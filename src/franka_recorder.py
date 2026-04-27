#!/usr/bin/env python3
"""Record Franka joint state samples and synchronized camera frames from ROS2."""

from __future__ import annotations

import argparse
import csv
import json
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np

from src.ros2_utils import image_msg_to_rgb8, stamp_to_nanoseconds


@dataclass(frozen=True)
class CameraFrameRef:
    index: int
    stamp_ns: int
    receive_time_ns: int
    relative_path: str
    width: int
    height: int
    encoding: str


@dataclass(frozen=True)
class CameraWriteJob:
    frame_ref: CameraFrameRef
    rgb: np.ndarray


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("recordings") / f"franka_recording_{timestamp}"


def _parse_joint_names(values: list[str] | None) -> list[str] | None:
    if not values:
        return None

    names: list[str] = []
    for value in values:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    return names or None


def _safe_header_name(name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip())
    return safe.strip("_") or "joint"


def _joint_number(name: str) -> int | None:
    match = re.search(r"(?:^|_)joint_?(\d+)$", name)
    if match is None:
        return None
    return int(match.group(1))


def _make_qos_profile(qos_module: object, reliability: str, depth: int) -> object:
    if reliability == "reliable":
        reliability_policy = qos_module.QoSReliabilityPolicy.RELIABLE
    elif reliability == "best_effort":
        reliability_policy = qos_module.QoSReliabilityPolicy.BEST_EFFORT
    else:
        raise ValueError(f"Unsupported reliability policy: {reliability}")

    return qos_module.QoSProfile(
        history=qos_module.QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=reliability_policy,
        durability=qos_module.QoSDurabilityPolicy.VOLATILE,
    )


class FrankaRecorder:
    """ROS2 subscriber that records q/qd at 1 kHz plus camera frames."""

    def __init__(self, args: argparse.Namespace) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy import qos
        from sensor_msgs.msg import Image, JointState

        class _Node(Node):
            pass

        self.args = args
        self.rclpy = rclpy
        self.node = _Node("franka_recorder")
        self.output_dir = args.output_dir
        self.frames_dir = self.output_dir / "frames"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.start_perf = time.perf_counter()
        self.created_at = datetime.now().isoformat(timespec="seconds")

        self.joint_names: list[str] | None = args.joint_names
        self.joint_indices: list[int] | None = None
        self.latest_q: np.ndarray | None = None
        self.latest_qd: np.ndarray | None = None
        self.latest_joint_stamp_ns = 0
        self.latest_joint_receive_time_ns = 0
        self.latest_camera_frame: CameraFrameRef | None = None

        self.joint_file: TextIO | None = None
        self.joint_writer: csv.writer | None = None
        self.frames_file = (self.output_dir / "frames.csv").open("w", newline="", buffering=1024 * 1024)
        self.frames_writer = csv.writer(self.frames_file)
        self.frames_writer.writerow(
            [
                "frame_index",
                "image_stamp_ns",
                "image_receive_time_ns",
                "width",
                "height",
                "encoding",
                "relative_path",
            ]
        )

        self.frame_queue: queue.Queue[CameraWriteJob] = queue.Queue(maxsize=args.frame_queue_size)
        self.frame_writer_thread = threading.Thread(target=self._frame_writer_loop, name="camera-frame-writer")
        self.frame_writer_thread.start()

        self.joint_sample_count = 0
        self.joint_message_count = 0
        self.camera_frame_count = 0
        self.camera_frames_written = 0
        self.dropped_camera_frames = 0
        self.failed_camera_writes = 0
        self.image_decode_errors = 0

        topic_qos = _make_qos_profile(qos, args.qos_reliability, args.qos_depth)
        self.node.create_subscription(JointState, args.joint_topic, self._on_joint_state, topic_qos)
        self.node.create_subscription(Image, args.image_topic, self._on_image, topic_qos)

        if args.sample_mode == "timer":
            self.node.create_timer(1.0 / max(args.rate_hz, 1e-3), self._sample_latest_joint)
        if args.status_every_sec > 0.0:
            self.node.create_timer(args.status_every_sec, self._report_status)

        self.node.get_logger().info(
            f"Recording joints from {args.joint_topic} and images from {args.image_topic}"
        )
        self.node.get_logger().info(f"Output directory: {self.output_dir}")
        if args.sample_mode == "timer":
            self.node.get_logger().info(f"Joint samples will be written at {args.rate_hz:.1f} Hz.")
        else:
            self.node.get_logger().info("Joint samples will be written once per incoming JointState message.")
        if not args.allow_missing_camera:
            self.node.get_logger().info("Waiting for both joint state and camera frames before writing samples.")

    def _resolve_joint_indices(self, msg: object) -> None:
        if self.joint_indices is not None:
            return

        message_names = list(getattr(msg, "name", []))
        if self.joint_names is not None:
            missing = [name for name in self.joint_names if name not in message_names]
            if missing:
                raise ValueError(
                    "JointState is missing requested joints: "
                    + ", ".join(missing)
                    + f". Available joints: {', '.join(message_names)}"
                )
            self.joint_indices = [message_names.index(name) for name in self.joint_names]
        elif self.args.joint_prefix:
            prefix = self.args.joint_prefix
            if prefix.endswith("_joint"):
                self.joint_names = [f"{prefix}{idx}" for idx in range(1, self.args.num_joints + 1)]
            else:
                self.joint_names = [f"{prefix}_joint{idx}" for idx in range(1, self.args.num_joints + 1)]
            missing = [name for name in self.joint_names if name not in message_names]
            if missing:
                raise ValueError(
                    "JointState is missing requested prefix joints: "
                    + ", ".join(missing)
                    + f". Available joints: {', '.join(message_names)}"
                )
            self.joint_indices = [message_names.index(name) for name in self.joint_names]
        else:
            self._infer_joint_indices(message_names)

        self._ensure_joint_writer()
        self.node.get_logger().info("Recording joint order: " + ", ".join(self.joint_names or []))

    def _infer_joint_indices(self, message_names: list[str]) -> None:
        positions = list(range(self.args.num_joints))
        if not message_names:
            self.joint_names = [f"joint_{idx}" for idx in range(1, self.args.num_joints + 1)]
            self.joint_indices = positions
            self.node.get_logger().warning(
                "JointState messages do not include names; recording the first "
                f"{self.args.num_joints} position entries."
            )
            return

        numbered: dict[int, tuple[int, str]] = {}
        for idx, name in enumerate(message_names):
            number = _joint_number(name)
            if number is None or number < 1 or number > self.args.num_joints:
                continue
            numbered.setdefault(number, (idx, name))

        if all(idx in numbered for idx in range(1, self.args.num_joints + 1)):
            ordered = [numbered[idx] for idx in range(1, self.args.num_joints + 1)]
            self.joint_indices = [idx for idx, _ in ordered]
            self.joint_names = [name for _, name in ordered]
            return

        self.joint_indices = positions
        self.joint_names = message_names[: self.args.num_joints]
        if len(self.joint_names) < self.args.num_joints:
            self.joint_names.extend(
                f"joint_{idx}" for idx in range(len(self.joint_names) + 1, self.args.num_joints + 1)
            )
        self.node.get_logger().warning(
            "Could not infer numbered Franka joints; recording the first "
            f"{self.args.num_joints} JointState entries: {', '.join(self.joint_names)}"
        )

    def _ensure_joint_writer(self) -> None:
        if self.joint_writer is not None:
            return
        if self.joint_names is None:
            raise RuntimeError("Cannot open joint writer before joint names are known.")

        self.joint_file = (self.output_dir / "joint_states.csv").open(
            "w",
            newline="",
            buffering=1024 * 1024,
        )
        self.joint_writer = csv.writer(self.joint_file)
        safe_names = [_safe_header_name(name) for name in self.joint_names]
        self.joint_writer.writerow(
            [
                "sample_index",
                "sample_ros_time_ns",
                "sample_wall_time_ns",
                "joint_msg_stamp_ns",
                "joint_receive_time_ns",
                "camera_frame_index",
                "camera_stamp_ns",
                "camera_receive_time_ns",
                "camera_relative_path",
                "camera_age_ms",
                *[f"q_{name}" for name in safe_names],
                *[f"qd_{name}" for name in safe_names],
            ]
        )

    def _on_joint_state(self, msg: object) -> None:
        receive_time_ns = time.time_ns()
        try:
            self._resolve_joint_indices(msg)
            q, qd = self._extract_joint_state(msg)
        except Exception as exc:
            self.node.get_logger().error(f"Could not parse JointState message: {exc}")
            return

        stamp_ns = stamp_to_nanoseconds(msg.header.stamp)
        with self.lock:
            self.latest_q = q
            self.latest_qd = qd
            self.latest_joint_stamp_ns = stamp_ns
            self.latest_joint_receive_time_ns = receive_time_ns
            self.joint_message_count += 1

        if self.args.sample_mode == "messages":
            self._write_joint_sample(
                q=q,
                qd=qd,
                joint_stamp_ns=stamp_ns,
                joint_receive_time_ns=receive_time_ns,
                sample_ros_time_ns=self.node.get_clock().now().nanoseconds,
                sample_wall_time_ns=time.time_ns(),
            )

    def _extract_joint_state(self, msg: object) -> tuple[np.ndarray, np.ndarray]:
        if self.joint_indices is None:
            raise RuntimeError("Joint indices have not been resolved.")

        positions = list(getattr(msg, "position", []))
        velocities = list(getattr(msg, "velocity", []))
        if max(self.joint_indices) >= len(positions):
            raise ValueError(
                f"JointState position length {len(positions)} is too short for indices {self.joint_indices}."
            )

        q = np.asarray([positions[idx] for idx in self.joint_indices], dtype=np.float64)
        qd_values = [
            velocities[idx] if idx < len(velocities) else float("nan")
            for idx in self.joint_indices
        ]
        qd = np.asarray(qd_values, dtype=np.float64)
        return q, qd

    def _on_image(self, msg: object) -> None:
        try:
            rgb = image_msg_to_rgb8(msg)
        except Exception as exc:
            self.image_decode_errors += 1
            if self.image_decode_errors <= 3 or self.image_decode_errors % 10 == 0:
                self.node.get_logger().warning(f"Could not decode image message: {exc}")
            return

        frame_index = self.camera_frame_count
        extension = "jpg" if self.args.image_format == "jpg" else "png"
        relative_path = f"frames/frame_{frame_index:06d}.{extension}"
        frame_ref = CameraFrameRef(
            index=frame_index,
            stamp_ns=stamp_to_nanoseconds(msg.header.stamp),
            receive_time_ns=time.time_ns(),
            relative_path=relative_path,
            width=int(msg.width),
            height=int(msg.height),
            encoding=str(msg.encoding),
        )
        job = CameraWriteJob(frame_ref=frame_ref, rgb=rgb)

        try:
            if self.args.block_on_frame_queue:
                self.frame_queue.put(job)
            else:
                self.frame_queue.put_nowait(job)
        except queue.Full:
            self.dropped_camera_frames += 1
            if self.dropped_camera_frames <= 3 or self.dropped_camera_frames % 10 == 0:
                self.node.get_logger().warning(
                    "Camera frame writer queue is full; dropping frame "
                    f"{frame_index}. Increase --frame-queue-size or use --block-on-frame-queue."
                )
            return

        with self.lock:
            self.latest_camera_frame = frame_ref
            self.camera_frame_count += 1

    def _sample_latest_joint(self) -> None:
        with self.lock:
            if self.latest_q is None or self.latest_qd is None:
                return
            q = self.latest_q.copy()
            qd = self.latest_qd.copy()
            joint_stamp_ns = self.latest_joint_stamp_ns
            joint_receive_time_ns = self.latest_joint_receive_time_ns

        self._write_joint_sample(
            q=q,
            qd=qd,
            joint_stamp_ns=joint_stamp_ns,
            joint_receive_time_ns=joint_receive_time_ns,
            sample_ros_time_ns=self.node.get_clock().now().nanoseconds,
            sample_wall_time_ns=time.time_ns(),
        )

    def _write_joint_sample(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        joint_stamp_ns: int,
        joint_receive_time_ns: int,
        sample_ros_time_ns: int,
        sample_wall_time_ns: int,
    ) -> None:
        if self.joint_writer is None:
            return

        with self.lock:
            camera_frame = self.latest_camera_frame

        if camera_frame is None and not self.args.allow_missing_camera:
            return

        camera_index = -1
        camera_stamp_ns = 0
        camera_receive_time_ns = 0
        camera_relative_path = ""
        camera_age_ms = float("nan")
        if camera_frame is not None:
            camera_index = camera_frame.index
            camera_stamp_ns = camera_frame.stamp_ns
            camera_receive_time_ns = camera_frame.receive_time_ns
            camera_relative_path = camera_frame.relative_path
            if camera_stamp_ns > 0:
                camera_age_ms = (sample_ros_time_ns - camera_stamp_ns) / 1e6

        row = [
            self.joint_sample_count,
            sample_ros_time_ns,
            sample_wall_time_ns,
            joint_stamp_ns,
            joint_receive_time_ns,
            camera_index,
            camera_stamp_ns,
            camera_receive_time_ns,
            camera_relative_path,
            f"{camera_age_ms:.6f}",
            *[f"{value:.10g}" for value in q],
            *[f"{value:.10g}" for value in qd],
        ]
        self.joint_writer.writerow(row)
        self.joint_sample_count += 1

    def _frame_writer_loop(self) -> None:
        while not self.stop_event.is_set() or not self.frame_queue.empty():
            try:
                job = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._write_camera_frame(job)
            except Exception as exc:
                self.failed_camera_writes += 1
                self.node.get_logger().warning(
                    f"Could not write camera frame {job.frame_ref.index}: {exc}"
                )
            finally:
                self.frame_queue.task_done()

    def _write_camera_frame(self, job: CameraWriteJob) -> None:
        output_path = self.output_dir / job.frame_ref.relative_path
        bgr = cv2.cvtColor(job.rgb, cv2.COLOR_RGB2BGR)
        if self.args.image_format == "jpg":
            params = [cv2.IMWRITE_JPEG_QUALITY, int(self.args.jpeg_quality)]
        else:
            params = [cv2.IMWRITE_PNG_COMPRESSION, int(self.args.png_compression)]

        if not cv2.imwrite(str(output_path), bgr, params):
            raise RuntimeError(f"cv2.imwrite returned false for {output_path}")

        self.frames_writer.writerow(
            [
                job.frame_ref.index,
                job.frame_ref.stamp_ns,
                job.frame_ref.receive_time_ns,
                job.frame_ref.width,
                job.frame_ref.height,
                job.frame_ref.encoding,
                job.frame_ref.relative_path,
            ]
        )
        self.camera_frames_written += 1

    def _report_status(self) -> None:
        elapsed = max(time.perf_counter() - self.start_perf, 1e-9)
        joint_rate = self.joint_sample_count / elapsed
        image_rate = self.camera_frames_written / elapsed
        self.node.get_logger().info(
            "Recorded "
            f"{self.joint_sample_count} joint samples ({joint_rate:.1f} Hz), "
            f"{self.camera_frames_written} camera frames ({image_rate:.1f} Hz), "
            f"queue={self.frame_queue.qsize()}, dropped_frames={self.dropped_camera_frames}."
        )

    def duration_reached(self) -> bool:
        return self.args.duration_sec > 0.0 and (time.perf_counter() - self.start_perf) >= self.args.duration_sec

    def shutdown(self) -> None:
        self.node.get_logger().info("Stopping recorder and flushing camera frames.")
        self.stop_event.set()
        self.frame_queue.join()
        self.frame_writer_thread.join(timeout=5.0)
        if self.frame_writer_thread.is_alive():
            self.node.get_logger().warning("Camera writer thread did not stop within 5 seconds.")

        if self.joint_file is not None:
            self.joint_file.close()
        self.frames_file.close()
        self._write_metadata()
        self.node.get_logger().info(f"Saved recording to {self.output_dir}")
        self.node.destroy_node()

    def _write_metadata(self) -> None:
        metadata = {
            "created_at": self.created_at,
            "closed_at": datetime.now().isoformat(timespec="seconds"),
            "joint_topic": self.args.joint_topic,
            "image_topic": self.args.image_topic,
            "sample_mode": self.args.sample_mode,
            "target_rate_hz": self.args.rate_hz,
            "joint_names": self.joint_names,
            "joint_samples": self.joint_sample_count,
            "joint_messages_received": self.joint_message_count,
            "camera_frames_received": self.camera_frame_count,
            "camera_frames_written": self.camera_frames_written,
            "camera_frames_dropped": self.dropped_camera_frames,
            "camera_writes_failed": self.failed_camera_writes,
            "image_decode_errors": self.image_decode_errors,
            "image_format": self.args.image_format,
            "allow_missing_camera": self.args.allow_missing_camera,
        }
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record Franka q/qd from sensor_msgs/JointState at 1000 Hz while saving "
            "camera frames from a ROS2 Image topic."
        )
    )
    parser.add_argument("--joint-topic", type=str, default="/joint_states")
    parser.add_argument("--image-topic", type=str, default="/camera/color/image_raw")
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--rate-hz", type=float, default=1000.0, help="Timer sampling rate for --sample-mode timer.")
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="Stop automatically after this many seconds. The default records until Ctrl-C.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("timer", "messages"),
        default="timer",
        help="Use a fixed-rate timer, or write one row per incoming JointState message.",
    )
    parser.add_argument(
        "--joint-names",
        nargs="*",
        default=None,
        help=(
            "Explicit joint order. Accepts space-separated names or one comma-separated string, "
            "for example: panda_joint1,panda_joint2,..."
        ),
    )
    parser.add_argument(
        "--joint-prefix",
        type=str,
        default=None,
        help="Convenience prefix for Franka joints, for example 'panda' or 'fr3'.",
    )
    parser.add_argument("--num-joints", type=int, default=7, help="Number of arm joints to record.")
    parser.add_argument(
        "--allow-missing-camera",
        action="store_true",
        help="Start writing joint samples before the first camera frame arrives.",
    )
    parser.add_argument("--image-format", choices=("png", "jpg"), default="png")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--png-compression", type=int, default=3)
    parser.add_argument("--frame-queue-size", type=int, default=128)
    parser.add_argument(
        "--block-on-frame-queue",
        action="store_true",
        help="Block image callbacks instead of dropping frames when disk writing falls behind.",
    )
    parser.add_argument(
        "--qos-reliability",
        choices=("reliable", "best_effort"),
        default="best_effort",
        help="Subscriber QoS reliability for both topics.",
    )
    parser.add_argument("--qos-depth", type=int, default=100)
    parser.add_argument("--status-every-sec", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.joint_names = _parse_joint_names(args.joint_names)

    import rclpy

    rclpy.init()
    recorder: FrankaRecorder | None = None
    try:
        recorder = FrankaRecorder(args)
        while rclpy.ok():
            rclpy.spin_once(recorder.node, timeout_sec=0.1)
            if recorder.duration_reached():
                recorder.node.get_logger().info(f"Reached duration {args.duration_sec:.3f} seconds.")
                break
    except KeyboardInterrupt:
        pass
    finally:  # pragma: no cover - ROS runtime path
        if recorder is not None:
            recorder.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
