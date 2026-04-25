#!/usr/bin/env python3
"""ROS2 node that runs segmentation tracking from the RealSense RGB stream."""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from tracking.ros2_utils import (
    binary_mask_to_bbox_xywh,
    bbox_xywh_to_polygon_msg,
    image_msg_to_rgb8,
    logits_to_binary_mask,
    numpy_to_image_msg,
    stamp_to_nanoseconds,
)
from vision.cfg.segmentation.seg_cfg import SegmentationCfg
from vision.wrapper.seg_wrapper import SegWrapper


def bbox_xywh_to_xyxy(bbox_xywh: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox_xywh
    return (x, y, x + w, y + h)


def _to_python_int(value: object) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _object_ids_to_list(obj_ids: object) -> list[int]:
    try:
        return [_to_python_int(value) for value in list(obj_ids)]
    except TypeError:
        return [_to_python_int(obj_ids)]


def extract_object_mask(result: object, object_id: int, threshold: float) -> np.ndarray:
    """Extract the requested object mask from an add_new_prompt or track result."""

    if not isinstance(result, tuple):
        raise ValueError(f"Expected tuple segmentation result, got {type(result)!r}.")
    if len(result) == 3:
        _, obj_ids, mask_logits = result
    elif len(result) == 2:
        obj_ids, mask_logits = result
    else:
        raise ValueError(f"Expected a 2- or 3-item segmentation result, got {len(result)} items.")

    object_ids = _object_ids_to_list(obj_ids)
    object_index = object_ids.index(object_id) if object_id in object_ids else 0
    return logits_to_binary_mask(mask_logits, object_index=object_index, threshold=threshold)


def build_prompt(args: argparse.Namespace) -> tuple[tuple[float, float, float, float] | None, np.ndarray | None, np.ndarray | None]:
    bbox_xyxy = None
    if args.bbox is not None:
        bbox_xyxy = bbox_xywh_to_xyxy([float(value) for value in args.bbox])
    if args.bbox_xyxy is not None:
        bbox_xyxy = tuple(float(value) for value in args.bbox_xyxy)

    if not args.point:
        return bbox_xyxy, None, None

    points = []
    labels = []
    for x, y, label in args.point:
        points.append([float(x), float(y)])
        labels.append(int(label))
    return bbox_xyxy, np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32)


class SegmentationTrackingNode:
    """Subscribe to RealSense RGB frames, initialize once, then track every new frame."""

    def __init__(self, args: argparse.Namespace) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from geometry_msgs.msg import PolygonStamped
        from sensor_msgs.msg import Image

        class _Node(Node):
            pass

        self.args = args
        self.rclpy = rclpy
        self.node = _Node("run_segmentation")
        self.lock = threading.Lock()
        self.latest_color_msg: Image | None = None
        self.processing = False
        self.initialized = False
        self.prompt_cancelled = False
        self.first_stamp_ns: int | None = None
        self.last_processed_stamp_ns: int | None = None
        self.tracked_frame_count = 0

        self.prompt_bbox_xyxy, self.prompt_points, self.prompt_labels = build_prompt(args)
        cfg = SegmentationCfg.from_yaml(args.seg_cfg)
        self.segmenter = SegWrapper(cfg=cfg, device=args.device)

        self.mask_pub = self.node.create_publisher(Image, args.mask_topic, 10)
        self.bbox_pub = self.node.create_publisher(PolygonStamped, args.bbox_topic, 10)
        self.node.create_subscription(Image, args.color_topic, self._on_color, qos_profile_sensor_data)
        self.node.create_timer(1.0 / max(args.process_rate_hz, 1e-3), self._process_latest_frame)

        self.node.get_logger().info(
            f"Subscribing to {args.color_topic}; publishing masks on {args.mask_topic} "
            f"and bboxes on {args.bbox_topic}."
        )
        self.node.get_logger().info(
            f"Loaded {self.segmenter.model_type} segmentation model on {args.device}."
        )
        if self._needs_window_prompt():
            self.node.get_logger().info(
                "No initial prompt was provided. The first RGB frame will open a window for bbox selection."
            )

    def _on_color(self, msg: object) -> None:
        with self.lock:
            self.latest_color_msg = msg

    def _process_latest_frame(self) -> None:
        if self.processing or self.prompt_cancelled:
            return

        with self.lock:
            color_msg = self.latest_color_msg
        if color_msg is None:
            return

        stamp_ns = stamp_to_nanoseconds(color_msg.header.stamp)
        if self.last_processed_stamp_ns == stamp_ns:
            return

        self.processing = True
        try:
            frame_rgb = image_msg_to_rgb8(color_msg)
            if not self.initialized:
                self._initialize_tracking(frame_rgb, color_msg)
            else:
                self._track_frame(frame_rgb, color_msg)
            self.last_processed_stamp_ns = stamp_ns
        except Exception as exc:
            self.node.get_logger().error(f"Segmentation processing failed: {exc}")
            if not self.initialized and self._needs_window_prompt():
                self.prompt_cancelled = True
                self.node.get_logger().warning(
                    "Interactive prompt initialization failed. Restart run_segmentation with a display, "
                    "or pass --bbox/--bbox-xyxy/--point explicitly."
                )
        finally:
            self.processing = False

    def _initialize_tracking(self, frame_rgb: np.ndarray, color_msg: object) -> None:
        start = time.perf_counter()
        if self._needs_window_prompt():
            bbox_xyxy = self._select_prompt_bbox(frame_rgb)
            if bbox_xyxy is None:
                self.prompt_cancelled = True
                self.node.get_logger().warning(
                    "Prompt selection was cancelled. Restart run_segmentation to initialize tracking."
                )
                return
            self.prompt_bbox_xyxy = bbox_xyxy

        self.segmenter.load_first_frame(frame_rgb)
        result = self.segmenter.add_new_prompt(
            obj_id=self.args.object_id,
            points=self.prompt_points,
            labels=self.prompt_labels,
            bbox=self.prompt_bbox_xyxy,
            frame_idx=0,
        )
        mask = extract_object_mask(result, object_id=self.args.object_id, threshold=self.args.mask_threshold)
        self.initialized = True
        self.first_stamp_ns = stamp_to_nanoseconds(color_msg.header.stamp)
        bbox_xywh = binary_mask_to_bbox_xywh(mask)
        self._publish_result(mask, bbox_xywh, color_msg)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.node.get_logger().info(
            "Initialized segmentation from first image timestamp "
            f"{self.first_stamp_ns} ns with bbox_xywh={bbox_xywh} in {elapsed_ms:.1f} ms."
        )

    def _needs_window_prompt(self) -> bool:
        has_prompt = (
            self.prompt_bbox_xyxy is not None
            or self.prompt_points is not None
            or self.prompt_labels is not None
        )
        return bool(self.args.prompt_window and not has_prompt)

    def _select_prompt_bbox(self, frame_rgb: np.ndarray) -> tuple[float, float, float, float] | None:
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "OpenCV is required for interactive prompt selection. "
                "Install OpenCV or pass --bbox/--bbox-xyxy/--point explicitly."
            ) from exc

        frame_bgr = frame_rgb[:, :, ::-1].copy()
        window_name = self.args.prompt_window_name
        self.node.get_logger().info(
            "Draw the target bbox in the prompt window, then press Enter or Space. Press c or Esc to cancel."
        )
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            x, y, width, height = cv2.selectROI(
                window_name,
                frame_bgr,
                fromCenter=False,
                showCrosshair=True,
            )
        finally:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass

        if width <= 0 or height <= 0:
            return None
        return (float(x), float(y), float(x + width), float(y + height))

    def _track_frame(self, frame_rgb: np.ndarray, color_msg: object) -> None:
        start = time.perf_counter()
        result = self.segmenter.track(frame_rgb)
        mask = extract_object_mask(result, object_id=self.args.object_id, threshold=self.args.mask_threshold)
        bbox_xywh = binary_mask_to_bbox_xywh(mask)
        self._publish_result(mask, bbox_xywh, color_msg)
        self.tracked_frame_count += 1

        if self.args.log_every_frames > 0 and self.tracked_frame_count % self.args.log_every_frames == 0:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.node.get_logger().info(
                f"Tracked frame {self.tracked_frame_count}: bbox_xywh={bbox_xywh}, track={elapsed_ms:.1f} ms."
            )

    def _publish_result(self, mask: np.ndarray, bbox_xywh: list[int], color_msg: object) -> None:
        self._publish_mask(mask, color_msg)
        self._publish_bbox(bbox_xywh, color_msg)

    def _publish_mask(self, mask: np.ndarray, color_msg: object) -> None:
        mask_u8 = np.asarray(mask, dtype=np.uint8) * 255
        mask_msg = numpy_to_image_msg(
            array=mask_u8,
            encoding="mono8",
            frame_id=color_msg.header.frame_id,
            stamp=color_msg.header.stamp,
        )
        self.mask_pub.publish(mask_msg)

    def _publish_bbox(self, bbox_xywh: list[int], color_msg: object) -> None:
        bbox_msg = bbox_xywh_to_polygon_msg(
            bbox_xywh=bbox_xywh,
            frame_id=color_msg.header.frame_id,
            stamp=color_msg.header.stamp,
        )
        self.bbox_pub.publish(bbox_msg)

    def shutdown(self) -> None:
        self.node.destroy_node()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Subscribe to the RealSense RGB topic and run segmentation tracking."
    )
    parser.add_argument("--color-topic", type=str, default="/camera/color/image_raw")
    parser.add_argument("--mask-topic", type=str, default="/tracking/segmentation/mask")
    parser.add_argument("--bbox-topic", type=str, default="/tracking/segmentation/bbox")
    parser.add_argument("--seg-cfg", type=str, default="config/tracking/seg_cfg_etam.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--object-id", type=int, default=0)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--process-rate-hz", type=float, default=30.0)
    parser.add_argument("--log-every-frames", type=int, default=30)
    parser.add_argument(
        "--prompt-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open a first-frame OpenCV window for bbox selection when no CLI prompt is provided.",
    )
    parser.add_argument("--prompt-window-name", type=str, default="Segmentation Prompt")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="Initial prompt bounding box in xywh pixels on the first received RGB frame.",
    )
    parser.add_argument(
        "--bbox-xyxy",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Initial prompt bounding box in xyxy pixels on the first received RGB frame.",
    )
    parser.add_argument(
        "--point",
        type=float,
        nargs=3,
        action="append",
        metavar=("X", "Y", "LABEL"),
        help="Initial point prompt in pixels. Repeat for multiple points; LABEL is usually 1 for foreground or 0 for background.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.bbox is not None and args.bbox_xyxy is not None:
        parser.error("Use only one of --bbox or --bbox-xyxy.")
    if args.bbox is None and args.bbox_xyxy is None and not args.point and not args.prompt_window:
        parser.error("Provide an initial prompt with --bbox, --bbox-xyxy, or at least one --point.")


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    import rclpy

    rclpy.init()
    runner = SegmentationTrackingNode(args)
    try:
        rclpy.spin(runner.node)
    except KeyboardInterrupt:
        pass
    finally:
        runner.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
