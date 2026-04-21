"""ROS2 node for live object pose tracking with FoundationPose plus a 2D tracker."""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from efficient_track_anything.wrapper import EfficientTrackAnythingWrapper
from foundationpose_wrapper import FoundationPoseWrapper
from sam2.wrapper import Sam2Wrapper

from src.ros2_utils import (
    bbox_xywh_to_xyxy,
    binary_mask_to_bbox_xywh,
    camera_info_to_matrix,
    depth_msg_to_meters,
    image_msg_to_rgb8,
    logits_to_binary_mask,
    make_pose_stamped,
    numpy_to_image_msg,
    stamp_to_nanoseconds,
)


def get_pose_xy_from_image_point(
    ob_in_cam: torch.Tensor,
    camera_matrix: torch.Tensor,
    x: float = -1.0,
    y: float = -1.0,
) -> tuple[float, float]:
    """Compute the translation that projects to the provided image point."""

    pose = ob_in_cam[0] if ob_in_cam.ndim == 3 else ob_in_cam
    if x < 0.0 or y < 0.0:
        return float(pose[0, 3].item()), float(pose[1, 3].item())

    tz = float(pose[2, 3].item())
    fx = float(camera_matrix[0, 0].item())
    fy = float(camera_matrix[1, 1].item())
    cx = float(camera_matrix[0, 2].item())
    cy = float(camera_matrix[1, 2].item())
    tx = (x - cx) * tz / fx
    ty = (y - cy) * tz / fy
    return tx, ty


def adjust_pose_to_image_point(
    ob_in_cam: torch.Tensor,
    camera_matrix: torch.Tensor,
    x: float = -1.0,
    y: float = -1.0,
) -> torch.Tensor:
    """Adjust the translation of a pose tensor to align its image projection center."""

    pose_tensor = ob_in_cam if ob_in_cam.ndim == 3 else ob_in_cam.unsqueeze(0)
    pose_new = pose_tensor.clone()

    for idx in range(pose_tensor.shape[0]):
        tx, ty = get_pose_xy_from_image_point(pose_tensor[idx], camera_matrix, x=x, y=y)
        pose_new[idx, 0, 3] = tx
        pose_new[idx, 1, 3] = ty

    return pose_new if ob_in_cam.ndim == 3 else pose_new[0]


@dataclass
class SegmentationResult:
    mask: np.ndarray
    bbox_xywh: list[int]


class CameraTrackerAdapter:
    """Thin adapter that normalizes SAM2 and EfficientTAM camera predictor usage."""

    def __init__(
        self,
        tracker_name: str,
        config_file: str,
        checkpoint_path: str,
        device: str,
        object_id: int,
        mask_threshold: float,
    ) -> None:
        self.tracker_name = tracker_name
        self.object_id = object_id
        self.mask_threshold = mask_threshold

        if tracker_name == "sam2":
            wrapper = Sam2Wrapper(device=device)
            self.predictor = wrapper.load_camera_predictor(
                config_file=config_file,
                checkpoint_path=checkpoint_path,
            )
        elif tracker_name == "efficienttam":
            wrapper = EfficientTrackAnythingWrapper(device=device)
            self.predictor = wrapper.load_camera_predictor(
                config_file=config_file,
                checkpoint_path=checkpoint_path,
            )
        else:
            raise ValueError(f"Unsupported tracker: {tracker_name}")

    def initialize(
        self,
        frame_rgb: np.ndarray,
        init_mask: np.ndarray | None,
        init_bbox_xywh: list[float] | None,
    ) -> SegmentationResult:
        self.predictor.load_first_frame(frame_rgb)
        if init_mask is not None:
            _, obj_ids, mask_logits = self.predictor.add_new_mask(
                frame_idx=0,
                obj_id=self.object_id,
                mask=np.asarray(init_mask, dtype=bool),
            )
        elif init_bbox_xywh is not None:
            _, obj_ids, mask_logits = self.predictor.add_new_prompt(
                frame_idx=0,
                obj_id=self.object_id,
                bbox=bbox_xywh_to_xyxy(init_bbox_xywh),
            )
        else:
            raise ValueError("Either init_mask or init_bbox_xywh must be provided.")

        mask = self._extract_object_mask(obj_ids, mask_logits)
        return SegmentationResult(mask=mask, bbox_xywh=binary_mask_to_bbox_xywh(mask))

    def track(self, frame_rgb: np.ndarray) -> SegmentationResult:
        obj_ids, mask_logits = self.predictor.track(frame_rgb)
        mask = self._extract_object_mask(obj_ids, mask_logits)
        return SegmentationResult(mask=mask, bbox_xywh=binary_mask_to_bbox_xywh(mask))

    def _extract_object_mask(self, obj_ids: Any, mask_logits: Any) -> np.ndarray:
        object_ids = list(obj_ids)
        object_index = object_ids.index(self.object_id) if self.object_id in object_ids else 0
        return logits_to_binary_mask(mask_logits, object_index=object_index, threshold=self.mask_threshold)


class LiveObjectPoseTrackerNode:
    """ROS2 node wrapper kept separate from module import to simplify testing."""

    def __init__(self, args: argparse.Namespace) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        from geometry_msgs.msg import PoseStamped

        class _Node(Node):
            pass

        self.args = args
        self.rclpy = rclpy
        self.node = _Node("live_object_pose_tracker")
        self.lock = threading.Lock()
        self.processing = False
        self.latest_color_msg: Image | None = None
        self.latest_depth_msg: Image | None = None
        self.latest_camera_info_msg: CameraInfo | None = None
        self.last_processed_color_stamp_ns: int | None = None
        self.camera_matrix_fallback = (
            np.asarray(args.cam_k, dtype=np.float32).reshape(3, 3) if args.cam_k is not None else None
        )

        force_apply_color = [int(v) for v in args.apply_color] if args.force_apply_color else None
        self.foundationpose = FoundationPoseWrapper(device=args.device, debug_dir=args.debug_dir)
        self.foundationpose.load(
            mesh_path=args.mesh_path,
            apply_scale=args.apply_scale,
            force_apply_color=force_apply_color,
        )
        self.segmenter = CameraTrackerAdapter(
            tracker_name=args.tracker,
            config_file=args.tracker_config,
            checkpoint_path=args.tracker_checkpoint,
            device=args.device,
            object_id=args.object_id,
            mask_threshold=args.mask_threshold,
        )

        self.init_mask = self._load_init_mask(args.init_mask_path) if args.init_mask_path else None
        self.init_bbox_xywh = list(args.init_bbox) if args.init_bbox is not None else None
        if self.init_mask is None and self.init_bbox_xywh is None:
            raise ValueError("Provide either --init-mask-path or --init-bbox to initialize tracking.")

        self.initialized = False

        self.pose_pub = self.node.create_publisher(PoseStamped, args.pose_topic, 10)
        self.mask_pub = self.node.create_publisher(Image, args.mask_topic, 10) if args.publish_mask else None

        self.node.create_subscription(Image, args.color_topic, self._on_color, qos_profile_sensor_data)
        self.node.create_subscription(Image, args.depth_topic, self._on_depth, qos_profile_sensor_data)
        self.node.create_subscription(
            CameraInfo, args.camera_info_topic, self._on_camera_info, qos_profile_sensor_data
        )
        self.node.create_timer(1.0 / max(args.process_rate_hz, 1e-3), self._process_latest_frame)

        self.node.get_logger().info(
            f"Tracking {args.tracker} + FoundationPose from {args.color_topic} and {args.depth_topic}"
        )

    def _load_init_mask(self, init_mask_path: str) -> np.ndarray:
        mask = cv2.imread(init_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read init mask: {init_mask_path}")
        return mask.astype(bool)

    def _on_color(self, msg: object) -> None:
        with self.lock:
            self.latest_color_msg = msg

    def _on_depth(self, msg: object) -> None:
        with self.lock:
            self.latest_depth_msg = msg

    def _on_camera_info(self, msg: object) -> None:
        with self.lock:
            self.latest_camera_info_msg = msg

    def _process_latest_frame(self) -> None:
        if self.processing:
            return

        with self.lock:
            color_msg = self.latest_color_msg
            depth_msg = self.latest_depth_msg
            camera_info_msg = self.latest_camera_info_msg

        if color_msg is None or depth_msg is None:
            return

        color_stamp_ns = stamp_to_nanoseconds(color_msg.header.stamp)
        if self.last_processed_color_stamp_ns == color_stamp_ns:
            return

        depth_stamp_ns = stamp_to_nanoseconds(depth_msg.header.stamp)
        if abs(color_stamp_ns - depth_stamp_ns) > int(self.args.max_sync_offset_sec * 1e9):
            return

        camera_matrix = self.camera_matrix_fallback
        if camera_info_msg is not None:
            camera_matrix = camera_info_to_matrix(camera_info_msg)
        if camera_matrix is None:
            return

        self.processing = True
        try:
            self._process_frame_pair(color_msg, depth_msg, camera_info_msg, camera_matrix)
            self.last_processed_color_stamp_ns = color_stamp_ns
        except Exception as exc:  # pragma: no cover - ROS runtime path
            self.node.get_logger().error(f"Tracking step failed: {exc}")
        finally:
            self.processing = False

    def _process_frame_pair(
        self,
        color_msg: object,
        depth_msg: object,
        camera_info_msg: object | None,
        camera_matrix: np.ndarray,
    ) -> None:
        color_rgb = image_msg_to_rgb8(color_msg)
        depth_m = depth_msg_to_meters(depth_msg, depth_scale_for_uint16=self.args.depth_scale)

        if not self.initialized:
            segmentation = self.segmenter.initialize(
                frame_rgb=color_rgb,
                init_mask=self._match_init_mask_shape(self.init_mask, color_rgb.shape[:2]),
                init_bbox_xywh=self.init_bbox_xywh,
            )
            if segmentation.bbox_xywh[2] <= 0 or segmentation.bbox_xywh[3] <= 0:
                raise RuntimeError("Initialization failed because the 2D tracker produced an empty mask.")

            pose = self.foundationpose.register(
                rgb=color_rgb,
                depth=depth_m,
                camera_matrix=camera_matrix,
                object_mask=segmentation.mask.astype(np.uint8) * 255,
                iteration=self.args.est_refine_iter,
            )
            self.initialized = True
            self.node.get_logger().info("Tracker initialized from the first live frame.")
        else:
            segmentation = self.segmenter.track(color_rgb)
            estimator = self.foundationpose.get()
            bbox = segmentation.bbox_xywh
            if bbox[2] > 0 and bbox[3] > 0:
                camera_matrix_tensor = torch.as_tensor(
                    camera_matrix,
                    device=estimator.pose_last.device,
                    dtype=estimator.pose_last.dtype,
                )
                estimator.pose_last = adjust_pose_to_image_point(
                    ob_in_cam=estimator.pose_last,
                    camera_matrix=camera_matrix_tensor,
                    x=float(bbox[0] + bbox[2] / 2.0),
                    y=float(bbox[1] + bbox[3] / 2.0),
                )

            pose = self.foundationpose.track_one(
                rgb=color_rgb,
                depth=depth_m,
                camera_matrix=camera_matrix,
                iteration=self.args.track_refine_iter,
            )

        frame_id = (
            camera_info_msg.header.frame_id
            if camera_info_msg is not None and camera_info_msg.header.frame_id
            else color_msg.header.frame_id
        )
        pose_msg = make_pose_stamped(pose, frame_id=frame_id, stamp=color_msg.header.stamp)
        self.pose_pub.publish(pose_msg)

        if self.mask_pub is not None:
            mask_msg = numpy_to_image_msg(
                array=(segmentation.mask.astype(np.uint8) * 255),
                encoding="mono8",
                frame_id=frame_id,
                stamp=color_msg.header.stamp,
            )
            self.mask_pub.publish(mask_msg)

    def _match_init_mask_shape(self, mask: np.ndarray | None, hw: tuple[int, int]) -> np.ndarray | None:
        if mask is None:
            return None
        if mask.shape == hw:
            return mask
        return cv2.resize(mask.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def _parse_json_list(value: str) -> list[float]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError(f"Expected a JSON list, got: {value}")
    return [float(v) for v in parsed]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live ROS2 object pose tracker using FoundationPose.")
    parser.add_argument("--tracker", choices=["sam2", "efficienttam"], default="sam2")
    parser.add_argument("--tracker-config", type=str, default=None)
    parser.add_argument("--tracker-checkpoint", type=str, required=True)
    parser.add_argument("--mesh-path", type=str, required=True)
    parser.add_argument("--color-topic", type=str, default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", type=str, default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic", type=str, default="/camera/color/camera_info")
    parser.add_argument("--pose-topic", type=str, default="/vision/tracked_pose")
    parser.add_argument("--mask-topic", type=str, default="/vision/tracked_mask")
    parser.add_argument("--publish-mask", action="store_true")
    parser.add_argument("--process-rate-hz", type=float, default=15.0)
    parser.add_argument("--max-sync-offset-sec", type=float, default=0.08)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--init-mask-path", type=str, default=None)
    parser.add_argument("--init-bbox", type=_parse_json_list, default=None, help='JSON list "[x, y, w, h]".')
    parser.add_argument("--cam-k", type=_parse_json_list, default=None, help='Optional JSON list with 9 intrinsics.')
    parser.add_argument("--est-refine-iter", type=int, default=10)
    parser.add_argument("--track-refine-iter", type=int, default=5)
    parser.add_argument("--apply-scale", type=float, default=0.01)
    parser.add_argument("--force-apply-color", action="store_true")
    parser.add_argument("--apply-color", type=_parse_json_list, default=[0.0, 159.0, 237.0])
    parser.add_argument("--debug-dir", type=str, default="./debug/foundationpose_ros2")
    return parser


def _resolve_tracker_config(args: argparse.Namespace) -> str:
    if args.tracker_config:
        return args.tracker_config
    if args.tracker == "sam2":
        return "configs/sam2.1/sam2.1_hiera_t.yaml"
    return "configs/efficienttam/efficienttam_ti_512x512.yaml"


def main() -> None:
    args = build_argparser().parse_args()
    args.tracker_config = _resolve_tracker_config(args)
    args.mesh_path = str(Path(args.mesh_path).expanduser().resolve())
    if args.init_mask_path:
        args.init_mask_path = str(Path(args.init_mask_path).expanduser().resolve())
    args.tracker_checkpoint = str(Path(args.tracker_checkpoint).expanduser().resolve())
    if args.cam_k is not None and len(args.cam_k) != 9:
        raise ValueError("--cam-k must contain 9 values.")
    if args.init_bbox is not None and len(args.init_bbox) != 4:
        raise ValueError("--init-bbox must contain 4 values.")

    import rclpy

    rclpy.init()
    tracker = None
    try:
        tracker = LiveObjectPoseTrackerNode(args)
        rclpy.spin(tracker.node)
    finally:  # pragma: no cover - ROS runtime path
        if tracker is not None:
            tracker.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
