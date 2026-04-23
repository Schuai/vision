"""ROS2 node for live object pose tracking with FoundationPose plus a 2D tracker."""

from __future__ import annotations

import argparse
import contextlib
import json
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from efficient_track_anything.wrapper import EfficientTrackAnythingWrapper
from foundationpose_wrapper import FoundationPoseWrapper
from sam2.wrapper import Sam2Wrapper
from tracking.FoundationPose_plus_plus.src.utils.kalman_filter_6d import KalmanFilter6D

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


def get_mat_from_6d_pose_arr(pose_arr: np.ndarray) -> np.ndarray:
    """Convert [tx, ty, tz, rx, ry, rz] into a 4x4 pose matrix."""

    pose_vector = np.asarray(pose_arr, dtype=np.float32).reshape(-1)
    xyz = pose_vector[:3]
    euler_angles = pose_vector[3:6]
    rotation_matrix = Rotation.from_euler("xyz", euler_angles, degrees=False).as_matrix().astype(np.float32)
    transformation_matrix = np.eye(4, dtype=np.float32)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = xyz
    return transformation_matrix


def get_6d_pose_arr_from_mat(pose: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a 4x4 pose matrix into [tx, ty, tz, rx, ry, rz]."""

    if torch.is_tensor(pose):
        pose_np = pose[0].detach().cpu().numpy() if pose.ndim == 3 else pose.detach().cpu().numpy()
    else:
        pose_np = np.asarray(pose, dtype=np.float32)

    xyz = pose_np[:3, 3]
    euler_angles = Rotation.from_matrix(pose_np[:3, :3]).as_euler("xyz", degrees=False).astype(np.float32)
    return np.concatenate((xyz.astype(np.float32), euler_angles), axis=0)


def pose_tensor_from_6d_pose_arr(pose_arr: np.ndarray, template_pose: torch.Tensor) -> torch.Tensor:
    """Create a pose tensor from a 6D pose vector, matching a template tensor's device/dtype/batch shape."""

    pose_tensor = torch.from_numpy(get_mat_from_6d_pose_arr(pose_arr)).to(
        device=template_pose.device,
        dtype=template_pose.dtype,
    )
    if template_pose.ndim == 3:
        pose_tensor = pose_tensor.unsqueeze(0)
    return pose_tensor


@dataclass
class SegmentationResult:
    mask: np.ndarray
    bbox_xywh: list[int]


@dataclass
class TrackingTimingBreakdown:
    total_ms: float = 0.0
    two_d_tracker_ms: float = 0.0
    kalman_ms: float = 0.0
    foundationpose_ms: float = 0.0
    publish_ms: float = 0.0
    visualization_ms: float = 0.0


@dataclass
class BboxSelectionState:
    drawing: bool = False
    start_point: tuple[int, int] | None = None
    end_point: tuple[int, int] | None = None
    bbox_xyxy: tuple[tuple[int, int], tuple[int, int]] | None = None
    new_bbox: bool = False

    def reset(self) -> None:
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.bbox_xyxy = None
        self.new_bbox = False


TAB10_RGB = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
)


def _mask_rgba_for_object(obj_id: int | None) -> np.ndarray:
    rgb = np.asarray(TAB10_RGB[(0 if obj_id is None else obj_id) % len(TAB10_RGB)], dtype=np.float32) / 255.0
    return np.concatenate((rgb, np.asarray([0.6], dtype=np.float32)))


def show_mask(image_bgr: np.ndarray, mask: np.ndarray, obj_id: int | None = None) -> np.ndarray:
    """Blend a binary mask over a BGR frame using the same style as the reference tracker."""

    output_image = image_bgr.copy()
    mask_bool = np.asarray(mask, dtype=bool)
    if not mask_bool.any():
        return output_image

    color = _mask_rgba_for_object(obj_id)
    alpha = float(color[3])
    color_bgr = (color[:3][::-1] * 255.0).astype(np.float32)
    masked_pixels = output_image[mask_bool].astype(np.float32)
    masked_pixels *= 1.0 - alpha
    masked_pixels += color_bgr * alpha
    output_image[mask_bool] = masked_pixels.astype(np.uint8)
    return output_image


def draw_rectangle(event: int, x: int, y: int, flags: int, param: BboxSelectionState) -> None:
    """Mouse callback used by the initialization edit-mode window."""

    del flags
    if event == cv2.EVENT_LBUTTONDOWN:
        param.drawing = True
        param.start_point = (x, y)
        param.end_point = (x, y)
        param.bbox_xyxy = None
        param.new_bbox = False
    elif event == cv2.EVENT_MOUSEMOVE and param.drawing:
        param.end_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and param.start_point is not None:
        param.drawing = False
        param.end_point = (x, y)
        param.bbox_xyxy = (param.start_point, param.end_point)
        param.new_bbox = True


def bbox_points_to_xywh(
    bbox_xyxy: tuple[tuple[int, int], tuple[int, int]] | None,
    image_shape: tuple[int, int],
) -> list[float] | None:
    """Convert two drag points into an xywh box clipped to the image."""

    if bbox_xyxy is None:
        return None

    (x0, y0), (x1, y1) = bbox_xyxy
    height, width = image_shape
    x_min = max(0, min(x0, x1))
    y_min = max(0, min(y0, y1))
    x_max = min(width - 1, max(x0, x1))
    y_max = min(height - 1, max(y0, y1))
    box_width = x_max - x_min
    box_height = y_max - y_min
    if box_width <= 0 or box_height <= 0:
        return None
    return [float(x_min), float(y_min), float(box_width), float(box_height)]


def mesh_requires_color_fallback(mesh_path: str) -> bool:
    """Return whether the mesh advertises texture visuals but has no usable image."""

    try:
        import trimesh
    except ModuleNotFoundError:
        return False

    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    visual = getattr(mesh, "visual", None)
    if not isinstance(visual, trimesh.visual.texture.TextureVisuals):
        return False

    material = getattr(visual, "material", None)
    if material is None:
        return False

    return getattr(material, "image", None) is None and getattr(visual, "uv", None) is not None


def inspect_mesh_scale(mesh_path: str, apply_scale: float) -> dict[str, np.ndarray | float] | None:
    """Return basic raw/scaled mesh size diagnostics for unit sanity checks."""

    try:
        import trimesh
    except ModuleNotFoundError:
        return None

    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    raw_extents = vertices.max(axis=0) - vertices.min(axis=0)
    scaled_extents = raw_extents * float(apply_scale)
    return {
        "raw_extents": raw_extents,
        "scaled_extents": scaled_extents,
        "raw_diag": float(np.linalg.norm(raw_extents)),
        "scaled_diag": float(np.linalg.norm(scaled_extents)),
    }


def restore_torch_cpu_default_tensor_type() -> None:
    """Undo FoundationPose's global CUDA default-tensor side effect for the rest of the pipeline."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        torch.set_default_tensor_type(torch.FloatTensor)


def maybe_cuda_synchronize(device: str | torch.device | None) -> None:
    """Synchronize the current CUDA stream when timing GPU work."""

    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.synchronize()
        return
    torch.cuda.synchronize(device=device)


def project_object_points(
    object_points: np.ndarray,
    object_pose_in_camera: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project object-frame 3D points into image coordinates."""

    points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    pose = np.asarray(object_pose_in_camera, dtype=np.float32).reshape(4, 4)
    intrinsics = np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3)

    points_cam = (pose[:3, :3] @ points.T).T + pose[:3, 3]
    depths = points_cam[:, 2]
    valid = depths > 1e-6
    pixels = np.zeros((len(points), 2), dtype=np.int32)
    if valid.any():
        projected = (intrinsics @ points_cam[valid].T).T
        pixels_valid = projected[:, :2] / projected[:, 2:3]
        pixels[valid] = np.round(pixels_valid).astype(np.int32)
    return pixels, valid


def draw_pose_axes_on_image(
    frame_bgr: np.ndarray,
    object_pose_in_camera: np.ndarray,
    camera_matrix: np.ndarray,
    axis_scale: float,
    thickness: int = 2,
) -> np.ndarray:
    """Draw XYZ axes for the tracked 6D pose."""

    axis_points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_scale, 0.0, 0.0],
            [0.0, axis_scale, 0.0],
            [0.0, 0.0, axis_scale],
        ],
        dtype=np.float32,
    )
    pixels, valid = project_object_points(axis_points, object_pose_in_camera, camera_matrix)
    if not valid[0]:
        return frame_bgr

    origin = tuple(pixels[0])
    colors_bgr = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    for index, color in enumerate(colors_bgr, start=1):
        if valid[index]:
            cv2.arrowedLine(
                frame_bgr,
                origin,
                tuple(pixels[index]),
                color=color,
                thickness=thickness,
                line_type=cv2.LINE_AA,
                tipLength=0.15,
            )
    return frame_bgr


def draw_pose_box_on_image(
    frame_bgr: np.ndarray,
    object_pose_in_camera: np.ndarray,
    camera_matrix: np.ndarray,
    object_bbox: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
    near_plane: float = 1e-4,
) -> np.ndarray:
    """Draw a projected 3D bounding box for the tracked object pose with near-plane and viewport clipping."""

    bbox = np.asarray(object_bbox, dtype=np.float32).reshape(2, 3)
    min_xyz, max_xyz = bbox
    corners = np.asarray(
        [
            [min_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], max_xyz[1], max_xyz[2]],
            [min_xyz[0], max_xyz[1], max_xyz[2]],
        ],
        dtype=np.float32,
    )
    pose = np.asarray(object_pose_in_camera, dtype=np.float32).reshape(4, 4)
    intrinsics = np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3)
    corners_cam = (pose[:3, :3] @ corners.T).T + pose[:3, 3]
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    image_rect = (0, 0, int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))

    def clip_edge_to_near_plane(start_cam: np.ndarray, end_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        z_start = float(start_cam[2])
        z_end = float(end_cam[2])
        if z_start < near_plane and z_end < near_plane:
            return None
        if z_start >= near_plane and z_end >= near_plane:
            return start_cam, end_cam
        denom = z_end - z_start
        if abs(denom) < 1e-8:
            return None
        t = (near_plane - z_start) / denom
        intersection = start_cam + t * (end_cam - start_cam)
        if z_start < near_plane:
            return intersection, end_cam
        return start_cam, intersection

    for start, end in edges:
        clipped_edge = clip_edge_to_near_plane(corners_cam[start], corners_cam[end])
        if clipped_edge is None:
            continue
        start_cam, end_cam = clipped_edge
        projected = (intrinsics @ np.stack((start_cam, end_cam), axis=0).T).T
        pixels = np.round(projected[:, :2] / projected[:, 2:3]).astype(np.int32)
        is_visible, clipped_start, clipped_end = cv2.clipLine(
            image_rect,
            tuple(pixels[0]),
            tuple(pixels[1]),
        )
        if not is_visible:
            continue
        cv2.line(
            frame_bgr,
            clipped_start,
            clipped_end,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    return frame_bgr


def ensure_foundationpose_optional_extensions(logger: Any | None = None) -> None:
    """Install lightweight fallbacks for optional FoundationPose extensions."""

    try:
        import Utils
        import torch
    except ModuleNotFoundError:
        return

    if getattr(Utils, "mycpp", None) is None:
        class _MyCppFallback:
            @staticmethod
            def cluster_poses(
                angle_diff: float,
                dist_diff: float,
                poses_in: np.ndarray,
                symmetry_tfs: np.ndarray,
            ) -> np.ndarray:
                del angle_diff, dist_diff, symmetry_tfs
                return poses_in

        Utils.mycpp = _MyCppFallback()
        if logger is not None:
            logger.warning(
                "FoundationPose mycpp extension is not available. "
                "Falling back to the unclustered rotation grid, which may be a bit slower at initialization."
            )

    if not getattr(Utils, "_ros2_dtype_patch_installed", False):
        def _resolve_tensor_device_dtype(*references: object) -> tuple[torch.device, torch.dtype]:
            tensors = [value for value in references if torch.is_tensor(value)]
            for tensor in tensors:
                if tensor.device.type == "cuda":
                    return tensor.device, torch.float32
            if tensors:
                return tensors[0].device, torch.float32
            if torch.cuda.is_available():
                return torch.device("cuda"), torch.float32
            return torch.device("cpu"), torch.float32

        def _as_aligned_tensor(
            value: object,
            device: torch.device,
            dtype: torch.dtype | None = None,
        ) -> torch.Tensor:
            if torch.is_tensor(value):
                if dtype is None:
                    return value.to(device=device)
                return value.to(device=device, dtype=dtype)
            if dtype is None:
                return torch.as_tensor(value, device=device)
            return torch.as_tensor(value, device=device, dtype=dtype)

        def _transform_pts_patched(pts: object, tf: object) -> torch.Tensor:
            device, dtype = _resolve_tensor_device_dtype(pts, tf)
            tf_tensor = _as_aligned_tensor(tf, device=device, dtype=dtype)
            pts_tensor = _as_aligned_tensor(pts, device=device, dtype=dtype)
            if len(tf_tensor.shape) >= 3 and tf_tensor.shape[-3] != pts_tensor.shape[-2]:
                tf_tensor = tf_tensor[..., None, :, :]
            return (tf_tensor[..., :-1, :-1] @ pts_tensor[..., None] + tf_tensor[..., :-1, -1:])[..., 0]

        def _transform_dirs_patched(dirs: object, tf: object) -> torch.Tensor:
            device, dtype = _resolve_tensor_device_dtype(dirs, tf)
            tf_tensor = _as_aligned_tensor(tf, device=device, dtype=dtype)
            dirs_tensor = _as_aligned_tensor(dirs, device=device, dtype=dtype)
            if len(tf_tensor.shape) >= 3 and tf_tensor.shape[-3] != dirs_tensor.shape[-2]:
                tf_tensor = tf_tensor[..., None, :, :]
            return (tf_tensor[..., :3, :3] @ dirs_tensor[..., None])[..., 0]

        def _compute_crop_window_tf_batch_patched(
            pts: object = None,
            H: object = None,
            W: object = None,
            poses: object = None,
            K: object = None,
            crop_ratio: float = 1.2,
            out_size: tuple[float, float] | None = None,
            rgb: object = None,
            uvs: object = None,
            method: str = "min_box",
            mesh_diameter: float | None = None,
        ) -> torch.Tensor:
            del H, W, rgb, uvs
            if poses is None or K is None or out_size is None or mesh_diameter is None:
                raise ValueError("poses, K, out_size, and mesh_diameter are required.")
            if method != "box_3d":
                raise RuntimeError

            device, dtype = _resolve_tensor_device_dtype(poses, K, pts)
            poses_tensor = _as_aligned_tensor(poses, device=device, dtype=dtype)
            K_tensor = _as_aligned_tensor(K, device=device, dtype=dtype)
            B = len(poses_tensor)
            radius = float(mesh_diameter) * float(crop_ratio) / 2.0
            offsets = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [radius, 0.0, 0.0],
                    [-radius, 0.0, 0.0],
                    [0.0, radius, 0.0],
                    [0.0, -radius, 0.0],
                ],
                device=device,
                dtype=dtype,
            )
            pts_tensor = poses_tensor[:, :3, 3].reshape(-1, 1, 3) + offsets.reshape(1, -1, 3)
            projected = (K_tensor @ pts_tensor.reshape(-1, 3).T).T
            uvs_tensor = projected[:, :2] / projected[:, 2:3]
            uvs_tensor = uvs_tensor.reshape(B, -1, 2)
            center = uvs_tensor[:, 0]
            crop_radius = torch.abs(uvs_tensor - center.reshape(-1, 1, 2)).reshape(B, -1).max(dim=-1)[0]
            left = crop_radius.neg() + center[:, 0]
            right = crop_radius + center[:, 0]
            top = crop_radius.neg() + center[:, 1]
            bottom = crop_radius + center[:, 1]
            left = left.round()
            right = right.round()
            top = top.round()
            bottom = bottom.round()

            tf_tensor = torch.eye(3, device=device, dtype=dtype)[None].expand(B, -1, -1).contiguous()
            tf_tensor[:, 0, 2] = -left
            tf_tensor[:, 1, 2] = -top
            scale_tf = torch.eye(3, device=device, dtype=dtype)[None].expand(B, -1, -1).contiguous()
            scale_tf[:, 0, 0] = float(out_size[0]) / (right - left)
            scale_tf[:, 1, 1] = float(out_size[1]) / (bottom - top)
            return scale_tf @ tf_tensor

        def _depth2xyzmap_batch_patched(depths: object, Ks: object, zfar: float) -> torch.Tensor:
            device, dtype = _resolve_tensor_device_dtype(depths, Ks)
            depths_tensor = _as_aligned_tensor(depths, device=device, dtype=dtype)
            Ks_tensor = _as_aligned_tensor(Ks, device=device, dtype=dtype)
            bs = depths_tensor.shape[0]
            invalid_mask = (depths_tensor < 0.001) | (depths_tensor > zfar)
            H, W = depths_tensor.shape[-2:]
            vs, us = torch.meshgrid(
                torch.arange(0, H, device=device, dtype=dtype),
                torch.arange(0, W, device=device, dtype=dtype),
                indexing="ij",
            )
            vs = vs.reshape(-1)[None].expand(bs, -1)
            us = us.reshape(-1)[None].expand(bs, -1)
            zs = depths_tensor.reshape(bs, -1)
            Ks_tensor = Ks_tensor[:, None].expand(bs, zs.shape[-1], 3, 3)
            xs = (us - Ks_tensor[..., 0, 2]) * zs / Ks_tensor[..., 0, 0]
            ys = (vs - Ks_tensor[..., 1, 2]) * zs / Ks_tensor[..., 1, 1]
            pts_tensor = torch.stack([xs, ys, zs], dim=-1)
            xyz_maps = pts_tensor.reshape(bs, H, W, 3)
            xyz_maps[invalid_mask] = 0
            return xyz_maps

        Utils.transform_pts = _transform_pts_patched
        Utils.transform_dirs = _transform_dirs_patched
        Utils.compute_crop_window_tf_batch = _compute_crop_window_tf_batch_patched
        Utils.depth2xyzmap_batch = _depth2xyzmap_batch_patched
        Utils._ros2_dtype_patch_installed = True
        if logger is not None:
            logger.info("Installed runtime dtype/device compatibility patches for FoundationPose utilities.")

    try:
        import estimater
    except ModuleNotFoundError:
        return

    if getattr(estimater, "_ros2_device_patch_installed", False):
        return

    def _infer_estimator_device(estimator: object) -> torch.device:
        mesh_tensors = getattr(estimator, "mesh_tensors", None)
        if isinstance(mesh_tensors, dict):
            for value in mesh_tensors.values():
                if torch.is_tensor(value):
                    return value.device
        for tensor_name in ("pts", "normals", "rot_grid", "symmetry_tfs", "poses", "pose_last"):
            value = getattr(estimator, tensor_name, None)
            if torch.is_tensor(value):
                return value.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_tf_to_centered_mesh_patched(self: object) -> torch.Tensor:
        device = _infer_estimator_device(self)
        tf_to_center = torch.eye(4, dtype=torch.float32, device=device)
        tf_to_center[:3, 3] = -torch.as_tensor(self.model_center, device=device, dtype=torch.float32)
        return tf_to_center

    estimater.FoundationPose.get_tf_to_centered_mesh = _get_tf_to_centered_mesh_patched
    estimater._ros2_device_patch_installed = True
    if logger is not None:
        logger.info("Installed runtime device compatibility patch for FoundationPose centered-mesh transforms.")


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
        self.device = device
        self.use_autocast = (
            tracker_name == "efficient_tam"
            and isinstance(device, str)
            and device.startswith("cuda")
            and torch.cuda.is_available()
        )

        if tracker_name == "sam2":
            wrapper = Sam2Wrapper(device=device)
            self.predictor = wrapper.load_camera_predictor(
                config_file=config_file,
                checkpoint_path=checkpoint_path,
            )
        elif tracker_name == "efficient_tam":
            wrapper = EfficientTrackAnythingWrapper(device=device)
            self.predictor = wrapper.load_camera_predictor(
                config_file=config_file,
                checkpoint_path=checkpoint_path,
                vos_optimized=True,
            )
        else:
            raise ValueError(f"Unsupported tracker: {tracker_name}")

    @contextlib.contextmanager
    def _predictor_context(self) -> Any:
        if self.use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                yield
        else:
            yield

    def synchronize(self) -> None:
        maybe_cuda_synchronize(self.device)

    def initialize(
        self,
        frame_rgb: np.ndarray,
        init_bbox_xywh: list[float],
    ) -> SegmentationResult:
        with self._predictor_context():
            self.predictor.load_first_frame(frame_rgb)
            _, obj_ids, mask_logits = self.predictor.add_new_prompt(
                frame_idx=0,
                obj_id=self.object_id,
                bbox=bbox_xywh_to_xyxy(init_bbox_xywh),
            )

        mask = self._extract_object_mask(obj_ids, mask_logits)
        return SegmentationResult(mask=mask, bbox_xywh=binary_mask_to_bbox_xywh(mask))

    def track(self, frame_rgb: np.ndarray) -> SegmentationResult:
        with self._predictor_context():
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
        self.init_prompt_logged = False
        self.visualization_window_name = f"PoseTracker {args.tracker}"
        self.edit_window_name = f"Edit Mode {args.tracker}"
        self.visualization_window_visible = False
        self.edit_window_visible = False
        self.visualization_enabled = True
        self.edit_mode = False
        self.selection_state = BboxSelectionState()
        self.pending_init_color_rgb: np.ndarray | None = None
        self.pending_init_depth_m: np.ndarray | None = None
        self.pending_init_camera_matrix: np.ndarray | None = None
        self.pending_init_segmentation: SegmentationResult | None = None
        self.pending_init_bbox_xywh: list[float] | None = None
        self.last_visualization_time: float | None = None
        self.last_visualization_render_time: float | None = None
        self.visualization_fps = 0.0
        self.visualization_rate_hz = max(0.0, args.visualization_rate_hz)
        self.tracking_window_canvas: np.ndarray | None = None
        self.log_timing = args.log_timing
        self.timing_report_every = max(1, args.timing_report_every)
        self.timing_breakdown_sums = TrackingTimingBreakdown()
        self.timing_breakdown_count = 0
        self.camera_matrix_fallback = (
            np.asarray(args.cam_k, dtype=np.float32).reshape(3, 3) if args.cam_k is not None else None
        )

        force_apply_color = [int(v) for v in args.apply_color] if args.force_apply_color else None
        if force_apply_color is None and mesh_requires_color_fallback(args.mesh_path):
            force_apply_color = [int(v) for v in args.apply_color]
            self.node.get_logger().warning(
                "Mesh texture metadata is incomplete (UVs present but no texture image). "
                "Enabling the pure-color mesh fallback automatically."
            )
        mesh_scale_info = inspect_mesh_scale(args.mesh_path, args.apply_scale)
        if mesh_scale_info is not None:
            raw_extents = np.asarray(mesh_scale_info["raw_extents"])
            scaled_extents = np.asarray(mesh_scale_info["scaled_extents"])
            raw_diag = float(mesh_scale_info["raw_diag"])
            scaled_diag = float(mesh_scale_info["scaled_diag"])
            self.node.get_logger().info(
                "Mesh size diagnostics: "
                f"raw extents={raw_extents.tolist()} raw diag={raw_diag:.6f}, "
                f"apply_scale={args.apply_scale}, "
                f"scaled extents={scaled_extents.tolist()} scaled diag={scaled_diag:.6f}"
            )
            if (
                args.apply_scale != 1.0
                and raw_diag >= 0.05
                and scaled_diag <= 0.01
            ):
                self.node.get_logger().warning(
                    "The scaled mesh is very small compared with the raw mesh. "
                    "This often means the mesh already uses meters and `--apply-scale` is shrinking it too much. "
                    "If the tracked pose looks tiny or unstable, try `--apply-scale 1.0`."
                )
        ensure_foundationpose_optional_extensions(self.node.get_logger())
        self.foundationpose = FoundationPoseWrapper(device=args.device, debug_dir=args.debug_dir)
        self.foundationpose.load(
            mesh_path=args.mesh_path,
            apply_scale=args.apply_scale,
            force_apply_color=force_apply_color,
        )
        import trimesh

        oriented_to_origin, oriented_extents = trimesh.bounds.oriented_bounds(self.foundationpose.mesh)
        self.reference_pose_transform = np.linalg.inv(oriented_to_origin).astype(np.float32)
        self.reference_bbox = np.stack((-oriented_extents / 2.0, oriented_extents / 2.0), axis=0).astype(np.float32)
        self.reference_pose_axis_scale = 0.1
        self.reference_draw_posed_3d_box = None
        self.reference_draw_xyz_axis = None
        try:
            import Utils
        except ModuleNotFoundError:
            pass
        else:
            self.reference_draw_posed_3d_box = getattr(Utils, "draw_posed_3d_box", None)
            self.reference_draw_xyz_axis = getattr(Utils, "draw_xyz_axis", None)
        self.segmenter = CameraTrackerAdapter(
            tracker_name=args.tracker,
            config_file=args.tracker_config,
            checkpoint_path=args.tracker_checkpoint,
            device=args.device,
            object_id=args.object_id,
            mask_threshold=args.mask_threshold,
        )
        self.kalman_filter = KalmanFilter6D(args.kf_measurement_noise_scale) if args.kalman_filter else None
        self.kf_mean: np.ndarray | None = None
        self.kf_covariance: np.ndarray | None = None

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
        if self.kalman_filter is not None:
            self.node.get_logger().info(
                "Kalman filter enabled for pose smoothing with "
                f"measurement noise scale={args.kf_measurement_noise_scale}."
            )
        if self.log_timing:
            self.node.get_logger().info(
                "Per-frame timing stats enabled. "
                f"Reporting rolling averages every {self.timing_report_every} tracking frames."
            )
        if self.visualization_rate_hz > 0.0:
            self.node.get_logger().info(
                f"Visualization refresh is capped at {self.visualization_rate_hz:.1f} Hz to reduce display overhead."
            )
        self.node.get_logger().info(
            "A live RGB window will appear. Press 'e' to enter edit mode, draw a box, then press 'e' again to confirm initialization."
        )

    def _on_color(self, msg: object) -> None:
        with self.lock:
            self.latest_color_msg = msg

    def _on_depth(self, msg: object) -> None:
        with self.lock:
            self.latest_depth_msg = msg

    def _on_camera_info(self, msg: object) -> None:
        with self.lock:
            self.latest_camera_info_msg = msg

    def _as_float32_camera_matrix(self, camera_matrix: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3))

    def _foundationpose_target_device(self, estimator: object) -> torch.device:
        mesh_tensors = getattr(estimator, "mesh_tensors", None)
        if isinstance(mesh_tensors, dict):
            for value in mesh_tensors.values():
                if torch.is_tensor(value):
                    return value.device
        for attr_name in ("pts", "normals", "rot_grid", "symmetry_tfs", "poses", "pose_last"):
            value = getattr(estimator, attr_name, None)
            if torch.is_tensor(value):
                return value.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _normalize_foundationpose_state(self, estimator: object) -> None:
        target_device = self._foundationpose_target_device(estimator)
        for attr_name in ("pose_last", "rot_grid", "pts", "normals", "symmetry_tfs", "poses"):
            value = getattr(estimator, attr_name, None)
            if torch.is_tensor(value):
                target_dtype = torch.float32 if value.is_floating_point() else value.dtype
                if value.device != target_device or value.dtype != target_dtype:
                    setattr(estimator, attr_name, value.to(device=target_device, dtype=target_dtype))

        mesh_tensors = getattr(estimator, "mesh_tensors", None)
        if isinstance(mesh_tensors, dict):
            for key, value in mesh_tensors.items():
                if torch.is_tensor(value) and value.device != target_device:
                    mesh_tensors[key] = value.to(device=target_device)

    def _apply_kalman_filter_to_pose_last(
        self,
        estimator: object,
        camera_matrix_tensor: torch.Tensor,
        bbox_xywh: list[int],
    ) -> None:
        if self.kalman_filter is None or self.kf_mean is None or self.kf_covariance is None:
            return

        self.kf_mean, self.kf_covariance = self.kalman_filter.update(
            self.kf_mean,
            self.kf_covariance,
            get_6d_pose_arr_from_mat(estimator.pose_last),
        )

        if bbox_xywh[2] > 0 and bbox_xywh[3] > 0:
            measurement_xy = np.asarray(
                get_pose_xy_from_image_point(
                    ob_in_cam=estimator.pose_last,
                    camera_matrix=camera_matrix_tensor,
                    x=float(bbox_xywh[0] + bbox_xywh[2] / 2.0),
                    y=float(bbox_xywh[1] + bbox_xywh[3] / 2.0),
                ),
                dtype=np.float32,
            )
            self.kf_mean, self.kf_covariance = self.kalman_filter.update_from_xy(
                self.kf_mean,
                self.kf_covariance,
                measurement_xy,
            )

        estimator.pose_last = pose_tensor_from_6d_pose_arr(self.kf_mean[:6], estimator.pose_last)

    def _record_tracking_timing(self, breakdown: TrackingTimingBreakdown) -> None:
        if not self.log_timing:
            return

        self.timing_breakdown_sums.total_ms += breakdown.total_ms
        self.timing_breakdown_sums.two_d_tracker_ms += breakdown.two_d_tracker_ms
        self.timing_breakdown_sums.kalman_ms += breakdown.kalman_ms
        self.timing_breakdown_sums.foundationpose_ms += breakdown.foundationpose_ms
        self.timing_breakdown_sums.publish_ms += breakdown.publish_ms
        self.timing_breakdown_sums.visualization_ms += breakdown.visualization_ms
        self.timing_breakdown_count += 1

        if self.timing_breakdown_count < self.timing_report_every:
            return

        n = float(self.timing_breakdown_count)
        averages = TrackingTimingBreakdown(
            total_ms=self.timing_breakdown_sums.total_ms / n,
            two_d_tracker_ms=self.timing_breakdown_sums.two_d_tracker_ms / n,
            kalman_ms=self.timing_breakdown_sums.kalman_ms / n,
            foundationpose_ms=self.timing_breakdown_sums.foundationpose_ms / n,
            publish_ms=self.timing_breakdown_sums.publish_ms / n,
            visualization_ms=self.timing_breakdown_sums.visualization_ms / n,
        )
        fps_str = f"{1000.0 / averages.total_ms:.2f}" if averages.total_ms > 1e-6 else "inf"
        self.node.get_logger().info(
            "Tracking timing over the last "
            f"{self.timing_breakdown_count} frames: "
            f"total={averages.total_ms:.1f} ms, "
            f"2d_tracker={averages.two_d_tracker_ms:.1f} ms, "
            f"kalman={averages.kalman_ms:.1f} ms, "
            f"foundationpose={averages.foundationpose_ms:.1f} ms, "
            f"publish={averages.publish_ms:.1f} ms, "
            f"visualization={averages.visualization_ms:.1f} ms, "
            f"fps={fps_str}"
        )
        self.timing_breakdown_sums = TrackingTimingBreakdown()
        self.timing_breakdown_count = 0

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
            self.node.get_logger().error(f"Tracking step failed: {exc}\n{traceback.format_exc()}")
        finally:
            self.processing = False

    def _process_frame_pair(
        self,
        color_msg: object,
        depth_msg: object,
        camera_info_msg: object | None,
        camera_matrix: np.ndarray,
    ) -> None:
        frame_start_time = time.perf_counter()
        timing_breakdown = TrackingTimingBreakdown()
        is_tracking_frame = False
        color_rgb = image_msg_to_rgb8(color_msg)
        depth_m = depth_msg_to_meters(depth_msg, depth_scale_for_uint16=self.args.depth_scale)
        camera_matrix = self._as_float32_camera_matrix(camera_matrix)

        if not self.initialized:
            init_result = self._try_initialize_from_live_bbox(
                color_rgb=color_rgb,
                depth_m=depth_m,
                camera_matrix=camera_matrix,
            )
            if init_result is None:
                return
            segmentation, pose = init_result
        else:
            is_tracking_frame = True
            self.segmenter.synchronize()
            step_start_time = time.perf_counter()
            segmentation = self.segmenter.track(color_rgb)
            self.segmenter.synchronize()
            timing_breakdown.two_d_tracker_ms = (time.perf_counter() - step_start_time) * 1000.0
            estimator = self.foundationpose.get()
            self._normalize_foundationpose_state(estimator)
            bbox = segmentation.bbox_xywh
            camera_matrix_tensor = torch.as_tensor(
                camera_matrix,
                device=estimator.pose_last.device,
                dtype=estimator.pose_last.dtype,
            )
            if self.kalman_filter is not None:
                step_start_time = time.perf_counter()
                self._apply_kalman_filter_to_pose_last(
                    estimator=estimator,
                    camera_matrix_tensor=camera_matrix_tensor,
                    bbox_xywh=bbox,
                )
                timing_breakdown.kalman_ms = (time.perf_counter() - step_start_time) * 1000.0
            elif bbox[2] > 0 and bbox[3] > 0:
                estimator.pose_last = adjust_pose_to_image_point(
                    ob_in_cam=estimator.pose_last,
                    camera_matrix=camera_matrix_tensor,
                    x=float(bbox[0] + bbox[2] / 2.0),
                    y=float(bbox[1] + bbox[3] / 2.0),
                )

            try:
                maybe_cuda_synchronize(self.args.device)
                step_start_time = time.perf_counter()
                pose = self.foundationpose.track_one(
                    rgb=color_rgb,
                    depth=depth_m,
                    camera_matrix=camera_matrix,
                    iteration=self.args.track_refine_iter,
                )
                maybe_cuda_synchronize(self.args.device)
                timing_breakdown.foundationpose_ms = (time.perf_counter() - step_start_time) * 1000.0
                if self.kalman_filter is not None and self.kf_mean is not None and self.kf_covariance is not None:
                    step_start_time = time.perf_counter()
                    self.kf_mean, self.kf_covariance = self.kalman_filter.predict(
                        self.kf_mean,
                        self.kf_covariance,
                    )
                    timing_breakdown.kalman_ms += (time.perf_counter() - step_start_time) * 1000.0
            finally:
                restore_torch_cpu_default_tensor_type()

        frame_id = (
            camera_info_msg.header.frame_id
            if camera_info_msg is not None and camera_info_msg.header.frame_id
            else color_msg.header.frame_id
        )
        step_start_time = time.perf_counter()
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
        timing_breakdown.publish_ms = (time.perf_counter() - step_start_time) * 1000.0

        step_start_time = time.perf_counter()
        self._update_tracking_visualization(
            color_rgb=color_rgb,
            segmentation=segmentation,
            pose=pose,
            camera_matrix=camera_matrix,
        )
        timing_breakdown.visualization_ms = (time.perf_counter() - step_start_time) * 1000.0
        timing_breakdown.total_ms = (time.perf_counter() - frame_start_time) * 1000.0
        if is_tracking_frame:
            self._record_tracking_timing(timing_breakdown)

    def _update_tracking_visualization(
        self,
        color_rgb: np.ndarray,
        segmentation: SegmentationResult,
        pose: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> None:
        if not self.initialized or not self.visualization_enabled:
            return

        if self.visualization_rate_hz > 0.0:
            now = time.perf_counter()
            min_frame_period = 1.0 / self.visualization_rate_hz
            if (
                self.last_visualization_render_time is not None
                and now - self.last_visualization_render_time < min_frame_period
            ):
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    self.visualization_enabled = False
                    self._destroy_visualization_window()
                    self.node.get_logger().info(
                        "Visualization hidden by user input. Tracking continues in the background."
                    )
                return

        self._update_visualization_fps()
        self._show_visualization_window(
            self._build_tracking_window_frame(color_rgb, segmentation, pose, camera_matrix)
        )
        self.last_visualization_render_time = time.perf_counter()
        if not self.visualization_enabled:
            return
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.visualization_enabled = False
            self._destroy_visualization_window()
            self.node.get_logger().info("Visualization hidden by user input. Tracking continues in the background.")

    def _update_visualization_fps(self) -> None:
        now = time.time()
        if self.last_visualization_time is not None:
            dt = now - self.last_visualization_time
            if dt > 1e-6:
                instantaneous_fps = 1.0 / dt
                if self.visualization_fps <= 0.0:
                    self.visualization_fps = instantaneous_fps
                else:
                    self.visualization_fps = 0.85 * self.visualization_fps + 0.15 * instantaneous_fps
        self.last_visualization_time = now

    def _annotate_frame_in_place(self, frame_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
        y = 30
        for line in lines:
            cv2.putText(
                frame_bgr,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 32
        return frame_bgr

    def _show_visualization_window(self, frame_bgr: np.ndarray) -> None:
        if self.visualization_window_visible:
            try:
                if cv2.getWindowProperty(self.visualization_window_name, cv2.WND_PROP_VISIBLE) < 1:
                    if self.initialized:
                        self.visualization_enabled = False
                        self.visualization_window_visible = False
                        self.node.get_logger().info(
                            "Visualization window closed. Tracking continues without the live display."
                        )
                        return
                    self.visualization_window_visible = False
            except cv2.error:
                self.visualization_window_visible = False

        if self.initialized and not self.visualization_enabled:
            return

        if not self.visualization_window_visible:
            cv2.namedWindow(self.visualization_window_name, cv2.WINDOW_NORMAL)
            self.visualization_window_visible = True

        cv2.imshow(self.visualization_window_name, frame_bgr)

    def _show_edit_window(self, frame_bgr: np.ndarray) -> None:
        if self.edit_window_visible:
            try:
                if cv2.getWindowProperty(self.edit_window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self._cancel_edit_mode("Edit mode window closed. Press 'e' to start again.")
                    return
            except cv2.error:
                self._cancel_edit_mode("Edit mode window closed. Press 'e' to start again.")
                return

        if not self.edit_window_visible:
            cv2.namedWindow(self.edit_window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.edit_window_name, draw_rectangle, self.selection_state)
            self.edit_window_visible = True

        cv2.imshow(self.edit_window_name, frame_bgr)

    def _destroy_visualization_window(self) -> None:
        if not self.visualization_window_visible:
            self.last_visualization_render_time = None
            return
        try:
            cv2.destroyWindow(self.visualization_window_name)
        except cv2.error:
            cv2.destroyAllWindows()
        self.visualization_window_visible = False
        self.last_visualization_render_time = None

    def _destroy_edit_window(self) -> None:
        if not self.edit_window_visible:
            return
        try:
            cv2.destroyWindow(self.edit_window_name)
        except cv2.error:
            cv2.destroyAllWindows()
        self.edit_window_visible = False

    def _build_live_window_frame(self, color_rgb: np.ndarray) -> np.ndarray:
        frame_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        lines = [f"FPS: {self.visualization_fps:.2f}"]
        if self.edit_mode:
            lines.append("Edit mode active")
            if self.pending_init_segmentation is not None:
                lines.append("Press 'e' again to confirm")
            else:
                lines.append("Draw a box in the edit window")
        else:
            lines.append("Press 'e' to enter edit mode")
        return self._annotate_frame_in_place(frame_bgr, lines)

    def _build_reference_pose_panel(
        self,
        base_bgr: np.ndarray,
        pose: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> np.ndarray:
        reference_pose = np.asarray(pose, dtype=np.float32).reshape(4, 4) @ self.reference_pose_transform
        pose_overlay_bgr = base_bgr.copy()
        draw_pose_box_on_image(
            pose_overlay_bgr,
            object_pose_in_camera=reference_pose,
            camera_matrix=camera_matrix,
            object_bbox=self.reference_bbox,
            color=(0, 255, 0),
            thickness=3,
        )
        draw_pose_axes_on_image(
            pose_overlay_bgr,
            object_pose_in_camera=reference_pose,
            camera_matrix=camera_matrix,
            axis_scale=self.reference_pose_axis_scale,
            thickness=3,
        )
        draw_pose_box_on_image(
            pose_overlay_bgr,
            object_pose_in_camera=reference_pose,
            camera_matrix=camera_matrix,
            object_bbox=self.reference_bbox,
            color=(0, 255, 0),
            thickness=3,
        )
        return pose_overlay_bgr

    def _compose_tracking_window_frame(
        self,
        left_bgr: np.ndarray,
        middle_bgr: np.ndarray,
        right_bgr: np.ndarray,
    ) -> np.ndarray:
        panel_height, panel_width = left_bgr.shape[:2]
        canvas_shape = (panel_height, panel_width * 3, 3)
        if self.tracking_window_canvas is None or self.tracking_window_canvas.shape != canvas_shape:
            self.tracking_window_canvas = np.empty(canvas_shape, dtype=np.uint8)

        canvas = self.tracking_window_canvas
        canvas[:, :panel_width] = left_bgr
        canvas[:, panel_width : 2 * panel_width] = middle_bgr
        canvas[:, 2 * panel_width :] = right_bgr
        return canvas

    def _build_tracking_window_frame(
        self,
        color_rgb: np.ndarray,
        segmentation: SegmentationResult,
        pose: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> np.ndarray:
        base_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        left = base_bgr.copy()
        overlay_bgr = show_mask(base_bgr, segmentation.mask, obj_id=self.args.object_id)
        bbox = segmentation.bbox_xywh
        if bbox[2] > 0 and bbox[3] > 0:
            x, y, width, height = (int(v) for v in bbox)
            cv2.rectangle(overlay_bgr, (x, y), (x + width, y + height), (0, 255, 0), 2)
        pose_overlay_bgr = self._build_reference_pose_panel(base_bgr=base_bgr, pose=pose, camera_matrix=camera_matrix)

        self._annotate_frame_in_place(left, [f"RGB  FPS: {self.visualization_fps:.2f}"])
        self._annotate_frame_in_place(overlay_bgr, ["2D Tracker Overlay"])
        self._annotate_frame_in_place(
            pose_overlay_bgr,
            ["Reference Pose Overlay", "6D pose + oriented bbox", "Press q/Esc to hide visualization"],
        )
        return self._compose_tracking_window_frame(left, overlay_bgr, pose_overlay_bgr)

    def _build_edit_window_frame(self) -> np.ndarray:
        if self.pending_init_color_rgb is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        if self.pending_init_segmentation is not None:
            frame_bgr = show_mask(
                cv2.cvtColor(self.pending_init_color_rgb, cv2.COLOR_RGB2BGR),
                self.pending_init_segmentation.mask,
                obj_id=self.args.object_id,
            )
        else:
            frame_bgr = cv2.cvtColor(self.pending_init_color_rgb, cv2.COLOR_RGB2BGR)

        if self.selection_state.bbox_xyxy is not None:
            cv2.rectangle(
                frame_bgr,
                self.selection_state.bbox_xyxy[0],
                self.selection_state.bbox_xyxy[1],
                (0, 255, 0),
                2,
            )
        elif self.selection_state.drawing and self.selection_state.start_point and self.selection_state.end_point:
            cv2.rectangle(
                frame_bgr,
                self.selection_state.start_point,
                self.selection_state.end_point,
                (0, 255, 0),
                2,
            )

        lines = ["Draw a box with the mouse", "Press 'e' to confirm", "Press q/Esc to cancel"]
        if self.pending_init_segmentation is None:
            lines.insert(1, "Release the mouse to preview the 2D mask")
        else:
            lines.insert(1, "2D tracker preview ready")
        return self._annotate_frame_in_place(frame_bgr, lines)

    def _enter_edit_mode(self, color_rgb: np.ndarray, depth_m: np.ndarray, camera_matrix: np.ndarray) -> None:
        self.edit_mode = True
        self.selection_state.reset()
        self.pending_init_color_rgb = color_rgb.copy()
        self.pending_init_depth_m = depth_m.copy()
        self.pending_init_camera_matrix = camera_matrix.copy()
        self.pending_init_segmentation = None
        self.pending_init_bbox_xywh = None
        self.node.get_logger().info("Edit mode enabled. Draw a box, then press 'e' again to confirm.")

    def _cancel_edit_mode(self, reason: str) -> None:
        self.edit_mode = False
        self.selection_state.reset()
        self.pending_init_color_rgb = None
        self.pending_init_depth_m = None
        self.pending_init_camera_matrix = None
        self.pending_init_segmentation = None
        self.pending_init_bbox_xywh = None
        self._destroy_edit_window()
        if reason:
            self.node.get_logger().info(reason)

    def _update_pending_initialization_preview(self) -> None:
        if not self.edit_mode or not self.selection_state.new_bbox or self.pending_init_color_rgb is None:
            return

        self.selection_state.new_bbox = False
        bbox_xywh = bbox_points_to_xywh(self.selection_state.bbox_xyxy, self.pending_init_color_rgb.shape[:2])
        if bbox_xywh is None:
            self.pending_init_segmentation = None
            self.pending_init_bbox_xywh = None
            self.node.get_logger().info("Initialization box selection cancelled. Draw a new box to try again.")
            return

        segmentation = self.segmenter.initialize(
            frame_rgb=self.pending_init_color_rgb,
            init_bbox_xywh=bbox_xywh,
        )
        if segmentation.bbox_xywh[2] <= 0 or segmentation.bbox_xywh[3] <= 0:
            self.pending_init_segmentation = None
            self.pending_init_bbox_xywh = None
            self.node.get_logger().warning(
                "Initialization preview failed because the 2D tracker produced an empty mask. Draw a new box to try again."
            )
            return

        self.pending_init_segmentation = segmentation
        self.pending_init_bbox_xywh = bbox_xywh
        self.node.get_logger().info("2D tracker preview updated. Press 'e' again to confirm initialization.")

    def shutdown(self) -> None:
        self._destroy_edit_window()
        self._destroy_visualization_window()
        self.node.destroy_node()

    def _try_initialize_from_live_bbox(
        self,
        color_rgb: np.ndarray,
        depth_m: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> tuple[SegmentationResult, torch.Tensor] | None:
        self._update_visualization_fps()
        self._show_visualization_window(self._build_live_window_frame(color_rgb))

        if not self.init_prompt_logged:
            self.node.get_logger().info(
                "Waiting for initialization. Press 'e' in the live RGB window to enter edit mode."
            )
            self.init_prompt_logged = True

        if self.edit_mode:
            self._update_pending_initialization_preview()
            self._show_edit_window(self._build_edit_window_frame())

        key = cv2.waitKey(1) & 0xFF

        if not self.edit_mode:
            if key == ord("e"):
                self._enter_edit_mode(color_rgb, depth_m, camera_matrix)
            return None

        if key in (27, ord("q")):
            self._cancel_edit_mode("Edit mode cancelled. Press 'e' to start again.")
            return None

        if key != ord("e"):
            return None

        if (
            self.pending_init_segmentation is None
            or self.pending_init_color_rgb is None
            or self.pending_init_depth_m is None
            or self.pending_init_camera_matrix is None
        ):
            self.node.get_logger().info("Draw a bounding box before confirming initialization.")
            return None

        try:
            pose = self.foundationpose.register(
                rgb=self.pending_init_color_rgb,
                depth=self.pending_init_depth_m,
                camera_matrix=self.pending_init_camera_matrix,
                object_mask=self.pending_init_segmentation.mask.astype(np.uint8) * 255,
                iteration=self.args.est_refine_iter,
            )
        finally:
            restore_torch_cpu_default_tensor_type()
        self._normalize_foundationpose_state(self.foundationpose.get())
        estimator = self.foundationpose.get()
        if self.kalman_filter is not None and getattr(estimator, "pose_last", None) is not None:
            self.kf_mean, self.kf_covariance = self.kalman_filter.initiate(
                get_6d_pose_arr_from_mat(estimator.pose_last)
            )
        segmentation = self.pending_init_segmentation
        self.initialized = True
        self._cancel_edit_mode("")
        self.node.get_logger().info("Tracker initialized from the user-confirmed edit-mode bounding box.")
        return segmentation, pose


def _parse_json_list(value: str) -> list[float]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError(f"Expected a JSON list, got: {value}")
    return [float(v) for v in parsed]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live ROS2 object pose tracker using FoundationPose.")
    parser.add_argument("--tracker", choices=["sam2", "efficient_tam"], default="sam2")
    parser.add_argument("--tracker-config", type=str, default=None)
    parser.add_argument("--tracker-checkpoint", type=str, required=True)
    parser.add_argument("--mesh-path", type=str, required=True)
    parser.add_argument("--color-topic", type=str, default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", type=str, default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic", type=str, default="/camera/color/camera_info")
    parser.add_argument("--pose-topic", type=str, default="/vision/tracked_pose")
    parser.add_argument("--mask-topic", type=str, default="/vision/tracked_mask")
    parser.add_argument("--publish-mask", action="store_true")
    parser.add_argument("--process-rate-hz", type=float, default=30.0)
    parser.add_argument("--max-sync-offset-sec", type=float, default=0.08)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--cam-k", type=_parse_json_list, default=None, help='Optional JSON list with 9 intrinsics.')
    parser.add_argument("--est-refine-iter", type=int, default=10)
    parser.add_argument("--track-refine-iter", type=int, default=5)
    parser.add_argument("--apply-scale", type=float, default=1.0)
    parser.add_argument("--force-apply-color", action="store_true")
    parser.add_argument("--apply-color", type=_parse_json_list, default=[0.0, 159.0, 237.0])
    parser.add_argument(
        "--kalman-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the reference-style 6D Kalman filter between the 2D tracker anchor and FoundationPose.",
    )
    parser.add_argument(
        "--kf-measurement-noise-scale",
        type=float,
        default=0.05,
        help="Kalman measurement noise scale; larger values smooth more aggressively.",
    )
    parser.add_argument(
        "--log-timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log rolling timing breakdowns for initialized tracking frames.",
    )
    parser.add_argument(
        "--timing-report-every",
        type=int,
        default=30,
        help="Number of tracking frames to average before printing timing stats.",
    )
    parser.add_argument(
        "--visualization-rate-hz",
        type=float,
        default=10.0,
        help="Cap live visualization refresh to this rate to reduce display overhead. Use 0 to redraw every frame.",
    )
    parser.add_argument("--debug-dir", type=str, default="./debug/foundationpose_ros2")
    return parser


def _normalize_hydra_config_path(config_file: str) -> str:
    """Convert absolute or repo-relative config file paths into Hydra config names."""

    config_path = Path(config_file).expanduser()
    path_parts = config_path.parts
    if "configs" in path_parts:
        configs_index = path_parts.index("configs")
        return Path(*path_parts[configs_index:]).as_posix()
    return config_file


def _resolve_tracker_config(args: argparse.Namespace) -> str:
    if args.tracker_config:
        return _normalize_hydra_config_path(args.tracker_config)
    if args.tracker == "sam2":
        return "configs/sam2.1/sam2.1_hiera_t.yaml"
    return "configs/efficienttam/efficienttam_ti_512x512.yaml"


def main() -> None:
    args = build_argparser().parse_args()
    args.tracker_config = _resolve_tracker_config(args)
    args.mesh_path = str(Path(args.mesh_path).expanduser().resolve())
    args.tracker_checkpoint = str(Path(args.tracker_checkpoint).expanduser().resolve())
    if args.cam_k is not None and len(args.cam_k) != 9:
        raise ValueError("--cam-k must contain 9 values.")

    import rclpy

    rclpy.init()
    tracker = None
    try:
        tracker = LiveObjectPoseTrackerNode(args)
        rclpy.spin(tracker.node)
    finally:  # pragma: no cover - ROS runtime path
        if tracker is not None:
            tracker.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
