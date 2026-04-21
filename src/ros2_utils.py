"""Shared ROS2 image and pose helpers for the live vision pipeline."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def stamp_to_nanoseconds(stamp: object) -> int:
    """Convert a ROS2 builtin_interfaces/Time-like object into nanoseconds."""

    sec = getattr(stamp, "sec", 0)
    nanosec = getattr(stamp, "nanosec", 0)
    return int(sec) * 1_000_000_000 + int(nanosec)


def camera_info_to_matrix(camera_info_msg: object) -> np.ndarray:
    """Return the 3x3 intrinsic matrix from a CameraInfo message."""

    return np.asarray(camera_info_msg.k, dtype=np.float32).reshape(3, 3)


def binary_mask_to_bbox_xywh(mask: np.ndarray) -> list[int]:
    """Convert a binary mask into an `xywh` bbox."""

    mask_bool = np.asarray(mask, dtype=bool)
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    if not np.any(rows) or not np.any(cols):
        return [-1, -1, 0, 0]

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]


def bbox_xywh_to_xyxy(bbox_xywh: Iterable[float]) -> tuple[float, float, float, float]:
    """Convert an `xywh` bbox into `xyxy` form."""

    x, y, w, h = [float(v) for v in bbox_xywh]
    return (x, y, x + w, y + h)


def logits_to_binary_mask(mask_logits: object, object_index: int = 0, threshold: float = 0.0) -> np.ndarray:
    """Convert model mask logits into a boolean mask."""

    if hasattr(mask_logits, "detach"):
        mask_array = mask_logits.detach().float().cpu().numpy()
    else:
        mask_array = np.asarray(mask_logits)

    if mask_array.ndim == 3:
        selected = mask_array[object_index]
    elif mask_array.ndim == 2:
        selected = mask_array
    else:
        raise ValueError(f"Unsupported mask logits shape: {mask_array.shape}")

    return np.asarray(selected > threshold, dtype=bool)


def _reshape_image_buffer(msg: object, dtype: np.dtype, channels: int) -> np.ndarray:
    """Reshape the raw ROS image buffer while respecting `step`."""

    itemsize = np.dtype(dtype).itemsize
    row_items = int(msg.step) // itemsize
    array = np.frombuffer(msg.data, dtype=dtype)
    if channels == 1:
        array = array.reshape((msg.height, row_items))
        return array[:, : msg.width]

    array = array.reshape((msg.height, row_items))
    expected_items = int(msg.width) * channels
    array = array[:, :expected_items]
    return array.reshape((msg.height, msg.width, channels))


def image_msg_to_rgb8(msg: object) -> np.ndarray:
    """Convert a ROS2 Image message into an RGB uint8 numpy array."""

    encoding = msg.encoding.lower()
    if encoding == "rgb8":
        return _reshape_image_buffer(msg, np.uint8, 3).copy()
    if encoding == "bgr8":
        return cv2.cvtColor(_reshape_image_buffer(msg, np.uint8, 3), cv2.COLOR_BGR2RGB)
    if encoding == "rgba8":
        return cv2.cvtColor(_reshape_image_buffer(msg, np.uint8, 4), cv2.COLOR_RGBA2RGB)
    if encoding == "bgra8":
        return cv2.cvtColor(_reshape_image_buffer(msg, np.uint8, 4), cv2.COLOR_BGRA2RGB)
    if encoding == "mono8":
        mono = _reshape_image_buffer(msg, np.uint8, 1)
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB)
    raise ValueError(f"Unsupported color image encoding: {msg.encoding}")


def depth_msg_to_meters(msg: object, depth_scale_for_uint16: float = 0.001) -> np.ndarray:
    """Convert a ROS2 depth Image message into meters."""

    encoding = msg.encoding.lower()
    if encoding in {"16uc1", "mono16"}:
        depth = _reshape_image_buffer(msg, np.uint16, 1).astype(np.float32)
        depth *= float(depth_scale_for_uint16)
    elif encoding == "32fc1":
        depth = _reshape_image_buffer(msg, np.float32, 1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported depth image encoding: {msg.encoding}")

    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 1e-6] = 0.0
    return depth


def numpy_to_image_msg(
    array: np.ndarray,
    encoding: str,
    frame_id: str,
    stamp: object,
) -> object:
    """Create a ROS2 Image message from a numpy array."""

    from sensor_msgs.msg import Image

    np_array = np.ascontiguousarray(array)
    if np_array.ndim == 2:
        height, width = np_array.shape
        channels = 1
    elif np_array.ndim == 3:
        height, width, channels = np_array.shape
    else:
        raise ValueError(f"Unsupported image array shape: {np_array.shape}")

    msg = Image()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = int(height)
    msg.width = int(width)
    msg.encoding = encoding
    msg.is_bigendian = False
    msg.step = int(width * channels * np_array.dtype.itemsize)
    msg.data = np_array.tobytes()
    return msg


def make_pose_stamped(pose_matrix: np.ndarray, frame_id: str, stamp: object) -> object:
    """Convert a 4x4 pose matrix into a PoseStamped message."""

    from geometry_msgs.msg import PoseStamped

    if hasattr(pose_matrix, "detach"):
        pose_array = pose_matrix.detach().float().cpu().numpy()
    else:
        pose_array = np.asarray(pose_matrix)
    if pose_array.ndim == 3:
        pose_array = pose_array[0]

    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.pose.position.x = float(pose_array[0, 3])
    msg.pose.position.y = float(pose_array[1, 3])
    msg.pose.position.z = float(pose_array[2, 3])

    quat = Rotation.from_matrix(pose_array[:3, :3]).as_quat()
    msg.pose.orientation.x = float(quat[0])
    msg.pose.orientation.y = float(quat[1])
    msg.pose.orientation.z = float(quat[2])
    msg.pose.orientation.w = float(quat[3])
    return msg


def make_camera_info_msg(
    width: int,
    height: int,
    frame_id: str,
    stamp: object,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion: Iterable[float] | None = None,
) -> object:
    """Create a CameraInfo message from camera intrinsics."""

    from sensor_msgs.msg import CameraInfo

    distortion_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0] if distortion is None else list(distortion)

    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.width = int(width)
    msg.height = int(height)
    msg.distortion_model = "plumb_bob"
    msg.d = distortion_coeffs
    msg.k = [
        float(fx),
        0.0,
        float(cx),
        0.0,
        float(fy),
        float(cy),
        0.0,
        0.0,
        1.0,
    ]
    msg.r = [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    msg.p = [
        float(fx),
        0.0,
        float(cx),
        0.0,
        0.0,
        float(fy),
        float(cy),
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    return msg
