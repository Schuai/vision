"""ROS2 message helpers used by the tracking package."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def stamp_to_nanoseconds(stamp: object) -> int:
    """Convert a ROS2 builtin_interfaces/Time-like object into nanoseconds."""

    sec = getattr(stamp, "sec", 0)
    nanosec = getattr(stamp, "nanosec", 0)
    return int(sec) * 1_000_000_000 + int(nanosec)


def _reshape_image_buffer(msg: object, dtype: np.dtype, channels: int) -> np.ndarray:
    """Reshape a ROS Image buffer while respecting row stride."""

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
    """Convert a ROS2 Image message into an RGB uint8 array."""

    encoding = msg.encoding.lower()
    if encoding == "rgb8":
        return _reshape_image_buffer(msg, np.uint8, 3).copy()
    if encoding == "bgr8":
        return _reshape_image_buffer(msg, np.uint8, 3)[:, :, ::-1].copy()
    if encoding == "rgba8":
        return _reshape_image_buffer(msg, np.uint8, 4)[:, :, :3].copy()
    if encoding == "bgra8":
        return _reshape_image_buffer(msg, np.uint8, 4)[:, :, [2, 1, 0]].copy()
    if encoding == "mono8":
        mono = _reshape_image_buffer(msg, np.uint8, 1)
        return np.repeat(mono[:, :, None], 3, axis=2)
    raise ValueError(f"Unsupported color image encoding: {msg.encoding}")


def logits_to_binary_mask(mask_logits: object, object_index: int = 0, threshold: float = 0.0) -> np.ndarray:
    """Convert model mask logits into a boolean mask."""

    if hasattr(mask_logits, "detach"):
        mask_array = mask_logits.detach().float().cpu().numpy()
    else:
        mask_array = np.asarray(mask_logits)

    mask_array = np.squeeze(mask_array)
    if mask_array.ndim == 3:
        if object_index < 0 or object_index >= mask_array.shape[0]:
            raise ValueError(
                f"object_index {object_index} is out of range for mask logits shape {mask_array.shape}"
            )
        selected = mask_array[object_index]
    elif mask_array.ndim == 2:
        selected = mask_array
    elif mask_array.ndim > 3:
        collapsed = mask_array.reshape((-1, mask_array.shape[-2], mask_array.shape[-1]))
        if object_index < 0 or object_index >= collapsed.shape[0]:
            raise ValueError(
                f"object_index {object_index} is out of range for collapsed mask logits shape {mask_array.shape}"
            )
        selected = collapsed[object_index]
    else:
        raise ValueError(f"Unsupported mask logits shape: {mask_array.shape}")

    return np.asarray(selected > threshold, dtype=bool)


def binary_mask_to_bbox_xywh(mask: np.ndarray) -> list[int]:
    """Convert a binary mask into an xywh bounding box."""

    mask_bool = np.asarray(mask, dtype=bool)
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    if not np.any(rows) or not np.any(cols):
        return [-1, -1, 0, 0]

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]


def bbox_xywh_to_polygon_msg(
    bbox_xywh: list[int] | tuple[int, int, int, int],
    frame_id: str,
    stamp: object,
) -> object:
    """Create a PolygonStamped bbox in image pixel coordinates."""

    from geometry_msgs.msg import Point32, PolygonStamped

    x, y, width, height = [float(value) for value in bbox_xywh]
    msg = PolygonStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    if width <= 0 or height <= 0:
        return msg

    x2 = x + width
    y2 = y + height
    msg.polygon.points = [
        Point32(x=x, y=y, z=0.0),
        Point32(x=x2, y=y, z=0.0),
        Point32(x=x2, y=y2, z=0.0),
        Point32(x=x, y=y2, z=0.0),
    ]
    return msg


def polygon_msg_to_bbox_xywh(msg: object) -> list[int] | None:
    """Convert a PolygonStamped bbox message into xywh pixels."""

    points = list(msg.polygon.points)
    if not points:
        return None

    xs = [float(point.x) for point in points]
    ys = [float(point.y) for point in points]
    x_min = min(xs)
    y_min = min(ys)
    width = max(xs) - x_min
    height = max(ys) - y_min
    if width <= 0 or height <= 0:
        return None
    return [int(round(x_min)), int(round(y_min)), int(round(width)), int(round(height))]


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
