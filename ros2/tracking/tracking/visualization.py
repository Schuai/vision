#!/usr/bin/env python3
"""Publish an RViz-friendly image with the tracked bbox drawn on top."""

from __future__ import annotations

import argparse
import threading

import numpy as np

from tracking.ros2_utils import (
    image_msg_to_rgb8,
    numpy_to_image_msg,
    polygon_msg_to_bbox_xywh,
    stamp_to_nanoseconds,
)


def draw_bbox(frame_rgb: np.ndarray, bbox_xywh: list[int] | None, color: tuple[int, int, int], thickness: int) -> np.ndarray:
    """Return an RGB frame with the bbox overlay applied."""

    annotated = frame_rgb.copy()
    if bbox_xywh is None:
        return annotated

    x, y, width, height = bbox_xywh
    if width <= 0 or height <= 0:
        return annotated

    try:
        import cv2
    except ModuleNotFoundError:
        x2 = min(annotated.shape[1], max(0, x + width))
        y2 = min(annotated.shape[0], max(0, y + height))
        x1 = min(annotated.shape[1], max(0, x))
        y1 = min(annotated.shape[0], max(0, y))
        annotated[y1:y2, x1 : min(x1 + thickness, x2)] = color
        annotated[y1:y2, max(x2 - thickness, x1) : x2] = color
        annotated[y1 : min(y1 + thickness, y2), x1:x2] = color
        annotated[max(y2 - thickness, y1) : y2, x1:x2] = color
        return annotated

    x1 = int(x)
    y1 = int(y)
    x2 = int(x + width)
    y2 = int(y + height)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        annotated,
        f"bbox {x},{y},{width},{height}",
        (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        max(1, thickness // 2),
        cv2.LINE_AA,
    )
    return annotated


class BboxVisualizationNode:
    """Subscribe to the camera image and bbox result, then publish an annotated image for RViz."""

    def __init__(self, args: argparse.Namespace) -> None:
        import rclpy
        from geometry_msgs.msg import PolygonStamped
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        class _Node(Node):
            pass

        self.args = args
        self.rclpy = rclpy
        self.node = _Node("tracking_visualization")
        self.lock = threading.Lock()
        self.latest_image_msg: Image | None = None
        self.latest_bbox_msg: PolygonStamped | None = None
        self.last_published_stamp_ns: int | None = None
        self.color = tuple(int(value) for value in args.bbox_color)

        self.image_pub = self.node.create_publisher(Image, args.output_image_topic, 10)
        self.node.create_subscription(Image, args.image_topic, self._on_image, qos_profile_sensor_data)
        self.node.create_subscription(PolygonStamped, args.bbox_topic, self._on_bbox, 10)

        self.node.get_logger().info(
            f"Subscribing to image={args.image_topic} and bbox={args.bbox_topic}; "
            f"publishing RViz image={args.output_image_topic}."
        )

    def _on_image(self, msg: object) -> None:
        with self.lock:
            self.latest_image_msg = msg
            bbox_msg = self.latest_bbox_msg
        self._publish_visualization(msg, bbox_msg)

    def _on_bbox(self, msg: object) -> None:
        with self.lock:
            self.latest_bbox_msg = msg
            image_msg = self.latest_image_msg
        if image_msg is not None:
            self._publish_visualization(image_msg, msg)

    def _publish_visualization(self, image_msg: object, bbox_msg: object | None) -> None:
        image_stamp_ns = stamp_to_nanoseconds(image_msg.header.stamp)
        bbox_xywh = polygon_msg_to_bbox_xywh(bbox_msg) if bbox_msg is not None else None
        annotated = draw_bbox(
            frame_rgb=image_msg_to_rgb8(image_msg),
            bbox_xywh=bbox_xywh,
            color=self.color,
            thickness=max(1, self.args.bbox_thickness),
        )
        output_msg = numpy_to_image_msg(
            array=annotated,
            encoding="rgb8",
            frame_id=image_msg.header.frame_id,
            stamp=image_msg.header.stamp,
        )
        self.image_pub.publish(output_msg)
        self.last_published_stamp_ns = image_stamp_ns

    def shutdown(self) -> None:
        self.node.destroy_node()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish an annotated image for RViz from the RealSense image and tracking bbox topics."
    )
    parser.add_argument("--image-topic", type=str, default="/camera/color/image_raw")
    parser.add_argument("--bbox-topic", type=str, default="/tracking/segmentation/bbox")
    parser.add_argument("--output-image-topic", type=str, default="/tracking/visualization/image")
    parser.add_argument("--bbox-color", type=int, nargs=3, default=(255, 64, 64), metavar=("R", "G", "B"))
    parser.add_argument("--bbox-thickness", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    import rclpy

    rclpy.init()
    node = BboxVisualizationNode(args)
    try:
        rclpy.spin(node.node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
