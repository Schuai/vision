"""ROS2 publisher for a USB RealSense camera when realsense-ros is unavailable."""

from __future__ import annotations

import argparse

import numpy as np

from src.ros2_utils import make_camera_info_msg, numpy_to_image_msg

DEFAULT_PHYSICAL_PORT_HINT = "2-9"
DEFAULT_SHARED_COLOR_DEPTH_MODES = (
    (640, 480, (6, 15, 30)),
    (1280, 720, (6,)),
)


def _format_shared_mode_help() -> str:
    """Return a compact help string for the default camera's shared RGB+depth modes."""

    mode_chunks = []
    for width, height, fps_values in DEFAULT_SHARED_COLOR_DEPTH_MODES:
        fps_text = "/".join(str(fps) for fps in fps_values)
        mode_chunks.append(f"{width}x{height}@{fps_text}")
    return (
        "Verified shared RGB+depth modes on the default 1-2 Intel RealSense D415: "
        + ", ".join(mode_chunks)
        + "."
    )


def _device_info_or_unknown(device: object, camera_info: object) -> str:
    if device.supports(camera_info):
        return str(device.get_info(camera_info))
    return "unknown"


def enumerate_realsense_devices(rs: object) -> list[dict[str, str]]:
    """Return connected RealSense devices with user-facing metadata."""

    ctx = rs.context()
    devices = []
    for index, device in enumerate(ctx.query_devices()):
        devices.append(
            {
                "index": str(index),
                "name": _device_info_or_unknown(device, rs.camera_info.name),
                "serial_number": _device_info_or_unknown(device, rs.camera_info.serial_number),
                "physical_port": _device_info_or_unknown(device, rs.camera_info.physical_port),
                "usb_type": _device_info_or_unknown(device, rs.camera_info.usb_type_descriptor),
            }
        )
    return devices


def select_realsense_device(
    rs: object,
    serial_number: str | None,
    device_index: int | None,
    physical_port_hint: str | None,
) -> dict[str, str] | None:
    """Resolve the RealSense device to use from explicit selectors or the default USB path."""

    devices = enumerate_realsense_devices(rs)
    if not devices:
        return None

    if serial_number:
        for device in devices:
            if device["serial_number"] == serial_number:
                return device
        raise ValueError(f"No RealSense device found with serial number {serial_number}.")

    if device_index is not None:
        if device_index < 0 or device_index >= len(devices):
            raise ValueError(
                f"--device-index {device_index} is out of range for {len(devices)} connected RealSense devices."
            )
        return devices[device_index]

    if physical_port_hint:
        for device in devices:
            if physical_port_hint in device["physical_port"]:
                return device

    return devices[0]


def resolve_device_serial(
    rs: object,
    serial_number: str | None,
    device_index: int | None,
    physical_port_hint: str | None,
) -> tuple[str | None, dict[str, str] | None]:
    """Resolve the device serial plus the selected device metadata."""

    device = select_realsense_device(
        rs=rs,
        serial_number=serial_number,
        device_index=device_index,
        physical_port_hint=physical_port_hint,
    )
    if device is None:
        return None, None

    resolved_serial = device["serial_number"]
    if not resolved_serial or resolved_serial == "unknown":
        raise ValueError(
            "The selected RealSense device did not expose a serial number through librealsense, "
            "so it cannot be targeted reliably."
        )
    return resolved_serial, device


def resolve_stream_intrinsics(
    rs: object,
    serial_number: str,
    stream_type: object,
    stream_format: object,
    width: int,
    height: int,
    fps: int,
) -> object:
    """Return stream intrinsics for a specific device/profile without starting a live stream."""

    ctx = rs.context()
    for device in ctx.query_devices():
        if not device.supports(rs.camera_info.serial_number):
            continue
        if device.get_info(rs.camera_info.serial_number) != serial_number:
            continue
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                try:
                    video_profile = profile.as_video_stream_profile()
                except RuntimeError:
                    continue
                if (
                    profile.stream_type() == stream_type
                    and profile.format() == stream_format
                    and profile.fps() == fps
                    and video_profile.width() == width
                    and video_profile.height() == height
                ):
                    return video_profile.get_intrinsics()
        break
    raise ValueError(
        "Could not find a matching RealSense stream profile for "
        f"serial={serial_number}, size={width}x{height}, fps={fps}."
    )


def build_ros_qos_profile(qos_module: object, reliability: str, depth: int) -> object:
    """Build a ROS2 QoS profile for image-like topics."""

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


class RealSensePublisherNode:
    """Publish RGB, aligned depth, and camera info topics from a USB RealSense camera."""

    def __init__(self, args: argparse.Namespace) -> None:
        import pyrealsense2 as rs
        import rclpy
        from rclpy.node import Node
        from rclpy import qos
        from sensor_msgs.msg import CameraInfo, Image

        class _Node(Node):
            pass

        self.args = args
        self.rs = rs
        self.rclpy = rclpy
        self.node = _Node("realsense_ros2_publisher")
        self.pipeline = None
        self.align = None
        self.color_pub = None
        self.depth_pub = None
        self.consecutive_timeouts = 0
        self.selected_serial, self.selected_device = resolve_device_serial(
            rs=rs,
            serial_number=args.serial_number,
            device_index=args.device_index,
            physical_port_hint=args.physical_port_hint,
        )
        self.color_intrinsics = resolve_stream_intrinsics(
            rs=rs,
            serial_number=self.selected_serial,
            stream_type=rs.stream.color,
            stream_format=rs.format.rgb8,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        publisher_qos = build_ros_qos_profile(
            qos_module=qos,
            reliability=args.qos_reliability,
            depth=args.qos_depth,
        )

        if not args.camera_info_only:
            self.color_pub = self.node.create_publisher(Image, args.color_topic, publisher_qos)
            if not args.rgb_only:
                self.depth_pub = self.node.create_publisher(Image, args.depth_topic, publisher_qos)
        self.camera_info_pub = self.node.create_publisher(CameraInfo, args.camera_info_topic, publisher_qos)
        self.node.get_logger().info(
            f"Publishing RealSense topics on {args.color_topic}, {args.depth_topic}, and {args.camera_info_topic}"
        )
        self.node.get_logger().info(
            f"Publisher QoS: reliability={args.qos_reliability}, depth={args.qos_depth}"
        )
        if self.selected_serial:
            self.node.get_logger().info(
                f"Selected RealSense serial: {self.selected_serial} "
                f"(port={self.selected_device['physical_port']})"
            )
        usb_type = self.selected_device["usb_type"]
        if usb_type.startswith("2"):
            self.node.get_logger().warning(
                "The selected RealSense is enumerating as USB "
                f"{usb_type}. Continuous RGB+depth streaming is often unreliable on USB 2.x. "
                "If frames time out, reconnect the camera with a USB 3.x cable/port or reduce expectations to RGB-only tooling."
            )
        if args.camera_info_only:
            self.node.get_logger().info(
                "Running in camera-info-only mode. No RealSense frames will be acquired."
            )
            self.node.create_timer(1.0 / max(args.fps, 1), self._publish_camera_info_only)
            return
        if args.rgb_only:
            self.node.get_logger().info("Running in RGB-only mode. Depth stream publishing is disabled.")

        self.pipeline = rs.pipeline()
        if not args.rgb_only:
            self.align = rs.align(rs.stream.color)
        config = rs.config()
        if self.selected_serial:
            config.enable_device(self.selected_serial)
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
        if not args.rgb_only:
            config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        self.profile = self.pipeline.start(config)
        self.node.create_timer(1.0 / max(args.fps, 1), self._publish_frame)

    def _publish_camera_info(self, stamp: object) -> None:
        camera_info_msg = make_camera_info_msg(
            width=self.color_intrinsics.width,
            height=self.color_intrinsics.height,
            frame_id=self.args.frame_id,
            stamp=stamp,
            fx=self.color_intrinsics.fx,
            fy=self.color_intrinsics.fy,
            cx=self.color_intrinsics.ppx,
            cy=self.color_intrinsics.ppy,
            distortion=self.color_intrinsics.coeffs,
        )
        self.camera_info_pub.publish(camera_info_msg)

    def _publish_camera_info_only(self) -> None:
        stamp = self.node.get_clock().now().to_msg()
        self._publish_camera_info(stamp)

    def _publish_frame(self) -> None:
        try:
            frames = self.pipeline.wait_for_frames(self.args.frame_timeout_ms)
        except RuntimeError as exc:
            self.consecutive_timeouts += 1
            if self.consecutive_timeouts <= 3 or self.consecutive_timeouts % 10 == 0:
                self.node.get_logger().warning(
                    "Timed out waiting for RealSense frames "
                    f"({self.consecutive_timeouts} consecutive timeouts, timeout={self.args.frame_timeout_ms} ms): {exc}. "
                    "This usually means the camera is busy, disconnected, or negotiated down to an unstable USB mode."
                )
            return

        stamp = self.node.get_clock().now().to_msg()
        if self.args.rgb_only:
            color_frame = frames.get_color_frame()
            if not color_frame:
                return
            self.consecutive_timeouts = 0
            color = np.asarray(color_frame.get_data())
            color_msg = numpy_to_image_msg(
                array=color,
                encoding="rgb8",
                frame_id=self.args.frame_id,
                stamp=stamp,
            )
            self.color_pub.publish(color_msg)
            self._publish_camera_info(stamp)
            return

        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return
        self.consecutive_timeouts = 0

        color = np.asarray(color_frame.get_data())
        depth = np.asarray(depth_frame.get_data(), dtype=np.uint16)

        color_msg = numpy_to_image_msg(
            array=color,
            encoding="rgb8",
            frame_id=self.args.frame_id,
            stamp=stamp,
        )
        depth_msg = numpy_to_image_msg(
            array=depth,
            encoding="16UC1",
            frame_id=self.args.frame_id,
            stamp=stamp,
        )
        self.color_pub.publish(color_msg)
        self.depth_pub.publish(depth_msg)
        self._publish_camera_info(stamp)

    def shutdown(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
        self.node.destroy_node()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish ROS2 image topics from a USB RealSense camera.")
    shared_mode_help = _format_shared_mode_help()
    parser.add_argument("--color-topic", type=str, default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", type=str, default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic", type=str, default="/camera/color/camera_info")
    parser.add_argument("--frame-id", type=str, default="camera_color_optical_frame")
    parser.add_argument("--serial-number", type=str, default=None)
    parser.add_argument("--device-index", type=int, default=None, help="Select the Nth connected RealSense device.")
    parser.add_argument(
        "--physical-port-hint",
        type=str,
        default=DEFAULT_PHYSICAL_PORT_HINT,
        help="Prefer the RealSense connected on a physical port containing this substring.",
    )
    parser.add_argument("--list-devices", action="store_true", help="Print connected RealSense devices and exit.")
    parser.add_argument("--width", type=int, default=640, help=f"Stream width for both color and depth. {shared_mode_help}")
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help=f"Stream height for both color and depth. {shared_mode_help}",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help=f"Stream rate for both color and depth. {shared_mode_help}",
    )
    parser.add_argument(
        "--frame-timeout-ms",
        type=int,
        default=5000,
        help="Timeout for waiting on a new RealSense frameset before logging a warning and retrying.",
    )
    parser.add_argument(
        "--camera-info-only",
        action="store_true",
        help="Publish only CameraInfo using cached intrinsics and do not acquire live RealSense frames.",
    )
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help="Publish only RGB plus CameraInfo and do not enable the depth stream.",
    )
    parser.add_argument(
        "--qos-reliability",
        choices=("reliable", "best_effort"),
        default="reliable",
        help="ROS2 publisher reliability policy. Use 'reliable' for RViz compatibility, 'best_effort' for sensor-data style transport.",
    )
    parser.add_argument(
        "--qos-depth",
        type=int,
        default=10,
        help="ROS2 publisher queue depth for image and CameraInfo topics.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.camera_info_only and args.rgb_only:
        raise SystemExit("Choose either --camera-info-only or --rgb-only, not both.")

    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "pyrealsense2 is not installed in the current Python environment. "
            "Install librealsense Python bindings before running this publisher."
        ) from exc

    if args.list_devices:
        devices = enumerate_realsense_devices(rs)
        if not devices:
            print("No RealSense devices found.")
            return
        for device in devices:
            print(
                f"[{device['index']}] {device['name']} "
                f"serial={device['serial_number']} usb={device['usb_type']} port={device['physical_port']}"
            )
        return

    import rclpy

    rclpy.init()
    publisher = None
    try:
        publisher = RealSensePublisherNode(args)
        rclpy.spin(publisher.node)
    except KeyboardInterrupt:
        pass
    finally:  # pragma: no cover - ROS runtime path
        if publisher is not None:
            publisher.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
