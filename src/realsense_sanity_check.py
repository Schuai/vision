"""Minimal librealsense streaming sanity check without ROS2 dependencies."""

from __future__ import annotations

import argparse
import time

from src.realsense_ros2_publisher import DEFAULT_PHYSICAL_PORT_HINT, enumerate_realsense_devices, resolve_device_serial


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start a RealSense stream and report whether frames arrive reliably without ROS2."
    )
    parser.add_argument("--serial-number", type=str, default=None)
    parser.add_argument("--device-index", type=int, default=None, help="Select the Nth connected RealSense device.")
    parser.add_argument(
        "--physical-port-hint",
        type=str,
        default=DEFAULT_PHYSICAL_PORT_HINT,
        help="Prefer the RealSense connected on a physical port containing this substring.",
    )
    parser.add_argument("--list-devices", action="store_true", help="Print connected RealSense devices and exit.")
    parser.add_argument(
        "--stream",
        choices=("paired", "color", "depth"),
        default="paired",
        help="Which stream setup to test: both color+depth, color only, or depth only.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--frames",
        type=int,
        default=20,
        help="Number of frames to wait for before declaring success.",
    )
    parser.add_argument(
        "--frame-timeout-ms",
        type=int,
        default=5000,
        help="Timeout for each frame wait operation.",
    )
    return parser


def _print_devices(rs: object) -> None:
    devices = enumerate_realsense_devices(rs)
    if not devices:
        print("No RealSense devices found.")
        return
    for device in devices:
        print(
            f"[{device['index']}] {device['name']} "
            f"serial={device['serial_number']} usb={device['usb_type']} port={device['physical_port']}"
        )


def _enable_requested_streams(config: object, rs: object, args: argparse.Namespace) -> None:
    if args.stream in ("paired", "color"):
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    if args.stream in ("paired", "depth"):
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)


def _frames_arrived(frames: object, args: argparse.Namespace) -> tuple[bool, str]:
    checks: list[tuple[bool, str]] = []
    if args.stream in ("paired", "color"):
        color = frames.get_color_frame()
        checks.append((bool(color), f"color={bool(color)}"))
    if args.stream in ("paired", "depth"):
        depth = frames.get_depth_frame()
        checks.append((bool(depth), f"depth={bool(depth)}"))
    ok = all(result for result, _ in checks)
    return ok, " ".join(part for _, part in checks)


def run_check(args: argparse.Namespace) -> int:
    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "pyrealsense2 is not installed in the current Python environment."
        ) from exc

    if args.list_devices:
        _print_devices(rs)
        return 0

    selected_serial, selected_device = resolve_device_serial(
        rs=rs,
        serial_number=args.serial_number,
        device_index=args.device_index,
        physical_port_hint=args.physical_port_hint,
    )
    if selected_serial is None or selected_device is None:
        print("No RealSense device found.")
        return 1

    print(
        f"Selected RealSense: serial={selected_serial} "
        f"usb={selected_device['usb_type']} port={selected_device['physical_port']}"
    )
    if selected_device["usb_type"].startswith("2"):
        print(
            "Warning: the selected camera is enumerating as USB 2.x. "
            "Streaming may be unreliable even in this ROS-free sanity check."
        )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(selected_serial)
    _enable_requested_streams(config, rs, args)

    success_count = 0
    timeout_count = 0
    start_time = time.monotonic()
    try:
        pipeline.start(config)
        print(
            f"Started {args.stream} stream at {args.width}x{args.height}@{args.fps}. "
            f"Waiting for {args.frames} frame(s)..."
        )
        for frame_idx in range(1, args.frames + 1):
            try:
                frames = pipeline.wait_for_frames(args.frame_timeout_ms)
            except RuntimeError as exc:
                timeout_count += 1
                print(
                    f"[{frame_idx}/{args.frames}] timeout after {args.frame_timeout_ms} ms: {exc}"
                )
                continue

            ok, detail = _frames_arrived(frames, args)
            if ok:
                success_count += 1
                elapsed = time.monotonic() - start_time
                print(f"[{frame_idx}/{args.frames}] ok after {elapsed:.2f}s {detail}")
            else:
                print(f"[{frame_idx}/{args.frames}] incomplete frameset {detail}")
    finally:
        try:
            pipeline.stop()
        except RuntimeError:
            pass

    print(
        f"Summary: successes={success_count} timeouts={timeout_count} "
        f"requested={args.frames} stream={args.stream} mode={args.width}x{args.height}@{args.fps}"
    )
    if success_count == args.frames and timeout_count == 0:
        print("Sanity check passed.")
        return 0

    print("Sanity check failed.")
    return 1


def main() -> None:
    args = build_argparser().parse_args()
    raise SystemExit(run_check(args))


if __name__ == "__main__":
    main()
