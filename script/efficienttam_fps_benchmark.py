#!/usr/bin/env python3
"""Benchmark EfficientTAM camera-tracking FPS on one or more live RealSense cameras."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from efficient_track_anything.wrapper import EfficientTrackAnythingWrapper

from script.realsense_publisher import (
    DEFAULT_SERIAL_NUMBER,
    enumerate_realsense_devices,
    resolve_device_serial,
)
from script.ros2_utils import binary_mask_to_bbox_xywh, bbox_xywh_to_xyxy, logits_to_binary_mask


TAB10_BGR = (
    (180, 119, 31),
    (14, 127, 255),
    (44, 160, 44),
    (40, 39, 214),
    (189, 103, 148),
    (75, 86, 140),
    (194, 119, 227),
    (127, 127, 127),
    (34, 189, 188),
    (207, 190, 23),
)


@dataclass
class BenchmarkStats:
    count: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    @property
    def fps(self) -> float:
        return 1000.0 / self.mean_ms if self.mean_ms > 1e-6 else float("inf")


class RealSenseColorStream:
    """Minimal color-only RealSense stream for tracker benchmarking."""

    def __init__(
        self,
        rs: object,
        serial_number: str,
        width: int,
        height: int,
        fps: int,
        frame_timeout_ms: int,
    ) -> None:
        self.rs = rs
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_timeout_ms = frame_timeout_ms
        self.pipeline: object | None = None

    def start(self) -> None:
        config = self.rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(self.rs.stream.color, self.width, self.height, self.rs.format.rgb8, self.fps)
        self.pipeline = self.rs.pipeline()
        self.pipeline.start(config)

    def read_rgb(self) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("RealSense stream has not been started.")
        frames = self.pipeline.wait_for_frames(self.frame_timeout_ms)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"Failed to receive a color frame from serial {self.serial_number}.")
        return np.asarray(color_frame.get_data()).copy()

    def stop(self) -> None:
        if self.pipeline is None:
            return
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass
        self.pipeline = None


class EfficientTAMBenchmark:
    """Stateful benchmark runner modeled after the multi-camera TAM tracker flow."""

    def __init__(self, args: argparse.Namespace, n_cams: int) -> None:
        self.args = args
        self.n_cams = n_cams
        self.predictors: list[Any] = []
        self.object_id = args.object_id
        self.parallel_warning_emitted = False

        for _ in range(n_cams):
            wrapper = EfficientTrackAnythingWrapper(device=args.device)
            predictor = wrapper.load_camera_predictor(
                config_file=args.tracker_config,
                checkpoint_path=args.tracker_checkpoint,
                vos_optimized=args.vos_optimized,
            )
            self.predictors.append(predictor)

    def load_first_frames(self, frames: list[np.ndarray]) -> float:
        if len(frames) != self.n_cams:
            raise ValueError("Number of frames must match the number of predictors.")
        maybe_cuda_synchronize(self.args.device)
        start_time = time.perf_counter()
        with inference_context(self.args.device, enable_autocast=self.args.autocast):
            for predictor, frame in zip(self.predictors, frames, strict=True):
                predictor.load_first_frame(frame)
        maybe_cuda_synchronize(self.args.device)
        return (time.perf_counter() - start_time) * 1000.0

    def add_new_prompts(self, bboxes_xywh: list[list[float]]) -> tuple[list[np.ndarray], float]:
        if len(bboxes_xywh) != self.n_cams:
            raise ValueError("Number of bounding boxes must match the number of predictors.")

        preview_masks: list[np.ndarray] = []
        maybe_cuda_synchronize(self.args.device)
        start_time = time.perf_counter()
        with inference_context(self.args.device, enable_autocast=self.args.autocast):
            for predictor, bbox_xywh in zip(self.predictors, bboxes_xywh, strict=True):
                _, _, mask_logits = predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=self.object_id,
                    bbox=bbox_xywh_to_xyxy(bbox_xywh),
                )
                preview_masks.append(logits_to_binary_mask(mask_logits, threshold=self.args.mask_threshold))
        maybe_cuda_synchronize(self.args.device)
        return preview_masks, (time.perf_counter() - start_time) * 1000.0

    def track(self, frames: list[np.ndarray]) -> tuple[list[tuple[Any, Any]], float]:
        if len(frames) != self.n_cams:
            raise ValueError("Number of frames must match the number of predictors.")

        maybe_cuda_synchronize(self.args.device)
        start_time = time.perf_counter()
        with inference_context(self.args.device, enable_autocast=self.args.autocast):
            if self.args.parallel_cameras and self.n_cams > 1:
                try:
                    futures = [
                        torch.jit.fork(predictor.track, frame)
                        for predictor, frame in zip(self.predictors, frames, strict=True)
                    ]
                    results = [torch.jit.wait(future) for future in futures]
                except Exception as exc:  # pragma: no cover - runtime fallback
                    if not self.parallel_warning_emitted:
                        logging.warning(
                            "Parallel camera tracking fell back to sequential execution: %s",
                            exc,
                        )
                        self.parallel_warning_emitted = True
                    results = [predictor.track(frame) for predictor, frame in zip(self.predictors, frames, strict=True)]
            else:
                results = [predictor.track(frame) for predictor, frame in zip(self.predictors, frames, strict=True)]
        maybe_cuda_synchronize(self.args.device)
        return results, (time.perf_counter() - start_time) * 1000.0


def parse_bbox(value: str) -> list[float]:
    bbox = json.loads(value)
    if len(bbox) != 4:
        raise argparse.ArgumentTypeError("A bbox must contain 4 values: [x, y, w, h].")
    return [float(v) for v in bbox]


def parse_bbox_list(value: str) -> list[list[float]]:
    bboxes = json.loads(value)
    if not isinstance(bboxes, list):
        raise argparse.ArgumentTypeError("Expected a JSON list of bounding boxes.")
    parsed = []
    for bbox in bboxes:
        if len(bbox) != 4:
            raise argparse.ArgumentTypeError("Each bbox must contain 4 values: [x, y, w, h].")
        parsed.append([float(v) for v in bbox])
    return parsed


def normalize_hydra_config_path(config_file: str) -> str:
    config_path = Path(config_file).expanduser()
    path_parts = config_path.parts
    if "configs" in path_parts:
        configs_index = path_parts.index("configs")
        return Path(*path_parts[configs_index:]).as_posix()
    return config_file


def resolve_tracker_checkpoint(checkpoint_path: str | None) -> str:
    if checkpoint_path:
        return str(Path(checkpoint_path).expanduser().resolve())
    default_checkpoint = Path("checkpoints/etam/efficienttam_s_512x512.pt").resolve()
    if default_checkpoint.is_file():
        return str(default_checkpoint)
    raise FileNotFoundError(
        "No EfficientTAM checkpoint was provided and the default "
        "`checkpoints/etam/efficienttam_s_512x512.pt` was not found."
    )


def maybe_cuda_synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


@contextlib.contextmanager
def inference_context(device: str, enable_autocast: bool) -> Any:
    with torch.inference_mode():
        if enable_autocast and device.startswith("cuda") and torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                yield
        else:
            yield


def summarize_ms(values_ms: list[float]) -> BenchmarkStats:
    values = np.asarray(values_ms, dtype=np.float64)
    return BenchmarkStats(
        count=int(values.size),
        mean_ms=float(values.mean()),
        median_ms=float(np.median(values)),
        p90_ms=float(np.percentile(values, 90)),
        p95_ms=float(np.percentile(values, 95)),
        min_ms=float(values.min()),
        max_ms=float(values.max()),
    )


def format_stats(label: str, stats: BenchmarkStats, camera_count: int) -> str:
    aggregate_camera_fps = stats.fps * camera_count
    return (
        f"{label}: mean={stats.mean_ms:.2f} ms median={stats.median_ms:.2f} ms "
        f"p90={stats.p90_ms:.2f} ms p95={stats.p95_ms:.2f} ms "
        f"min={stats.min_ms:.2f} ms max={stats.max_ms:.2f} ms "
        f"step_fps={stats.fps:.2f} aggregate_camera_fps={aggregate_camera_fps:.2f}"
    )


def make_mask_overlay(frame_rgb: np.ndarray, mask: np.ndarray, color_index: int = 0) -> np.ndarray:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.any():
        color = np.asarray(TAB10_BGR[color_index % len(TAB10_BGR)], dtype=np.float32)
        blended_pixels = frame_bgr[mask_bool].astype(np.float32)
        blended_pixels *= 0.4
        blended_pixels += color * 0.6
        frame_bgr[mask_bool] = blended_pixels.astype(np.uint8)
    return frame_bgr


def build_visualization_frame(
    frames_rgb: list[np.ndarray],
    results: list[tuple[Any, Any]],
    serial_numbers: list[str],
    latest_track_ms: float,
    latest_loop_ms: float,
) -> np.ndarray:
    panels: list[np.ndarray] = []
    for serial_number, frame_rgb, (_, mask_logits) in zip(serial_numbers, frames_rgb, results, strict=True):
        try:
            mask = logits_to_binary_mask(mask_logits, object_index=0)
        except ValueError:
            mask = np.zeros(frame_rgb.shape[:2], dtype=bool)
        panel = make_mask_overlay(frame_rgb, mask, color_index=0)
        bbox = binary_mask_to_bbox_xywh(mask)
        if bbox[2] > 0 and bbox[3] > 0:
            x, y, width, height = bbox
            cv2.rectangle(panel, (x, y), (x + width, y + height), (0, 255, 0), 2)
        lines = [
            f"serial={serial_number}",
            f"track={latest_track_ms:.1f} ms",
            f"loop={latest_loop_ms:.1f} ms",
        ]
        y = 28
        for line in lines:
            cv2.putText(
                panel,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 28
        panels.append(panel)

    max_cols = min(2, len(panels))
    rows = math.ceil(len(panels) / max_cols)
    panel_height, panel_width = panels[0].shape[:2]
    blank_panel = np.zeros_like(panels[0])
    row_images = []
    for row_index in range(rows):
        start_index = row_index * max_cols
        row_panels = panels[start_index : start_index + max_cols]
        while len(row_panels) < max_cols:
            row_panels.append(blank_panel)
        row_images.append(np.hstack(row_panels))
    grid = np.vstack(row_images)
    expected_height = rows * panel_height
    expected_width = max_cols * panel_width
    return grid[:expected_height, :expected_width]


def capture_color_frames(streams: list[RealSenseColorStream], grabs_per_step: int) -> list[np.ndarray]:
    frames: list[np.ndarray] | None = None
    for _ in range(max(1, grabs_per_step)):
        frames = [stream.read_rgb() for stream in streams]
    assert frames is not None
    return frames


def select_bboxes_interactively(
    frames_rgb: list[np.ndarray],
    serial_numbers: list[str],
) -> list[list[float]]:
    bboxes: list[list[float]] = []
    for serial_number, frame_rgb in zip(serial_numbers, frames_rgb, strict=True):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        window_name = f"Select ROI {serial_number}"
        x, y, width, height = cv2.selectROI(window_name, frame_bgr, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)
        if width <= 0 or height <= 0:
            raise RuntimeError(f"No ROI selected for camera {serial_number}.")
        bboxes.append([float(x), float(y), float(width), float(height)])
    return bboxes


def print_devices(rs: object) -> None:
    devices = enumerate_realsense_devices(rs)
    if not devices:
        print("No RealSense devices found.")
        return
    for device in devices:
        print(
            f"[{device['index']}] {device['name']} "
            f"serial={device['serial_number']} usb={device['usb_type']} port={device['physical_port']}"
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark EfficientTAM camera-tracking FPS on live RealSense frames.")
    parser.add_argument(
        "--serial-number",
        action="append",
        default=None,
        help=(
            "Select a RealSense serial number. Repeat this flag to benchmark multiple cameras in parallel. "
            f"Defaults to the repo default serial {DEFAULT_SERIAL_NUMBER}."
        ),
    )
    parser.add_argument("--list-devices", action="store_true", help="List connected RealSense devices and exit.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=5000)
    parser.add_argument("--capture-warmup-grabs", type=int, default=5)
    parser.add_argument("--tracker-config", type=str, default="configs/efficienttam/efficienttam_s_512x512.yaml")
    parser.add_argument("--tracker-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument(
        "--bbox",
        type=parse_bbox,
        default=None,
        help='Single-camera initialization bbox in JSON form, for example \'[240, 140, 120, 120]\'.',
    )
    parser.add_argument(
        "--bboxes",
        type=parse_bbox_list,
        default=None,
        help='Multi-camera initialization bboxes in JSON form, for example \'[[x,y,w,h],[...]]\'.',
    )
    parser.add_argument("--warmup-steps", type=int, default=20, help="Number of untimed tracking steps before benchmarking.")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of timed tracking steps.")
    parser.add_argument("--report-every", type=int, default=20, help="Print a rolling timing update every N timed steps.")
    parser.add_argument("--vis", action="store_true", help="Show live mask overlays during benchmarking.")
    parser.add_argument(
        "--parallel-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torch.jit.fork to track multiple cameras concurrently, similar to the reference tracker.",
    )
    parser.add_argument(
        "--vos-optimized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the EfficientTAM VOS-optimized camera predictor.",
    )
    parser.add_argument(
        "--autocast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run EfficientTAM under CUDA autocast with bfloat16 when available.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("pyrealsense2 is not installed in the current Python environment.") from exc

    if args.list_devices:
        print_devices(rs)
        return

    args.tracker_config = normalize_hydra_config_path(args.tracker_config)
    args.tracker_checkpoint = resolve_tracker_checkpoint(args.tracker_checkpoint)

    requested_serials = args.serial_number or [DEFAULT_SERIAL_NUMBER]
    selected_serials: list[str] = []
    for serial in requested_serials:
        resolved_serial, _ = resolve_device_serial(
            rs=rs,
            serial_number=serial,
            device_index=None,
            physical_port_hint=None,
        )
        if resolved_serial is None:
            raise RuntimeError(f"Could not resolve RealSense serial {serial}.")
        selected_serials.append(resolved_serial)

    logging.info("Benchmarking EfficientTAM on %d camera(s): %s", len(selected_serials), selected_serials)
    logging.info("Using config=%s checkpoint=%s", args.tracker_config, args.tracker_checkpoint)

    streams = [
        RealSenseColorStream(
            rs=rs,
            serial_number=serial_number,
            width=args.width,
            height=args.height,
            fps=args.fps,
            frame_timeout_ms=args.frame_timeout_ms,
        )
        for serial_number in selected_serials
    ]

    benchmark = EfficientTAMBenchmark(args=args, n_cams=len(selected_serials))
    window_name = "EfficientTAM FPS Benchmark"

    try:
        for stream in streams:
            stream.start()

        initial_frames = capture_color_frames(streams, grabs_per_step=args.capture_warmup_grabs)
        load_first_frame_ms = benchmark.load_first_frames(initial_frames)
        logging.info("load_first_frame completed in %.2f ms across %d camera(s).", load_first_frame_ms, len(streams))

        if args.bboxes is not None:
            if len(args.bboxes) != len(streams):
                raise ValueError("--bboxes must contain exactly one bbox per camera.")
            bboxes = args.bboxes
        elif args.bbox is not None:
            if len(streams) != 1:
                raise ValueError("--bbox only supports the single-camera case. Use --bboxes instead.")
            bboxes = [args.bbox]
        else:
            logging.info("Select one ROI per camera to initialize EfficientTAM tracking.")
            bboxes = select_bboxes_interactively(initial_frames, selected_serials)

        preview_masks, add_prompt_ms = benchmark.add_new_prompts(bboxes)
        logging.info("add_new_prompt completed in %.2f ms across %d camera(s).", add_prompt_ms, len(streams))

        if args.vis:
            preview_results = [([args.object_id], mask[None, None, ...]) for mask in preview_masks]
            preview_frame = build_visualization_frame(
                frames_rgb=initial_frames,
                results=preview_results,
                serial_numbers=selected_serials,
                latest_track_ms=0.0,
                latest_loop_ms=0.0,
            )
            cv2.imshow(window_name, preview_frame)
            cv2.waitKey(1)

        logging.info("Running %d warmup steps.", args.warmup_steps)
        for _ in range(args.warmup_steps):
            frames = capture_color_frames(streams, grabs_per_step=1)
            benchmark.track(frames)

        logging.info("Benchmarking %d timed steps.", args.num_steps)
        track_times_ms: list[float] = []
        loop_times_ms: list[float] = []
        interrupted = False

        for step_idx in range(1, args.num_steps + 1):
            loop_start_time = time.perf_counter()
            frames = capture_color_frames(streams, grabs_per_step=1)
            results, track_ms = benchmark.track(frames)
            loop_ms = (time.perf_counter() - loop_start_time) * 1000.0
            track_times_ms.append(track_ms)
            loop_times_ms.append(loop_ms)

            if args.vis:
                visualization = build_visualization_frame(
                    frames_rgb=frames,
                    results=results,
                    serial_numbers=selected_serials,
                    latest_track_ms=track_ms,
                    latest_loop_ms=loop_ms,
                )
                cv2.imshow(window_name, visualization)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    interrupted = True
                    logging.info("Benchmark interrupted by user input.")
                    break

            if args.report_every > 0 and step_idx % args.report_every == 0:
                recent_track = summarize_ms(track_times_ms[-args.report_every :])
                logging.info(
                    "Step %d/%d recent track stats: mean=%.2f ms median=%.2f ms p90=%.2f ms step_fps=%.2f",
                    step_idx,
                    args.num_steps,
                    recent_track.mean_ms,
                    recent_track.median_ms,
                    recent_track.p90_ms,
                    recent_track.fps,
                )

        if not track_times_ms:
            raise RuntimeError("No benchmark steps completed.")

        track_stats = summarize_ms(track_times_ms)
        loop_stats = summarize_ms(loop_times_ms)
        logging.info("EfficientTAM benchmark complete%s.", " (interrupted)" if interrupted else "")
        logging.info(format_stats("track()", track_stats, camera_count=len(streams)))
        logging.info(format_stats("capture+track+vis loop", loop_stats, camera_count=len(streams)))
    finally:
        for stream in streams:
            stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
