"""ROS2 pose tracker entrypoint backed by the foundationpose_cpp runtime."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


def _install_foundationpose_wrapper_stub() -> None:
    """Let this entrypoint reuse ros2_pose_tracker without the Python FoundationPose package."""

    if importlib.util.find_spec("foundationpose_wrapper") is not None:
        return

    stub = types.ModuleType("foundationpose_wrapper")

    class FoundationPoseWrapper:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError(
                "The Python FoundationPose wrapper is not installed. "
                "Use src.ros2_pose_tracker_cpp for the foundationpose_cpp backend."
            )

    stub.FoundationPoseWrapper = FoundationPoseWrapper
    sys.modules["foundationpose_wrapper"] = stub


DEFAULT_FOUNDATIONPOSE_CPP_ROOT = "/home/shuai/github_repos/foundationpose_cpp"
_FOUNDATIONPOSE_SESSION_CLASS: Any | None = None
_TRACKER_BASE: Any | None = None


def _foundationpose_cpp_root_from_argv(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--foundationpose-cpp-root", type=str, default=DEFAULT_FOUNDATIONPOSE_CPP_ROOT)
    args, _ = parser.parse_known_args(argv)
    return str(Path(args.foundationpose_cpp_root).expanduser().resolve())


def _preload_foundationpose_cpp(repo_root: str) -> Any:
    """Import foundationpose_cpp before torch so CUDA libraries resolve consistently."""

    global _FOUNDATIONPOSE_SESSION_CLASS
    if _FOUNDATIONPOSE_SESSION_CLASS is not None:
        return _FOUNDATIONPOSE_SESSION_CLASS

    repo_root_path = Path(repo_root).expanduser().resolve()
    build_python_path = repo_root_path / "build-pixi" / "python"
    package_dir = build_python_path / "foundationpose_cpp"
    extensions = sorted(package_dir.glob("_core*.so"))
    abi_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"

    if not extensions:
        raise RuntimeError(
            f"No foundationpose_cpp Python extension found in {package_dir}. "
            "Build it first with `pixi run build-python` in the foundationpose_cpp repo."
        )
    if not any(abi_tag in path.name for path in extensions):
        found = ", ".join(path.name for path in extensions)
        raise RuntimeError(
            f"foundationpose_cpp is not built for this Python ABI ({abi_tag}). "
            f"Found: {found}. Rebuild the binding with the Python environment that runs this tracker."
        )

    build_python_path_str = str(build_python_path)
    if build_python_path_str not in sys.path:
        sys.path.insert(0, build_python_path_str)

    try:
        from foundationpose_cpp import FoundationPoseSession
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Unable to import `foundationpose_cpp`. Make sure its runtime libraries are visible. "
            "Launch this tracker before importing Torch in the same process; this script preloads "
            "foundationpose_cpp before loading the base ROS2 tracker."
        ) from exc

    _FOUNDATIONPOSE_SESSION_CLASS = FoundationPoseSession
    return FoundationPoseSession


def _load_tracker_base() -> Any:
    global _TRACKER_BASE
    if _TRACKER_BASE is not None:
        return _TRACKER_BASE

    _install_foundationpose_wrapper_stub()
    from src import ros2_pose_tracker as tracker_base

    _TRACKER_BASE = tracker_base
    return tracker_base


class FoundationPoseCppWrapper:
    """Adapter exposing foundationpose_cpp with the interface expected by ros2_pose_tracker."""

    _args: argparse.Namespace | None = None

    @classmethod
    def configure(cls, args: argparse.Namespace) -> None:
        cls._args = args

    def __init__(self, device: str = "cuda", debug_dir: str = "./debug/foundationpose_ros2") -> None:
        if self._args is None:
            raise RuntimeError("FoundationPoseCppWrapper.configure(args) must be called before construction.")

        args = self._args
        self.repo_root = Path(args.foundationpose_cpp_root).expanduser().resolve()
        self.refiner_engine_path = str(Path(args.refiner_engine_path).expanduser().resolve())
        self.scorer_engine_path = str(Path(args.scorer_engine_path).expanduser().resolve())
        self.target_name = args.target_name
        self.max_image_height = int(args.foundationpose_max_image_height)
        self.max_image_width = int(args.foundationpose_max_image_width)
        self.mem_buf_size = int(args.foundationpose_mem_buf_size)
        self.device = device
        self.debug_dir = Path(debug_dir).expanduser()
        self.mesh: Any | None = None
        self.mesh_path: str | None = None
        self.session: Any | None = None
        self.camera_matrix: np.ndarray | None = None
        self.pose_last: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self.mesh is not None

    def load(
        self,
        mesh_path: str,
        apply_scale: float = 1.0,
        force_apply_color: list[int] | None = None,
    ) -> "FoundationPoseCppWrapper":
        import trimesh

        source_mesh_path = Path(mesh_path).expanduser().resolve()
        mesh = trimesh.load(source_mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh.apply_scale(apply_scale)
        if force_apply_color is not None:
            color = [int(value) for value in force_apply_color[:3]]
            mesh.visual.vertex_colors = np.tile(np.asarray([*color, 255], dtype=np.uint8), (len(mesh.vertices), 1))

        self.mesh = mesh
        self.mesh_path = str(source_mesh_path)
        self.target_name = self.target_name or source_mesh_path.stem

        if apply_scale != 1.0 or force_apply_color is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            prepared_mesh_path = self.debug_dir / f"{source_mesh_path.stem}_foundationpose_cpp.obj"
            mesh.export(prepared_mesh_path)
            self.mesh_path = str(prepared_mesh_path)

        return self

    def _torch_device(self) -> torch.device:
        import torch

        if isinstance(self.device, str) and self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.device(self.device)
        return torch.device("cpu")

    def _pose_tensor(self, pose: np.ndarray) -> torch.Tensor:
        import torch

        return torch.as_tensor(np.asarray(pose, dtype=np.float32).reshape(4, 4), device=self._torch_device())

    def _pose_numpy(self) -> np.ndarray | None:
        import torch

        if self.pose_last is None:
            return None
        if torch.is_tensor(self.pose_last):
            return self.pose_last.detach().float().cpu().numpy().reshape(4, 4).astype(np.float32)
        return np.asarray(self.pose_last, dtype=np.float32).reshape(4, 4)

    def _session_class(self) -> Any:
        return _preload_foundationpose_cpp(str(self.repo_root))

    def _ensure_session(self, camera_matrix: np.ndarray) -> None:
        if self.mesh_path is None or self.target_name is None:
            raise RuntimeError("FoundationPoseCppWrapper.load() must be called before register() or track_one().")

        camera_matrix = np.ascontiguousarray(np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3))
        if self.session is not None and self.camera_matrix is not None and np.allclose(self.camera_matrix, camera_matrix):
            return

        session_class = self._session_class()
        self.session = session_class(
            refiner_engine_path=self.refiner_engine_path,
            scorer_engine_path=self.scorer_engine_path,
            mesh_path=self.mesh_path,
            target_name=self.target_name,
            camera_intrinsic=camera_matrix,
            max_image_height=self.max_image_height,
            max_image_width=self.max_image_width,
            mem_buf_size=self.mem_buf_size,
        )
        self.camera_matrix = camera_matrix.copy()
        pose_last = self._pose_numpy()
        if pose_last is not None:
            self.session.set_last_pose(pose_last)

    def register(
        self,
        rgb: Any,
        depth: Any,
        camera_matrix: Any,
        object_mask: Any,
        iteration: int = 1,
        object_id: Any | None = None,
    ) -> np.ndarray:
        del object_id
        self._ensure_session(camera_matrix)
        if self.session is None:
            raise RuntimeError("FoundationPose C++ session was not created.")

        pose = self.session.register(
            np.ascontiguousarray(rgb, dtype=np.uint8),
            np.ascontiguousarray(depth, dtype=np.float32),
            np.ascontiguousarray(object_mask, dtype=np.uint8),
            refine_iter=max(1, int(iteration)),
        )
        pose_np = np.asarray(pose, dtype=np.float32).reshape(4, 4)
        self.pose_last = self._pose_tensor(pose_np)
        return pose_np

    def track_one(
        self,
        rgb: Any,
        depth: Any,
        camera_matrix: Any,
        iteration: int = 1,
        extra: dict[str, Any] | None = None,
    ) -> np.ndarray:
        del extra
        self._ensure_session(camera_matrix)
        if self.session is None:
            raise RuntimeError("FoundationPose C++ session was not created.")

        pose = self.session.track_one(
            np.ascontiguousarray(rgb, dtype=np.uint8),
            np.ascontiguousarray(depth, dtype=np.float32),
            last_pose=self._pose_numpy(),
            refine_iter=max(1, int(iteration)),
        )
        pose_np = np.asarray(pose, dtype=np.float32).reshape(4, 4)
        self.pose_last = self._pose_tensor(pose_np)
        return pose_np

    def get(self) -> "FoundationPoseCppWrapper":
        return self


def build_argparser() -> argparse.ArgumentParser:
    tracker_base = _load_tracker_base()
    parser = tracker_base.build_argparser()
    parser.description = "Live ROS2 object pose tracker using foundationpose_cpp."
    parser.add_argument(
        "--foundationpose-cpp-root",
        type=str,
        default=DEFAULT_FOUNDATIONPOSE_CPP_ROOT,
        help="Path to the foundationpose_cpp repository.",
    )
    parser.add_argument(
        "--refiner-engine-path",
        type=str,
        default=None,
        help="TensorRT refiner engine. Defaults to <foundationpose-cpp-root>/models/refiner_hwc_dynamic_fp16.engine.",
    )
    parser.add_argument(
        "--scorer-engine-path",
        type=str,
        default=None,
        help="TensorRT scorer engine. Defaults to <foundationpose-cpp-root>/models/scorer_hwc_dynamic_fp16.engine.",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default=None,
        help="Object target name passed to foundationpose_cpp. Defaults to the mesh filename stem.",
    )
    parser.add_argument("--foundationpose-max-image-height", type=int, default=1080)
    parser.add_argument("--foundationpose-max-image-width", type=int, default=1920)
    parser.add_argument("--foundationpose-mem-buf-size", type=int, default=1)
    return parser


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    tracker_base = _load_tracker_base()
    args.tracker_config = tracker_base._resolve_tracker_config(args)
    args.mesh_path = str(Path(args.mesh_path).expanduser().resolve())
    args.tracker_checkpoint = str(Path(args.tracker_checkpoint).expanduser().resolve())
    args.foundationpose_cpp_root = str(Path(args.foundationpose_cpp_root).expanduser().resolve())

    foundationpose_cpp_root = Path(args.foundationpose_cpp_root)
    if args.refiner_engine_path is None:
        args.refiner_engine_path = str(foundationpose_cpp_root / "models" / "refiner_hwc_dynamic_fp16.engine")
    else:
        args.refiner_engine_path = str(Path(args.refiner_engine_path).expanduser().resolve())

    if args.scorer_engine_path is None:
        args.scorer_engine_path = str(foundationpose_cpp_root / "models" / "scorer_hwc_dynamic_fp16.engine")
    else:
        args.scorer_engine_path = str(Path(args.scorer_engine_path).expanduser().resolve())

    for path_name in ("foundationpose_cpp_root", "refiner_engine_path", "scorer_engine_path"):
        path = Path(getattr(args, path_name))
        if not path.exists():
            raise FileNotFoundError(f"{path_name.replace('_', '-')} does not exist: {path}")

    if args.cam_k is not None and len(args.cam_k) != 9:
        raise ValueError("--cam-k must contain 9 values.")

    return args


def main() -> None:
    if "-h" not in sys.argv and "--help" not in sys.argv:
        _preload_foundationpose_cpp(_foundationpose_cpp_root_from_argv(sys.argv[1:]))

    tracker_base = _load_tracker_base()
    args = _resolve_args(build_argparser().parse_args())
    FoundationPoseCppWrapper.configure(args)
    tracker_base.FoundationPoseWrapper = FoundationPoseCppWrapper

    import rclpy

    rclpy.init()
    tracker = None
    try:
        tracker = tracker_base.LiveObjectPoseTrackerNode(args)
        tracker.node.get_logger().info(
            f"Using foundationpose_cpp from {args.foundationpose_cpp_root} "
            f"with target={tracker.foundationpose.target_name!r}."
        )
        rclpy.spin(tracker.node)
    finally:  # pragma: no cover - ROS runtime path
        if tracker is not None:
            tracker.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
