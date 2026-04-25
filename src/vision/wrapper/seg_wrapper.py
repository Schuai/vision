import contextlib
from pathlib import Path
from typing import Any

import torch

from sam2.build_sam import build_sam2_camera_predictor
from efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor
from vision.cfg.segmentation.seg_cfg import SegmentationCfg
from vision.utils.path_utils import resolve_seg_checkpoint_path


class SegWrapper:
    """Single-camera segmentation wrapper for ETAM and SAM2 camera predictors."""

    def __init__(self, cfg: SegmentationCfg, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device
        self.model_type = cfg.seg_model_type
        self.seg_config = self._normalize_hydra_config_path(
            self._require_cfg_value("seg_config", cfg.seg_config)
        )
        self.seg_checkpoint = str(resolve_seg_checkpoint_path(
            self.model_type,
            self._require_cfg_value("seg_checkpoint", cfg.seg_checkpoint),
        ))
        self.predictor = self._load_predictor()
        self.frame_idx = 0
        self.obj_idx = 0
        self.is_init = False
        self.is_load_first_frame = False

    @staticmethod
    def _require_cfg_value(name: str, value: str | None) -> str:
        if not value:
            raise ValueError(f"{name} must be set in SegmentationCfg.")
        return value

    @staticmethod
    def _normalize_hydra_config_path(config_file: str) -> str:
        """Convert absolute or repo-relative config paths into Hydra config names."""

        config_path = Path(config_file).expanduser()
        path_parts = config_path.parts
        if "configs" in path_parts:
            configs_index = path_parts.index("configs")
            return Path(*path_parts[configs_index:]).as_posix()
        return config_file

    @staticmethod
    def _initialize_hydra_config_module(config_module: str) -> None:
        """Point Hydra at the model package whose configs are about to be composed."""

        from hydra import initialize_config_module
        from hydra.core.global_hydra import GlobalHydra

        hydra = GlobalHydra.instance()
        if hydra.is_initialized():
            hydra.clear()
        initialize_config_module(config_module=config_module, version_base="1.2")

    def _load_predictor(self) -> Any:
        if self.model_type == "SAM2":
            self._initialize_hydra_config_module("sam2")
            return build_sam2_camera_predictor(
                config_file=self.seg_config,
                ckpt_path=self.seg_checkpoint,
                device=self.device,
            )
        if self.model_type == "ETAM":
            self._initialize_hydra_config_module("efficient_track_anything")
            return build_efficienttam_camera_predictor(
                config_file=self.seg_config,
                ckpt_path=self.seg_checkpoint,
                device=self.device,
                vos_optimized=True,
            )
        raise ValueError(f"Unsupported seg_model_type: {self.model_type}")

    @contextlib.contextmanager
    def _predictor_context(self):
        if self.device.startswith("cuda") and torch.cuda.is_available():
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                yield
        else:
            with torch.inference_mode():
                yield

    def load_first_frame(self, frame) -> None:
        with self._predictor_context():
            self.predictor.load_first_frame(frame)
        self.frame_idx = 0
        self.is_load_first_frame = True

    def add_new_prompt(
        self,
        obj_id: int | None = None,
        points=None,
        labels=None,
        bbox=None,
        frame_idx: int = 0,
    ):
        if not self.is_load_first_frame:
            raise RuntimeError("SegWrapper must load the first frame before adding prompts.")

        if obj_id is None:
            obj_id = self.obj_idx
            self.obj_idx += 1

        with self._predictor_context():
            result = self.predictor.add_new_prompt(
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
                bbox=bbox,
            )
        self.is_init = True
        return result

    def track(self, frame):
        if not self.is_init:
            raise RuntimeError("SegWrapper must be initialized with a prompt before tracking.")

        with self._predictor_context():
            return self.predictor.track(frame)
