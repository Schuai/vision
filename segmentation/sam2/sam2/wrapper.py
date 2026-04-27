"""Lazy wrapper API for SAM2."""

from __future__ import annotations

from typing import Any


class Sam2Wrapper:
    """Stateful loader for SAM2 models and predictors."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.kind: str | None = None
        self.instance: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self.instance is not None

    def load_model(
        self,
        config_file: str,
        checkpoint_path: str | None = None,
        **kwargs: Any,
    ) -> Any:
        from sam2.build_sam import build_sam2

        self.kind = "model"
        self.instance = build_sam2(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=self.device,
            **kwargs,
        )
        return self.instance

    def load_image_predictor(
        self,
        config_file: str,
        checkpoint_path: str | None = None,
        **kwargs: Any,
    ) -> Any:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.kind = "image_predictor"
        model = build_sam2(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=self.device,
            **kwargs,
        )
        self.instance = SAM2ImagePredictor(model)
        return self.instance

    def load_camera_predictor(
        self,
        config_file: str,
        checkpoint_path: str | None = None,
        **kwargs: Any,
    ) -> Any:
        from sam2.build_sam import build_sam2_camera_predictor

        kwargs.setdefault("vos_optimized", True)
        self.kind = "camera_predictor"
        self.instance = build_sam2_camera_predictor(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=self.device,
            **kwargs,
        )
        return self.instance

    def load_video_predictor(
        self,
        config_file: str,
        checkpoint_path: str | None = None,
        **kwargs: Any,
    ) -> Any:
        from sam2.build_sam import build_sam2_video_predictor

        self.kind = "video_predictor"
        self.instance = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=self.device,
            **kwargs,
        )
        return self.instance

    def load_model_from_hf(self, model_id: str, **kwargs: Any) -> Any:
        from sam2.build_sam import build_sam2_hf

        self.kind = "model_hf"
        self.instance = build_sam2_hf(
            model_id=model_id,
            device=self.device,
            **kwargs,
        )
        return self.instance

    def load_video_predictor_from_hf(self, model_id: str, **kwargs: Any) -> Any:
        from sam2.build_sam import build_sam2_video_predictor_hf

        self.kind = "video_predictor_hf"
        self.instance = build_sam2_video_predictor_hf(
            model_id=model_id,
            device=self.device,
            **kwargs,
        )
        return self.instance

    def get(self) -> Any:
        if self.instance is None:
            raise RuntimeError("Sam2Wrapper has not loaded a model yet.")
        return self.instance
