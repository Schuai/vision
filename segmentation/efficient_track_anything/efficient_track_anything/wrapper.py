"""Lazy wrapper API for Efficient Track Anything."""

from __future__ import annotations

from typing import Any


class EfficientTrackAnythingWrapper:
    """Stateful loader for Efficient Track Anything models and predictors."""

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
        from efficient_track_anything.build_efficienttam import build_efficienttam

        self.kind = "model"
        self.instance = build_efficienttam(
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
        from efficient_track_anything.build_efficienttam import build_efficienttam
        from efficient_track_anything.efficienttam_image_predictor import (
            EfficientTAMImagePredictor,
        )

        self.kind = "image_predictor"
        model = build_efficienttam(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=self.device,
            **kwargs,
        )
        self.instance = EfficientTAMImagePredictor(model)
        return self.instance

    def load_camera_predictor(
        self,
        config_file: str,
        checkpoint_path: str | None = None,
        **kwargs: Any,
    ) -> Any:
        from efficient_track_anything.build_efficienttam import (
            build_efficienttam_camera_predictor,
        )

        self.kind = "camera_predictor"
        self.instance = build_efficienttam_camera_predictor(
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
        from efficient_track_anything.build_efficienttam import (
            build_efficienttam_video_predictor,
        )

        self.kind = "video_predictor"
        self.instance = build_efficienttam_video_predictor(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=self.device,
            **kwargs,
        )
        return self.instance

    def load_video_predictor_from_hf(self, model_id: str, **kwargs: Any) -> Any:
        from efficient_track_anything.build_efficienttam import (
            build_efficienttam_video_predictor_hf,
        )

        self.kind = "video_predictor_hf"
        self.instance = build_efficienttam_video_predictor_hf(
            model_id=model_id,
            device=self.device,
            **kwargs,
        )
        return self.instance

    def get(self) -> Any:
        if self.instance is None:
            raise RuntimeError("EfficientTrackAnythingWrapper has not loaded a model yet.")
        return self.instance
