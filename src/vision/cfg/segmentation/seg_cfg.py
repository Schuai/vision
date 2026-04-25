import json
import yaml

from dataclasses import dataclass
from typing import Literal


SegModelType = Literal["ETAM", "SAM2"]


@dataclass
class SegmentationCfg:
    """Configuration for tracking a target obj"""
    # ============================= Tracking parameters ============================= #

    # model
    seg_model_type: SegModelType = "ETAM"

    seg_config: str | None = None
    seg_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.seg_model_type not in ("ETAM", "SAM2"):
            raise ValueError("seg_model_type must be one of: ETAM, SAM2.")

    @classmethod
    def from_json(cls, file_path: str) -> "SegmentationCfg":
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_yaml(cls, file_path: str) -> "SegmentationCfg":
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
