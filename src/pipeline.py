"""Root integration entrypoint for a monolithic vision pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict, dataclass

from efficient_track_anything.wrapper import EfficientTrackAnythingWrapper
from sam2.wrapper import Sam2Wrapper


@dataclass
class PipelineStatus:
    efficient_track_anything_loaded: bool
    sam2_loaded: bool
    foundationpose_loaded: bool
    device: str


class VisionPipeline:
    """Owns long-lived model wrappers for a single-process pipeline."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.efficient_track_anything = EfficientTrackAnythingWrapper(device=device)
        self.sam2 = Sam2Wrapper(device=device)
        self.foundationpose = None
        if importlib.util.find_spec("foundationpose_wrapper") is not None:
            from foundationpose_wrapper import FoundationPoseWrapper

            self.foundationpose = FoundationPoseWrapper(device=device)

    def status(self) -> PipelineStatus:
        return PipelineStatus(
            efficient_track_anything_loaded=self.efficient_track_anything.is_loaded,
            sam2_loaded=self.sam2.is_loaded,
            foundationpose_loaded=bool(self.foundationpose and self.foundationpose.is_loaded),
            device=self.device,
        )

    def dry_run(self) -> dict[str, object]:
        """Return a lightweight summary without loading heavy models."""
        return asdict(self.status())


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vision pipeline integration entrypoint.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device string to assign to wrapper instances.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Instantiate wrappers and print the pipeline status without loading model weights.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    pipeline = VisionPipeline(device=args.device)
    if args.dry_run:
        print(json.dumps(pipeline.dry_run(), indent=2, sort_keys=True))
        return
    print(json.dumps(pipeline.status().__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
