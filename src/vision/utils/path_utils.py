import os
from pathlib import Path


SEG_CHECKPOINT_DIRS = {
    "ETAM": "etam",
    "SAM2": "sam2",
}


def _load_root_dir() -> Path:
    root_dir = os.environ.get("ROOT_DIR")
    if not root_dir:
        raise RuntimeError("ROOT_DIR is not set. Source env.sh before running vision code.")
    return Path(root_dir).expanduser().resolve()


ROOT_DIR = _load_root_dir()


def root_path(*parts: str | Path) -> Path:
    return ROOT_DIR.joinpath(*parts)


def resolve_seg_checkpoint_path(seg_model_type: str, checkpoint: str, must_exist: bool = True) -> Path:
    model_type = seg_model_type.strip().upper()
    if model_type not in SEG_CHECKPOINT_DIRS:
        supported = ", ".join(SEG_CHECKPOINT_DIRS)
        raise ValueError(f"Unsupported seg_model_type '{seg_model_type}'. Supported model types: {supported}.")

    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.suffix != ".pt":
        checkpoint_path = checkpoint_path.with_suffix(".pt")

    if checkpoint_path.is_absolute():
        resolved = checkpoint_path.resolve()
    elif checkpoint_path.parts and checkpoint_path.parts[0] == "checkpoints":
        resolved = root_path(checkpoint_path).resolve()
    elif checkpoint_path.parts and checkpoint_path.parts[0] in SEG_CHECKPOINT_DIRS.values():
        resolved = root_path("checkpoints", checkpoint_path).resolve()
    else:
        resolved = root_path("checkpoints", SEG_CHECKPOINT_DIRS[model_type], checkpoint_path).resolve()

    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {resolved}")
    return resolved
