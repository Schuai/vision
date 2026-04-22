# Visual Models

## Environment Setup

This repository uses `pixi` to manage Python environments.

The repo root is the primary integration environment for the full pipeline:

- `src/pipeline.py` runs as one Python process.
- The root environment imports submodule wrappers directly.
- Each submodule also keeps its own `pyproject.toml` so it can still be used on its own.

### Repository Layout

- `src/`: root integration layer
- `segmentation/efficient_track_anything`: Efficient Track Anything package and Pixi env
- `segmentation/sam2`: SAM2 package and Pixi env
- `tracking/FoundationPose_plus_plus/FoundationPose`: FoundationPose wrapper package and Pixi env

### Prerequisites

Install the following before setting up the project:

1. `pixi`
2. NVIDIA driver and CUDA toolkit if you plan to use GPU features
3. A local CUDA 12.6 toolkit at `/usr/local/cuda-12.6` if you want to use the provided `cuda126-shell`

### Root Environment

From the repository root:

```bash
pixi install
pixi shell
```

This creates the main integration environment defined in [pyproject.toml](vision/pyproject.toml:1).

Inside the root Pixi shell, you can run:

```bash
python -m src.pipeline --help
```

Quick checks:

```bash
pixi run import-smoke
pixi run wrapper-smoke
pixi run pipeline-dry-run
pixi run env-diagnostics
```

What these do:

- `import-smoke`: verifies the root env can import `efficient_track_anything.wrapper` and `sam2.wrapper`
- `wrapper-smoke`: instantiates the two root-env tracker wrappers without loading model weights
- `pipeline-dry-run`: starts the root integration entrypoint and prints wrapper status
- `env-diagnostics`: prints the active Python path, PyTorch version, CUDA runtime version, and `CUDA_HOME`

### Standalone Submodule Environments

Each submodule can also be installed and tested independently.

#### Efficient Track Anything

```bash
pixi install -m segmentation/efficient_track_anything
pixi run -m segmentation/efficient_track_anything import-smoke
pixi run -m segmentation/efficient_track_anything wrapper-smoke
```

#### SAM2

```bash
pixi install -m segmentation/sam2
pixi run -m segmentation/sam2 import-smoke
pixi run -m segmentation/sam2 wrapper-smoke
```

#### FoundationPose

```bash
pixi install -m tracking/FoundationPose_plus_plus/FoundationPose
pixi run -m tracking/FoundationPose_plus_plus/FoundationPose import-smoke
pixi run -m tracking/FoundationPose_plus_plus/FoundationPose wrapper-smoke
```

### CUDA 12.6 Build Shell

The repository includes a Pixi task that launches `bash` with CUDA build variables set to `/usr/local/cuda-12.6`.

From the repo root:

```bash
pixi run cuda126-shell
```

This sets:

- `CUDA_HOME=/usr/local/cuda-12.6`
- `CUDA_PATH=/usr/local/cuda-12.6`
- `CUDACXX=/usr/local/cuda-12.6/bin/nvcc`

The same `cuda126-shell` task also exists in each submodule manifest.

### Current Runtime Notes

- The validated root environment uses Python `3.12`.
- The current PyTorch runtime resolves to `torch 2.6.0+cu124`.
- The `cuda126-shell` task controls the CUDA toolkit used for local native builds, but it does not change the CUDA runtime bundled with the installed PyTorch wheel.

### Wrapper Imports

The root pipeline imports these stable wrapper entrypoints:

```python
from efficient_track_anything.wrapper import EfficientTrackAnythingWrapper
from sam2.wrapper import Sam2Wrapper
```

`FoundationPoseWrapper` now lives in the dedicated FoundationPose Pixi environment:

```python
from foundationpose_wrapper import FoundationPoseWrapper
```

### Known Caveat

Use `pixi run -m tracking/FoundationPose_plus_plus/FoundationPose ...` for FoundationPose-backed commands such as `src.ros2_pose_tracker`. The root Pixi environment intentionally does not solve FoundationPose anymore, because it uses a different Torch and torchvision stack than the root tracker environment.
