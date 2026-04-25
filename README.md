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
2. NVIDIA driver if you plan to use GPU features

The CUDA toolkit used for local builds is now installed inside the Pixi environment instead of being taken from `/usr/local/cuda-*`.

### Root Environment

From the repository root:

```bash
pixi install
pixi shell
```

This creates the default root integration environment defined in [pyproject.toml](vision/pyproject.toml:1), which is `jazzy-cuda126`.

The root workspace also exposes an opt-in `humble-cuda124` environment:

```bash
pixi install -e humble-cuda124
pixi shell -e humble-cuda124
```

Inside the root Pixi shell, you can run:

```bash
python -m src.pipeline --help
```

Plain `pixi run ...` uses the default Jazzy environment. For Humble, use `pixi run -e humble-cuda124 ...`.

### Tracking ROS2 Package

Build the dedicated tracking package from a Pixi shell:

```bash
pixi run ros2-tracking-build
source ros2/tracking/install/setup.bash
```

Start the publisher with:

```bash
ros2 run tracking realsense_publisher --serial-number 251622061129
```

Start segmentation tracking from the publisher's RGB stream. If no prompt is passed, the first RGB frame opens in a prompt window for bbox selection:

```bash
ros2 run tracking run_segmentation
```

Publish an annotated image for RViz from the RGB stream and segmentation bbox:

```bash
ros2 run tracking visualization
```

In RViz, add an `Image` display and select `/tracking/visualization/image`.

For a packaging/import check without starting streams:

```bash
ros2 run tracking realsense_publisher --help
ros2 run tracking realsense_publisher --list-devices
```

### Standalone Submodule Environments

Each submodule can also be installed independently.

#### Efficient Track Anything

```bash
pixi install -m segmentation/efficient_track_anything
```

#### SAM2

```bash
pixi install -m segmentation/sam2
```

#### FoundationPose

```bash
pixi install -m tracking/FoundationPose_plus_plus/FoundationPose
```

### CUDA Build Shells

The root workspace includes Pixi tasks that launch `bash` with CUDA build variables set to the CUDA toolkit installed inside the selected Pixi environment.

For the default Jazzy environment:

```bash
pixi run cuda126-shell
```

This shell resolves `nvcc` from the active Pixi environment and sets `CUDA_HOME`, `CUDA_PATH`, and `CUDACXX` to that in-environment toolkit.

For the Humble environment, use the explicit environment selection pattern:

```bash
pixi run -e humble-cuda124 cuda124-shell
```

This does the same for the `humble-cuda124` environment, using the CUDA toolkit installed into that environment.

For FoundationPose-specific native builds, use the same `cuda126-shell` task from that submodule environment:

```bash
pixi run -m tracking/FoundationPose_plus_plus/FoundationPose cuda126-shell
```

The same `cuda126-shell` task also exists in each submodule manifest.

### Current Runtime Notes

- The validated root environments use Python `3.12`.
- The default root workflow is `jazzy-cuda126`.
- The opt-in Humble root workflow is `humble-cuda124`.
- The current shared PyTorch runtime resolves to `torch 2.6.0+cu124`.
- The `cuda126-shell` and `cuda124-shell` tasks now use the CUDA toolkit installed by Pixi inside the environment for local native builds.
- Those tasks still do not change the CUDA runtime bundled with the installed PyTorch wheel.
- The NVIDIA driver still comes from the host machine.

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

### ROS2 Pose Tracker

`src.ros2_pose_tracker` currently depends on the dedicated FoundationPose environment plus the local SAM2 source tree on `PYTHONPATH`.

Recommended setup on a new machine:

```bash
CONDA_OVERRIDE_CUDA=12.1 pixi install -m tracking/FoundationPose_plus_plus/FoundationPose
```

Then open the FoundationPose CUDA build shell:

```bash
pixi run -m tracking/FoundationPose_plus_plus/FoundationPose cuda126-shell
```

Inside that shell, install the heavier runtime extras and build the local native extension once:

```bash
pixi run -m tracking/FoundationPose_plus_plus/FoundationPose install-runtime
pixi run -m tracking/FoundationPose_plus_plus/FoundationPose build-native
exit
```

`pixi install -m tracking/FoundationPose_plus_plus/FoundationPose` now installs `warp-lang` and the Boost development package needed for the local `mycpp` build. The `install-runtime` step is still needed for heavier runtime extras such as `pytorch3d` and `nvdiffrast`.

Download tracker checkpoints into the repository `checkpoints/` folder:

```bash
cd checkpoints
bash download_ckpts.sh
cd ..
```

The script stores SAM2 checkpoints under `checkpoints/sam2/` and EfficientTAM checkpoints under `checkpoints/etam/`.

Run the camera publisher in one terminal:

```bash
ros2 run tracking realsense_publisher --serial-number 251622061129
```

Run the tracker with EfficientTAM in a second terminal:

```bash
pixi run -m tracking/FoundationPose_plus_plus/FoundationPose bash -lc 'export PYTHONPATH="$PWD/segmentation/sam2${PYTHONPATH:+:$PYTHONPATH}"; python -m src.ros2_pose_tracker --tracker efficient_tam --tracker-config configs/efficienttam/efficienttam_s_512x512.yaml --tracker-checkpoint checkpoints/etam/efficienttam_s_512x512.pt --mesh-path demo_data/t_shape/t_shape.obj --timing-report-every 1'
```

Use the dedicated FoundationPose Pixi environment for `src.ros2_pose_tracker`. The root Pixi environment does not install `foundationpose_wrapper`, so direct root-environment launches will fail with `ModuleNotFoundError: No module named 'foundationpose_wrapper'`.

The tracker no longer takes an initialization mask or bbox from the command line. After startup, focus the RGB window, press `e` to enter edit mode, draw the initialization box, then press `e` again to confirm and start tracking. If `mycpp` is unavailable, `src.ros2_pose_tracker` falls back to an unclustered rotation-grid path, but `build-native` is the recommended setup.

The OpenCV visualization and edit windows open at `1.5x` the rendered frame size by default. Pass `--visualization-window-scale 2.0` in a custom launch command if you want them larger.

During tracking, the node checks FoundationPose confidence by rendering the current 6D pose and comparing rendered depth against the camera depth inside the 2D tracker mask. If the masked depth error is above `--pose-depth-error-threshold 0.05`, the next FoundationPose loop restarts from `register(...)` using the current 2D tracker mask; the 2D tracker itself is not reinitialized. The Kalman filter is reinitialized from the registered pose. Use `--no-retrack-on-low-confidence` to disable this recovery path.
