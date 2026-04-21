# FoundationPose Wrapper

## Purpose

This package exposes an importable wrapper for FoundationPose that can be used directly from the root pipeline:

```python
from foundationpose_wrapper import FoundationPoseWrapper
```

The wrapper hides the legacy module layout behind a stable import path and delays heavyweight imports until `load(...)` is called.

## Wrapper Class

`FoundationPoseWrapper` lives in `wrapper.py`.

It is a stateful object that stores:

- the target `device`
- the `debug_dir`
- the loaded estimator
- the mesh
- the scorer, refiner, and rasterizer context created during setup

## Lifecycle

1. Construct the wrapper.
2. Call `load(mesh_path, ...)`.
3. Call `register(...)` for the initial pose estimate.
4. Call `track_one(...)` for subsequent updates.

## Supported Methods

- `load(mesh_path, apply_scale=1.0, force_apply_color=None)`
  Loads the mesh, creates the FoundationPose estimator stack, and moves it to the selected device.

- `register(rgb, depth, camera_matrix, object_mask, iteration=10, object_id=None)`
  Runs the initial pose registration step.

- `track_one(rgb, depth, camera_matrix, iteration=5, extra=None)`
  Runs one pose tracking update.

- `get()`
  Returns the loaded estimator. Raises an error if `load(...)` has not been called.

- `is_loaded`
  Property that reports whether the estimator has been created.

## Example

```python
from foundationpose_wrapper import FoundationPoseWrapper

wrapper = FoundationPoseWrapper(device="cuda:0")
wrapper.load(mesh_path="/path/to/object_mesh.obj", apply_scale=1.0)

pose = wrapper.register(
    rgb=rgb,
    depth=depth,
    camera_matrix=K,
    object_mask=mask,
)

tracked_pose = wrapper.track_one(
    rgb=rgb,
    depth=depth,
    camera_matrix=K,
)
```

## Notes

- The wrapper imports `Utils` and `estimater` lazily inside `load(...)` so that `import foundationpose_wrapper` stays lightweight.
- `load(...)` may still fail if the full upstream FoundationPose runtime dependencies are not installed.
- The current upstream native build flow is not fully repaired yet. In particular, the existing build script still references a missing `bundlesdf/mycuda` path.
- This wrapper focuses on providing a clean import surface for the root integration pipeline, not on rewriting the full upstream FoundationPose runtime.
