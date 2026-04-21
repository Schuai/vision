# Efficient Track Anything Wrapper

## Purpose

This package exposes a lightweight wrapper API for the Efficient Track Anything builders.

The wrapper is designed for the root integration pipeline, where we want a stable import path and lazy model loading:

```python
from efficient_track_anything.wrapper import EfficientTrackAnythingWrapper
```

## Wrapper Class

`EfficientTrackAnythingWrapper` lives in `wrapper.py`.

It is a stateful object that:

- remembers the target `device`
- tracks the loaded object in `instance`
- records the current loaded object type in `kind`
- delays heavy imports until a `load_*` method is called

## Supported Methods

- `load_model(config_file, checkpoint_path=None, **kwargs)`
  Loads the base EfficientTAM model via `build_efficienttam(...)`.

- `load_image_predictor(config_file, checkpoint_path=None, **kwargs)`
  Builds the base model and wraps it with `EfficientTAMImagePredictor`.

- `load_camera_predictor(config_file, checkpoint_path=None, **kwargs)`
  Loads the camera predictor via `build_efficienttam_camera_predictor(...)`.

- `load_video_predictor(config_file, checkpoint_path=None, **kwargs)`
  Loads the video predictor via `build_efficienttam_video_predictor(...)`.

- `load_video_predictor_from_hf(model_id, **kwargs)`
  Loads a video predictor from Hugging Face.

- `get()`
  Returns the currently loaded object. Raises an error if nothing has been loaded yet.

- `is_loaded`
  Property that reports whether `instance` is set.

## Example

```python
from efficient_track_anything.wrapper import EfficientTrackAnythingWrapper

wrapper = EfficientTrackAnythingWrapper(device="cuda")

predictor = wrapper.load_video_predictor(
    config_file="configs/efficienttam/efficienttam_ti_512x512.yaml",
    checkpoint_path="/path/to/checkpoint.pt",
)

assert wrapper.is_loaded
same_predictor = wrapper.get()
```

## Notes

- The wrapper only manages loading and access to the underlying ETA object.
- It does not yet add a higher-level inference API such as `predict(...)` or `reset(...)`.
- Loading a new object replaces the previous `instance`.
- The wrapper is also re-exported from `efficient_track_anything.__init__`.
