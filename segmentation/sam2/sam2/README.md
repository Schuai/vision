# SAM2 Wrapper

## Purpose

This package exposes a lightweight wrapper API for SAM2 builders and predictors.

The intended stable import path is:

```python
from sam2.wrapper import Sam2Wrapper
```

The wrapper is used by the root pipeline so that SAM2 can be imported without loading heavy model code at import time.

## Wrapper Class

`Sam2Wrapper` lives in `wrapper.py`.

It is a stateful object that:

- stores the target `device`
- keeps the loaded object in `instance`
- records the active object type in `kind`
- performs heavy imports only inside `load_*` methods

## Supported Methods

- `load_model(config_file, checkpoint_path=None, **kwargs)`
  Loads the base SAM2 model with `build_sam2(...)`.

- `load_image_predictor(config_file, checkpoint_path=None, **kwargs)`
  Builds the base model and wraps it with `SAM2ImagePredictor`.

- `load_camera_predictor(config_file, checkpoint_path=None, **kwargs)`
  Loads the camera predictor with `build_sam2_camera_predictor(...)`.

- `load_video_predictor(config_file, checkpoint_path=None, **kwargs)`
  Loads the video predictor with `build_sam2_video_predictor(...)`.

- `load_model_from_hf(model_id, **kwargs)`
  Loads a base model from Hugging Face.

- `load_video_predictor_from_hf(model_id, **kwargs)`
  Loads a video predictor from Hugging Face.

- `get()`
  Returns the current loaded object. Raises an error if nothing has been loaded yet.

- `is_loaded`
  Property that reports whether a model or predictor has been loaded.

## Example

```python
from sam2.wrapper import Sam2Wrapper

wrapper = Sam2Wrapper(device="cuda")

predictor = wrapper.load_video_predictor(
    config_file="configs/sam2.1/sam2.1_hiera_t.yaml",
    checkpoint_path="/path/to/checkpoint.pt",
)

assert wrapper.is_loaded
same_predictor = wrapper.get()
```

## Notes

- The wrapper is intentionally thin and stays close to upstream SAM2 behavior.
- Extra keyword arguments are forwarded directly to the underlying builder functions.
- Loading a new object replaces the previous `instance`.
- The wrapper is also re-exported from `sam2.__init__`.
