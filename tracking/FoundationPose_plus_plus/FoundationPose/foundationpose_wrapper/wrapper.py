"""Lazy wrapper API for FoundationPose."""

from __future__ import annotations

from typing import Any


class FoundationPoseWrapper:
    """Stateful wrapper around the FoundationPose estimator lifecycle."""

    def __init__(self, device: str = "cuda:0", debug_dir: str = "./debug/foundationpose") -> None:
        self.device = device
        self.debug_dir = debug_dir
        self.estimator: Any | None = None
        self.mesh: Any | None = None
        self.scorer: Any | None = None
        self.refiner: Any | None = None
        self.glctx: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self.estimator is not None

    def load(
        self,
        mesh_path: str,
        apply_scale: float = 1.0,
        force_apply_color: list[int] | None = None,
    ) -> "FoundationPoseWrapper":
        try:
            import trimesh
            from Utils import trimesh_add_pure_colored_texture
            from estimater import (
                FoundationPose,
                PoseRefinePredictor,
                ScorePredictor,
                dr,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "FoundationPose runtime dependencies are incomplete. "
                "Install the FoundationPose Pixi environment and any native extras required "
                "by the upstream project before calling FoundationPoseWrapper.load()."
            ) from exc

        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh.apply_scale(apply_scale)
        if force_apply_color is not None:
            mesh = trimesh_add_pure_colored_texture(mesh, color=force_apply_color, resolution=10)

        self.mesh = mesh
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.estimator = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            glctx=self.glctx,
            debug_dir=self.debug_dir,
        )
        self.estimator.to_device(self.device)
        return self

    def register(
        self,
        rgb: Any,
        depth: Any,
        camera_matrix: Any,
        object_mask: Any,
        iteration: int = 10,
        object_id: Any | None = None,
    ) -> Any:
        if self.estimator is None:
            raise RuntimeError("FoundationPoseWrapper.load() must be called before register().")
        return self.estimator.register(
            K=camera_matrix,
            rgb=rgb,
            depth=depth,
            ob_mask=object_mask,
            ob_id=object_id,
            iteration=iteration,
        )

    def track_one(
        self,
        rgb: Any,
        depth: Any,
        camera_matrix: Any,
        iteration: int = 5,
        extra: dict[str, Any] | None = None,
    ) -> Any:
        if self.estimator is None:
            raise RuntimeError("FoundationPoseWrapper.load() must be called before track_one().")
        return self.estimator.track_one(
            rgb=rgb,
            depth=depth,
            K=camera_matrix,
            iteration=iteration,
            extra={} if extra is None else extra,
        )

    def get(self) -> Any:
        if self.estimator is None:
            raise RuntimeError("FoundationPoseWrapper has not loaded an estimator yet.")
        return self.estimator
