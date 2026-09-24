"""Lazy registry for public simulation model entry points."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from collections.abc import Callable, Sequence


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    module: str
    smoke_test: bool = False
    input_path_options: tuple[str, ...] = ()

    def load_main(self) -> Callable[[Sequence[str] | None], int | None]:
        return import_module(self.module).main


_MODELS = {
    spec.model_id: spec
    for spec in (
        ModelSpec("curved-tube-2d-arc", "softrobot.curved_tube.arc_2d"),
        ModelSpec(
            "straight-tube-mesh-3d",
            "softrobot.straight_tube.quasi_static_mesh_3d",
            input_path_options=("--mesh-obj",),
        ),
        ModelSpec("straight-tube-v3", "softrobot.straight_tube.v3"),
        ModelSpec(
            "straight-tube-v4",
            "softrobot.straight_tube.v4",
            input_path_options=("--mesh-obj", "--centerline-csv"),
        ),
    )
}


def model_ids() -> tuple[str, ...]:
    return tuple(sorted(_MODELS))


def get_model(model_id: str) -> ModelSpec:
    try:
        return _MODELS[model_id]
    except KeyError as error:
        raise KeyError(f"unknown model {model_id!r}") from error
