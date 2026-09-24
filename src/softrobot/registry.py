"""Lazy registry for public simulation model entry points."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    module: str
    smoke_test: bool = False
    input_path_options: tuple[str, ...] = ()
    task: str = "guidewire"
    status: str = "current"
    dimension: str = "3d"
    geometry: str = "straight"
    contact: str = "none"
    backend: str = "torch-cpu"
    source: str = "repository"

    def load_main(self) -> Callable[[Sequence[str] | None], int | None]:
        return import_module(self.module).main


_MODELS = {
    spec.model_id: spec
    for spec in (
        ModelSpec(
            "contact-2d-dynamic-pi", "softrobot.contact.dynamic_pi_2d", True,
            task="contact", status="historical", dimension="2d",
            geometry="parallel-walls", contact="pi-soft-wall",
            source="血管导丝/01_基础施力模型/rod_soft_contact_PI_dynamic.py",
        ),
        ModelSpec(
            "contact-2d-quasistatic-pi-v1", "softrobot.contact.quasistatic_pi_v1", True,
            task="contact", status="historical", dimension="2d",
            geometry="parallel-walls", contact="pi-soft-wall",
            source="SoftRobot.zip root untracked SHA256 1a0b234e...",
        ),
        ModelSpec(
            "contact-2d-quasistatic-pi-v2", "softrobot.contact.quasistatic_pi_v2", True,
            task="contact", status="historical", dimension="2d",
            geometry="parallel-walls", contact="pi-soft-wall",
            source="SoftRobot.zip root untracked SHA256 baa0c14e...",
        ),
        ModelSpec("curved-tube-2d-arc", "softrobot.curved_tube.arc_2d", dimension="2d", geometry="curved", contact="pi-soft-wall"),
        ModelSpec(
            "foundation-2d-stretch-bend-torch", "softrobot.foundations.stretch_bend_2d_torch", True,
            task="foundation", status="historical", dimension="2d",
            source="血管导丝/00_导丝受力变化物理模型/rod_stretch_bend_2d_pytorch.py",
        ),
        ModelSpec(
            "foundation-2d-stretch-numpy", "softrobot.foundations.stretch_2d_numpy", True,
            task="foundation", status="historical", dimension="2d", backend="numpy",
            source="血管导丝/00_导丝受力变化物理模型/rod_simulation_2d_stretch.py",
        ),
        ModelSpec(
            "foundation-3d-implicit-newton", "softrobot.foundations.implicit_newton_3d", True,
            task="foundation", status="historical", dimension="3d",
            source="血管导丝/00_导丝受力变化物理模型/rodsim_implicit_euler_newton3d.py",
        ),
        ModelSpec(
            "foundation-3d-stretch-bend-torch", "softrobot.foundations.stretch_bend_3d_torch", True,
            task="foundation", status="historical", dimension="3d",
            source="血管导丝/00_导丝受力变化物理模型/rodsim_str_bend_pytorch3d.py",
        ),
        ModelSpec(
            "foundation-cantilever-der", "softrobot.foundations.cantilever.run_cantilever_demo", True,
            task="foundation", status="historical", dimension="3d", geometry="cantilever",
            source="SoftRobot.zip commit eb33bf3:Stretching_bending",
        ),
        ModelSpec(
            "straight-tube-mesh-3d", "softrobot.straight_tube.quasi_static_mesh_3d",
            input_path_options=("--mesh-obj",), contact="mesh",
        ),
        ModelSpec("straight-tube-v3", "softrobot.straight_tube.v3", contact="pi-soft-wall"),
        ModelSpec(
            "straight-tube-v4", "softrobot.straight_tube.v4",
            input_path_options=("--mesh-obj", "--centerline-csv"), contact="tube-or-mesh",
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
