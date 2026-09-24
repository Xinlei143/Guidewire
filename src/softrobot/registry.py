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
    default_model_args: tuple[str, ...] = ()
    timeout_s: float = 60.0
    smoke_timeout_s: float | None = None

    def __post_init__(self) -> None:
        if self.smoke_timeout_s is None:
            object.__setattr__(self, "smoke_timeout_s", self.timeout_s)

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
            source="SoftRobot.zip root untracked SHA256 1a0b234ef0693053d2296a99e5b023888c31fe6f913d099f6b551610a40a894b",
        ),
        ModelSpec(
            "contact-2d-quasistatic-pi-v2", "softrobot.contact.quasistatic_pi_v2", True,
            task="contact", status="historical", dimension="2d",
            geometry="parallel-walls", contact="pi-soft-wall",
            source="SoftRobot.zip root untracked SHA256 baa0c14e3b9b12bd93b14b0def26c03006b494b4eed61f5f69cb5398945d76cf",
        ),
        ModelSpec("curved-tube-2d-arc", "softrobot.curved_tube.arc_2d", True, dimension="2d", geometry="curved", contact="pi-soft-wall", backend="python+cpp", smoke_timeout_s=300.0),
        ModelSpec("curved-tube-2d-legacy-python", "softrobot.curved_tube.legacy_arc_2d", True, status="historical", dimension="2d", geometry="curved", contact="pi-soft-wall", source="血管导丝/03_arc_tube_model/二维弧形管/rod_soft_contact_PI_dynamic_quasi_static_2d_arc.py"),
        ModelSpec("curved-tube-3d-legacy-analytic", "softrobot.curved_tube.legacy_arc_3d_analytic", True, status="historical", geometry="curved", contact="analytic-arc", source="血管导丝/03_arc_tube_model/三维弧形管/rod_soft_contact_PI_dynamic_quasi_static_3d_arc.py", default_model_args=("--contact-type", "analytic")),
        ModelSpec("curved-tube-3d-legacy-mesh", "softrobot.curved_tube.legacy_arc_3d_mesh", True, status="historical", geometry="curved", contact="mesh", source="血管导丝/03_arc_tube_model/三维弧形管/rod_soft_contact_PI_dynamic_quasi_static_3d_arc.py", default_model_args=("--contact-type", "mesh")),
        ModelSpec("curved-tube-v4-analytic", "softrobot.curved_tube.v4_analytic", True, geometry="curved", contact="analytic-centerline", backend="python+cpp", default_model_args=("--contact-type", "curved-tube"), smoke_timeout_s=300.0),
        ModelSpec("curved-tube-v4-mesh", "softrobot.curved_tube.v4_mesh", True, input_path_options=("--mesh-obj", "--centerline-csv"), geometry="curved", contact="mesh", backend="python+cpp", default_model_args=("--contact-type", "mesh", "--mesh-obj", "__ARC_MESH__", "--centerline-csv", "__ARC_CENTERLINE__"), smoke_timeout_s=300.0),
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
        ModelSpec("straight-tube-2d-pi", "softrobot.straight_tube.legacy_pi_2d", True, status="historical", dimension="2d", contact="pi-soft-wall", source="血管导丝/02_直圆管模型/二维直圆管/rodsim_2d_vessel_boundary_PI.py"),
        ModelSpec(
            "straight-tube-mesh-3d", "softrobot.straight_tube.quasi_static_mesh_3d",
            True, input_path_options=("--mesh-obj",), contact="mesh",
        ),
        ModelSpec("straight-tube-v2-3d", "softrobot.straight_tube.v2", True, status="historical", contact="pi-soft-wall", source="血管导丝/02_直圆管模型/三维直圆管/rod_soft_contact_PI_dynamic_quasi_static_3d_v2.py"),
        ModelSpec("straight-tube-v3", "softrobot.straight_tube.v3", True, contact="pi-soft-wall"),
        ModelSpec(
            "straight-tube-v4", "softrobot.straight_tube.v4", True,
            input_path_options=("--mesh-obj", "--centerline-csv"), contact="tube-or-mesh", backend="python+cpp", smoke_timeout_s=300.0,
        ),
    )
}


_TASK_ORDER = {"foundation": 0, "contact": 1, "straight": 2, "curved": 3}


def model_ids() -> tuple[str, ...]:
    return tuple(sorted(_MODELS, key=lambda model_id: (_TASK_ORDER[model_id.split("-", 1)[0]], model_id)))


def get_model(model_id: str) -> ModelSpec:
    try:
        return _MODELS[model_id]
    except KeyError as error:
        raise KeyError(f"unknown model {model_id!r}") from error
