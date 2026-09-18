"""CPU float64 quasi-static insertion of a DER wire through a 2-D sine tube."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
import math

import numpy as np
import torch
import torch.nn.functional as F

from rod_soft_contact_PI_dynamic_quasi_static_2d_arc import (
    NewtonInfo,
    assemble_elastic_banded,
    elastic_energy,
    get_block_plan,
)
from sine_tube_2d import SineTube2D


@dataclass
class SineNewtonInfo(NewtonInfo):
    """Sine solver diagnostics without changing the public Arc2D contract."""

    dense_fallback_count: int = 0


@dataclass(frozen=True)
class SineInsertConfig:
    """Physical, geometric, contact, insertion, and output configuration."""

    amplitude: float = 0.225
    x_span: float = 1.5
    tube_half_width: float = 0.05
    safe_distance: float = 0.01

    wire_length: float = 2.5
    num_nodes: int = 800
    initial_inserted_length: float = 0.12
    total_insertion: float = 1.5
    num_load_steps: int = 60
    outlet_margin: float = 0.1

    feeder_half_width: float = 0.002
    feeder_contact_layer: float = 0.0005
    feeder_kp_line: float = 1_118_600.0

    contact_kp_line: float = 1_118_600.0
    contact_ki_line: float = 31_960.0
    contact_force_cap_line: float = 6_392.0
    integral_leak_rate: float = 5.0
    max_integral_state: float = 5.0e-2
    contact_smooth_length: float = 2.0e-5
    direction_epsilon: float = 1.0e-6
    friction_mu: float = 0.10
    friction_kt_line: float = 1_118_600.0

    diameter: float = 1.0e-3
    density: float = 7850.0
    young_modulus: float = 200.0e9
    dt: float = 0.005

    damping_ratio: float = 1.0
    max_newton_iters: int = 35
    residual_tol: float = 1.0e-7
    step_tol: float = 1.0e-10
    settle_speed_tol: float = 1.0e-3
    settle_residual_tol: float = 1.0e-6
    contact_tolerance: float = 5.0e-4
    settle_confirm_steps: int = 4
    max_settle_steps: int = 1600
    prebalance: bool = True
    prebalance_max_steps: int = 1600
    ramp_steps: int = 25
    history_save_interval: int = 20
    kp_grow_factor: float = 1.5
    kp_max_factor: float = 8.0
    max_kp_retries: int = 3
    ki_reduction_factor: float = 0.5
    ki_zero_threshold: float = 1.0
    max_ki_retries: int = 8
    retry_subdivisions: tuple[int, ...] = (2, 4)
    projection_trust_radius: float = 0.02

    output_dir: str = "2d_npz"
    run_name: str = "rod_soft_contact_PI_quasi_static_2d_sine_insert"
    backend: str = "cpp"

    def __post_init__(self) -> None:
        if isinstance(self.friction_mu, (bool, np.bool_)):
            raise ValueError("friction_mu must be numeric")
        if isinstance(self.friction_kt_line, (bool, np.bool_)):
            raise ValueError("friction_kt_line must be numeric")
        positive = (
            "amplitude", "x_span", "tube_half_width", "safe_distance",
            "wire_length", "initial_inserted_length", "total_insertion",
            "outlet_margin", "feeder_half_width", "feeder_contact_layer",
            "feeder_kp_line", "contact_kp_line", "contact_force_cap_line",
            "max_integral_state", "contact_smooth_length", "direction_epsilon",
            "friction_kt_line", "diameter", "density", "young_modulus", "dt",
            "damping_ratio", "residual_tol", "step_tol", "settle_speed_tol",
            "settle_residual_tol", "contact_tolerance", "kp_max_factor",
            "projection_trust_radius",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("contact_ki_line", "integral_leak_rate", "friction_mu"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.num_nodes, (int, np.integer)) or isinstance(self.num_nodes, bool) or self.num_nodes < 4:
            raise ValueError("num_nodes must be an integer of at least four")
        if not isinstance(self.num_load_steps, (int, np.integer)) or isinstance(self.num_load_steps, bool) or self.num_load_steps < 1:
            raise ValueError("num_load_steps must be a positive integer")
        integer_positive = (
            "max_newton_iters", "settle_confirm_steps", "max_settle_steps",
            "prebalance_max_steps", "ramp_steps", "history_save_interval",
        )
        for name in integer_positive:
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_kp_retries", "max_ki_retries"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.prebalance, (bool, np.bool_)):
            raise ValueError("prebalance must be bool")
        if (
            not isinstance(self.retry_subdivisions, tuple)
            or any(
                not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_))
                or value < 2
                for value in self.retry_subdivisions
            )
        ):
            raise ValueError("retry_subdivisions must be a tuple of integers at least two")
        if not math.isfinite(float(self.kp_grow_factor)) or self.kp_grow_factor <= 1.0:
            raise ValueError("kp_grow_factor must be finite and greater than one")
        if self.kp_max_factor < 1.0:
            raise ValueError("kp_max_factor must be at least one")
        if not math.isfinite(float(self.ki_reduction_factor)) or not 0.0 < self.ki_reduction_factor < 1.0:
            raise ValueError("ki_reduction_factor must be finite and lie in (0, 1)")
        if (
            isinstance(self.ki_zero_threshold, (bool, np.bool_))
            or not math.isfinite(float(self.ki_zero_threshold))
            or self.ki_zero_threshold < 0.0
        ):
            raise ValueError("ki_zero_threshold must be finite and non-negative")
        if self.safe_distance >= self.tube_half_width:
            raise ValueError("safe_distance must be smaller than tube_half_width")
        if self.feeder_contact_layer >= self.feeder_half_width:
            raise ValueError("feeder_contact_layer must be smaller than feeder_half_width")
        if self.initial_inserted_length >= self.wire_length:
            raise ValueError("initial_inserted_length must be smaller than wire_length")
        if self.backend not in ("python", "cpp"):
            raise ValueError("backend must be 'python' or 'cpp'")
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("output_dir must be a non-empty string")
        if (
            not isinstance(self.run_name, str)
            or not self.run_name.strip()
            or self.run_name in (".", "..")
            or "/" in self.run_name
            or "\\" in self.run_name
        ):
            raise ValueError("run_name must be a safe, non-empty directory name")

        tube = SineTube2D(self.amplitude, self.x_span, self.tube_half_width)
        available = tube.vessel_length - self.outlet_margin
        if available <= 0.0 or self.initial_inserted_length + self.total_insertion >= available:
            raise ValueError("initial inserted length plus insertion must remain before the outlet margin")


@dataclass(frozen=True)
class RodProperties:
    total_mass: float
    axial_stiffness: float
    bending_stiffness: float


@dataclass(frozen=True)
class RodInitialization:
    positions: np.ndarray
    velocities: np.ndarray
    rest_lengths: np.ndarray
    reference_curvature: np.ndarray
    nodal_lengths: np.ndarray
    masses: np.ndarray
    integrals: np.ndarray
    previous_errors: np.ndarray
    properties: RodProperties


@dataclass(frozen=True)
class PrescribedState:
    mask: np.ndarray
    positions: np.ndarray


@dataclass(frozen=True, init=False)
class SineContactFrame:
    _q: torch.Tensor
    _center: torch.Tensor
    _tangent: torch.Tensor
    _normal: torch.Tensor
    _region: torch.Tensor

    def __init__(
        self,
        q: torch.Tensor,
        center: torch.Tensor,
        tangent: torch.Tensor,
        normal: torch.Tensor,
        region: torch.Tensor,
    ) -> None:
        if not isinstance(q, torch.Tensor):
            raise TypeError("frame.q must be a CPU torch.float64 tensor")
        if q.ndim != 1:
            raise ValueError("frame.q must have shape (N,)")
        count = len(q)
        _require_cpu_float64(q, "frame.q", (count,))
        _require_cpu_float64(center, "frame.center", (count, 2))
        _require_cpu_float64(tangent, "frame.tangent", (count, 2))
        _require_cpu_float64(normal, "frame.normal", (count, 2))
        _require_cpu_bool(region, "frame.region", count)
        tensors = (q, center, tangent, normal)
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("contact frame tensors must be finite")
        tangent_norm = torch.linalg.vector_norm(tangent, dim=1)
        normal_norm = torch.linalg.vector_norm(normal, dim=1)
        ones = torch.ones(count, dtype=torch.float64)
        tolerance = 5.0e-10
        if not torch.allclose(tangent_norm, ones, rtol=tolerance, atol=tolerance):
            raise ValueError("frame.tangent rows must be unit vectors")
        if not torch.allclose(normal_norm, ones, rtol=tolerance, atol=tolerance):
            raise ValueError("frame.normal rows must be unit vectors")
        orthogonality = torch.sum(tangent * normal, dim=1)
        if bool(torch.any(torch.abs(orthogonality) > tolerance)):
            raise ValueError("frame tangent and normal rows must be orthogonal")
        object.__setattr__(self, "_q", q.clone())
        object.__setattr__(self, "_center", center.clone())
        object.__setattr__(self, "_tangent", tangent.clone())
        object.__setattr__(self, "_normal", normal.clone())
        object.__setattr__(self, "_region", region.clone())

    @property
    def q(self) -> torch.Tensor:
        return self._q.clone()

    @property
    def center(self) -> torch.Tensor:
        return self._center.clone()

    @property
    def tangent(self) -> torch.Tensor:
        return self._tangent.clone()

    @property
    def normal(self) -> torch.Tensor:
        return self._normal.clone()

    @property
    def region(self) -> torch.Tensor:
        return self._region.clone()

    @property
    def vessel_region(self) -> torch.Tensor:
        return self._region.clone()

    @property
    def feeder_region(self) -> torch.Tensor:
        return ~self._region


def _frame_data(
    frame: SineContactFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return validated private frame storage for trusted internal evaluation."""
    if not isinstance(frame, SineContactFrame):
        raise TypeError("frame must be a SineContactFrame")
    return frame._q, frame._center, frame._tangent, frame._normal, frame._region


@dataclass(frozen=True)
class SineContactResult:
    vessel_normal_force: torch.Tensor
    feeder_force: torch.Tensor
    total_force: torch.Tensor
    new_integrals: torch.Tensor
    active: torch.Tensor
    weight: torch.Tensor
    error: torch.Tensor
    clearance: torch.Tensor
    q: torch.Tensor
    tangent: torch.Tensor
    normal: torch.Tensor
    region: torch.Tensor
    signed_distance: torch.Tensor

    @property
    def vessel_force(self) -> torch.Tensor:
        return self.vessel_normal_force

    @property
    def total_normal_force(self) -> torch.Tensor:
        return self.total_force

    @property
    def integrals(self) -> torch.Tensor:
        return self.new_integrals


def _clone_tensor_fields(instance: object) -> None:
    """Break caller aliases for tensor-valued state held by a frozen dataclass."""
    for item in fields(instance):
        value = getattr(instance, item.name)
        if isinstance(value, torch.Tensor):
            object.__setattr__(instance, item.name, value.clone())


def _validate_friction_state(
    elastic_slip: torch.Tensor,
    anchors: torch.Tensor,
    q: torch.Tensor,
    contact_active: torch.Tensor,
    owner: str,
) -> int:
    if not isinstance(elastic_slip, torch.Tensor) or elastic_slip.ndim != 1:
        raise ValueError(f"{owner}.elastic_slip must be a length-N tensor")
    count = len(elastic_slip)
    _require_cpu_float64(elastic_slip, f"{owner}.elastic_slip", (count,))
    _require_cpu_float64(anchors, f"{owner}.anchors", (count, 2))
    _require_cpu_float64(q, f"{owner}.q", (count,))
    if (
        not isinstance(contact_active, torch.Tensor)
        or contact_active.dtype != torch.bool
        or contact_active.device.type != "cpu"
        or tuple(contact_active.shape) != (count,)
    ):
        raise TypeError(f"{owner}.contact_active must be a length-N CPU bool tensor")
    if not bool(torch.isfinite(elastic_slip).all()):
        raise ValueError(f"{owner}.elastic_slip must be finite")
    if not bool(torch.isfinite(anchors).all()) or not bool(torch.isfinite(q).all()):
        raise ValueError(f"{owner} metadata must be finite")
    return count


@dataclass(frozen=True, init=False)
class FrictionHistory:
    """Committed tangential state; construction breaks all caller tensor aliases."""

    _elastic_slip: torch.Tensor
    _anchors: torch.Tensor
    _q: torch.Tensor
    _contact_active: torch.Tensor

    def __init__(
        self,
        elastic_slip: torch.Tensor,
        anchors: torch.Tensor,
        q: torch.Tensor,
        contact_active: torch.Tensor,
    ) -> None:
        _validate_friction_state(
            elastic_slip, anchors, q, contact_active, "history"
        )
        object.__setattr__(self, "_elastic_slip", elastic_slip.clone())
        object.__setattr__(self, "_anchors", anchors.clone())
        object.__setattr__(self, "_q", q.clone())
        object.__setattr__(self, "_contact_active", contact_active.clone())

    @property
    def elastic_slip(self) -> torch.Tensor:
        return self._elastic_slip.clone()

    @property
    def anchors(self) -> torch.Tensor:
        return self._anchors.clone()

    @property
    def q(self) -> torch.Tensor:
        return self._q.clone()

    @property
    def contact_active(self) -> torch.Tensor:
        return self._contact_active.clone()

    @property
    def xi(self) -> torch.Tensor:
        return self._elastic_slip.clone()

    @property
    def committed_q(self) -> torch.Tensor:
        return self._q.clone()

    @property
    def committed_contact(self) -> torch.Tensor:
        return self._contact_active.clone()


@dataclass(frozen=True, init=False)
class FrictionIncrementContext:
    """Frozen committed state plus accepted positions at one increment's start."""

    _elastic_slip: torch.Tensor
    _anchors: torch.Tensor
    _q: torch.Tensor
    _contact_active: torch.Tensor

    def __init__(
        self,
        elastic_slip: torch.Tensor,
        anchors: torch.Tensor,
        q: torch.Tensor,
        contact_active: torch.Tensor,
    ) -> None:
        _validate_friction_state(
            elastic_slip, anchors, q, contact_active, "context"
        )
        object.__setattr__(self, "_elastic_slip", elastic_slip.clone())
        object.__setattr__(self, "_anchors", anchors.clone())
        object.__setattr__(self, "_q", q.clone())
        object.__setattr__(self, "_contact_active", contact_active.clone())

    @property
    def elastic_slip(self) -> torch.Tensor:
        return self._elastic_slip.clone()

    @property
    def anchors(self) -> torch.Tensor:
        return self._anchors.clone()

    @property
    def q(self) -> torch.Tensor:
        return self._q.clone()

    @property
    def contact_active(self) -> torch.Tensor:
        return self._contact_active.clone()

    @property
    def committed_elastic_slip(self) -> torch.Tensor:
        return self._elastic_slip.clone()

    @property
    def committed_q(self) -> torch.Tensor:
        return self._q.clone()

    @property
    def committed_contact(self) -> torch.Tensor:
        return self._contact_active.clone()


def _friction_state_data(
    state: FrictionHistory | FrictionIncrementContext,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return private immutable-snapshot storage for trusted solver internals."""
    if not isinstance(state, (FrictionHistory, FrictionIncrementContext)):
        raise TypeError("state must be friction history or increment context")
    return state._elastic_slip, state._anchors, state._q, state._contact_active


@dataclass(frozen=True)
class FrictionResult:
    """One non-committing friction evaluation and its isolated public metadata."""

    friction_force: torch.Tensor
    tangent_traction: torch.Tensor
    trial_slip: torch.Tensor
    candidate_slip: torch.Tensor
    stick: torch.Tensor
    slip: torch.Tensor
    current_active: torch.Tensor
    coulomb_limit: torch.Tensor
    q: torch.Tensor
    tangent: torch.Tensor

    def __post_init__(self) -> None:
        _clone_tensor_fields(self)

    @property
    def force(self) -> torch.Tensor:
        return self.friction_force

    @property
    def traction(self) -> torch.Tensor:
        return self.tangent_traction


def nodal_voronoi_lengths(rest_lengths: np.ndarray) -> np.ndarray:
    """Return node-associated lengths using half adjacent edges at the ends."""
    rest = np.asarray(rest_lengths, dtype=np.float64)
    if rest.ndim != 1 or rest.size < 1 or not np.all(np.isfinite(rest)) or np.any(rest <= 0.0):
        raise ValueError("rest_lengths must be a non-empty vector of finite positive lengths")
    nodal = np.empty(rest.size + 1, dtype=np.float64)
    nodal[0] = 0.5 * rest[0]
    nodal[-1] = 0.5 * rest[-1]
    if rest.size > 1:
        nodal[1:-1] = 0.5 * (rest[:-1] + rest[1:])
    return nodal


def _composite_centerline_samples(config: SineInsertConfig, tube: SineTube2D) -> np.ndarray:
    """Sample the straight feeder and sine centerline with an explicit entry node."""
    interval_count = config.num_nodes - 1
    feeder_length = config.wire_length - config.initial_inserted_length
    feeder_intervals = int(round(interval_count * feeder_length / config.wire_length))
    feeder_intervals = min(max(feeder_intervals, 1), interval_count - 1)
    vessel_intervals = interval_count - feeder_intervals

    feeder_x = np.linspace(-feeder_length, 0.0, feeder_intervals + 1, dtype=np.float64)
    feeder = np.column_stack((feeder_x, np.full_like(feeder_x, config.amplitude)))
    inserted_s = np.linspace(0.0, config.initial_inserted_length, vessel_intervals + 1, dtype=np.float64)
    vessel = np.asarray(tube.centerline(tube.parameter_at_arclength(inserted_s)), dtype=np.float64)
    return np.vstack((feeder[:-1], vessel))


def initialize_rod(config: SineInsertConfig, tube: SineTube2D) -> RodInitialization:
    """Create the C1 feeder/centerline insert with actual chords as rest lengths."""
    expected = (config.amplitude, config.x_span, config.tube_half_width)
    actual = (tube.amplitude, tube.x_span, tube.tube_half_width)
    if not np.allclose(actual, expected, rtol=0.0, atol=1.0e-15):
        raise ValueError("tube geometry must match config")
    positions = _composite_centerline_samples(config, tube)
    rest = np.linalg.vector_norm(np.diff(positions, axis=0), axis=1).astype(np.float64)
    nodal = nodal_voronoi_lengths(rest)
    area = math.pi * config.diameter**2 / 4.0
    second_moment = math.pi * config.diameter**4 / 64.0
    properties = RodProperties(
        total_mass=float(config.density * area * np.sum(rest)),
        axial_stiffness=float(config.young_modulus * area),
        bending_stiffness=float(config.young_modulus * second_moment),
    )
    masses = config.density * area * nodal
    return RodInitialization(
        positions=positions,
        velocities=np.zeros_like(positions),
        rest_lengths=rest,
        reference_curvature=np.zeros(config.num_nodes - 2, dtype=np.float64),
        nodal_lengths=nodal,
        masses=masses.astype(np.float64, copy=False),
        integrals=np.zeros(config.num_nodes, dtype=np.float64),
        previous_errors=np.zeros(config.num_nodes, dtype=np.float64),
        properties=properties,
    )


def proximal_prescribed_state(
    initial_positions: np.ndarray,
    insertion: float,
    amplitude: float | None = None,
) -> PrescribedState:
    """Prescribe only nodes 0 and 1, translated along the infinite feeder."""
    initial = np.asarray(initial_positions, dtype=np.float64)
    if initial.ndim != 2 or initial.shape[1] != 2 or len(initial) < 4 or not np.all(np.isfinite(initial)):
        raise ValueError("initial_positions must contain at least four finite 2-D nodes")
    if not math.isfinite(float(insertion)) or insertion < 0.0:
        raise ValueError("insertion must be finite and non-negative")
    feeder_height = float(initial[0, 1]) if amplitude is None else float(amplitude)
    if not math.isfinite(feeder_height):
        raise ValueError("amplitude must be finite")
    positions = initial.copy()
    positions[:2, 0] += insertion
    positions[:2, 1] = feeder_height
    mask = np.zeros_like(positions, dtype=bool)
    mask[:2, :] = True
    return PrescribedState(mask=mask, positions=positions)


def _require_cpu_float64(value: torch.Tensor, name: str, shape: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float64 or value.device.type != "cpu":
        raise TypeError(f"{name} must be a CPU torch.float64 tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")


def freeze_contact_frames(positions: torch.Tensor, tube: SineTube2D) -> SineContactFrame:
    """Project an accepted state once and freeze its separable contact frame."""
    if not isinstance(positions, torch.Tensor) or positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must be an N-by-2 tensor")
    _require_cpu_float64(positions, "positions", (len(positions), 2))
    if not bool(torch.isfinite(positions).all()):
        raise ValueError("positions must be finite")
    projection = tube.project(positions.detach().numpy())

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(value), dtype=torch.float64).clone()

    return SineContactFrame(
        q=tensor(projection.q),
        center=tensor(projection.point),
        tangent=tensor(projection.tangent),
        normal=tensor(projection.normal),
        region=(positions[:, 0] >= 0.0).detach().clone(),
    )


build_contact_frames = freeze_contact_frames


def _validate_friction_frame(frame: SineContactFrame, count: int) -> None:
    q, _, _, _, _ = _frame_data(frame)
    if len(q) != count:
        raise ValueError(f"frame node count {len(q)} does not match expected count {count}")


def _require_cpu_bool(value: torch.Tensor, name: str, count: int) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bool
        or value.device.type != "cpu"
        or tuple(value.shape) != (count,)
    ):
        raise TypeError(f"{name} must be a length-N CPU bool tensor")


def initialize_friction_history(
    prebalanced_positions: torch.Tensor,
    frame: SineContactFrame,
    current_active: torch.Tensor,
) -> FrictionHistory:
    """Start zero-slip history after the frictionless prebalance is accepted."""
    if not isinstance(prebalanced_positions, torch.Tensor) or prebalanced_positions.ndim != 2:
        raise ValueError("prebalanced_positions must be an N-by-2 tensor")
    count = len(prebalanced_positions)
    _require_cpu_float64(prebalanced_positions, "prebalanced_positions", (count, 2))
    _validate_friction_frame(frame, count)
    _require_cpu_bool(current_active, "current_active", count)
    if not bool(torch.isfinite(prebalanced_positions).all()):
        raise ValueError("prebalanced_positions must be finite")
    frame_q, _, _, _, frame_region = _frame_data(frame)
    vessel_active = current_active & frame_region
    return FrictionHistory(
        elastic_slip=torch.zeros(count, dtype=torch.float64),
        anchors=prebalanced_positions,
        q=frame_q,
        contact_active=vessel_active,
    )


def begin_friction_increment(
    committed: FrictionHistory,
    start_positions: torch.Tensor,
) -> FrictionIncrementContext:
    """Freeze committed history and the accepted increment-start anchors."""
    if not isinstance(committed, FrictionHistory):
        raise TypeError("committed must be a FrictionHistory")
    elastic_slip, _, committed_q, contact_active = _friction_state_data(committed)
    count = len(elastic_slip)
    _require_cpu_float64(start_positions, "start_positions", (count, 2))
    if not bool(torch.isfinite(start_positions).all()):
        raise ValueError("start_positions must be finite")
    return FrictionIncrementContext(
        elastic_slip=elastic_slip,
        anchors=start_positions,
        q=committed_q,
        contact_active=contact_active,
    )


def evaluate_friction(
    positions: torch.Tensor,
    frame: SineContactFrame,
    vessel_normal_force: torch.Tensor,
    current_active: torch.Tensor,
    nodal_lengths: torch.Tensor,
    context: FrictionIncrementContext,
    config: SineInsertConfig,
    enabled: bool = True,
) -> FrictionResult:
    """Evaluate frozen-frame Coulomb friction without changing committed state.

    The increment anchor and committed slip come only from ``context``; neither
    Newton trials nor line-search probes reset them.  A changing frozen geometry
    frame may change the projected trial displacement, but cannot mutate that
    context.  Raw ``traction * delta_u`` is not required to be non-positive on
    reversal because the tangential spring can return stored energy.  The
    discrete thermodynamic check is instead contact work plus the change in
    ``0.5 * k_t * xi**2``: endpoint stick evaluation contributes its expected
    non-positive discretization term, while slip leaves non-negative plastic
    dissipation after the elastic-energy change is accounted for.
    """
    if not isinstance(context, FrictionIncrementContext):
        raise TypeError("context must be a FrictionIncrementContext")
    if not isinstance(config, SineInsertConfig):
        raise TypeError("config must be a SineInsertConfig")
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be bool")
    if not isinstance(positions, torch.Tensor) or positions.ndim != 2:
        raise ValueError("positions must be an N-by-2 tensor")
    count = len(positions)
    _require_cpu_float64(positions, "positions", (count, 2))
    _require_cpu_float64(vessel_normal_force, "vessel_normal_force", (count, 2))
    _require_cpu_float64(nodal_lengths, "nodal_lengths", (count,))
    _require_cpu_bool(current_active, "current_active", count)
    _validate_friction_frame(frame, count)
    frame_q, _, frame_tangent, _, frame_region = _frame_data(frame)
    context_slip, context_anchors, _, context_active = _friction_state_data(context)
    if len(context_slip) != count:
        raise ValueError("context node count must match positions")
    if not bool(torch.isfinite(positions).all()) or not bool(torch.isfinite(vessel_normal_force).all()):
        raise ValueError("friction inputs must be finite")
    if not bool(torch.isfinite(nodal_lengths).all()) or bool(torch.any(nodal_lengths <= 0.0)):
        raise ValueError("nodal_lengths must be finite and positive")

    vessel_active = current_active & frame_region
    base_slip = torch.where(
        context_active,
        context_slip,
        torch.zeros_like(context_slip),
    )
    delta_u = torch.sum((positions - context_anchors) * frame_tangent, dim=1)
    trial_slip = base_slip + delta_u
    stiffness = config.friction_kt_line * nodal_lengths
    uncapped_traction = -stiffness * trial_slip
    coulomb_limit = config.friction_mu * torch.linalg.vector_norm(vessel_normal_force, dim=1)

    law_active = vessel_active & (config.friction_mu > 0.0) & enabled
    stick = law_active & (torch.abs(uncapped_traction) <= coulomb_limit)
    slip = law_active & ~stick
    capped_traction = -coulomb_limit * torch.sign(trial_slip)
    tangent_traction = torch.where(stick, uncapped_traction, capped_traction)
    tangent_traction = torch.where(law_active, tangent_traction, torch.zeros_like(tangent_traction))
    candidate_slip = torch.where(
        law_active,
        -tangent_traction / stiffness,
        torch.zeros_like(tangent_traction),
    )
    friction_force = tangent_traction[:, None] * frame_tangent

    return FrictionResult(
        friction_force=friction_force,
        tangent_traction=tangent_traction,
        trial_slip=trial_slip,
        candidate_slip=candidate_slip,
        stick=stick,
        slip=slip,
        current_active=vessel_active,
        coulomb_limit=coulomb_limit,
        q=frame_q,
        tangent=frame_tangent,
    )


def commit_friction_increment(
    context: FrictionIncrementContext,
    settled_positions: torch.Tensor,
    frame: SineContactFrame,
    current_active: torch.Tensor,
    candidate_slip: torch.Tensor,
) -> FrictionHistory:
    """Explicitly commit a settled increment; inactive nodes lose slip history."""
    if not isinstance(context, FrictionIncrementContext):
        raise TypeError("context must be a FrictionIncrementContext")
    context_slip, _, _, _ = _friction_state_data(context)
    count = len(context_slip)
    _require_cpu_float64(settled_positions, "settled_positions", (count, 2))
    _require_cpu_float64(candidate_slip, "candidate_slip", (count,))
    _require_cpu_bool(current_active, "current_active", count)
    _validate_friction_frame(frame, count)
    if not bool(torch.isfinite(settled_positions).all()) or not bool(torch.isfinite(candidate_slip).all()):
        raise ValueError("settled friction state must be finite")
    frame_q, _, _, _, frame_region = _frame_data(frame)
    vessel_active = current_active & frame_region
    committed_slip = torch.where(vessel_active, candidate_slip, torch.zeros_like(candidate_slip))
    return FrictionHistory(
        elastic_slip=committed_slip,
        anchors=settled_positions,
        q=frame_q,
        contact_active=vessel_active,
    )


def contact_quantities(
    positions: torch.Tensor,
    old_integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    nodal_lengths: torch.Tensor,
    frame: SineContactFrame,
    config: SineInsertConfig,
    dt: float,
) -> SineContactResult:
    """Evaluate frozen-frame vessel PI contact plus straight-feeder P contact."""
    if not isinstance(positions, torch.Tensor) or positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must be an N-by-2 tensor")
    count = len(positions)
    _require_cpu_float64(positions, "positions", (count, 2))
    _require_cpu_float64(old_integrals, "old_integrals", (count,))
    _require_cpu_float64(previous_errors, "previous_errors", (count,))
    _require_cpu_float64(nodal_lengths, "nodal_lengths", (count,))
    _validate_friction_frame(frame, count)
    if not math.isfinite(float(dt)) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not bool(torch.isfinite(positions).all()) or not bool(torch.isfinite(old_integrals).all()):
        raise ValueError("contact state must be finite")
    if not bool(torch.isfinite(previous_errors).all()) or not bool(torch.isfinite(nodal_lengths).all()):
        raise ValueError("contact state must be finite")
    if bool(torch.any(nodal_lengths <= 0.0)):
        raise ValueError("nodal_lengths must be positive")

    frame_q, frame_center, frame_tangent, frame_normal, vessel_region = _frame_data(frame)
    signed = torch.sum((positions - frame_center) * frame_normal, dim=1)
    clearance = config.tube_half_width - torch.abs(signed)
    z = torch.abs(signed) - (config.tube_half_width - config.safe_distance)
    beta = 1.0 / config.contact_smooth_length
    raw_error = F.softplus(z, beta=beta)
    raw_weight = torch.sigmoid(beta * z)
    error = torch.where(vessel_region, raw_error, torch.zeros_like(raw_error))
    weight = torch.where(vessel_region, raw_weight, torch.zeros_like(raw_weight))

    leaked = old_integrals * math.exp(-config.integral_leak_rate * dt)
    candidate = weight * (old_integrals + error * dt) + (1.0 - weight) * leaked
    candidate = torch.clamp(candidate, 0.0, config.max_integral_state)
    kp = config.contact_kp_line * nodal_lengths
    ki = config.contact_ki_line * nodal_lengths
    cap = config.contact_force_cap_line * nodal_lengths
    unsaturated = kp * error + ki * candidate
    blocked = (ki > 0.0) & (unsaturated >= cap) & (candidate > old_integrals)
    decreasing = error < previous_errors
    blocked |= (ki > 0.0) & decreasing & (ki * old_integrals > kp * error) & (candidate > old_integrals)
    new_integrals = torch.where(blocked, old_integrals, candidate)
    magnitude = torch.clamp(kp * error + ki * new_integrals, min=0.0)
    magnitude = torch.minimum(magnitude, cap)
    direction = -signed[:, None] * frame_normal / torch.sqrt(
        signed.square()[:, None] + config.direction_epsilon**2
    )
    vessel_force = magnitude[:, None] * direction
    vessel_force = torch.where(vessel_region[:, None], vessel_force, torch.zeros_like(vessel_force))

    feeder_region = ~vessel_region
    feeder_signed = positions[:, 1] - config.amplitude
    feeder_z = torch.abs(feeder_signed) - (config.feeder_half_width - config.feeder_contact_layer)
    feeder_error = F.softplus(feeder_z, beta=beta)
    feeder_magnitude = config.feeder_kp_line * nodal_lengths * feeder_error
    feeder_y = -feeder_magnitude * feeder_signed / torch.sqrt(
        feeder_signed.square() + config.direction_epsilon**2
    )
    feeder_y = torch.where(feeder_region, feeder_y, torch.zeros_like(feeder_y))
    feeder_force = torch.stack((torch.zeros_like(feeder_y), feeder_y), dim=1)

    return SineContactResult(
        vessel_normal_force=vessel_force,
        feeder_force=feeder_force,
        total_force=vessel_force + feeder_force,
        new_integrals=new_integrals,
        active=vessel_region & (z > 0.0),
        weight=weight,
        error=error,
        clearance=clearance,
        q=frame_q.clone(),
        tangent=frame_tangent.clone(),
        normal=frame_normal.clone(),
        region=vessel_region.clone(),
        signed_distance=signed,
    )


evaluate_contact = contact_quantities


@dataclass(frozen=True)
class ImplicitStepResult:
    positions: torch.Tensor
    velocities: torch.Tensor
    integrals: torch.Tensor
    previous_errors: torch.Tensor
    normal_force: torch.Tensor
    feeder_force: torch.Tensor
    friction_force: torch.Tensor
    active: torch.Tensor
    max_contact_error: float
    frame: SineContactFrame
    friction_result: FrictionResult
    info: SineNewtonInfo
    dynamic_force: torch.Tensor
    prescribed_mask: torch.Tensor


@dataclass(frozen=True)
class EquilibriumStatus:
    settled: bool
    max_free_speed: float
    residual_norm: float
    max_contact_error: float
    confirm_count: int


@dataclass(frozen=True)
class SimulationState:
    positions: torch.Tensor
    velocities: torch.Tensor
    integrals: torch.Tensor
    previous_errors: torch.Tensor
    friction: FrictionHistory
    frame: SineContactFrame
    time: float
    insertion: float


@dataclass(frozen=True)
class HistoryRecord:
    time: float
    insertion: float
    load_step: int
    phase: str
    settled: bool
    positions: torch.Tensor
    velocities: torch.Tensor
    normal_force: torch.Tensor
    friction_force: torch.Tensor
    feeder_force: torch.Tensor
    integrals: torch.Tensor
    xi: torch.Tensor
    active: torch.Tensor
    contact_error: torch.Tensor
    max_contact_error: float
    newton: NewtonInfo
    tip_position: torch.Tensor
    tip_s: float
    elastic_stretch: float
    elastic_bend: float
    elastic_total: float
    reaction: ReactionDiagnostics


@dataclass(frozen=True)
class ReactionDiagnostics:
    """Global guidewire loads using a single actuator-on-wire sign convention."""

    actuator_force_on_wire: torch.Tensor
    proximal_reaction: torch.Tensor
    dynamic_actuator_force_on_wire: torch.Tensor
    insertion_force: float
    normal_resultant: torch.Tensor
    friction_resultant: torch.Tensor
    feeder_resultant: torch.Tensor
    wall_resultant: torch.Tensor
    dynamic_proximal_remainder: torch.Tensor
    global_dynamic_resultant: torch.Tensor
    internal_elastic_resultant: torch.Tensor
    static_imbalance: torch.Tensor
    static_balance_abs: float
    static_balance_rel: float
    full_balance: torch.Tensor
    full_balance_abs: float
    full_balance_rel: float

    def __post_init__(self) -> None:
        _clone_tensor_fields(self)


def compute_reaction_diagnostics(
    elastic_gradient: torch.Tensor,
    dynamic_force: torch.Tensor,
    normal_force: torch.Tensor,
    friction_force: torch.Tensor,
    feeder_force: torch.Tensor,
    prescribed_mask: torch.Tensor,
) -> ReactionDiagnostics:
    """Resolve actuator reactions and static/full global force balances.

    ``actuator_force_on_wire`` is the settled static force applied by the
    actuator to the guidewire.  ``proximal_reaction`` is the equal-and-opposite
    guidewire force on the actuator.  The dynamic-inclusive value has its own
    explicit name and is used only by the full balance diagnostic.  Wall
    resultants are never folded into the reported insertion-force scalar.
    """
    if not isinstance(elastic_gradient, torch.Tensor) or elastic_gradient.ndim != 2:
        raise ValueError("elastic_gradient must be an N-by-2 tensor")
    count = len(elastic_gradient)
    for value, name in (
        (elastic_gradient, "elastic_gradient"),
        (dynamic_force, "dynamic_force"),
        (normal_force, "normal_force"),
        (friction_force, "friction_force"),
        (feeder_force, "feeder_force"),
    ):
        _require_cpu_float64(value, name, (count, 2))
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
    if (
        not isinstance(prescribed_mask, torch.Tensor)
        or prescribed_mask.dtype != torch.bool
        or prescribed_mask.device.type != "cpu"
        or tuple(prescribed_mask.shape) != (count, 2)
    ):
        raise TypeError("prescribed_mask must be an N-by-2 CPU bool tensor")
    if bool(torch.any(prescribed_mask[:, 0] != prescribed_mask[:, 1])):
        raise ValueError("prescribed nodes must constrain both coordinates")

    prescribed_nodes = prescribed_mask[:, 0]
    wall_force = normal_force + friction_force + feeder_force
    static_nodal_actuator = elastic_gradient - wall_force
    static_actuator = static_nodal_actuator[prescribed_nodes].sum(dim=0)
    dynamic_remainder = dynamic_force[prescribed_nodes].sum(dim=0)
    dynamic_actuator = static_actuator + dynamic_remainder
    normal_resultant = normal_force.sum(dim=0)
    friction_resultant = friction_force.sum(dim=0)
    feeder_resultant = feeder_force.sum(dim=0)
    wall_resultant = normal_resultant + friction_resultant + feeder_resultant
    global_dynamic = dynamic_force.sum(dim=0)
    internal_elastic = elastic_gradient.sum(dim=0)
    static_imbalance = static_actuator + wall_resultant
    full_balance = dynamic_actuator + wall_resultant - global_dynamic
    denominator = max(float(torch.linalg.vector_norm(static_actuator).item()), 1.0e-12)
    static_abs = float(torch.linalg.vector_norm(static_imbalance).item())
    full_abs = float(torch.linalg.vector_norm(full_balance).item())
    return ReactionDiagnostics(
        actuator_force_on_wire=static_actuator,
        proximal_reaction=-static_actuator,
        dynamic_actuator_force_on_wire=dynamic_actuator,
        insertion_force=float(static_actuator[0].item()),
        normal_resultant=normal_resultant,
        friction_resultant=friction_resultant,
        feeder_resultant=feeder_resultant,
        wall_resultant=wall_resultant,
        dynamic_proximal_remainder=dynamic_remainder,
        global_dynamic_resultant=global_dynamic,
        internal_elastic_resultant=internal_elastic,
        static_imbalance=static_imbalance,
        static_balance_abs=static_abs,
        static_balance_rel=static_abs / denominator,
        full_balance=full_balance,
        full_balance_abs=full_abs,
        full_balance_rel=full_abs / denominator,
    )


@dataclass(frozen=True)
class SimulationResult:
    config: SineInsertConfig
    success: bool
    abort_reason: str | None
    completed_load_steps: int
    history: tuple[HistoryRecord, ...]
    final_state: SimulationState
    prebalanced_state: SimulationState
    prebalance_history_length: int

    @property
    def settled_records(self) -> tuple[HistoryRecord, ...]:
        return tuple(record for record in self.history if record.settled)


def smooth_ramp_fraction(step: int, steps: int) -> float:
    """Return the exact-endpoint cubic smoothstep fraction for a ramp."""
    if not isinstance(step, (int, np.integer)) or isinstance(step, (bool, np.bool_)):
        raise ValueError("step must be an integer")
    if not isinstance(steps, (int, np.integer)) or isinstance(steps, (bool, np.bool_)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if step < 0 or step > steps:
        raise ValueError("step must lie between zero and steps")
    if step == 0:
        return 0.0
    if step == steps:
        return 1.0
    value = float(step) / float(steps)
    return value * value * (3.0 - 2.0 * value)


def proximal_ramp_state(
    initial_positions: np.ndarray,
    start_insertion: float,
    target_insertion: float,
    step: int,
    steps: int,
    amplitude: float | None = None,
) -> PrescribedState:
    """Prescribe the proximal two nodes on a smooth insertion ramp."""
    if not math.isfinite(float(start_insertion)) or start_insertion < 0.0:
        raise ValueError("start_insertion must be finite and non-negative")
    if not math.isfinite(float(target_insertion)) or target_insertion < start_insertion:
        raise ValueError("target_insertion must be finite and not smaller than start_insertion")
    fraction = smooth_ramp_fraction(step, steps)
    insertion = start_insertion + fraction * (target_insertion - start_insertion)
    if step == 0:
        insertion = float(start_insertion)
    elif step == steps:
        insertion = float(target_insertion)
    return proximal_prescribed_state(initial_positions, insertion, amplitude)


def _frame_from_projection(positions: torch.Tensor, projection: object) -> SineContactFrame:
    def tensor(value: object) -> torch.Tensor:
        return torch.as_tensor(np.asarray(value), dtype=torch.float64).clone()

    return SineContactFrame(
        q=tensor(projection.q),
        center=tensor(projection.point),
        tangent=tensor(projection.tangent),
        normal=tensor(projection.normal),
        region=(positions[:, 0] >= 0.0).detach().clone(),
    )


def _tracked_frame(
    positions: torch.Tensor,
    tube: SineTube2D,
    previous: SineContactFrame | None,
    config: SineInsertConfig,
) -> SineContactFrame:
    if previous is None:
        return freeze_contact_frames(positions, tube)
    previous_q, _, _, _, _ = _frame_data(previous)
    projection = tube.project_tracked(
        positions.detach().numpy(),
        previous_q.detach().numpy(),
        trust_radius=config.projection_trust_radius,
    )
    return _frame_from_projection(positions, projection)


def _zero_friction_context(
    positions: torch.Tensor,
    frame: SineContactFrame,
) -> FrictionIncrementContext:
    count = len(positions)
    frame_q, _, _, _, _ = _frame_data(frame)
    return FrictionIncrementContext(
        elastic_slip=torch.zeros(count, dtype=torch.float64),
        anchors=positions,
        q=frame_q,
        contact_active=torch.zeros(count, dtype=torch.bool),
    )


def _local_external_force(
    point: torch.Tensor,
    center: torch.Tensor,
    tangent: torch.Tensor,
    normal: torch.Tensor,
    vessel_region: torch.Tensor,
    old_integral: torch.Tensor,
    previous_error: torch.Tensor,
    nodal_length: torch.Tensor,
    anchor: torch.Tensor,
    committed_slip: torch.Tensor,
    committed_active: torch.Tensor,
    config: SineInsertConfig,
    dt: float,
    friction_enabled: bool,
) -> torch.Tensor:
    """Differentiable frozen-frame force for one independent contact node."""
    signed = torch.dot(point - center, normal)
    z = torch.abs(signed) - (config.tube_half_width - config.safe_distance)
    beta = 1.0 / config.contact_smooth_length
    raw_error = F.softplus(z, beta=beta)
    weight = torch.sigmoid(beta * z)
    error = torch.where(vessel_region, raw_error, torch.zeros_like(raw_error))
    weight = torch.where(vessel_region, weight, torch.zeros_like(weight))
    leaked = old_integral * math.exp(-config.integral_leak_rate * dt)
    candidate = weight * (old_integral + error * dt) + (1.0 - weight) * leaked
    candidate = torch.clamp(candidate, 0.0, config.max_integral_state)
    kp = config.contact_kp_line * nodal_length
    ki = config.contact_ki_line * nodal_length
    cap = config.contact_force_cap_line * nodal_length
    unsaturated = kp * error + ki * candidate
    blocked = (ki > 0.0) & (unsaturated >= cap) & (candidate > old_integral)
    blocked |= (
        (ki > 0.0)
        & (error < previous_error)
        & (ki * old_integral > kp * error)
        & (candidate > old_integral)
    )
    integral = torch.where(blocked, old_integral, candidate)
    magnitude = torch.minimum(torch.clamp(kp * error + ki * integral, min=0.0), cap)
    direction = -signed * normal / torch.sqrt(signed.square() + config.direction_epsilon**2)
    normal_force = torch.where(vessel_region, magnitude * direction, torch.zeros_like(point))

    feeder_signed = point[1] - config.amplitude
    feeder_z = torch.abs(feeder_signed) - (
        config.feeder_half_width - config.feeder_contact_layer
    )
    feeder_error = F.softplus(feeder_z, beta=beta)
    feeder_magnitude = config.feeder_kp_line * nodal_length * feeder_error
    feeder_y = -feeder_magnitude * feeder_signed / torch.sqrt(
        feeder_signed.square() + config.direction_epsilon**2
    )
    feeder_force = torch.stack((torch.zeros_like(feeder_y), feeder_y))
    feeder_force = torch.where(vessel_region, torch.zeros_like(feeder_force), feeder_force)

    active = vessel_region & (z > 0.0)
    base_slip = torch.where(committed_active, committed_slip, torch.zeros_like(committed_slip))
    trial_slip = base_slip + torch.dot(point - anchor, tangent)
    stiffness = config.friction_kt_line * nodal_length
    uncapped = -stiffness * trial_slip
    limit = config.friction_mu * torch.linalg.vector_norm(normal_force)
    law_active = active & friction_enabled & (config.friction_mu > 0.0)
    traction = torch.where(
        torch.abs(uncapped) <= limit,
        uncapped,
        -limit * torch.sign(trial_slip),
    )
    traction = torch.where(law_active, traction, torch.zeros_like(traction))
    return normal_force + feeder_force + traction * tangent


def _external_force_and_blocks(
    positions: torch.Tensor,
    frame: SineContactFrame,
    old_integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    nodal_lengths: torch.Tensor,
    context: FrictionIncrementContext,
    config: SineInsertConfig,
    dt: float,
    friction_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    def force_aux(
        point: torch.Tensor,
        center: torch.Tensor,
        tangent: torch.Tensor,
        normal: torch.Tensor,
        region: torch.Tensor,
        old_integral: torch.Tensor,
        previous_error: torch.Tensor,
        nodal_length: torch.Tensor,
        anchor: torch.Tensor,
        committed_slip: torch.Tensor,
        committed_active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = _local_external_force(
            point, center, tangent, normal, region, old_integral,
            previous_error, nodal_length, anchor, committed_slip,
            committed_active, config, dt, friction_enabled,
        )
        return value, value

    _, frame_center, frame_tangent, frame_normal, frame_region = _frame_data(frame)
    context_slip, context_anchors, _, context_active = _friction_state_data(context)
    blocks, forces = torch.func.vmap(
        torch.func.jacrev(force_aux, argnums=0, has_aux=True)
    )(
        positions, frame_center, frame_tangent, frame_normal, frame_region,
        old_integrals, previous_errors, nodal_lengths, context_anchors,
        context_slip, context_active,
    )
    return forces, blocks


def _external_quantities(
    positions: torch.Tensor,
    frame: SineContactFrame,
    old_integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    nodal_lengths: torch.Tensor,
    context: FrictionIncrementContext,
    config: SineInsertConfig,
    dt: float,
    friction_enabled: bool,
) -> tuple[SineContactResult, FrictionResult]:
    contact = contact_quantities(
        positions, old_integrals, previous_errors, nodal_lengths, frame, config, dt
    )
    friction = evaluate_friction(
        positions, frame, contact.vessel_normal_force, contact.active,
        nodal_lengths, context, config, enabled=friction_enabled,
    )
    return contact, friction


def _unpack_free(
    free: torch.Tensor,
    prescribed: torch.Tensor,
    free_start: int,
    free_end: int,
) -> torch.Tensor:
    positions = prescribed.clone()
    positions[free_start:free_end] = free.reshape(-1, 2)
    return positions


def _residual(
    free: torch.Tensor,
    old_positions: torch.Tensor,
    old_velocities: torch.Tensor,
    rest_lengths: torch.Tensor,
    reference_curvature: torch.Tensor,
    masses: torch.Tensor,
    old_integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    nodal_lengths: torch.Tensor,
    prescribed: torch.Tensor,
    frame: SineContactFrame,
    context: FrictionIncrementContext,
    config: SineInsertConfig,
    dt: float,
    damping: float,
    friction_enabled: bool,
    with_jacobian: bool,
) -> tuple[torch.Tensor, object | None, torch.Tensor]:
    plan = get_block_plan(len(old_positions), False)
    positions = _unpack_free(free, prescribed, plan.free_start, plan.free_end)
    if with_jacobian:
        gradient, system = assemble_elastic_banded(
            positions, rest_lengths, reference_curvature, plan,
            config.young_modulus * math.pi * config.diameter**2 / 4.0,
            config.young_modulus * math.pi * config.diameter**4 / 64.0,
        )
        external, blocks = _external_force_and_blocks(
            positions, frame, old_integrals, previous_errors, nodal_lengths,
            context, config, dt, friction_enabled=friction_enabled,
        )
    else:
        differentiable = positions.detach().clone().requires_grad_(True)
        _, _, energy = elastic_energy(
            differentiable, rest_lengths,
            config.young_modulus * math.pi * config.diameter**2 / 4.0,
            config.young_modulus * math.pi * config.diameter**4 / 64.0,
            reference_curvature,
        )
        gradient = torch.autograd.grad(energy, differentiable)[0][plan.free_start:plan.free_end].reshape(-1)
        contact, friction = _external_quantities(
            positions, frame, old_integrals, previous_errors, nodal_lengths,
            context, config, dt, friction_enabled,
        )
        external = contact.total_force + friction.friction_force
        system = None
        blocks = None
    old_free = old_positions[plan.free_start:plan.free_end].reshape(-1)
    old_velocity = old_velocities[plan.free_start:plan.free_end].reshape(-1)
    mass = masses[plan.free_start:plan.free_end, None].expand(-1, 2).reshape(-1)
    residual = (
        mass * (free - old_free - dt * old_velocity) / dt**2
        + gradient
        + damping * (free - old_free) / dt
        - external[plan.free_start:plan.free_end].reshape(-1)
    )
    if with_jacobian:
        diagonal = (mass / dt**2 + damping / dt).reshape(-1, 2)
        system.diagonal.diagonal(dim1=1, dim2=2).add_(diagonal)
        system.diagonal.add_(-blocks[plan.free_start:plan.free_end])
    return residual, system, positions


def implicit_euler_step(
    old_positions: torch.Tensor,
    old_velocities: torch.Tensor,
    rest_lengths: torch.Tensor,
    reference_curvature: torch.Tensor,
    masses: torch.Tensor,
    old_integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    nodal_lengths: torch.Tensor,
    prescribed_mask: torch.Tensor,
    prescribed_positions: torch.Tensor,
    tube: SineTube2D,
    config: SineInsertConfig,
    friction_context: FrictionIncrementContext | None,
    *,
    friction_enabled: bool,
    accepted_frame: SineContactFrame | None = None,
    evaluation_observer=None,
) -> ImplicitStepResult:
    """Advance one proximal-prescribed, distal-free implicit Euler step."""
    count = len(old_positions)
    for value, name, shape in (
        (old_positions, "old_positions", (count, 2)),
        (old_velocities, "old_velocities", (count, 2)),
        (rest_lengths, "rest_lengths", (count - 1,)),
        (reference_curvature, "reference_curvature", (count - 2,)),
        (masses, "masses", (count,)),
        (old_integrals, "old_integrals", (count,)),
        (previous_errors, "previous_errors", (count,)),
        (nodal_lengths, "nodal_lengths", (count,)),
        (prescribed_positions, "prescribed_positions", (count, 2)),
    ):
        _require_cpu_float64(value, name, shape)
    _require_cpu_bool(prescribed_mask.reshape(-1), "prescribed_mask", 2 * count)
    expected_mask = torch.zeros_like(prescribed_mask)
    expected_mask[:2] = True
    if tuple(prescribed_mask.shape) != (count, 2) or not torch.equal(prescribed_mask, expected_mask):
        raise ValueError("only proximal nodes 0 and 1 may be prescribed; the distal tip is free")
    if not isinstance(friction_enabled, bool):
        raise TypeError("friction_enabled must be bool")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (
            old_positions, old_velocities, rest_lengths, reference_curvature,
            masses, old_integrals, previous_errors, nodal_lengths,
            prescribed_positions,
        )
    ):
        raise ValueError("implicit step inputs must be finite")

    plan = get_block_plan(count, False)
    free = (
        old_positions[plan.free_start:plan.free_end]
        + config.dt * old_velocities[plan.free_start:plan.free_end]
    ).reshape(-1).detach()
    tracking_frame = accepted_frame or freeze_contact_frames(old_positions, tube)
    context = friction_context or _zero_friction_context(old_positions, tracking_frame)
    total_mass = float(masses.sum().item())
    ei = config.young_modulus * math.pi * config.diameter**4 / 64.0
    damping = 2.0 * config.damping_ratio * math.sqrt(
        (3.0 * ei / config.wire_length**3) * total_mass
    )
    if config.backend == "cpp":
        if evaluation_observer is not None:
            raise ValueError("evaluation_observer is available only for the Python reference backend")
        import der_cpp_backend

        context_slip, context_anchors, _, context_active = _friction_state_data(context)
        ea = config.young_modulus * math.pi * config.diameter**2 / 4.0
        core = der_cpp_backend.Sine2DCore(
            rest_lengths, masses, ea, ei, damping, config.amplitude,
            config.x_span, config.tube_half_width, config.safe_distance,
            config.feeder_half_width, config.feeder_contact_layer,
            config.feeder_kp_line, config.contact_kp_line,
            config.contact_ki_line, config.contact_force_cap_line,
            config.friction_mu, config.friction_kt_line,
            config.max_integral_state, config.integral_leak_rate,
            config.contact_smooth_length, config.direction_epsilon,
        )
        tracking_q, _, _, _, _ = _frame_data(tracking_frame)
        raw = core.implicit_euler_step(
            old_positions, old_velocities, old_integrals, previous_errors,
            nodal_lengths, prescribed_positions, reference_curvature,
            context_slip, context_anchors, context_active, friction_enabled,
            tracking_q, config.dt, config.max_newton_iters,
            config.residual_tol, config.step_tol,
            config.projection_trust_radius,
            kp_active=config.contact_kp_line,
            ki_active=config.contact_ki_line,
        )

        def tensor(name: str, dtype=torch.float64) -> torch.Tensor:
            return torch.as_tensor(np.asarray(raw[name]), dtype=dtype).clone()

        result_frame = SineContactFrame(
            q=tensor("q"), center=tensor("center"), tangent=tensor("tangent"),
            normal=tensor("normal"), region=tensor("region", torch.bool),
        )
        friction = FrictionResult(
            friction_force=tensor("friction_force"),
            tangent_traction=tensor("traction"), trial_slip=tensor("trial_slip"),
            candidate_slip=tensor("candidate_slip"), stick=tensor("stick", torch.bool),
            slip=tensor("slip", torch.bool), current_active=tensor("active", torch.bool),
            coulomb_limit=tensor("limit"), tangent=result_frame.tangent,
            q=result_frame.q,
        )
        return ImplicitStepResult(
            positions=tensor("positions"), velocities=tensor("velocities"),
            integrals=tensor("integrals"), previous_errors=tensor("previous_errors"),
            normal_force=tensor("normal_force"), feeder_force=tensor("feeder_force"),
            friction_force=tensor("friction_force"), active=tensor("active", torch.bool),
            max_contact_error=float(raw["max_contact_error"]), frame=result_frame,
            friction_result=friction,
            info=SineNewtonInfo(
                bool(raw["converged"]), int(raw["iterations"]),
                float(raw["residual_norm"]), float(raw["step_norm"]),
                int(raw["line_search_failures"]), int(raw["dense_fallback_count"]),
            ),
            dynamic_force=tensor("dynamic_force"),
            prescribed_mask=prescribed_mask.clone(),
        )
    failures = 0
    dense_fallback_count = 0
    residual_norm = math.inf
    step_norm = math.inf
    converged = False
    iteration = 0

    for iteration in range(1, config.max_newton_iters + 1):
        current_positions = _unpack_free(
            free, prescribed_positions, plan.free_start, plan.free_end
        )
        frame = _tracked_frame(current_positions, tube, tracking_frame, config)
        if evaluation_observer is not None:
            evaluation_observer(iteration, "jacobian", frame)
        residual, system, _ = _residual(
            free, old_positions, old_velocities, rest_lengths,
            reference_curvature, masses, old_integrals, previous_errors,
            nodal_lengths, prescribed_positions, frame, context,
            config, config.dt, damping, friction_enabled, True,
        )
        residual_norm = float(torch.linalg.vector_norm(residual).item())
        if residual_norm < config.residual_tol:
            converged = True
            step_norm = 0.0
            tracking_frame = frame
            break
        try:
            delta = system.solve(-residual)
        except RuntimeError:
            dense_fallback_count += 1
            delta = torch.linalg.solve(system.dense(), -residual)
        step_norm = float(torch.linalg.vector_norm(delta).item())
        accepted = False
        for exponent in range(10):
            candidate = free + (0.5**exponent) * delta
            if evaluation_observer is not None:
                evaluation_observer(iteration, "line_search", frame)
            candidate_residual, _, _ = _residual(
                candidate, old_positions, old_velocities, rest_lengths,
                reference_curvature, masses, old_integrals, previous_errors,
                nodal_lengths, prescribed_positions, frame, context,
                config, config.dt, damping, friction_enabled, False,
            )
            candidate_norm = float(torch.linalg.vector_norm(candidate_residual).item())
            if math.isfinite(candidate_norm) and candidate_norm < residual_norm:
                free = candidate.detach()
                residual_norm = candidate_norm
                accepted = True
                break
        if not accepted:
            failures += 1
            forced = free + 0.05 * delta
            if evaluation_observer is not None:
                evaluation_observer(iteration, "line_search", frame)
            forced_residual, _, _ = _residual(
                forced, old_positions, old_velocities, rest_lengths,
                reference_curvature, masses, old_integrals, previous_errors,
                nodal_lengths, prescribed_positions, frame, context,
                config, config.dt, damping, friction_enabled, False,
            )
            forced_norm = float(torch.linalg.vector_norm(forced_residual).item())
            if math.isfinite(forced_norm):
                free = forced.detach()
                residual_norm = forced_norm
            else:
                # Keep the last accepted iterate, and never let an unevaluable
                # forced candidate satisfy the relaxed small-step criterion.
                residual_norm = math.inf
        tracking_frame = frame
        if step_norm < config.step_tol:
            converged = residual_norm < 10.0 * config.residual_tol
            break

    positions = _unpack_free(
        free, prescribed_positions, plan.free_start, plan.free_end
    ).detach()
    frame = _tracked_frame(positions, tube, tracking_frame, config)
    contact, friction = _external_quantities(
        positions, frame, old_integrals, previous_errors, nodal_lengths,
        context, config, config.dt, friction_enabled,
    )
    velocities = ((positions - old_positions) / config.dt).detach()
    velocities[prescribed_mask] = 0.0
    dynamic_force = (
        masses[:, None]
        * (positions - old_positions - config.dt * old_velocities)
        / config.dt**2
        + damping * (positions - old_positions) / config.dt
    ).detach()
    intrusion = geometric_layer_intrusion(positions, frame, config)
    return ImplicitStepResult(
        positions=positions,
        velocities=velocities,
        integrals=contact.new_integrals.detach(),
        previous_errors=contact.error.detach(),
        normal_force=contact.vessel_normal_force.detach(),
        feeder_force=contact.feeder_force.detach(),
        friction_force=friction.friction_force.detach(),
        active=contact.active.detach(),
        max_contact_error=float(intrusion.max().item()),
        frame=frame,
        friction_result=friction,
        info=SineNewtonInfo(
            converged, iteration, residual_norm, step_norm, failures,
            dense_fallback_count,
        ),
        dynamic_force=dynamic_force,
        prescribed_mask=prescribed_mask.clone(),
    )


def geometric_layer_intrusion(
    positions: torch.Tensor,
    frame: SineContactFrame,
    config: SineInsertConfig,
) -> torch.Tensor:
    """Return exact vessel/feeder protection-layer intrusion per node."""
    _, frame_center, _, frame_normal, frame_region = _frame_data(frame)
    signed = torch.sum((positions - frame_center) * frame_normal, dim=1)
    vessel = torch.clamp(
        torch.abs(signed) - (config.tube_half_width - config.safe_distance), min=0.0
    )
    feeder_signed = positions[:, 1] - config.amplitude
    feeder = torch.clamp(
        torch.abs(feeder_signed)
        - (config.feeder_half_width - config.feeder_contact_layer),
        min=0.0,
    )
    return torch.where(frame_region, vessel, feeder)


def check_equilibrium(
    step: ImplicitStepResult,
    prescribed_mask: torch.Tensor,
    config: SineInsertConfig,
    prior_confirm_count: int,
) -> EquilibriumStatus:
    """Check free-node speed, Newton residual, geometry, and confirmation count."""
    free_nodes = torch.any(~prescribed_mask, dim=1)
    speed = float(
        torch.linalg.vector_norm(step.velocities[free_nodes], dim=1).max().item()
    )
    ok = (
        step.info.converged
        and speed < config.settle_speed_tol
        and step.info.residual_norm < config.settle_residual_tol
        and step.max_contact_error < config.contact_tolerance
    )
    count = prior_confirm_count + 1 if ok else 0
    return EquilibriumStatus(
        settled=count >= config.settle_confirm_steps,
        max_free_speed=speed,
        residual_norm=step.info.residual_norm,
        max_contact_error=step.max_contact_error,
        confirm_count=count,
    )


def guard_tip_before_outlet(
    positions: torch.Tensor,
    tube: SineTube2D,
    config: SineInsertConfig,
    frame: SineContactFrame | None = None,
) -> float:
    """Return tip centerline arclength, raising at the configured outlet margin."""
    if frame is None:
        projection = tube.project(positions[-1].detach().numpy())
        q = float(projection.q)
    else:
        frame_q, _, _, _, _ = _frame_data(frame)
        q = float(frame_q[-1].item())
    tip_s = float(tube.arclength(q))
    if tip_s >= tube.vessel_length - config.outlet_margin:
        raise RuntimeError(
            f"tip reached outlet guard: s={tip_s:.9g} m, "
            f"limit={tube.vessel_length - config.outlet_margin:.9g} m"
        )
    return tip_s


def _clone_frame(frame: SineContactFrame) -> SineContactFrame:
    q, center, tangent, normal, region = _frame_data(frame)
    return SineContactFrame(
        q=q, center=center, tangent=tangent, normal=normal, region=region,
    )


def _clone_state(state: SimulationState) -> SimulationState:
    slip, anchors, friction_q, contact_active = _friction_state_data(state.friction)
    return SimulationState(
        positions=state.positions.clone(),
        velocities=state.velocities.clone(),
        integrals=state.integrals.clone(),
        previous_errors=state.previous_errors.clone(),
        friction=FrictionHistory(
            slip, anchors, friction_q, contact_active,
        ),
        frame=_clone_frame(state.frame),
        time=float(state.time),
        insertion=float(state.insertion),
    )


def _record(
    state: SimulationState,
    step: ImplicitStepResult,
    rest_lengths: torch.Tensor,
    reference_curvature: torch.Tensor,
    config: SineInsertConfig,
    tube: SineTube2D,
    load_step: int,
    phase: str,
    settled: bool,
) -> HistoryRecord:
    ea = config.young_modulus * math.pi * config.diameter**2 / 4.0
    ei = config.young_modulus * math.pi * config.diameter**4 / 64.0
    differentiable_positions = state.positions.detach().clone().requires_grad_(True)
    stretch, bend, total = elastic_energy(
        differentiable_positions, rest_lengths, ea, ei, reference_curvature
    )
    elastic_gradient = torch.autograd.grad(total, differentiable_positions)[0].detach()
    reaction = compute_reaction_diagnostics(
        elastic_gradient,
        step.dynamic_force,
        step.normal_force,
        step.friction_force,
        step.feeder_force,
        step.prescribed_mask,
    )
    tip_s = guard_tip_before_outlet(state.positions, tube, config, state.frame)
    friction_slip, _, _, _ = _friction_state_data(state.friction)
    return HistoryRecord(
        time=float(state.time), insertion=float(state.insertion), load_step=load_step,
        phase=phase, settled=settled, positions=state.positions.clone(),
        velocities=state.velocities.clone(), normal_force=step.normal_force.clone(),
        friction_force=step.friction_force.clone(), feeder_force=step.feeder_force.clone(),
        integrals=state.integrals.clone(), xi=friction_slip.clone(),
        active=step.active.clone(), contact_error=state.previous_errors.clone(),
        max_contact_error=step.max_contact_error,
        newton=step.info, tip_position=state.positions[-1].clone(), tip_s=tip_s,
        elastic_stretch=float(stretch.item()), elastic_bend=float(bend.item()),
        elastic_total=float(total.item()), reaction=reaction,
    )


@dataclass(frozen=True)
class _AttemptResult:
    success: bool
    state: SimulationState
    records: tuple[HistoryRecord, ...]
    reason: str | None
    max_contact_error: float


def _attempt_increment(
    start: SimulationState,
    start_insertion: float,
    target_insertion: float,
    initial_positions: np.ndarray,
    rest_lengths: torch.Tensor,
    reference_curvature: torch.Tensor,
    masses: torch.Tensor,
    nodal_lengths: torch.Tensor,
    tube: SineTube2D,
    config: SineInsertConfig,
    load_step: int,
) -> _AttemptResult:
    context = begin_friction_increment(start.friction, start.positions)
    state = _clone_state(start)
    local_records: list[HistoryRecord] = []
    last_step = None
    mask = None
    for ramp_step in range(1, config.ramp_steps + 1):
        prescription = proximal_ramp_state(
            initial_positions, start_insertion, target_insertion,
            ramp_step, config.ramp_steps, config.amplitude,
        )
        mask = torch.as_tensor(prescription.mask, dtype=torch.bool)
        prescribed = torch.as_tensor(prescription.positions, dtype=torch.float64)
        last_step = implicit_euler_step(
            state.positions, state.velocities, rest_lengths, reference_curvature,
            masses, state.integrals, state.previous_errors, nodal_lengths,
            mask, prescribed, tube, config, context, friction_enabled=True,
            accepted_frame=state.frame,
        )
        if not last_step.info.converged:
            return _AttemptResult(False, start, (), "Newton failure during ramp", last_step.max_contact_error)
        insertion = start_insertion + smooth_ramp_fraction(
            ramp_step, config.ramp_steps
        ) * (target_insertion - start_insertion)
        state = SimulationState(
            last_step.positions, last_step.velocities, last_step.integrals,
            last_step.previous_errors, state.friction, last_step.frame,
            state.time + config.dt, insertion,
        )
        guard_tip_before_outlet(state.positions, tube, config, state.frame)
        if ramp_step % config.history_save_interval == 0:
            local_records.append(_record(
                state, last_step, rest_lengths, reference_curvature, config,
                tube, load_step, "ramp", False,
            ))

    stable = 0
    for relax_step in range(1, config.max_settle_steps + 1):
        prescription = proximal_prescribed_state(
            initial_positions, target_insertion, config.amplitude
        )
        mask = torch.as_tensor(prescription.mask, dtype=torch.bool)
        prescribed = torch.as_tensor(prescription.positions, dtype=torch.float64)
        last_step = implicit_euler_step(
            state.positions, state.velocities, rest_lengths, reference_curvature,
            masses, state.integrals, state.previous_errors, nodal_lengths,
            mask, prescribed, tube, config, context, friction_enabled=True,
            accepted_frame=state.frame,
        )
        state = SimulationState(
            last_step.positions, last_step.velocities, last_step.integrals,
            last_step.previous_errors, state.friction, last_step.frame,
            state.time + config.dt, target_insertion,
        )
        guard_tip_before_outlet(state.positions, tube, config, state.frame)
        equilibrium = check_equilibrium(last_step, mask, config, stable)
        stable = equilibrium.confirm_count
        if relax_step % config.history_save_interval == 0 and not equilibrium.settled:
            local_records.append(_record(
                state, last_step, rest_lengths, reference_curvature, config,
                tube, load_step, "relax", False,
            ))
        if equilibrium.settled:
            committed = commit_friction_increment(
                context, state.positions, last_step.frame, last_step.active,
                last_step.friction_result.candidate_slip,
            )
            state = SimulationState(
                state.positions, state.velocities, state.integrals,
                state.previous_errors, committed, state.frame, state.time,
                target_insertion,
            )
            local_records.append(_record(
                state, last_step, rest_lengths, reference_curvature, config,
                tube, load_step, "settled", True,
            ))
            return _AttemptResult(True, state, tuple(local_records), None, last_step.max_contact_error)
    reason = f"load step {load_step} did not settle in {config.max_settle_steps} steps"
    error = last_step.max_contact_error if last_step is not None else math.inf
    return _AttemptResult(False, start, (), reason, error)


def _adaptive_configs(config: SineInsertConfig):
    """Yield bounded Kp growth then Ki-reduction retry configurations."""
    yield config
    kp = config.contact_kp_line
    for _ in range(config.max_kp_retries):
        kp = min(kp * config.kp_grow_factor, config.contact_kp_line * config.kp_max_factor)
        yield replace(config, contact_kp_line=kp)
        if kp >= config.contact_kp_line * config.kp_max_factor:
            break
    ki = config.contact_ki_line
    for _ in range(config.max_ki_retries):
        ki *= config.ki_reduction_factor
        if ki <= config.ki_zero_threshold:
            ki = 0.0
        yield replace(config, contact_kp_line=kp, contact_ki_line=ki)
        if ki == 0.0:
            break


def run_simulation(config: SineInsertConfig) -> SimulationResult:
    """Run an in-memory quasi-static insertion with transactional load steps."""
    if not isinstance(config, SineInsertConfig):
        raise TypeError("config must be a SineInsertConfig")
    tube = SineTube2D(config.amplitude, config.x_span, config.tube_half_width)
    initial = initialize_rod(config, tube)
    positions = torch.as_tensor(initial.positions, dtype=torch.float64)
    velocities = torch.as_tensor(initial.velocities, dtype=torch.float64)
    rest = torch.as_tensor(initial.rest_lengths, dtype=torch.float64)
    reference = torch.as_tensor(initial.reference_curvature, dtype=torch.float64)
    masses = torch.as_tensor(initial.masses, dtype=torch.float64)
    nodal = torch.as_tensor(initial.nodal_lengths, dtype=torch.float64)
    integrals = torch.as_tensor(initial.integrals, dtype=torch.float64)
    errors = torch.as_tensor(initial.previous_errors, dtype=torch.float64)
    frame = freeze_contact_frames(positions, tube)
    initial_contact = contact_quantities(
        positions, integrals, errors, nodal, frame, config, config.dt
    )
    placeholder_friction = initialize_friction_history(
        positions, frame, torch.zeros(config.num_nodes, dtype=torch.bool)
    )
    state = SimulationState(
        positions, velocities, integrals, errors, placeholder_friction,
        frame, 0.0, 0.0,
    )
    history: list[HistoryRecord] = []
    prebalance_start = _clone_state(state)

    if config.prebalance:
        stable = 0
        prebalanced = False
        for relax_step in range(1, config.prebalance_max_steps + 1):
            prescription = proximal_prescribed_state(
                initial.positions, 0.0, config.amplitude
            )
            mask = torch.as_tensor(prescription.mask, dtype=torch.bool)
            prescribed = torch.as_tensor(prescription.positions, dtype=torch.float64)
            step = implicit_euler_step(
                state.positions, state.velocities, rest, reference, masses,
                state.integrals, state.previous_errors, nodal, mask, prescribed,
                tube, config, None, friction_enabled=False,
                accepted_frame=state.frame,
            )
            state = SimulationState(
                step.positions, step.velocities, step.integrals,
                step.previous_errors, state.friction, step.frame,
                state.time + config.dt, 0.0,
            )
            equilibrium = check_equilibrium(step, mask, config, stable)
            stable = equilibrium.confirm_count
            if relax_step % config.history_save_interval == 0 and not equilibrium.settled:
                history.append(_record(
                    state, step, rest, reference, config, tube, -1,
                    "prebalance", False,
                ))
            if equilibrium.settled:
                friction = initialize_friction_history(
                    state.positions, step.frame, step.active
                )
                state = SimulationState(
                    state.positions, state.velocities, state.integrals,
                    state.previous_errors, friction, state.frame, state.time, 0.0,
                )
                history.append(_record(
                    state, step, rest, reference, config, tube, -1,
                    "prebalance", True,
                ))
                prebalanced = True
                break
        if not prebalanced:
            snapshot = _clone_state(prebalance_start)
            return SimulationResult(
                config, False,
                f"prebalance did not settle in {config.prebalance_max_steps} steps",
                0, (), snapshot, snapshot, 0,
            )
    else:
        friction = initialize_friction_history(
            state.positions, state.frame, initial_contact.active
        )
        state = SimulationState(
            state.positions, state.velocities, state.integrals,
            state.previous_errors, friction, state.frame, state.time, 0.0,
        )

    prebalanced_state = _clone_state(state)
    prebalance_history_length = len(history)
    completed = 0
    abort_reason = None
    for load_step in range(1, config.num_load_steps + 1):
        increment_start = _clone_state(state)
        start_insertion = config.total_insertion * (load_step - 1) / config.num_load_steps
        target_insertion = config.total_insertion * load_step / config.num_load_steps
        accepted = None
        final_retry_config = config
        for retry_config in _adaptive_configs(config):
            final_retry_config = retry_config
            attempt = _attempt_increment(
                increment_start, start_insertion, target_insertion,
                initial.positions, rest, reference, masses, nodal, tube,
                retry_config, load_step,
            )
            if attempt.success:
                accepted = attempt
                break
        if accepted is None:
            for subdivisions in config.retry_subdivisions:
                local_state = _clone_state(increment_start)
                local_records: list[HistoryRecord] = []
                subdivision_ok = True
                for subdivision in range(1, subdivisions + 1):
                    sub_start = start_insertion + (
                        target_insertion - start_insertion
                    ) * (subdivision - 1) / subdivisions
                    sub_target = start_insertion + (
                        target_insertion - start_insertion
                    ) * subdivision / subdivisions
                    attempt = _attempt_increment(
                        local_state, sub_start, sub_target, initial.positions,
                        rest, reference, masses, nodal, tube,
                        final_retry_config, load_step,
                    )
                    if not attempt.success:
                        subdivision_ok = False
                        break
                    local_state = attempt.state
                    local_records.extend(attempt.records)
                if subdivision_ok:
                    accepted = _AttemptResult(
                        True, local_state, tuple(local_records), None,
                        attempt.max_contact_error,
                    )
                    break
        if accepted is None:
            state = increment_start
            abort_reason = attempt.reason
            break
        state = accepted.state
        history.extend(accepted.records)
        completed += 1

    success = completed == config.num_load_steps
    return SimulationResult(
        config=config, success=success, abort_reason=None if success else abort_reason,
        completed_load_steps=completed, history=tuple(history),
        final_state=_clone_state(state), prebalanced_state=prebalanced_state,
        prebalance_history_length=prebalance_history_length,
    )


def parse_args(argv: list[str] | None = None) -> SineInsertConfig:
    """Parse the complete in-memory sine insertion configuration."""
    defaults = SineInsertConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    integer_fields = {
        "num_nodes", "num_load_steps", "max_newton_iters",
        "settle_confirm_steps", "max_settle_steps", "prebalance_max_steps",
        "ramp_steps", "history_save_interval", "max_kp_retries",
        "max_ki_retries",
    }
    string_fields = {"output_dir", "run_name"}
    excluded = string_fields | integer_fields | {
        "backend", "prebalance", "retry_subdivisions",
    }
    for item in fields(SineInsertConfig):
        name = item.name
        option = "--" + name.replace("_", "-")
        if name in excluded:
            continue
        parser.add_argument(option, type=float, default=getattr(defaults, name))
    for name in sorted(integer_fields):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in sorted(string_fields):
        parser.add_argument("--" + name.replace("_", "-"), default=getattr(defaults, name))
    parser.add_argument(
        "--prebalance", action=argparse.BooleanOptionalAction,
        default=defaults.prebalance,
    )
    parser.add_argument(
        "--retry-subdivisions", type=int, nargs="*",
        default=defaults.retry_subdivisions,
    )
    parser.add_argument("--backend", choices=("python", "cpp"), default=defaults.backend)
    values = vars(parser.parse_args(argv))
    values["retry_subdivisions"] = tuple(values["retry_subdivisions"])
    try:
        return SineInsertConfig(**values)
    except ValueError as exc:
        parser.error(str(exc))


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    result = run_simulation(config)
    from sine2d_benchmark_io import export_simulation

    manifest = export_simulation(result)
    for name, path in manifest.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
