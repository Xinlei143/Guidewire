"""Free-distal 3-D guidewire insertion into an analytic straight circular tube.

The two rightmost nodes model a short translating actuator guide.  Every node
to their left is free.  This module is a Python/CPU/float64 reference demo and
does not use or alter the v4 C++ topology.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
import functools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

import rod_soft_contact_PI_dynamic_quasi_static_3d_v4 as v4


DTYPE = torch.float64


@dataclass(frozen=True)
class StraightFreeInsertConfig:
    total_length: float = 1.5
    num_nodes: int = 250
    tube_radius: float = 60.0e-3
    safe_distance: float = 10.0e-3
    imperfection_fraction: float = 0.25
    imperfection_penetration: float = 0.2e-3
    imperfection_z_amplitude: float = 0.5e-3

    diameter: float = 1.0e-3
    density: float = 7850.0
    young_modulus: float = 200.0e9
    damping_ratio: float = 1.0

    contact_kp: float = 3000.0
    contact_ki: float = 100.0
    max_contact_force: float = 20.0
    max_integral_state: float = 5.0e-2
    integral_leak_rate: float = 5.0
    contact_smooth_length: float = 2.0e-5
    contact_active_weight: float = 0.01
    direction_epsilon: float = 1.0e-6
    friction_mu: float = 1.0
    friction_kt_line: float = 1_118_600.0

    dt: float = 0.05
    max_newton_iters: int = 200
    residual_tol: float = 1.0e-7
    step_tol: float = 1.0e-10
    settle_speed_tol: float = 1.0e-3
    settle_residual_tol: float = 1.0e-6
    contact_tolerance: float = 5.0e-4
    settle_confirm_steps: int = 4
    max_settle_steps: int = 16_000
    prebalance_max_steps: int = 16_000
    ramp_steps: int = 25
    total_insertion: float = 90.0e-3
    num_load_steps: int = 20
    retry_subdivisions: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    ki_reduction_factor: float = 0.5
    ki_min_scale: float = 0.125
    diagnostic_interval: int = 50
    progress: bool = True

    output_dir: str = "3d_npz/straight_free_insert"
    run_name: str = "straight_free_insert"

    def __post_init__(self) -> None:
        positive = (
            "total_length", "tube_radius", "safe_distance", "imperfection_fraction",
            "imperfection_penetration", "imperfection_z_amplitude", "diameter",
            "density", "young_modulus", "damping_ratio", "contact_kp",
            "max_contact_force", "max_integral_state", "contact_smooth_length",
            "contact_active_weight", "direction_epsilon", "friction_kt_line", "dt", "residual_tol",
            "step_tol", "settle_speed_tol", "settle_residual_tol",
            "contact_tolerance", "total_insertion", "ki_reduction_factor", "ki_min_scale",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("contact_ki", "integral_leak_rate", "friction_mu"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 < self.safe_distance < self.tube_radius:
            raise ValueError("safe_distance must lie strictly between zero and tube_radius")
        if not 0.0 < self.imperfection_fraction <= 0.5:
            raise ValueError("imperfection_fraction must lie in (0, 0.5]")
        if self.contact_active_weight >= 0.5:
            raise ValueError("contact_active_weight must be smaller than 0.5")
        if self.ki_reduction_factor >= 1.0:
            raise ValueError("ki_reduction_factor must be smaller than one")
        if self.ki_min_scale > 1.0:
            raise ValueError("ki_min_scale must not exceed one")
        if self.num_nodes < 6:
            raise ValueError("num_nodes must be at least six")
        for name in (
            "max_newton_iters", "settle_confirm_steps", "max_settle_steps",
            "prebalance_max_steps", "ramp_steps", "num_load_steps",
            "diagnostic_interval",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.total_insertion >= self.total_length:
            raise ValueError("total_insertion must be smaller than total_length")


def reduced_contact_ki_config(
    config: StraightFreeInsertConfig, scale: float
) -> StraightFreeInsertConfig:
    """Return a solver-equivalent configuration with a reduced PI integral gain.

    This mirrors the v4 local-continuation fallback: all retry attempts begin
    from the accepted state and only the *next* full retry ladder sees the
    lower Ki.  The caller owns the accepted checkpoint and friction context.
    """
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError("scale must be finite and lie in (0, 1]")
    return replace(config, contact_ki=config.contact_ki * scale)


@dataclass(frozen=True)
class RodInitialization:
    positions: np.ndarray
    velocities: np.ndarray
    rest_lengths: np.ndarray
    nodal_lengths: np.ndarray
    masses: np.ndarray
    integrals: np.ndarray
    previous_errors: np.ndarray


@dataclass(frozen=True)
class PrescribedState:
    mask: np.ndarray
    positions: np.ndarray


def _smoothstep(value: np.ndarray) -> np.ndarray:
    return value * value * (3.0 - 2.0 * value)


def initialize_rod(config: StraightFreeInsertConfig) -> RodInitialization:
    """Build a deterministic near-actuator imperfection in the last quarter."""
    x = np.linspace(0.0, config.total_length, config.num_nodes)
    positions = np.zeros((config.num_nodes, 3), dtype=np.float64)
    positions[:, 0] = x
    start_x = (1.0 - config.imperfection_fraction) * config.total_length
    transition_end = x[-3]
    if transition_end <= start_x:
        start_x = x[-4]
    u = np.clip((x - start_x) / (transition_end - start_x), 0.0, 1.0)
    h = _smoothstep(u)
    radial_target = config.tube_radius - config.safe_distance + config.imperfection_penetration
    positions[:, 1] = radial_target * h
    positions[:, 2] = config.imperfection_z_amplitude * np.sin(np.pi * u) ** 2
    positions[-2:, 1] = radial_target
    positions[-2:, 2] = 0.0
    rest_lengths = np.linalg.vector_norm(np.diff(positions, axis=0), axis=1)
    nodal = np.empty(config.num_nodes, dtype=np.float64)
    nodal[0] = 0.5 * rest_lengths[0]
    nodal[-1] = 0.5 * rest_lengths[-1]
    nodal[1:-1] = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
    area = math.pi * config.diameter**2 / 4.0
    masses = config.density * area * nodal
    return RodInitialization(
        positions=positions,
        velocities=np.zeros_like(positions),
        rest_lengths=rest_lengths,
        nodal_lengths=nodal,
        masses=masses,
        integrals=np.zeros(config.num_nodes, dtype=np.float64),
        previous_errors=np.zeros(config.num_nodes, dtype=np.float64),
    )


def make_prescribed_state(initial: np.ndarray, displacement: float) -> PrescribedState:
    if initial.ndim != 2 or initial.shape[1] != 3 or len(initial) < 6:
        raise ValueError("initial must be an N-by-3 array with at least six nodes")
    if not math.isfinite(displacement) or displacement < 0.0:
        raise ValueError("displacement must be finite and non-negative")
    mask = np.zeros_like(initial, dtype=bool)
    mask[-2:] = True
    positions = np.array(initial, dtype=np.float64, copy=True)
    positions[-2:, 0] -= displacement
    return PrescribedState(mask=mask, positions=positions)


@functools.lru_cache(maxsize=8)
def get_block_plan(device_str: str, num_nodes: int) -> v4.BlockAssemblyPlan:
    device = torch.device(device_str)
    free_start = 0
    free_end = num_nodes - 2
    mask = torch.zeros((num_nodes, 3), dtype=torch.bool, device=device)
    mask[-2:] = True
    free_indices = torch.nonzero((~mask).reshape(-1), as_tuple=False).squeeze(1)
    return v4.BlockAssemblyPlan(
        device=device,
        num_nodes=num_nodes,
        free_start=free_start,
        free_end=free_end,
        block_count=free_end,
        free_indices=free_indices,
        stretch=v4._make_block_local_assembly_plan(
            device, num_nodes - 1, 2, 0, free_start, free_end
        ),
        bend=v4._make_block_local_assembly_plan(
            device, num_nodes - 2, 3, 0, free_start, free_end
        ),
    )


def _validate_friction_state(
    elastic_slip: torch.Tensor,
    anchors: torch.Tensor,
    contact_active: torch.Tensor,
    owner: str,
) -> int:
    if not isinstance(elastic_slip, torch.Tensor) or elastic_slip.ndim != 2 or elastic_slip.shape[1] != 3:
        raise ValueError(f"{owner}.elastic_slip must be an N-by-3 tensor")
    count = len(elastic_slip)
    for value, name in ((elastic_slip, "elastic_slip"), (anchors, "anchors")):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float64
            or value.device.type != "cpu"
            or tuple(value.shape) != (count, 3)
        ):
            raise TypeError(f"{owner}.{name} must be an N-by-3 CPU float64 tensor")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{owner}.{name} must be finite")
    if (
        not isinstance(contact_active, torch.Tensor)
        or contact_active.dtype != torch.bool
        or contact_active.device.type != "cpu"
        or tuple(contact_active.shape) != (count,)
    ):
        raise TypeError(f"{owner}.contact_active must be a length-N CPU bool tensor")
    return count


@dataclass(frozen=True, init=False)
class FrictionHistory3D:
    _elastic_slip: torch.Tensor
    _anchors: torch.Tensor
    _contact_active: torch.Tensor

    def __init__(
        self,
        elastic_slip: torch.Tensor,
        anchors: torch.Tensor,
        contact_active: torch.Tensor,
    ) -> None:
        _validate_friction_state(elastic_slip, anchors, contact_active, "history")
        object.__setattr__(self, "_elastic_slip", elastic_slip.clone())
        object.__setattr__(self, "_anchors", anchors.clone())
        object.__setattr__(self, "_contact_active", contact_active.clone())

    @property
    def elastic_slip(self) -> torch.Tensor:
        return self._elastic_slip.clone()

    @property
    def anchors(self) -> torch.Tensor:
        return self._anchors.clone()

    @property
    def contact_active(self) -> torch.Tensor:
        return self._contact_active.clone()


@dataclass(frozen=True, init=False)
class FrictionIncrementContext3D:
    _elastic_slip: torch.Tensor
    _anchors: torch.Tensor
    _contact_active: torch.Tensor

    def __init__(
        self,
        elastic_slip: torch.Tensor,
        anchors: torch.Tensor,
        contact_active: torch.Tensor,
    ) -> None:
        _validate_friction_state(elastic_slip, anchors, contact_active, "context")
        object.__setattr__(self, "_elastic_slip", elastic_slip.clone())
        object.__setattr__(self, "_anchors", anchors.clone())
        object.__setattr__(self, "_contact_active", contact_active.clone())

    @property
    def elastic_slip(self) -> torch.Tensor:
        return self._elastic_slip.clone()

    @property
    def anchors(self) -> torch.Tensor:
        return self._anchors.clone()

    @property
    def contact_active(self) -> torch.Tensor:
        return self._contact_active.clone()


def begin_friction_increment(
    committed: FrictionHistory3D, start_positions: torch.Tensor
) -> FrictionIncrementContext3D:
    if not isinstance(committed, FrictionHistory3D):
        raise TypeError("committed must be FrictionHistory3D")
    count = len(committed._elastic_slip)
    if (
        not isinstance(start_positions, torch.Tensor)
        or start_positions.dtype != torch.float64
        or start_positions.device.type != "cpu"
        or tuple(start_positions.shape) != (count, 3)
        or not bool(torch.isfinite(start_positions).all())
    ):
        raise TypeError("start_positions must be a finite N-by-3 CPU float64 tensor")
    return FrictionIncrementContext3D(
        committed._elastic_slip, start_positions, committed._contact_active
    )


def commit_friction_increment(
    context: FrictionIncrementContext3D,
    settled_positions: torch.Tensor,
    current_active: torch.Tensor,
    candidate_slip: torch.Tensor,
) -> FrictionHistory3D:
    if not isinstance(context, FrictionIncrementContext3D):
        raise TypeError("context must be FrictionIncrementContext3D")
    count = len(context._elastic_slip)
    _validate_friction_state(candidate_slip, settled_positions, current_active, "commit")
    if len(candidate_slip) != count:
        raise ValueError("commit node count must match context")
    committed = torch.where(
        current_active[:, None], candidate_slip, torch.zeros_like(candidate_slip)
    )
    return FrictionHistory3D(committed, settled_positions, current_active)


@dataclass(frozen=True)
class ContactFrictionResult:
    total_force: torch.Tensor
    normal_force: torch.Tensor
    friction_force: torch.Tensor
    new_integrals: torch.Tensor
    error: torch.Tensor
    weight: torch.Tensor
    active: torch.Tensor
    normal: torch.Tensor
    tangent_projector: torch.Tensor
    trial_slip: torch.Tensor
    candidate_slip: torch.Tensor
    stick: torch.Tensor
    slip: torch.Tensor
    coulomb_limit: torch.Tensor
    distance: torch.Tensor

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, torch.Tensor):
                object.__setattr__(self, item.name, value.clone())


def _local_contact_friction_force(
    point: torch.Tensor,
    old_integral: torch.Tensor,
    previous_error: torch.Tensor,
    nodal_length: torch.Tensor,
    anchor: torch.Tensor,
    committed_slip: torch.Tensor,
    committed_active: torch.Tensor,
    config: StraightFreeInsertConfig,
    dt: float,
    friction_enabled: bool,
) -> torch.Tensor:
    yz = point[1:]
    radial = torch.linalg.vector_norm(yz)
    z = radial - (config.tube_radius - config.safe_distance)
    beta = 1.0 / config.contact_smooth_length
    error = F.softplus(z, beta=beta)
    weight = torch.sigmoid(beta * z)
    candidate_integral = torch.clamp(
        weight * (old_integral + error * dt)
        + (1.0 - weight) * old_integral * math.exp(-config.integral_leak_rate * dt),
        0.0,
        config.max_integral_state,
    )
    unsaturated = config.contact_kp * error + config.contact_ki * candidate_integral
    stop_growth = (
        (config.contact_ki > 0.0)
        & (unsaturated >= config.max_contact_force)
        & (candidate_integral > old_integral)
    ) | (
        (config.contact_ki > 0.0)
        & (error < previous_error)
        & (config.contact_ki * old_integral > config.contact_kp * error)
        & (candidate_integral > old_integral)
    )
    integral = torch.where(stop_growth, old_integral, candidate_integral)
    magnitude = torch.clamp(
        config.contact_kp * error + config.contact_ki * integral,
        0.0,
        config.max_contact_force,
    )
    normal_yz = yz / radial.clamp_min(1.0e-30)
    normal = torch.cat((torch.zeros(1, dtype=point.dtype), normal_yz))
    direction_scale = torch.sqrt(radial.square() + config.direction_epsilon**2)
    normal_force = -normal * magnitude * radial / direction_scale
    projector = torch.eye(3, dtype=point.dtype) - torch.outer(normal, normal)
    base_slip = torch.where(
        committed_active, projector @ committed_slip, torch.zeros_like(committed_slip)
    )
    trial_slip = base_slip + projector @ (point - anchor)
    stiffness = config.friction_kt_line * nodal_length
    uncapped = -stiffness * trial_slip
    limit = config.friction_mu * torch.linalg.vector_norm(normal_force)
    uncapped_norm = torch.linalg.vector_norm(uncapped)
    active = weight >= config.contact_active_weight
    law_active = active & friction_enabled & (config.friction_mu > 0.0)
    capped = -limit * trial_slip / torch.linalg.vector_norm(trial_slip).clamp_min(1.0e-30)
    friction = torch.where(uncapped_norm <= limit, uncapped, capped)
    friction = torch.where(law_active, friction, torch.zeros_like(friction))
    return normal_force + friction


def evaluate_straight_tube_contact_friction(
    positions: torch.Tensor,
    old_integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    nodal_lengths: torch.Tensor,
    context: FrictionIncrementContext3D,
    config: StraightFreeInsertConfig,
    dt: float,
    friction_enabled: bool = True,
) -> ContactFrictionResult:
    if not isinstance(context, FrictionIncrementContext3D):
        raise TypeError("context must be FrictionIncrementContext3D")
    count = len(positions)
    if tuple(positions.shape) != (count, 3) or positions.dtype != torch.float64 or positions.device.type != "cpu":
        raise TypeError("positions must be an N-by-3 CPU float64 tensor")
    for value, name in (
        (old_integrals, "old_integrals"),
        (previous_errors, "previous_errors"),
        (nodal_lengths, "nodal_lengths"),
    ):
        if value.dtype != torch.float64 or value.device.type != "cpu" or tuple(value.shape) != (count,):
            raise TypeError(f"{name} must be a length-N CPU float64 tensor")
    if len(context._elastic_slip) != count:
        raise ValueError("context node count must match positions")
    yz = positions[:, 1:]
    radial = torch.linalg.vector_norm(yz, dim=1)
    distance = config.tube_radius - radial
    z = radial - (config.tube_radius - config.safe_distance)
    beta = 1.0 / config.contact_smooth_length
    error = F.softplus(z, beta=beta)
    weight = torch.sigmoid(beta * z)
    candidate_integrals = torch.clamp(
        weight * (old_integrals + error * dt)
        + (1.0 - weight) * old_integrals * math.exp(-config.integral_leak_rate * dt),
        0.0,
        config.max_integral_state,
    )
    unsaturated = config.contact_kp * error + config.contact_ki * candidate_integrals
    stop_growth = (
        (config.contact_ki > 0.0)
        & (unsaturated >= config.max_contact_force)
        & (candidate_integrals > old_integrals)
    ) | (
        (config.contact_ki > 0.0)
        & (error < previous_errors)
        & (config.contact_ki * old_integrals > config.contact_kp * error)
        & (candidate_integrals > old_integrals)
    )
    new_integrals = torch.where(stop_growth, old_integrals, candidate_integrals)
    magnitude = torch.clamp(
        config.contact_kp * error + config.contact_ki * new_integrals,
        0.0,
        config.max_contact_force,
    )
    normal = torch.zeros_like(positions)
    normal[:, 1:] = yz / radial.clamp_min(1.0e-30)[:, None]
    normal_force = -normal * (magnitude * radial / torch.sqrt(
        radial.square() + config.direction_epsilon**2
    ))[:, None]
    eye = torch.eye(3, dtype=positions.dtype).expand(count, 3, 3)
    projector = eye - normal[:, :, None] * normal[:, None, :]
    base_slip = torch.where(
        context._contact_active[:, None],
        torch.einsum("nij,nj->ni", projector, context._elastic_slip),
        torch.zeros_like(context._elastic_slip),
    )
    delta = torch.einsum("nij,nj->ni", projector, positions - context._anchors)
    trial_slip = base_slip + delta
    stiffness = config.friction_kt_line * nodal_lengths
    uncapped = -stiffness[:, None] * trial_slip
    coulomb_limit = config.friction_mu * torch.linalg.vector_norm(normal_force, dim=1)
    uncapped_norm = torch.linalg.vector_norm(uncapped, dim=1)
    active = weight >= config.contact_active_weight
    law_active = active & friction_enabled & (config.friction_mu > 0.0)
    stick = law_active & (uncapped_norm <= coulomb_limit)
    slip = law_active & ~stick
    capped = -coulomb_limit[:, None] * trial_slip / torch.linalg.vector_norm(
        trial_slip, dim=1
    ).clamp_min(1.0e-30)[:, None]
    friction_force = torch.where(stick[:, None], uncapped, capped)
    friction_force = torch.where(law_active[:, None], friction_force, torch.zeros_like(friction_force))
    candidate_slip = torch.where(
        law_active[:, None],
        -friction_force / stiffness[:, None],
        torch.zeros_like(friction_force),
    )
    return ContactFrictionResult(
        total_force=normal_force + friction_force,
        normal_force=normal_force,
        friction_force=friction_force,
        new_integrals=new_integrals,
        error=error,
        weight=weight,
        active=active,
        normal=normal,
        tangent_projector=projector,
        trial_slip=trial_slip,
        candidate_slip=candidate_slip,
        stick=stick,
        slip=slip,
        coulomb_limit=coulomb_limit,
        distance=distance,
    )


def external_force_and_blocks(
    positions: torch.Tensor,
    old_integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    nodal_lengths: torch.Tensor,
    context: FrictionIncrementContext3D,
    config: StraightFreeInsertConfig,
    dt: float,
    friction_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    def force_aux(
        point: torch.Tensor,
        old_integral: torch.Tensor,
        previous_error: torch.Tensor,
        nodal_length: torch.Tensor,
        anchor: torch.Tensor,
        committed_slip: torch.Tensor,
        committed_active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        force = _local_contact_friction_force(
            point, old_integral, previous_error, nodal_length, anchor,
            committed_slip, committed_active, config, dt, friction_enabled,
        )
        return force, force

    blocks, forces = torch.func.vmap(
        torch.func.jacrev(force_aux, argnums=0, has_aux=True)
    )(
        positions, old_integrals, previous_errors, nodal_lengths,
        context._anchors, context._elastic_slip, context._contact_active,
    )
    return forces, blocks


def _mechanical_properties(config: StraightFreeInsertConfig) -> tuple[float, float, float]:
    area = math.pi * config.diameter**2 / 4.0
    second_moment = math.pi * config.diameter**4 / 64.0
    ea = config.young_modulus * area
    ei = config.young_modulus * second_moment
    total_mass = config.density * area * config.total_length
    damping = 2.0 * config.damping_ratio * math.sqrt(
        (3.0 * ei / config.total_length**3) * total_mass
    )
    return ea, ei, damping


def update_velocities(
    new_positions: torch.Tensor,
    old_positions: torch.Tensor,
    prescribed_mask: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if new_positions.shape != old_positions.shape or prescribed_mask.shape != old_positions.shape:
        raise ValueError("position and mask shapes must match")
    # The same kinematic difference is valid for free and prescribed DOFs.  In
    # particular, there is no special case that silently clamps the left tip.
    return ((new_positions - old_positions) / dt).detach()


def _unpack_free(free: torch.Tensor, prescribed: torch.Tensor) -> torch.Tensor:
    positions = prescribed.clone()
    positions[:-2] = free.reshape(-1, 3)
    return positions


def residual_and_jacobian(
    free: torch.Tensor,
    old_positions: torch.Tensor,
    old_velocities: torch.Tensor,
    rest_lengths: torch.Tensor,
    masses: torch.Tensor,
    old_integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    nodal_lengths: torch.Tensor,
    prescribed: torch.Tensor,
    friction_context: FrictionIncrementContext3D,
    config: StraightFreeInsertConfig,
    dt: float,
    friction_enabled: bool,
    with_jacobian: bool,
) -> tuple[torch.Tensor, v4.BlockPentaSystem | None, torch.Tensor]:
    positions = _unpack_free(free, prescribed)
    plan = get_block_plan("cpu", len(positions))
    ea, ei, damping = _mechanical_properties(config)
    if with_jacobian:
        prescribed_mask = torch.zeros_like(positions, dtype=torch.bool)
        prescribed_mask[-2:] = True
        gradient, system, free_start, free_end = v4.assemble_elastic_block_penta_v4(
            positions, rest_lengths, prescribed_mask, ea, ei, plan
        )
        external, blocks = external_force_and_blocks(
            positions, old_integrals, previous_errors, nodal_lengths,
            friction_context, config, dt, friction_enabled,
        )
    else:
        differentiable = positions.detach().clone().requires_grad_(True)
        _, _, energy = v4.elastic_energy(differentiable, rest_lengths, ea, ei)
        full_gradient = torch.autograd.grad(energy, differentiable)[0]
        gradient = full_gradient[:-2].reshape(-1)
        external = evaluate_straight_tube_contact_friction(
            positions, old_integrals, previous_errors, nodal_lengths,
            friction_context, config, dt, friction_enabled,
        ).total_force
        system = None
        blocks = None
        free_start, free_end = 0, len(positions) - 2
    old_free = old_positions[:-2].reshape(-1)
    old_velocity = old_velocities[:-2].reshape(-1)
    free_masses = masses[:-2, None].expand(-1, 3).reshape(-1)
    residual = (
        free_masses * (free - old_free - dt * old_velocity) / dt**2
        + gradient
        + damping * (free - old_free) / dt
        - external[:-2].reshape(-1)
    )
    if system is not None:
        diagonal = (free_masses / dt**2 + damping / dt).reshape(-1, 3)
        system.diagonal.diagonal(dim1=1, dim2=2).add_(diagonal)
        system.diagonal.add_(-blocks[free_start:free_end])
    return residual, system, positions


@dataclass(frozen=True)
class ImplicitStepResult:
    positions: torch.Tensor
    velocities: torch.Tensor
    integrals: torch.Tensor
    previous_errors: torch.Tensor
    normal_force: torch.Tensor
    friction_force: torch.Tensor
    active: torch.Tensor
    stick: torch.Tensor
    slip: torch.Tensor
    candidate_slip: torch.Tensor
    dynamic_force: torch.Tensor
    prescribed_mask: torch.Tensor
    info: v4.NewtonInfo


def implicit_euler_step(
    old_positions: torch.Tensor,
    old_velocities: torch.Tensor,
    rest_lengths: torch.Tensor,
    masses: torch.Tensor,
    old_integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    nodal_lengths: torch.Tensor,
    prescribed_mask: torch.Tensor,
    prescribed_positions: torch.Tensor,
    friction_context: FrictionIncrementContext3D,
    config: StraightFreeInsertConfig,
    dt: float,
    *,
    friction_enabled: bool,
) -> ImplicitStepResult:
    expected_mask = torch.zeros_like(old_positions, dtype=torch.bool)
    expected_mask[-2:] = True
    if not torch.equal(prescribed_mask, expected_mask):
        raise ValueError("only the last two nodes may be prescribed")
    free = (old_positions[:-2] + dt * old_velocities[:-2]).reshape(-1).detach()
    residual_norm = math.inf
    step_norm = math.inf
    failures = 0
    converged = False
    iteration = 0
    for iteration in range(1, config.max_newton_iters + 1):
        residual, system, _ = residual_and_jacobian(
            free, old_positions, old_velocities, rest_lengths, masses,
            old_integrals, previous_errors, nodal_lengths, prescribed_positions,
            friction_context, config, dt, friction_enabled, True,
        )
        residual_norm = float(torch.linalg.vector_norm(residual))
        if residual_norm < config.residual_tol:
            converged = True
            step_norm = 0.0
            break
        try:
            delta, _used_dense_fallback = system.solve(-residual)
        except RuntimeError:
            delta = torch.linalg.solve(system.dense(), -residual)
        step_norm = float(torch.linalg.vector_norm(delta))
        base_norm = residual_norm
        accepted = False
        for exponent in range(12):
            candidate = free + (0.5**exponent) * delta
            trial_residual, _, _ = residual_and_jacobian(
                candidate, old_positions, old_velocities, rest_lengths, masses,
                old_integrals, previous_errors, nodal_lengths, prescribed_positions,
                friction_context, config, dt, friction_enabled, False,
            )
            trial_norm = float(torch.linalg.vector_norm(trial_residual))
            if math.isfinite(trial_norm) and trial_norm < base_norm:
                free = candidate.detach()
                residual_norm = trial_norm
                accepted = True
                break
        if not accepted:
            failures += 1
            forced = free + 0.05 * delta
            forced_residual, _, _ = residual_and_jacobian(
                forced, old_positions, old_velocities, rest_lengths, masses,
                old_integrals, previous_errors, nodal_lengths, prescribed_positions,
                friction_context, config, dt, friction_enabled, False,
            )
            forced_norm = float(torch.linalg.vector_norm(forced_residual))
            if math.isfinite(forced_norm):
                free = forced.detach()
                residual_norm = forced_norm
            else:
                residual_norm = math.inf
        if step_norm < config.step_tol:
            converged = residual_norm < 10.0 * config.residual_tol
            break
    positions = _unpack_free(free, prescribed_positions).detach()
    contact = evaluate_straight_tube_contact_friction(
        positions, old_integrals, previous_errors, nodal_lengths,
        friction_context, config, dt, friction_enabled,
    )
    velocities = update_velocities(positions, old_positions, prescribed_mask, dt)
    _, _, damping = _mechanical_properties(config)
    dynamic_force = (
        masses[:, None] * (positions - old_positions - dt * old_velocities) / dt**2
        + damping * (positions - old_positions) / dt
    ).detach()
    return ImplicitStepResult(
        positions=positions,
        velocities=velocities,
        integrals=contact.new_integrals.detach(),
        previous_errors=contact.error.detach(),
        normal_force=contact.normal_force.detach(),
        friction_force=contact.friction_force.detach(),
        active=contact.active.detach(),
        stick=contact.stick.detach(),
        slip=contact.slip.detach(),
        candidate_slip=contact.candidate_slip.detach(),
        dynamic_force=dynamic_force,
        prescribed_mask=prescribed_mask.clone(),
        info=v4.NewtonInfo(converged, iteration, residual_norm, step_norm, failures),
    )


def compute_actuator_reaction(
    positions: torch.Tensor,
    old_positions: torch.Tensor,
    old_velocities: torch.Tensor,
    rest_lengths: torch.Tensor,
    masses: torch.Tensor,
    old_integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    nodal_lengths: torch.Tensor,
    friction_context: FrictionIncrementContext3D,
    config: StraightFreeInsertConfig,
    dt: float,
    friction_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    ea, ei, damping = _mechanical_properties(config)
    differentiable = positions.detach().clone().requires_grad_(True)
    _, _, energy = v4.elastic_energy(differentiable, rest_lengths, ea, ei)
    elastic_gradient = torch.autograd.grad(energy, differentiable)[0].detach()
    external = evaluate_straight_tube_contact_friction(
        positions, old_integrals, previous_errors, nodal_lengths,
        friction_context, config, dt, friction_enabled,
    ).total_force
    dynamic = (
        masses[:, None] * (positions - old_positions - dt * old_velocities) / dt**2
        + damping * (positions - old_positions) / dt
    ).detach()
    actuator_on_wire = dynamic + elastic_gradient - external
    return actuator_on_wire[-2:].detach(), dynamic


@dataclass(frozen=True)
class SimulationResult:
    config: StraightFreeInsertConfig
    history: dict[str, np.ndarray]
    completed_load_steps: int
    friction_commit_count: int
    config_path: Path
    settled_path: Path
    history_path: Path
    figure_path: Path


@dataclass(frozen=True)
class ComparisonResult:
    results: dict[float, SimulationResult]
    figure_path: Path


def _tensor(value: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64).clone()


def _make_mask(count: int) -> torch.Tensor:
    mask = torch.zeros((count, 3), dtype=torch.bool)
    mask[-2:] = True
    return mask


def _settled(
    result: ImplicitStepResult, config: StraightFreeInsertConfig
) -> bool:
    free_speed = torch.linalg.vector_norm(result.velocities[:-2], dim=1).max()
    max_intrusion = torch.clamp(
        torch.linalg.vector_norm(result.positions[:, 1:], dim=1)
        - (config.tube_radius - config.safe_distance),
        min=0.0,
    ).max()
    return bool(
        result.info.converged
        and float(free_speed) < config.settle_speed_tol
        and result.info.residual_norm < config.settle_residual_tol
        and float(max_intrusion) < config.contact_tolerance
    )


def dynamic_balance_acceptable(
    dynamic_remainder: float, push_force: float, absolute_tolerance: float
) -> bool:
    """Require a 5% dynamic/reaction ratio, with an absolute near-zero-force floor."""
    return dynamic_remainder <= max(0.05 * abs(push_force), absolute_tolerance)


def free_tip_advance(
    positions: torch.Tensor, prebalanced_reference: torch.Tensor
) -> float:
    return float(prebalanced_reference[0, 0] - positions[0, 0])


def _advance_interval(
    state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    initial_positions: np.ndarray,
    rest_lengths: torch.Tensor,
    masses: torch.Tensor,
    nodal_lengths: torch.Tensor,
    context: FrictionIncrementContext3D,
    config: StraightFreeInsertConfig,
    start_displacement: float,
    end_displacement: float,
    *,
    friction_enabled: bool,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ImplicitStepResult]:
    """Retry from an immutable checkpoint; friction context never participates in rollback."""
    checkpoint = tuple(value.clone() for value in state)
    last_info: v4.NewtonInfo | None = None
    for subdivisions in config.retry_subdivisions:
        attempt = tuple(value.clone() for value in checkpoint)
        latest: ImplicitStepResult | None = None
        failed = False
        for index in range(1, subdivisions + 1):
            fraction = index / subdivisions
            displacement = start_displacement + (end_displacement - start_displacement) * fraction
            prescribed_np = make_prescribed_state(initial_positions, displacement)
            prescribed = _tensor(prescribed_np.positions)
            mask = torch.as_tensor(prescribed_np.mask, dtype=torch.bool)
            latest = implicit_euler_step(
                attempt[0], attempt[1], rest_lengths, masses, attempt[2], attempt[3],
                nodal_lengths, mask, prescribed, context, config, config.dt / subdivisions,
                friction_enabled=friction_enabled,
            )
            last_info = latest.info
            if not latest.info.converged:
                failed = True
                break
            attempt = (
                latest.positions, latest.velocities,
                latest.integrals, latest.previous_errors,
            )
        if not failed and latest is not None:
            return attempt, latest
    detail = "unknown" if last_info is None else f"residual={last_info.residual_norm:.3e}"
    raise RuntimeError(f"implicit continuation failed after transactional retries: {detail}")


def _diagnostics(
    positions: torch.Tensor,
    dynamic_force: torch.Tensor,
    rest_lengths: torch.Tensor,
    masses: torch.Tensor,
    integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    nodal_lengths: torch.Tensor,
    context: FrictionIncrementContext3D,
    config: StraightFreeInsertConfig,
    displacement: float,
    prebalanced_reference: torch.Tensor,
    friction_enabled: bool,
    effective_ki_scale: float = 1.0,
) -> dict[str, object]:
    ea, ei, _ = _mechanical_properties(config)
    differentiable = positions.detach().clone().requires_grad_(True)
    stretch, bend, total = v4.elastic_energy(differentiable, rest_lengths, ea, ei)
    elastic_gradient = torch.autograd.grad(total, differentiable)[0].detach()
    contact = evaluate_straight_tube_contact_friction(
        positions, integrals, previous_errors, nodal_lengths,
        context, config, config.dt, friction_enabled,
    )
    actuator_reaction = (dynamic_force + elastic_gradient - contact.total_force)[-2:]
    push_force = -float(actuator_reaction[:, 0].sum())
    dynamic_remainder = float(torch.linalg.vector_norm(dynamic_force[:-2]))
    free_nodes = torch.nonzero(contact.active[:-2], as_tuple=False).squeeze(1)
    return {
        "actuator_displacement_m": displacement,
        "free_tip_advance_m": free_tip_advance(positions, prebalanced_reference),
        "positions": positions.detach().numpy().copy(),
        "normal_force_n": contact.normal_force.detach().numpy().copy(),
        "friction_force_n": contact.friction_force.detach().numpy().copy(),
        "actuator_reaction_n": actuator_reaction.detach().numpy().copy(),
        "push_force_n": push_force,
        "normal_resultant_n": float(torch.linalg.vector_norm(contact.normal_force.sum(dim=0))),
        "friction_resultant_n": float(torch.linalg.vector_norm(contact.friction_force.sum(dim=0))),
        "friction_axial_n": float(contact.friction_force[:, 0].sum()),
        "contact_count": int(torch.count_nonzero(contact.active)),
        "free_contact_start": int(free_nodes[0]) if len(free_nodes) else -1,
        "free_contact_end": int(free_nodes[-1]) if len(free_nodes) else -1,
        "stick_count": int(torch.count_nonzero(contact.stick)),
        "slip_count": int(torch.count_nonzero(contact.slip)),
        "stretching_energy_j": float(stretch.detach()),
        "bending_energy_j": float(bend.detach()),
        "elastic_energy_j": float(total.detach()),
        "dynamic_remainder_n": dynamic_remainder,
        "dynamic_to_push_ratio": dynamic_remainder / max(abs(push_force), config.settle_residual_tol),
        "effective_contact_ki": config.contact_ki,
        "effective_ki_scale": effective_ki_scale,
        "coulomb_max_ratio": float(
            torch.max(
                torch.linalg.vector_norm(contact.friction_force, dim=1)
                / torch.clamp(
                    config.friction_mu * torch.linalg.vector_norm(contact.normal_force, dim=1),
                    min=1.0e-30,
                )
            )
        ) if config.friction_mu > 0.0 else 0.0,
    }


def _history_arrays(records: list[dict[str, object]]) -> dict[str, np.ndarray]:
    return {key: np.asarray([record[key] for record in records]) for key in records[0]}


def _write_single_figure(
    history: dict[str, np.ndarray], config: StraightFreeInsertConfig, path: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(12, 8))
    axis = figure.add_subplot(2, 2, 1, projection="3d")
    axis.plot(*history["positions"][0].T, "k--", label="prebalanced")
    axis.plot(*history["positions"][-1].T, color="tab:blue", label="final")
    axis.set_title(f"3-D configuration, mu={config.friction_mu:g}")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.legend()
    x = history["actuator_displacement_m"] * 1.0e3
    axis = figure.add_subplot(2, 2, 2)
    axis.plot(x, history["free_tip_advance_m"] * 1.0e3, marker="o")
    axis.plot(x, x, "k--", label="rigid translation")
    axis.set(xlabel="actuator [mm]", ylabel="free tip [mm]")
    axis.legend()
    axis = figure.add_subplot(2, 2, 3)
    axis.plot(x, history["push_force_n"], label="push")
    axis.plot(x, history["normal_resultant_n"], label="normal resultant")
    axis.plot(x, history["friction_resultant_n"], label="friction resultant")
    axis.set(xlabel="actuator [mm]", ylabel="force [N]")
    axis.legend()
    axis = figure.add_subplot(2, 2, 4)
    axis.plot(x, history["bending_energy_j"], label="bending energy")
    twin = axis.twinx()
    twin.plot(x, history["contact_count"], color="tab:red", label="contacts")
    axis.set(xlabel="actuator [mm]", ylabel="energy [J]")
    twin.set_ylabel("contact nodes")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_outputs(
    config: StraightFreeInsertConfig, history: dict[str, np.ndarray]
) -> tuple[Path, Path, Path, Path]:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    settled_path = output / "settled.tsv"
    history_path = output / "history.npz"
    figure_path = output / "result.png"
    with config_path.open("w", encoding="utf-8") as stream:
        json.dump(asdict(config), stream, indent=2, sort_keys=True)
    scalar_keys = [
        key for key, value in history.items()
        if value.ndim == 1 and len(value) == len(history["actuator_displacement_m"])
    ]
    matrix = np.column_stack([history[key] for key in scalar_keys])
    np.savetxt(settled_path, matrix, delimiter="\t", header="\t".join(scalar_keys), comments="")
    np.savez_compressed(history_path, **history)
    _write_single_figure(history, config, figure_path)
    return config_path, settled_path, history_path, figure_path


def run_simulation(config: StraightFreeInsertConfig) -> SimulationResult:
    """Run normal-only prebalance followed by settled transactional insertion."""
    def progress(message: str) -> None:
        if config.progress:
            print(message, flush=True)

    active_config = config
    effective_ki_scale = 1.0

    initial = initialize_rod(config)
    initial_positions = initial.positions.copy()
    positions = _tensor(initial.positions)
    velocities = _tensor(initial.velocities)
    rest_lengths = _tensor(initial.rest_lengths)
    nodal_lengths = _tensor(initial.nodal_lengths)
    masses = _tensor(initial.masses)
    integrals = _tensor(initial.integrals)
    previous_errors = _tensor(initial.previous_errors)
    mask = _make_mask(config.num_nodes)
    inactive = torch.zeros(config.num_nodes, dtype=torch.bool)
    zero_history = FrictionHistory3D(torch.zeros_like(positions), positions, inactive)
    normal_context = begin_friction_increment(zero_history, positions)
    prebalance_result: ImplicitStepResult | None = None
    state = (positions, velocities, integrals, previous_errors)

    def advance_with_ki_adaptation(
        accepted_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        context: FrictionIncrementContext3D,
        start_displacement: float,
        end_displacement: float,
        *,
        friction_enabled: bool,
        phase_label: str,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ImplicitStepResult]:
        """Use v4's Ki ladder without allowing a rejected trial to mutate history."""
        nonlocal active_config, effective_ki_scale
        while True:
            try:
                return _advance_interval(
                    accepted_state, initial_positions, rest_lengths, masses, nodal_lengths,
                    context, active_config, start_displacement, end_displacement,
                    friction_enabled=friction_enabled,
                )
            except RuntimeError as error:
                if not str(error).startswith("implicit continuation failed after transactional retries:"):
                    raise
                next_scale = effective_ki_scale * config.ki_reduction_factor
                if config.contact_ki <= 0.0 or next_scale < config.ki_min_scale - 1.0e-15:
                    raise
                old_ki = active_config.contact_ki
                effective_ki_scale = next_scale
                active_config = reduced_contact_ki_config(config, effective_ki_scale)
                progress(
                    f"[KI ADAPT] {phase_label}; local {active_config.retry_subdivisions} retries failed. "
                    f"Ki_eff {old_ki:g} -> {active_config.contact_ki:g} N/(m*s) "
                    f"(scale={effective_ki_scale:g}); restoring accepted checkpoint."
                )

    progress(
        "[START] 3-D straight free-insert "
        f"| mu={config.friction_mu:g} | nodes={config.num_nodes} "
        f"| insertion={config.total_insertion * 1.0e3:.3f} mm "
        f"| Kp={config.contact_kp:g} | Ki={config.contact_ki:g}"
    )
    progress("[PREBALANCE] zero actuator displacement; left end is fully free")
    for prebalance_step in range(1, config.prebalance_max_steps + 1):
        state, prebalance_result = advance_with_ki_adaptation(
            state, normal_context, 0.0, 0.0, friction_enabled=False,
            phase_label="zero-displacement prebalance",
        )
        is_settled = _settled(prebalance_result, active_config)
        if prebalance_step % config.diagnostic_interval == 0 or is_settled:
            free_speed = float(torch.linalg.vector_norm(prebalance_result.velocities[:-2], dim=1).max())
            progress(
                f"[PREBALANCE {prebalance_step:5d}] v={free_speed:.2e} m/s "
                f"| R={prebalance_result.info.residual_norm:.2e} N "
                f"| contact={int(torch.count_nonzero(prebalance_result.active))} "
                f"| Ki_eff={active_config.contact_ki:g}"
            )
        if is_settled:
            break
    if prebalance_result is None or not _settled(prebalance_result, active_config):
        raise RuntimeError("zero-displacement normal-only prebalance did not settle")
    progress(f"[PREBALANCE COMPLETE] steps={prebalance_step} | Ki_eff={active_config.contact_ki:g}")
    positions, velocities, integrals, previous_errors = state
    if not bool(torch.any(prebalance_result.active[:-2])):
        raise RuntimeError("prebalance left no free node in the contact protection layer")
    committed = FrictionHistory3D(
        torch.zeros_like(positions), positions, prebalance_result.active
    )
    prebalanced_reference = positions.detach().clone()
    initial_context = begin_friction_increment(committed, positions)
    records = [_diagnostics(
        positions, prebalance_result.dynamic_force, rest_lengths, masses,
        integrals, previous_errors, nodal_lengths, initial_context, active_config, 0.0,
        prebalanced_reference, False, effective_ki_scale,
    )]
    completed = 0
    commit_count = 0
    for load_step in range(1, config.num_load_steps + 1):
        previous_target = (load_step - 1) * config.total_insertion / config.num_load_steps
        target = load_step * config.total_insertion / config.num_load_steps
        progress(
            f"[LOAD {load_step:02d}/{config.num_load_steps:02d}] "
            f"target={target * 1.0e3:.3f} mm"
        )
        context = begin_friction_increment(committed, positions)
        result: ImplicitStepResult | None = None
        for ramp in range(1, config.ramp_steps + 1):
            start = previous_target + (target - previous_target) * (ramp - 1) / config.ramp_steps
            end = previous_target + (target - previous_target) * ramp / config.ramp_steps
            state, result = advance_with_ki_adaptation(
                state, context, start, end, friction_enabled=True,
                phase_label=(
                    f"load {load_step}/{config.num_load_steps}, "
                    f"ramp {ramp}/{config.ramp_steps}"
                ),
            )
            if ramp % config.diagnostic_interval == 0 or ramp == config.ramp_steps:
                progress(
                    f"[RAMP {load_step:02d}/{config.num_load_steps:02d} | {ramp:3d}/{config.ramp_steps:3d}] "
                    f"tip={end * 1.0e3:.3f} mm | R={result.info.residual_norm:.2e} N "
                    f"| contact={int(torch.count_nonzero(result.active))} "
                    f"| Ki_eff={active_config.contact_ki:g}"
                )
        stable = 0
        for relaxation in range(1, config.max_settle_steps + 1):
            state, result = advance_with_ki_adaptation(
                state, context, target, target, friction_enabled=True,
                phase_label=f"load {load_step}/{config.num_load_steps}, relaxation {relaxation}",
            )
            equilibrium = _settled(result, active_config)
            dynamic_ratio: float | None = None
            if equilibrium:
                candidate_diagnostics = _diagnostics(
                    state[0], result.dynamic_force, rest_lengths, masses,
                    state[2], state[3], nodal_lengths, context, active_config, target,
                    prebalanced_reference, True, effective_ki_scale,
                )
                equilibrium = dynamic_balance_acceptable(
                    float(candidate_diagnostics["dynamic_remainder_n"]),
                    float(candidate_diagnostics["push_force_n"]),
                    active_config.settle_residual_tol,
                )
                dynamic_ratio = float(candidate_diagnostics["dynamic_to_push_ratio"])
            stable = stable + 1 if equilibrium else 0
            if relaxation % config.diagnostic_interval == 0 or stable >= config.settle_confirm_steps:
                free_speed = float(torch.linalg.vector_norm(result.velocities[:-2], dim=1).max())
                progress(
                    f"[RELAX {load_step:02d}/{config.num_load_steps:02d} | {relaxation:5d}] "
                    f"v={free_speed:.2e} m/s | R={result.info.residual_norm:.2e} N "
                    f"| contact={int(torch.count_nonzero(result.active))} | stable={stable} "
                    f"| Ki_eff={active_config.contact_ki:g}"
                    + ("" if dynamic_ratio is None else f" | dyn/push={dynamic_ratio:.2e}")
                )
            if stable >= config.settle_confirm_steps:
                break
        if result is None or stable < config.settle_confirm_steps:
            raise RuntimeError(f"load step {load_step} did not settle; friction history not committed")
        positions, velocities, integrals, previous_errors = state
        committed = commit_friction_increment(
            context, positions, result.active, result.candidate_slip
        )
        commit_count += 1
        completed += 1
        committed_context = begin_friction_increment(committed, positions)
        records.append(_diagnostics(
            positions, result.dynamic_force, rest_lengths, masses, integrals,
            previous_errors, nodal_lengths, committed_context, active_config, target,
            prebalanced_reference, True, effective_ki_scale,
        ))
        progress(
            f"[LOAD COMPLETE {load_step:02d}/{config.num_load_steps:02d}] "
            f"tip={target * 1.0e3:.3f} mm | relax={relaxation} "
            f"| Ki_eff={active_config.contact_ki:g}"
        )
    history = _history_arrays(records)
    paths = _write_outputs(config, history)
    return SimulationResult(
        config=config,
        history=history,
        completed_load_steps=completed,
        friction_commit_count=commit_count,
        config_path=paths[0],
        settled_path=paths[1],
        history_path=paths[2],
        figure_path=paths[3],
    )


def _write_comparison_figure(results: dict[float, SimulationResult], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(15, 8))
    axis3d = figure.add_subplot(2, 3, 1, projection="3d")
    for mu, result in sorted(results.items()):
        axis3d.plot(*result.history["positions"][-1].T, label=f"mu={mu:g}")
    axis3d.set_title("Final 3-D configuration")
    axis3d.legend()
    panels = (
        ("free_tip_advance_m", "free-tip advance [mm]", 1.0e3),
        ("push_force_n", "push force [N]", 1.0),
        ("friction_resultant_n", "friction resultant [N]", 1.0),
        ("contact_count", "contact nodes", 1.0),
        ("bending_energy_j", "bending energy [J]", 1.0),
    )
    for panel, (key, ylabel, scale) in enumerate(panels, start=2):
        axis = figure.add_subplot(2, 3, panel)
        for mu, result in sorted(results.items()):
            x = result.history["actuator_displacement_m"] * 1.0e3
            axis.plot(x, result.history[key] * scale, marker="o", label=f"mu={mu:g}")
        axis.set_xlabel("actuator [mm]")
        axis.set_ylabel(ylabel)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_comparison(
    base_config: StraightFreeInsertConfig,
    mus: Iterable[float] = (0.0, 1.0),
) -> ComparisonResult:
    root = Path(base_config.output_dir)
    results: dict[float, SimulationResult] = {}
    for value in mus:
        mu = float(value)
        label = f"mu{mu:.2f}".replace(".", "p")
        config = replace(
            base_config,
            friction_mu=mu,
            output_dir=str(root / label),
            run_name=f"{base_config.run_name}_{label}",
        )
        results[mu] = run_simulation(config)
    comparison_path = root / "comparison.png"
    root.mkdir(parents=True, exist_ok=True)
    _write_comparison_figure(results, comparison_path)
    return ComparisonResult(results=results, figure_path=comparison_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-nodes", type=int, default=250)
    parser.add_argument("--total-insertion", type=float, default=90.0e-3)
    parser.add_argument("--num-load-steps", type=int, default=20)
    parser.add_argument("--friction-mu", type=float, default=1.0)
    parser.add_argument("--output-dir", default="3d_npz/straight_free_insert")
    parser.add_argument("--single", action="store_true", help="run one mu instead of the default mu=0/1 comparison")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = StraightFreeInsertConfig(
        num_nodes=args.num_nodes,
        total_insertion=args.total_insertion,
        num_load_steps=args.num_load_steps,
        friction_mu=args.friction_mu,
        output_dir=args.output_dir,
    )
    if args.single:
        result = run_simulation(config)
        print(f"completed {result.completed_load_steps}/{config.num_load_steps}; {result.figure_path}")
    else:
        comparison = run_comparison(config)
        print(f"comparison complete: {comparison.figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
