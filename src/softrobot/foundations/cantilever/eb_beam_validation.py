"""Euler-Bernoulli cantilever reference solution for validation plots."""

from __future__ import annotations

import torch

from .soft_robot_common import Tensor, validate_positions


def eb_tip_load_skip_reason(
    *,
    force_node: int,
    num_nodes: int,
    force_vector: Tensor,
    B: float,
    gravity: float,
    tolerance: float = 1e-12,
) -> str | None:
    """Return why the EB tip-load comparison is unavailable, or None if valid."""
    if B <= 0.0:
        return "B must be positive for the Euler-Bernoulli beam reference."
    if force_node < 2 or force_node >= num_nodes:
        return "the Euler-Bernoulli reference requires a free point-load node."
    if abs(gravity) > tolerance:
        return "gravity is not included in the Euler-Bernoulli point-load reference."
    if abs(float(force_vector[0])) > tolerance:
        return "axial force-x is not included in the Euler-Bernoulli transverse-load reference."
    return None


def euler_bernoulli_tip_load_positions(
    initial_positions: Tensor,
    force_vector: Tensor,
    B: float,
    fixed_indices: tuple[int, ...] = (0, 1),
    min_length: float = 1e-12,
) -> Tensor:
    """Return small-deflection EB cantilever positions for a transverse tip load."""
    return euler_bernoulli_point_load_positions(
        initial_positions, force_vector, B, fixed_indices=fixed_indices,
        min_length=min_length,
    )


def euler_bernoulli_point_load_positions(
    initial_positions: Tensor,
    force_vector: Tensor,
    B: float,
    *,
    force_node: int = -1,
    fixed_indices: tuple[int, ...] = (0, 1),
    min_length: float = 1e-12,
) -> Tensor:
    """Small-deflection, zero-gravity EB shape for a transverse nodal point load."""
    positions = validate_positions(initial_positions).detach().clone()
    if positions.shape[0] < 3:
        raise ValueError("At least three nodes are required for the node-1 clamped EB reference.")
    if len(fixed_indices) < 2:
        raise ValueError("fixed_indices must include node 0 and node 1 for this EB reference.")
    if B <= 0.0:
        raise ValueError("B must be positive.")

    clamp_index = fixed_indices[1]
    if clamp_index < 0:
        clamp_index += positions.shape[0]
    if clamp_index < 0 or clamp_index >= positions.shape[0] - 1:
        raise ValueError("The EB clamp node must be an interior node before the tip.")
    if force_node < 0:
        force_node += positions.shape[0]
    if force_node <= clamp_index or force_node >= positions.shape[0]:
        raise ValueError("force_node must be after the clamp and within the rod.")

    force = torch.as_tensor(force_vector, dtype=positions.dtype, device=positions.device)
    if force.shape != (3,):
        raise ValueError("force_vector must have shape (3,).")

    clamp_x = positions[clamp_index, 0]
    load_distance = positions[force_node, 0] - clamp_x
    if load_distance <= min_length or positions[-1, 0] - clamp_x <= min_length:
        raise ValueError("The EB length from clamp to load and tip must be positive.")

    s = torch.clamp(positions[:, 0] - clamp_x, min=0.0)
    transverse_shape = torch.where(
        s <= load_distance,
        s**2 * (3.0 * load_distance - s) / (6.0 * B),
        load_distance**2 * (3.0 * s - load_distance) / (6.0 * B),
    )

    displacements = torch.zeros_like(positions)
    displacements[:, 1] = force[1] * transverse_shape
    displacements[:, 2] = force[2] * transverse_shape
    displacements[: clamp_index + 1] = 0.0
    return positions + displacements
