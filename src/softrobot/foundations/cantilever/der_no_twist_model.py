"""Torch-native DER-style extensible centerline rod model without twist.

This module owns the phase-1 physical formulas: stretch energy, bending energy,
external potentials, kinetic energy, and elastic internal forces.  Forces are
computed from the total elastic energy with PyTorch autograd, so the simulation
loop no longer performs a NumPy -> Torch -> NumPy force-conversion cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .soft_robot_common import (
    Tensor,
    as_tensor,
    compute_curvature_binormal_magnitudes,
    compute_curvature_binormals,
    compute_edge_geometry,
    compute_edge_stiffness_from_EA,
    compute_vertex_rest_lengths,
    validate_positions,
)


@dataclass(frozen=True)
class RodRestState:
    """Reference geometry for a no-twist centerline rod."""

    positions: Tensor
    lengths: Tensor
    vertex_lengths: Tensor
    curvature_binormals: Tensor


@dataclass(frozen=True)
class RodParams:
    """Physical parameters used by the phase-1 rod model."""

    EA: float
    B: float
    node_masses: Tensor


def make_rest_state_from_positions(
    positions,
    lengths=None,
    curvature_binormals=None,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    min_length: float = 1e-12,
) -> RodRestState:
    """Build a Torch rest-state object from reference positions."""
    rest_positions = validate_positions(as_tensor(positions, dtype=dtype, device=device), "rest positions")
    if rest_positions.shape[0] < 2:
        raise ValueError("At least two rest nodes are required.")

    _, measured_lengths, _ = compute_edge_geometry(rest_positions, min_length)
    if lengths is None:
        rest_lengths = measured_lengths.detach().clone()
    else:
        rest_lengths = as_tensor(lengths, dtype=rest_positions.dtype, device=rest_positions.device)
        if rest_lengths.shape != measured_lengths.shape:
            raise ValueError("lengths must have one entry per edge.")

    if torch.any(rest_lengths <= min_length):
        raise ValueError("All rest lengths must be positive.")

    if curvature_binormals is None:
        rest_curvature_binormals = compute_curvature_binormals(rest_positions, min_length).detach().clone()
    else:
        rest_curvature_binormals = as_tensor(
            curvature_binormals,
            dtype=rest_positions.dtype,
            device=rest_positions.device,
        )
        expected_shape = (max(rest_positions.shape[0] - 2, 0), rest_positions.shape[1])
        if rest_curvature_binormals.shape != expected_shape:
            raise ValueError("curvature_binormals has an incompatible shape.")

    return RodRestState(
        positions=rest_positions,
        lengths=rest_lengths,
        vertex_lengths=compute_vertex_rest_lengths(rest_lengths, min_length),
        curvature_binormals=rest_curvature_binormals,
    )


def make_straight_rest_state(
    num_links: int,
    total_length: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> RodRestState:
    """Create a straight 3D rest-state chain with uniform link lengths."""
    if num_links < 1:
        raise ValueError("num_links must be at least 1.")
    if total_length <= 0.0:
        raise ValueError("total_length must be positive.")

    num_nodes = num_links + 1
    x_coordinates = torch.linspace(0.0, total_length, num_nodes, dtype=dtype, device=device)
    positions = torch.stack(
        (
            x_coordinates,
            torch.zeros(num_nodes, dtype=dtype, device=device),
            torch.zeros(num_nodes, dtype=dtype, device=device),
        ),
        dim=1,
    )
    lengths = torch.full((num_links,), total_length / num_links, dtype=dtype, device=device)
    return make_rest_state_from_positions(positions, lengths=lengths, dtype=dtype, device=device)


def make_uniform_node_masses(
    num_nodes: int,
    value: float,
    fixed_indices: tuple[int, ...] = (),
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """Create uniform nodal masses and set fixed-node masses to zero."""
    if num_nodes < 1:
        raise ValueError("num_nodes must be positive.")
    if value <= 0.0:
        raise ValueError("node mass value must be positive.")

    node_masses = torch.full((num_nodes,), value, dtype=dtype, device=device)
    for index in fixed_indices:
        if index < 0 or index >= num_nodes:
            raise ValueError("fixed_indices contains an out-of-range node index.")
        node_masses[index] = 0.0
    return node_masses


def compute_kappa_b_magnitudes(
    positions: Tensor,
    min_length: float = 1e-12,
) -> Tensor:
    """Return ||kappa_b||, not the turning angle theta."""
    return compute_curvature_binormal_magnitudes(positions, min_length)


def stretching_energy(positions: Tensor, rest: RodRestState, EA: float) -> Tensor:
    """Return extensible-rod stretching energy with k_i = EA / L_i^0."""
    positions = validate_positions(positions)
    _, current_lengths, _ = compute_edge_geometry(positions)
    extensions = current_lengths - rest.lengths.to(dtype=positions.dtype, device=positions.device)
    edge_stiffness = compute_edge_stiffness_from_EA(
        rest.lengths.to(dtype=positions.dtype, device=positions.device),
        EA,
    )
    return 0.5 * torch.sum(edge_stiffness * extensions**2)


def bending_energy(
    positions: Tensor,
    rest: RodRestState,
    B: float,
    min_length: float = 1e-12,
) -> Tensor:
    """Return centerline bending energy using kappa_b - kappa_b_rest."""
    positions = validate_positions(positions)
    if B == 0.0 or positions.shape[0] < 3:
        return positions.new_tensor(0.0)

    curvature_binormals = compute_curvature_binormals(positions, min_length)
    rest_curvature_binormals = rest.curvature_binormals.to(
        dtype=positions.dtype,
        device=positions.device,
    )
    if rest_curvature_binormals.shape != curvature_binormals.shape:
        raise ValueError("rest curvature_binormals has an incompatible shape.")

    vertex_lengths = torch.clamp(
        rest.vertex_lengths.to(dtype=positions.dtype, device=positions.device),
        min=min_length,
    )
    delta_kappa_b = curvature_binormals - rest_curvature_binormals
    return 0.5 * B * torch.sum(torch.sum(delta_kappa_b**2, dim=1) / vertex_lengths)


def total_elastic_energy(
    positions: Tensor,
    rest: RodRestState,
    params: RodParams,
) -> dict[str, Tensor]:
    """Compute stretching, bending, and total elastic energy."""
    stretch = stretching_energy(positions, rest, params.EA)
    bend = bending_energy(positions, rest, params.B)
    return {
        "stretching_energy": stretch,
        "bending_energy": bend,
        "elastic_energy": stretch + bend,
    }


def elastic_internal_forces(
    positions: Tensor,
    rest: RodRestState,
    params: RodParams,
) -> Tensor:
    """Return elastic internal forces as -grad(E_stretch + E_bend)."""
    positions_for_grad = positions.detach().clone().requires_grad_(True)
    energy = total_elastic_energy(positions_for_grad, rest, params)["elastic_energy"]
    (gradient,) = torch.autograd.grad(energy, positions_for_grad)
    return -gradient.detach()


def gravitational_potential(
    positions: Tensor,
    node_masses: Tensor,
    gravity: float,
) -> Tensor:
    """Return gravitational potential energy, sum(m_j * g * z_j)."""
    positions = validate_positions(positions)
    masses = node_masses.to(dtype=positions.dtype, device=positions.device)
    return torch.sum(masses * gravity * positions[:, 2])


def external_potential(
    positions: Tensor,
    external_forces: Tensor,
) -> Tensor:
    """Return external-force potential, -sum(F_j dot x_j)."""
    positions = validate_positions(positions)
    forces = external_forces.to(dtype=positions.dtype, device=positions.device)
    return -torch.sum(forces * positions)


def kinetic_energy(
    velocities: Tensor,
    node_masses: Tensor,
    free_indices: Tensor,
) -> Tensor:
    """Return kinetic energy over free nodes."""
    masses = node_masses.to(dtype=velocities.dtype, device=velocities.device)
    return 0.5 * torch.sum(
        masses[free_indices] * torch.sum(velocities[free_indices] ** 2, dim=1)
    )


def total_energy(
    positions: Tensor,
    velocities: Tensor,
    rest: RodRestState,
    params: RodParams,
    gravity: float,
    external_forces: Tensor,
    free_indices: Tensor,
) -> dict[str, Tensor]:
    """Compute elastic, potential, kinetic, and total energy values."""
    elastic = total_elastic_energy(positions, rest, params)
    gravity_energy = gravitational_potential(positions, params.node_masses, gravity)
    external_energy = external_potential(positions, external_forces)
    kinetic = kinetic_energy(velocities, params.node_masses, free_indices)
    total = (
        elastic["stretching_energy"]
        + elastic["bending_energy"]
        + gravity_energy
        + external_energy
        + kinetic
    )
    return {
        "stretching_energy": elastic["stretching_energy"],
        "bending_energy": elastic["bending_energy"],
        "gravitational_energy": gravity_energy,
        "external_energy": external_energy,
        "kinetic_energy": kinetic,
        "total_energy": total,
    }
