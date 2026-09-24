"""Low-level Torch geometry utilities for staged soft-rod simulations.

This module is intentionally independent from solvers, plotting, command-line
interfaces, and model-specific energy assembly.  All core arrays are Torch
tensors so the simulation loop can stay on one tensor/autograd path.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def as_tensor(
    values,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """Convert values to a detached Torch tensor with the requested dtype/device."""
    return torch.as_tensor(values, dtype=dtype, device=device).clone().detach()


def to_numpy(values):
    """Detach a tensor-like value for plotting or legacy display boundaries."""
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return values


def validate_positions(positions: Tensor, name: str = "positions") -> Tensor:
    """Validate a 3D nodal centerline tensor."""
    if not isinstance(positions, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{name} must have shape (num_nodes, 3).")
    return positions


def compute_edge_geometry(
    positions: Tensor,
    min_length: float = 1e-12,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute edge vectors, edge lengths, and unit tangents."""
    positions = validate_positions(positions)
    edge_vectors = positions[1:] - positions[:-1]
    lengths = torch.linalg.norm(edge_vectors, dim=1)
    safe_lengths = torch.clamp(lengths, min=min_length)
    tangents = edge_vectors / safe_lengths[:, None]
    tangents = torch.where(
        (lengths > min_length)[:, None],
        tangents,
        torch.zeros_like(tangents),
    )
    return edge_vectors, lengths, tangents


def compute_edge_stiffness_from_EA(
    rest_lengths: Tensor,
    EA: float | Tensor,
    min_length: float = 1e-12,
) -> Tensor:
    """Return per-edge axial stiffness k_i = EA / L_i^0."""
    if torch.any(rest_lengths <= min_length):
        raise ValueError("All rest lengths must be positive.")
    EA_tensor = torch.as_tensor(EA, dtype=rest_lengths.dtype, device=rest_lengths.device)
    if torch.any(EA_tensor < 0.0):
        raise ValueError("EA must be non-negative.")
    return EA_tensor / rest_lengths


def compute_curvature_binormals(
    positions: Tensor,
    min_length: float = 1e-12,
) -> Tensor:
    """Return discrete curvature-binormal vectors at interior nodes."""
    positions = validate_positions(positions)
    if positions.shape[0] < 3:
        return positions.new_zeros((0, positions.shape[1]))

    _, lengths, tangents = compute_edge_geometry(positions, min_length)
    dot_products = torch.sum(tangents[:-1] * tangents[1:], dim=1)
    denominators = torch.clamp(1.0 + dot_products, min=min_length)
    cross_products = torch.linalg.cross(tangents[:-1], tangents[1:], dim=1)
    curvature_binormals = 2.0 * cross_products / denominators[:, None]

    valid_vertices = (lengths[:-1] > min_length) & (lengths[1:] > min_length)
    return torch.where(
        valid_vertices[:, None],
        curvature_binormals,
        torch.zeros_like(curvature_binormals),
    )


def compute_curvature_binormal_magnitudes(
    positions: Tensor,
    min_length: float = 1e-12,
) -> Tensor:
    """Return ||kappa_b|| at interior nodes."""
    curvature_binormals = compute_curvature_binormals(positions, min_length)
    return torch.linalg.norm(curvature_binormals, dim=1)


def compute_vertex_rest_lengths(
    rest_lengths: Tensor,
    min_length: float = 1e-12,
) -> Tensor:
    """Return Voronoi-style rest lengths around interior vertices."""
    if rest_lengths.numel() < 2:
        return rest_lengths.new_empty(0)
    vertex_lengths = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
    return torch.clamp(vertex_lengths, min=min_length)


def compute_node_arc_positions(rest_lengths: Tensor) -> Tensor:
    """Return rest-configuration arc-length positions at chain nodes."""
    return torch.cat((rest_lengths.new_zeros(1), torch.cumsum(rest_lengths, dim=0)))


def compute_edge_midpoint_arc_positions(rest_lengths: Tensor) -> Tensor:
    """Return rest-configuration arc-length positions at edge midpoints."""
    node_arc_positions = compute_node_arc_positions(rest_lengths)
    return node_arc_positions[:-1] + 0.5 * rest_lengths


def compute_vertex_arc_positions(rest_lengths: Tensor) -> Tensor:
    """Return rest-configuration arc-length positions at interior vertices."""
    node_arc_positions = compute_node_arc_positions(rest_lengths)
    return node_arc_positions[1:-1]
