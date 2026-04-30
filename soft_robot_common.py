"""Shared utility functions for the staged soft robot chain scripts.

This module intentionally contains only stable, low-level helpers used by more
than one physical model. Model-specific solvers, energy assembly, visualization,
and command-line interfaces stay in the stage scripts.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())


def load_matplotlib():
    """Import Matplotlib only when plotting or animation is requested."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Matplotlib is required for plotting or animation. "
            "Run with --no-plot, or install matplotlib."
        ) from exc
    return plt, FuncAnimation


def compute_edge_geometry(
    positions: np.ndarray,
    min_length: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute edge vectors, edge lengths, and unit tangents."""
    edge_vectors = positions[1:] - positions[:-1]
    lengths = np.linalg.norm(edge_vectors, axis=1)
    tangents = np.zeros_like(edge_vectors)
    valid_edges = lengths > min_length
    tangents[valid_edges] = edge_vectors[valid_edges] / lengths[valid_edges, None]
    return edge_vectors, lengths, tangents


def stretching_energy(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    stiffness: float,
) -> float:
    """Return sum of axial spring stretching energies."""
    edge_vectors = positions[1:] - positions[:-1]
    current_lengths = np.linalg.norm(edge_vectors, axis=1)
    extensions = current_lengths - rest_lengths
    return 0.5 * stiffness * float(np.sum(extensions**2))


def compute_stretching_forces(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    stiffness: float,
    min_length: float = 1e-12,
) -> np.ndarray:
    """Assemble nodal forces from the negative gradient of stretching energy."""
    _, current_lengths, tangents = compute_edge_geometry(positions, min_length)
    extensions = current_lengths - rest_lengths
    edge_forces = stiffness * extensions[:, None] * tangents

    forces = np.zeros_like(positions)
    forces[:-1] += edge_forces
    forces[1:] -= edge_forces
    return forces


def gravitational_potential(
    positions: np.ndarray,
    node_masses: np.ndarray,
    g: float,
) -> float:
    """Return gravitational potential energy, sum(m_j * g * z_j)."""
    z_coordinates = positions[:, 2]
    return float(np.sum(node_masses * g * z_coordinates))


def external_potential(
    positions: np.ndarray,
    external_forces: np.ndarray,
) -> float:
    """Return external-force potential, -sum(F_j dot x_j)."""
    return -float(np.sum(external_forces * positions))


def compute_vertex_rest_lengths(
    rest_lengths: np.ndarray,
    min_length: float = 1e-12,
) -> np.ndarray:
    """Return Voronoi-style rest lengths around interior vertices."""
    if rest_lengths.size < 2:
        return np.empty(0)
    vertex_lengths = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
    return np.maximum(vertex_lengths, min_length)


def compute_node_arc_positions(rest_lengths: np.ndarray) -> np.ndarray:
    """Return rest-configuration arc-length positions at chain nodes."""
    return np.concatenate(([0.0], np.cumsum(rest_lengths)))


def compute_edge_midpoint_arc_positions(rest_lengths: np.ndarray) -> np.ndarray:
    """Return rest-configuration arc-length positions at edge midpoints."""
    node_arc_positions = compute_node_arc_positions(rest_lengths)
    return node_arc_positions[:-1] + 0.5 * rest_lengths


def compute_vertex_arc_positions(rest_lengths: np.ndarray) -> np.ndarray:
    """Return rest-configuration arc-length positions at interior vertices."""
    node_arc_positions = compute_node_arc_positions(rest_lengths)
    return node_arc_positions[1:-1]
