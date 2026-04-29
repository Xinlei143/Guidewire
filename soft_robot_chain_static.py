"""Static 2D soft robot chain with axial stretching springs only.

The robot is modeled as 10 serial elastic links and 11 point nodes by default.
The left node is fixed, all other nodes are optimized by minimizing total
potential energy.
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
from scipy.optimize import minimize

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib.pyplot as plt


def initialize_chain(num_links: int, total_length: float) -> tuple[np.ndarray, np.ndarray]:
    """Create a straight horizontal chain and uniform link rest lengths."""
    if num_links < 1:
        raise ValueError("num_links must be at least 1.")
    if total_length <= 0:
        raise ValueError("total_length must be positive.")

    num_nodes = num_links + 1
    link_length = total_length / num_links
    x_coordinates = np.linspace(0.0, total_length, num_nodes)
    positions = np.column_stack((x_coordinates, np.zeros(num_nodes)))
    rest_lengths = np.full(num_links, link_length)
    return positions, rest_lengths


def stretching_energy(
    positions: np.ndarray, rest_lengths: np.ndarray, stiffness: float
) -> float:
    """Return sum of axial spring stretching energies."""
    # Axial spring stretching energy is computed here for each link.
    link_vectors = positions[1:] - positions[:-1]
    current_lengths = np.linalg.norm(link_vectors, axis=1)
    extensions = current_lengths - rest_lengths
    return 0.5 * stiffness * float(np.sum(extensions**2))


def gravitational_potential(
    positions: np.ndarray, node_masses: np.ndarray, g: float
) -> float:
    """Return gravitational potential energy, sum(m_j * g * y_j)."""
    # Gravity is added here through the vertical coordinate of each node.
    y_coordinates = positions[:, 1]
    return float(np.sum(node_masses * g * y_coordinates))


def external_potential(positions: np.ndarray, external_forces: np.ndarray) -> float:
    """Return external-force potential, -sum(F_j dot x_j)."""
    return -float(np.sum(external_forces * positions))


def total_potential_energy(
    free_dofs: np.ndarray,
    fixed_positions: np.ndarray,
    free_indices: np.ndarray,
    rest_lengths: np.ndarray,
    stiffness: float,
    node_masses: np.ndarray,
    g: float,
    external_forces: np.ndarray,
) -> float:
    """Reconstruct all node positions and evaluate total potential energy."""
    positions = fixed_positions.copy()
    positions[free_indices] = free_dofs.reshape((-1, 2))
    return (
        stretching_energy(positions, rest_lengths, stiffness)
        + gravitational_potential(positions, node_masses, g)
        + external_potential(positions, external_forces)
    )


def solve_static_equilibrium(
    initial_positions: np.ndarray,
    rest_lengths: np.ndarray,
    stiffness: float,
    node_masses: np.ndarray,
    g: float,
    external_forces: np.ndarray,
    fixed_indices: tuple[int, ...] = (0,),
    method: str = "L-BFGS-B",
) -> tuple[np.ndarray, object]:
    """Minimize total potential energy subject to fixed-node constraints."""
    num_nodes = initial_positions.shape[0]
    all_indices = np.arange(num_nodes)
    fixed_indices_array = np.array(fixed_indices, dtype=int)

    if np.any(fixed_indices_array < 0) or np.any(fixed_indices_array >= num_nodes):
        raise ValueError("fixed_indices contains an out-of-range node index.")

    # Fixed boundary condition is enforced by excluding fixed nodes from the
    # optimization variables. Node 0 stays at its initial position.
    free_indices = np.setdiff1d(all_indices, fixed_indices_array)
    fixed_positions = initial_positions.copy()
    initial_free_dofs = initial_positions[free_indices].reshape(-1)

    result = minimize(
        total_potential_energy,
        initial_free_dofs,
        args=(
            fixed_positions,
            free_indices,
            rest_lengths,
            stiffness,
            node_masses,
            g,
            external_forces,
        ),
        method=method,
        options={"gtol": 1e-10, "ftol": 1e-14, "maxiter": 10000, "maxfun": 200000},
    )

    deformed_positions = fixed_positions.copy()
    deformed_positions[free_indices] = result.x.reshape((-1, 2))
    return deformed_positions, result


def visualize_chain(
    initial_positions: np.ndarray,
    deformed_positions: np.ndarray,
    external_forces: np.ndarray,
) -> None:
    """Plot undeformed chain, deformed chain, external force, and displacements."""
    displacements = deformed_positions - initial_positions

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        initial_positions[:, 0],
        initial_positions[:, 1],
        "o--",
        label="Undeformed chain",
        color="tab:blue",
    )
    ax.plot(
        deformed_positions[:, 0],
        deformed_positions[:, 1],
        "o-",
        label="Deformed chain",
        color="tab:orange",
    )

    ax.quiver(
        initial_positions[:, 0],
        initial_positions[:, 1],
        displacements[:, 0],
        displacements[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="tab:green",
        width=0.004,
        label="Node displacement",
    )

    force_nodes = np.where(np.linalg.norm(external_forces, axis=1) > 0.0)[0]
    if force_nodes.size:
        force_scale = 0.1 / max(np.max(np.linalg.norm(external_forces[force_nodes], axis=1)), 1e-12)
        ax.quiver(
            deformed_positions[force_nodes, 0],
            deformed_positions[force_nodes, 1],
            external_forces[force_nodes, 0] * force_scale,
            external_forces[force_nodes, 1] * force_scale,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="tab:red",
            width=0.006,
            label="External force",
        )

    ax.scatter(initial_positions[0, 0], initial_positions[0, 1], s=90, c="black", marker="s", label="Fixed node")
    ax.text(
        0.02,
        0.95,
        "Gravity acts downward on free nodes",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title("2D Soft Robot Chain Static Equilibrium")
    ax.axis("equal")
    ax.grid(True, alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.show()


def print_displacements(
    initial_positions: np.ndarray,
    deformed_positions: np.ndarray,
    force_node: int,
) -> None:
    """Print nodal displacement vectors and selected-node displacement magnitude."""
    displacements = deformed_positions - initial_positions

    print("Node displacements:")
    for node_index, displacement in enumerate(displacements):
        magnitude = np.linalg.norm(displacement)
        print(
            f"  Node {node_index:2d}: "
            f"ux = {displacement[0]: .8e} m, "
            f"uy = {displacement[1]: .8e} m, "
            f"|u| = {magnitude: .8e} m"
        )

    force_node_magnitude = np.linalg.norm(displacements[force_node])
    print(
        f"\nTotal displacement magnitude of force-applied node {force_node}: "
        f"{force_node_magnitude:.8e} m"
    )


def build_external_forces(
    num_nodes: int,
    force_node: int,
    force_vector: tuple[float, float],
    fixed_node: int = 0,
) -> np.ndarray:
    """Create the nodal external-force array."""
    if force_node == fixed_node:
        raise ValueError("External force cannot be applied to the fixed left node.")
    if force_node < 0 or force_node >= num_nodes:
        raise ValueError(f"force_node must be between 0 and {num_nodes - 1}.")

    external_forces = np.zeros((num_nodes, 2), dtype=float)
    # The user-configurable additional external force is added here.
    external_forces[force_node] = np.array(force_vector, dtype=float)
    return external_forces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static equilibrium of a 2D soft robot chain with axial springs."
    )
    parser.add_argument("--num-links", type=int, default=10, help="Number of serial links.")
    parser.add_argument("--total-length", type=float, default=1.0, help="Initial total length in meters.")
    parser.add_argument("--stiffness", type=float, default=1000.0, help="Axial spring stiffness in N/m.")
    parser.add_argument("--node-mass", type=float, default=0.01, help="Mass of each free node in kg.")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravitational acceleration in m/s^2.")
    parser.add_argument("--force-node", type=int, default=10, help="Node index where the external force is applied.")
    parser.add_argument("--force-x", type=float, default=1.0, help="External force x component in N.")
    parser.add_argument("--force-y", type=float, default=0.0, help="External force y component in N.")
    parser.add_argument("--no-plot", action="store_true", help="Skip visualization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    initial_positions, rest_lengths = initialize_chain(args.num_links, args.total_length)
    num_nodes = initial_positions.shape[0]
    node_masses = np.full(num_nodes, args.node_mass)
    node_masses[0] = 0.0
    external_forces = build_external_forces(
        num_nodes,
        args.force_node,
        (args.force_x, args.force_y),
        fixed_node=0,
    )

    deformed_positions, result = solve_static_equilibrium(
        initial_positions=initial_positions,
        rest_lengths=rest_lengths,
        stiffness=args.stiffness,
        node_masses=node_masses,
        g=args.gravity,
        external_forces=external_forces,
        fixed_indices=(0,),
    )

    if not result.success:
        print(f"Warning: optimizer did not report convergence: {result.message}")

    print_displacements(initial_positions, deformed_positions, args.force_node)

    if not args.no_plot:
        visualize_chain(initial_positions, deformed_positions, external_forces)


if __name__ == "__main__":
    main()
