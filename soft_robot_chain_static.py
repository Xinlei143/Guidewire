"""Stretching-only DER-style dynamic simulation of a 2D soft robot chain.

The robot is modeled as 10 serial elastic links and 11 point nodes by default.
The left node is fixed, the right node is free, and only axial stretching is
implemented. This is not a full DER model because bending and twisting are not
included yet.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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


def compute_edge_geometry(
    positions: np.ndarray, min_length: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute edge vectors, edge lengths, and unit tangents."""
    edge_vectors = positions[1:] - positions[:-1]
    lengths = np.linalg.norm(edge_vectors, axis=1)
    tangents = np.zeros_like(edge_vectors)
    valid_edges = lengths > min_length
    tangents[valid_edges] = edge_vectors[valid_edges] / lengths[valid_edges, None]
    return edge_vectors, lengths, tangents


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


def compute_total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    rest_lengths: np.ndarray,
    stiffness: float,
    node_masses: np.ndarray,
    g: float,
    external_forces: np.ndarray,
    free_indices: np.ndarray,
) -> dict[str, float]:
    """Compute stretching, potential, kinetic, and total energy values."""
    stretch = stretching_energy(positions, rest_lengths, stiffness)
    gravity = gravitational_potential(positions, node_masses, g)
    external = external_potential(positions, external_forces)
    kinetic = 0.5 * float(
        np.sum(node_masses[free_indices] * np.sum(velocities[free_indices] ** 2, axis=1))
    )
    return {
        "stretching_energy": stretch,
        "gravitational_energy": gravity,
        "external_energy": external,
        "kinetic_energy": kinetic,
        "total_energy": stretch + gravity + external + kinetic,
    }


def compute_gravity_forces(
    node_masses: np.ndarray, g: float, fixed_indices: np.ndarray
) -> np.ndarray:
    """Return nodal gravity forces in the negative y direction."""
    gravity_forces = np.zeros((node_masses.size, 2), dtype=float)
    gravity_forces[:, 1] = -node_masses * g
    gravity_forces[fixed_indices] = 0.0
    return gravity_forces


def solve_dynamic_explicit(
    initial_positions: np.ndarray,
    rest_lengths: np.ndarray,
    stiffness: float,
    node_masses: np.ndarray,
    g: float,
    external_forces: np.ndarray,
    dt: float,
    steps: int,
    damping: float,
    convergence_tol: float,
    save_every: int,
    fixed_indices: tuple[int, ...] = (0,),
) -> tuple[np.ndarray, dict[str, np.ndarray | bool | int | None]]:
    """Run a stretching-only DER-style explicit/semi-implicit time integration.

    The current geometry is updated each step, stretching forces are assembled
    analytically, damping is used only for time integration, and convergence is
    checked with the undamped mechanical residual on free nodes.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if damping < 0.0:
        raise ValueError("damping must be non-negative.")
    if convergence_tol <= 0.0:
        raise ValueError("convergence_tol must be positive.")
    if save_every < 1:
        raise ValueError("save_every must be at least 1.")

    num_nodes = initial_positions.shape[0]
    fixed_indices_array = np.array(fixed_indices, dtype=int)
    if np.any(fixed_indices_array < 0) or np.any(fixed_indices_array >= num_nodes):
        raise ValueError("fixed_indices contains an out-of-range node index.")

    all_indices = np.arange(num_nodes)
    free_indices = np.setdiff1d(all_indices, fixed_indices_array)
    if np.any(node_masses[free_indices] <= 0.0):
        raise ValueError("All free nodes must have positive mass.")

    positions = initial_positions.copy()
    velocities = np.zeros_like(initial_positions)
    fixed_positions = initial_positions[fixed_indices_array].copy()
    gravity_forces = compute_gravity_forces(node_masses, g, fixed_indices_array)

    history: dict[str, list | bool | int | None] = {
        "positions": [],
        "velocities": [],
        "time": [],
        "total_energy": [],
        "stretching_energy": [],
        "gravitational_energy": [],
        "external_energy": [],
        "kinetic_energy": [],
        "max_displacement": [],
        "max_velocity_norm": [],
        "max_force_norm": [],
        "converged": False,
        "converged_step": None,
        "unstable": False,
        "unstable_step": None,
    }
    last_saved_step: int | None = None

    def append_history(step: int, residual_forces: np.ndarray | None) -> None:
        nonlocal last_saved_step
        if residual_forces is None:
            max_force_norm = np.nan
        else:
            max_force_norm = float(np.max(np.linalg.norm(residual_forces[free_indices], axis=1)))
        max_velocity_norm = float(np.max(np.linalg.norm(velocities[free_indices], axis=1)))
        max_displacement = float(
            np.max(np.linalg.norm(positions - initial_positions, axis=1))
        )
        energies = compute_total_energy(
            positions,
            velocities,
            rest_lengths,
            stiffness,
            node_masses,
            g,
            external_forces,
            free_indices,
        )

        history["positions"].append(positions.copy())
        history["velocities"].append(velocities.copy())
        history["time"].append(step * dt)
        history["total_energy"].append(energies["total_energy"])
        history["stretching_energy"].append(energies["stretching_energy"])
        history["gravitational_energy"].append(energies["gravitational_energy"])
        history["external_energy"].append(energies["external_energy"])
        history["kinetic_energy"].append(energies["kinetic_energy"])
        history["max_displacement"].append(max_displacement)
        history["max_velocity_norm"].append(max_velocity_norm)
        history["max_force_norm"].append(max_force_norm)
        last_saved_step = step

    def append_unstable_history(step: int) -> None:
        nonlocal last_saved_step
        history["positions"].append(positions.copy())
        history["velocities"].append(velocities.copy())
        history["time"].append(step * dt)
        history["total_energy"].append(np.nan)
        history["stretching_energy"].append(np.nan)
        history["gravitational_energy"].append(np.nan)
        history["external_energy"].append(np.nan)
        history["kinetic_energy"].append(np.nan)
        history["max_displacement"].append(np.nan)
        history["max_velocity_norm"].append(np.nan)
        history["max_force_norm"].append(np.nan)
        last_saved_step = step

    internal_forces = compute_stretching_forces(positions, rest_lengths, stiffness)
    residual_forces = internal_forces + external_forces + gravity_forces
    append_history(0, residual_forces)

    final_step = 0
    for step in range(1, steps + 1):
        internal_forces = compute_stretching_forces(positions, rest_lengths, stiffness)
        residual_forces = internal_forces + external_forces + gravity_forces
        damping_forces = -damping * velocities
        total_forces = residual_forces + damping_forces

        # Semi-implicit Euler: update free-node velocities first, then positions.
        velocities[free_indices] += (
            dt * total_forces[free_indices] / node_masses[free_indices, None]
        )
        positions[free_indices] += dt * velocities[free_indices]

        positions[fixed_indices_array] = fixed_positions
        velocities[fixed_indices_array] = 0.0
        final_step = step

        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            history["unstable"] = True
            history["unstable_step"] = step
            if last_saved_step != step:
                append_unstable_history(step)
            print("Simulation became unstable. Try reducing --dt or increasing --damping.")
            break

        internal_forces = compute_stretching_forces(positions, rest_lengths, stiffness)
        residual_forces = internal_forces + external_forces + gravity_forces
        max_velocity_norm = float(np.max(np.linalg.norm(velocities[free_indices], axis=1)))
        max_force_norm = float(np.max(np.linalg.norm(residual_forces[free_indices], axis=1)))

        if step % save_every == 0:
            append_history(step, residual_forces)

        if max_velocity_norm < convergence_tol and max_force_norm < convergence_tol:
            history["converged"] = True
            history["converged_step"] = step
            break

    if not history["unstable"] and last_saved_step != final_step:
        internal_forces = compute_stretching_forces(positions, rest_lengths, stiffness)
        residual_forces = internal_forces + external_forces + gravity_forces
        append_history(final_step, residual_forces)

    for key in (
        "positions",
        "velocities",
        "time",
        "total_energy",
        "stretching_energy",
        "gravitational_energy",
        "external_energy",
        "kinetic_energy",
        "max_displacement",
        "max_velocity_norm",
        "max_force_norm",
    ):
        history[key] = np.asarray(history[key])

    return positions, history


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
    ax.set_title("2D Soft Robot Chain Dynamic Equilibrium")
    ax.axis("equal")
    ax.grid(True, alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.show()


def animate_chain_evolution(
    initial_positions: np.ndarray,
    history: dict[str, np.ndarray | bool | int | None],
    external_forces: np.ndarray,
    force_node: int,
    animation_output: str | None = None,
    show: bool = False,
) -> None:
    """Animate saved chain states and optionally save the animation as a GIF."""
    positions_history = history["positions"]
    time_history = history["time"]
    all_positions = np.concatenate((positions_history, initial_positions[None, :, :]), axis=0)
    min_xy = np.min(all_positions.reshape(-1, 2), axis=0)
    max_xy = np.max(all_positions.reshape(-1, 2), axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    margin = 0.12 * max(float(np.max(span)), 1e-3)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        initial_positions[:, 0],
        initial_positions[:, 1],
        "o--",
        color="tab:blue",
        label="Initial chain",
    )
    current_line, = ax.plot([], [], "o-", color="tab:orange", label="Current chain")
    fixed_marker = ax.scatter(
        initial_positions[0, 0],
        initial_positions[0, 1],
        s=90,
        c="black",
        marker="s",
        label="Fixed node",
    )
    force_marker = ax.scatter(
        [],
        [],
        s=100,
        facecolors="none",
        edgecolors="tab:red",
        linewidths=2,
        label="Force node",
    )

    force_vector = external_forces[force_node]
    force_norm = np.linalg.norm(force_vector)
    force_scale = 0.15 * max(float(np.max(span)), 1e-3) / max(force_norm, 1e-12)
    force_quiver = ax.quiver(
        [initial_positions[force_node, 0]],
        [initial_positions[force_node, 1]],
        [force_vector[0] * force_scale],
        [force_vector[1] * force_scale],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="tab:red",
        width=0.006,
        label="External force",
    )

    ax.set_xlim(min_xy[0] - margin, max_xy[0] + margin)
    ax.set_ylim(min_xy[1] - margin, max_xy[1] + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.legend(loc="best")

    def update(frame_index: int):
        positions = positions_history[frame_index]
        current_line.set_data(positions[:, 0], positions[:, 1])
        force_marker.set_offsets(positions[force_node : force_node + 1])
        force_quiver.set_offsets(positions[force_node : force_node + 1])
        force_quiver.set_UVC(
            [force_vector[0] * force_scale],
            [force_vector[1] * force_scale],
        )
        ax.set_title(f"Stretching-only DER-style dynamics, t = {time_history[frame_index]:.4f} s")
        return current_line, fixed_marker, force_marker, force_quiver

    animation = FuncAnimation(
        fig,
        update,
        frames=len(positions_history),
        interval=40,
        blit=False,
    )

    if animation_output:
        output_path = Path(animation_output)
        if output_path.suffix.lower() != ".gif":
            output_path = output_path.with_suffix(".gif")
            print(f"Animation output is saved as GIF: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(output_path, writer="pillow", fps=25)
        print(f"Animation saved to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_time_history(history: dict[str, np.ndarray | bool | int | None]) -> None:
    """Plot energy and convergence-related time histories."""
    time = history["time"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(time, history["total_energy"], label="Total energy")
    axes[0].plot(time, history["stretching_energy"], label="Stretching energy")
    axes[0].plot(time, history["kinetic_energy"], label="Kinetic energy")
    axes[0].set_ylabel("Energy (J)")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend()

    axes[1].plot(time, history["max_displacement"], color="tab:green")
    axes[1].set_ylabel("Max displacement (m)")
    axes[1].grid(True, alpha=0.35)

    axes[2].plot(time, history["max_velocity_norm"], color="tab:purple", label="Max velocity")
    axes[2].plot(time, history["max_force_norm"], color="tab:red", label="Max residual force")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Convergence metrics")
    axes[2].grid(True, alpha=0.35)
    axes[2].legend()

    fig.suptitle("Dynamic Simulation History")
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
        description=(
            "Stretching-only DER-style explicit dynamics for a 2D soft robot "
            "chain. Bending and twisting are not implemented."
        )
    )
    parser.add_argument("--num-links", type=int, default=10, help="Number of serial links.")
    parser.add_argument("--total-length", type=float, default=1.0, help="Initial total length in meters.")
    parser.add_argument("--stiffness", type=float, default=1000.0, help="Axial spring stiffness in N/m.")
    parser.add_argument("--node-mass", type=float, default=0.01, help="Mass of each free node in kg.")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravitational acceleration in m/s^2.")
    parser.add_argument("--force-node", type=int, default=10, help="Node index where the external force is applied.")
    parser.add_argument("--force-x", type=float, default=1.0, help="External force x component in N.")
    parser.add_argument("--force-y", type=float, default=0.0, help="External force y component in N.")
    parser.add_argument("--dt", type=float, default=1e-4, help="Time step size in seconds.")
    parser.add_argument("--steps", type=int, default=20000, help="Maximum number of time steps.")
    parser.add_argument("--damping", type=float, default=2.0, help="Nodal viscous damping coefficient.")
    parser.add_argument(
        "--convergence-tol",
        type=float,
        default=1e-5,
        help="Tolerance for free-node max velocity and undamped residual force.",
    )
    parser.add_argument("--save-every", type=int, default=10, help="Save history every N steps.")
    parser.add_argument("--animate", action="store_true", help="Display the chain evolution animation.")
    parser.add_argument("--animation-output", type=str, default=None, help="Path for saving animation GIF.")
    parser.add_argument("--plot-history", action="store_true", help="Display energy and convergence history plots.")
    parser.add_argument("--no-plot", action="store_true", help="Disable interactive plot windows.")
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

    deformed_positions, history = solve_dynamic_explicit(
        initial_positions=initial_positions,
        rest_lengths=rest_lengths,
        stiffness=args.stiffness,
        node_masses=node_masses,
        g=args.gravity,
        external_forces=external_forces,
        dt=args.dt,
        steps=args.steps,
        damping=args.damping,
        convergence_tol=args.convergence_tol,
        save_every=args.save_every,
        fixed_indices=(0,),
    )

    if history["unstable"]:
        print(f"Simulation stopped at unstable step {history['unstable_step']}.")
    elif history["converged"]:
        print(f"Simulation converged at step {history['converged_step']}.")
    else:
        print(f"Simulation did not converge within {args.steps} steps.")

    print_displacements(initial_positions, deformed_positions, args.force_node)

    if not args.no_plot:
        visualize_chain(initial_positions, deformed_positions, external_forces)

    if args.plot_history:
        if args.no_plot:
            print("--plot-history was requested, but --no-plot is enabled; skipping interactive history plots.")
        else:
            plot_time_history(history)

    show_animation = args.animate and not args.no_plot
    if args.animate and args.no_plot:
        print("--animate was requested, but --no-plot is enabled; skipping interactive animation.")
    if args.animation_output or show_animation:
        animate_chain_evolution(
            initial_positions=initial_positions,
            history=history,
            external_forces=external_forces,
            force_node=args.force_node,
            animation_output=args.animation_output,
            show=show_animation,
        )


if __name__ == "__main__":
    main()
