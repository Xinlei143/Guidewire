"""Stretching + bending + torsion dynamic simulation of a 3D soft robot chain.

This third-stage script intentionally lives separately from the 2D stretching
and 2D stretching + bending scripts. The centerline uses 3D nodal positions,
while torsion is represented by one scalar twist angle per edge. The torsion
model is intentionally simple: it penalizes twist-angle differences between
neighboring edges and integrates twist angles with their own angular inertia.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soft_robot_common import (
    compute_edge_geometry,
    compute_edge_midpoint_arc_positions,
    compute_stretching_forces,
    compute_vertex_arc_positions,
    compute_vertex_rest_lengths,
    external_potential,
    gravitational_potential,
    load_matplotlib,
    stretching_energy,
)


@dataclass
class RodState:
    """Minimal 3D rod state for the torsion stage."""

    positions: np.ndarray
    velocities: np.ndarray
    twist_angles: np.ndarray
    twist_velocities: np.ndarray


def initialize_chain(
    num_links: int,
    total_length: float,
    initial_tip_twist: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a straight 3D chain, uniform rest lengths, and edge twists."""
    if num_links < 1:
        raise ValueError("num_links must be at least 1.")
    if total_length <= 0:
        raise ValueError("total_length must be positive.")

    num_nodes = num_links + 1
    link_length = total_length / num_links
    x_coordinates = np.linspace(0.0, total_length, num_nodes)
    positions = np.column_stack(
        (x_coordinates, np.zeros(num_nodes), np.zeros(num_nodes))
    )
    rest_lengths = np.full(num_links, link_length)
    twist_angles = np.zeros(num_links, dtype=float)
    if num_links > 1:
        twist_angles[-1] = initial_tip_twist
    return positions, rest_lengths, twist_angles


def compute_bending_angles(
    positions: np.ndarray,
    min_length: float = 1e-12,
) -> np.ndarray:
    """Return discrete curvature-binormal magnitudes between neighboring edges."""
    if positions.shape[0] < 3:
        return np.empty(0)

    _, lengths, tangents = compute_edge_geometry(positions, min_length)
    dot_products = np.sum(tangents[:-1] * tangents[1:], axis=1)
    denominators = np.maximum(1.0 + dot_products, min_length)
    cross_products = np.cross(tangents[:-1], tangents[1:])
    curvature_binormals = 2.0 * cross_products / denominators[:, None]
    curvature_magnitudes = np.linalg.norm(curvature_binormals, axis=1)

    valid_vertices = (lengths[:-1] > min_length) & (lengths[1:] > min_length)
    curvature_magnitudes[~valid_vertices] = 0.0
    return curvature_magnitudes


def bending_energy(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    stiffness: float,
) -> float:
    """Return 3D centerline bending energy with zero rest curvature."""
    if stiffness == 0.0 or positions.shape[0] < 3:
        return 0.0

    angles = compute_bending_angles(positions)
    vertex_lengths = compute_vertex_rest_lengths(rest_lengths)
    return 0.5 * stiffness * float(np.sum(angles**2 / vertex_lengths))


def compute_bending_forces(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    stiffness: float,
    finite_difference_step: float = 1e-6,
) -> np.ndarray:
    """Assemble bending forces as the negative numerical energy gradient."""
    if stiffness == 0.0 or positions.shape[0] < 3:
        return np.zeros_like(positions)
    if finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be positive.")

    forces = np.zeros_like(positions)
    for node_index in range(positions.shape[0]):
        for coordinate_index in range(positions.shape[1]):
            plus_positions = positions.copy()
            minus_positions = positions.copy()
            plus_positions[node_index, coordinate_index] += finite_difference_step
            minus_positions[node_index, coordinate_index] -= finite_difference_step
            plus_energy = bending_energy(plus_positions, rest_lengths, stiffness)
            minus_energy = bending_energy(minus_positions, rest_lengths, stiffness)
            forces[node_index, coordinate_index] = -(
                plus_energy - minus_energy
            ) / (2.0 * finite_difference_step)

    return forces


def torsion_energy(
    twist_angles: np.ndarray,
    rest_lengths: np.ndarray,
    stiffness: float,
) -> float:
    """Return torsional energy from twist-angle differences between edges."""
    if stiffness == 0.0 or twist_angles.size < 2:
        return 0.0

    twist_differences = compute_twist_angle_differences(twist_angles)
    vertex_lengths = compute_vertex_rest_lengths(rest_lengths)
    return 0.5 * stiffness * float(np.sum(twist_differences**2 / vertex_lengths))


def compute_twist_angle_differences(twist_angles: np.ndarray) -> np.ndarray:
    """Return twist-angle differences between neighboring edges."""
    if twist_angles.size < 2:
        return np.empty(0)
    return twist_angles[1:] - twist_angles[:-1]


def compute_torsion_torques(
    twist_angles: np.ndarray,
    rest_lengths: np.ndarray,
    stiffness: float,
) -> np.ndarray:
    """Return edge torques from the negative gradient of torsional energy."""
    torques = np.zeros_like(twist_angles)
    if stiffness == 0.0 or twist_angles.size < 2:
        return torques

    vertex_lengths = compute_vertex_rest_lengths(rest_lengths)
    twist_differences = compute_twist_angle_differences(twist_angles)
    gradient_terms = stiffness * twist_differences / vertex_lengths
    torques[:-1] += gradient_terms
    torques[1:] -= gradient_terms
    return torques


def compute_gravity_forces(
    node_masses: np.ndarray,
    g: float,
    fixed_indices: np.ndarray,
) -> np.ndarray:
    """Return 3D nodal gravity forces in the negative z direction."""
    gravity_forces = np.zeros((node_masses.size, 3), dtype=float)
    gravity_forces[:, 2] = -node_masses * g
    gravity_forces[fixed_indices] = 0.0
    return gravity_forces


def compute_internal_forces(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    stretch_stiffness: float,
    bend_stiffness: float,
) -> np.ndarray:
    """Return the sum of stretching and bending nodal forces."""
    return compute_stretching_forces(
        positions,
        rest_lengths,
        stretch_stiffness,
    ) + compute_bending_forces(
        positions,
        rest_lengths,
        bend_stiffness,
    )


def compute_total_energy(
    state: RodState,
    rest_lengths: np.ndarray,
    stretch_stiffness: float,
    bend_stiffness: float,
    twist_stiffness: float,
    node_masses: np.ndarray,
    edge_inertias: np.ndarray,
    g: float,
    external_forces: np.ndarray,
    free_indices: np.ndarray,
    free_twist_indices: np.ndarray,
) -> dict[str, float]:
    """Compute translational, rotational, elastic, and potential energies."""
    stretch = stretching_energy(state.positions, rest_lengths, stretch_stiffness)
    bend = bending_energy(state.positions, rest_lengths, bend_stiffness)
    twist = torsion_energy(state.twist_angles, rest_lengths, twist_stiffness)
    gravity = gravitational_potential(state.positions, node_masses, g)
    external = external_potential(state.positions, external_forces)
    kinetic = 0.5 * float(
        np.sum(
            node_masses[free_indices]
            * np.sum(state.velocities[free_indices] ** 2, axis=1)
        )
    )
    twist_kinetic = 0.5 * float(
        np.sum(
            edge_inertias[free_twist_indices]
            * state.twist_velocities[free_twist_indices] ** 2
        )
    )
    return {
        "stretching_energy": stretch,
        "bending_energy": bend,
        "torsion_energy": twist,
        "gravitational_energy": gravity,
        "external_energy": external,
        "kinetic_energy": kinetic,
        "twist_kinetic_energy": twist_kinetic,
        "total_energy": stretch
        + bend
        + twist
        + gravity
        + external
        + kinetic
        + twist_kinetic,
    }


def solve_dynamic_explicit(
    initial_positions: np.ndarray,
    rest_lengths: np.ndarray,
    initial_twist_angles: np.ndarray,
    stretch_stiffness: float,
    bend_stiffness: float,
    twist_stiffness: float,
    node_masses: np.ndarray,
    edge_inertias: np.ndarray,
    g: float,
    external_forces: np.ndarray,
    dt: float,
    steps: int,
    damping: float,
    twist_damping: float,
    convergence_tol: float,
    save_every: int,
    fixed_indices: tuple[int, ...] = (0,),
    fixed_twist_indices: tuple[int, ...] = (0,),
) -> tuple[RodState, dict[str, np.ndarray | bool | int | None]]:
    """Run 3D stretching + bending + torsion semi-implicit dynamics."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if damping < 0.0 or twist_damping < 0.0:
        raise ValueError("damping values must be non-negative.")
    if convergence_tol <= 0.0:
        raise ValueError("convergence_tol must be positive.")
    if save_every < 1:
        raise ValueError("save_every must be at least 1.")

    num_nodes = initial_positions.shape[0]
    num_edges = rest_lengths.size
    if initial_positions.shape != (num_nodes, 3):
        raise ValueError("initial_positions must have shape (num_nodes, 3).")
    if initial_twist_angles.shape != (num_edges,):
        raise ValueError("initial_twist_angles must have shape (num_links,).")
    if np.any(edge_inertias <= 0.0):
        raise ValueError("All edge inertias must be positive.")

    fixed_indices_array = np.array(fixed_indices, dtype=int)
    fixed_twist_indices_array = np.array(fixed_twist_indices, dtype=int)
    if np.any(fixed_indices_array < 0) or np.any(fixed_indices_array >= num_nodes):
        raise ValueError("fixed_indices contains an out-of-range node index.")
    if np.any(fixed_twist_indices_array < 0) or np.any(
        fixed_twist_indices_array >= num_edges
    ):
        raise ValueError("fixed_twist_indices contains an out-of-range edge index.")

    all_indices = np.arange(num_nodes)
    all_twist_indices = np.arange(num_edges)
    free_indices = np.setdiff1d(all_indices, fixed_indices_array)
    free_twist_indices = np.setdiff1d(all_twist_indices, fixed_twist_indices_array)
    if np.any(node_masses[free_indices] <= 0.0):
        raise ValueError("All free nodes must have positive mass.")

    state = RodState(
        positions=initial_positions.copy(),
        velocities=np.zeros_like(initial_positions),
        twist_angles=initial_twist_angles.copy(),
        twist_velocities=np.zeros_like(initial_twist_angles),
    )
    fixed_positions = initial_positions[fixed_indices_array].copy()
    fixed_twist_angles = initial_twist_angles[fixed_twist_indices_array].copy()
    gravity_forces = compute_gravity_forces(node_masses, g, fixed_indices_array)

    history: dict[str, list | bool | int | None] = {
        "positions": [],
        "velocities": [],
        "twist_angles": [],
        "twist_velocities": [],
        "bending_angles": [],
        "twist_angle_differences": [],
        "time": [],
        "total_energy": [],
        "stretching_energy": [],
        "bending_energy": [],
        "torsion_energy": [],
        "gravitational_energy": [],
        "external_energy": [],
        "kinetic_energy": [],
        "twist_kinetic_energy": [],
        "max_displacement": [],
        "max_velocity_norm": [],
        "max_force_norm": [],
        "max_twist_velocity_norm": [],
        "max_torque_norm": [],
        "max_bending_angle": [],
        "max_abs_twist_angle": [],
        "max_abs_twist_difference": [],
        "converged": False,
        "converged_step": None,
        "unstable": False,
        "unstable_step": None,
    }
    last_saved_step: int | None = None

    def append_history(
        step: int,
        residual_forces: np.ndarray | None,
        residual_torques: np.ndarray | None,
    ) -> None:
        nonlocal last_saved_step
        if residual_forces is None:
            max_force_norm = np.nan
        else:
            max_force_norm = float(
                np.max(np.linalg.norm(residual_forces[free_indices], axis=1))
            )
        if residual_torques is None or free_twist_indices.size == 0:
            max_torque_norm = np.nan
        else:
            max_torque_norm = float(
                np.max(np.abs(residual_torques[free_twist_indices]))
            )
        max_velocity_norm = float(
            np.max(np.linalg.norm(state.velocities[free_indices], axis=1))
        )
        if free_twist_indices.size == 0:
            max_twist_velocity_norm = 0.0
        else:
            max_twist_velocity_norm = float(
                np.max(np.abs(state.twist_velocities[free_twist_indices]))
            )
        max_displacement = float(
            np.max(np.linalg.norm(state.positions - initial_positions, axis=1))
        )
        bending_angles = compute_bending_angles(state.positions)
        twist_angle_differences = compute_twist_angle_differences(state.twist_angles)
        if bending_angles.size:
            max_bending_angle = float(np.max(bending_angles))
        else:
            max_bending_angle = 0.0
        if state.twist_angles.size:
            max_abs_twist_angle = float(np.max(np.abs(state.twist_angles)))
        else:
            max_abs_twist_angle = 0.0
        if twist_angle_differences.size:
            max_abs_twist_difference = float(
                np.max(np.abs(twist_angle_differences))
            )
        else:
            max_abs_twist_difference = 0.0
        energies = compute_total_energy(
            state,
            rest_lengths,
            stretch_stiffness,
            bend_stiffness,
            twist_stiffness,
            node_masses,
            edge_inertias,
            g,
            external_forces,
            free_indices,
            free_twist_indices,
        )

        history["positions"].append(state.positions.copy())
        history["velocities"].append(state.velocities.copy())
        history["twist_angles"].append(state.twist_angles.copy())
        history["twist_velocities"].append(state.twist_velocities.copy())
        history["bending_angles"].append(bending_angles.copy())
        history["twist_angle_differences"].append(twist_angle_differences.copy())
        history["time"].append(step * dt)
        for key, value in energies.items():
            history[key].append(value)
        history["max_displacement"].append(max_displacement)
        history["max_velocity_norm"].append(max_velocity_norm)
        history["max_force_norm"].append(max_force_norm)
        history["max_twist_velocity_norm"].append(max_twist_velocity_norm)
        history["max_torque_norm"].append(max_torque_norm)
        history["max_bending_angle"].append(max_bending_angle)
        history["max_abs_twist_angle"].append(max_abs_twist_angle)
        history["max_abs_twist_difference"].append(max_abs_twist_difference)
        last_saved_step = step

    def append_unstable_history(step: int) -> None:
        nonlocal last_saved_step
        num_bending_angles = max(initial_positions.shape[0] - 2, 0)
        num_twist_differences = max(initial_twist_angles.size - 1, 0)
        history["positions"].append(state.positions.copy())
        history["velocities"].append(state.velocities.copy())
        history["twist_angles"].append(state.twist_angles.copy())
        history["twist_velocities"].append(state.twist_velocities.copy())
        history["bending_angles"].append(np.full(num_bending_angles, np.nan))
        history["twist_angle_differences"].append(
            np.full(num_twist_differences, np.nan)
        )
        history["time"].append(step * dt)
        for key in (
            "total_energy",
            "stretching_energy",
            "bending_energy",
            "torsion_energy",
            "gravitational_energy",
            "external_energy",
            "kinetic_energy",
            "twist_kinetic_energy",
            "max_displacement",
            "max_velocity_norm",
            "max_force_norm",
            "max_twist_velocity_norm",
            "max_torque_norm",
            "max_bending_angle",
            "max_abs_twist_angle",
            "max_abs_twist_difference",
        ):
            history[key].append(np.nan)
        last_saved_step = step

    internal_forces = compute_internal_forces(
        state.positions,
        rest_lengths,
        stretch_stiffness,
        bend_stiffness,
    )
    residual_forces = internal_forces + external_forces + gravity_forces
    residual_torques = compute_torsion_torques(
        state.twist_angles,
        rest_lengths,
        twist_stiffness,
    )
    append_history(0, residual_forces, residual_torques)

    final_step = 0
    for step in range(1, steps + 1):
        internal_forces = compute_internal_forces(
            state.positions,
            rest_lengths,
            stretch_stiffness,
            bend_stiffness,
        )
        residual_forces = internal_forces + external_forces + gravity_forces
        damping_forces = -damping * state.velocities
        total_forces = residual_forces + damping_forces

        residual_torques = compute_torsion_torques(
            state.twist_angles,
            rest_lengths,
            twist_stiffness,
        )
        damping_torques = -twist_damping * state.twist_velocities
        total_torques = residual_torques + damping_torques

        state.velocities[free_indices] += (
            dt * total_forces[free_indices] / node_masses[free_indices, None]
        )
        state.positions[free_indices] += dt * state.velocities[free_indices]

        state.twist_velocities[free_twist_indices] += (
            dt
            * total_torques[free_twist_indices]
            / edge_inertias[free_twist_indices]
        )
        state.twist_angles[free_twist_indices] += (
            dt * state.twist_velocities[free_twist_indices]
        )

        state.positions[fixed_indices_array] = fixed_positions
        state.velocities[fixed_indices_array] = 0.0
        state.twist_angles[fixed_twist_indices_array] = fixed_twist_angles
        state.twist_velocities[fixed_twist_indices_array] = 0.0
        final_step = step

        if (
            not np.all(np.isfinite(state.positions))
            or not np.all(np.isfinite(state.velocities))
            or not np.all(np.isfinite(state.twist_angles))
            or not np.all(np.isfinite(state.twist_velocities))
        ):
            history["unstable"] = True
            history["unstable_step"] = step
            if last_saved_step != step:
                append_unstable_history(step)
            print("Simulation became unstable. Try reducing --dt or increasing damping.")
            break

        internal_forces = compute_internal_forces(
            state.positions,
            rest_lengths,
            stretch_stiffness,
            bend_stiffness,
        )
        residual_forces = internal_forces + external_forces + gravity_forces
        residual_torques = compute_torsion_torques(
            state.twist_angles,
            rest_lengths,
            twist_stiffness,
        )
        max_velocity_norm = float(
            np.max(np.linalg.norm(state.velocities[free_indices], axis=1))
        )
        max_force_norm = float(
            np.max(np.linalg.norm(residual_forces[free_indices], axis=1))
        )
        if free_twist_indices.size == 0:
            max_twist_velocity_norm = 0.0
            max_torque_norm = 0.0
        else:
            max_twist_velocity_norm = float(
                np.max(np.abs(state.twist_velocities[free_twist_indices]))
            )
            max_torque_norm = float(
                np.max(np.abs(residual_torques[free_twist_indices]))
            )

        if step % save_every == 0:
            append_history(step, residual_forces, residual_torques)

        if (
            max_velocity_norm < convergence_tol
            and max_force_norm < convergence_tol
            and max_twist_velocity_norm < convergence_tol
            and max_torque_norm < convergence_tol
        ):
            history["converged"] = True
            history["converged_step"] = step
            break

    if not history["unstable"] and last_saved_step != final_step:
        internal_forces = compute_internal_forces(
            state.positions,
            rest_lengths,
            stretch_stiffness,
            bend_stiffness,
        )
        residual_forces = internal_forces + external_forces + gravity_forces
        residual_torques = compute_torsion_torques(
            state.twist_angles,
            rest_lengths,
            twist_stiffness,
        )
        append_history(final_step, residual_forces, residual_torques)

    for key in (
        "positions",
        "velocities",
        "twist_angles",
        "twist_velocities",
        "bending_angles",
        "twist_angle_differences",
        "time",
        "total_energy",
        "stretching_energy",
        "bending_energy",
        "torsion_energy",
        "gravitational_energy",
        "external_energy",
        "kinetic_energy",
        "twist_kinetic_energy",
        "max_displacement",
        "max_velocity_norm",
        "max_force_norm",
        "max_twist_velocity_norm",
        "max_torque_norm",
        "max_bending_angle",
        "max_abs_twist_angle",
        "max_abs_twist_difference",
    ):
        history[key] = np.asarray(history[key])

    return state, history


def build_external_forces(
    num_nodes: int,
    force_node: int,
    force_vector: tuple[float, float, float],
    fixed_node: int = 0,
) -> np.ndarray:
    """Create the 3D nodal external-force array."""
    if force_node == fixed_node:
        raise ValueError("External force cannot be applied to the fixed left node.")
    if force_node < 0 or force_node >= num_nodes:
        raise ValueError(f"force_node must be between 0 and {num_nodes - 1}.")

    external_forces = np.zeros((num_nodes, 3), dtype=float)
    external_forces[force_node] = np.array(force_vector, dtype=float)
    return external_forces


def set_axes_equal_3d(ax) -> None:
    """Set equal scale on all axes of a Matplotlib 3D plot."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    centers = np.array(
        [
            0.5 * sum(x_limits),
            0.5 * sum(y_limits),
            0.5 * sum(z_limits),
        ]
    )
    radius = 0.5 * max(
        x_limits[1] - x_limits[0],
        y_limits[1] - y_limits[0],
        z_limits[1] - z_limits[0],
    )
    radius = max(radius, 1e-6)
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)


def visualize_chain_3d(
    initial_positions: np.ndarray,
    deformed_positions: np.ndarray,
    rest_lengths: np.ndarray,
    twist_angles: np.ndarray,
    external_forces: np.ndarray,
) -> None:
    """Plot undeformed chain, equilibrium chain, and external force."""
    plt, _ = load_matplotlib()
    displacements = deformed_positions - initial_positions

    fig = plt.figure(figsize=(9, 6))
    ax_chain = fig.add_subplot(111, projection="3d")

    ax_chain.plot(
        initial_positions[:, 0],
        initial_positions[:, 1],
        initial_positions[:, 2],
        "o--",
        label="Undeformed chain",
        color="tab:blue",
    )
    ax_chain.plot(
        deformed_positions[:, 0],
        deformed_positions[:, 1],
        deformed_positions[:, 2],
        "o-",
        label="Deformed chain",
        color="tab:orange",
    )

    ax_chain.quiver(
        initial_positions[:, 0],
        initial_positions[:, 1],
        initial_positions[:, 2],
        displacements[:, 0],
        displacements[:, 1],
        displacements[:, 2],
        color="tab:green",
        label="Node displacement",
    )

    force_nodes = np.where(np.linalg.norm(external_forces, axis=1) > 0.0)[0]
    if force_nodes.size:
        force_scale = 0.1 / max(
            np.max(np.linalg.norm(external_forces[force_nodes], axis=1)),
            1e-12,
        )
        ax_chain.quiver(
            deformed_positions[force_nodes, 0],
            deformed_positions[force_nodes, 1],
            deformed_positions[force_nodes, 2],
            external_forces[force_nodes, 0] * force_scale,
            external_forces[force_nodes, 1] * force_scale,
            external_forces[force_nodes, 2] * force_scale,
            color="tab:red",
            label="External force",
        )

    ax_chain.scatter(
        initial_positions[0, 0],
        initial_positions[0, 1],
        initial_positions[0, 2],
        s=90,
        c="black",
        marker="s",
        label="Fixed node",
    )
    ax_chain.set_xlabel("x position (m)")
    ax_chain.set_ylabel("y position (m)")
    ax_chain.set_zlabel("z position (m)")
    ax_chain.set_title("3D Soft Robot Chain Dynamic Equilibrium")
    ax_chain.text2D(
        0.02,
        0.95,
        "Gravity acts in the -z direction on free nodes",
        transform=ax_chain.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    set_axes_equal_3d(ax_chain)
    ax_chain.legend()
    plt.tight_layout()
    plt.show()


def animate_chain_evolution_3d(
    initial_positions: np.ndarray,
    history: dict[str, np.ndarray | bool | int | None],
    rest_lengths: np.ndarray,
    external_forces: np.ndarray,
    force_node: int,
    animation_output: str | None = None,
    show: bool = False,
) -> None:
    """Animate saved 3D chain states and optionally save a GIF."""
    plt, FuncAnimation = load_matplotlib()
    positions_history = history["positions"]
    bending_angles_history = history["bending_angles"]
    twist_angles_history = history["twist_angles"]
    twist_angle_differences_history = history["twist_angle_differences"]
    time_history = history["time"]
    vertex_arc_positions = compute_vertex_arc_positions(rest_lengths)
    edge_midpoint_arc_positions = compute_edge_midpoint_arc_positions(rest_lengths)
    all_positions = np.concatenate(
        (positions_history, initial_positions[None, :, :]),
        axis=0,
    )
    min_xyz = np.min(all_positions.reshape(-1, 3), axis=0)
    max_xyz = np.max(all_positions.reshape(-1, 3), axis=0)
    span = np.maximum(max_xyz - min_xyz, 1e-6)
    margin = 0.12 * max(float(np.max(span)), 1e-3)

    fig = plt.figure(figsize=(12, 9))
    gridspec = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.0])
    ax_chain = fig.add_subplot(gridspec[0, :], projection="3d")
    ax_bending = fig.add_subplot(gridspec[1, 0])
    ax_twist = fig.add_subplot(gridspec[1, 1])
    ax_twist_difference = fig.add_subplot(gridspec[1, 2])

    ax_chain.plot(
        initial_positions[:, 0],
        initial_positions[:, 1],
        initial_positions[:, 2],
        "o--",
        color="tab:blue",
        label="Initial chain",
    )
    current_line, = ax_chain.plot([], [], [], "o-", color="tab:orange", label="Current chain")
    ax_chain.scatter(
        initial_positions[0, 0],
        initial_positions[0, 1],
        initial_positions[0, 2],
        s=90,
        c="black",
        marker="s",
        label="Fixed node",
    )
    force_vector = external_forces[force_node]
    force_norm = float(np.linalg.norm(force_vector))
    force_scale = 0.1 / max(force_norm, 1e-12)
    scaled_force = force_vector * force_scale
    force_quiver = None
    if force_norm > 0.0:
        force_quiver = ax_chain.quiver(
            initial_positions[force_node, 0],
            initial_positions[force_node, 1],
            initial_positions[force_node, 2],
            scaled_force[0],
            scaled_force[1],
            scaled_force[2],
            color="tab:red",
            label="External force",
        )

    ax_chain.set_xlim(min_xyz[0] - margin, max_xyz[0] + margin)
    ax_chain.set_ylim(min_xyz[1] - margin, max_xyz[1] + margin)
    ax_chain.set_zlim(min_xyz[2] - margin, max_xyz[2] + margin)
    set_axes_equal_3d(ax_chain)
    ax_chain.set_xlabel("x position (m)")
    ax_chain.set_ylabel("y position (m)")
    ax_chain.set_zlabel("z position (m)")
    ax_chain.legend(loc="best")

    max_bending_angle = float(np.nanmax(np.abs(bending_angles_history))) if bending_angles_history.size else 0.0
    max_abs_twist_angle = float(np.nanmax(np.abs(twist_angles_history))) if twist_angles_history.size else 0.0
    max_abs_twist_difference = (
        float(np.nanmax(np.abs(twist_angle_differences_history)))
        if twist_angle_differences_history.size
        else 0.0
    )

    bending_line, = ax_bending.plot([], [], "o-", color="tab:purple", label="Curvature magnitude")
    twist_line, = ax_twist.plot([], [], "o-", color="tab:brown", label="Edge twist angle")
    twist_difference_line, = ax_twist_difference.plot(
        [],
        [],
        "o-",
        color="tab:red",
        label="Twist difference",
    )

    total_length = max(float(np.sum(rest_lengths)), 1e-12)
    ax_bending.set_xlim(0.0, total_length)
    ax_bending.set_ylim(-0.05 * max(max_bending_angle, 1.0), max_bending_angle + 0.1 * max(max_bending_angle, 1.0))
    ax_bending.set_xlabel("arc length at interior node (m)")
    ax_bending.set_ylabel("||kappa_b||")
    ax_bending.grid(True, alpha=0.35)
    ax_bending.legend(loc="best")

    twist_margin = 0.1 * max(max_abs_twist_angle, 1.0)
    ax_twist.axhline(0.0, color="0.45", linewidth=1.0, alpha=0.6)
    ax_twist.set_xlim(0.0, total_length)
    ax_twist.set_ylim(
        -max_abs_twist_angle - twist_margin,
        max_abs_twist_angle + twist_margin,
    )
    ax_twist.set_xlabel("arc length at edge midpoint (m)")
    ax_twist.set_ylabel("twist angle (rad)")
    ax_twist.grid(True, alpha=0.35)
    ax_twist.legend(loc="best")

    twist_difference_margin = 0.1 * max(max_abs_twist_difference, 1.0)
    ax_twist_difference.axhline(0.0, color="0.45", linewidth=1.0, alpha=0.6)
    ax_twist_difference.set_xlim(0.0, total_length)
    ax_twist_difference.set_ylim(
        -max_abs_twist_difference - twist_difference_margin,
        max_abs_twist_difference + twist_difference_margin,
    )
    ax_twist_difference.set_xlabel("arc length at interior node (m)")
    ax_twist_difference.set_ylabel("twist difference (rad)")
    ax_twist_difference.grid(True, alpha=0.35)
    ax_twist_difference.legend(loc="best")

    def update(frame_index: int):
        nonlocal force_quiver
        positions = positions_history[frame_index]
        bending_angles = bending_angles_history[frame_index]
        twist_angles = twist_angles_history[frame_index]
        twist_angle_differences = twist_angle_differences_history[frame_index]
        max_frame_bending_angle = (
            float(np.nanmax(bending_angles)) if bending_angles.size else 0.0
        )
        max_frame_twist_difference = (
            float(np.nanmax(np.abs(twist_angle_differences)))
            if twist_angle_differences.size
            else 0.0
        )
        current_line.set_data(positions[:, 0], positions[:, 1])
        current_line.set_3d_properties(positions[:, 2])
        if force_quiver is not None:
            force_quiver.remove()
            force_quiver = None
        if force_norm > 0.0:
            force_quiver = ax_chain.quiver(
                positions[force_node, 0],
                positions[force_node, 1],
                positions[force_node, 2],
                scaled_force[0],
                scaled_force[1],
                scaled_force[2],
                color="tab:red",
            )
        bending_line.set_data(vertex_arc_positions, bending_angles)
        twist_line.set_data(edge_midpoint_arc_positions, twist_angles)
        twist_difference_line.set_data(
            vertex_arc_positions,
            twist_angle_differences,
        )
        ax_chain.set_title(
            f"3D stretching + bending + torsion dynamics, "
            f"t = {time_history[frame_index]:.4f} s, "
            f"max bend = {max_frame_bending_angle:.4e} rad, "
            f"max |twist diff| = {max_frame_twist_difference:.4e} rad"
        )
        return current_line, bending_line, twist_line, twist_difference_line

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


def plot_time_history(
    history: dict[str, np.ndarray | bool | int | None],
    rest_lengths: np.ndarray | None = None,
) -> None:
    """Plot final angle profiles plus energy and convergence diagnostics."""
    plt, _ = load_matplotlib()
    time = history["time"]
    final_bending_angles = history["bending_angles"][-1]
    final_twist_angles = history["twist_angles"][-1]
    final_twist_differences = history["twist_angle_differences"][-1]
    if rest_lengths is None:
        vertex_arc_positions = np.arange(final_bending_angles.size)
        edge_midpoint_arc_positions = np.arange(final_twist_angles.size)
        vertex_xlabel = "interior node index"
        edge_xlabel = "edge index"
    else:
        vertex_arc_positions = compute_vertex_arc_positions(rest_lengths)
        edge_midpoint_arc_positions = compute_edge_midpoint_arc_positions(rest_lengths)
        vertex_xlabel = "arc length at interior node (m)"
        edge_xlabel = "arc length at edge midpoint (m)"

    fig = plt.figure(figsize=(13, 14))
    gridspec = fig.add_gridspec(6, 3, height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ax_bending = fig.add_subplot(gridspec[0, 0])
    ax_twist = fig.add_subplot(gridspec[0, 1])
    ax_twist_difference = fig.add_subplot(gridspec[0, 2])
    ax_energy = fig.add_subplot(gridspec[1, :])
    ax_displacement = fig.add_subplot(gridspec[2, :], sharex=ax_energy)
    ax_velocity = fig.add_subplot(gridspec[3, :], sharex=ax_energy)
    ax_angle_summary = fig.add_subplot(gridspec[4, :], sharex=ax_energy)
    ax_convergence = fig.add_subplot(gridspec[5, :], sharex=ax_energy)

    ax_bending.plot(
        vertex_arc_positions,
        final_bending_angles,
        "o-",
        color="tab:purple",
        label="Final curvature magnitude",
    )
    ax_bending.set_xlabel(vertex_xlabel)
    ax_bending.set_ylabel("||kappa_b||")
    ax_bending.grid(True, alpha=0.35)
    ax_bending.legend()

    ax_twist.axhline(0.0, color="0.45", linewidth=1.0, alpha=0.6)
    ax_twist.plot(
        edge_midpoint_arc_positions,
        final_twist_angles,
        "o-",
        color="tab:brown",
        label="Final edge twist angle",
    )
    ax_twist.set_xlabel(edge_xlabel)
    ax_twist.set_ylabel("twist angle (rad)")
    ax_twist.grid(True, alpha=0.35)
    ax_twist.legend()

    ax_twist_difference.axhline(0.0, color="0.45", linewidth=1.0, alpha=0.6)
    ax_twist_difference.plot(
        vertex_arc_positions,
        final_twist_differences,
        "o-",
        color="tab:red",
        label="Final twist difference",
    )
    ax_twist_difference.set_xlabel(vertex_xlabel)
    ax_twist_difference.set_ylabel("twist difference (rad)")
    ax_twist_difference.grid(True, alpha=0.35)
    ax_twist_difference.legend()

    ax_energy.plot(time, history["total_energy"], label="Total energy")
    ax_energy.plot(time, history["stretching_energy"], label="Stretching energy")
    ax_energy.plot(time, history["bending_energy"], label="Bending energy")
    ax_energy.plot(time, history["torsion_energy"], label="Torsion energy")
    ax_energy.plot(time, history["kinetic_energy"], label="Translational kinetic")
    ax_energy.plot(time, history["twist_kinetic_energy"], label="Twist kinetic")
    ax_energy.set_ylabel("Energy (J)")
    ax_energy.grid(True, alpha=0.35)
    ax_energy.legend()

    ax_displacement.plot(time, history["max_displacement"], color="tab:green")
    ax_displacement.set_ylabel("Max displacement (m)")
    ax_displacement.grid(True, alpha=0.35)

    ax_velocity.plot(time, history["max_velocity_norm"], color="tab:purple")
    ax_velocity.set_ylabel("Max node velocity")
    ax_velocity.grid(True, alpha=0.35)

    ax_angle_summary.plot(time, history["max_bending_angle"], color="tab:purple", label="Curvature magnitude")
    ax_angle_summary.plot(time, history["max_abs_twist_angle"], color="tab:brown", label="|Twist angle|")
    ax_angle_summary.plot(
        time,
        history["max_abs_twist_difference"],
        color="tab:red",
        label="|Twist difference|",
    )
    ax_angle_summary.set_ylabel("Angle summary (rad)")
    ax_angle_summary.grid(True, alpha=0.35)
    ax_angle_summary.legend()

    ax_convergence.plot(time, history["max_force_norm"], color="tab:red", label="Force")
    ax_convergence.plot(time, history["max_torque_norm"], color="tab:brown", label="Torque")
    ax_convergence.plot(
        time,
        history["max_twist_velocity_norm"],
        color="tab:gray",
        label="Twist velocity",
    )
    ax_convergence.set_xlabel("Time (s)")
    ax_convergence.set_ylabel("Convergence metrics")
    ax_convergence.grid(True, alpha=0.35)
    ax_convergence.legend()

    fig.suptitle("Stretching + Bending + Torsion Diagnostics")
    plt.tight_layout()
    plt.show()


def print_displacements_and_twist(
    initial_positions: np.ndarray,
    final_state: RodState,
    force_node: int,
) -> None:
    """Print nodal displacement vectors and final edge twist angles."""
    displacements = final_state.positions - initial_positions

    print("Node displacements:")
    for node_index, displacement in enumerate(displacements):
        magnitude = np.linalg.norm(displacement)
        print(
            f"  Node {node_index:2d}: "
            f"ux = {displacement[0]: .8e} m, "
            f"uy = {displacement[1]: .8e} m, "
            f"uz = {displacement[2]: .8e} m, "
            f"|u| = {magnitude: .8e} m"
        )

    force_node_magnitude = np.linalg.norm(displacements[force_node])
    print(
        f"\nTotal displacement magnitude of force-applied node {force_node}: "
        f"{force_node_magnitude:.8e} m"
    )

    print("\nFinal edge twist angles:")
    for edge_index, twist_angle in enumerate(final_state.twist_angles):
        print(f"  Edge {edge_index:2d}: theta = {twist_angle: .8e} rad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stretching + bending + torsion explicit dynamics for a 3D soft "
            "robot chain."
        )
    )
    parser.add_argument("--num-links", type=int, default=10, help="Number of serial links.")
    parser.add_argument("--total-length", type=float, default=2.0, help="Initial total length in meters.")
    parser.add_argument("--stretch-stiffness", type=float, default=15000.0, help="Axial stretching stiffness in N/m.")
    parser.add_argument("--bend-stiffness", type=float, default=0.01, help="3D bending stiffness.")
    parser.add_argument("--twist-stiffness", type=float, default=0.01, help="Torsional stiffness.")
    parser.add_argument("--node-mass", type=float, default=0.2, help="Mass of each free node in kg.")
    parser.add_argument("--edge-inertia", type=float, default=1e-4, help="Rotational inertia of each twist edge.")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravitational acceleration in m/s^2.")
    parser.add_argument("--force-node", type=int, default=10, help="Node index where the external force is applied.")
    parser.add_argument("--force-x", type=float, default=10.0, help="External force x component in N.")
    parser.add_argument("--force-y", type=float, default=0.0, help="External force y component in N.")
    parser.add_argument("--force-z", type=float, default=0.0, help="External force z component in N.")
    parser.add_argument("--initial-tip-twist", type=float, default=0.0, help="Initial twist angle on the last edge in radians.")
    parser.add_argument("--dt", type=float, default=2e-4, help="Time step size in seconds.")
    parser.add_argument("--steps", type=int, default=20000, help="Maximum number of time steps.")
    parser.add_argument("--damping", type=float, default=2.0, help="Nodal viscous damping coefficient.")
    parser.add_argument("--twist-damping", type=float, default=0.02, help="Viscous damping coefficient for twist angles.")
    parser.add_argument(
        "--convergence-tol",
        type=float,
        default=1e-5,
        help="Tolerance for free-node velocity, force, twist velocity, and torque.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save history every N steps. Default 1000 gives 0.1 s between saved frames when dt=1e-4.",
    )
    parser.add_argument("--animate", action="store_true", help="Compatibility flag; animation is displayed by default unless --no-plot is used.")
    parser.add_argument("--animation-output", type=str, default=None, help="Path for saving the dynamic-process GIF.")
    parser.add_argument("--plot-history", action="store_true", help="Compatibility flag; diagnostics are displayed by default unless --no-plot is used.")
    parser.add_argument("--no-plot", action="store_true", help="Disable interactive plot windows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    initial_positions, rest_lengths, initial_twist_angles = initialize_chain(
        args.num_links,
        args.total_length,
        args.initial_tip_twist,
    )
    num_nodes = initial_positions.shape[0]
    num_edges = rest_lengths.size
    node_masses = np.full(num_nodes, args.node_mass)
    node_masses[0] = 0.0
    edge_inertias = np.full(num_edges, args.edge_inertia)
    external_forces = build_external_forces(
        num_nodes,
        args.force_node,
        (args.force_x, args.force_y, args.force_z),
        fixed_node=0,
    )

    final_state, history = solve_dynamic_explicit(
        initial_positions=initial_positions,
        rest_lengths=rest_lengths,
        initial_twist_angles=initial_twist_angles,
        stretch_stiffness=args.stretch_stiffness,
        bend_stiffness=args.bend_stiffness,
        twist_stiffness=args.twist_stiffness,
        node_masses=node_masses,
        edge_inertias=edge_inertias,
        g=args.gravity,
        external_forces=external_forces,
        dt=args.dt,
        steps=args.steps,
        damping=args.damping,
        twist_damping=args.twist_damping,
        convergence_tol=args.convergence_tol,
        save_every=args.save_every,
        fixed_indices=(0,),
        fixed_twist_indices=(0,),
    )

    if history["unstable"]:
        print(f"Simulation stopped at unstable step {history['unstable_step']}.")
    elif history["converged"]:
        print(f"Simulation converged at step {history['converged_step']}.")
    else:
        print(f"Simulation did not converge within {args.steps} steps.")

    print_displacements_and_twist(initial_positions, final_state, args.force_node)

    if args.no_plot:
        if args.animate:
            print("--animate was requested, but --no-plot is enabled; skipping interactive animation.")
        if args.animation_output:
            print("--animation-output was requested, but --no-plot is enabled; skipping GIF output.")
        if args.plot_history:
            print("--plot-history was requested, but --no-plot is enabled; skipping diagnostics.")
    else:
        animate_chain_evolution_3d(
            initial_positions=initial_positions,
            history=history,
            rest_lengths=rest_lengths,
            external_forces=external_forces,
            force_node=args.force_node,
            animation_output=args.animation_output,
            show=True,
        )
        visualize_chain_3d(
            initial_positions,
            final_state.positions,
            rest_lengths,
            final_state.twist_angles,
            external_forces,
        )
        plot_time_history(history, rest_lengths)


if __name__ == "__main__":
    main()
