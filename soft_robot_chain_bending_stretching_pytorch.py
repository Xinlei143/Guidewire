"""Stretching + bending DER-style dynamic simulation of a 3D soft robot chain.

The robot is modeled as 10 serial elastic links and 11 point nodes by default.
The left node is fixed, the right node is free, and both axial stretching and
3D centerline bending are implemented. Bending forces are computed from the
bending energy with PyTorch automatic differentiation. Twisting is intentionally
left for the separate 3D stretching + bending + torsion script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from soft_robot_common import (
    compute_edge_geometry,
    compute_stretching_forces,
    compute_vertex_arc_positions,
    compute_vertex_rest_lengths,
    external_potential,
    gravitational_potential,
    load_matplotlib,
    stretching_energy,
)


def initialize_chain(num_links: int, total_length: float) -> tuple[np.ndarray, np.ndarray]:
    """Create a straight 3D chain and uniform link rest lengths."""
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
    return positions, rest_lengths


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


def bending_energy_torch(
    positions: torch.Tensor,
    rest_lengths: torch.Tensor,
    stiffness: float,
    min_length: float = 1e-12,
) -> torch.Tensor:
    """Return the current bending energy as a PyTorch scalar tensor."""
    if stiffness == 0.0 or positions.shape[0] < 3:
        return positions.new_tensor(0.0)

    edge_vectors = positions[1:] - positions[:-1]
    lengths = torch.linalg.norm(edge_vectors, dim=1)
    safe_lengths = torch.clamp(lengths, min=min_length)
    tangents = edge_vectors / safe_lengths[:, None]

    dot_products = torch.sum(tangents[:-1] * tangents[1:], dim=1)
    denominators = torch.clamp(1.0 + dot_products, min=min_length)
    cross_products = torch.linalg.cross(tangents[:-1], tangents[1:], dim=1)
    curvature_binormals = 2.0 * cross_products / denominators[:, None]
    curvature_magnitudes = torch.linalg.norm(curvature_binormals, dim=1)

    valid_vertices = (lengths[:-1] > min_length) & (lengths[1:] > min_length)
    curvature_magnitudes = torch.where(
        valid_vertices,
        curvature_magnitudes,
        torch.zeros_like(curvature_magnitudes),
    )
    vertex_lengths = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
    vertex_lengths = torch.clamp(vertex_lengths, min=min_length)
    return 0.5 * stiffness * torch.sum(curvature_magnitudes**2 / vertex_lengths)


def compute_bending_forces_autograd(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    stiffness: float,
) -> np.ndarray:
    """Assemble bending forces as the negative autograd energy gradient."""
    if stiffness == 0.0 or positions.shape[0] < 3:
        return np.zeros_like(positions)

    torch_positions = torch.as_tensor(
        positions,
        dtype=torch.float64,
        device="cpu",
    ).clone().detach().requires_grad_(True)
    torch_rest_lengths = torch.as_tensor(
        rest_lengths,
        dtype=torch.float64,
        device="cpu",
    )
    energy = bending_energy_torch(torch_positions, torch_rest_lengths, stiffness)
    (gradient,) = torch.autograd.grad(energy, torch_positions)
    return -gradient.detach().cpu().numpy().astype(positions.dtype, copy=False)


def compute_internal_forces(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    stretch_stiffness: float,
    bend_stiffness: float,
) -> np.ndarray:
    """Return the sum of stretching and bending nodal forces."""
    stretching_forces = compute_stretching_forces(
        positions,
        rest_lengths,
        stretch_stiffness,
    )
    bending_forces = compute_bending_forces_autograd(
        positions,
        rest_lengths,
        bend_stiffness,
    )
    return stretching_forces + bending_forces


def compute_total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    rest_lengths: np.ndarray,
    stretch_stiffness: float,
    bend_stiffness: float,
    node_masses: np.ndarray,
    g: float,
    external_forces: np.ndarray,
    free_indices: np.ndarray,
) -> dict[str, float]:
    """Compute stretching, bending, potential, kinetic, and total energy values."""
    stretch = stretching_energy(positions, rest_lengths, stretch_stiffness)
    bend = bending_energy(positions, rest_lengths, bend_stiffness)
    gravity = gravitational_potential(positions, node_masses, g)
    external = external_potential(positions, external_forces)
    kinetic = 0.5 * float(
        np.sum(node_masses[free_indices] * np.sum(velocities[free_indices] ** 2, axis=1))
    )
    return {
        "stretching_energy": stretch,
        "bending_energy": bend,
        "gravitational_energy": gravity,
        "external_energy": external,
        "kinetic_energy": kinetic,
        "total_energy": stretch + bend + gravity + external + kinetic,
    }


def compute_gravity_forces(
    node_masses: np.ndarray, g: float, fixed_indices: np.ndarray
) -> np.ndarray:
    """Return 3D nodal gravity forces in the negative z direction."""
    gravity_forces = np.zeros((node_masses.size, 3), dtype=float)
    gravity_forces[:, 2] = -node_masses * g
    gravity_forces[fixed_indices] = 0.0
    return gravity_forces


def solve_dynamic_explicit(
    initial_positions: np.ndarray,
    rest_lengths: np.ndarray,
    stretch_stiffness: float,
    bend_stiffness: float,
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
    """Run a 3D stretching + bending explicit/semi-implicit time integration.

    Stretching forces are assembled analytically. Bending forces are assembled
    from the PyTorch autograd gradient of the bending energy, which keeps this
    research script easy to check against the bending energy formula without a
    finite-difference perturbation step.
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
        "bending_energy": [],
        "gravitational_energy": [],
        "external_energy": [],
        "kinetic_energy": [],
        "bending_angles": [],
        "max_displacement": [],
        "max_velocity_norm": [],
        "max_force_norm": [],
        "max_abs_bending_angle": [],
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
        bending_angles = compute_bending_angles(positions)
        if bending_angles.size:
            max_abs_bending_angle = float(np.max(np.abs(bending_angles)))
        else:
            max_abs_bending_angle = 0.0
        energies = compute_total_energy(
            positions,
            velocities,
            rest_lengths,
            stretch_stiffness,
            bend_stiffness,
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
        history["bending_energy"].append(energies["bending_energy"])
        history["gravitational_energy"].append(energies["gravitational_energy"])
        history["external_energy"].append(energies["external_energy"])
        history["kinetic_energy"].append(energies["kinetic_energy"])
        history["bending_angles"].append(bending_angles.copy())
        history["max_displacement"].append(max_displacement)
        history["max_velocity_norm"].append(max_velocity_norm)
        history["max_force_norm"].append(max_force_norm)
        history["max_abs_bending_angle"].append(max_abs_bending_angle)
        last_saved_step = step

    def append_unstable_history(step: int) -> None:
        nonlocal last_saved_step
        num_bending_angles = max(initial_positions.shape[0] - 2, 0)
        history["positions"].append(positions.copy())
        history["velocities"].append(velocities.copy())
        history["time"].append(step * dt)
        history["total_energy"].append(np.nan)
        history["stretching_energy"].append(np.nan)
        history["bending_energy"].append(np.nan)
        history["gravitational_energy"].append(np.nan)
        history["external_energy"].append(np.nan)
        history["kinetic_energy"].append(np.nan)
        history["bending_angles"].append(np.full(num_bending_angles, np.nan))
        history["max_displacement"].append(np.nan)
        history["max_velocity_norm"].append(np.nan)
        history["max_force_norm"].append(np.nan)
        history["max_abs_bending_angle"].append(np.nan)
        last_saved_step = step

    internal_forces = compute_internal_forces(
        positions,
        rest_lengths,
        stretch_stiffness,
        bend_stiffness,
    )
    residual_forces = internal_forces + external_forces + gravity_forces
    append_history(0, residual_forces)

    final_step = 0
    for step in range(1, steps + 1):
        internal_forces = compute_internal_forces(
            positions,
            rest_lengths,
            stretch_stiffness,
            bend_stiffness,
        )
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

        internal_forces = compute_internal_forces(
            positions,
            rest_lengths,
            stretch_stiffness,
            bend_stiffness,
        )
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
        internal_forces = compute_internal_forces(
            positions,
            rest_lengths,
            stretch_stiffness,
            bend_stiffness,
        )
        residual_forces = internal_forces + external_forces + gravity_forces
        append_history(final_step, residual_forces)

    for key in (
        "positions",
        "velocities",
        "time",
        "total_energy",
        "stretching_energy",
        "bending_energy",
        "gravitational_energy",
        "external_energy",
        "kinetic_energy",
        "bending_angles",
        "max_displacement",
        "max_velocity_norm",
        "max_force_norm",
        "max_abs_bending_angle",
    ):
        history[key] = np.asarray(history[key])

    return positions, history


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


def visualize_chain(
    initial_positions: np.ndarray,
    deformed_positions: np.ndarray,
    rest_lengths: np.ndarray,
    external_forces: np.ndarray,
) -> None:
    """Plot undeformed chain, equilibrium chain, external force, and displacements."""
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
        force_scale = 0.1 / max(np.max(np.linalg.norm(external_forces[force_nodes], axis=1)), 1e-12)
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

    ax_chain.scatter(initial_positions[0, 0], initial_positions[0, 1], initial_positions[0, 2], s=90, c="black", marker="s", label="Fixed node")
    ax_chain.text2D(
        0.02,
        0.95,
        "Gravity acts in the -z direction on free nodes",
        transform=ax_chain.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax_chain.set_xlabel("x position (m)")
    ax_chain.set_ylabel("y position (m)")
    ax_chain.set_zlabel("z position (m)")
    ax_chain.set_title("3D Soft Robot Chain Dynamic Equilibrium")
    set_axes_equal_3d(ax_chain)
    ax_chain.legend()
    plt.tight_layout()
    plt.show()


def animate_chain_evolution(
    initial_positions: np.ndarray,
    history: dict[str, np.ndarray | bool | int | None],
    rest_lengths: np.ndarray,
    external_forces: np.ndarray,
    force_node: int,
    animation_output: str | None = None,
    show: bool = False,
) -> None:
    """Animate saved 3D chain states and optionally save the animation as a GIF."""
    plt, FuncAnimation = load_matplotlib()
    positions_history = history["positions"]
    time_history = history["time"]
    bending_angles_history = history["bending_angles"]
    vertex_arc_positions = compute_vertex_arc_positions(rest_lengths)
    all_positions = np.concatenate((positions_history, initial_positions[None, :, :]), axis=0)
    min_xyz = np.min(all_positions.reshape(-1, 3), axis=0)
    max_xyz = np.max(all_positions.reshape(-1, 3), axis=0)
    span = np.maximum(max_xyz - min_xyz, 1e-6)
    margin = 0.12 * max(float(np.max(span)), 1e-3)

    fig = plt.figure(figsize=(10, 8))
    gridspec = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0])
    ax_chain = fig.add_subplot(gridspec[0], projection="3d")
    ax_angles = fig.add_subplot(gridspec[1])
    ax_chain.plot(
        initial_positions[:, 0],
        initial_positions[:, 1],
        initial_positions[:, 2],
        "o--",
        color="tab:blue",
        label="Initial chain",
    )
    current_line, = ax_chain.plot([], [], [], "o-", color="tab:orange", label="Current chain")
    fixed_marker = ax_chain.scatter(
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

    max_abs_bending_angle = float(np.nanmax(np.abs(bending_angles_history))) if bending_angles_history.size else 0.0
    angle_margin = 0.1 * max(max_abs_bending_angle, 1.0)
    ax_angles.axhline(0.0, color="0.45", linewidth=1.0, alpha=0.6)
    bending_angle_line, = ax_angles.plot(
        [],
        [],
        "o-",
        color="tab:purple",
        label="Curvature magnitude",
    )
    ax_angles.set_xlim(
        0.0,
        max(float(np.sum(rest_lengths)), 1e-12),
    )
    ax_angles.set_ylim(
        -max_abs_bending_angle - angle_margin,
        max_abs_bending_angle + angle_margin,
    )
    ax_angles.set_xlabel("arc length at interior node (m)")
    ax_angles.set_ylabel("||kappa_b||")
    ax_angles.grid(True, alpha=0.35)
    ax_angles.legend(loc="best")

    def update(frame_index: int):
        nonlocal force_quiver
        positions = positions_history[frame_index]
        bending_angles = bending_angles_history[frame_index]
        if bending_angles.size:
            max_frame_bending_angle = float(np.nanmax(np.abs(bending_angles)))
        else:
            max_frame_bending_angle = 0.0
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
        bending_angle_line.set_data(vertex_arc_positions, bending_angles)
        ax_chain.set_title(
            f"3D stretching + bending DER-style dynamics, "
            f"t = {time_history[frame_index]:.4f} s, "
            f"max ||kappa_b|| = {max_frame_bending_angle:.4e}"
        )
        return current_line, fixed_marker, bending_angle_line

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
    """Plot final bending profile plus energy and convergence diagnostics."""
    plt, _ = load_matplotlib()
    time = history["time"]
    final_bending_angles = history["bending_angles"][-1]
    if rest_lengths is None:
        vertex_arc_positions = np.arange(final_bending_angles.size)
        angle_xlabel = "interior node index"
    else:
        vertex_arc_positions = compute_vertex_arc_positions(rest_lengths)
        angle_xlabel = "arc length at interior node (m)"

    fig, axes = plt.subplots(5, 1, figsize=(9, 13))
    axes[0].axhline(0.0, color="0.45", linewidth=1.0, alpha=0.6)
    axes[0].plot(
        vertex_arc_positions,
        final_bending_angles,
        "o-",
        color="tab:purple",
        label="Final curvature magnitude",
    )
    axes[0].set_xlabel(angle_xlabel)
    axes[0].set_ylabel("||kappa_b||")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend()

    axes[1].plot(time, history["total_energy"], label="Total energy")
    axes[1].plot(time, history["stretching_energy"], label="Stretching energy")
    axes[1].plot(time, history["bending_energy"], label="Bending energy")
    axes[1].plot(time, history["kinetic_energy"], label="Kinetic energy")
    axes[1].set_ylabel("Energy (J)")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend()

    axes[2].plot(time, history["max_displacement"], color="tab:green")
    axes[2].set_ylabel("Max displacement (m)")
    axes[2].grid(True, alpha=0.35)

    axes[3].plot(time, history["max_abs_bending_angle"], color="tab:purple")
    axes[3].set_ylabel("Max ||kappa_b||")
    axes[3].grid(True, alpha=0.35)

    axes[4].plot(time, history["max_velocity_norm"], color="tab:purple", label="Max velocity")
    axes[4].plot(time, history["max_force_norm"], color="tab:red", label="Max residual force")
    axes[4].set_xlabel("Time (s)")
    axes[4].set_ylabel("Convergence metrics")
    axes[4].grid(True, alpha=0.35)
    axes[4].legend()

    fig.suptitle("Stretching + Bending Diagnostics")
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
            f"uz = {displacement[2]: .8e} m, "
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
    force_vector: tuple[float, float, float],
    fixed_node: int = 0,
) -> np.ndarray:
    """Create the 3D nodal external-force array."""
    if force_node == fixed_node:
        raise ValueError("External force cannot be applied to the fixed left node.")
    if force_node < 0 or force_node >= num_nodes:
        raise ValueError(f"force_node must be between 0 and {num_nodes - 1}.")

    external_forces = np.zeros((num_nodes, 3), dtype=float)
    # The user-configurable additional external force is added here.
    external_forces[force_node] = np.array(force_vector, dtype=float)
    return external_forces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stretching + bending DER-style explicit dynamics for a 3D soft "
            "robot chain. Twisting is not implemented in this stage."
        )
    )
    parser.add_argument("--num-links", type=int, default=10, help="Number of serial links.")
    parser.add_argument("--total-length", type=float, default=2.0, help="Initial total length in meters.")
    parser.add_argument(
        "--stretch-stiffness",
        type=float,
        default=5000.0,
        help="Axial stretching stiffness in N/m.",
    )
    parser.add_argument(
        "--bend-stiffness",
        type=float,
        default=0.5,
        help="3D bending stiffness.",
    )
    parser.add_argument("--node-mass", type=float, default=0.1, help="Mass of each free node in kg.")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravitational acceleration in m/s^2.")
    parser.add_argument("--force-node", type=int, default=5, help="Node index where the external force is applied.")
    parser.add_argument("--force-x", type=float, default=5.0, help="External force x component in N.")
    parser.add_argument("--force-y", type=float, default=5.0, help="External force y component in N.")
    parser.add_argument("--force-z", type=float, default=5.0, help="External force z component in N.")
    parser.add_argument("--dt", type=float, default=2e-4, help="Time step size in seconds.")
    parser.add_argument("--steps", type=int, default=60000, help="Maximum number of time steps.")
    parser.add_argument("--damping", type=float, default=0.5, help="Nodal viscous damping coefficient.")
    parser.add_argument(
        "--convergence-tol",
        type=float,
        default=1e-5,
        help="Tolerance for free-node max velocity and undamped residual force.",
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

    initial_positions, rest_lengths = initialize_chain(args.num_links, args.total_length)
    num_nodes = initial_positions.shape[0]
    node_masses = np.full(num_nodes, args.node_mass)
    node_masses[0] = 0.0
    external_forces = build_external_forces(
        num_nodes,
        args.force_node,
        (args.force_x, args.force_y, args.force_z),
        fixed_node=0,
    )

    deformed_positions, history = solve_dynamic_explicit(
        initial_positions=initial_positions,
        rest_lengths=rest_lengths,
        stretch_stiffness=args.stretch_stiffness,
        bend_stiffness=args.bend_stiffness,
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

    if args.no_plot:
        if args.animate:
            print("--animate was requested, but --no-plot is enabled; skipping interactive animation.")
        if args.animation_output:
            print("--animation-output was requested, but --no-plot is enabled; skipping GIF output.")
        if args.plot_history:
            print("--plot-history was requested, but --no-plot is enabled; skipping diagnostics.")
    else:
        animate_chain_evolution(
            initial_positions=initial_positions,
            history=history,
            rest_lengths=rest_lengths,
            external_forces=external_forces,
            force_node=args.force_node,
            animation_output=args.animation_output,
            show=True,
        )
        visualize_chain(
            initial_positions,
            deformed_positions,
            rest_lengths,
            external_forces,
        )
        plot_time_history(history, rest_lengths)


if __name__ == "__main__":
    main()
