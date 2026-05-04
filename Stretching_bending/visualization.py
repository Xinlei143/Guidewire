"""Plotting and animation helpers for soft-rod simulation results."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from soft_robot_common import compute_vertex_arc_positions, to_numpy

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())


def _vertex_arc_positions_numpy(rest_lengths) -> np.ndarray:
    """Compute interior arc positions for plotting regardless of input type."""
    rest_lengths_tensor = torch.as_tensor(rest_lengths, dtype=torch.float64)
    return to_numpy(compute_vertex_arc_positions(rest_lengths_tensor))


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
    initial_positions,
    deformed_positions,
    rest_lengths,
    external_forces,
    fixed_indices: tuple[int, ...] = (0, 1),
) -> None:
    """Plot undeformed chain, equilibrium chain, external force, and displacements."""
    plt, _ = load_matplotlib()
    initial_positions = to_numpy(initial_positions)
    deformed_positions = to_numpy(deformed_positions)
    external_forces = to_numpy(external_forces)
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

    fixed_positions = initial_positions[np.array(fixed_indices, dtype=int)]
    ax_chain.scatter(
        fixed_positions[:, 0],
        fixed_positions[:, 1],
        fixed_positions[:, 2],
        s=90,
        c="black",
        marker="s",
        label="Fixed nodes",
    )
    ax_chain.text2D(
        0.02,
        0.95,
        "Gravity, if enabled, acts in the -z direction on free nodes",
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
    initial_positions,
    history: dict[str, np.ndarray | bool | int | None],
    rest_lengths,
    external_forces,
    force_node: int,
    fixed_indices: tuple[int, ...] = (0, 1),
    animation_output: str | None = None,
    show: bool = False,
) -> None:
    """Animate saved 3D chain states and optionally save the animation as a GIF."""
    plt, FuncAnimation = load_matplotlib()
    initial_positions = to_numpy(initial_positions)
    rest_lengths = to_numpy(rest_lengths)
    external_forces = to_numpy(external_forces)
    positions_history = to_numpy(history["positions"])
    time_history = to_numpy(history["time"])
    kappa_b_magnitudes_history = to_numpy(history["kappa_b_magnitudes"])
    vertex_arc_positions = _vertex_arc_positions_numpy(rest_lengths)
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
    fixed_positions = initial_positions[np.array(fixed_indices, dtype=int)]
    fixed_marker = ax_chain.scatter(
        fixed_positions[:, 0],
        fixed_positions[:, 1],
        fixed_positions[:, 2],
        s=90,
        c="black",
        marker="s",
        label="Fixed nodes",
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

    max_kappa_b_magnitude = (
        float(np.nanmax(kappa_b_magnitudes_history))
        if kappa_b_magnitudes_history.size
        else 0.0
    )
    angle_margin = 0.1 * max(max_kappa_b_magnitude, 1.0)
    ax_angles.axhline(0.0, color="0.45", linewidth=1.0, alpha=0.6)
    kappa_b_line, = ax_angles.plot(
        [],
        [],
        "o-",
        color="tab:purple",
        label="Curvature-binormal magnitude",
    )
    ax_angles.set_xlim(0.0, max(float(np.sum(rest_lengths)), 1e-12))
    ax_angles.set_ylim(-angle_margin, max_kappa_b_magnitude + angle_margin)
    ax_angles.set_xlabel("arc length at interior node (m)")
    ax_angles.set_ylabel("||kappa_b||")
    ax_angles.grid(True, alpha=0.35)
    ax_angles.legend(loc="best")

    def update(frame_index: int):
        nonlocal force_quiver
        positions = positions_history[frame_index]
        kappa_b_magnitudes = kappa_b_magnitudes_history[frame_index]
        if kappa_b_magnitudes.size:
            max_frame_kappa_b = float(np.nanmax(kappa_b_magnitudes))
        else:
            max_frame_kappa_b = 0.0
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
        kappa_b_line.set_data(vertex_arc_positions, kappa_b_magnitudes)
        ax_chain.set_title(
            f"3D stretching + bending DER-style dynamics, "
            f"t = {time_history[frame_index]:.4f} s, "
            f"max ||kappa_b|| = {max_frame_kappa_b:.4e}"
        )
        return current_line, fixed_marker, kappa_b_line

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
    rest_lengths=None,
) -> None:
    """Plot final bending profile plus energy and convergence diagnostics."""
    plt, _ = load_matplotlib()
    time = to_numpy(history["time"])
    final_kappa_b_magnitudes = to_numpy(history["kappa_b_magnitudes"])[-1]
    if rest_lengths is None:
        vertex_arc_positions = np.arange(final_kappa_b_magnitudes.size)
        angle_xlabel = "interior node index"
    else:
        vertex_arc_positions = _vertex_arc_positions_numpy(to_numpy(rest_lengths))
        angle_xlabel = "arc length at interior node (m)"

    fig, axes = plt.subplots(5, 1, figsize=(9, 13))
    axes[0].axhline(0.0, color="0.45", linewidth=1.0, alpha=0.6)
    axes[0].plot(
        vertex_arc_positions,
        final_kappa_b_magnitudes,
        "o-",
        color="tab:purple",
        label="Final curvature-binormal magnitude",
    )
    axes[0].set_xlabel(angle_xlabel)
    axes[0].set_ylabel("||kappa_b||")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend()

    axes[1].plot(time, to_numpy(history["total_energy"]), label="Total energy")
    axes[1].plot(time, to_numpy(history["stretching_energy"]), label="Stretching energy")
    axes[1].plot(time, to_numpy(history["bending_energy"]), label="Bending energy")
    axes[1].plot(time, to_numpy(history["kinetic_energy"]), label="Kinetic energy")
    axes[1].set_ylabel("Energy (J)")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend()

    axes[2].plot(time, to_numpy(history["max_displacement"]), color="tab:green")
    axes[2].set_ylabel("Max displacement (m)")
    axes[2].grid(True, alpha=0.35)

    axes[3].plot(time, to_numpy(history["max_kappa_b_magnitude"]), color="tab:purple")
    axes[3].set_ylabel("Max ||kappa_b||")
    axes[3].grid(True, alpha=0.35)

    axes[4].plot(time, to_numpy(history["max_velocity_norm"]), color="tab:purple", label="Max velocity")
    axes[4].plot(time, to_numpy(history["max_force_norm"]), color="tab:red", label="Max residual force")
    axes[4].set_xlabel("Time (s)")
    axes[4].set_ylabel("Convergence metrics")
    axes[4].grid(True, alpha=0.35)
    axes[4].legend()

    fig.suptitle("Stretching + Bending Diagnostics")
    plt.tight_layout()
    plt.show()


def plot_energy_history(history: dict[str, np.ndarray | bool | int | None]) -> None:
    """Plot only the saved energy components."""
    plt, _ = load_matplotlib()
    time = to_numpy(history["time"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(time, to_numpy(history["total_energy"]), label="Total energy")
    ax.plot(time, to_numpy(history["stretching_energy"]), label="Stretching energy")
    ax.plot(time, to_numpy(history["bending_energy"]), label="Bending energy")
    ax.plot(time, to_numpy(history["kinetic_energy"]), label="Kinetic energy")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.grid(True, alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_stability_metrics(history: dict[str, np.ndarray | bool | int | None]) -> None:
    """Plot saved stability metrics."""
    plt, _ = load_matplotlib()
    time = to_numpy(history["time"])
    fig, axes = plt.subplots(3, 1, figsize=(9, 8))
    axes[0].plot(time, to_numpy(history["max_stretch_ratio"]))
    axes[0].set_ylabel("Max stretch ratio")
    axes[0].grid(True, alpha=0.35)
    axes[1].plot(time, to_numpy(history["min_edge_length"]))
    axes[1].set_ylabel("Min edge length (m)")
    axes[1].grid(True, alpha=0.35)
    axes[2].plot(time, to_numpy(history["energy_growth_ratio"]))
    axes[2].set_ylabel("Energy growth ratio")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()
