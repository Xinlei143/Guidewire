"""
Two-dimensional discrete elastic rod with node-wise PI soft contact.

The model is a 1 m long, 1 mm diameter steel/iron guidewire between two
horizontal walls.  Elastic stretching and curvature-binormal bending energies
are differentiated by PyTorch autograd.  Every time step is solved with an
implicit-Euler residual and damped Newton iterations.

Contact is force-only: there is no projection, position correction, bounce, or
hard contact constraint.  Before first contact, the tip x coordinate is driven
toward the fixed end after a short transverse pulse.  On first entry into a
wall protection layer that kinematic constraint is removed, without resetting
the tip velocity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch


DTYPE = torch.float64

# Geometry and material
TOTAL_LENGTH = 1.0
DIAMETER = 0.001
EA = 1.5e5
EI = 1.0e-2
MATERIAL_DENSITY = 7850.0
NUM_NODES = 30

# Walls and node-wise PI contact
WALL_HALF_GAP = 5.0e-3
SAFE_DISTANCE = 2.0e-3
KP = 400.0
KI = 200.0
MAX_PI_FORCE = 10.0
INTEGRAL_LEAK_RATE = 5.0

# Dynamics: resolve the short contact transient accurately, then use a larger
# step after every node has left the protection layer.
DT = 1.0e-3
RELAX_DT = 1.0e-2
MIN_DT = 2.5e-4
CONTACT_CLEAR_HOLD_TIME = 5.0e-2
TOTAL_TIME = 5.0
DAMPING = 3e-3
SETTLE_SPEED_TOL = 1.0e-7
SETTLE_RESIDUAL_TOL = 1.0e-9
SETTLE_HOLD_TIME = 0.10
PULSE_FORCE = 1.0e-2
PULSE_DURATION = 1.0e-2
TIP_VELOCITY = -0.2


@dataclass
class NewtonInfo:
    converged: bool
    iterations: int
    residual_norm: float
    step_norm: float
    line_search_failures: int


def initialize_rod(
    total_length: float = TOTAL_LENGTH,
    num_nodes: int = NUM_NODES,
    diameter: float = DIAMETER,
    density: float = MATERIAL_DENSITY,
):
    """Create a horizontal 2-D rod and lump its physical steel mass to nodes."""
    if num_nodes < 4:
        raise ValueError("num_nodes must be at least 4")
    positions = np.zeros((num_nodes, 2), dtype=np.float64)
    positions[:, 0] = np.linspace(0.0, total_length, num_nodes)
    rest_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    area = np.pi * diameter**2 / 4.0
    total_mass = density * area * total_length
    segment_mass = total_mass / (num_nodes - 1)
    node_masses = np.full(num_nodes, segment_mass, dtype=np.float64)
    node_masses[[0, -1]] *= 0.5
    velocities = np.zeros_like(positions)
    integral_states = np.zeros(num_nodes, dtype=np.float64)
    return positions, velocities, rest_lengths, node_masses, integral_states


def _positions_3d(positions_2d: torch.Tensor) -> torch.Tensor:
    """Embed a planar rod in 3-D so the DER curvature-binormal formula applies."""
    zeros = torch.zeros((positions_2d.shape[0], 1), dtype=positions_2d.dtype)
    return torch.cat((positions_2d, zeros), dim=1)


def compute_elastic_energy_torch(
    positions_2d: torch.Tensor,
    rest_lengths: torch.Tensor,
    EA_value: float,
    EI_value: float,
):
    """Return stretching, bending, and total DER energy for a planar rod."""
    positions = _positions_3d(positions_2d)
    edges = positions[1:] - positions[:-1]
    lengths = torch.linalg.norm(edges, dim=1).clamp_min(1.0e-12)
    tangents = edges / lengths[:, None]

    stretch = 0.5 * EA_value * torch.sum((lengths - rest_lengths) ** 2 / rest_lengths)
    cross = torch.linalg.cross(tangents[:-1], tangents[1:], dim=1)
    dot = torch.sum(tangents[:-1] * tangents[1:], dim=1)
    kappa_binormal = 2.0 * cross / (1.0 + dot).clamp_min(1.0e-12)[:, None]
    voronoi_lengths = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
    bend = 0.5 * EI_value * torch.sum(
        torch.sum(kappa_binormal**2, dim=1) / voronoi_lengths
    )
    return stretch, bend, stretch + bend


def _contact_quantities_torch(
    positions: torch.Tensor,
    old_integrals: torch.Tensor,
    dt: float,
    wall_half_gap: float,
    safe_distance: float,
    kp: float,
    ki: float,
    max_force: float,
    leak_rate: float,
):
    """Differentiable, node-wise PI contact evaluated at candidate positions."""
    y = positions[:, 1]
    # Signed inward clearance: positive inside the channel and negative after
    # crossing a real wall.  This makes error continue growing on penetration.
    distances = wall_half_gap - torch.abs(y)
    errors = torch.clamp(safe_distance - distances, min=0.0)
    active = errors > 0.0
    leaked = old_integrals * np.exp(-leak_rate * dt)
    new_integrals = torch.where(active, old_integrals + errors * dt, leaked)
    magnitudes = torch.clamp(kp * errors + ki * new_integrals, 0.0, max_force)
    directions_y = -torch.sign(y)
    forces = torch.zeros_like(positions)
    forces[:, 1] = torch.where(active, directions_y * magnitudes, torch.zeros_like(y))
    return forces, new_integrals, active, errors, distances


def compute_PI_contact_force(
    positions: np.ndarray,
    integral_states: np.ndarray,
    dt: float,
    wall_half_gap: float = WALL_HALF_GAP,
    safe_distance: float = SAFE_DISTANCE,
    kp: float = KP,
    ki: float = KI,
    max_force: float = MAX_PI_FORCE,
    leak_rate: float = INTEGRAL_LEAK_RATE,
    update_integral: bool = True,
):
    """Compute independent PI normal forces and states for every node.

    Inactive nodes always receive exactly zero force.  Their integrals leak
    exponentially instead of being reset.
    """
    p = torch.as_tensor(positions, dtype=DTYPE)
    integrals = torch.as_tensor(integral_states, dtype=DTYPE)
    with torch.no_grad():
        force, new_i, active, error, distance = _contact_quantities_torch(
            p, integrals, dt, wall_half_gap, safe_distance,
            kp, ki, max_force, leak_rate,
        )
    result_integrals = new_i.numpy() if update_integral else integral_states.copy()
    return (
        force.numpy(), result_integrals, active.numpy(), error.numpy(), distance.numpy()
    )


def apply_tip_velocity_boundary(
    prescribed_values: np.ndarray,
    old_positions: np.ndarray,
    dt: float,
    tip_velocity: float,
):
    """Advance only the prescribed tip x coordinate by constant velocity."""
    prescribed_values = prescribed_values.copy()
    prescribed_values[-1, 0] = old_positions[-1, 0] + tip_velocity * dt
    return prescribed_values


def detect_contact_event(active_mask: np.ndarray) -> bool:
    """Return True when any node is inside either protection layer."""
    return bool(np.any(active_mask))


def _pack_dofs(values: torch.Tensor, free_mask: torch.Tensor) -> torch.Tensor:
    return values[free_mask]


def _unpack_dofs(
    q: torch.Tensor,
    template: torch.Tensor,
    free_mask: torch.Tensor,
    prescribed_values: torch.Tensor,
):
    positions = prescribed_values.clone()
    positions[free_mask] = q
    return positions


def _implicit_residual(
    q: torch.Tensor,
    old_positions: torch.Tensor,
    old_velocities: torch.Tensor,
    rest_lengths: torch.Tensor,
    node_masses: torch.Tensor,
    pulse_forces: torch.Tensor,
    old_integrals: torch.Tensor,
    free_mask: torch.Tensor,
    prescribed_values: torch.Tensor,
    dt: float,
    damping: float,
    EA_value: float,
    EI_value: float,
    wall_half_gap: float,
    safe_distance: float,
    kp: float,
    ki: float,
    max_pi_force: float,
    integral_leak_rate: float,
    create_graph: bool,
):
    positions_new = _unpack_dofs(q, old_positions, free_mask, prescribed_values)
    _, _, energy = compute_elastic_energy_torch(
        positions_new, rest_lengths, EA_value, EI_value
    )
    elastic_gradient = torch.autograd.grad(
        energy, q, create_graph=create_graph, retain_graph=True
    )[0]

    pi_forces, _, _, _, _ = _contact_quantities_torch(
        positions_new, old_integrals, dt, wall_half_gap, safe_distance,
        kp, ki, max_pi_force, integral_leak_rate,
    )
    external_forces = pulse_forces + pi_forces
    q_old = _pack_dofs(old_positions, free_mask)
    v_old = _pack_dofs(old_velocities, free_mask)
    masses = node_masses[:, None].expand_as(old_positions)[free_mask]
    v_new = (q - q_old) / dt
    inertia = masses * (q - q_old - dt * v_old) / dt**2
    # M*a + elastic_gradient + damping*v - external_force = 0
    return inertia + elastic_gradient + damping * v_new - external_forces[free_mask]


def _residual_and_jacobian(q: torch.Tensor, residual_args):
    q_work = q.detach().clone().requires_grad_(True)
    residual = _implicit_residual(q_work, *residual_args, create_graph=True)
    rows = []
    for residual_i in residual:
        (row,) = torch.autograd.grad(residual_i, q_work, retain_graph=True)
        rows.append(row)
    return residual.detach(), torch.stack(rows).detach()


def _residual_norm(q: torch.Tensor, residual_args) -> float:
    q_work = q.detach().clone().requires_grad_(True)
    residual = _implicit_residual(q_work, *residual_args, create_graph=False)
    return float(torch.linalg.norm(residual.detach()))


def implicit_euler_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    rest_lengths: np.ndarray,
    node_masses: np.ndarray,
    pulse_forces: np.ndarray,
    integral_states: np.ndarray,
    prescribed_mask: np.ndarray,
    prescribed_values: np.ndarray,
    dt: float,
    damping: float,
    EA_value: float = EA,
    EI_value: float = EI,
    wall_half_gap: float = WALL_HALF_GAP,
    safe_distance: float = SAFE_DISTANCE,
    kp: float = KP,
    ki: float = KI,
    max_pi_force: float = MAX_PI_FORCE,
    integral_leak_rate: float = INTEGRAL_LEAK_RATE,
    max_newton_iters: int = 20,
    residual_tol: float = 1.0e-7,
    step_tol: float = 1.0e-10,
):
    """Solve one implicit-Euler step with Newton and residual line search."""
    old_p = torch.as_tensor(positions, dtype=DTYPE)
    old_v = torch.as_tensor(velocities, dtype=DTYPE)
    rest = torch.as_tensor(rest_lengths, dtype=DTYPE)
    masses = torch.as_tensor(node_masses, dtype=DTYPE)
    pulse = torch.as_tensor(pulse_forces, dtype=DTYPE)
    integrals = torch.as_tensor(integral_states, dtype=DTYPE)
    prescribed = torch.as_tensor(prescribed_values, dtype=DTYPE)
    prescribed_mask_t = torch.as_tensor(prescribed_mask, dtype=torch.bool)
    free_mask = ~prescribed_mask_t

    # Implicit-Euler predictor is particularly important when the tip has just
    # been released: its current velocity is retained in this initial guess.
    predictor = old_p + dt * old_v
    predictor[prescribed_mask_t] = prescribed[prescribed_mask_t]
    q = _pack_dofs(predictor, free_mask).detach()
    residual_args = (
        old_p, old_v, rest, masses, pulse, integrals, free_mask, prescribed,
        dt, damping, EA_value, EI_value, wall_half_gap, safe_distance,
        kp, ki, max_pi_force, integral_leak_rate,
    )

    residual_norm = _residual_norm(q, residual_args)
    converged = residual_norm < residual_tol
    step_norm = 0.0
    iterations = 0
    line_search_failures = 0
    for iteration in range(1, max_newton_iters + 1):
        if converged:
            break
        residual, jacobian = _residual_and_jacobian(q, residual_args)
        try:
            delta = torch.linalg.solve(jacobian, -residual)
        except RuntimeError:
            delta = torch.linalg.lstsq(jacobian, -residual).solution
        step_norm = float(torch.linalg.norm(delta))
        iterations = iteration
        if step_norm < step_tol:
            converged = True
            break

        current_norm = float(torch.linalg.norm(residual))
        accepted = False
        alpha = 1.0
        for _ in range(12):
            trial = (q + alpha * delta).detach()
            trial_norm = _residual_norm(trial, residual_args)
            if np.isfinite(trial_norm) and trial_norm < current_norm:
                q, residual_norm, accepted = trial, trial_norm, True
                break
            alpha *= 0.5
        if not accepted:
            line_search_failures += 1
            q = (q + 0.05 * delta).detach()
            residual_norm = _residual_norm(q, residual_args)
        converged = residual_norm < residual_tol

    new_p_t = _unpack_dofs(q, old_p, free_mask, prescribed).detach()
    new_v_t = (new_p_t - old_p) / dt
    # Fully fixed base nodes have zero velocity.  The prescribed tip x velocity
    # remains whatever displacement/dt imposed; it is not zeroed on release.
    new_v_t[0:2] = 0.0
    new_p = new_p_t.numpy()
    new_v = new_v_t.numpy()
    pi_force, new_integrals, active, error, distance = compute_PI_contact_force(
        new_p, integral_states, dt, wall_half_gap, safe_distance,
        kp, ki, max_pi_force, integral_leak_rate, update_integral=True,
    )
    info = NewtonInfo(
        converged, iterations, residual_norm, step_norm, line_search_failures
    )
    return new_p, new_v, new_integrals, pi_force, active, error, distance, info


def _make_prescribed_state(initial_positions: np.ndarray, tip_driven: bool):
    mask = np.zeros_like(initial_positions, dtype=bool)
    mask[0:2, :] = True
    if tip_driven:
        mask[-1, 0] = True
    return mask, initial_positions.copy()


def run_simulation(args: argparse.Namespace):
    positions, velocities, rest_lengths, masses, integrals = initialize_rod(
        args.total_length, args.num_nodes, args.diameter, args.density
    )
    initial_positions = positions.copy()
    pulse_node = args.pulse_node if args.pulse_node is not None else args.num_nodes // 2
    tip_driven = False
    contact_event_time = np.nan
    relaxation_phase = False
    contact_clear_time = 0.0
    time_value = 0.0
    step = 0
    settled_time = 0.0
    next_progress_time = args.total_time / 10.0

    history = {
        "time": [0.0],
        "time_step": [0.0],
        "positions": [positions.copy()],
        "velocities": [velocities.copy()],
        "pi_forces": [np.zeros_like(positions)],
        "pi_integrals": [integrals.copy()],
        "contact_active": [np.zeros(args.num_nodes, dtype=bool)],
        "max_protection_error": [0.0],
        "max_pi_force": [0.0],
        "newton_iterations": [0],
        "newton_residual": [0.0],
        "newton_converged": [True],
        "tip_velocity_constraint_active": [False],
    }

    print("=" * 76)
    print("2-D DER + implicit Euler/Newton + node-wise PI soft contact")
    print("=" * 76)
    print(
        f"nodes={args.num_nodes}, length={args.total_length:g} m, diameter={args.diameter:g} m, "
        f"mass={np.sum(masses):.6g} kg"
    )
    print(
        f"EA={args.EA:g} N, EI={args.EI:g} N*m^2, "
        f"transient_dt={args.dt:g} s, relax_dt={args.relax_dt:g} s, min_dt={args.min_dt:g} s, "
        f"max_time={args.total_time:g} s"
    )
    print(
        f"walls=+/-{args.wall_half_gap:g} m, safe_distance={args.safe_distance:g} m, "
        f"Kp={args.Kp:g}, Ki={args.Ki:g}, cap={args.max_pi_force:g} N"
    )

    failed_steps = 0
    wall_start = time.perf_counter()
    while time_value < args.total_time - 1.0e-15:
        # A large step is permitted only after the rod has stayed clear of both
        # protection layers for a while.  A renewed contact immediately returns
        # the simulation to the small-step phase.
        dt_step = args.relax_dt if relaxation_phase else args.dt
        if time_value < args.pulse_duration:
            dt_step = min(dt_step, args.pulse_duration - time_value)
        dt_step = min(dt_step, args.total_time - time_value)
        pulse_active = time_value < args.pulse_duration - 1.0e-15
        if not pulse_active and not np.isfinite(contact_event_time):
            tip_driven = True
        tip_constraint_used = tip_driven

        pulse_forces = np.zeros_like(positions)
        if pulse_active:
            pulse_forces[pulse_node, 1] = args.pulse_force

        positions_before = positions.copy()
        velocities_before = velocities.copy()
        integrals_before = integrals.copy()

        def solve_trial(trial_dt: float, residual_tol: float):
            prescribed_mask, prescribed_values = _make_prescribed_state(
                initial_positions, tip_driven
            )
            prescribed_values[0:2] = initial_positions[0:2]
            if tip_driven:
                prescribed_values = apply_tip_velocity_boundary(
                    prescribed_values, positions, trial_dt, args.tip_velocity
                )
            return implicit_euler_step(
                positions, velocities, rest_lengths, masses, pulse_forces, integrals,
                prescribed_mask, prescribed_values, trial_dt, args.damping,
                args.EA, args.EI, args.wall_half_gap, args.safe_distance,
                args.Kp, args.Ki, args.max_pi_force, args.integral_leak_rate,
                args.max_newton_iters, residual_tol, args.step_tol,
            )

        solver_residual_tol = (
            min(args.residual_tol, args.settle_residual_tol)
            if relaxation_phase else args.residual_tol
        )
        (
            positions, velocities, integrals, pi_forces, active, errors,
            _distances, info,
        ) = solve_trial(dt_step, solver_residual_tol)

        # Never accept a large-step entry into the protection layer.  Restore
        # the exact old state and resolve that same interval with the small dt.
        if relaxation_phase and (detect_contact_event(active) or not info.converged):
            positions, velocities, integrals = (
                positions_before.copy(), velocities_before.copy(), integrals_before.copy()
            )
            dt_step = min(args.dt, args.total_time - time_value)
            print(f"adaptive retry at t={time_value:.6f} s: returning to transient_dt={dt_step:g} s")
            (
                positions, velocities, integrals, pi_forces, active, errors,
                _distances, info,
            ) = solve_trial(dt_step, args.residual_tol)
            relaxation_phase = False
            contact_clear_time = 0.0

        # If Newton still fails, retry the same interval with progressively
        # smaller steps instead of accepting a large residual.
        while not info.converged and dt_step > args.min_dt + 1.0e-15:
            positions, velocities, integrals = (
                positions_before.copy(), velocities_before.copy(), integrals_before.copy()
            )
            dt_step = max(args.min_dt, 0.5 * dt_step)
            print(f"Newton retry at t={time_value:.6f} s with dt={dt_step:g} s")
            (
                positions, velocities, integrals, pi_forces, active, errors,
                _distances, info,
            ) = solve_trial(dt_step, args.residual_tol)

        t_new = time_value + dt_step
        step += 1

        if detect_contact_event(active) and not np.isfinite(contact_event_time):
            contact_event_time = t_new
            tip_driven = False
            print(f"first protection-layer contact at t={t_new:.6f} s; tip x constraint released")

        if np.isfinite(contact_event_time):
            if detect_contact_event(active):
                contact_clear_time = 0.0
                relaxation_phase = False
            else:
                contact_clear_time += dt_step
                if (
                    not relaxation_phase
                    and contact_clear_time >= args.contact_clear_hold_time
                ):
                    relaxation_phase = True
                    print(
                        f"clear of protection layers for {contact_clear_time:.3f} s at t={t_new:.6f} s; "
                        f"switching to relax_dt={args.relax_dt:g} s"
                    )

        if not info.converged:
            failed_steps += 1
        history["time"].append(t_new)
        history["time_step"].append(dt_step)
        history["positions"].append(positions.copy())
        history["velocities"].append(velocities.copy())
        history["pi_forces"].append(pi_forces.copy())
        history["pi_integrals"].append(integrals.copy())
        history["contact_active"].append(active.copy())
        history["max_protection_error"].append(float(np.max(errors)))
        history["max_pi_force"].append(float(np.max(np.linalg.norm(pi_forces, axis=1))))
        history["newton_iterations"].append(info.iterations)
        history["newton_residual"].append(info.residual_norm)
        history["newton_converged"].append(info.converged)
        history["tip_velocity_constraint_active"].append(tip_constraint_used)

        max_speed = float(np.max(np.linalg.norm(velocities[2:], axis=1)))
        unloaded = (
            relaxation_phase
            and not pulse_active
            and not tip_driven
            and not detect_contact_event(active)
        )
        is_settled = (
            unloaded
            and max_speed <= args.settle_speed_tol
            and info.residual_norm <= args.settle_residual_tol
        )
        settled_time = settled_time + dt_step if is_settled else 0.0

        if t_new >= next_progress_time - 1.0e-15 or time_value + dt_step >= args.total_time - 1.0e-15:
            print(
                f"step {step:4d}, t={t_new:.3f}, dt={dt_step:.3g}, iters={info.iterations:2d}, "
                f"res={info.residual_norm:.2e}, max_error={np.max(errors):.3e}, "
                f"max_PI={history['max_pi_force'][-1]:.3e}, max_speed={max_speed:.3e}"
            )
            while next_progress_time <= t_new + 1.0e-15:
                next_progress_time += args.total_time / 10.0

        time_value = t_new
        if not args.disable_auto_stop and settled_time >= args.settle_hold_time:
            print(
                f"settled at t={time_value:.6f} s after {settled_time:.3f} s below "
                "the velocity and residual thresholds"
            )
            break

    for key in history:
        history[key] = np.asarray(history[key])
    history["initial_positions"] = initial_positions
    history["contact_event_time"] = np.asarray(contact_event_time)
    elapsed = time.perf_counter() - wall_start
    print(f"done in {elapsed:.2f} s; unconverged steps={failed_steps}")

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_prefix.with_suffix(".npz")
    np.savez_compressed(npz_path, **history)
    print(f"history saved: {npz_path}")
    visualize_results(history, args)
    return history


def visualize_results(history, args: argparse.Namespace):
    """Show the live animation, then summary and final-frame figures; save none."""
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if not args.show:
        return
    positions = history["positions"]
    time_values = history["time"]
    max_bend_index = int(np.argmax(np.ptp(positions[:, :, 1], axis=1)))
    protection_y = args.wall_half_gap - args.safe_distance

    if not args.skip_animation:
        fig_anim, ax = plt.subplots(figsize=(12, 3.8))
        rod_line, = ax.plot([], [], "b-o", ms=3, lw=1.5)
        fixed_scatter = ax.scatter([], [], marker="s", c="red", s=45, zorder=4)
        contact_scatter = ax.scatter([], [], c="magenta", s=32, zorder=5)
        title = ax.set_title("")
        for wall in (-args.wall_half_gap, args.wall_half_gap):
            ax.axhline(wall, color="black", lw=1.3)
        for layer in (-protection_y, protection_y):
            ax.axhline(layer, color="tab:red", ls=":", lw=1.1)
        ax.set_xlim(-0.02, args.total_length + 0.02)
        ax.set_ylim(-1.35 * args.wall_half_gap, 1.35 * args.wall_half_gap)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(alpha=0.2)

        frame_indices = np.arange(0, len(time_values), max(1, args.animation_stride))
        if frame_indices[-1] != len(time_values) - 1:
            frame_indices = np.append(frame_indices, len(time_values) - 1)

        def update(frame_number):
            index = int(frame_indices[frame_number])
            p = positions[index]
            rod_line.set_data(p[:, 0], p[:, 1])
            fixed_scatter.set_offsets(p[:2])
            contact_scatter.set_offsets(p[history["contact_active"][index]])
            title.set_text(f"Node-wise PI soft contact, t={time_values[index]:.3f} s")
            return rod_line, fixed_scatter, contact_scatter, title

        animation = FuncAnimation(
            fig_anim,
            update,
            frames=len(frame_indices),
            interval=1000.0 / args.animation_fps,
            blit=False,
        )
        plt.show()
        plt.close(fig_anim)

    fig, (ax_shape, ax_hist) = plt.subplots(2, 1, figsize=(12, 8))
    ax_shape.plot(positions[0, :, 0], positions[0, :, 1], "k--", label="initial")
    ax_shape.plot(
        positions[max_bend_index, :, 0], positions[max_bend_index, :, 1],
        color="tab:orange", label=f"maximum bend (t={time_values[max_bend_index]:.3f}s)",
    )
    ax_shape.plot(positions[-1, :, 0], positions[-1, :, 1], "b-o", ms=2.5, label="final")
    ax_shape.scatter(positions[-1, :2, 0], positions[-1, :2, 1], marker="s", c="red", label="fixed")
    for wall in (-args.wall_half_gap, args.wall_half_gap):
        ax_shape.axhline(wall, color="black", lw=1.4)
    for layer in (-protection_y, protection_y):
        ax_shape.axhline(layer, color="tab:red", ls=":", lw=1.1)
    ax_shape.set(xlabel="x (m)", ylabel="y (m)", title="Initial, maximum-bending, and final states")
    ax_shape.set_ylim(-1.35 * args.wall_half_gap, 1.35 * args.wall_half_gap)
    ax_shape.grid(alpha=0.25)
    ax_shape.legend(loc="upper right", ncol=3)

    ax_hist.plot(time_values, history["max_protection_error"] * 1e3, label="max error (mm)")
    ax_hist.plot(time_values, history["max_pi_force"], label="max PI force (N)")
    ax_hist.plot(time_values, history["newton_iterations"], label="Newton iterations")
    ax_hist.set(xlabel="time (s)", title="Contact and solver histories")
    ax_hist.grid(alpha=0.25)
    ax_hist.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    final_displacement = np.linalg.norm(positions[-1] - positions[0], axis=1)
    fig_final, ax_final = plt.subplots(figsize=(12, 3.8))
    ax_final.plot(positions[-1, :, 0], positions[-1, :, 1], "b-o", ms=3, lw=1.5)
    ax_final.scatter(positions[-1, :2, 0], positions[-1, :2, 1], marker="s", c="red", s=45, label="fixed")
    for wall in (-args.wall_half_gap, args.wall_half_gap):
        ax_final.axhline(wall, color="black", lw=1.3)
    for layer in (-protection_y, protection_y):
        ax_final.axhline(layer, color="tab:red", ls=":", lw=1.1)
    ax_final.set(
        xlim=(-0.02, args.total_length + 0.02),
        ylim=(-1.35 * args.wall_half_gap, 1.35 * args.wall_half_gap),
        xlabel="x (m)",
        ylabel="y (m)",
        title=(
            f"Final frame: t={time_values[-1]:.3f} s, "
            f"max displacement={np.max(final_displacement) * 1e3:.3f} mm"
        ),
    )
    ax_final.grid(alpha=0.2)
    ax_final.legend()
    fig_final.tight_layout()
    plt.show()
    plt.close(fig_final)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-length", type=float, default=TOTAL_LENGTH)
    parser.add_argument("--diameter", type=float, default=DIAMETER)
    parser.add_argument("--density", type=float, default=MATERIAL_DENSITY)
    parser.add_argument("--num-nodes", type=int, default=NUM_NODES)
    parser.add_argument("--EA", type=float, default=EA)
    parser.add_argument("--EI", type=float, default=EI)
    parser.add_argument(
        "--dt",
        type=float,
        default=DT,
        help="small time step used before and during the first contact episode",
    )
    parser.add_argument(
        "--relax-dt",
        type=float,
        default=RELAX_DT,
        help="larger time step used after a sustained contact-free interval",
    )
    parser.add_argument(
        "--min-dt",
        type=float,
        default=MIN_DT,
        help="smallest retry time step after a Newton failure",
    )
    parser.add_argument(
        "--contact-clear-hold-time",
        type=float,
        default=CONTACT_CLEAR_HOLD_TIME,
        help="contact-free time required before switching to relax_dt (s)",
    )
    parser.add_argument("--total-time", type=float, default=TOTAL_TIME)
    parser.add_argument("--damping", type=float, default=DAMPING)
    parser.add_argument("--pulse-force", type=float, default=PULSE_FORCE)
    parser.add_argument("--pulse-duration", type=float, default=PULSE_DURATION)
    parser.add_argument("--pulse-node", type=int)
    parser.add_argument("--tip-velocity", type=float, default=TIP_VELOCITY)
    parser.add_argument("--wall-half-gap", type=float, default=WALL_HALF_GAP)
    parser.add_argument("--safe-distance", type=float, default=SAFE_DISTANCE)
    parser.add_argument("--Kp", type=float, default=KP)
    parser.add_argument("--Ki", type=float, default=KI)
    parser.add_argument("--max-pi-force", type=float, default=MAX_PI_FORCE)
    parser.add_argument("--integral-leak-rate", type=float, default=INTEGRAL_LEAK_RATE)
    parser.add_argument("--max-newton-iters", type=int, default=40)
    parser.add_argument("--residual-tol", type=float, default=1.0e-7)
    parser.add_argument("--step-tol", type=float, default=1.0e-10)
    parser.add_argument(
        "--settle-speed-tol",
        type=float,
        default=SETTLE_SPEED_TOL,
        help="automatic-stop maximum free-node speed threshold (m/s)",
    )
    parser.add_argument(
        "--settle-residual-tol",
        type=float,
        default=SETTLE_RESIDUAL_TOL,
        help="automatic-stop Newton residual threshold (N)",
    )
    parser.add_argument(
        "--settle-hold-time",
        type=float,
        default=SETTLE_HOLD_TIME,
        help="time for which both settling conditions must hold before stopping (s)",
    )
    parser.add_argument(
        "--disable-auto-stop",
        action="store_true",
        help="always integrate through total_time even after the settling criteria are met",
    )
    parser.add_argument("--output-prefix", default="rod_soft_contact_PI_dynamic")
    parser.add_argument(
        "--animation-stride",
        type=int,
        default=1,
        help="show one frame every N time steps; default 1 keeps every step",
    )
    parser.add_argument(
        "--animation-fps",
        type=float,
        default=8.0,
        help="live animation speed in frames per second",
    )
    parser.add_argument("--skip-animation", action="store_true")
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        default=True,
        help="run without opening the animation or figures",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args(argv)
    if args.num_nodes != 30:
        parser.error("this test model is specified to use exactly 30 nodes")
    if args.dt <= 0 or args.relax_dt <= 0 or args.min_dt <= 0 or args.total_time <= 0:
        parser.error("dt, relax_dt, min_dt, and total_time must be positive")
    if args.min_dt > args.dt:
        parser.error("min_dt must not exceed dt")
    if args.contact_clear_hold_time <= 0:
        parser.error("contact_clear_hold_time must be positive")
    if args.settle_speed_tol <= 0 or args.settle_residual_tol <= 0 or args.settle_hold_time <= 0:
        parser.error("all settling thresholds and settle_hold_time must be positive")
    if not 0 < args.safe_distance < args.wall_half_gap:
        parser.error("safe_distance must be between zero and wall_half_gap")
    if args.tip_velocity >= 0:
        parser.error("tip_velocity must be negative for contraction toward the fixed end")
    if args.animation_stride <= 0 or args.animation_fps <= 0:
        parser.error("animation_stride and animation_fps must be positive")
    if args.pulse_node is not None and not 2 <= args.pulse_node < args.num_nodes:
        parser.error("pulse_node must be a free node from 2 through num_nodes-1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke_test:
        import sys
        from ._smoke import run_contact_smoke

        run_contact_smoke(
            sys.modules[__name__], 'contact-2d-dynamic-pi', args.output_dir,
            quasistatic=False,
        )
        return 0
    run_simulation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
