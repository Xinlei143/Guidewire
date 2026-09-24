"""
3D discrete elastic rod dynamics with implicit Euler and Newton iterations.

This is a first implicit-dynamics prototype.  It intentionally does not modify
the existing semi-implicit Euler scripts.  The elastic model is reused from
rodtoolbox_ks_kb_torch3d.py:

- extensible edges with stretching energy
- curvature-binormal bending energy
- PyTorch autograd for elastic gradients

The time update is different: each step solves the implicit Euler residual for
free-node positions with damped Newton and a simple residual-decreasing line
search.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np
import torch

from .toolbox_3d_torch import (
    compute_bending_angles_degrees,
    compute_curvature_binormal_magnitudes,
    compute_initial_curvature_binormal,
    compute_total_elastic_energy_torch,
)


DTYPE = torch.float64


# ==================== 用户参数设置区 ====================
# 你平时主要改这里即可；下面的命令行参数只是辅助覆盖这些默认值。

# --- 几何参数 ---
default_total_length = 50.0          # 导管/杆中心线总长度
default_num_segments = 50            # 离散段数；节点数 = default_num_segments + 1

# --- 材料参数 ---
default_EA = 1000.0                  # 轴向拉伸刚度
default_EI = 200.0                   # 弯曲刚度

# --- 质量参数 ---
# "total": 固定自由节点总质量，适合做不同离散段数的收敛比较
# "density": 固定线密度，total_mass = mass_density * total_length
# "node": 固定每个自由节点质量，和旧版脚本一致；段数增加时总质量也增加
default_mass_mode = "total"
default_total_mass = 15.0            # mass_mode="total" 时，自由节点总质量
default_mass_density = 0.3           # mass_mode="density" 时，单位长度质量
default_node_mass = 0.5              # mass_mode="node" 时，每个自由节点质量

# --- 动力学参数 ---
default_dt = 0.01                   # 隐式欧拉时间步长
default_total_time = 5            # 总仿真时间；第一版建议先短时间跑通
default_damping = 0.5                # 速度阻尼系数
default_gravity = -9.81              # 重力加速度，负 Z 方向可设为 -9.81

# --- 外力参数 ---
default_force_mode = "constant"      # "none", "constant", "pulse"
default_force_magnitude = 100        # 外力大小
default_force_direction = [0, 1, 0]  # 外力方向 [Fx, Fy, Fz]，程序会自动归一化
default_force_node = None            # None 表示默认施加在末端节点；也可填具体节点编号
default_pulse_duration = 0.002       # pulse 模式下的持续时间

# --- Newton 求解参数 ---
default_max_newton_iters = 15        # 每个时间步最多 Newton 迭代次数
default_residual_tol = 1e-7          # 残差收敛阈值
default_step_tol = 1e-10             # 位移增量收敛阈值
default_jacobian_mode = "residual_jacobian"  # "energy_hessian" or "residual_jacobian"

# --- 输出/可视化参数 ---
default_save_every = 1               # 每隔多少步记录一次状态
default_show_final = True            # 显示三维最终形状和位移图
default_show_history = False         # 显示能量、Newton 残差等历史曲线
default_save_npz = None              # 如 "implicit_history.npz"，None 表示不保存

# ========================================================


@dataclass
class NewtonInfo:
    converged: bool
    iterations: int
    residual_norm: float
    step_norm: float
    line_search_failures: int
    jacobian_time: float = 0.0
    solve_time: float = 0.0
    line_search_time: float = 0.0


def create_force_function(force_mode: str, magnitude: float, direction: Iterable[float], duration: float = 1.0):
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    unit_dir = direction / norm if norm > 0 else np.array([1.0, 0.0, 0.0])

    if force_mode == "none":
        return lambda t: 0.0, unit_dir
    if force_mode == "constant":
        return lambda t: magnitude, unit_dir
    if force_mode == "pulse":
        return lambda t: magnitude if t < duration else 0.0, unit_dir
    raise ValueError(f"Unknown force_mode: {force_mode}")


def initialize_rod(
    total_length: float,
    num_segments: int,
    mass_mode: str,
    total_mass: float,
    mass_density: float,
    node_mass: float,
):
    num_nodes = num_segments + 1
    segment_length = total_length / num_segments

    positions = np.zeros((num_nodes, 3), dtype=np.float64)
    positions[:, 0] = np.arange(num_nodes) * segment_length

    rest_lengths = np.full(num_segments, segment_length, dtype=np.float64)
    node_masses = np.zeros(num_nodes, dtype=np.float64)
    num_free_nodes = num_nodes - 2
    if mass_mode == "total":
        free_node_mass = total_mass / num_free_nodes
    elif mass_mode == "density":
        free_node_mass = mass_density * total_length / num_free_nodes
    elif mass_mode == "node":
        free_node_mass = node_mass
    else:
        raise ValueError(f"Unknown mass_mode: {mass_mode}")

    node_masses[2:] = free_node_mass
    node_masses[0] = 0.0
    node_masses[1] = 0.0
    return positions, rest_lengths, node_masses


def build_external_forces(num_nodes: int, force_node: int, force_magnitude: float, force_direction_unit: np.ndarray):
    external_forces = np.zeros((num_nodes, 3), dtype=np.float64)
    if force_magnitude != 0.0:
        external_forces[force_node] = force_magnitude * force_direction_unit
    return external_forces


def _pack_free(positions_t: torch.Tensor, free_indices_t: torch.Tensor) -> torch.Tensor:
    return positions_t[free_indices_t].reshape(-1)


def _unpack_free(
    q_flat: torch.Tensor,
    template_positions: torch.Tensor,
    free_indices_t: torch.Tensor,
) -> torch.Tensor:
    positions = template_positions.clone()
    positions[free_indices_t] = q_flat.reshape(-1, 3)
    return positions


def _implicit_residual_from_q(
    q_flat: torch.Tensor,
    old_positions_t: torch.Tensor,
    old_velocities_t: torch.Tensor,
    rest_lengths_t: torch.Tensor,
    rest_kappa_b_t: torch.Tensor,
    node_masses_t: torch.Tensor,
    external_forces_t: torch.Tensor,
    gravity_g: float,
    damping: float,
    EA: float,
    EI: float,
    dt: float,
    fixed_positions_t: torch.Tensor,
    fixed_indices_t: torch.Tensor,
    free_indices_t: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    positions_new = _unpack_free(q_flat, old_positions_t, free_indices_t)
    positions_new[fixed_indices_t] = fixed_positions_t

    _, _, elastic_energy = compute_total_elastic_energy_torch(
        positions_new, rest_lengths_t, EA, EI, rest_kappa_b_t
    )
    grad_free = torch.autograd.grad(
        elastic_energy,
        q_flat,
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    q_old = _pack_free(old_positions_t, free_indices_t)
    v_old = _pack_free(old_velocities_t, free_indices_t)
    masses_free = node_masses_t[free_indices_t].repeat_interleave(3)

    v_new = (q_flat - q_old) / dt
    inertia = masses_free * (q_flat - q_old - dt * v_old) / (dt * dt)
    damping_force_lhs = damping * v_new

    external_free = _pack_free(external_forces_t, free_indices_t)
    gravity_forces = torch.zeros_like(external_forces_t)
    gravity_forces[:, 2] = node_masses_t * gravity_g
    gravity_free = _pack_free(gravity_forces, free_indices_t)

    # Equation: M*a = f_int(x_new) + f_ext + f_g - damping*v_new.
    # Since f_int = -dE/dx, the residual is:
    # M*a + dE/dx - f_ext - f_g + damping*v_new = 0.
    return inertia + grad_free - external_free - gravity_free + damping_force_lhs


def _residual_and_jacobian(
    q_flat: torch.Tensor,
    *residual_args,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = q_flat.detach().clone().requires_grad_(True)
    residual = _implicit_residual_from_q(q, *residual_args, create_graph=True)

    rows = []
    for residual_i in residual:
        (grad_i,) = torch.autograd.grad(residual_i, q, retain_graph=True)
        rows.append(grad_i)
    jacobian = torch.stack(rows, dim=0)
    return residual.detach(), jacobian.detach()


def _elastic_energy_from_q(
    q_flat: torch.Tensor,
    template_positions: torch.Tensor,
    rest_lengths_t: torch.Tensor,
    rest_kappa_b_t: torch.Tensor,
    EA: float,
    EI: float,
    fixed_positions_t: torch.Tensor,
    fixed_indices_t: torch.Tensor,
    free_indices_t: torch.Tensor,
) -> torch.Tensor:
    positions_new = _unpack_free(q_flat, template_positions, free_indices_t)
    positions_new[fixed_indices_t] = fixed_positions_t
    _, _, elastic_energy = compute_total_elastic_energy_torch(
        positions_new, rest_lengths_t, EA, EI, rest_kappa_b_t
    )
    return elastic_energy


def _residual_and_jacobian_from_energy_hessian(
    q_flat: torch.Tensor,
    old_positions_t: torch.Tensor,
    old_velocities_t: torch.Tensor,
    rest_lengths_t: torch.Tensor,
    rest_kappa_b_t: torch.Tensor,
    node_masses_t: torch.Tensor,
    external_forces_t: torch.Tensor,
    gravity_g: float,
    damping: float,
    EA: float,
    EI: float,
    dt: float,
    fixed_positions_t: torch.Tensor,
    fixed_indices_t: torch.Tensor,
    free_indices_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = q_flat.detach().clone().requires_grad_(True)
    elastic_energy = _elastic_energy_from_q(
        q,
        old_positions_t,
        rest_lengths_t,
        rest_kappa_b_t,
        EA,
        EI,
        fixed_positions_t,
        fixed_indices_t,
        free_indices_t,
    )
    (grad_free,) = torch.autograd.grad(elastic_energy, q, create_graph=True)

    rows = []
    for grad_i in grad_free:
        (hess_i,) = torch.autograd.grad(grad_i, q, retain_graph=True)
        rows.append(hess_i)
    hessian = torch.stack(rows, dim=0)

    q_old = _pack_free(old_positions_t, free_indices_t)
    v_old = _pack_free(old_velocities_t, free_indices_t)
    masses_free = node_masses_t[free_indices_t].repeat_interleave(3)

    diagonal = masses_free / (dt * dt) + damping / dt
    inertia = masses_free * (q - q_old - dt * v_old) / (dt * dt)
    damping_force_lhs = damping * (q - q_old) / dt

    external_free = _pack_free(external_forces_t, free_indices_t)
    gravity_forces = torch.zeros_like(external_forces_t)
    gravity_forces[:, 2] = node_masses_t * gravity_g
    gravity_free = _pack_free(gravity_forces, free_indices_t)

    residual = inertia + grad_free - external_free - gravity_free + damping_force_lhs
    jacobian = hessian + torch.diag(diagonal)
    return residual.detach(), jacobian.detach()


def _residual_norm(q_flat: torch.Tensor, *residual_args) -> float:
    q = q_flat.detach().clone().requires_grad_(True)
    residual = _implicit_residual_from_q(q, *residual_args, create_graph=False)
    return torch.linalg.norm(residual.detach()).item()


def integrate_implicit_euler_newton(
    positions: np.ndarray,
    velocities: np.ndarray,
    rest_lengths: np.ndarray,
    node_masses: np.ndarray,
    EA: float,
    EI: float,
    external_forces: np.ndarray,
    gravity_g: float,
    damping: float,
    dt: float,
    fixed_indices: np.ndarray,
    fixed_positions: np.ndarray,
    rest_kappa_b: np.ndarray,
    max_newton_iters: int = 20,
    residual_tol: float = 1e-7,
    step_tol: float = 1e-9,
    max_line_search_iters: int = 10,
    jacobian_mode: str = "energy_hessian",
) -> tuple[np.ndarray, np.ndarray, NewtonInfo]:
    positions_t = torch.as_tensor(positions, dtype=DTYPE)
    velocities_t = torch.as_tensor(velocities, dtype=DTYPE)
    rest_lengths_t = torch.as_tensor(rest_lengths, dtype=DTYPE)
    rest_kappa_b_t = torch.as_tensor(rest_kappa_b, dtype=DTYPE)
    node_masses_t = torch.as_tensor(node_masses, dtype=DTYPE)
    external_forces_t = torch.as_tensor(external_forces, dtype=DTYPE)
    fixed_positions_t = torch.as_tensor(fixed_positions, dtype=DTYPE)

    fixed_indices_t = torch.as_tensor(fixed_indices, dtype=torch.long)
    all_indices = np.arange(len(positions))
    free_indices = np.setdiff1d(all_indices, fixed_indices)
    free_indices_t = torch.as_tensor(free_indices, dtype=torch.long)

    q = _pack_free(positions_t, free_indices_t).detach().clone()
    residual_args = (
        positions_t,
        velocities_t,
        rest_lengths_t,
        rest_kappa_b_t,
        node_masses_t,
        external_forces_t,
        gravity_g,
        damping,
        EA,
        EI,
        dt,
        fixed_positions_t,
        fixed_indices_t,
        free_indices_t,
    )

    line_search_failures = 0
    residual_norm = _residual_norm(q, *residual_args)
    step_norm = 0.0
    jacobian_time = 0.0
    solve_time = 0.0
    line_search_time = 0.0

    converged = residual_norm < residual_tol
    iterations = 0
    for iteration in range(1, max_newton_iters + 1):
        if converged:
            break

        t0 = time.perf_counter()
        if jacobian_mode == "energy_hessian":
            residual, jacobian = _residual_and_jacobian_from_energy_hessian(q, *residual_args)
        elif jacobian_mode == "residual_jacobian":
            residual, jacobian = _residual_and_jacobian(q, *residual_args)
        else:
            raise ValueError(f"Unknown jacobian_mode: {jacobian_mode}")
        jacobian_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            delta_q = torch.linalg.solve(jacobian, -residual)
        except RuntimeError:
            delta_q = torch.linalg.lstsq(jacobian, -residual).solution
        solve_time += time.perf_counter() - t0

        step_norm = torch.linalg.norm(delta_q).item()
        if step_norm < step_tol:
            converged = True
            iterations = iteration
            break

        accepted = False
        alpha = 1.0
        current_norm = torch.linalg.norm(residual).item()
        best_q = q
        best_norm = current_norm

        t0 = time.perf_counter()
        for _ in range(max_line_search_iters):
            trial_q = q + alpha * delta_q
            trial_norm = _residual_norm(trial_q, *residual_args)
            if np.isfinite(trial_norm) and trial_norm < best_norm:
                best_q = trial_q.detach()
                best_norm = trial_norm
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            line_search_failures += 1
            best_q = q + 0.1 * delta_q
            best_q = best_q.detach()
            best_norm = _residual_norm(best_q, *residual_args)
        line_search_time += time.perf_counter() - t0

        q = best_q
        residual_norm = best_norm
        iterations = iteration
        converged = residual_norm < residual_tol

    new_positions_t = _unpack_free(q, positions_t, free_indices_t).detach()
    new_positions_t[fixed_indices_t] = fixed_positions_t
    new_velocities_t = (new_positions_t - positions_t) / dt
    new_velocities_t[fixed_indices_t] = 0.0

    info = NewtonInfo(
        converged=converged,
        iterations=iterations,
        residual_norm=residual_norm,
        step_norm=step_norm,
        line_search_failures=line_search_failures,
        jacobian_time=jacobian_time,
        solve_time=solve_time,
        line_search_time=line_search_time,
    )
    return new_positions_t.numpy(), new_velocities_t.numpy(), info


def compute_energy_snapshot(
    positions: np.ndarray,
    velocities: np.ndarray,
    rest_lengths: np.ndarray,
    rest_kappa_b: np.ndarray,
    node_masses: np.ndarray,
    EA: float,
    EI: float,
    gravity_g: float,
):
    positions_t = torch.as_tensor(positions, dtype=DTYPE)
    rest_lengths_t = torch.as_tensor(rest_lengths, dtype=DTYPE)
    rest_kappa_b_t = torch.as_tensor(rest_kappa_b, dtype=DTYPE)
    stretch, bend, elastic = compute_total_elastic_energy_torch(
        positions_t, rest_lengths_t, EA, EI, rest_kappa_b_t
    )
    gravity_energy = float(np.sum(node_masses * gravity_g * positions[:, 2]))
    kinetic = float(0.5 * np.sum(node_masses[:, None] * velocities * velocities))
    return {
        "stretch_energy": stretch.item(),
        "bend_energy": bend.item(),
        "elastic_energy": elastic.item(),
        "gravity_energy": gravity_energy,
        "kinetic_energy": kinetic,
        "total_energy": elastic.item() + gravity_energy + kinetic,
    }


def run_implicit_simulation(args: argparse.Namespace):
    wall_start = time.perf_counter()
    positions0, rest_lengths, node_masses = initialize_rod(
        args.total_length,
        args.num_segments,
        args.mass_mode,
        args.total_mass,
        args.mass_density,
        args.node_mass,
    )
    fixed_indices = np.array([0, 1], dtype=int)
    fixed_positions = positions0[fixed_indices].copy()
    rest_kappa_b = compute_initial_curvature_binormal(positions0, rest_lengths)

    force_func, force_direction_unit = create_force_function(
        args.force_mode, args.force_magnitude, args.force_direction, args.pulse_duration
    )

    positions = positions0.copy()
    velocities = np.zeros_like(positions)
    num_steps = int(round(args.total_time / args.dt))
    save_every = max(1, args.save_every)

    history = {
        "time": [],
        "positions": [],
        "velocities": [],
        "newton_iterations": [],
        "newton_residual": [],
        "newton_converged": [],
        "max_displacement": [],
        "max_velocity": [],
        "bending_angles": [],
        "kappa_b_magnitudes": [],
        "stretch_energy": [],
        "bend_energy": [],
        "kinetic_energy": [],
        "total_energy": [],
    }

    def record(t: float, info: NewtonInfo | None):
        energy = compute_energy_snapshot(
            positions, velocities, rest_lengths, rest_kappa_b,
            node_masses, args.EA, args.EI, args.gravity
        )
        free_indices = np.setdiff1d(np.arange(len(positions)), fixed_indices)
        history["time"].append(t)
        history["positions"].append(positions.copy())
        history["velocities"].append(velocities.copy())
        history["newton_iterations"].append(0 if info is None else info.iterations)
        history["newton_residual"].append(0.0 if info is None else info.residual_norm)
        history["newton_converged"].append(True if info is None else info.converged)
        history["max_displacement"].append(float(np.max(np.linalg.norm(positions - positions0, axis=1))))
        history["max_velocity"].append(float(np.max(np.linalg.norm(velocities[free_indices], axis=1))))
        history["bending_angles"].append(compute_bending_angles_degrees(positions))
        history["kappa_b_magnitudes"].append(compute_curvature_binormal_magnitudes(positions))
        history["stretch_energy"].append(energy["stretch_energy"])
        history["bend_energy"].append(energy["bend_energy"])
        history["kinetic_energy"].append(energy["kinetic_energy"])
        history["total_energy"].append(energy["total_energy"])

    print("=" * 72)
    print("3D DER implicit Euler dynamics with Newton line search")
    print("=" * 72)
    print(f"nodes={args.num_segments + 1}, free_dof={(args.num_segments - 1) * 3}")
    print(f"total_length={args.total_length:g}, EA={args.EA:g}, EI={args.EI:g}")
    print(
        f"mass_mode={args.mass_mode}, total_free_mass={np.sum(node_masses):g}, "
        f"free_node_mass={node_masses[2]:g}"
    )
    print(f"dt={args.dt:g}, total_time={args.total_time:g}, steps={num_steps}, damping={args.damping:g}")
    print(
        f"gravity={args.gravity:g}, force={args.force_magnitude:g} on node {args.force_node}, "
        f"direction={force_direction_unit}"
    )
    print(f"jacobian_mode={args.jacobian_mode}")
    print("-" * 72)

    record(0.0, None)

    failed_steps = 0
    newton_iterations_all = []
    jacobian_time_all = []
    solve_time_all = []
    line_search_time_all = []
    for step in range(1, num_steps + 1):
        current_time = step * args.dt
        force_magnitude = force_func(current_time)
        external_forces = build_external_forces(
            len(positions), args.force_node, force_magnitude, force_direction_unit
        )

        positions, velocities, info = integrate_implicit_euler_newton(
            positions,
            velocities,
            rest_lengths,
            node_masses,
            args.EA,
            args.EI,
            external_forces,
            args.gravity,
            args.damping,
            args.dt,
            fixed_indices,
            fixed_positions,
            rest_kappa_b,
            max_newton_iters=args.max_newton_iters,
            residual_tol=args.residual_tol,
            step_tol=args.step_tol,
            jacobian_mode=args.jacobian_mode,
        )

        if not info.converged:
            failed_steps += 1
            if args.stop_on_newton_failure:
                print(
                    f"Newton failed at step {step}, t={current_time:.6g}, "
                    f"residual={info.residual_norm:.3e}"
                )
                break
        newton_iterations_all.append(info.iterations)
        jacobian_time_all.append(info.jacobian_time)
        solve_time_all.append(info.solve_time)
        line_search_time_all.append(info.line_search_time)

        if step % save_every == 0 or step == num_steps:
            record(current_time, info)

        if step % max(1, num_steps // 10) == 0 or step == num_steps:
            end_disp = np.linalg.norm(positions[-1] - positions0[-1])
            print(
                f"step {step:5d}/{num_steps} t={current_time:.5g} "
                f"iters={info.iterations:2d} residual={info.residual_norm:.3e} "
                f"end_disp={end_disp:.6e} converged={info.converged}"
            )

    final_disp = np.linalg.norm(positions[-1] - positions0[-1])
    final_disp_vec = positions[-1] - positions0[-1]
    all_disp = positions - positions0
    max_disp = np.max(np.linalg.norm(positions - positions0, axis=1))
    max_abs_components = np.max(np.abs(all_disp), axis=0)
    wall_time = time.perf_counter() - wall_start
    finished_steps = max(1, len(newton_iterations_all))
    avg_iters = np.mean(newton_iterations_all) if newton_iterations_all else 0.0
    print("-" * 72)
    print(f"done: failed_steps={failed_steps}, avg_recorded_newton_iters={avg_iters:.2f}")
    print(
        f"timing: wall_time={wall_time:.3f}s, "
        f"avg_step_time={wall_time / finished_steps:.4f}s, finished_steps={finished_steps}"
    )
    print(
        "solver_timing: "
        f"jacobian={np.sum(jacobian_time_all):.3f}s, "
        f"linear_solve={np.sum(solve_time_all):.3f}s, "
        f"line_search={np.sum(line_search_time_all):.3f}s"
    )
    print(f"final_end_displacement={final_disp:.8e}, max_displacement={max_disp:.8e}")
    print(
        "final_end_displacement_components: "
        f"dx={final_disp_vec[0]:+.8e}, dy={final_disp_vec[1]:+.8e}, dz={final_disp_vec[2]:+.8e}"
    )
    print(
        "max_abs_displacement_components: "
        f"|dx|={max_abs_components[0]:.8e}, |dy|={max_abs_components[1]:.8e}, "
        f"|dz|={max_abs_components[2]:.8e}"
    )
    print("=" * 72)

    if args.save_npz:
        np.savez(
            args.save_npz,
            time=np.asarray(history["time"]),
            positions=np.asarray(history["positions"]),
            velocities=np.asarray(history["velocities"]),
            newton_iterations=np.asarray(history["newton_iterations"]),
            newton_residual=np.asarray(history["newton_residual"]),
            newton_converged=np.asarray(history["newton_converged"]),
            max_displacement=np.asarray(history["max_displacement"]),
            max_velocity=np.asarray(history["max_velocity"]),
            stretch_energy=np.asarray(history["stretch_energy"]),
            bend_energy=np.asarray(history["bend_energy"]),
            kinetic_energy=np.asarray(history["kinetic_energy"]),
            total_energy=np.asarray(history["total_energy"]),
        )
        print(f"saved history to {args.save_npz}")

    if args.show_final or args.show_history:
        visualize_results(history, positions0, rest_lengths, args, force_direction_unit)

    return history


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_mid = 0.5 * (x_limits[0] + x_limits[1])
    y_mid = 0.5 * (y_limits[0] + y_limits[1])
    z_mid = 0.5 * (z_limits[0] + z_limits[1])
    radius = 0.5 * max(
        x_limits[1] - x_limits[0],
        y_limits[1] - y_limits[0],
        z_limits[1] - z_limits[0],
    )
    if radius <= 0:
        radius = 1.0

    ax.set_xlim3d(x_mid - radius, x_mid + radius)
    ax.set_ylim3d(y_mid - radius, y_mid + radius)
    ax.set_zlim3d(z_mid - radius, z_mid + radius)


def visualize_results(history, initial_positions, rest_lengths, args, force_direction_unit):
    import matplotlib.pyplot as plt

    time = np.asarray(history["time"])
    final_positions = history["positions"][-1]
    displacements = final_positions - initial_positions
    displacement_magnitudes = np.linalg.norm(displacements, axis=1)

    num_nodes = initial_positions.shape[0]
    arc_lengths = np.zeros(num_nodes)
    arc_lengths[1:] = np.cumsum(rest_lengths)

    if args.show_final:
        fig = plt.figure(figsize=(14, 9))
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        ax1.plot(
            initial_positions[:, 0],
            initial_positions[:, 1],
            initial_positions[:, 2],
            "b--",
            linewidth=1.5,
            label="Initial",
        )
        ax1.plot(
            final_positions[:, 0],
            final_positions[:, 1],
            final_positions[:, 2],
            "r-",
            linewidth=2,
            label="Final",
        )
        ax1.scatter(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], c="red", s=18)
        ax1.scatter(
            initial_positions[:2, 0],
            initial_positions[:2, 1],
            initial_positions[:2, 2],
            c="black",
            marker="s",
            s=50,
            label="Fixed nodes",
        )
        force_vector = args.force_magnitude * force_direction_unit
        force_norm = np.linalg.norm(force_vector)
        if force_norm > 0:
            arrow_scale = max(args.total_length * 0.08 / force_norm, 1e-12)
            p = final_positions[args.force_node]
            ax1.quiver(
                p[0],
                p[1],
                p[2],
                force_vector[0] * arrow_scale,
                force_vector[1] * arrow_scale,
                force_vector[2] * arrow_scale,
                color="green",
                arrow_length_ratio=0.25,
                label="External force",
            )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title("Implicit Euler final rod shape")
        ax1.legend()
        set_axes_equal(ax1)

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(arc_lengths, displacement_magnitudes, "k-o", markersize=3)
        ax2.set_xlabel("Arc length")
        ax2.set_ylabel("Displacement")
        ax2.set_title("Node displacement magnitude")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(time, history["max_displacement"], "b-", label="Max displacement")
        ax3.plot(time, history["max_velocity"], "r-", label="Max velocity")
        ax3.set_xlabel("Time")
        ax3.set_title("Motion summary")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(time, history["newton_iterations"], "m-o", markersize=3)
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Iterations")
        ax4.set_title("Newton iterations")
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()

    if args.show_history:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(time, history["stretch_energy"], label="Stretch")
        axes[0, 0].plot(time, history["bend_energy"], label="Bend")
        axes[0, 0].plot(time, history["kinetic_energy"], label="Kinetic")
        axes[0, 0].set_title("Energy components")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(time, history["total_energy"], "b-")
        axes[0, 1].set_title("Total energy")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].semilogy(time, np.maximum(history["newton_residual"], 1e-30), "g-o", markersize=3)
        axes[1, 0].set_title("Newton residual")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(time, history["max_displacement"], "k-")
        axes[1, 1].set_title("Max displacement")
        axes[1, 1].grid(True, alpha=0.3)

        for ax in axes.flat:
            ax.set_xlabel("Time")
        fig.tight_layout()

    plt.show()


def parse_force_direction(text: str):
    values = [float(part.strip()) for part in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("force direction must be like '0,1,0'")
    return values


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-length", type=float, default=default_total_length)
    parser.add_argument("--num-segments", type=int, default=default_num_segments)
    parser.add_argument("--EA", type=float, default=default_EA)
    parser.add_argument("--EI", type=float, default=default_EI)
    parser.add_argument("--mass-mode", choices=["total", "density", "node"], default=default_mass_mode)
    parser.add_argument("--total-mass", type=float, default=default_total_mass)
    parser.add_argument("--mass-density", type=float, default=default_mass_density)
    parser.add_argument("--node-mass", type=float, default=default_node_mass)
    parser.add_argument("--dt", type=float, default=default_dt)
    parser.add_argument("--total-time", type=float, default=default_total_time)
    parser.add_argument("--damping", type=float, default=default_damping)
    parser.add_argument("--gravity", type=float, default=default_gravity)
    parser.add_argument("--force-mode", choices=["none", "constant", "pulse"], default=default_force_mode)
    parser.add_argument("--force-magnitude", type=float, default=default_force_magnitude)
    parser.add_argument("--force-direction", type=parse_force_direction, default=default_force_direction)
    parser.add_argument("--force-node", type=int, default=default_force_node)
    parser.add_argument("--pulse-duration", type=float, default=default_pulse_duration)
    parser.add_argument("--save-every", type=int, default=default_save_every)
    parser.add_argument("--max-newton-iters", type=int, default=default_max_newton_iters)
    parser.add_argument("--residual-tol", type=float, default=default_residual_tol)
    parser.add_argument("--step-tol", type=float, default=default_step_tol)
    parser.add_argument(
        "--jacobian-mode",
        choices=["energy_hessian", "residual_jacobian"],
        default=default_jacobian_mode,
    )
    parser.add_argument("--stop-on-newton-failure", action="store_true")
    parser.add_argument("--save-npz", default=default_save_npz)
    parser.add_argument(
        "--show-final",
        action=argparse.BooleanOptionalAction,
        default=default_show_final,
        help="show 3D final shape and displacement plots",
    )
    parser.add_argument(
        "--show-history",
        action=argparse.BooleanOptionalAction,
        default=default_show_history,
        help="show energy and Newton residual plots",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args(argv)

    if args.num_segments < 2:
        parser.error("--num-segments must be at least 2 because the first two nodes are fixed")
    if args.mass_mode == "total" and args.total_mass <= 0:
        parser.error("--total-mass must be positive when --mass-mode total")
    if args.mass_mode == "density" and args.mass_density <= 0:
        parser.error("--mass-density must be positive when --mass-mode density")
    if args.mass_mode == "node" and args.node_mass <= 0:
        parser.error("--node-mass must be positive when --mass-mode node")
    if args.force_node is None:
        args.force_node = args.num_segments
    if args.force_node < 0 or args.force_node > args.num_segments:
        parser.error("--force-node must be between 0 and num_segments")
    if args.dt <= 0:
        parser.error("--dt must be positive")
    if args.total_time <= 0:
        parser.error("--total-time must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke_test:
        from softrobot._smoke import configure_cpu_runtime, write_smoke_result

        configure_cpu_runtime()
        args.num_segments = 3
        args.total_length = 0.3
        args.mass_mode = "node"
        args.node_mass = 0.1
        args.dt = 1.0e-3
        args.total_time = 1.0e-3
        args.gravity = 0.0
        args.force_magnitude = 0.01
        args.force_node = args.num_segments
        args.save_every = 1
        args.max_newton_iters = 5
        args.show_final = False
        args.show_history = False
        args.save_npz = None
        history = run_implicit_simulation(args)
        write_smoke_result(
            args.output_dir, "foundation-3d-implicit-newton",
            history["positions"][-1], steps=1,
            newton_iterations=int(history["newton_iterations"][-1]),
        )
        return 0
    run_implicit_simulation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
