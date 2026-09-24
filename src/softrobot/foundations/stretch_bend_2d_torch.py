"""
二维离散弹性杆动力仿真 - 自动微分版本（PyTorch）
基于 PyTorch 自动微分，所有力都从能量自动求导获得

使用方法：
    直接运行 python rod_stretch_bend_2d_pytorch.py
    修改下方参数即可
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .toolbox_2d_torch import (
    compute_total_internal_forces,
    integrate_semi_implicit_euler,
    calculate_critical_dt,
    compute_bending_angles_degrees,
    compute_total_energy
)


# ==================== 用户参数设置区 ====================

# --- 几何参数 ---
total_length = 2.0          # 改短一点，方便计算
num_segments = 10

# --- 材料参数 ---
EA = 1000.0
EI = 100.0                  # 调大，让理论挠度合理
node_mass = 0.1

# --- 仿真参数 ---
dt = 0.0005                 # 更小的步长
total_time = 1.0            # 短时间
damping = 0.5               # 大阻尼，快速稳定
gravity = 0.0               # 关闭重力！

# --- 外力参数 ---
force_mode = "constant"
force_magnitude = 10.0      # 力大小 10N
force_direction = [1.0, 0.0]  # 测试一：纯X方向
# force_direction = [0.0, 1.0]  # 测试二：纯Y方向
# force_direction = [1.0, 1.0]  # 测试三：45度方向
force_node = 10              # 最后一个节点（端点）
pulse_duration = 1.0        # 脉冲持续时间 (s)

# --- 输出参数 ---
save_every = 100            # 每多少步记录一次状态
show_animation = True       # 是否显示动画
show_energy_plot = True     # 是否显示能量曲线
show_final_shape = False    # 是否显示最终形状对比
save_animation_path = None  # 保存动画路径，如 "simulation.gif"

# ========================================================


def create_force_function(force_mode, magnitude, direction, duration=1.0):
    """创建时间相关的力函数"""
    norm = np.linalg.norm(direction)
    if norm > 0:
        unit_dir = np.array(direction) / norm
    else:
        unit_dir = np.array([1.0, 0.0])

    if force_mode == "none":
        return lambda t: 0.0, unit_dir * 0.0
    elif force_mode == "constant":
        return lambda t: magnitude, unit_dir * magnitude
    elif force_mode == "pulse":
        def pulse_func(t):
            return magnitude if t < duration else 0.0
        return pulse_func, unit_dir * magnitude
    else:
        raise ValueError(f"未知的力模式: {force_mode}")


def initialize_rod(total_length, num_segments, node_mass):
    """初始化弹性杆"""
    num_nodes = num_segments + 1
    segment_length = total_length / num_segments

    positions = np.zeros((num_nodes, 2))
    for i in range(num_nodes):
        positions[i] = [i * segment_length, 0.0]

    rest_lengths = np.full(num_segments, segment_length)
    node_masses = np.full(num_nodes, node_mass)
    node_masses[0] = 0.0  # 左端点固定

    return positions, rest_lengths, node_masses


def run_simulation(positions, rest_lengths, node_masses,
                   EA, EI, gravity, damping, dt, total_time,
                   fixed_indices, force_func, force_node, save_every):
    """运行离散弹性杆显式动力学仿真"""
    num_nodes = positions.shape[0]
    free_indices = np.setdiff1d(np.arange(num_nodes), fixed_indices)

    pos = positions.copy()
    vel = np.zeros_like(pos)
    fixed_positions_initial = pos[fixed_indices].copy()

    num_steps = int(total_time / dt)

    history = {
        'positions': [],
        'velocities': [],
        'time': [],
        'stretch_energy': [],
        'bend_energy': [],
        'kinetic_energy': [],
        'total_energy': [],
        'bending_angles': [],
        'max_displacement': [],
        'max_velocity': []
    }

    def record_state(t, external_forces_current):
        energies = compute_total_energy(
            pos, vel, rest_lengths, EA, EI,
            node_masses, gravity, external_forces_current,
            fixed_indices
        )
        bending_angles = compute_bending_angles_degrees(pos)
        max_disp = np.max(np.linalg.norm(pos - positions, axis=1))
        max_vel = np.max(np.linalg.norm(vel[free_indices], axis=1)) if len(free_indices) > 0 else 0

        history['positions'].append(pos.copy())
        history['velocities'].append(vel.copy())
        history['time'].append(t)
        history['stretch_energy'].append(energies['stretching_energy'])
        history['bend_energy'].append(energies['bending_energy'])
        history['kinetic_energy'].append(energies['kinetic_energy'])
        history['total_energy'].append(energies['total_energy'])
        history['bending_angles'].append(bending_angles.copy())
        history['max_displacement'].append(max_disp)
        history['max_velocity'].append(max_vel)

    # 打印仿真信息
    print("=" * 60)
    print("二维离散弹性杆动力仿真 (PyTorch 自动微分版本)")
    print("=" * 60)
    print(f"节点数: {num_nodes}, 边数: {num_nodes - 1}")
    print(f"总长度: {np.sum(rest_lengths):.3f} m")
    print(f"拉伸刚度 EA: {EA:.2e} N")
    print(f"弯曲刚度 EI: {EI:.2e} N·m²")
    print(f"节点质量: {node_mass:.4f} kg (自由节点)")
    print(f"阻尼系数: {damping}")
    print(f"重力加速度: {gravity} m/s²")
    print(f"时间步长 dt: {dt:.2e} s")
    print(f"总时间: {total_time:.2f} s")
    print(f"总步数: {num_steps}")
    if force_mode != "none":
        print(f"外力节点: {force_node}, 大小: {force_magnitude} N, 方向: {force_direction}")

    # CFL检查
    min_mass = np.min(node_masses[free_indices]) if len(free_indices) > 0 else node_mass
    min_len = np.min(rest_lengths)
    critical_dt = calculate_critical_dt(min_mass, min_len, EA, EI, 1.0)
    if dt > critical_dt:
        print(f"⚠️ 警告: dt={dt:.2e}s 超过CFL限制 {critical_dt:.2e}s!")
    else:
        print(f"✅ CFL检查通过: dt={dt:.2e}s ≤ {critical_dt:.2e}s")

    print("-" * 60)

    # 记录初始状态
    record_state(0.0, np.zeros_like(pos))

    # 获取外力方向向量
    _, force_vector_display = create_force_function(
        force_mode, force_magnitude, force_direction, pulse_duration
    )
    if np.linalg.norm(force_vector_display) > 0:
        unit_dir = force_vector_display / np.linalg.norm(force_vector_display)
    else:
        unit_dir = np.array([0.0, 0.0])

    # 主仿真循环
    for step in range(1, num_steps + 1):
        current_time = step * dt

        # 构建外力数组
        external_forces = np.zeros_like(pos)
        force_mag = force_func(current_time)
        if force_mag != 0 and force_mode != "none":
            external_forces[force_node] = force_mag * unit_dir

        # 计算内力（自动微分）
        internal_forces = compute_total_internal_forces(pos, rest_lengths, EA, EI)

        # 积分
        pos, vel = integrate_semi_implicit_euler(
            pos, vel, internal_forces, external_forces,
            node_masses, dt, damping, gravity, fixed_indices,
            fixed_positions_initial
        )

        # 定期记录
        if step % save_every == 0:
            record_state(current_time, external_forces)

        # 打印进度
        if step % max(1, num_steps // 10) == 0:
            progress = 100 * step / num_steps
            total_energy = history['total_energy'][-1] if history['total_energy'] else 0
            print(f"  进度: {progress:.0f}% | 总能量: {total_energy:.4e} J | t={current_time:.3f}s")

    # 记录最终状态
    if num_steps % save_every != 0:
        external_forces = np.zeros_like(pos)
        force_mag = force_func(total_time)
        if force_mag != 0 and force_mode != "none":
            external_forces[force_node] = force_mag * unit_dir
        record_state(total_time, external_forces)

    print("-" * 60)
    print("仿真完成!")
    print(f"最终总能量: {history['total_energy'][-1]:.4e} J")
    print(f"拉伸势能: {history['stretch_energy'][-1]:.4e} J")
    print(f"弯曲势能: {history['bend_energy'][-1]:.4e} J")
    print(f"动能: {history['kinetic_energy'][-1]:.4e} J")
    print("=" * 60)

    for key in history:
        if key not in ['positions', 'velocities', 'bending_angles']:
            history[key] = np.array(history[key])

    return history


# ==================== 可视化函数 ====================

def visualize_animation(history, rest_lengths, initial_positions,
                        fixed_node=0, force_node=None, save_path=None):
    """动画显示杆件变形过程"""
    positions_history = history['positions']
    bending_angles_history = history['bending_angles']
    time_history = history['time']

    num_nodes = initial_positions.shape[0]
    num_frames = len(positions_history)

    # 弧长坐标
    arc_lengths = np.zeros(num_nodes)
    for i in range(1, num_nodes):
        arc_lengths[i] = arc_lengths[i-1] + rest_lengths[i-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(wspace=0.3)

    # ========== 左侧：杆件变形动画 ==========
    all_positions = np.concatenate(positions_history, axis=0)
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    margin = max((x_max - x_min) * 0.1, (y_max - y_min) * 0.1, 0.5)

    ax1.set_xlim(x_min - margin, x_max + margin)
    ax1.set_ylim(y_min - margin, y_max + margin)
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Rod Deformation (PyTorch Autograd)')

    # 初始构型
    ax1.plot(initial_positions[:, 0], initial_positions[:, 1],
             'r--', alpha=0.5, linewidth=1, label='Initial shape')

    # 固定节点
    ax1.scatter(initial_positions[fixed_node, 0], initial_positions[fixed_node, 1],
                c='black', s=80, marker='s', label='Fixed Node', zorder=5)

    # 当前构型
    current_line, = ax1.plot([], [], 'b-', linewidth=2, label='Current shape')
    current_nodes, = ax1.plot([], [], 'go', markersize=6, label='Nodes')

    # 施力节点标记
    force_marker = None
    if force_node is not None:
        force_marker, = ax1.plot([], [], 'r^', markersize=10, label='Force node')

    # 时间显示
    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                         fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.legend(loc='upper right')

    # ========== 右侧：弯曲角度分布 ==========
    ax2.set_xlim(0, arc_lengths[-1])
    ax2.set_ylim(-60, 60)
    ax2.set_xlabel('Arc Length (m)')
    ax2.set_ylabel('Bending Angle (deg)')
    ax2.set_title('Bending Angle Distribution')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.axhline(0, color='gray', linewidth=0.8)

    bend_line, = ax2.plot([], [], 'm-o', markersize=4, linewidth=1.5, label='Current angle')
    ax2.legend(loc='upper right')

    def update(frame):
        positions = positions_history[frame]
        bending_angles = bending_angles_history[frame]
        current_time = time_history[frame]

        current_line.set_data(positions[:, 0], positions[:, 1])
        current_nodes.set_data(positions[:, 0], positions[:, 1])

        if force_marker:
            force_marker.set_data([positions[force_node, 0]], [positions[force_node, 1]])

        time_text.set_text(f't = {current_time:.3f} s')
        bend_line.set_data(arc_lengths, bending_angles)

        max_angle = np.max(np.abs(bending_angles))
        ax1.set_title(f'Rod Deformation (max bend angle = {max_angle:.1f}°)')

        return current_line, current_nodes, bend_line, time_text

    anim = animation.FuncAnimation(fig, update, frames=num_frames,
                                   interval=50, blit=False, repeat=False)

    if save_path:
        anim.save(save_path, writer='pillow', fps=25)
        print(f"动画已保存到: {save_path}")

    plt.show()
    return anim


def visualize_energy_history(history):
    """绘制能量历史曲线"""
    time = history['time']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(time, history['stretch_energy'], label='Stretching', linewidth=1.5)
    axes[0, 0].plot(time, history['bend_energy'], label='Bending', linewidth=1.5)
    axes[0, 0].plot(time, history['kinetic_energy'], label='Kinetic', linewidth=1.5)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Energy (J)')
    axes[0, 0].set_title('Energy Components')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(time, history['total_energy'], 'b-', linewidth=1.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Total Energy (J)')
    axes[0, 1].set_title('Total Energy')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(time, history['max_displacement'], 'g-', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Max Displacement (m)')
    axes[1, 0].set_title('Maximum Node Displacement')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(time, history['max_velocity'], 'r-', linewidth=1.5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Max Velocity (m/s)')
    axes[1, 1].set_title('Maximum Node Velocity')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_final_shape(initial_positions, final_positions, rest_lengths,
                          fixed_node=0, force_node=None, force_vector=None):
    """绘制最终变形形状对比"""
    num_nodes = initial_positions.shape[0]

    arc_lengths = np.zeros(num_nodes)
    for i in range(1, num_nodes):
        arc_lengths[i] = arc_lengths[i-1] + rest_lengths[i-1]

    displacements = final_positions - initial_positions
    displacement_magnitudes = np.linalg.norm(displacements, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 形状对比
    axes[0].plot(initial_positions[:, 0], initial_positions[:, 1],
                 'b--', linewidth=2, label='Initial', alpha=0.7)
    axes[0].plot(final_positions[:, 0], final_positions[:, 1],
                 'r-', linewidth=2, label='Final')
    axes[0].scatter(initial_positions[fixed_node, 0], initial_positions[fixed_node, 1],
                    c='black', s=100, marker='s', label='Fixed', zorder=5)

    if force_node is not None and force_vector is not None:
        force_norm = np.linalg.norm(force_vector)
        if force_norm > 0:
            scale = 0.2 / force_norm
            axes[0].arrow(final_positions[force_node, 0], final_positions[force_node, 1],
                          force_vector[0] * scale, force_vector[1] * scale,
                          head_width=0.05, head_length=0.08, fc='green', ec='green',
                          label=f'Force ({force_norm:.1f} N)')

    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title('Shape Comparison')
    axes[0].set_aspect('equal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 位移分布
    axes[1].plot(arc_lengths, displacement_magnitudes, 'b-o', markersize=6, linewidth=1.5)
    axes[1].set_xlabel('Arc Length (m)')
    axes[1].set_ylabel('Displacement (m)')
    axes[1].set_title('Node Displacements')
    axes[1].grid(True, alpha=0.3)

    # 弯曲角度分布
    final_angles = compute_bending_angles_degrees(final_positions)
    axes[2].plot(arc_lengths, final_angles, 'm-o', markersize=6, linewidth=1.5)
    axes[2].axhline(0, color='gray', linewidth=0.8)
    axes[2].set_xlabel('Arc Length (m)')
    axes[2].set_ylabel('Bending Angle (deg)')
    axes[2].set_title('Final Bending Angle')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_displacements(initial_positions, final_positions, force_node):
    """打印节点位移信息 - 详细版"""
    displacements = final_positions - initial_positions

    print("\n" + "=" * 80)
    print("最终节点位移详细信息")
    print("=" * 80)
    print(f"{'节点':>4} | {'X位移 (m)':>14} | {'Y位移 (m)':>14} | {'总位移 (m)':>14}")
    print("-" * 80)

    for i in range(len(initial_positions)):
        disp_mag = np.linalg.norm(displacements[i])
        print(f"{i:4d} | {displacements[i, 0]:+14.8e} | {displacements[i, 1]:+14.8e} | {disp_mag:14.8e}")

    print("-" * 80)
    print(f"施力节点 {force_node} 总位移: {np.linalg.norm(displacements[force_node]):.8e} m")
    print("=" * 80)


# ==================== 主程序 ====================

def _run_smoke(output_dir: str) -> None:
    from .._smoke import configure_cpu_runtime, write_smoke_result

    configure_cpu_runtime()
    positions, rest_lengths, node_masses = initialize_rod(0.4, 4, node_mass)
    force_func, _ = create_force_function("constant", 0.1, [1.0, 0.0])
    history = run_simulation(
        positions, rest_lengths, node_masses, EA, EI, 0.0, damping,
        1.0e-6, 2.0e-6, np.array([0], dtype=int), force_func, 4, 1,
    )
    write_smoke_result(
        output_dir, "foundation-2d-stretch-bend-torch",
        history["positions"][-1], steps=2,
        energy=float(history["total_energy"][-1]),
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Historical 2-D stretch-bend rod")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args(argv)
    if args.smoke_test:
        _run_smoke(args.output_dir)
        return 0
    # 1. 初始化杆件
    positions, rest_lengths, node_masses = initialize_rod(total_length, num_segments, node_mass)

    # 2. 设置固定节点（左端点）
    fixed_indices = np.array([0], dtype=int)

    # 3. 设置外力函数
    force_func, force_vector_display = create_force_function(
        force_mode, force_magnitude, force_direction, pulse_duration
    )

    # 4. 运行仿真
    history = run_simulation(
        positions=positions,
        rest_lengths=rest_lengths,
        node_masses=node_masses,
        EA=EA,
        EI=EI,
        gravity=gravity,
        damping=damping,
        dt=dt,
        total_time=total_time,
        fixed_indices=fixed_indices,
        force_func=force_func,
        force_node=force_node,
        save_every=save_every
    )

    # 5. 可视化
    final_positions = history['positions'][-1]

    if show_animation:
        visualize_animation(
            history, rest_lengths, positions,
            fixed_node=0,
            force_node=force_node if force_mode != "none" else None,
            save_path=save_animation_path
        )

    if show_final_shape:
        force_vector_plot = None
        if force_mode != "none":
            force_vector_plot = force_vector_display
        visualize_final_shape(
            positions, final_positions, rest_lengths,
            fixed_node=0,
            force_node=force_node if force_mode != "none" else None,
            force_vector=force_vector_plot
        )

    if show_energy_plot:
        visualize_energy_history(history)

    # 6. 打印位移结果
    if force_mode != "none":
        print_displacements(positions, final_positions, force_node)
    else:
        print_displacements(positions, final_positions, 0)


if __name__ == "__main__":
    raise SystemExit(main())
