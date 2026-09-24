"""
三维离散弹性杆动力仿真 - 自动微分版本（PyTorch）
基于 PyTorch 自动微分，所有力都从能量自动求导获得
优化版：整合 curvature-binormal 弯曲模型

使用方法：
    直接运行 python rodsim_str_bend_pytorch3d.py
    修改下方参数即可
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from .toolbox_3d_torch import (
    compute_total_internal_forces,
    integrate_semi_implicit_euler,
    calculate_critical_dt,
    compute_bending_angles_degrees,
    compute_total_energy,
    compute_curvature_binormal_magnitudes,
    compute_initial_curvature_binormal
)


# ==================== 用户参数设置区 ====================

# --- 几何参数 ---
total_length = 50.0          # 杆的总长度 (m)
num_segments = 50           # 分段数（边数），节点数 = num_segments + 1

# --- 材料参数 ---
EA = 1000.0                 # 拉伸刚度 (N)
EI = 200                    # 弯曲刚度 (N·m²)

# --- 质量参数 ---
# "total": 固定自由节点总质量，适合不同离散段数之间比较
# "density": 固定线密度，total_mass = mass_density * total_length
# "node": 固定每个自由节点质量，和旧版逻辑一致
mass_mode = "total"
total_mass = 15.0           # mass_mode="total" 时，自由节点总质量 (kg)
mass_density = 0.3          # mass_mode="density" 时，单位长度质量 (kg/m)
node_mass = 0.5             # mass_mode="node" 时，每个自由节点质量 (kg)

# --- 仿真参数 ---
dt = 5e-5                 # 时间步长 (s)
total_time = 0.2            # 总仿真时间 (s)
damping = 0.5               # 速度阻尼系数
gravity = -9.81             # 重力加速度 (m/s²) - 沿 Z 轴负方向
max_velocity = 100.0        # 最大速度限制

# --- 外力参数 ---
force_mode = "constant"     # "none", "constant", "pulse"
force_magnitude = 100      # 外力大小 (N)
force_direction = [0, 1, 0]  # 外力方向 [Fx, Fy, Fz]（会自动归一化）
force_node = 50              # 施力节点索引 (0 到 num_segments)
pulse_duration = 1.0        # 脉冲持续时间 (s)

# --- 输出参数 ---
save_every = 100            # 每多少步记录一次状态
show_animation = True       # 是否显示动画
show_energy_plot = True     # 是否显示能量曲线
show_final_shape = True     # 是否显示最终形状对比
show_kappa_b_plot = True    # 是否显示曲率副法线图
save_animation_path = None  # 保存动画路径，如 "simulation.gif"

# ========================================================


def create_force_function(force_mode, magnitude, direction, duration=1.0):
    """创建时间相关的力函数"""
    norm = np.linalg.norm(direction)
    if norm > 0:
        unit_dir = np.array(direction) / norm
    else:
        unit_dir = np.array([1.0, 0.0, 0.0])

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


def initialize_rod(total_length, num_segments, mass_mode, total_mass, mass_density, node_mass):
    """初始化弹性杆 (3D) - 固定端约束版（固定前两个节点）"""
    num_nodes = num_segments + 1
    segment_length = total_length / num_segments

    # 沿 X 轴放置
    positions = np.zeros((num_nodes, 3))
    for i in range(num_nodes):
        positions[i] = [i * segment_length, 0.0, 0.0]

    rest_lengths = np.full(num_segments, segment_length)
    node_masses = np.zeros(num_nodes)
    num_free_nodes = num_nodes - 2
    if mass_mode == "total":
        free_node_mass = total_mass / num_free_nodes
    elif mass_mode == "density":
        free_node_mass = mass_density * total_length / num_free_nodes
    elif mass_mode == "node":
        free_node_mass = node_mass
    else:
        raise ValueError(f"未知质量模式: {mass_mode}")
    node_masses[2:] = free_node_mass

    # 固定前两个节点（模拟固定端，约束位移和转角）
    node_masses[0] = 0.0
    node_masses[1] = 0.0

    return positions, rest_lengths, node_masses


def run_simulation(positions, rest_lengths, node_masses,
                   EA, EI, gravity, damping, dt, total_time,
                   fixed_indices, force_func, force_node, force_direction_unit,
                   save_every, max_velocity=100.0):
    """运行离散弹性杆显式动力学仿真 (3D)"""
    num_nodes = positions.shape[0]

    # 注意：fixed_indices 应该已经包含 [0, 1]
    free_indices = np.setdiff1d(np.arange(num_nodes), fixed_indices)

    pos = positions.copy()
    vel = np.zeros_like(pos)

    # 固定节点的初始位置（用于后续强制位置）
    fixed_positions_initial = pos[fixed_indices].copy()

    # 计算初始曲率副法线（用于非零初始弯曲）
    rest_kappa_b = compute_initial_curvature_binormal(positions, rest_lengths)

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
        'kappa_b_magnitudes': [],
        'max_displacement': [],
        'max_velocity': []
    }

    def record_state(t, external_forces_current):
        energies = compute_total_energy(
            pos, vel, rest_lengths, EA, EI,
            node_masses, gravity, external_forces_current,
            fixed_indices, rest_kappa_b
        )
        bending_angles = compute_bending_angles_degrees(pos)
        kappa_b_mags = compute_curvature_binormal_magnitudes(pos)
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
        history['kappa_b_magnitudes'].append(kappa_b_mags.copy())
        history['max_displacement'].append(max_disp)
        history['max_velocity'].append(max_vel)

    # 打印仿真信息
    print("=" * 60)
    print("三维离散弹性杆动力仿真 (PyTorch 自动微分版本)")
    print("优化版：使用 curvature-binormal 弯曲模型")
    print("=" * 60)
    print(f"节点数: {num_nodes}, 边数: {num_nodes - 1}")
    print(f"总长度: {np.sum(rest_lengths):.3f} m")
    print(f"拉伸刚度 EA: {EA:.2e} N")
    print(f"弯曲刚度 EI: {EI:.2e} N·m²")
    print(
        f"质量模式: {mass_mode}, 自由节点总质量: {np.sum(node_masses):.6g} kg, "
        f"单个自由节点质量: {node_masses[2]:.6g} kg"
    )
    print(f"阻尼系数: {damping}")
    print(f"重力加速度: {gravity} m/s² (沿Z轴)")
    print(f"时间步长 dt: {dt:.2e} s")
    print(f"总时间: {total_time:.2f} s")
    print(f"总步数: {num_steps}")
    print(f"速度限制: {max_velocity} m/s")
    if force_mode != "none":
        print(f"外力节点: {force_node}, 大小: {force_magnitude} N, 方向: {force_direction_unit}")

    # CFL检查
    min_mass = np.min(node_masses[free_indices]) if len(free_indices) > 0 else node_mass
    min_len = np.min(rest_lengths)
    critical_dt = calculate_critical_dt(min_mass, min_len, EA, EI, 0.5)
    if dt > critical_dt:
        print(f"⚠️ 警告: dt={dt:.2e}s 超过CFL限制 {critical_dt:.2e}s!")
    else:
        print(f"✅ CFL检查通过: dt={dt:.2e}s ≤ {critical_dt:.2e}s")

    print("-" * 60)

    # 记录初始状态
    record_state(0.0, np.zeros_like(pos))

    # 主仿真循环
    for step in range(1, num_steps + 1):
        current_time = step * dt

        # 构建外力数组
        external_forces = np.zeros_like(pos)
        force_mag = force_func(current_time)
        if force_mag != 0 and force_mode != "none":
            external_forces[force_node] = force_mag * force_direction_unit

        # 计算内力（自动微分，支持初始曲率）
        internal_forces = compute_total_internal_forces(pos, rest_lengths, EA, EI, rest_kappa_b)

        # 积分
        pos, vel = integrate_semi_implicit_euler(
            pos, vel, internal_forces, external_forces,
            node_masses, dt, damping, gravity, fixed_indices,
            fixed_positions_initial, max_velocity
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
            external_forces[force_node] = force_mag * force_direction_unit
        record_state(total_time, external_forces)

    print("-" * 60)
    print("仿真完成!")
    print(f"最终总能量: {history['total_energy'][-1]:.4e} J")
    print(f"拉伸势能: {history['stretch_energy'][-1]:.4e} J")
    print(f"弯曲势能: {history['bend_energy'][-1]:.4e} J")
    print(f"动能: {history['kinetic_energy'][-1]:.4e} J")
    print("=" * 60)

    for key in history:
        if key not in ['positions', 'velocities', 'bending_angles', 'kappa_b_magnitudes']:
            history[key] = np.array(history[key])

    return history


# ==================== 可视化函数 (3D) ====================

def set_axes_equal(ax):
    """设置3D坐标轴等比例"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range) / 2.0

    mid_x = (x_limits[0] + x_limits[1]) * 0.5
    mid_y = (y_limits[0] + y_limits[1]) * 0.5
    mid_z = (z_limits[0] + z_limits[1]) * 0.5

    ax.set_xlim3d(mid_x - max_range, mid_x + max_range)
    ax.set_ylim3d(mid_y - max_range, mid_y + max_range)
    ax.set_zlim3d(mid_z - max_range, mid_z + max_range)


def visualize_animation(history, rest_lengths, initial_positions,
                        fixed_node=0, force_node=None, save_path=None):
    """动画显示杆件变形过程 (3D)"""
    positions_history = history['positions']
    time_history = history['time']
    bending_angles_history = history['bending_angles']

    num_nodes = initial_positions.shape[0]
    num_frames = len(positions_history)

    # 弧长坐标
    arc_lengths = np.zeros(num_nodes)
    for i in range(1, num_nodes):
        arc_lengths[i] = arc_lengths[i-1] + rest_lengths[i-1]

    # 创建图形 - 改为 1x2 布局
    fig = plt.figure(figsize=(14, 8))

    # 左侧：3D动画
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    # 右侧：弯曲角度分布
    ax2 = fig.add_subplot(1, 2, 2)

    # ========== 左侧：3D杆件变形动画 ==========
    all_positions = np.concatenate(positions_history, axis=0)
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()
    margin = 0.2

    ax1.set_xlim(x_min - margin, x_max + margin)
    ax1.set_ylim(y_min - margin, y_max + margin)
    ax1.set_zlim(z_min - margin, z_max + margin)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_zlabel('Z Position (m)')
    ax1.set_title('3D Rod Deformation (PyTorch Autograd)')

    # 初始构型（红色虚线）
    ax1.plot(initial_positions[:, 0], initial_positions[:, 1], initial_positions[:, 2],
             'r--', alpha=0.5, linewidth=1, label='Initial shape')

    # 固定节点（黑色方块）
    ax1.scatter(initial_positions[fixed_node, 0], initial_positions[fixed_node, 1],
                initial_positions[fixed_node, 2], c='black', s=80, marker='s', label='Fixed Node')

    # 当前构型
    current_line, = ax1.plot([], [], [], 'b-', linewidth=2, label='Current shape')
    current_nodes, = ax1.plot([], [], [], 'go', markersize=6, label='Nodes')

    # 施力节点标记
    force_marker = None
    if force_node is not None:
        force_marker, = ax1.plot([], [], [], 'r^', markersize=10, label='Force node')

    # 时间显示
    time_text = ax1.text2D(0.02, 0.98, '', transform=ax1.transAxes,
                           fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.legend(loc='upper right')
    set_axes_equal(ax1)

    # ========== 右侧：弯曲角度分布 ==========
    ax2.set_xlim(0, arc_lengths[-1])
    ax2.set_ylim(0, 180)
    ax2.set_xlabel('Arc Length (m)')
    ax2.set_ylabel('Bending Angle (deg)')
    ax2.set_title('Bending Angle Distribution')
    ax2.grid(True, linestyle='--', alpha=0.6)

    bend_line, = ax2.plot([], [], 'm-o', markersize=4, linewidth=1.5, label='Current angle')
    ax2.legend(loc='upper right')

    def update(frame):
        positions = positions_history[frame]
        current_time = time_history[frame]
        bending_angles = bending_angles_history[frame]

        # 更新左侧3D动画
        current_line.set_data(positions[:, 0], positions[:, 1])
        current_line.set_3d_properties(positions[:, 2])
        current_nodes.set_data(positions[:, 0], positions[:, 1])
        current_nodes.set_3d_properties(positions[:, 2])

        if force_marker:
            force_marker.set_data([positions[force_node, 0]], [positions[force_node, 1]])
            force_marker.set_3d_properties([positions[force_node, 2]])

        time_text.set_text(f't = {current_time:.3f} s')

        # 更新右侧弯曲角度图
        if len(bending_angles) > 0:
            node_positions = arc_lengths[1:-1]
            bend_line.set_data(node_positions, bending_angles[1:-1])

        # 更新标题
        if len(bending_angles) > 0:
            max_angle = np.max(bending_angles)
            ax1.set_title(f'3D Rod Deformation (max bend angle = {max_angle:.1f}°)')

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


def visualize_curvature_evolution(history, rest_lengths):
    """绘制曲率副法线演化"""
    if 'kappa_b_magnitudes' not in history or len(history['kappa_b_magnitudes']) == 0:
        print("没有曲率数据可显示")
        return

    time = history['time']
    kappa_b_history = history['kappa_b_magnitudes']

    max_kappa_b = [np.max(kb) if len(kb) > 0 else 0 for kb in kappa_b_history]
    mean_kappa_b = [np.mean(kb) if len(kb) > 0 else 0 for kb in kappa_b_history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(time, max_kappa_b, 'b-', linewidth=1.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Max ||κb|| (1/m)')
    axes[0].set_title('Maximum Curvature-Binormal Magnitude')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, mean_kappa_b, 'r-', linewidth=1.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Mean ||κb|| (1/m)')
    axes[1].set_title('Mean Curvature-Binormal Magnitude')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_final_shape(initial_positions, final_positions, rest_lengths,
                          fixed_node=0, force_node=None, force_vector=None):
    """绘制最终变形形状对比 (3D) - 添加节点标记"""
    num_nodes = initial_positions.shape[0]

    # 计算位移
    displacements = final_positions - initial_positions
    displacement_magnitudes = np.linalg.norm(displacements, axis=1)

    # 计算曲率副法线（最终显示用）
    final_kappa_b = compute_curvature_binormal_magnitudes(final_positions)

    fig = plt.figure(figsize=(15, 10))

    # 子图1：3D形状对比（添加节点）
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(initial_positions[:, 0], initial_positions[:, 1], initial_positions[:, 2],
             'b--', linewidth=2, label='Initial', alpha=0.7)
    ax1.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2],
             'r-', linewidth=2, label='Final')
    # 添加初始构型的节点
    ax1.scatter(initial_positions[:, 0], initial_positions[:, 1], initial_positions[:, 2],
                c='blue', s=30, marker='o', alpha=0.5, label='Initial nodes')
    # 添加最终构型的节点
    ax1.scatter(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2],
                c='red', s=40, marker='o', label='Final nodes')
    # 固定节点用特殊标记
    ax1.scatter(initial_positions[fixed_node, 0], initial_positions[fixed_node, 1],
                initial_positions[fixed_node, 2], c='black', s=100, marker='s', label='Fixed')

    if force_node is not None and force_vector is not None:
        force_norm = np.linalg.norm(force_vector)
        if force_norm > 0:
            scale = 0.2 / force_norm
            ax1.quiver(final_positions[force_node, 0], final_positions[force_node, 1],
                       final_positions[force_node, 2],
                       force_vector[0] * scale, force_vector[1] * scale, force_vector[2] * scale,
                       color='green', arrow_length_ratio=0.3, label=f'Force ({force_norm:.1f} N)')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Shape Comparison')
    ax1.legend()
    set_axes_equal(ax1)

    # 子图2：位移分布
    ax2 = fig.add_subplot(2, 2, 2)
    arc_lengths = np.zeros(num_nodes)
    for i in range(1, num_nodes):
        arc_lengths[i] = arc_lengths[i-1] + rest_lengths[i-1]
    ax2.plot(arc_lengths, displacement_magnitudes, 'b-o', markersize=6, linewidth=1.5)
    ax2.set_xlabel('Arc Length (m)')
    ax2.set_ylabel('Displacement (m)')
    ax2.set_title('Node Displacements')
    ax2.grid(True, alpha=0.3)

    # 子图3：弯曲角度分布
    ax3 = fig.add_subplot(2, 2, 3)
    final_angles = compute_bending_angles_degrees(final_positions)
    initial_angles = compute_bending_angles_degrees(initial_positions)
    ax3.plot(arc_lengths, final_angles, 'r-o', markersize=6, linewidth=1.5, label='Final')
    ax3.plot(arc_lengths, initial_angles, 'b--', markersize=4, linewidth=1, label='Initial')
    ax3.axhline(0, color='gray', linewidth=0.8)
    ax3.set_xlabel('Arc Length (m)')
    ax3.set_ylabel('Bending Angle (deg)')
    ax3.set_title('Bending Angle Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 子图4：最终曲率副法线
    ax4 = fig.add_subplot(2, 2, 4)
    vertex_arc_lengths = arc_lengths[1:-1]
    if len(final_kappa_b) > 0:
        ax4.plot(vertex_arc_lengths, final_kappa_b, 'c-o', markersize=6, linewidth=1.5)
    ax4.set_xlabel('Arc Length (m)')
    ax4.set_ylabel('||κb|| (1/m)')
    ax4.set_title('Final Curvature-Binormal Magnitude')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_displacements(initial_positions, final_positions, force_node):
    """打印节点位移信息 (3D) - 详细版"""
    displacements = final_positions - initial_positions

    print("\n" + "=" * 80)
    print("最终节点位移详细信息")
    print("=" * 80)
    print(f"{'节点':>4} | {'X位移 (m)':>12} | {'Y位移 (m)':>12} | {'Z位移 (m)':>12} | {'总位移 (m)':>12}")
    print("-" * 80)

    for i in range(len(initial_positions)):
        disp_mag = np.linalg.norm(displacements[i])
        print(f"{i:4d} | {displacements[i, 0]:+12.6e} | {displacements[i, 1]:+12.6e} | "
              f"{displacements[i, 2]:+12.6e} | {disp_mag:12.6e}")

    print("-" * 80)
    print(f"施力节点 {force_node} 总位移: {np.linalg.norm(displacements[force_node]):.6e} m")
    print("=" * 80)


def _run_smoke(output_dir: str) -> None:
    from .._smoke import cpu_smoke_runtime, write_smoke_result

    with cpu_smoke_runtime():
        _run_smoke_in_runtime(output_dir, write_smoke_result)


def _run_smoke_in_runtime(output_dir: str, write_smoke_result) -> None:
    positions, rest_lengths, node_masses = initialize_rod(
        0.4, 4, "node", total_mass, mass_density, 0.1
    )
    force_func, force_vector = create_force_function(
        "constant", 0.1, [0.0, 1.0, 0.0]
    )
    force_direction_unit = force_vector / np.linalg.norm(force_vector)
    history = run_simulation(
        positions, rest_lengths, node_masses, EA, EI, 0.0, damping,
        1.0e-6, 2.0e-6, np.array([0, 1], dtype=int), force_func, 4,
        force_direction_unit, 1, max_velocity,
    )
    write_smoke_result(
        output_dir, "foundation-3d-stretch-bend-torch",
        history["positions"][-1], steps=2,
        energy=float(history["total_energy"][-1]),
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Historical 3-D stretch-bend rod")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args(argv)
    if args.smoke_test:
        _run_smoke(args.output_dir)
        return 0
    # 1. 初始化杆件
    positions, rest_lengths, node_masses = initialize_rod(
        total_length, num_segments, mass_mode, total_mass, mass_density, node_mass
    )

    # 2. 设置固定节点（前两个节点，形成固定端约束）
    fixed_indices = np.array([0, 1], dtype=int)

    # 3. 设置外力函数
    force_func, force_vector_display = create_force_function(
        force_mode, force_magnitude, force_direction, pulse_duration
    )

    # 获取外力单位方向
    if np.linalg.norm(force_vector_display) > 0:
        force_direction_unit = force_vector_display / np.linalg.norm(force_vector_display)
    else:
        force_direction_unit = np.array([0.0, 0.0, 0.0])

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
        force_direction_unit=force_direction_unit,
        save_every=save_every,
        max_velocity=max_velocity
    )

    # 5. 可视化
    final_positions = history['positions'][-1]

    if show_animation:
        visualize_animation(
            history, rest_lengths, positions,
            fixed_node=0,  # 固定端用第一个节点标记
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

    if show_kappa_b_plot:
        visualize_curvature_evolution(history, rest_lengths)

    # 6. 打印位移结果
    if force_mode != "none":
        print_displacements(positions, final_positions, force_node)
    else:
        print_displacements(positions, final_positions, 0)


if __name__ == "__main__":
    raise SystemExit(main())
