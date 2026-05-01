"""
二维离散弹性杆动力仿真 - 拉伸+弯曲（解析版本）
基于导师文档中的解析公式，使用半隐式欧拉积分

物理模型：
- 杆件由N个节点和N-1个边组成
- 左端点固定，右端点自由
- 重力沿-Y方向
- 可选在指定节点施加恒力或脉冲力

使用方法：
    直接运行 python rod_strecth_bending_2d.py
    修改下方参数即可
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 导入工具箱函数
from rodtoolbox_2d_ks_kb import (
    compute_total_internal_forces,
    integrate_semi_implicit_euler,
    calculate_critical_dt,
    compute_bending_angles_degrees,
    compute_total_energy)


# ==================== 用户参数设置区 ====================
# 在这里手动修改你的参数

# --- 几何参数 ---
total_length = 2.0          # 杆的总长度 (m)
num_segments = 10           # 分段数（增加分段使弯曲更平滑）

# --- 材料参数 ---
EA = 1000.0                 # 拉伸刚度 (N)
EI = 0.5                    # 弯曲刚度 (N·m²)
node_mass = 0.1             # 每个节点的质量 (kg)

# --- 仿真参数 ---
dt = 0.0004                 # 时间步长 (s)
total_time = 10            # 总仿真时间 (s)
damping = 1
# 速度阻尼系数
gravity = -9.81             # 重力加速度 (m/s²)

# --- 外力参数 ---
force_mode = "constant"     # "none", "constant", "pulse"
force_magnitude = 5.0       # 外力大小 (N)
force_direction = [1.0, 0.0]  # 外力方向 [x, y]
force_node = 5              # 施力节点索引
pulse_duration = 1.0        # 脉冲持续时间 (s)

# --- 输出参数 ---
save_every = 100            # 每多少步记录一次状态
show_animation = True       # 是否显示动画
show_energy_plot = True     # 是否显示能量曲线
show_final_shape = True     # 是否显示最终形状对比
save_animation_path = None  # 保存动画路径，如 "simulation.gif"

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


def initialize_rod(total_length, num_segments):
    """初始化弹性杆的几何构型"""
    num_nodes = num_segments + 1
    segment_length = total_length / num_segments

    positions = np.zeros((num_nodes, 2))
    for i in range(num_nodes):
        positions[i] = [i * segment_length, 0.0]

    rest_lengths = np.full(num_segments, segment_length)
    node_masses_arr = np.full(num_nodes, node_mass)
    node_masses_arr[0] = 0.0  # 左端点固定

    return positions, rest_lengths, node_masses_arr


def run_simulation(positions, rest_lengths, node_masses,
                   EA, EI, gravity, damping, dt, total_time,
                   fixed_indices, force_func, save_every,
                   convergence_tol=1e-3):  # 新增：收敛容差
    """
    运行离散弹性杆显式动力学仿真

    新增参数:
        convergence_tol: 收敛容差（最大速度小于此值时停止），默认1e-6

    返回:
        history: 字典，包含所有记录的状态和能量，新增 'converged' 和 'converged_step'
    """
    num_nodes = positions.shape[0]
    free_indices = np.setdiff1d(np.arange(num_nodes), fixed_indices)

    # 复制状态变量
    pos = positions.copy()
    vel = np.zeros_like(pos)

    # 保存固定节点的初始位置
    fixed_positions_initial = pos[fixed_indices].copy()

    # 时间步数
    num_steps = int(total_time / dt)

    # 记录历史（新增收敛相关字段）
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
        'max_velocity': [],
        'converged': False,  # 新增：是否收敛
        'converged_step': None,  # 新增：收敛步数
    }

    # 记录初始状态
    def record_state(t):
        # 计算能量
        energies = compute_total_energy(
            pos, vel, rest_lengths, EA, EI,
            node_masses, gravity, np.zeros_like(pos),
            fixed_indices
        )

        # 弯曲角度
        bending_angles = compute_bending_angles_degrees(pos)

        # 最大位移和速度
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
    print("二维离散弹性杆动力仿真 (拉伸 + 弯曲)")
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
    print(f"收敛容差: {convergence_tol:.2e} (最大速度)")

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
    record_state(0.0)

    # 主仿真循环
    for step in range(1, num_steps + 1):
        current_time = step * dt

        # 构建外力数组
        external_forces = np.zeros_like(pos)
        force_mag = force_func(current_time)
        force_vector = force_func.force_vector if hasattr(force_func, 'force_vector') else None
        if force_mag != 0 and force_vector is not None:
            external_forces[force_node] = force_mag * (force_vector / np.linalg.norm(force_vector))

        # 计算内力
        internal_forces = compute_total_internal_forces(
            pos, rest_lengths, EA, EI
        )

        # 半隐式欧拉积分
        pos, vel = integrate_semi_implicit_euler(
            pos, vel, internal_forces, external_forces,
            node_masses, dt, damping, gravity, fixed_indices,
            fixed_positions=fixed_positions_initial
        )

        # 计算当前最大速度（用于收敛判断）
        max_velocity = np.max(np.linalg.norm(vel[free_indices], axis=1)) if len(free_indices) > 0 else 0

        # 【新增】收敛性判断
        if max_velocity < convergence_tol:
            history['converged'] = True
            history['converged_step'] = step
            # 记录最终收敛状态
            if step % save_every != 0:
                record_state(current_time)
            print(f"\n✅ 仿真收敛! 步数={step}, 最大速度={max_velocity:.2e} m/s < {convergence_tol:.2e}")
            break

        # 定期记录
        if step % save_every == 0:
            record_state(current_time)

        # 打印进度
        if step % max(1, num_steps // 10) == 0:
            progress = 100 * step / num_steps
            total_energy = history['total_energy'][-1] if history['total_energy'] else 0
            print(
                f"  进度: {progress:.0f}% | 总能量: {total_energy:.4e} J | 最大速度: {max_velocity:.2e} m/s | t={current_time:.3f}s")

    # 记录最终状态（如果未在收敛时记录）
    if not history['converged'] and (num_steps % save_every != 0):
        record_state(total_time)

    print("-" * 60)
    if history['converged']:
        print(f"仿真在步数 {history['converged_step']} 处收敛!")
    else:
        print("仿真完成（达到最大步数）!")
    print(f"最终总能量: {history['total_energy'][-1]:.4e} J")
    print(f"拉伸势能: {history['stretch_energy'][-1]:.4e} J")
    print(f"弯曲势能: {history['bend_energy'][-1]:.4e} J")
    print(f"动能: {history['kinetic_energy'][-1]:.4e} J")
    print("=" * 60)

    # 转换为numpy数组
    for key in history:
        if key not in ['positions', 'velocities', 'bending_angles', 'converged', 'converged_step']:
            history[key] = np.array(history[key])

    return history

def visualize_animation(history, rest_lengths, initial_positions,
                        fixed_node=0, force_node=None, save_path=None):
    """动画显示杆件变形过程"""
    positions_history = history['positions']
    bending_angles_history = history['bending_angles']
    time_history = history['time']

    num_nodes = initial_positions.shape[0]
    num_frames = len(positions_history)

    arc_lengths = np.zeros(num_nodes)
    for i in range(1, num_nodes):
        arc_lengths[i] = arc_lengths[i - 1] + rest_lengths[i - 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(wspace=0.3)

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
    ax1.set_title('Rod Deformation (Stretching + Bending)')

    ax1.plot(initial_positions[:, 0], initial_positions[:, 1],
             'r--', alpha=0.5, linewidth=1, label='Initial shape')
    ax1.scatter(initial_positions[fixed_node, 0], initial_positions[fixed_node, 1],
                c='black', s=80, marker='s', label='Fixed Node', zorder=5)

    current_line, = ax1.plot([], [], 'b-', linewidth=2, label='Current shape')
    current_nodes, = ax1.plot([], [], 'go', markersize=6, label='Nodes')

    force_marker = None
    if force_node is not None:
        force_marker, = ax1.plot([], [], 'r^', markersize=10, label='Force node')

    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                         fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.legend(loc='upper right')

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
    axes[0, 1].set_title('Total Energy (should be nearly conserved)')
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
        arc_lengths[i] = arc_lengths[i - 1] + rest_lengths[i - 1]

    displacements = final_positions - initial_positions
    displacement_magnitudes = np.linalg.norm(displacements, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

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

    axes[1].plot(arc_lengths, displacement_magnitudes, 'b-o', markersize=6, linewidth=1.5)
    axes[1].set_xlabel('Arc Length (m)')
    axes[1].set_ylabel('Displacement (m)')
    axes[1].set_title('Node Displacements')
    axes[1].grid(True, alpha=0.3)

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
    """打印节点位移信息"""
    displacements = final_positions - initial_positions

    print("\n" + "=" * 60)
    print("最终节点位移:")
    print("=" * 60)
    for i in range(len(initial_positions)):
        disp_mag = np.linalg.norm(displacements[i])
        print(f"  节点 {i}: ux={displacements[i, 0]:+.6f}, uy={displacements[i, 1]:+.6f}, |u|={disp_mag:.6f} m")

    print(f"\n施力节点 {force_node} 最终位移: {np.linalg.norm(displacements[force_node]):.6f} m")


# ==================== 主程序 ====================

def main():

    # 1. 初始化杆件
    positions, rest_lengths, node_masses_arr = initialize_rod(total_length, num_segments)

    # 2. 设置固定节点（左端点）
    fixed_indices = np.array([0], dtype=int)

    # 3. 设置外力函数
    force_func, force_vector_display = create_force_function(
        force_mode, force_magnitude, force_direction, pulse_duration
    )
    force_func.force_vector = force_vector_display

    # 4. 运行仿真
    history = run_simulation(
        positions=positions,
        rest_lengths=rest_lengths,
        node_masses=node_masses_arr,
        EA=EA,
        EI=EI,
        gravity=gravity,
        damping=damping,
        dt=dt,
        total_time=total_time,
        fixed_indices=fixed_indices,
        force_func=force_func,
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
            final_force_mag = force_func(total_time)
            force_vector_plot = final_force_mag * (np.array(force_direction) / np.linalg.norm(force_direction))

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
    main()