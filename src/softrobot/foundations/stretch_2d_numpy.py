from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .stretch_2d_numpy_toolbox import compute_internal_forces, integrate_semi_implicit_euler, calculate_critical_dt

# ================= 物理参数设置区 =================
total_length = 2      # 杆的总长度 (m)
segment_length = 0.1    # 每一段杆的长度 (m)
num_nodes = 10          # 中间的节点数量

EA = 1000.0             # 轴向抗拉刚度 (N)
node_mass = 0.1        # 每个节点的质量 (kg)
total_time = 10         # 总仿真时间 (s)
dt = 0.0005             # 时间步长 (s) -> 程序会自动检查这个值是否安全
damping_c = 0.5         # 阻尼系数（稍微调大一点，防止初始下垂时震荡太剧烈）
external_force_mag = 5 # 末端拉力 (N)

# 0-9 代表施加在第1到第10个节点上
# 10 代表施加在杆子的最末端（尾巴尖端）
force_node_index = 5  # 默认施加在末端

# 🎮 施力模式选择
force_mode = "constant"
pulse_duration = 0.01

# ==================================================

# 【自动计算杆段数量】
num_segments = int(round(total_length / segment_length))
print(f"总长 {total_length}m, 杆段长 {segment_length}m -> 共 {num_segments} 段")
if num_segments > num_nodes:
    print(f"注意：有 {num_segments - num_nodes} 段杆作为自由尾巴延伸在 {num_nodes} 个节点之后。")


# ==================================================

def run_simulation():
    # 1. 初始化节点位置
    positions = np.zeros((num_nodes, 2))
    for i in range(num_nodes):
        x = (i + 1) * segment_length
        positions[i] = [x, 0.0]

    # 所有杆段（包括尾巴）的初始无应力长度
    rest_lengths = np.full(num_segments, segment_length)

    velocities = np.zeros_like(positions)
    external_forces = np.zeros_like(positions)

    # ==================================================
    # CFL 稳定性检查
    # ==================================================
    critical_dt = calculate_critical_dt(node_mass, segment_length, EA, num_segments, num_nodes)

    print(f"计算得出的最大安全时间步长 (CFL Limit): {critical_dt:.6f} s")
    print(f"当前设定的时间步长 (dt): {dt:.6f} s")

    if dt > critical_dt:
        raise ValueError(f"时间步长 dt={dt} 过大！会导致数值不稳定。请将其减小到 {critical_dt} 以下。")
    else:
        print("时间步长检查通过：仿真数值稳定。")
    # ==================================================

    # ==================================================
    # 预松弛阶段 (Pre-relaxation)
    # 在正式记录动画前，先悄悄跑几百步，让杆子在重力作用下自然下垂一点
    # ==================================================
    print("正在进行预松弛（让杆子自然下垂）...")
    relax_steps = 500
    for _ in range(relax_steps):
        internal_forces = compute_internal_forces(positions, rest_lengths, EA, num_segments, num_nodes)
        # 预松弛阶段不施加外部拉力，只受重力和内力
        positions, velocities = integrate_semi_implicit_euler(
            positions, velocities, internal_forces, np.zeros_like(positions),
            node_mass, dt, damping_c, gravity_g=9.81
        )
    print("预松弛完成，开始正式仿真计算...")
    # ==================================================

    time_steps = int(total_time / dt)
    record_interval = 5
    recorded_positions = []

    for step in range(time_steps):
        current_time = step * dt

        # 🎮 【核心修改】根据选择的模式动态更新外力
        external_forces.fill(0.0)  # 先把每一步的外力清零

        # 确定当前应该施加多大的力
        current_force = 0.0
        if force_mode == "constant":
            current_force = external_force_mag
        elif force_mode == "pulse":
            if current_time < pulse_duration:
                current_force = external_force_mag

        # 根据选择的索引，把力施加到对应的位置
        if current_force != 0.0:
            if force_node_index < num_nodes:
                # 情况1：施加在真实的节点上（0到9）
                external_forces[force_node_index, 0] = current_force
            else:
                # 情况2：施加在末端（索引为10，即尾巴尖端）
                # 尾巴尖端没有质量，它的受力会通过杆段内力传递给最后一个真实节点
                # 所以我们直接把力加在最后一个节点（num_nodes - 1）上即可
                external_forces[num_nodes - 1, 0] = current_force

        # 传入 num_segments 和 num_nodes 供工具库判断尾巴
        internal_forces = compute_internal_forces(positions, rest_lengths, EA, num_segments, num_nodes)

        # 显式传入 gravity_g=9.81 开启重力
        positions, velocities = integrate_semi_implicit_euler(
            positions, velocities, internal_forces, external_forces,
            node_mass, dt, damping_c, gravity_g=9.81
        )

        if step % record_interval == 0:
            recorded_positions.append(positions.copy())

        if step % (time_steps // 5) == 0:
            print(f"进度: {step / time_steps * 100:.0f}%")

    print("仿真计算完成，开始生成可视化...")
    return np.array(recorded_positions)

def visualize(recorded_positions):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-0.2, total_length + 1.0)
    # 扩大 Y 轴范围，防止杆子掉出画面 (向下扩展到 -1.5m)
    ax.set_ylim(-1.5, 0.5)
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(f'Force Mode: {force_mode}, Force Magnitude: {external_force_mag}N, Gravity: ON')

    # 1. 生成初始位置的节点坐标 (不包含尾巴尖端，只看node)
    init_x = np.array([(i + 1) * segment_length for i in range(num_nodes)])
    init_y = np.zeros(num_nodes)

    # 2. 画出静态的初始节点位置（红色实心点），作为背景参照，没有任何连线
    ax.scatter(init_x, init_y, color='red', marker='o', s=25, label='Original Nodes (Horizontal)')

    # 3. 准备动态的当前杆线（蓝色实线）
    line, = ax.plot([], [], 'b-', lw=2, label='Current Rod')
    # 4. 准备动态的当前节点（蓝色空心圆圈）
    current_dots, = ax.plot([], [], 'go', markersize=5, label='Current Nodes')

    ax.legend(loc='upper right') # 显示图例

    def init():
        line.set_data([], [])
        current_dots.set_data([], [])
        return line, current_dots

    def update(frame):
        pos_frame = recorded_positions[frame]
        x = pos_frame[:, 0]
        y = pos_frame[:, 1]

        # 绘图逻辑：固定端(0,0) -> 10个节点 -> 尾巴自由端
        tail_length = (num_segments - num_nodes) * segment_length
        plot_x = np.concatenate(([0.0], x, [x[-1] + tail_length]))
        plot_y = np.concatenate(([0.0], y, [y[-1]]))

        # 更新当前杆线和当前节点的位置
        line.set_data(plot_x, plot_y)
        current_dots.set_data(x, y)
        return line, current_dots

    anim = animation.FuncAnimation(fig, update, frames=len(recorded_positions),
                                   init_func=init, blit=True, interval=20)
    plt.show()

def _run_smoke(output_dir: str) -> None:
    from .._smoke import configure_cpu_runtime, write_smoke_result

    configure_cpu_runtime()
    smoke_nodes = 4
    smoke_segments = 4
    smoke_positions = np.column_stack(
        (np.arange(1, smoke_nodes + 1, dtype=float) * 0.1, np.zeros(smoke_nodes))
    )
    smoke_velocities = np.zeros_like(smoke_positions)
    smoke_rest_lengths = np.full(smoke_segments, 0.1)
    smoke_external = np.zeros_like(smoke_positions)
    smoke_external[-1, 0] = 0.1
    for _ in range(2):
        internal = compute_internal_forces(
            smoke_positions, smoke_rest_lengths, EA, smoke_segments, smoke_nodes
        )
        smoke_positions, smoke_velocities = integrate_semi_implicit_euler(
            smoke_positions, smoke_velocities, internal, smoke_external,
            node_mass, dt, damping_c, gravity_g=0.0,
        )
    write_smoke_result(
        output_dir, "foundation-2d-stretch-numpy", smoke_positions, steps=2,
        velocity_shape=list(smoke_velocities.shape),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Historical 2-D stretching rod")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args(argv)
    if args.smoke_test:
        _run_smoke(args.output_dir)
        return 0
    all_positions = run_simulation()
    visualize(all_positions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
