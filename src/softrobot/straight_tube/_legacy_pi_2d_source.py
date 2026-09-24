"""
二维血管内离散导丝动力学：隐式 Euler + Newton + PI 血管壁保护力。

这个文件是独立可运行的二维原型，不修改任何已有的三维程序。
它保留离散弹性杆的两种内部能量：
    1. stretching energy（拉伸能）
    2. bending energy（弯曲能）

血管壁不作为弹性能的一部分。每个时间步都根据节点到上下壁的最近距离，
在保护层内施加朝血管中心的 PI 外力。以后换成三维 OBJ 血管时，只需替换
"最近距离 + 最近点法向"这一部分，其余隐式动力学框架仍可沿用。

单位统一采用 SI：m, kg, s, N。
依赖：numpy, torch, matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import matplotlib.pyplot as plt
import numpy as np
import torch


DTYPE = torch.float64


# ============================================================================
# 用户参数区：日常只修改这一段即可
# ============================================================================

# --- geometry（几何，单位 m） ---
rod_length = 0.05                 # 导丝总长度：50 mm
num_nodes = 20                    # 节点数，边数 = num_nodes - 1
vessel_radius = 0.0045            # 血管半径：4.5 mm；较紧管腔，便于观察 PI 调节作用
vessel_length = 1.30              # 血管轴向长度；60 s 后导丝前进 1.2 m，仍在显示范围内
# 血管中心线：先保留水平入口，再平滑过渡到 S 型弯曲。
vessel_inlet_straight_length = 0.06   # 入口直段长度，m；初始 5 cm 导丝完全位于该直段
vessel_s_amplitude = 0.004           # S 型中心线振幅，m；固定水平入口推进时仍保持在血管内
vessel_s_wavelength = 0.16            # S 型中心线波长，m
vessel_s_ramp_length = 0.02           # S 型振幅平滑建立的过渡长度，m

# --- rod material（离散弹性杆参数） ---
EA = 100.0                        # 拉伸刚度，N
EI = 1.0e-4                       # 弯曲刚度，N*m^2；原型值，可继续标定
total_free_mass = 0.002           # 自由节点总质量，kg；节点数变化时总质量保持不变
damping = 0.005                   # 线性速度阻尼，N*s/m；保留接触后的横向动力学响应

# --- boundary（血管壁保护层） ---
safe_distance = 0.003            # 保护层厚度：1.5 mm，小于它才产生壁力

# --- PI（壁面力控制器） ---
# Kp 单位 N/m，Ki 单位 N/(m*s)。误差为距离不足量 e = d_safe - d。
# 1 mm 的距离误差初始约产生 Kp * 0.001 = 0.20 N 的斥力。
Kp = 200.0
Ki = 20
max_wall_force_per_node = 2.0     # 每个节点壁力上限，N；防止极端数值状态
# PI 平滑与抗积分饱和：只影响保护层边缘的数值过渡，不改变弹性杆能量。
pi_fade_width = 0.0002            # m；在保护层边界内 0.2 mm 使总 PI 力平滑淡入/淡出
integral_leak_rate = 1.0          # 1/s；节点离壁时积分记忆按指数近似逐步衰减
integral_error_limit = 0.0015     # m*s；限制每节点积分状态，避免长期贴壁累计过大

# --- motion（入口速度边界） ---
velocity = 0.02                   # 推进速度，m/s
advance_angle_deg = 0.0           # 入口推进速度方向：0 deg 为向右
initial_angle_deg = 0.0           # 初始导丝直线方向；默认与水平入口段一致

# --- simulation（隐式动力学） ---
dt = 0.02                         # 固定隐式时间步长，s；兼顾动态准确性与计算量
total_time = 10                 # 总时间，s；以当前 0.02 m/s 推进 1.2 m
max_newton_iters = 16
residual_tol = 1.0e-5
step_tol = 1.0e-10
line_search_min_alpha = 1.0 / 128.0

# --- output（输出与显示） ---
print_every = 1                   # 每多少步打印一次；设为 5 可少打印一些
save_every = 1                     # 每多少步保存一次历史状态
show_animation = True              # True 时显示导丝进入过程动画
show_final_result = True           # True 时显示最终图
save_animation = False             # True 时保存 GIF；False 时仅弹出动画窗口
save_animation_path = "2d_vessel_boundary_PI_motion.gif"  # GIF 输出文件名
animation_fps = 12                 # GIF 播放帧率
animation_frame_stride = 2         # 每隔多少保存状态取一帧，减小 GIF 文件体积
animation_follow_window = 0.16     # 动画跟随窗口宽度，m；保证 5 cm 导丝可清楚看见
animation_repeat = True            # True 时动画循环播放，直到手动关闭图形窗口


@dataclass
class NewtonInfo:
    """记录单个隐式时间步的 Newton 收敛信息。"""

    converged: bool
    iterations: int
    residual_norm: float
    step_norm: float
    line_search_failures: int


def initialize_rod(
    length: float,
    number_of_nodes: int,
    initial_angle_degrees: float,
    free_mass: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """建立初始直导丝，节点 0 是受速度约束的入口节点。"""
    if number_of_nodes < 3:
        raise ValueError("num_nodes 至少应为 3，弯曲能需要三个节点。")

    number_of_edges = number_of_nodes - 1
    segment_length = length / number_of_edges
    angle = np.deg2rad(initial_angle_degrees)
    tangent = np.array([np.cos(angle), np.sin(angle)])

    positions = np.arange(number_of_nodes, dtype=np.float64)[:, None] * segment_length * tangent
    rest_lengths = np.full(number_of_edges, segment_length, dtype=np.float64)

    # 入口节点由速度边界直接指定，不参与惯性方程；其余节点均分总质量。
    node_masses = np.full(number_of_nodes, free_mass / (number_of_nodes - 1), dtype=np.float64)
    node_masses[0] = 0.0
    return positions, rest_lengths, node_masses


def compute_total_elastic_energy_torch(
    positions: torch.Tensor,
    rest_lengths: torch.Tensor,
    stretch_stiffness: float,
    bend_stiffness: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算二维 DER 的拉伸能、弯曲能和总弹性能。"""
    edges = positions[1:] - positions[:-1]
    edge_lengths = torch.linalg.vector_norm(edges, dim=1).clamp_min(1.0e-12)
    tangents = edges / edge_lengths[:, None]

    stretch_energy = 0.5 * stretch_stiffness * torch.sum(
        (edge_lengths - rest_lengths).square() / rest_lengths
    )

    # 离散曲率 kappa = 2 cross(t_i, t_{i+1}) / (1 + dot(t_i, t_{i+1}))。
    # 相邻边接近反向时用 clamp 防止除零；正常小弯曲情况下不影响结果。
    cross = tangents[:-1, 0] * tangents[1:, 1] - tangents[:-1, 1] * tangents[1:, 0]
    dot = torch.sum(tangents[:-1] * tangents[1:], dim=1)
    kappa = 2.0 * cross / (1.0 + dot).clamp_min(1.0e-10)
    vertex_lengths = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
    bend_energy = 0.5 * bend_stiffness * torch.sum(kappa.square() / vertex_lengths)

    return stretch_energy, bend_energy, stretch_energy + bend_energy


def compute_elastic_force(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    stretch_stiffness: float,
    bend_stiffness: float,
) -> np.ndarray:
    """用于诊断的 NumPy 接口：通过自动微分由弹性能计算全部节点内力。"""
    positions_t = torch.as_tensor(positions, dtype=DTYPE).clone().detach().requires_grad_(True)
    rest_lengths_t = torch.as_tensor(rest_lengths, dtype=DTYPE)
    _, _, energy = compute_total_elastic_energy_torch(
        positions_t, rest_lengths_t, stretch_stiffness, bend_stiffness
    )
    (gradient,) = torch.autograd.grad(energy, positions_t)
    return (-gradient).detach().cpu().numpy()


def _centerline_numpy(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """S 型中心线及其一、二阶导数。入口直段后平滑建立振幅。"""
    s = np.maximum(x - vessel_inlet_straight_length, 0.0)
    active = x > vessel_inlet_straight_length
    k = 2.0 * np.pi / vessel_s_wavelength
    ramp = vessel_s_ramp_length
    exp_term = np.exp(-s / ramp)
    growth = 1.0 - exp_term

    y = vessel_s_amplitude * np.sin(k * s) * growth
    dy = vessel_s_amplitude * (k * np.cos(k * s) * growth + np.sin(k * s) * exp_term / ramp)
    ddy = vessel_s_amplitude * (
        -k * k * np.sin(k * s) * growth
        + 2.0 * k * np.cos(k * s) * exp_term / ramp
        - np.sin(k * s) * exp_term / (ramp * ramp)
    )
    return np.where(active, y, 0.0), np.where(active, dy, 0.0), np.where(active, ddy, 0.0)


def _centerline_torch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """与 NumPy 版本一致的可微 S 型中心线表达式。"""
    s = torch.clamp(x - vessel_inlet_straight_length, min=0.0)
    active = x > vessel_inlet_straight_length
    k = 2.0 * np.pi / vessel_s_wavelength
    exp_term = torch.exp(-s / vessel_s_ramp_length)
    growth = 1.0 - exp_term
    y = vessel_s_amplitude * torch.sin(k * s) * growth
    dy = vessel_s_amplitude * (
        k * torch.cos(k * s) * growth + torch.sin(k * s) * exp_term / vessel_s_ramp_length
    )
    ddy = vessel_s_amplitude * (
        -k * k * torch.sin(k * s) * growth
        + 2.0 * k * torch.cos(k * s) * exp_term / vessel_s_ramp_length
        - torch.sin(k * s) * exp_term / (vessel_s_ramp_length * vessel_s_ramp_length)
    )
    zeros = torch.zeros_like(x)
    return torch.where(active, y, zeros), torch.where(active, dy, zeros), torch.where(active, ddy, zeros)


def _closest_centerline_coordinates_numpy(positions: np.ndarray) -> np.ndarray:
    """对每个节点解二维曲线的最近点投影；u 是该最近中心线点的 x 坐标。"""
    u = np.clip(positions[:, 0].copy(), 0.0, vessel_length)
    for _ in range(6):
        cy, dy, ddy = _centerline_numpy(u)
        residual_x = u - positions[:, 0]
        residual_y = cy - positions[:, 1]
        numerator = residual_x + residual_y * dy
        denominator = 1.0 + dy * dy + residual_y * ddy
        u = np.clip(u - numerator / np.where(np.abs(denominator) > 1.0e-10, denominator, 1.0e-10), 0.0, vessel_length)
    return u


def compute_boundary_distance(positions: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算节点到 S 型血管壁的最近距离，并返回指向血管中心的局部法向。"""
    u = _closest_centerline_coordinates_numpy(positions)
    center_y, center_dy, _ = _centerline_numpy(u)
    center_points = np.column_stack((u, center_y))
    center_normals = np.column_stack((-center_dy, np.ones_like(center_dy)))
    center_normals /= np.linalg.norm(center_normals, axis=1, keepdims=True)

    signed_normal_distance = np.sum((positions - center_points) * center_normals, axis=1)
    distance_to_wall = radius - np.abs(signed_normal_distance)
    # 节点位于中心线法向正侧时，回推方向为 -normal；反侧则为 +normal。
    force_normals = -np.sign(signed_normal_distance)[:, None] * center_normals
    force_normals[np.abs(signed_normal_distance) < 1.0e-14] = 0.0
    return distance_to_wall, force_normals, signed_normal_distance


def sample_vessel_boundaries(number_of_samples: int = 1200) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成中心线、两侧真实血管壁和两条保护层边界，供可视化使用。"""
    x = np.linspace(0.0, vessel_length, number_of_samples)
    center_y, center_dy, _ = _centerline_numpy(x)
    centerline = np.column_stack((x, center_y))
    normal = np.column_stack((-center_dy, np.ones_like(center_dy)))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    wall_positive = centerline + vessel_radius * normal
    wall_negative = centerline - vessel_radius * normal
    safe_positive = centerline + (vessel_radius - safe_distance) * normal
    safe_negative = centerline - (vessel_radius - safe_distance) * normal
    return centerline, wall_positive, wall_negative, safe_positive, safe_negative


def _smooth_pi_engagement_numpy(error: np.ndarray) -> np.ndarray:
    """C1 smoothstep：e=0 时总 PI 力和导数均平滑趋于零。"""
    z = np.clip(error / pi_fade_width, 0.0, 1.0)
    return z * z * (3.0 - 2.0 * z)


def _smooth_pi_engagement_torch(error: torch.Tensor) -> torch.Tensor:
    """与 NumPy 版本一致的可微 smoothstep。"""
    z = torch.clamp(error / pi_fade_width, 0.0, 1.0)
    return z * z * (3.0 - 2.0 * z)


def compute_PI_wall_force(
    positions: np.ndarray,
    integral_error: np.ndarray,
    time_step: float,
    radius: float,
    protection_distance: float,
    proportional_gain: float,
    integral_gain: float,
    force_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算 PI 壁面外力，并返回本步要提交的每个节点积分量。"""
    distances, normals, _ = compute_boundary_distance(positions, radius)
    error = np.maximum(protection_distance - distances, 0.0)
    active = error > 0.0

    # 积分状态连续泄漏而不是离开保护层时突然清零，避免 Ki*I 产生力跳变。
    decay = max(0.0, 1.0 - integral_leak_rate * time_step)
    next_integral = np.clip(
        decay * integral_error + error * time_step,
        0.0,
        integral_error_limit,
    )
    engagement = _smooth_pi_engagement_numpy(error)
    magnitudes = engagement * (proportional_gain * error + integral_gain * next_integral)
    magnitudes = np.clip(magnitudes, 0.0, force_limit)
    return magnitudes[:, None] * normals, next_integral, distances, active


def _compute_PI_wall_force_torch(
    positions: torch.Tensor,
    integral_error_old: torch.Tensor,
    time_step: float,
) -> torch.Tensor:
    """可微 S 型血管版本，专供 Newton 残差/Jacobian 使用；壁力仍是外力。"""
    # 固定次数的 Newton 投影求最近中心线点；整个过程可由 autograd 求导。
    u = torch.clamp(positions[:, 0], 0.0, vessel_length)
    k = 2.0 * np.pi / vessel_s_wavelength
    for _ in range(6):
        s = torch.clamp(u - vessel_inlet_straight_length, min=0.0)
        active_curve = u > vessel_inlet_straight_length
        exp_term = torch.exp(-s / vessel_s_ramp_length)
        growth = 1.0 - exp_term
        center_y = vessel_s_amplitude * torch.sin(k * s) * growth
        center_dy = vessel_s_amplitude * (
            k * torch.cos(k * s) * growth + torch.sin(k * s) * exp_term / vessel_s_ramp_length
        )
        center_ddy = vessel_s_amplitude * (
            -k * k * torch.sin(k * s) * growth
            + 2.0 * k * torch.cos(k * s) * exp_term / vessel_s_ramp_length
            - torch.sin(k * s) * exp_term / (vessel_s_ramp_length * vessel_s_ramp_length)
        )
        center_y = torch.where(active_curve, center_y, torch.zeros_like(center_y))
        center_dy = torch.where(active_curve, center_dy, torch.zeros_like(center_dy))
        center_ddy = torch.where(active_curve, center_ddy, torch.zeros_like(center_ddy))
        residual_x = u - positions[:, 0]
        residual_y = center_y - positions[:, 1]
        numerator = residual_x + residual_y * center_dy
        denominator = 1.0 + center_dy.square() + residual_y * center_ddy
        safe_denominator = torch.where(
            torch.abs(denominator) > 1.0e-10,
            denominator,
            torch.where(denominator >= 0.0, torch.full_like(denominator, 1.0e-10), torch.full_like(denominator, -1.0e-10)),
        )
        u = torch.clamp(u - numerator / safe_denominator, 0.0, vessel_length)

    s = torch.clamp(u - vessel_inlet_straight_length, min=0.0)
    active_curve = u > vessel_inlet_straight_length
    exp_term = torch.exp(-s / vessel_s_ramp_length)
    growth = 1.0 - exp_term
    center_y = vessel_s_amplitude * torch.sin(k * s) * growth
    center_dy = vessel_s_amplitude * (
        k * torch.cos(k * s) * growth + torch.sin(k * s) * exp_term / vessel_s_ramp_length
    )
    center_y = torch.where(active_curve, center_y, torch.zeros_like(center_y))
    center_dy = torch.where(active_curve, center_dy, torch.zeros_like(center_dy))
    normal = torch.stack((-center_dy, torch.ones_like(center_dy)), dim=1)
    normal = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True)
    center_points = torch.stack((u, center_y), dim=1)
    signed_distance = torch.sum((positions - center_points) * normal, dim=1)
    distance = vessel_radius - torch.abs(signed_distance)
    error = torch.clamp(safe_distance - distance, min=0.0)
    # 积分状态使用泄漏和上限；smoothstep 让总 PI 力在 e=0 时连续淡出。
    decay = max(0.0, 1.0 - integral_leak_rate * time_step)
    integral_trial = torch.clamp(
        decay * integral_error_old + error * time_step,
        min=0.0,
        max=integral_error_limit,
    )
    engagement = _smooth_pi_engagement_torch(error)
    magnitude = engagement * (Kp * error + Ki * integral_trial)
    magnitude = torch.clamp(magnitude, min=0.0, max=max_wall_force_per_node)

    force_direction = -torch.sign(signed_distance)[:, None] * normal
    return magnitude[:, None] * force_direction


def _pack_free(positions: torch.Tensor, free_indices: torch.Tensor) -> torch.Tensor:
    return positions[free_indices].reshape(-1)


def _unpack_free(
    q_flat: torch.Tensor,
    old_positions: torch.Tensor,
    free_indices: torch.Tensor,
    driven_position_new: torch.Tensor,
) -> torch.Tensor:
    """把自由变量向量还原成完整位置，并强制入口节点位于指定新位置。"""
    positions = old_positions.clone()
    positions[free_indices] = q_flat.reshape(-1, 2)
    positions[0] = driven_position_new
    return positions


def _implicit_residual(
    q_flat: torch.Tensor,
    old_positions: torch.Tensor,
    old_velocities: torch.Tensor,
    rest_lengths: torch.Tensor,
    node_masses: torch.Tensor,
    integral_error_old: torch.Tensor,
    free_indices: torch.Tensor,
    driven_position_new: torch.Tensor,
    time_step: float,
) -> torch.Tensor:
    """隐式 Euler 残差：惯性 + 阻尼 + dE/dx - F_wall = 0。"""
    positions_new = _unpack_free(q_flat, old_positions, free_indices, driven_position_new)
    _, _, elastic_energy = compute_total_elastic_energy_torch(positions_new, rest_lengths, EA, EI)
    (elastic_gradient,) = torch.autograd.grad(elastic_energy, q_flat, create_graph=True, retain_graph=True)

    q_old = _pack_free(old_positions, free_indices)
    v_old = _pack_free(old_velocities, free_indices)
    masses_free = node_masses[free_indices].repeat_interleave(2)
    v_new = (q_flat - q_old) / time_step
    inertia = masses_free * (q_flat - q_old - time_step * v_old) / (time_step * time_step)
    damping_term = damping * v_new

    wall_forces = _compute_PI_wall_force_torch(positions_new, integral_error_old, time_step)
    wall_force_free = _pack_free(wall_forces, free_indices)
    return inertia + damping_term + elastic_gradient - wall_force_free


def _residual_and_jacobian(q_flat: torch.Tensor, *args) -> tuple[torch.Tensor, torch.Tensor]:
    """由自动微分构建 Newton 所需的残差和完整 Jacobian。"""
    q = q_flat.detach().clone().requires_grad_(True)
    residual = _implicit_residual(q, *args)
    jacobian_rows = []
    for residual_component in residual:
        (row,) = torch.autograd.grad(residual_component, q, retain_graph=True)
        jacobian_rows.append(row)
    return residual.detach(), torch.stack(jacobian_rows).detach()


def newton_solver(
    old_positions: torch.Tensor,
    old_velocities: torch.Tensor,
    rest_lengths: torch.Tensor,
    node_masses: torch.Tensor,
    integral_error_old: torch.Tensor,
    free_indices: torch.Tensor,
    driven_position_new: torch.Tensor,
    time_step: float,
) -> tuple[torch.Tensor, NewtonInfo]:
    """对一个时间步的自由节点位置执行 Newton-Raphson + 线搜索。"""
    q = _pack_free(old_positions, free_indices).detach().clone()
    residual_args = (
        old_positions, old_velocities, rest_lengths, node_masses, integral_error_old,
        free_indices, driven_position_new, time_step,
    )
    failures = 0
    last_step_norm = np.inf

    for iteration in range(1, max_newton_iters + 1):
        residual, jacobian = _residual_and_jacobian(q, *residual_args)
        residual_norm = float(torch.linalg.vector_norm(residual).item())
        if residual_norm <= residual_tol:
            return q, NewtonInfo(True, iteration - 1, residual_norm, last_step_norm, failures)

        # 小型二维原型直接稠密求解；奇异时做极弱正则化后再求。
        try:
            step = torch.linalg.solve(jacobian, -residual)
        except RuntimeError:
            regularized = jacobian + 1.0e-10 * torch.eye(jacobian.shape[0], dtype=DTYPE)
            step = torch.linalg.solve(regularized, -residual)

        last_step_norm = float(torch.linalg.vector_norm(step).item())
        alpha = 1.0
        accepted = False
        while alpha >= line_search_min_alpha:
            candidate = q + alpha * step
            candidate_residual = _implicit_residual(candidate.detach().clone().requires_grad_(True), *residual_args)
            candidate_norm = float(torch.linalg.vector_norm(candidate_residual).item())
            if candidate_norm < residual_norm:
                q = candidate.detach()
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            failures += 1
            return q, NewtonInfo(False, iteration, residual_norm, last_step_norm, failures)
        if alpha * last_step_norm <= step_tol:
            return q, NewtonInfo(True, iteration, candidate_norm, alpha * last_step_norm, failures)

    final_residual = _implicit_residual(q.detach().clone().requires_grad_(True), *residual_args)
    return q, NewtonInfo(False, max_newton_iters, float(torch.linalg.vector_norm(final_residual).item()), last_step_norm, failures)


def implicit_euler_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    rest_lengths: np.ndarray,
    node_masses: np.ndarray,
    integral_error: np.ndarray,
    drive_velocity: np.ndarray,
    time_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, NewtonInfo]:
    """推进一个时间步，并在收敛后更新 PI 积分状态。"""
    old_positions_t = torch.as_tensor(positions, dtype=DTYPE)
    old_velocities_t = torch.as_tensor(velocities, dtype=DTYPE)
    rest_lengths_t = torch.as_tensor(rest_lengths, dtype=DTYPE)
    node_masses_t = torch.as_tensor(node_masses, dtype=DTYPE)
    integral_old_t = torch.as_tensor(integral_error, dtype=DTYPE)
    free_indices_t = torch.arange(1, len(positions), dtype=torch.long)
    # 入口速度方向固定，不根据血管中心线改变。
    driven_position_new = old_positions_t[0] + torch.as_tensor(drive_velocity * time_step, dtype=DTYPE)

    q_new, info = newton_solver(
        old_positions_t, old_velocities_t, rest_lengths_t, node_masses_t,
        integral_old_t, free_indices_t, driven_position_new, time_step,
    )
    if not info.converged:
        raise RuntimeError(
            f"Newton 未收敛：residual={info.residual_norm:.3e}, iterations={info.iterations}. "
            "请减小 dt 或降低 velocity。"
        )

    positions_new_t = _unpack_free(q_new, old_positions_t, free_indices_t, driven_position_new)
    positions_new = positions_new_t.detach().cpu().numpy()
    velocities_new = (positions_new - positions) / time_step
    velocities_new[0] = drive_velocity

    wall_force, integral_new, distances, active = compute_PI_wall_force(
        positions_new, integral_error, time_step, vessel_radius, safe_distance, Kp, Ki, max_wall_force_per_node
    )
    penetrating = distances < 0.0
    penetration_depths = np.maximum(-distances, 0.0)
    diagnostics = {
        "wall_force": wall_force,
        "distances": distances,
        "active": active,
        "penetrating": penetrating,
        "penetration_depths": penetration_depths,
        "minimum_wall_distance": float(np.min(distances)),
        "max_penetration_depth": float(np.max(penetration_depths)),
        "max_wall_force": float(np.max(np.linalg.norm(wall_force, axis=1))),
    }
    return positions_new, velocities_new, integral_new, diagnostics, info


def run_simulation() -> tuple[dict, np.ndarray, np.ndarray]:
    """运行完整仿真。入口节点以固定速度推进，其余节点由隐式动力学求解。"""
    initial_positions, rest_lengths, node_masses = initialize_rod(
        rod_length, num_nodes, initial_angle_deg, total_free_mass
    )
    initial_distances, initial_normals, _ = compute_boundary_distance(initial_positions, vessel_radius)

    positions = initial_positions.copy()
    velocities = np.zeros_like(positions)
    integral_error = np.zeros(num_nodes, dtype=np.float64)
    advance_angle = np.deg2rad(advance_angle_deg)
    drive_velocity = velocity * np.array([np.cos(advance_angle), np.sin(advance_angle)])
    num_steps = int(round(total_time / dt))

    # t=0 也记录真实保护层状态；积分项此时仍为零，因此只有比例项 Kp*e。
    initial_error = np.maximum(safe_distance - initial_distances, 0.0)
    initial_active = initial_error > 0.0
    initial_magnitude = np.clip(
        _smooth_pi_engagement_numpy(initial_error) * Kp * initial_error,
        0.0,
        max_wall_force_per_node,
    )
    initial_wall_force = initial_magnitude[:, None] * initial_normals
    initial_penetrating = initial_distances < 0.0
    initial_penetration_depths = np.maximum(-initial_distances, 0.0)
    history = {"time": [0.0], "positions": [positions.copy()], "wall_force": [initial_wall_force],
               "active": [initial_active], "penetrating": [initial_penetrating],
               "newton_iterations": [0], "max_wall_force": [float(np.max(initial_magnitude))],
               "minimum_wall_distance": [float(np.min(initial_distances))],
               "max_penetration_depth": [float(np.max(initial_penetration_depths))]}

    print("=" * 78)
    print("2D vessel guidewire: implicit Euler + Newton + PI wall protection")
    print("=" * 78)
    print(f"nodes={num_nodes}, rod_length={rod_length:.4f} m, vessel_radius={vessel_radius:.4f} m")
    print(f"S vessel: inlet={vessel_inlet_straight_length:g} m, amplitude={vessel_s_amplitude:g} m, wavelength={vessel_s_wavelength:g} m")
    print(f"EA={EA:g} N, EI={EI:g} N*m^2, total_free_mass={total_free_mass:g} kg")
    print(f"velocity={velocity:g} m/s, advance_angle={advance_angle_deg:g} deg, initial_angle={initial_angle_deg:g} deg")
    print(f"safe_distance={safe_distance:.4f} m, Kp={Kp:g} N/m, Ki={Ki:g} N/(m*s)")
    print(f"PI fade={pi_fade_width * 1e3:g} mm, leak={integral_leak_rate:g} 1/s, I_limit={integral_error_limit:g} m*s")
    print(f"dt={dt:g} s, total_time={total_time:g} s, steps={num_steps}")
    print("-" * 78)

    start = time.perf_counter()
    for step in range(1, num_steps + 1):
        positions, velocities, integral_error, diagnostics, info = implicit_euler_step(
            positions, velocities, rest_lengths, node_masses, integral_error, drive_velocity, dt
        )
        current_time = step * dt

        if step % save_every == 0 or step == num_steps:
            history["time"].append(current_time)
            history["positions"].append(positions.copy())
            history["wall_force"].append(diagnostics["wall_force"].copy())
            history["active"].append(diagnostics["active"].copy())
            history["penetrating"].append(diagnostics["penetrating"].copy())
            history["newton_iterations"].append(info.iterations)
            history["max_wall_force"].append(diagnostics["max_wall_force"])
            history["minimum_wall_distance"].append(diagnostics["minimum_wall_distance"])
            history["max_penetration_depth"].append(diagnostics["max_penetration_depth"])

        if step % print_every == 0 or step == num_steps:
            print(
                f"step {step:4d}/{num_steps}  t={current_time:7.4f}  "
                f"Newton={info.iterations:2d}  residual={info.residual_norm:.3e}  "
                f"active={np.count_nonzero(diagnostics['active']):2d}  "
                f"max_PI={diagnostics['max_wall_force']:.3e} N  "
                f"min_d={diagnostics['minimum_wall_distance'] * 1e3:+.3f} mm  "
                f"max_pen={diagnostics['max_penetration_depth'] * 1e3:.3f} mm"
            )
            if np.any(diagnostics["penetrating"]):
                for node_index in np.flatnonzero(diagnostics["penetrating"]):
                    depth_mm = diagnostics["penetration_depths"][node_index] * 1e3
                    print(f"  Warning: node {node_index} penetration = {depth_mm:.3f} mm at t={current_time:.4f} s")

    print("-" * 78)
    print(f"done: wall_time={time.perf_counter() - start:.3f} s")
    print(f"final inlet position = {positions[0]}")
    print(f"final tip position   = {positions[-1]}")
    print(f"minimum wall distance = {np.min(compute_boundary_distance(positions, vessel_radius)[0]):.6e} m")
    print("=" * 78)
    return history, initial_positions, rest_lengths


def visualize_result(history: dict, initial_positions: np.ndarray, rest_lengths: np.ndarray) -> None:
    """显示血管壁、初始/最终导丝、安全层节点以及最终壁面力箭头。"""
    final_positions = history["positions"][-1]
    final_wall_force = history["wall_force"][-1]
    active = history["active"][-1]
    penetrating = history["penetrating"][-1]

    fig, (ax_shape, ax_force) = plt.subplots(1, 2, figsize=(15, 5.5))
    x_values = np.concatenate([initial_positions[:, 0], final_positions[:, 0]])
    x_margin = max(0.01, 0.1 * (x_values.max() - x_values.min()))
    centerline, wall_positive, wall_negative, safe_positive, safe_negative = sample_vessel_boundaries()
    # 最终图展示完整 S 型血管范围；动画会使用更近的跟随视窗。
    x_min = min(0.0, x_values.min() - x_margin)
    x_max = max(vessel_length, x_values.max() + x_margin)

    # 左图：S 型中心线、两侧曲线血管壁、保护层与最终受力。
    ax_shape.plot(wall_positive[:, 0], wall_positive[:, 1], color="black", linewidth=2.0, label="vessel wall")
    ax_shape.plot(wall_negative[:, 0], wall_negative[:, 1], color="black", linewidth=2.0)
    ax_shape.plot(safe_positive[:, 0], safe_positive[:, 1], color="orange", linestyle="--", alpha=0.8, label="safe layer")
    ax_shape.plot(safe_negative[:, 0], safe_negative[:, 1], color="orange", linestyle="--", alpha=0.8)
    ax_shape.plot(centerline[:, 0], centerline[:, 1], color="0.55", linestyle=":", linewidth=1.0, label="vessel centerline")
    ax_shape.plot(initial_positions[:, 0], initial_positions[:, 1], "--", color="royalblue", label="initial rod")
    ax_shape.plot(final_positions[:, 0], final_positions[:, 1], "-o", color="crimson", markersize=4, label="final rod")
    ax_shape.scatter(final_positions[active, 0], final_positions[active, 1], color="gold", edgecolor="black",
                     zorder=5, label="inside safe layer")
    ax_shape.scatter(final_positions[penetrating, 0], final_positions[penetrating, 1], marker="x", s=70,
                     linewidths=2.2, color="magenta", zorder=6, label="penetration")

    force_norm = np.linalg.norm(final_wall_force, axis=1)
    if np.any(force_norm > 0.0):
        arrow_scale = 0.002 / max(np.max(force_norm), 1.0e-12)
        ax_shape.quiver(
            final_positions[:, 0], final_positions[:, 1],
            final_wall_force[:, 0] * arrow_scale, final_wall_force[:, 1] * arrow_scale,
            angles="xy", scale_units="xy", scale=1.0, color="forestgreen", width=0.004,
            label="wall force direction",
        )

    ax_shape.set_xlim(x_min, x_max)
    wall_y = np.concatenate((wall_positive[:, 1], wall_negative[:, 1]))
    ax_shape.set_ylim(wall_y.min() - 0.002, wall_y.max() + 0.002)
    # 全长 1.3 m 而管径仅约 2 cm；总览使用 auto 才能清楚看见 S 型轨迹。
    ax_shape.set_aspect("auto")
    ax_shape.set_xlabel("x (m): insertion direction")
    ax_shape.set_ylabel("y (m): vessel height")
    ax_shape.set_title("2D guidewire inside S-shaped vessel")
    ax_shape.grid(alpha=0.25)
    ax_shape.legend(loc="upper right")
    parameter_text = (
        f"velocity = {velocity:g} m/s\n"
        f"advance angle = {advance_angle_deg:g} deg\n"
        f"initial angle = {initial_angle_deg:g} deg\n"
        f"d_safe = {safe_distance * 1e3:g} mm\n"
        f"Kp = {Kp:g} N/m\nKi = {Ki:g} N/(m*s)"
    )
    ax_shape.text(0.02, 0.03, parameter_text, transform=ax_shape.transAxes, va="bottom",
                  bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.7"})

    # 右图：壁力和进入保护层节点数随时间变化。
    times = np.asarray(history["time"])
    max_forces = np.asarray(history["max_wall_force"])
    active_counts = np.asarray([np.count_nonzero(mask) for mask in history["active"]])
    ax_force.plot(times, max_forces, color="forestgreen", label="max wall force (N)")
    ax_force.set_xlabel("time (s)")
    ax_force.set_ylabel("max wall force (N)", color="forestgreen")
    ax_force.tick_params(axis="y", labelcolor="forestgreen")
    ax_force.grid(alpha=0.25)
    ax_nodes = ax_force.twinx()
    ax_nodes.step(times, active_counts, where="post", color="darkorange", label="nodes in safe layer")
    ax_nodes.set_ylabel("active nodes", color="darkorange")
    ax_nodes.tick_params(axis="y", labelcolor="darkorange")
    ax_force.set_title("PI wall-protection response")
    lines = ax_force.get_lines() + ax_nodes.get_lines()
    ax_force.legend(lines, [line.get_label() for line in lines], loc="upper left")

    fig.tight_layout()
    plt.show()


def visualize_animation(history: dict, initial_positions: np.ndarray) -> None:
    """保存并显示导丝前进动画；窗口随导丝移动，避免 5 cm 导丝在 1 m 视图中太小。"""
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(11, 4))
    centerline, wall_positive, wall_negative, safe_positive, safe_negative = sample_vessel_boundaries()
    wall_y = np.concatenate((wall_positive[:, 1], wall_negative[:, 1]))
    ax.set_ylim(wall_y.min() - 0.002, wall_y.max() + 0.002)
    ax.set_aspect("equal", adjustable="box")
    ax.plot(wall_positive[:, 0], wall_positive[:, 1], color="black", linewidth=2, label="vessel wall")
    ax.plot(wall_negative[:, 0], wall_negative[:, 1], color="black", linewidth=2)
    ax.plot(safe_positive[:, 0], safe_positive[:, 1], color="orange", linestyle="--", label="safe layer")
    ax.plot(safe_negative[:, 0], safe_negative[:, 1], color="orange", linestyle="--")
    ax.plot(centerline[:, 0], centerline[:, 1], color="0.55", linestyle=":", linewidth=1, label="centerline")
    ax.plot(initial_positions[:, 0], initial_positions[:, 1], "--", color="royalblue", label="initial")
    (rod_line,) = ax.plot([], [], "-o", color="crimson", markersize=4, label="guidewire")
    active_scatter = ax.scatter([], [], color="gold", edgecolor="black", zorder=5)
    penetration_scatter = ax.scatter([], [], marker="x", s=65, linewidths=2.0, color="magenta", zorder=6)
    title = ax.set_title("")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()

    frame_indices = list(range(0, len(history["positions"]), animation_frame_stride))
    if frame_indices[-1] != len(history["positions"]) - 1:
        frame_indices.append(len(history["positions"]) - 1)

    def update(frame_index: int):
        state = history["positions"][frame_index]
        mask = history["active"][frame_index]
        penetration_mask = history["penetrating"][frame_index]
        rod_line.set_data(state[:, 0], state[:, 1])
        active_scatter.set_offsets(state[mask] if np.any(mask) else np.empty((0, 2)))
        penetration_scatter.set_offsets(state[penetration_mask] if np.any(penetration_mask) else np.empty((0, 2)))
        # 跟随入口节点，但不显示超过设定血管长度的区域。
        x_left = np.clip(state[0, 0] - 0.03, 0.0, max(vessel_length - animation_follow_window, 0.0))
        x_right = min(vessel_length, x_left + animation_follow_window)
        if x_right - x_left < animation_follow_window:
            x_left = max(0.0, x_right - animation_follow_window)
        ax.set_xlim(x_left, x_right)
        title.set_text(f"t = {history['time'][frame_index]:.2f} s | active wall nodes = {np.count_nonzero(mask)}")
        return rod_line, active_scatter, penetration_scatter, title

    animation = FuncAnimation(
        fig, update, frames=frame_indices, interval=1000 / animation_fps, blit=False, repeat=animation_repeat
    )
    if save_animation:
        try:
            animation.save(save_animation_path, writer="pillow", fps=animation_fps, dpi=100)
            print(f"animation saved: {save_animation_path}")
        except (ImportError, RuntimeError) as error:
            print(f"animation could not be saved: {error}")
    plt.show()


if __name__ == "__main__":
    simulation_history, initial_rod, rod_rest_lengths = run_simulation()
    # 先保存 GIF，避免最终静态图窗口关闭前动画文件尚未生成。
    if show_animation:
        visualize_animation(simulation_history, initial_rod)
    if show_final_result:
        visualize_result(simulation_history, initial_rod, rod_rest_lengths)
