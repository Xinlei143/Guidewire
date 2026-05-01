"""
二维离散弹性杆工具箱
基于导师文档和原始能工作的DER算法

核心公式（与原始 rodtoolbox_2d_ks_kb.py 一致）：
1. 拉伸力: F = EA * (l - l̄) / l̄
2. 离散曲率: κ = 2(t_{i-1} × t_i) / (1 + t_{i-1}·t_i)
3. 弯曲力: F_i = EI * κ / Δs_i * (n_next - n_prev)
   F_{i-1} = -EI * κ / Δs_i * n_prev * 0.5
   F_{i+1} = EI * κ / Δs_i * n_next * 0.5
"""

import numpy as np
from typing import Tuple


def normalize_vector(v: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    """归一化向量"""
    length = np.linalg.norm(v)
    if length < eps:
        return np.zeros_like(v), 0.0
    return v / length, length


def compute_edge_geometry(
    positions: np.ndarray,
    eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算边向量、边长和切向量"""
    num_edges = positions.shape[0] - 1
    edge_vectors = np.zeros((num_edges, 2))
    edge_lengths = np.zeros(num_edges)
    tangents = np.zeros((num_edges, 2))

    for i in range(num_edges):
        edge_vectors[i] = positions[i+1] - positions[i]
        tangents[i], edge_lengths[i] = normalize_vector(edge_vectors[i], eps)

    return edge_vectors, edge_lengths, tangents


def compute_stretching_forces(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    stiffness: float
) -> np.ndarray:
    """
    拉伸力计算（与原始代码一致）

    公式: F = EA * (l - l̄) / l̄
    """
    num_nodes = positions.shape[0]
    num_edges = num_nodes - 1
    forces = np.zeros_like(positions)

    if stiffness == 0.0:
        return forces

    _, edge_lengths, tangents = compute_edge_geometry(positions)

    for i in range(num_edges):
        current_len = edge_lengths[i]
        rest_len = rest_lengths[i]

        if rest_len < 1e-12:
            continue

        # 拉伸力大小（与原始代码一致）
        stretch_magnitude = stiffness * (current_len - rest_len) / rest_len

        force_vec = stretch_magnitude * tangents[i]

        forces[i] += force_vec
        forces[i+1] -= force_vec

    return forces


def compute_bending_forces(
        positions: np.ndarray,
        rest_lengths: np.ndarray,
        stiffness: float
) -> np.ndarray:
    """
    弯曲力计算（修改版：Voronoi长度使用初始无应力长度）

    公式:
        κ = 2 * cross(t_{i-1}, t_i) / (1 + dot(t_{i-1}, t_i))
        Δs_i = (l̄_{i-1} + l̄_i) / 2  （使用初始长度，而非当前长度）
        F_i = EI * κ / Δs_i * (n_next - n_prev)
        F_{i-1} -= EI * κ / Δs_i * n_prev * 0.5
        F_{i+1} += EI * κ / Δs_i * n_next * 0.5
    """
    num_nodes = positions.shape[0]
    forces = np.zeros_like(positions)

    if stiffness == 0.0 or num_nodes < 3:
        return forces

    _, edge_lengths, tangents = compute_edge_geometry(positions)

    for i in range(1, num_nodes - 1):
        # 获取相邻边
        t_prev = tangents[i - 1]
        t_curr = tangents[i]
        len_prev = edge_lengths[i - 1]
        len_curr = edge_lengths[i]

        if len_prev < 1e-12 or len_curr < 1e-12:
            continue

        # 计算离散曲率
        cross_prod = t_prev[0] * t_curr[1] - t_prev[1] * t_curr[0]
        dot_prod = np.dot(t_prev, t_curr)
        denom = 1.0 + dot_prod

        if denom < 1e-8:
            kappa = 0.0
        else:
            kappa = 2.0 * cross_prod / denom

        # 【修改点】Voronoi长度使用初始无应力长度（符合文档要求）
        # 修改为:
        rest_len_prev = rest_lengths[i - 1]
        rest_len_curr = rest_lengths[i]
        voronoi_len = 0.5 * (rest_len_prev + rest_len_curr)

        if voronoi_len < 1e-12:
            continue

        # 法向量（垂直于切向量）
        n_prev = np.array([-t_prev[1], t_prev[0]])
        n_curr = np.array([-t_curr[1], t_curr[0]])

        # 根据曲率符号调整方向
        if cross_prod >= 0:
            n_prev = np.array([-t_prev[1], t_prev[0]])
            n_curr = np.array([-t_curr[1], t_curr[0]])
        else:
            n_prev = np.array([t_prev[1], -t_prev[0]])
            n_curr = np.array([t_curr[1], -t_curr[0]])

        # 弯曲力系数
        bend_coeff = stiffness * kappa / voronoi_len

        # 分配弯曲力
        forces[i] += bend_coeff * (n_curr - n_prev)
        forces[i - 1] -= bend_coeff * n_prev * 0.5
        forces[i + 1] += bend_coeff * n_curr * 0.5

    return forces


def compute_total_internal_forces(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    stretch_stiffness: float,
    bend_stiffness: float
) -> np.ndarray:
    """计算总内力"""
    stretch_forces = compute_stretching_forces(positions, rest_lengths, stretch_stiffness)
    bend_forces = compute_bending_forces(positions, rest_lengths, bend_stiffness)
    return stretch_forces + bend_forces


def compute_stretching_energy(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    stiffness: float
) -> float:
    """计算拉伸势能"""
    if stiffness == 0.0:
        return 0.0

    _, edge_lengths, _ = compute_edge_geometry(positions)
    energy = 0.0

    for i, current_len in enumerate(edge_lengths):
        rest_len = rest_lengths[i]
        if rest_len < 1e-12:
            continue
        delta = current_len - rest_len
        energy += 0.5 * stiffness * delta * delta / rest_len

    return energy


def compute_bending_energy(
        positions: np.ndarray,
        rest_lengths: np.ndarray,
        stiffness: float
) -> float:
    """
    计算弯曲势能（修改版：Voronoi长度使用初始无应力长度）

    公式: E_bend = Σ 0.5 * EI * κ² / Δs_i
    其中 Δs_i 使用初始长度
    """
    num_nodes = positions.shape[0]

    if stiffness == 0.0 or num_nodes < 3:
        return 0.0

    _, edge_lengths, tangents = compute_edge_geometry(positions)
    energy = 0.0

    for i in range(1, num_nodes - 1):
        if edge_lengths[i - 1] < 1e-12 or edge_lengths[i] < 1e-12:
            continue

        cross_prod = tangents[i - 1][0] * tangents[i][1] - tangents[i - 1][1] * tangents[i][0]
        dot_prod = np.dot(tangents[i - 1], tangents[i])
        denom = 1.0 + dot_prod

        if denom < 1e-8:
            kappa = 0.0
        else:
            kappa = 2.0 * cross_prod / denom

        # 【修改点】Voronoi长度使用初始无应力长度
        rest_len_prev = rest_lengths[i - 1]
        rest_len_curr = rest_lengths[i]
        voronoi_len = 0.5 * (rest_len_prev + rest_len_curr)

        if voronoi_len < 1e-12:
            continue

        energy += 0.5 * stiffness * kappa * kappa / voronoi_len

    return energy


def compute_total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    rest_lengths: np.ndarray,
    stretch_stiffness: float,
    bend_stiffness: float,
    node_masses: np.ndarray,
    gravity_g: float,
    external_forces: np.ndarray,
    fixed_indices: np.ndarray
) -> dict:
    """计算系统总能量"""
    free_indices = np.setdiff1d(np.arange(len(positions)), fixed_indices)

    stretch_energy = compute_stretching_energy(positions, rest_lengths, stretch_stiffness)
    bend_energy = compute_bending_energy(positions, rest_lengths, bend_stiffness)
    gravity_energy = np.sum(node_masses * gravity_g * positions[:, 1])
    kinetic_energy = 0.5 * np.sum(
        node_masses[free_indices, np.newaxis] * velocities[free_indices]**2
    )

    total = stretch_energy + bend_energy + gravity_energy + kinetic_energy

    return {
        'stretching_energy': stretch_energy,
        'bending_energy': bend_energy,
        'gravitational_energy': gravity_energy,
        'kinetic_energy': kinetic_energy,
        'total_energy': total
    }


def calculate_critical_dt(
    node_mass: float,
    segment_length: float,
    stretch_stiffness: float,
    bend_stiffness: float = None,
    safety_factor: float = 0.5
) -> float:
    """计算CFL稳定性限制"""
    if stretch_stiffness > 0:
        dt_stretch = np.sqrt(node_mass * segment_length / stretch_stiffness)
    else:
        dt_stretch = float('inf')

    if bend_stiffness is not None and bend_stiffness > 0:
        dt_bend = segment_length**2 * np.sqrt(node_mass / bend_stiffness) * 0.1
    else:
        dt_bend = float('inf')

    critical_dt = min(dt_stretch, dt_bend)
    return critical_dt * safety_factor


def integrate_semi_implicit_euler(
        positions: np.ndarray,
        velocities: np.ndarray,
        internal_forces: np.ndarray,
        external_forces: np.ndarray,
        node_masses: np.ndarray,
        dt: float,
        damping: float,
        gravity_g: float = -9.81,
        fixed_indices: np.ndarray = None,
        fixed_positions: np.ndarray = None  # 新增参数：固定节点的初始位置
) -> Tuple[np.ndarray, np.ndarray]:
    """
    半隐式欧拉积分器（完整版：固定节点完全锁定）

    参数:
        fixed_positions: 固定节点的目标位置（初始位置），形状为 (len(fixed_indices), 2)
    """
    if fixed_indices is None:
        fixed_indices = np.array([], dtype=int)

    free_indices = np.setdiff1d(np.arange(len(positions)), fixed_indices)

    # 初始化总力数组
    total_forces = np.zeros_like(positions)

    # 只对自由节点计算力
    total_forces[free_indices] = internal_forces[free_indices] + external_forces[free_indices]
    total_forces[free_indices, 1] += node_masses[free_indices] * gravity_g

    if damping > 0:
        total_forces[free_indices] -= damping * velocities[free_indices]

    # 更新自由节点
    if len(free_indices) > 0:
        accelerations = total_forces[free_indices] / node_masses[free_indices, np.newaxis]
        velocities[free_indices] += accelerations * dt
        positions[free_indices] += velocities[free_indices] * dt

    # 固定节点：速度置零，位置恢复为初始值
    if len(fixed_indices) > 0:
        velocities[fixed_indices] = 0.0
        if fixed_positions is not None:
            positions[fixed_indices] = fixed_positions

    return positions, velocities


def compute_bending_angles_degrees(
    positions: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """计算弯曲角度（度数）用于可视化"""
    num_nodes = positions.shape[0]
    angles_deg = np.zeros(num_nodes)

    if num_nodes < 3:
        return angles_deg

    _, _, tangents = compute_edge_geometry(positions, eps)

    for i in range(1, num_nodes - 1):
        t_prev = tangents[i-1]
        t_curr = tangents[i]

        dot_prod = np.dot(t_prev, t_curr)
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        angle_rad = np.arccos(dot_prod)

        # 根据叉积确定方向
        cross_prod = t_prev[0] * t_curr[1] - t_prev[1] * t_curr[0]
        if cross_prod < 0:
            angle_rad = -angle_rad

        angles_deg[i] = np.degrees(angle_rad)

    return angles_deg
