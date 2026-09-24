"""
三维离散弹性杆工具箱 - 自动微分版本 (PyTorch)
优化版：整合 curvature-binormal 弯曲模型，保持高性能

所有力通过 PyTorch 自动微分从能量求导获得
"""

import numpy as np
import torch
from typing import Optional, Tuple


# ==================== 核心能量计算（PyTorch）====================

def compute_edge_geometry_torch(
    positions: torch.Tensor,
    min_length: float = 1e-12
):
    """
    计算边几何信息 (3D)

    参数:
        positions: (N, 3) 节点位置张量
        min_length: 最小边长，防止除零

    返回:
        edge_vectors: (N-1, 3) 边向量
        edge_lengths: (N-1,) 边长
        tangents: (N-1, 3) 单位切向量
    """
    edge_vectors = positions[1:] - positions[:-1]
    edge_lengths = torch.norm(edge_vectors, dim=1)
    safe_lengths = torch.clamp(edge_lengths, min=min_length)
    tangents = edge_vectors / safe_lengths[:, None]
    return edge_vectors, edge_lengths, tangents


def compute_stretching_energy_torch(
    positions: torch.Tensor,
    rest_lengths: torch.Tensor,
    EA: float,
    min_length: float = 1e-12
) -> torch.Tensor:
    """计算拉伸势能 (可扩展杆模型)"""
    if EA == 0.0:
        return positions.new_tensor(0.0)

    _, current_lengths, _ = compute_edge_geometry_torch(positions, min_length)
    rest_lengths_t = rest_lengths.to(dtype=positions.dtype, device=positions.device)
    delta = current_lengths - rest_lengths_t
    # k_i = EA / L_i^0
    energy = 0.5 * EA * torch.sum(delta ** 2 / rest_lengths_t)
    return energy


def compute_curvature_binormal_torch(
    tangents: torch.Tensor,
    min_length: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算离散曲率副法线向量 (3D)

    公式: (κb)_i = 2(t_{i-1} × t_i) / (1 + t_{i-1}·t_i)

    返回:
        kappa_b: (N-2, 3) 曲率副法线向量
        cross_products: (N-2, 3) 叉积结果
    """
    if len(tangents) < 2:
        return torch.empty(0, 3, device=tangents.device), torch.empty(0, 3, device=tangents.device)

    cross_products = torch.cross(tangents[:-1], tangents[1:], dim=1)
    dot_products = torch.sum(tangents[:-1] * tangents[1:], dim=1)
    denominators = torch.clamp(1.0 + dot_products, min=min_length)
    kappa_b = 2.0 * cross_products / denominators[:, None]

    return kappa_b, cross_products


def compute_bending_energy_torch(
    positions: torch.Tensor,
    rest_lengths: torch.Tensor,
    EI: float,
    rest_kappa_b: Optional[torch.Tensor] = None,
    min_length: float = 1e-12
) -> torch.Tensor:
    """
    计算弯曲势能 (3D) - 使用曲率副法线方法

    公式: E_bend = Σ 0.5 * EI * ‖κb_i - κb_i^0‖² / Δs_i

    其中 Δs_i 是 Voronoi 长度 = (L_{i-1} + L_i) / 2
    """
    num_nodes = positions.shape[0]
    if EI == 0.0 or num_nodes < 3:
        return positions.new_tensor(0.0)

    _, _, tangents = compute_edge_geometry_torch(positions, min_length)
    kappa_b, _ = compute_curvature_binormal_torch(tangents, min_length)

    if len(kappa_b) == 0:
        return positions.new_tensor(0.0)

    # Voronoi 顶点长度
    rest_lengths_t = rest_lengths.to(dtype=positions.dtype, device=positions.device)
    vertex_lengths = 0.5 * (rest_lengths_t[:-1] + rest_lengths_t[1:])
    vertex_lengths = torch.clamp(vertex_lengths, min=min_length)

    # 初始曲率副法线
    if rest_kappa_b is not None:
        rest_kappa_b = rest_kappa_b.to(dtype=positions.dtype, device=positions.device)
        delta_kappa_b = kappa_b - rest_kappa_b
    else:
        delta_kappa_b = kappa_b

    # 弯曲能量
    energy = 0.5 * EI * torch.sum(torch.sum(delta_kappa_b ** 2, dim=1) / vertex_lengths)

    return energy


def compute_total_elastic_energy_torch(
    positions: torch.Tensor,
    rest_lengths: torch.Tensor,
    EA: float,
    EI: float,
    rest_kappa_b: Optional[torch.Tensor] = None,
    min_length: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算总弹性势能，返回 (拉伸能, 弯曲能, 总能)"""
    stretch_energy = compute_stretching_energy_torch(positions, rest_lengths, EA, min_length)
    bend_energy = compute_bending_energy_torch(positions, rest_lengths, EI, rest_kappa_b, min_length)
    total_energy = stretch_energy + bend_energy
    return stretch_energy, bend_energy, total_energy


# ==================== 力计算（自动微分）====================

def compute_total_internal_forces(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    EA: float,
    EI: float,
    rest_kappa_b: Optional[np.ndarray] = None,
    min_length: float = 1e-12
) -> np.ndarray:
    """
    使用 PyTorch 自动微分计算总内力 (3D)

    参数:
        positions: (N, 3) 当前节点位置
        rest_lengths: (N-1,) 初始边长
        EA: 拉伸刚度 (N)
        EI: 弯曲刚度 (N·m²)
        rest_kappa_b: (N-2, 3) 初始曲率副法线
        min_length: 最小长度，防止除零

    返回:
        forces: (N, 3) 节点力向量
    """
    if positions.shape[0] < 2:
        return np.zeros_like(positions)

    tensor_pos = torch.as_tensor(
        positions, dtype=torch.float64, device="cpu"
    ).clone().detach().requires_grad_(True)

    tensor_rest = torch.as_tensor(rest_lengths, dtype=torch.float64, device="cpu")

    tensor_rest_kappa_b = None
    if rest_kappa_b is not None and len(rest_kappa_b) > 0:
        tensor_rest_kappa_b = torch.as_tensor(rest_kappa_b, dtype=torch.float64, device="cpu")

    _, _, total_energy = compute_total_elastic_energy_torch(
        tensor_pos, tensor_rest, EA, EI, tensor_rest_kappa_b, min_length
    )

    (gradient,) = torch.autograd.grad(total_energy, tensor_pos)
    forces = -gradient.detach().cpu().numpy().astype(positions.dtype, copy=False)

    return forces


# ==================== 辅助函数 ====================

def compute_stretching_energy(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    EA: float
) -> float:
    """计算拉伸势能（NumPy接口）"""
    tensor_pos = torch.as_tensor(positions, dtype=torch.float64)
    tensor_rest = torch.as_tensor(rest_lengths, dtype=torch.float64)
    return compute_stretching_energy_torch(tensor_pos, tensor_rest, EA).item()


def compute_bending_energy(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    EI: float,
    rest_kappa_b: Optional[np.ndarray] = None
) -> float:
    """计算弯曲势能（NumPy接口）"""
    tensor_pos = torch.as_tensor(positions, dtype=torch.float64)
    tensor_rest = torch.as_tensor(rest_lengths, dtype=torch.float64)
    tensor_rest_kappa_b = None
    if rest_kappa_b is not None:
        tensor_rest_kappa_b = torch.as_tensor(rest_kappa_b, dtype=torch.float64)
    return compute_bending_energy_torch(tensor_pos, tensor_rest, EI, tensor_rest_kappa_b).item()


def compute_total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    rest_lengths: np.ndarray,
    EA: float,
    EI: float,
    node_masses: np.ndarray,
    gravity_g: float,
    external_forces: np.ndarray,
    fixed_indices: np.ndarray,
    rest_kappa_b: Optional[np.ndarray] = None
) -> dict:
    """计算系统总能量 (3D)"""
    free_indices = np.setdiff1d(np.arange(len(positions)), fixed_indices)

    stretch_energy = compute_stretching_energy(positions, rest_lengths, EA)
    bend_energy = compute_bending_energy(positions, rest_lengths, EI, rest_kappa_b)

    # 重力势能 (Z方向)
    gravity_energy = np.sum(node_masses * gravity_g * positions[:, 2])

    # 动能
    if len(free_indices) > 0:
        kinetic_energy = 0.5 * np.sum(
            node_masses[free_indices, np.newaxis] * velocities[free_indices] ** 2
        )
    else:
        kinetic_energy = 0.0

    total_energy = stretch_energy + bend_energy + gravity_energy + kinetic_energy

    return {
        'stretching_energy': stretch_energy,
        'bending_energy': bend_energy,
        'gravitational_energy': gravity_energy,
        'kinetic_energy': kinetic_energy,
        'total_energy': total_energy
    }


def calculate_critical_dt(
    node_mass: float,
    segment_length: float,
    EA: float,
    EI: float = None,
    safety_factor: float = 0.5
) -> float:
    """计算CFL稳定性限制"""
    dt_values = []

    if EA > 0:
        dt_stretch = np.sqrt(node_mass * segment_length / EA)
        dt_values.append(dt_stretch)

    if EI is not None and EI > 0:
        # 弯曲稳定性条件通常更严格
        dt_bend = segment_length ** 2 * np.sqrt(node_mass / EI) * 0.1
        dt_values.append(dt_bend)

    if not dt_values:
        return 1.0

    critical_dt = min(dt_values) * safety_factor
    return max(critical_dt, 1e-12)


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
    fixed_positions: np.ndarray = None,
    max_velocity: float = 1e6
) -> tuple:
    """半隐式欧拉积分器 (3D)"""
    if fixed_indices is None:
        fixed_indices = np.array([], dtype=int)

    free_indices = np.setdiff1d(np.arange(len(positions)), fixed_indices)

    # 总力计算
    total_forces = internal_forces + external_forces
    total_forces[:, 2] += node_masses * gravity_g

    if damping > 0:
        total_forces -= damping * velocities

    if len(free_indices) > 0:
        accelerations = total_forces[free_indices] / node_masses[free_indices, np.newaxis]
        velocities[free_indices] += accelerations * dt

        # 速度限制（防止数值爆炸）
        vel_magnitudes = np.linalg.norm(velocities[free_indices], axis=1)
        mask = vel_magnitudes > max_velocity
        if np.any(mask):
            velocities[free_indices][mask] *= max_velocity / vel_magnitudes[mask, np.newaxis]

        positions[free_indices] += velocities[free_indices] * dt

    if len(fixed_indices) > 0:
        velocities[fixed_indices] = 0.0
        if fixed_positions is not None:
            positions[fixed_indices] = fixed_positions

    return positions, velocities


def compute_bending_angles_degrees(
    positions: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """
    计算弯曲角度（度数）用于可视化 (3D)

    返回每个内部节点的弯曲角度（0 到 180 度）
    """
    num_nodes = positions.shape[0]
    angles_deg = np.zeros(num_nodes)

    if num_nodes < 3:
        return angles_deg

    edges = positions[1:] - positions[:-1]
    lengths = np.linalg.norm(edges, axis=1)
    lengths = np.maximum(lengths, eps)
    tangents = edges / lengths[:, np.newaxis]

    for i in range(1, num_nodes - 1):
        t_prev = tangents[i - 1]
        t_curr = tangents[i]

        dot = np.dot(t_prev, t_curr)
        dot = np.clip(dot, -1.0, 1.0)
        angle_rad = np.arccos(dot)

        angles_deg[i] = np.degrees(angle_rad)

    return angles_deg


def compute_curvature_binormal_magnitudes(
    positions: np.ndarray,
    min_length: float = 1e-12
) -> np.ndarray:
    """计算曲率副法线向量的模长 |κb|"""
    num_nodes = positions.shape[0]
    if num_nodes < 3:
        return np.zeros(0)

    tensor_pos = torch.as_tensor(positions, dtype=torch.float64)
    _, _, tangents = compute_edge_geometry_torch(tensor_pos, min_length)
    kappa_b, _ = compute_curvature_binormal_torch(tangents, min_length)
    return torch.norm(kappa_b, dim=1).detach().cpu().numpy()


def compute_initial_curvature_binormal(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    min_length: float = 1e-12
) -> np.ndarray:
    """计算初始构型的曲率副法线（用于非零初始弯曲）"""
    num_nodes = positions.shape[0]
    if num_nodes < 3:
        return np.zeros((0, 3))

    tensor_pos = torch.as_tensor(positions, dtype=torch.float64)
    _, _, tangents = compute_edge_geometry_torch(tensor_pos, min_length)
    kappa_b, _ = compute_curvature_binormal_torch(tangents, min_length)
    return kappa_b.detach().cpu().numpy()