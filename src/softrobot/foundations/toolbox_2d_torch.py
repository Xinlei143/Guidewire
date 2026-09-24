"""
二维离散弹性杆工具箱 - 自动微分版本 (PyTorch)

所有力通过 PyTorch 自动微分从能量求导获得
"""

import numpy as np
import torch


# ==================== 核心能量计算（PyTorch）====================

def compute_edge_geometry_torch(
    positions: torch.Tensor,
    min_length: float = 1e-12
):
    """计算边几何信息"""
    edge_vectors = positions[1:] - positions[:-1]
    edge_lengths = torch.norm(edge_vectors, dim=1)
    safe_lengths = torch.clamp(edge_lengths, min=min_length)
    tangents = edge_vectors / safe_lengths[:, None]
    return edge_vectors, edge_lengths, tangents


def compute_stretching_energy_torch(
    positions: torch.Tensor,
    rest_lengths: torch.Tensor,
    stiffness: float
) -> torch.Tensor:
    """计算拉伸势能"""
    if stiffness == 0.0:
        return positions.new_tensor(0.0)
    _, edge_lengths, _ = compute_edge_geometry_torch(positions)
    delta = edge_lengths - rest_lengths
    energy = 0.5 * stiffness * torch.sum(delta ** 2 / rest_lengths)
    return energy


def compute_discrete_curvature_torch(tangents: torch.Tensor, min_length: float = 1e-12):
    """计算离散曲率（文档公式）"""
    if len(tangents) < 2:
        return torch.tensor([]), torch.tensor([])
    cross_prod = tangents[:-1, 0] * tangents[1:, 1] - tangents[:-1, 1] * tangents[1:, 0]
    dot_prod = torch.sum(tangents[:-1] * tangents[1:], dim=1)
    denom = 1.0 + dot_prod
    denom = torch.clamp(denom, min=min_length)
    kappa = 2.0 * cross_prod / denom
    return kappa, cross_prod


def compute_bending_energy_torch(
    positions: torch.Tensor,
    rest_lengths: torch.Tensor,
    stiffness: float
) -> torch.Tensor:
    """计算弯曲势能（使用文档的离散曲率公式）"""
    num_nodes = positions.shape[0]
    if stiffness == 0.0 or num_nodes < 3:
        return positions.new_tensor(0.0)
    _, _, tangents = compute_edge_geometry_torch(positions)
    kappa, _ = compute_discrete_curvature_torch(tangents)
    if len(kappa) == 0:
        return positions.new_tensor(0.0)
    vertex_lengths = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
    vertex_lengths = torch.clamp(vertex_lengths, min=1e-12)
    energy = 0.5 * stiffness * torch.sum(kappa ** 2 / vertex_lengths)
    return energy


def compute_total_energy_torch(
    positions: torch.Tensor,
    rest_lengths: torch.Tensor,
    stretch_stiffness: float,
    bend_stiffness: float
) -> torch.Tensor:
    """计算总势能"""
    stretch_energy = compute_stretching_energy_torch(positions, rest_lengths, stretch_stiffness)
    bend_energy = compute_bending_energy_torch(positions, rest_lengths, bend_stiffness)
    return stretch_energy + bend_energy


# ==================== 力计算（自动微分）====================

def compute_total_internal_forces(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    stretch_stiffness: float,
    bend_stiffness: float
) -> np.ndarray:
    """
    使用 PyTorch 自动微分计算总内力（核心函数）

    返回:
        forces: (N, 2) 节点力向量
    """
    if positions.shape[0] < 2:
        return np.zeros_like(positions)

    tensor_pos = torch.as_tensor(
        positions, dtype=torch.float64, device="cpu"
    ).clone().detach().requires_grad_(True)

    tensor_rest = torch.as_tensor(rest_lengths, dtype=torch.float64, device="cpu")

    energy = compute_total_energy_torch(
        tensor_pos, tensor_rest, stretch_stiffness, bend_stiffness
    )

    (gradient,) = torch.autograd.grad(energy, tensor_pos)
    forces = -gradient.detach().cpu().numpy().astype(positions.dtype, copy=False)

    return forces


# ==================== 辅助函数 ====================

def compute_stretching_energy(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    stiffness: float
) -> float:
    """计算拉伸势能（NumPy接口）"""
    tensor_pos = torch.as_tensor(positions, dtype=torch.float64)
    tensor_rest = torch.as_tensor(rest_lengths, dtype=torch.float64)
    return compute_stretching_energy_torch(tensor_pos, tensor_rest, stiffness).item()


def compute_bending_energy(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    stiffness: float
) -> float:
    """计算弯曲势能（NumPy接口）"""
    tensor_pos = torch.as_tensor(positions, dtype=torch.float64)
    tensor_rest = torch.as_tensor(rest_lengths, dtype=torch.float64)
    return compute_bending_energy_torch(tensor_pos, tensor_rest, stiffness).item()


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
        node_masses[free_indices, np.newaxis] * velocities[free_indices] ** 2
    )

    return {
        'stretching_energy': stretch_energy,
        'bending_energy': bend_energy,
        'gravitational_energy': gravity_energy,
        'kinetic_energy': kinetic_energy,
        'total_energy': stretch_energy + bend_energy + gravity_energy + kinetic_energy
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
        dt_bend = segment_length ** 2 * np.sqrt(node_mass / bend_stiffness) * 0.1
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
    fixed_positions: np.ndarray = None
) -> tuple:
    """半隐式欧拉积分器"""
    if fixed_indices is None:
        fixed_indices = np.array([], dtype=int)

    free_indices = np.setdiff1d(np.arange(len(positions)), fixed_indices)

    total_forces = internal_forces + external_forces
    total_forces[:, 1] += node_masses * gravity_g

    if damping > 0:
        total_forces -= damping * velocities

    if len(free_indices) > 0:
        accelerations = total_forces[free_indices] / node_masses[free_indices, np.newaxis]
        velocities[free_indices] += accelerations * dt
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
    """计算弯曲角度（度数）用于可视化"""
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

        cross = t_prev[0] * t_curr[1] - t_prev[1] * t_curr[0]
        if cross < 0:
            angle_rad = -angle_rad

        angles_deg[i] = np.degrees(angle_rad)

    return angles_deg