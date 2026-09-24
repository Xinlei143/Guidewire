"""Three-dimensional quasi-static guidewire compression inside a straight tube.

The guidewire is an extensible discrete elastic rod.  At each prescribed
right-tip displacement increment, implicit-Euler dynamics are damped to an
equilibrium state.  The tube is a rigid cylinder about the x axis; contact is
a differentiable node-wise PI force acting radially inward.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import functools
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


# ==================== 常用参数区：通常只需修改这里 ====================
# 全部长度单位为 m、力为 N、时间为 s；内部统一使用 float64，保证 Newton 求解精度。
DTYPE = torch.float64
# 计算设备："cuda" 强制使用 NVIDIA GPU；"cpu" 强制使用 CPU；"auto" 优先 GPU、不可用时自动回退 CPU。
DEVICE_DEFAULT = "cuda"

# --- 导丝材料与离散 ---
TOTAL_LENGTH = 1.5                 # 导丝初始长度；x 轴方向从 0 延伸到该值
DIAMETER = 0.001                   # 导丝直径，影响横截面积、质量、轴向/弯曲刚度
MATERIAL_DENSITY = 7850.0          # 材料密度，kg/m^3
YOUNG_MODULUS = 200.0e9            # 杨氏模量，越大表示导丝越不容易拉伸和弯曲
DAMPING_RATIO = 1.0                # 阻尼比例；增大可更快消除振动，但过大可能使收敛变慢
NUM_NODES = 250                    # 导丝节点数；越大越精细，但计算量也显著增加

# --- 直圆管与 PI 软接触 ---
TUBE_RADIUS = 60e-3                # 直管内半径；管轴为 x 轴
SAFE_DISTANCE = 10e-3              # 距离真实管壁小于此值时进入保护层并产生接触力
KP = 3000.0                        # 接触比例增益；越大，越强力地把节点推回管内
KI = 100.0                         # 接触积分增益；用于减小长期残余穿透
MAX_PI_FORCE = 20.0                # 单个节点允许的最大接触力，避免数值力无限增大
MAX_INTEGRAL_STATE = 5.0e-2        # 单个节点 PI 积分状态上限，防止积分饱和
INTEGRAL_LEAK_RATE = 5.0           # 节点离开接触区后，积分状态衰减的速度
CONTACT_SMOOTH_LENGTH = 2e-5       # 接触开启的平滑长度；越大接触切换越平滑
DIRECTION_EPSILON = 1e-6           # 节点在管轴上时用于避免径向单位向量除零

# --- 准静态求解与平衡判据 ---
DT = 5e-3                          # 隐式 Euler 单步时间；仅用于脉冲/松弛，不代表真实加载时间
MAX_NEWTON_ITERS = 200             # 每个隐式步允许的 Newton 最大迭代次数
NEWTON_RESIDUAL_TOL = 1e-7         # 单个 Newton 步的力残差收敛阈值
NEWTON_STEP_TOL = 1e-10            # Newton 位移修正很小时的停止阈值
SETTLE_SPEED_TOL = 1e-3            # 载荷步平衡时，自由节点允许的最大速度
SETTLE_RESIDUAL_TOL = 1e-6         # 载荷步平衡时，允许的 Newton 残差
CONTACT_TOLERANCE = 5e-4           # 载荷步平衡时，保护层允许的最大几何误差
SETTLE_CONFIRM_STEPS = 5           # 上述条件连续满足这么多步，才认定该载荷步平衡
MAX_SETTLE_STEPS = 16000           # 一个载荷步最多松弛步数，超出则中止并报告

# --- 右端位移控制压缩 ---
TIP_TOTAL_DISPLACEMENT = 90e-3     # 最后一个节点总共向 -x 压缩的距离
NUM_LOAD_STEPS = 45                # 总压缩位移分成多少个载荷步
TIP_RAMP_STEPS = 50                # 每个载荷步用平滑斜坡分成多少个隐式步施加

# --- 初始脉冲：数量、位置、强度和 y/z 方向都在每次运行时随机 ---
PULSE_COUNT_MIN = 4                 # 随机脉冲节点数的最小值；至少为 4
PULSE_COUNT_MAX = 8                 # 随机脉冲节点数的最大值；设成与最小值相同可固定数量
PULSE_FORCE_MIN = 2e-2              # 每个随机节点脉冲力大小的最小值，N
PULSE_FORCE_MAX = 7e-2              # 每个随机节点脉冲力大小的最大值，N
PULSE_DURATION = 2e-2              # 脉冲持续时间；此阶段右端最后一个节点保持自由
PULSE_SEED = None                  # 随机种子；None 表示每次运行重新随机，填整数可复现实验

# --- 自动输出：同一 run_stem 的 NPZ、PNG 和汇总表可以一一对应 ---
NPZ_OUTPUT_DIR = Path("3d_npz")
PICTURE_OUTPUT_DIR = Path("3d_picture")
SUMMARY_FILENAME = "random_pulse_experiment_summary_elastic_energy.tsv"

# --- 困难载荷步的自动处理与历史保存 ---
RAMP_RETRY_SUBDIVISIONS = (1, 2, 4, 8)  # Newton 失败时按这些子步数局部重试
KI_ADAPTIVE_REDUCTION_FACTOR = 0.5       # 所有子步重试失败后，Ki 每次乘以此系数
KI_ADAPTIVE_MIN_SCALE = 0.125            # Ki 最多降到初始值的该比例
HISTORY_SAVE_INTERVAL = 167              # 每隔多少个松弛步写入一帧完整历史到 NPZ
DIAGNOSTIC_INTERVAL = 50                 # 每隔多少个 ramp/relaxation 步打印一次进度诊断（与二维一致）


@dataclass
class NewtonInfo:
    converged: bool
    iterations: int
    residual_norm: float
    step_norm: float
    line_search_failures: int


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("--device must be auto, cpu, or cuda")
    return torch.device(requested)


def tensor_to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    return value.detach().cpu().numpy() if torch.is_tensor(value) else value


def rod_properties(length: float, diameter: float, density: float) -> tuple[float, float, float, float]:
    area = math.pi * diameter**2 / 4.0
    mass = density * area * length
    ea = YOUNG_MODULUS * area
    ei = YOUNG_MODULUS * math.pi * diameter**4 / 64.0
    effective_stiffness = 3.0 * ei / length**3
    damping = 2.0 * DAMPING_RATIO * math.sqrt(effective_stiffness * mass)
    return mass, ea, ei, damping


def initialize_rod(
    total_length: float = TOTAL_LENGTH,
    num_nodes: int = NUM_NODES,
    diameter: float = DIAMETER,
    density: float = MATERIAL_DENSITY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create an initially straight 3-D rod; the pulse selects buckling."""
    if num_nodes < 4:
        raise ValueError("num_nodes must be at least 4")
    positions = np.zeros((num_nodes, 3), dtype=np.float64)
    positions[:, 0] = np.linspace(0.0, total_length, num_nodes)
    rest_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_mass, _, _, _ = rod_properties(total_length, diameter, density)
    masses = np.full(num_nodes, total_mass / (num_nodes - 1), dtype=np.float64)
    masses[[0, -1]] *= 0.5
    return positions, np.zeros_like(positions), rest_lengths, masses, np.zeros(num_nodes)


def make_initial_pulse_profile(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Randomly select free nodes and assign independent y/z-plane directions."""
    rng = np.random.default_rng(args.pulse_seed)
    # 0、1 号节点固定，最后一个节点预留给位移控制，随机脉冲只作用于中间自由节点。
    candidate_nodes = np.arange(2, args.num_nodes - 1, dtype=int)
    pulse_count = int(rng.integers(args.pulse_count_min, args.pulse_count_max + 1))
    nodes = np.sort(rng.choice(candidate_nodes, size=pulse_count, replace=False))
    # 在 y/z 平面均匀随机取角度；x 分量严格为 0，因此不会人为施加轴向脉冲。
    angles_rad = rng.uniform(0.0, 2.0 * math.pi, size=pulse_count)
    magnitudes = rng.uniform(args.pulse_force_min, args.pulse_force_max, size=pulse_count)
    unit_directions = np.column_stack((
        np.zeros(pulse_count, dtype=np.float64),
        np.cos(angles_rad),
        np.sin(angles_rad),
    ))
    profile = np.zeros((args.num_nodes, 3), dtype=np.float64)
    profile[nodes] = magnitudes[:, None] * unit_directions
    return nodes, magnitudes, angles_rad, unit_directions, profile


def local_stretch_energy(points: torch.Tensor, rest_length: torch.Tensor, ea: float) -> torch.Tensor:
    length = torch.linalg.norm(points[1] - points[0]).clamp_min(1e-12)
    return 0.5 * ea * (length - rest_length).square() / rest_length


def local_bending_energy(
    points: torch.Tensor, rest_left: torch.Tensor, rest_right: torch.Tensor, ei: float
) -> torch.Tensor:
    left = points[1] - points[0]
    right = points[2] - points[1]
    tangent_left = left / torch.linalg.norm(left).clamp_min(1e-12)
    tangent_right = right / torch.linalg.norm(right).clamp_min(1e-12)
    curvature_binormal = 2.0 * torch.cross(tangent_left, tangent_right, dim=0) / (
        1.0 + torch.dot(tangent_left, tangent_right)
    ).clamp_min(1e-12)
    voronoi = 0.5 * (rest_left + rest_right)
    return 0.5 * ei * torch.dot(curvature_binormal, curvature_binormal) / voronoi


def elastic_energy(positions: torch.Tensor, rest_lengths: torch.Tensor, ea: float, ei: float):
    edges = positions[1:] - positions[:-1]
    lengths = torch.linalg.norm(edges, dim=1).clamp_min(1e-12)
    tangents = edges / lengths[:, None]
    curvature_binormal = 2.0 * torch.cross(tangents[:-1], tangents[1:], dim=1) / (
        1.0 + torch.sum(tangents[:-1] * tangents[1:], dim=1)
    ).clamp_min(1e-12)[:, None]
    stretch = 0.5 * ea * torch.sum((lengths - rest_lengths).square() / rest_lengths)
    bend = 0.5 * ei * torch.sum(
        torch.sum(curvature_binormal.square(), dim=1) / (0.5 * (rest_lengths[:-1] + rest_lengths[1:]))
    )
    return stretch, bend, stretch + bend


def build_dof_mapping(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    free_full_indices = torch.nonzero((~mask).reshape(-1), as_tuple=False).squeeze(1)
    full_to_free = torch.full((mask.numel(),), -1, dtype=torch.long, device=mask.device)
    full_to_free[free_full_indices] = torch.arange(len(free_full_indices), device=mask.device)
    return full_to_free, free_full_indices


@dataclass(frozen=True)
class ScatterPlan:
    """Precomputed free-coordinate scatter indices for one family of local terms.

    Rows/cols are the free-coordinate (rows>=0, cols>=0) pairs of every local
    block; valid selects the usable entries.  grad_dst/grad_valid are the
    flattened 1-D form used by the gradient index_add_.
    """
    rows: torch.Tensor        # (M,k,k) long
    cols: torch.Tensor        # (M,k,k) long
    valid: torch.Tensor       # (M,k,k) bool, (rows>=0)&(cols>=0)
    grad_dst: torch.Tensor    # (L,) long, destination of every valid gradient entry
    grad_valid: torch.Tensor  # (M,k) bool


@dataclass(frozen=True)
class AssemblyPattern:
    """Everything needed to scatter local DER/contact terms for one topology."""
    device: torch.device
    num_nodes: int
    count: int                # free free-coordinate count
    full_to_free: torch.Tensor
    free_indices: torch.Tensor
    stretch: ScatterPlan      # k=6,  M=num_nodes-1
    bend: ScatterPlan         # k=9,  M=num_nodes-2
    contact: ScatterPlan      # k=2,  M=num_nodes


@functools.lru_cache(maxsize=16)
def _build_assembly_pattern(
    device_str: str,
    num_nodes: int,
    first_prescribed: bool,
    tip_prescribed: bool,
) -> AssemblyPattern:
    device = torch.device(device_str)
    mask = torch.zeros((num_nodes, 3), dtype=torch.bool, device=device)
    if first_prescribed:
        mask[:2] = True
    if tip_prescribed:
        mask[-1] = True
    full_to_free, free_indices = build_dof_mapping(mask)

    def make_plan(width: int, element_count: int, offset: int = 0) -> ScatterPlan:
        indices = 3 * torch.arange(element_count, device=device)[:, None] + offset + torch.arange(width, device=device)
        local = full_to_free[indices]                          # (M,k)
        rows = local[:, :, None].expand(-1, -1, width)
        cols = local[:, None, :].expand(-1, width, -1)
        valid = (rows >= 0) & (cols >= 0)
        flat_valid = local >= 0
        return ScatterPlan(rows, cols, valid, local[flat_valid], flat_valid)

    return AssemblyPattern(
        device=device,
        num_nodes=num_nodes,
        count=len(free_indices),
        full_to_free=full_to_free,
        free_indices=free_indices,
        stretch=make_plan(6, num_nodes - 1),    # nodes (n, n+1)
        bend=make_plan(9, num_nodes - 2),       # nodes (n, n+1, n+2)
        contact=make_plan(2, num_nodes, offset=1),  # y/z dofs of node n
    )


def get_assembly_pattern(mask: torch.Tensor) -> AssemblyPattern:
    """Cached pattern for the canonical prescribed masks built by
    make_prescribed_state (left two nodes fixed, optionally the tip); the
    all-free mask used by self_test is also supported.  Other masks are
    unsupported and will silently map to the nearest canonical topology."""
    return _build_assembly_pattern(
        str(mask.device), mask.shape[0],
        bool(mask[:2].all()), bool(mask[-1].all()),
    )


def _scatter_terms(
    gradients: torch.Tensor,
    hessians: torch.Tensor | None,
    plan: ScatterPlan,
    gradient: torch.Tensor,
    hessian: torch.Tensor | None,
) -> None:
    gradient.index_add_(0, plan.grad_dst, gradients[plan.grad_valid])
    if hessians is None or hessian is None:
        return
    hessian.index_put_(
        (plan.rows[plan.valid], plan.cols[plan.valid]), hessians[plan.valid], accumulate=True
    )


def stretch_grad_hessian_analytic(
    positions: torch.Tensor,
    rest_lengths: torch.Tensor,
    ea: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form stretch gradient/Hessian of E = 1/2 (EA/l0) (l-l0)^2.

    Uses the same clamp_min(1e-12) on the edge length as local_stretch_energy,
    so it stays finite where an element is crushed to zero length.
    """
    d = positions[1:] - positions[:-1]
    l = torch.linalg.norm(d, dim=1).clamp_min(1e-12)
    k = ea / rest_lengths
    coeff = 1.0 - rest_lengths / l
    g_d = k[:, None] * coeff[:, None] * d
    grad = torch.cat((-g_d, g_d), dim=1)                      # (E,6)
    eye = torch.eye(3, dtype=d.dtype, device=d.device)
    K = k[:, None, None] * (
        coeff[:, None, None] * eye
        + (rest_lengths / l**3)[:, None, None] * (d[:, :, None] * d[:, None, :])
    )
    H = torch.empty(len(d), 6, 6, dtype=d.dtype, device=d.device)
    H[:, :3, :3] = K
    H[:, :3, 3:] = -K
    H[:, 3:, :3] = -K
    H[:, 3:, 3:] = K
    return grad, H


def assemble_elastic_batched(
    positions: torch.Tensor,
    rest_lengths: torch.Tensor,
    pattern: AssemblyPattern,
    ea: float,
    ei: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble 3-D local DER terms into the free-coordinate gradient/Hessian.

    Stretch is analytic; bending derives gradient and Hessian in a single
    forward pass (jacrev of the gradient with has_aux, where the aux output is
    the gradient itself).  Scatter indices come from the cached pattern.
    """
    gradient = torch.zeros(pattern.count, dtype=positions.dtype, device=positions.device)
    hessian = torch.zeros((pattern.count, pattern.count), dtype=positions.dtype, device=positions.device)

    stretch_grad, stretch_hess = stretch_grad_hessian_analytic(positions, rest_lengths, ea)
    _scatter_terms(stretch_grad, stretch_hess, pattern.stretch, gradient, hessian)

    def bend_energy(flat: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return local_bending_energy(flat.reshape(3, 3), left, right, ei)

    bend_grad_fn = torch.func.grad(bend_energy, argnums=0)

    def bend_grad_with_aux(flat: torch.Tensor, left: torch.Tensor, right: torch.Tensor):
        g = bend_grad_fn(flat, left, right)
        return g, g                                       # output=g (differentiated), aux=g

    bend_dofs = torch.cat((positions[:-2], positions[1:-1], positions[2:]), dim=1)
    bend_hess, bend_grad = torch.func.vmap(
        torch.func.jacrev(bend_grad_with_aux, argnums=0, has_aux=True)
    )(bend_dofs, rest_lengths[:-1], rest_lengths[1:])
    _scatter_terms(bend_grad, bend_hess, pattern.bend, gradient, hessian)
    return gradient, hessian


def _contact_single(
    yz: torch.Tensor,
    old_integral: torch.Tensor,
    previous_error: torch.Tensor,
    dt: float,
    radius: float,
    safe_distance: float,
    kp: float,
    ki: float,
    max_force: float,
    leak_rate: float,
    max_integral: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # 节点到管轴的径向距离；圆管接触只与 y/z 平面内的位置有关。
    radial = torch.linalg.norm(yz)
    distance = radius - radial
    z = radial - (radius - safe_distance)
    beta = 1.0 / CONTACT_SMOOTH_LENGTH
    error = F.softplus(z, beta=beta)
    weight = torch.sigmoid(beta * z)
    candidate = torch.clamp(
        weight * (old_integral + error * dt)
        + (1.0 - weight) * old_integral * math.exp(-leak_rate * dt),
        0.0,
        max_integral,
    )
    unsaturated = kp * error + ki * candidate
    stop_growth = ((ki > 0.0) & (unsaturated >= max_force) & (candidate > old_integral)) | (
        (ki > 0.0) & (error < previous_error) & (ki * old_integral > kp * error) & (candidate > old_integral)
    )
    new_integral = torch.where(stop_growth, old_integral, candidate)
    magnitude = torch.clamp(kp * error + ki * new_integral, 0.0, max_force)
    # 负号表示接触力始终指向管轴，将靠近/越过管壁的节点推回管内。
    force_yz = -yz * magnitude / torch.sqrt(torch.dot(yz, yz) + DIRECTION_EPSILON**2)
    return force_yz, new_integral, error, weight, z > 0.0, distance


def contact_quantities(
    positions: torch.Tensor,
    integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    args: argparse.Namespace,
):
    values = torch.func.vmap(_contact_single, in_dims=(0, 0, 0, None, None, None, None, None, None, None, None))(
        positions[:, 1:], integrals, previous_errors, args.dt_active, args.tube_radius,
        args.safe_distance, args.Kp, args.ki_active, args.max_pi_force,
        args.integral_leak_rate, args.max_integral_state,
    )
    force_yz, new_integrals, errors, weights, active, distances = values
    forces = torch.zeros_like(positions)
    forces[:, 1:] = force_yz
    return forces, new_integrals, active, weights, errors, distances


def assemble_contact_batched(
    positions: torch.Tensor,
    integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    pattern: AssemblyPattern,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the contact force on free coordinates and its per-node y/z
    Jacobian blocks (num_nodes, 2, 2), computed in a single batched pass."""
    def force_yz_aux(yz: torch.Tensor, old_i: torch.Tensor, previous: torch.Tensor):
        f = _contact_single(
            yz, old_i, previous, args.dt_active, args.tube_radius, args.safe_distance,
            args.Kp, args.ki_active, args.max_pi_force, args.integral_leak_rate,
            args.max_integral_state,
        )[0]
        return f, f                                       # output=f (differentiated), aux=f

    yz = positions[:, 1:]
    blocks, values = torch.func.vmap(
        torch.func.jacrev(force_yz_aux, argnums=0, has_aux=True)
    )(yz, integrals, previous_errors)
    force_full = torch.zeros(positions.numel(), dtype=positions.dtype, device=positions.device)
    force_full[1::3] = values[:, 0]
    force_full[2::3] = values[:, 1]
    return force_full[pattern.free_indices], blocks


def make_prescribed_state(
    initial: torch.Tensor,
    tip_constrained: bool,
    tip_displacement: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros_like(initial, dtype=torch.bool)
    mask[:2] = True  # 左端前两个节点全自由度固定，相当于夹持端
    values = initial.clone()
    if tip_constrained:
        mask[-1] = True  # 脉冲结束后，最后一个节点成为位移控制端
        values[-1, 0] -= tip_displacement  # 从初始约 1 m 位置沿 -x 向内压缩
    return mask, values


def unpack(q: torch.Tensor, free_mask: torch.Tensor, prescribed: torch.Tensor) -> torch.Tensor:
    result = prescribed.clone()
    result[free_mask] = q
    return result


def compute_protection_error(positions: torch.Tensor, radius: float, safe_distance: float) -> torch.Tensor:
    return torch.clamp(torch.linalg.norm(positions[:, 1:], dim=1) - (radius - safe_distance), min=0.0)


def residual_and_jacobian(
    q: torch.Tensor,
    old_positions: torch.Tensor,
    old_velocities: torch.Tensor,
    rest_lengths: torch.Tensor,
    masses: torch.Tensor,
    integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    pulse_forces: torch.Tensor,
    prescribed_mask: torch.Tensor,
    prescribed: torch.Tensor,
    args: argparse.Namespace,
    with_jacobian: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    free_mask = ~prescribed_mask
    positions = unpack(q, free_mask, prescribed)
    if with_jacobian:
        pattern = get_assembly_pattern(prescribed_mask)
        gradient, elastic_hessian = assemble_elastic_batched(positions, rest_lengths, pattern, args.EA, args.EI)
        contact_force, contact_blocks = assemble_contact_batched(
            positions, integrals, previous_errors, pattern, args
        )
    else:
        # Residual-only path: one global autograd call replaces the batched
        # assembly.  q_grad is a fresh leaf so no gradient leaks into the
        # caller's q; contact runs on the detached positions.
        q_grad = q.detach().requires_grad_(True)
        _, _, energy = elastic_energy(unpack(q_grad, free_mask, prescribed), rest_lengths, args.EA, args.EI)
        gradient = torch.autograd.grad(energy, q_grad)[0]
        all_contact, *_ = contact_quantities(positions, integrals, previous_errors, args)
        contact_force = all_contact[free_mask]
        contact_blocks = None
    q_old = old_positions[free_mask]
    v_old = old_velocities[free_mask]
    free_masses = masses[:, None].expand_as(old_positions)[free_mask]
    residual = (
        free_masses * (q - q_old - args.dt_active * v_old) / args.dt_active**2
        + gradient + args.damping * (q - q_old) / args.dt_active
        - contact_force - pulse_forces[free_mask]
    )
    if not with_jacobian:
        return residual, None, positions
    jacobian = elastic_hessian
    jacobian.diagonal().add_(free_masses / args.dt_active**2 + args.damping / args.dt_active)
    # Scatter the negated contact blocks in place, after the mass/damping
    # diagonal add, so the result matches the old diag-add-then-sub() ordering.
    contact_plan = pattern.contact
    jacobian.index_put_(
        (contact_plan.rows[contact_plan.valid], contact_plan.cols[contact_plan.valid]),
        (-contact_blocks)[contact_plan.valid],
        accumulate=True,
    )
    return residual, jacobian, positions


def implicit_euler_step(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    rest_lengths: torch.Tensor,
    masses: torch.Tensor,
    integrals: torch.Tensor,
    initial_positions: torch.Tensor,
    pulse_forces: torch.Tensor,
    tip_constrained: bool,
    tip_displacement: float,
    dt_step: float,
    args: argparse.Namespace,
):
    args.dt_active = dt_step
    prescribed_mask, prescribed = make_prescribed_state(initial_positions, tip_constrained, tip_displacement)
    free_mask = ~prescribed_mask
    previous_radius = torch.linalg.norm(positions[:, 1:], dim=1)
    previous_errors = F.softplus(
        previous_radius - (args.tube_radius - args.safe_distance), beta=1.0 / CONTACT_SMOOTH_LENGTH
    )
    predictor = positions + args.dt_active * velocities
    predictor[prescribed_mask] = prescribed[prescribed_mask]
    q = predictor[free_mask].detach()
    current_residual, _, _ = residual_and_jacobian(
        q, positions, velocities, rest_lengths, masses, integrals, previous_errors, pulse_forces,
        prescribed_mask, prescribed, args, False
    )
    residual_norm = float(torch.linalg.norm(current_residual))
    line_search_failures = 0
    step_norm = 0.0
    converged = residual_norm < args.residual_tol
    iteration = 0
    for iteration in range(1, args.max_newton_iters + 1):
        if converged:
            break
        residual, jacobian, _ = residual_and_jacobian(
            q, positions, velocities, rest_lengths, masses, integrals, previous_errors, pulse_forces,
            prescribed_mask, prescribed, args, True
        )
        try:
            delta = torch.linalg.solve(jacobian, -residual)
        except RuntimeError:
            delta = torch.linalg.lstsq(jacobian, -residual).solution
        step_norm = float(torch.linalg.norm(delta))
        if step_norm < args.step_tol:
            converged = True
            break
        base_norm = float(torch.linalg.norm(residual))
        alpha = 1.0
        accepted = False
        for _ in range(12):
            trial = (q + alpha * delta).detach()
            trial_residual, _, _ = residual_and_jacobian(
                trial, positions, velocities, rest_lengths, masses, integrals, previous_errors, pulse_forces,
                prescribed_mask, prescribed, args, False
            )
            trial_norm = float(torch.linalg.norm(trial_residual))
            if np.isfinite(trial_norm) and trial_norm < base_norm:
                q, residual_norm, accepted = trial, trial_norm, True
                break
            alpha *= 0.5
        if not accepted:
            line_search_failures += 1
            q = (q + 0.05 * delta).detach()
            trial_residual, _, _ = residual_and_jacobian(
                q, positions, velocities, rest_lengths, masses, integrals, previous_errors, pulse_forces,
                prescribed_mask, prescribed, args, False
            )
            residual_norm = float(torch.linalg.norm(trial_residual))
        converged = residual_norm < args.residual_tol

    new_positions = unpack(q, free_mask, prescribed).detach()
    new_velocities = (new_positions - positions) / args.dt_active
    new_velocities[:2] = 0.0
    if tip_constrained:
        new_velocities[-1] = (prescribed[-1] - positions[-1]) / args.dt_active
    with torch.no_grad():
        force, new_integrals, active, weights, errors, distances = contact_quantities(
            new_positions, integrals, previous_errors, args
        )
    return (
        new_positions, new_velocities, new_integrals, force, active, weights, errors, distances,
        NewtonInfo(converged, iteration, residual_norm, step_norm, line_search_failures), prescribed_mask,
    )


def max_free_speed(velocities: torch.Tensor, prescribed_mask: torch.Tensor) -> float:
    free_nodes = torch.any(~prescribed_mask, dim=1)
    return float(torch.linalg.vector_norm(velocities[free_nodes], dim=1).max()) if bool(torch.any(free_nodes)) else 0.0


def summarize_contact_segments(active: np.ndarray, forces: np.ndarray, positions: np.ndarray) -> dict[str, np.ndarray]:
    nodes = np.flatnonzero(active)
    if not len(nodes):
        return {
            "start": np.empty(0, dtype=int), "end": np.empty(0, dtype=int),
            "count": np.empty(0, dtype=int), "resultant": np.empty((0, 3)),
            "magnitude_sum": np.empty(0), "mean_angle_rad": np.empty(0),
        }
    groups = np.split(nodes, np.flatnonzero(np.diff(nodes) > 1) + 1)
    return {
        "start": np.asarray([g[0] for g in groups]),
        "end": np.asarray([g[-1] for g in groups]),
        "count": np.asarray([len(g) for g in groups]),
        "resultant": np.asarray([forces[g].sum(axis=0) for g in groups]),
        "magnitude_sum": np.asarray([np.linalg.norm(forces[g], axis=1).sum() for g in groups]),
        "mean_angle_rad": np.asarray([np.arctan2(positions[g, 2].mean(), positions[g, 1].mean()) for g in groups]),
    }


def run_simulation(args: argparse.Namespace) -> dict[str, np.ndarray]:
    positions, velocities, rest_lengths, masses, integrals = initialize_rod(
        args.total_length, args.num_nodes, args.diameter, args.density
    )
    device = args.torch_device
    positions = torch.as_tensor(positions, dtype=DTYPE, device=device)
    velocities = torch.as_tensor(velocities, dtype=DTYPE, device=device)
    rest_lengths = torch.as_tensor(rest_lengths, dtype=DTYPE, device=device)
    masses = torch.as_tensor(masses, dtype=DTYPE, device=device)
    integrals = torch.as_tensor(integrals, dtype=DTYPE, device=device)
    initial_positions = positions.detach().clone()
    args.dt_active = args.dt
    args.ki_active = args.Ki
    initial = tensor_to_numpy(positions).copy()
    (
        pulse_nodes,
        pulse_magnitudes,
        pulse_angles_rad,
        pulse_unit_directions,
        pulse_profile,
    ) = make_initial_pulse_profile(args)
    pulse_forces = torch.as_tensor(pulse_profile, dtype=DTYPE, device=device)
    zero_forces = torch.zeros_like(positions)
    history: dict[str, list] = {key: [] for key in (
        "time", "load_step", "tip_displacement", "positions", "velocities", "pi_forces",
        "pi_integrals", "contact_active", "contact_weight", "radial_distance",
        "max_protection_error", "max_pi_force", "newton_iterations", "newton_residual",
        "elastic_stretch_energy_j", "elastic_bend_energy_j", "elastic_total_energy_j",
        "newton_converged", "tip_position_constraint_active",
    )}
    start = time.perf_counter()
    time_value = 0.0
    abort_reason: str | None = None
    completed_steps = 0
    last_force = torch.zeros_like(positions)
    last_active = torch.zeros(args.num_nodes, dtype=torch.bool, device=device)
    last_info = NewtonInfo(True, 0, 0.0, 0.0, 0)
    total_newton_iterations = 0
    effective_ki_scale = 1.0
    effective_ki = args.Ki
    load_step_relaxation_steps: list[int] = []
    contact_event_time = np.nan

    def record(
        load_step: int,
        displacement: float,
        force: torch.Tensor,
        active: torch.Tensor,
        weights: torch.Tensor,
        info: NewtonInfo,
        tip_constrained: bool,
    ) -> None:
        p_np = tensor_to_numpy(positions).copy()
        history["time"].append(time_value)
        history["load_step"].append(load_step)
        history["tip_displacement"].append(displacement)
        history["positions"].append(p_np)
        history["velocities"].append(tensor_to_numpy(velocities).copy())
        history["pi_forces"].append(tensor_to_numpy(force).copy())
        history["pi_integrals"].append(tensor_to_numpy(integrals).copy())
        history["contact_active"].append(tensor_to_numpy(active).copy())
        history["contact_weight"].append(tensor_to_numpy(weights).copy())
        history["radial_distance"].append(np.linalg.norm(p_np[:, 1:], axis=1))
        protection = compute_protection_error(positions, args.tube_radius, args.safe_distance)
        history["max_protection_error"].append(float(torch.max(protection)))
        history["max_pi_force"].append(float(torch.linalg.vector_norm(force, dim=1).max()))
        history["newton_iterations"].append(info.iterations)
        history["newton_residual"].append(info.residual_norm)
        history["newton_converged"].append(info.converged)
        history["tip_position_constraint_active"].append(tip_constrained)
        stretch_energy, bend_energy, elastic_total = elastic_energy(positions, rest_lengths, args.EA, args.EI)
        history["elastic_stretch_energy_j"].append(float(stretch_energy))
        history["elastic_bend_energy_j"].append(float(bend_energy))
        history["elastic_total_energy_j"].append(float(elastic_total))

    def advance(
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        applied_forces: torch.Tensor,
        tip_constrained: bool,
        displacement: float,
        dt_step: float,
    ):
        p, v, i = state
        return implicit_euler_step(
            p, v, rest_lengths, masses, i, initial_positions, applied_forces,
            tip_constrained, displacement, dt_step, args,
        )

    def retry_step(
        checkpoint: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        applied_forces: torch.Tensor,
        tip_constrained: bool,
        start_displacement: float,
        end_displacement: float,
        phase_label: str,
    ):
        """沿用二维的 1/2/4/8 子步重试；失败时逐级降低 Ki。"""
        nonlocal total_newton_iterations, effective_ki, effective_ki_scale
        last_failed: NewtonInfo | None = None
        while True:
            for subdivision_count in RAMP_RETRY_SUBDIVISIONS:
                state = checkpoint
                latest = None
                failed = False
                for subdivision_index in range(1, subdivision_count + 1):
                    fraction = subdivision_index / subdivision_count
                    displacement = start_displacement + (end_displacement - start_displacement) * fraction
                    latest = advance(
                        state, applied_forces, tip_constrained, displacement,
                        args.dt / subdivision_count,
                    )
                    total_newton_iterations += latest[8].iterations
                    if not latest[8].converged and latest[8].residual_norm >= args.residual_tol:
                        failed = True
                        last_failed = latest[8]
                        break
                    state = latest[:3]
                if not failed:
                    return True, latest, subdivision_count, None
            next_scale = effective_ki_scale * KI_ADAPTIVE_REDUCTION_FACTOR
            if args.Ki <= 0.0 or next_scale < KI_ADAPTIVE_MIN_SCALE - 1.0e-15:
                return False, None, 0, last_failed
            old_ki = effective_ki
            effective_ki_scale = next_scale
            effective_ki = args.Ki * effective_ki_scale
            args.ki_active = effective_ki
            print(
                f"KI ADAPT: {phase_label}; local {RAMP_RETRY_SUBDIVISIONS} retries failed. "
                f"Ki_eff reduced from {old_ki:g} to {effective_ki:g} N/(m*s) "
                f"(scale={effective_ki_scale:g}); restoring state and retrying.",
                flush=True,
            )

    record(0, 0.0, last_force, last_active, torch.zeros_like(integrals), last_info, False)
    print("=" * 76)
    print("3-D DER + implicit Euler/Newton + node-wise PI soft contact (quasi-static displacement control)")
    print("=" * 76)
    print(
        f"nodes={args.num_nodes}, length={args.total_length:g} m, diameter={args.diameter:g} m, "
        f"mass={args.mtotal:.6g} kg"
    )
    print(
        f"mtotal={args.mtotal:.6e} kg, EA={args.EA:g} N, EI={args.EI:g} N*m^2, "
        f"damping={args.damping:.6e} N*s/m, relaxation_dt={args.dt:g} s"
    )
    print(
        f"execution_device={device}, dtype={str(DTYPE).replace('torch.', '')}, "
        f"device_request={args.device}"
    )
    print(
        f"tube_radius={args.tube_radius:g} m, safe_distance={args.safe_distance:g} m, "
        f"Kp={args.Kp:g}, Ki={args.Ki:g}, cap={args.max_pi_force:g} N"
    )
    print(
        f"tip_total_displacement={args.tip_total_displacement:g} m, load_steps={args.num_load_steps}, "
        f"increment={args.tip_displacement_step:g} m, ramp_steps={args.tip_ramp_steps}, "
        f"contact_tolerance={args.contact_tolerance:g} m"
    )
    print(
        f"initial random yz-plane pulses: count={len(pulse_nodes)}, nodes={pulse_nodes.tolist()}, "
        f"magnitudes={np.array2string(pulse_magnitudes, precision=4)}, seed={args.pulse_seed}, "
        f"duration={args.pulse_duration:.6e} s; right tip is free during pulse",
        flush=True,
    )

    # The initially straight rod receives only this short, user-configurable
    # force pulse.  Compression begins after the right tip has been free.
    while time_value < args.pulse_duration - 1.0e-15:
        dt_step = min(args.dt, args.pulse_duration - time_value)
        (
            positions, velocities, integrals, last_force, last_active, weights, _errors, _distances,
            last_info, prescribed_mask,
        ) = advance((positions, velocities, integrals), pulse_forces, False, 0.0, dt_step)
        total_newton_iterations += last_info.iterations
        time_value += dt_step
        if not last_info.converged and last_info.residual_norm >= args.residual_tol:
            abort_reason = f"Newton failure during initial pulse: residual={last_info.residual_norm:.3e}"
            print(f"EARLY ABORT: {abort_reason}")
            break
        record(0, 0.0, last_force, last_active, weights, last_info, False)

    for load_step in range(1, args.num_load_steps + 1):
        if abort_reason:
            break
        target = load_step * args.tip_displacement_step
        previous = (load_step - 1) * args.tip_displacement_step
        print(
            f"\n[LOAD {load_step:02d}/{args.num_load_steps:02d}] "
            f"target tip displacement = {target * 1e3:.3f} mm",
            flush=True,
        )
        max_speed_seen = 0.0
        max_contact_error_seen = 0.0
        for ramp in range(1, args.tip_ramp_steps + 1):
            f = ramp / args.tip_ramp_steps
            smooth = f * f * (3.0 - 2.0 * f)
            displacement = previous + (target - previous) * smooth
            previous_f = (ramp - 1) / args.tip_ramp_steps
            previous_smooth = previous_f * previous_f * (3.0 - 2.0 * previous_f)
            ramp_start = previous + (target - previous) * previous_smooth
            accepted, result, subdivisions, failed_info = retry_step(
                (positions, velocities, integrals), zero_forces, True, ramp_start, displacement,
                phase_label=f"load step {load_step}/{args.num_load_steps}, ramp step {ramp}/{args.tip_ramp_steps}",
            )
            if not accepted:
                abort_reason = (
                    f"Newton failure during ramp at load step {load_step}, ramp step {ramp}: "
                    f"residual={failed_info.residual_norm:.3e}" if failed_info else
                    f"Newton failure during ramp at load step {load_step}, ramp step {ramp}"
                )
                print(f"EARLY ABORT: {abort_reason}")
                break
            (
                positions, velocities, integrals, last_force, last_active, weights, _errors, _distances,
                last_info, prescribed_mask,
            ) = result
            time_value += args.dt
            if subdivisions > 1:
                print(
                    f"RAMP RETRY: load step {load_step:2d}/{args.num_load_steps}, "
                    f"ramp step {ramp}/{args.tip_ramp_steps} accepted with "
                    f"{subdivisions} local substeps, Ki_eff={effective_ki:g}",
                    flush=True,
                )
            current_max_speed = max_free_speed(velocities, prescribed_mask)
            max_speed_seen = max(max_speed_seen, current_max_speed)
            ramp_protection = compute_protection_error(positions, args.tube_radius, args.safe_distance)
            max_contact_error_seen = max(max_contact_error_seen, float(torch.max(ramp_protection)))
            if ramp % DIAGNOSTIC_INTERVAL == 0 or ramp == args.tip_ramp_steps:
                print(
                    f"[RAMP  {load_step:02d}/{args.num_load_steps:02d} | "
                    f"{ramp:4d}/{args.tip_ramp_steps} | "
                    f"{100.0 * ramp / args.tip_ramp_steps:5.1f}%] "
                    f"tip={displacement * 1e3:.4f} mm | "
                    f"v={current_max_speed:.2e} | Newton={last_info.iterations} | "
                    f"R={last_info.residual_norm:.2e} | "
                    f"contacts={int(torch.count_nonzero(last_active))}",
                    flush=True,
                )
            if ramp % args.history_save_interval == 0 or ramp == args.tip_ramp_steps:
                record(load_step, displacement, last_force, last_active, weights, last_info, True)
        if abort_reason:
            break
        print(
            f"[RAMP COMPLETE] increment={args.tip_displacement_step * 1e3:.3f} mm "
            f"over {args.tip_ramp_steps} steps; holding tip for relaxation",
            flush=True,
        )
        stable = 0
        settled = False
        max_speed_seen = 0.0
        max_contact_error_seen = 0.0
        max_pi_force_seen = 0.0
        for relaxation in range(1, args.max_settle_steps + 1):
            accepted, result, subdivisions, failed_info = retry_step(
                (positions, velocities, integrals), zero_forces, True, target, target,
                phase_label=f"load step {load_step}/{args.num_load_steps}, relaxation step {relaxation}",
            )
            if not accepted:
                abort_reason = (
                    f"Newton failure during relaxation at load step {load_step}: "
                    f"residual={failed_info.residual_norm:.3e}" if failed_info else
                    f"Newton failure during relaxation at load step {load_step}"
                )
                print(f"EARLY ABORT: {abort_reason}")
                break
            (
                positions, velocities, integrals, last_force, last_active, weights, _errors, _distances,
                last_info, prescribed_mask,
            ) = result
            time_value += args.dt
            if subdivisions > 1:
                print(
                    f"RELAXATION RETRY: load step {load_step:2d}/{args.num_load_steps}, "
                    f"relaxation step {relaxation} accepted with "
                    f"{subdivisions} local substeps, Ki_eff={effective_ki:g}",
                    flush=True,
                )
            protection = compute_protection_error(positions, args.tube_radius, args.safe_distance)
            max_speed = max_free_speed(velocities, prescribed_mask)
            max_contact_error = float(torch.max(protection))
            max_pi_force = float(torch.linalg.vector_norm(last_force, dim=1).max())
            max_speed_seen = max(max_speed_seen, max_speed)
            max_contact_error_seen = max(max_contact_error_seen, max_contact_error)
            max_pi_force_seen = max(max_pi_force_seen, max_pi_force)
            equilibrium = (
                last_info.converged and max_speed < args.settle_speed_tol
                and last_info.residual_norm < args.settle_residual_tol
                and max_contact_error < args.contact_tolerance
            )
            stable = stable + 1 if equilibrium else 0
            if relaxation % args.history_save_interval == 0 or stable >= args.settle_confirm_steps:
                record(load_step, target, last_force, last_active, weights, last_info, True)
            if relaxation % DIAGNOSTIC_INTERVAL == 0 or stable >= args.settle_confirm_steps:
                contact_count = int(torch.count_nonzero(last_active))
                contact_summary = f"contact={max_contact_error * 1e3:.4f} mm/{contact_count} nodes"
                pi_summary = f" | F_PI={max_pi_force:.2e} N" if contact_count else ""
                retry_summary = (
                    f" | retries={last_info.line_search_failures}"
                    if last_info.line_search_failures else ""
                )
                solver_state = "ok" if last_info.converged else "not-converged"
                print(
                    f"[RELAX {load_step:02d}/{args.num_load_steps:02d} | {relaxation:5d}] "
                    f"v={max_speed:.2e} m/s | R={last_info.residual_norm:.2e} N | "
                    f"{contact_summary} | stable={stable} | Newton={solver_state}"
                    f"{pi_summary}{retry_summary}",
                    flush=True,
                )
            if stable >= args.settle_confirm_steps:
                settled = True
                completed_steps += 1
                load_step_relaxation_steps.append(relaxation)
                print(
                    f"[LOAD COMPLETE {load_step:02d}/{args.num_load_steps:02d}] "
                    f"tip={target * 1e3:.3f} mm | relax={relaxation} steps | "
                    f"v={max_speed_seen:.2e} m/s | contact={max_contact_error_seen * 1e3:.4f} mm",
                    flush=True,
                )
                break
        if abort_reason:
            break
        if not settled:
            abort_reason = f"load step {load_step} did not settle within {args.max_settle_steps} steps"
            print(f"EARLY ABORT: {abort_reason}")
            break

    history_np = {key: np.asarray(value) for key, value in history.items()}
    final_positions = tensor_to_numpy(positions)
    final_forces = tensor_to_numpy(last_force)
    segments = summarize_contact_segments(tensor_to_numpy(last_active), final_forces, final_positions)
    stretch, bend, total = elastic_energy(positions, rest_lengths, args.EA, args.EI)
    elapsed = time.perf_counter() - start
    print(
        f"completed_load_steps={completed_steps}/{args.num_load_steps}, "
        f"wall_time={elapsed:.2f} s, total_newton_iterations={total_newton_iterations}, "
        f"final_Ki_eff={effective_ki:g} N/(m*s), Ki_scale={effective_ki_scale:g}"
    )
    run_stem, npz_path, picture_path, summary_path = resolve_output_paths(
        args, pulse_nodes, pulse_magnitudes
    )
    history_np.update({
        "run_stem": np.asarray(run_stem),
        "npz_path": np.asarray(str(npz_path)),
        "picture_path": np.asarray(str(picture_path)),
        "initial_positions": initial,
        "initial_pulse_count": np.asarray(len(pulse_nodes), dtype=int),
        "initial_pulse_seed": np.asarray(-1 if args.pulse_seed is None else args.pulse_seed, dtype=int),
        "initial_pulse_nodes": pulse_nodes,
        "initial_pulse_x_m": initial[pulse_nodes, 0],
        "initial_pulse_magnitudes_n": pulse_magnitudes,
        "initial_pulse_angles_rad": pulse_angles_rad,
        "initial_pulse_unit_directions": pulse_unit_directions,
        "initial_pulse_force_vectors_n": pulse_profile[pulse_nodes],
        "initial_pulse_duration_s": np.asarray(args.pulse_duration),
        "completed_load_steps": np.asarray(completed_steps),
        "load_step_implicit_steps": np.asarray(load_step_relaxation_steps, dtype=int),
        "total_newton_iterations": np.asarray(total_newton_iterations),
        "effective_ki_final": np.asarray(effective_ki),
        "abort_reason": np.asarray(abort_reason or ""),
        "wall_clock_seconds": np.asarray(elapsed),
        "final_elastic_stretch_energy_j": np.asarray(float(stretch)),
        "final_elastic_bend_energy_j": np.asarray(float(bend)),
        "final_elastic_energy_j": np.asarray(float(total)),
        "final_contact_segment_start_nodes": segments["start"],
        "final_contact_segment_end_nodes": segments["end"],
        "final_contact_segment_node_counts": segments["count"],
        "final_contact_segment_resultants": segments["resultant"],
        "final_contact_segment_force_magnitude_sums": segments["magnitude_sum"],
        "final_contact_segment_mean_angle_rad": segments["mean_angle_rad"],
        "final_contact_total_resultant": final_forces.sum(axis=0),
    })
    state_label = "final state" if abort_reason is None else "stopped state"
    total_contact_magnitude = float(np.sum(np.linalg.norm(final_forces, axis=1)))
    print("FINAL CONTACT-SEGMENT FORCE SUMMARY (node-wise PI force sums)")
    if len(segments["start"]):
        for segment_number, (start_node, end_node, count, resultant, magnitude_sum) in enumerate(
            zip(segments["start"], segments["end"], segments["count"], segments["resultant"], segments["magnitude_sum"]),
            start=1,
        ):
            print(
                f"  segment {segment_number}: nodes {start_node}-{end_node} "
                f"({count} nodes), sum|F_PI|={magnitude_sum:.6e} N, "
                f"resultant=({resultant[0]:.6e}, {resultant[1]:.6e}, {resultant[2]:.6e}) N, "
                f"|resultant|={np.linalg.norm(resultant):.6e} N"
            )
    else:
        print("  no protection-layer contact nodes at the final stable state")
    print(
        f"  total sum|F_PI|={total_contact_magnitude:.6e} N, "
        f"net resultant=({final_forces.sum(axis=0)[0]:.6e}, "
        f"{final_forces.sum(axis=0)[1]:.6e}, {final_forces.sum(axis=0)[2]:.6e}) N"
    )
    print(
        f"FINAL ENERGY ({state_label}, J): elastic_stretching={float(stretch):.8e} | "
        f"bending={float(bend):.8e} | total={float(total):.8e}"
    )
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **history_np)
    # 无论是否弹出窗口，都生成报告用 PNG；--no-show 时使用无界面后端保存。
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    final_figure = create_final_result_figure(history_np, args)
    picture_path.parent.mkdir(parents=True, exist_ok=True)
    final_figure.savefig(picture_path, dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(final_figure)
    append_experiment_summary(summary_path, run_stem, npz_path, picture_path, history_np)
    print(f"history saved: {npz_path}")
    print(f"final report picture saved: {picture_path}")
    print(f"experiment summary appended: {summary_path}")
    if abort_reason:
        print(f"simulation stopped: {abort_reason}")
    else:
        print("all load steps settled")
    if args.show:
        visualize_results(history_np, args)
    return history_np


def resolve_output_paths(
    args: argparse.Namespace,
    pulse_nodes: np.ndarray,
    pulse_magnitudes: np.ndarray,
) -> tuple[str, Path, Path, Path]:
    """Create one shared run name so NPZ, PNG and summary rows are traceable."""
    if args.output_prefix is not None:
        requested = Path(args.output_prefix)
        run_stem = requested.with_suffix("").name
        npz_path = requested.with_suffix(".npz")
    else:
        # 时间戳避免 seed=None 的随机试验覆盖上一次结果；节点/幅值仍完整保存在 NPZ 和汇总表内。
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        nodes_tag = "-".join(map(str, pulse_nodes.tolist()))
        run_stem = (
            f"quasi_static_3d_L{args.total_length:g}m_disp{args.tip_total_displacement * 1e3:g}mm"
            f"_N{args.num_nodes}_pulse{len(pulse_nodes)}_nodes{nodes_tag}"
            f"_F{pulse_magnitudes.min():.3g}-{pulse_magnitudes.max():.3g}_{timestamp}"
        ).replace(".", "p")
        npz_path = NPZ_OUTPUT_DIR / f"{run_stem}.npz"
    picture_path = PICTURE_OUTPUT_DIR / f"{run_stem}_final.png"
    summary_path = PICTURE_OUTPUT_DIR / SUMMARY_FILENAME
    return run_stem, npz_path, picture_path, summary_path


def create_final_result_figure(history: dict[str, np.ndarray], args: argparse.Namespace):
    """Build the report-ready final PNG: deformation, random loads and energies."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    positions = history["positions"]
    x = np.linspace(0.0, args.total_length, 48)
    theta = np.linspace(0.0, 2.0 * np.pi, 36)
    X, THETA = np.meshgrid(x, theta)
    Y, Z = args.tube_radius * np.cos(THETA), args.tube_radius * np.sin(THETA)
    fig = plt.figure(figsize=(16, 10), dpi=180)

    ax = fig.add_axes([0.02, 0.37, 0.96, 0.60], projection="3d")
    ax.plot_surface(X, Y, Z, color="lightsteelblue", alpha=0.18, linewidth=0)
    ax.plot(*positions[0].T, "k--", lw=1.2, label="initial")
    ax.plot(*positions[-1].T, color="tab:blue", lw=1.9, label="final deformation")
    active = history["contact_active"][-1]
    if np.any(active):
        ax.scatter(*positions[-1, active].T, c="magenta", s=18, label="final contact")
    ax.set(
        xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
        title="Final guidewire deformation in the straight cylindrical tube",
    )
    ax.set_xlim(0.0, args.total_length)
    ax.set_ylim(-1.15 * args.tube_radius, 1.15 * args.tube_radius)
    ax.set_zlim(-1.15 * args.tube_radius, 1.15 * args.tube_radius)
    ax.set_box_aspect((args.total_length, 6 * args.tube_radius, 6 * args.tube_radius))
    ax.view_init(elev=18, azim=-64)
    ax.legend(loc="upper right")

    load_ax = fig.add_axes([0.03, 0.05, 0.57, 0.23])
    load_ax.axis("off")
    nodes = np.asarray(history["initial_pulse_nodes"], dtype=int)
    x_positions = np.asarray(history["initial_pulse_x_m"], dtype=float)
    magnitudes = np.asarray(history["initial_pulse_magnitudes_n"], dtype=float)
    force_vectors = np.asarray(history["initial_pulse_force_vectors_n"], dtype=float)
    rows = [
        [str(node), f"{x_value:.4f}", f"{magnitude:.4e}", f"{force[1]:.4e}", f"{force[2]:.4e}"]
        for node, x_value, magnitude, force in zip(nodes, x_positions, magnitudes, force_vectors)
    ]
    table = load_ax.table(
        cellText=rows,
        colLabels=["node", "x (m)", "|F| (N)", "Fy (N)", "Fz (N)"],
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.25)
    load_ax.set_title("Random initial pulse loads (all directions are in the y/z plane)", fontsize=10, pad=5)

    energy_ax = fig.add_axes([0.66, 0.07, 0.30, 0.21])
    final_stretch = float(history["elastic_stretch_energy_j"][-1])
    final_bend = float(history["elastic_bend_energy_j"][-1])
    final_elastic = float(history["elastic_total_energy_j"][-1])
    labels = ["stretch", "bend", "total"]
    values = [final_stretch, final_bend, final_elastic]
    bars = energy_ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"])
    for bar, value in zip(bars, values):
        energy_ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2e}", ha="center", va="bottom", fontsize=7)
    energy_ax.set_ylabel("energy (J)")
    energy_ax.set_title("Final system energy", fontsize=10)
    energy_ax.grid(axis="y", alpha=0.25)
    return fig


def append_experiment_summary(
    summary_path: Path,
    run_stem: str,
    npz_path: Path,
    picture_path: Path,
    history: dict[str, np.ndarray],
) -> None:
    """Append one compact report row; six simulations become a ready comparison table."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()
    nodes = ";".join(map(str, np.asarray(history["initial_pulse_nodes"], dtype=int)))
    magnitudes = ";".join(f"{value:.8e}" for value in np.asarray(history["initial_pulse_magnitudes_n"], dtype=float))
    vectors = ";".join(
        f"({vector[0]:.6e},{vector[1]:.6e},{vector[2]:.6e})"
        for vector in np.asarray(history["initial_pulse_force_vectors_n"], dtype=float)
    )
    fields = [
        run_stem,
        str(history["abort_reason"].item() or "success"),
        str(npz_path),
        str(picture_path),
        str(int(history["initial_pulse_count"])),
        nodes,
        magnitudes,
        vectors,
        f"{float(history['elastic_stretch_energy_j'][-1]):.10e}",
        f"{float(history['elastic_bend_energy_j'][-1]):.10e}",
        f"{float(history['elastic_total_energy_j'][-1]):.10e}",
    ]
    with summary_path.open("a", encoding="utf-8", newline="") as file:
        if write_header:
            file.write(
                "run_stem\tstatus\tnpz_path\tpicture_path\tpulse_count\tpulse_nodes\t"
                "pulse_magnitudes_n\tpulse_force_vectors_n\tstretch_energy_j\tbend_energy_j\t"
                "elastic_total_energy_j\n"
            )
        file.write("\t".join(fields) + "\n")


def visualize_results(history: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    positions = history["positions"]
    x = np.linspace(0.0, args.total_length, 40)
    theta = np.linspace(0.0, 2.0 * np.pi, 32)
    X, THETA = np.meshgrid(x, theta)
    Y, Z = args.tube_radius * np.cos(THETA), args.tube_radius * np.sin(THETA)
    if not args.skip_animation:
        # 直管沿 x 方向很长，采用横向窗口可去掉上下无效白边，
        # 同时保持导丝在窗口中的实际显示宽度。
        fig_animation = plt.figure(figsize=(13, 5.4), dpi=120)
        # Use almost the full popup for the 3-D scene.  The y/z display aspect
        # is deliberately enlarged only for viewing: the simulation data and
        # axis labels remain in metres and are not rescaled numerically.
        ax_animation = fig_animation.add_axes([0.015, 0.07, 0.97, 0.87], projection="3d")
        ax_animation.plot_surface(X, Y, Z, color="lightsteelblue", alpha=0.18, linewidth=0)
        rod_line, = ax_animation.plot([], [], [], color="tab:blue", lw=1.8, label="guidewire")
        contact_scatter = ax_animation.scatter([], [], [], c="magenta", s=18, label="contact")
        ax_animation.set(
            xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
            title="3-D guidewire compression animation",
        )
        ax_animation.set_xlim(0.0, args.total_length)
        ax_animation.set_ylim(-1.15 * args.tube_radius, 1.15 * args.tube_radius)
        ax_animation.set_zlim(-1.15 * args.tube_radius, 1.15 * args.tube_radius)
        ax_animation.set_box_aspect((args.total_length, 6 * args.tube_radius, 6 * args.tube_radius))
        ax_animation.view_init(elev=18, azim=-64)
        ax_animation.legend(loc="upper right")
        frame_indices = np.arange(0, len(positions), max(1, args.animation_stride))
        if frame_indices[-1] != len(positions) - 1:
            frame_indices = np.append(frame_indices, len(positions) - 1)

        def update(frame_number: int):
            frame = int(frame_indices[frame_number])
            current = positions[frame]
            rod_line.set_data(current[:, 0], current[:, 1])
            rod_line.set_3d_properties(current[:, 2])
            active_nodes = history["contact_active"][frame]
            contact = current[active_nodes]
            contact_scatter._offsets3d = (contact[:, 0], contact[:, 1], contact[:, 2])
            ax_animation.set_title(
                f"3-D guidewire compression animation, t={history['time'][frame]:.3f} s"
            )
            return rod_line, contact_scatter

        animation = FuncAnimation(
            fig_animation, update, frames=len(frame_indices),
            interval=1000.0 / args.animation_fps, blit=False,
        )
        # Keep a reference alive until the animation window is closed.
        fig_animation._guidewire_animation = animation
        plt.show()
        plt.close(fig_animation)

    fig = create_final_result_figure(history, args)
    plt.show()
    plt.close(fig)


def self_test(args: argparse.Namespace) -> None:
    positions, _, rest, _, integrals = initialize_rod()
    assert positions.shape == (NUM_NODES, 3) and rest.shape == (NUM_NODES - 1,) and np.allclose(positions[:, 2], 0.0)
    p = torch.as_tensor(positions[:8], dtype=DTYPE)
    rest_t = torch.as_tensor(rest[:7], dtype=DTYPE)
    _, _, energy = elastic_energy(p, rest_t, args.EA, args.EI)
    assert bool(torch.isfinite(energy))
    args.dt_active = args.dt
    samples = torch.tensor([[0.051, 0.0], [0.0, 0.051], [0.051 / math.sqrt(2), 0.051 / math.sqrt(2)], [0.0, 0.0]], dtype=DTYPE)
    previous = torch.zeros(4, dtype=DTYPE)
    force, *_ = torch.func.vmap(_contact_single, in_dims=(0, 0, 0, None, None, None, None, None, None, None, None))(
        samples, torch.zeros(4, dtype=DTYPE), previous, args.dt, args.tube_radius, args.safe_distance,
        args.Kp, args.Ki, args.max_pi_force, args.integral_leak_rate, args.max_integral_state,
    )
    assert torch.all(torch.sum(force[:3] * samples[:3], dim=1) < 0.0)
    assert torch.allclose(force[-1], torch.zeros(2, dtype=DTYPE))
    probe = torch.zeros((4, 3), dtype=DTYPE)
    probe[:, 1:] = samples
    pattern = get_assembly_pattern(torch.zeros_like(probe, dtype=torch.bool))
    force_free, contact_blocks = assemble_contact_batched(
        probe, torch.zeros(4, dtype=DTYPE), previous, pattern, args,
    )
    assert torch.isfinite(force_free).all() and torch.isfinite(contact_blocks).all()
    # The 45-degree node has coupled y/z derivatives; this guards against a
    # diagonal-only planar-contact Jacobian accidentally returning here.
    assert abs(float(contact_blocks[2, 0, 1])) > 0.0
    args.pulse_seed = 20260806
    nodes, magnitudes, _angles, directions, profile = make_initial_pulse_profile(args)
    assert args.pulse_count_min <= len(nodes) <= args.pulse_count_max
    assert len(np.unique(nodes)) == len(nodes)
    assert np.all((2 <= nodes) & (nodes < args.num_nodes - 1))
    assert np.allclose(directions[:, 0], 0.0) and np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.all((args.pulse_force_min <= magnitudes) & (magnitudes <= args.pulse_force_max))
    assert np.allclose(profile[nodes], magnitudes[:, None] * directions)
    print(f"self-test passed: {NUM_NODES}-node geometry, 3-D energy, radial contact, and random yz pulse directions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-length", type=float, default=TOTAL_LENGTH)
    parser.add_argument("--diameter", type=float, default=DIAMETER)
    parser.add_argument("--density", type=float, default=MATERIAL_DENSITY)
    parser.add_argument("--num-nodes", type=int, default=NUM_NODES)
    parser.add_argument("--pulse-count-min", type=int, default=PULSE_COUNT_MIN)
    parser.add_argument("--pulse-count-max", type=int, default=PULSE_COUNT_MAX)
    parser.add_argument("--pulse-force-min", type=float, default=PULSE_FORCE_MIN)
    parser.add_argument("--pulse-force-max", type=float, default=PULSE_FORCE_MAX)
    parser.add_argument(
        "--pulse-duration", type=float, default=PULSE_DURATION,
        help="duration of the initial pulse in seconds; right tip remains free during this phase",
    )
    parser.add_argument(
        "--pulse-seed", type=int, default=PULSE_SEED,
        help="random seed for selected nodes and yz directions; omit for a new random realization each run",
    )
    parser.add_argument("--tube-radius", type=float, default=TUBE_RADIUS)
    parser.add_argument("--safe-distance", type=float, default=SAFE_DISTANCE)
    parser.add_argument("--Kp", type=float, default=KP)
    parser.add_argument("--Ki", type=float, default=KI)
    parser.add_argument("--max-pi-force", type=float, default=MAX_PI_FORCE)
    parser.add_argument("--max-integral-state", type=float, default=MAX_INTEGRAL_STATE)
    parser.add_argument("--integral-leak-rate", type=float, default=INTEGRAL_LEAK_RATE)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--max-newton-iters", type=int, default=MAX_NEWTON_ITERS)
    parser.add_argument("--residual-tol", type=float, default=NEWTON_RESIDUAL_TOL)
    parser.add_argument("--step-tol", type=float, default=NEWTON_STEP_TOL)
    parser.add_argument("--settle-speed-tol", type=float, default=SETTLE_SPEED_TOL)
    parser.add_argument("--settle-residual-tol", type=float, default=SETTLE_RESIDUAL_TOL)
    parser.add_argument("--contact-tolerance", type=float, default=CONTACT_TOLERANCE)
    parser.add_argument("--settle-confirm-steps", type=int, default=SETTLE_CONFIRM_STEPS)
    parser.add_argument("--max-settle-steps", type=int, default=MAX_SETTLE_STEPS)
    parser.add_argument("--tip-total-displacement", type=float, default=TIP_TOTAL_DISPLACEMENT)
    parser.add_argument("--num-load-steps", type=int, default=NUM_LOAD_STEPS)
    parser.add_argument("--tip-ramp-steps", type=int, default=TIP_RAMP_STEPS)
    parser.add_argument("--history-save-interval", type=int, default=HISTORY_SAVE_INTERVAL)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default=DEVICE_DEFAULT,
        help="calculation device; the default comes from DEVICE_DEFAULT in the parameter section",
    )
    parser.add_argument(
        "--output-prefix", default=None,
        help="NPZ path prefix; omitted uses a parameter-encoded name under 3d_npz/",
    )
    parser.add_argument("--no-show", dest="show", action="store_false", default=True)
    parser.add_argument("--skip-animation", action="store_true", help="show only the final result figure")
    parser.add_argument("--animation-stride", type=int, default=1)
    parser.add_argument("--animation-fps", type=float, default=30.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.num_nodes < 4 or args.total_length <= 0 or args.diameter <= 0 or args.density <= 0:
        parser.error("length, diameter, density must be positive and num-nodes must be at least 4")
    if not 0.0 < args.safe_distance < args.tube_radius:
        parser.error("safe-distance must lie strictly between zero and tube-radius")
    for name in ("dt", "max_newton_iters", "residual_tol", "step_tol", "settle_speed_tol", "settle_residual_tol", "settle_confirm_steps", "max_settle_steps", "tip_total_displacement", "num_load_steps", "tip_ramp_steps", "history_save_interval"):
        if getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    args.mtotal, args.EA, args.EI, args.damping = rod_properties(args.total_length, args.diameter, args.density)
    args.tip_displacement_step = args.tip_total_displacement / args.num_load_steps
    if args.animation_stride <= 0 or args.animation_fps <= 0:
        parser.error("animation-stride and animation-fps must be positive")
    if args.pulse_duration < 0 or args.pulse_force_min < 0 or args.pulse_force_max < 0:
        parser.error("pulse-duration and pulse-force bounds must be non-negative")
    if args.pulse_count_min < 4:
        parser.error("pulse-count-min must be at least 4")
    if args.pulse_count_max < args.pulse_count_min:
        parser.error("pulse-count-max must be greater than or equal to pulse-count-min")
    if args.pulse_count_max > args.num_nodes - 3:
        parser.error("pulse-count-max exceeds the number of available free interior nodes")
    if args.pulse_force_max < args.pulse_force_min:
        parser.error("pulse-force-max must be greater than or equal to pulse-force-min")
    args.ki_active = args.Ki
    args.torch_device = resolve_device(args.device)
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test:
        self_test(parsed)
    else:
        run_simulation(parsed)
