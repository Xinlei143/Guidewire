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

from ..common.cli import cli_arguments


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
DT = 0.005                          # 隐式 Euler 单步时间；仅用于脉冲/松弛，不代表真实加载时间
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
NUM_LOAD_STEPS = 10                # 总压缩位移分成多少个载荷步
TIP_RAMP_STEPS = 50                # fixed 模式的斜坡步数；adaptive 模式中也是允许的最小增量尺度
RAMP_MODE = "adaptive"             # adaptive=速度控制自适应加载；fixed=原程序固定平滑 ramp
ADAPTIVE_RELAX_STEPS = 8           # 每个自适应位移增量后，最多保持末端短暂松弛的隐式步数
ADAPTIVE_SPEED_GROW_FACTOR = 2.0   # v_max 小于该倍数×平衡速度阈值时，才允许放大下一增量
ADAPTIVE_SPEED_SHRINK_FACTOR = 10.0 # v_max 大于该倍数×平衡速度阈值时，回滚并缩小本次增量

# --- 初始脉冲：数量、位置、强度和 y/z 方向都在每次运行时随机 ---
PULSE_COUNT_MIN = 4                 # 随机脉冲节点数的最小值；至少为 4
PULSE_COUNT_MAX = 8                 # 随机脉冲节点数的最大值；设成与最小值相同可固定数量
PULSE_FORCE_MIN = 2e-2              # 每个随机节点脉冲力大小的最小值，N
PULSE_FORCE_MAX = 7e-2              # 每个随机节点脉冲力大小的最大值，N
PULSE_DURATION = 2e-2              # 脉冲持续时间；此阶段右端最后一个节点保持自由
PULSE_SEED = None                  # 随机种子；None 表示每次运行重新随机，填整数可复现实验
# 显式指定模式：两者同时提供时跳过随机（节点与力完全固定）；都不提供则全部随机。
PULSE_NODES = None                 # 逗号分隔的节点号字符串，如 "5,10,20"；None 表示随机选节点
PULSE_FYFZ = None                  # 逗号分隔的每节点 (fy,fz) 分量，如 "0.05,0.0,0.02,0.04"；None 表示随机力

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


def sync_device(value: torch.Tensor | torch.device) -> None:
    """Synchronize CUDA only at an externally meaningful timing boundary."""
    device = value.device if torch.is_tensor(value) else torch.device(value)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def parse_float_list(text: str) -> list[float]:
    """Parse a comma/whitespace separated string of floats, e.g. "5,10,20" or "0.05,0.0,0.02,0.04"."""
    return [float(token) for token in text.replace(",", " ").split()]


def make_explicit_pulse_profile(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed perturbation: node indices plus per-node (Fy, Fz) force components.

    The force direction is whatever the supplied (Fy, Fz) pair implies; the
    x component stays zero.  Node range and pair-count are validated here so a
    typo fails fast instead of silently perturbing the wrong node.
    """
    nodes = np.asarray(parse_float_list(args.pulse_nodes), dtype=int)
    force_values = np.asarray(parse_float_list(args.pulse_fyfz), dtype=float)
    if nodes.size == 0:
        raise ValueError("--pulse-nodes must list at least one node")
    if nodes.min() < 2 or nodes.max() > args.num_nodes - 2:
        raise ValueError(
            f"pulse nodes must lie in the free range [2, {args.num_nodes - 2}]; "
            f"got {nodes.tolist()}"
        )
    if np.unique(nodes).size != nodes.size:
        raise ValueError(f"pulse nodes must be unique; got {nodes.tolist()}")
    if force_values.size != 2 * nodes.size:
        raise ValueError(
            f"--pulse-fyfz must supply 2 values (fy,fz) per node: got {force_values.size} "
            f"values for {nodes.size} nodes"
        )
    force_yz = force_values.reshape(-1, 2)  # (count, 2)
    magnitudes = np.linalg.norm(force_yz, axis=1)
    unit_yz = np.divide(
        force_yz, magnitudes[:, None],
        out=np.zeros_like(force_yz), where=magnitudes[:, None] > 0.0,
    )
    angles_rad = np.arctan2(force_yz[:, 1], force_yz[:, 0])
    unit_directions = np.column_stack((np.zeros(nodes.size, dtype=np.float64), unit_yz))
    profile = np.zeros((args.num_nodes, 3), dtype=np.float64)
    profile[nodes, 1:] = force_yz
    return nodes, magnitudes, angles_rad, unit_directions, profile


def make_initial_pulse_profile(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select pulse nodes and forces: either fully explicit or fully random.

    If --pulse-nodes and --pulse-fyfz are both provided, the pulse is exactly
    those nodes with those (Fy, Fz) components.  If neither is provided, fall
    back to the historical fully-random selection (count, nodes, magnitude and
    y/z direction all drawn from the seed).  Providing only one of the two is
    an error so a half-configured run never silently falls back to random.
    """
    if getattr(args, "pulse_nodes", None) is not None or getattr(args, "pulse_fyfz", None) is not None:
        if getattr(args, "pulse_nodes", None) is None or getattr(args, "pulse_fyfz", None) is None:
            raise ValueError(
                "--pulse-nodes and --pulse-fyfz must be provided together for a fixed "
                "perturbation; omit both to keep fully-random pulses"
            )
        return make_explicit_pulse_profile(args)
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


def assemble_contact_analytic(
    positions: torch.Tensor,
    integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    pattern: AssemblyPattern,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contact force and radial 2x2 Jacobian blocks without autograd.

    This is algebraically equivalent to ``assemble_contact_batched``.  The
    PI state is treated as a function of the current radial distance while
    ``integrals`` and ``previous_errors`` remain Newton-step constants.
    Clamp and anti-windup branches explicitly zero their local derivative,
    matching the derivative used by PyTorch's clamp/where in the reference
    implementation away from switching-point ambiguity.
    """
    yz = positions[:, 1:]
    radial = torch.linalg.vector_norm(yz, dim=1)
    beta = 1.0 / CONTACT_SMOOTH_LENGTH
    z = radial - (args.tube_radius - args.safe_distance)
    error = F.softplus(z, beta=beta)
    error_prime = torch.sigmoid(beta * z)
    weight_prime = beta * error_prime * (1.0 - error_prime)

    decay = math.exp(-args.integral_leak_rate * args.dt_active)
    accumulated = integrals + error * args.dt_active
    leaked = integrals * decay
    raw_candidate = leaked + error_prime * (accumulated - leaked)
    candidate = torch.clamp(raw_candidate, 0.0, args.max_integral_state)
    candidate_prime = torch.where(
        (raw_candidate >= 0.0) & (raw_candidate <= args.max_integral_state),
        weight_prime * (accumulated - leaked) + args.dt_active * error_prime.square(),
        torch.zeros_like(raw_candidate),
    )

    unsaturated = args.Kp * error + args.ki_active * candidate
    stop_growth = (
        ((args.ki_active > 0.0) & (unsaturated >= args.max_pi_force) & (candidate > integrals))
        | (
            (args.ki_active > 0.0)
            & (error < previous_errors)
            & (args.ki_active * integrals > args.Kp * error)
            & (candidate > integrals)
        )
    )
    new_integrals = torch.where(stop_growth, integrals, candidate)
    new_integral_prime = torch.where(stop_growth, torch.zeros_like(candidate_prime), candidate_prime)
    raw_magnitude = args.Kp * error + args.ki_active * new_integrals
    magnitude = torch.clamp(raw_magnitude, 0.0, args.max_pi_force)
    magnitude_prime = torch.where(
        (raw_magnitude >= 0.0) & (raw_magnitude <= args.max_pi_force),
        args.Kp * error_prime + args.ki_active * new_integral_prime,
        torch.zeros_like(raw_magnitude),
    )

    direction_scale = torch.sqrt(radial.square() + DIRECTION_EPSILON**2)
    force_yz = -yz * (magnitude / direction_scale)[:, None]
    inverse_radial = torch.where(
        radial > 0.0,
        1.0 / radial.clamp_min(1.0e-30),
        torch.zeros_like(radial),
    )
    c1 = -magnitude / direction_scale
    c2 = magnitude / direction_scale.pow(3) - magnitude_prime * inverse_radial / direction_scale
    eye = torch.eye(2, dtype=positions.dtype, device=positions.device)
    blocks = c1[:, None, None] * eye + c2[:, None, None] * (yz[:, :, None] * yz[:, None, :])

    force_full = torch.zeros(positions.numel(), dtype=positions.dtype, device=positions.device)
    force_full[1::3] = force_yz[:, 0]
    force_full[2::3] = force_yz[:, 1]
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


@dataclass(frozen=True)
class BlockBandScatterPlan:
    """Precomputed local-Hessian entries for one block band."""

    element: torch.Tensor
    row: torch.Tensor
    col: torch.Tensor
    destination: torch.Tensor


@dataclass(frozen=True)
class BlockLocalAssemblyPlan:
    """Precomputed gradient and Hessian scatter locations for one DER family."""

    grad_element: torch.Tensor
    grad_local: torch.Tensor
    grad_destination: torch.Tensor
    diagonal: BlockBandScatterPlan
    upper1: BlockBandScatterPlan
    lower1: BlockBandScatterPlan
    upper2: BlockBandScatterPlan
    lower2: BlockBandScatterPlan


@dataclass(frozen=True)
class BlockAssemblyPlan:
    """Topology-only plan shared by v3 and the mesh wrapper."""

    device: torch.device
    num_nodes: int
    free_start: int
    free_end: int
    block_count: int
    free_indices: torch.Tensor
    stretch: BlockLocalAssemblyPlan
    bend: BlockLocalAssemblyPlan


def _make_block_local_assembly_plan(
    device: torch.device,
    element_count: int,
    local_nodes: int,
    first_node: int,
    free_start: int,
    free_end: int,
) -> BlockLocalAssemblyPlan:
    gradient_entries: list[tuple[int, int, int]] = []
    bands: dict[int, list[tuple[int, int, int, int]]] = {offset: [] for offset in (-2, -1, 0, 1, 2)}
    for element in range(element_count):
        for local_row in range(local_nodes):
            row_node = first_node + element + local_row
            if free_start <= row_node < free_end:
                gradient_entries.append((element, local_row, row_node - free_start))
            for local_col in range(local_nodes):
                col_node = first_node + element + local_col
                if not (free_start <= row_node < free_end and free_start <= col_node < free_end):
                    continue
                offset = local_col - local_row
                if offset not in bands:
                    continue
                row = row_node - free_start
                col = col_node - free_start
                bands[offset].append((element, local_row, local_col, row if offset >= 0 else col))

    def tensor_column(column: int) -> torch.Tensor:
        return torch.as_tensor(
            [entry[column] for entry in gradient_entries], dtype=torch.long, device=device
        )

    def make_band(offset: int) -> BlockBandScatterPlan:
        entries = bands[offset]
        return BlockBandScatterPlan(
            element=torch.as_tensor([entry[0] for entry in entries], dtype=torch.long, device=device),
            row=torch.as_tensor([entry[1] for entry in entries], dtype=torch.long, device=device),
            col=torch.as_tensor([entry[2] for entry in entries], dtype=torch.long, device=device),
            destination=torch.as_tensor([entry[3] for entry in entries], dtype=torch.long, device=device),
        )

    return BlockLocalAssemblyPlan(
        grad_element=tensor_column(0),
        grad_local=tensor_column(1),
        grad_destination=tensor_column(2),
        diagonal=make_band(0),
        upper1=make_band(1),
        lower1=make_band(-1),
        upper2=make_band(2),
        lower2=make_band(-2),
    )


@functools.lru_cache(maxsize=8)
def get_block_assembly_plan(
    device_str: str, num_nodes: int, tip_constrained: bool,
) -> BlockAssemblyPlan:
    """Build and cache the two fixed DER topologies used by one simulation."""
    device = torch.device(device_str)
    free_start = 2
    free_end = num_nodes - int(tip_constrained)
    mask = torch.zeros((num_nodes, 3), dtype=torch.bool, device=device)
    mask[:free_start] = True
    if tip_constrained:
        mask[-1] = True
    free_indices = torch.nonzero((~mask).reshape(-1), as_tuple=False).squeeze(1)
    return BlockAssemblyPlan(
        device=device,
        num_nodes=num_nodes,
        free_start=free_start,
        free_end=free_end,
        block_count=free_end - free_start,
        free_indices=free_indices,
        stretch=_make_block_local_assembly_plan(
            device, num_nodes - 1, 2, 0, free_start, free_end
        ),
        bend=_make_block_local_assembly_plan(
            device, num_nodes - 2, 3, 0, free_start, free_end
        ),
    )


def _fallback_block_assembly_plan(
    positions: torch.Tensor, prescribed_mask: torch.Tensor,
) -> BlockAssemblyPlan:
    """Compatibility fallback for direct callers outside an implicit step."""
    tip_constrained = bool(prescribed_mask[-1].all())
    return get_block_assembly_plan(str(positions.device), len(positions), tip_constrained)


def assemble_elastic_block_penta_v3(
    positions: torch.Tensor, rest_lengths: torch.Tensor, prescribed_mask: torch.Tensor,
    ea: float, ei: float, block_plan: BlockAssemblyPlan | None = None,
    profile_args: argparse.Namespace | None = None,
) -> tuple[torch.Tensor, "BlockPentaSystem", int, int]:
    """Assemble DER local terms using a topology-only block scatter plan."""
    block_plan = block_plan or _fallback_block_assembly_plan(positions, prescribed_mask)
    gradient = torch.zeros((block_plan.block_count, 3), dtype=positions.dtype, device=positions.device)
    system = BlockPentaSystem.zeros(block_plan.block_count, positions.dtype, positions.device)

    def scatter(local_grad: torch.Tensor, local_hess: torch.Tensor, plan: BlockLocalAssemblyPlan) -> None:
        local_nodes = local_grad.shape[1]
        local_blocks = local_hess.reshape(len(local_grad), local_nodes, 3, local_nodes, 3)
        gradient.index_add_(
            0,
            plan.grad_destination,
            local_grad[plan.grad_element, plan.grad_local],
        )
        for name in ("diagonal", "upper1", "lower1", "upper2", "lower2"):
            band_plan = getattr(plan, name)
            getattr(system, name).index_add_(
                0,
                band_plan.destination,
                local_blocks[band_plan.element, band_plan.row, :, band_plan.col, :],
            )

    derivative_started = time.perf_counter()
    stretch_grad, stretch_hess = stretch_grad_hessian_analytic(positions, rest_lengths, ea)
    def bend_energy(flat: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return local_bending_energy(flat.reshape(3, 3), left, right, ei)
    bend_grad_fn = torch.func.grad(bend_energy, argnums=0)
    def bend_grad_aux(flat: torch.Tensor, left: torch.Tensor, right: torch.Tensor):
        value = bend_grad_fn(flat, left, right)
        return value, value
    bend_dofs = torch.cat((positions[:-2], positions[1:-1], positions[2:]), dim=1)
    bend_hess, bend_grad = torch.func.vmap(torch.func.jacrev(bend_grad_aux, argnums=0, has_aux=True))(
        bend_dofs, rest_lengths[:-1], rest_lengths[1:]
    )
    derivative_elapsed = time.perf_counter() - derivative_started
    assembly_started = time.perf_counter()
    scatter(stretch_grad.reshape(-1, 2, 3), stretch_hess, block_plan.stretch)
    scatter(bend_grad.reshape(-1, 3, 3), bend_hess, block_plan.bend)
    assembly_elapsed = time.perf_counter() - assembly_started
    if profile_args is not None:
        profile = getattr(profile_args, "performance_profile", None)
        if profile is not None:
            profile["elastic_gradient_hessian_s"] = profile.get("elastic_gradient_hessian_s", 0.0) + derivative_elapsed
            profile["block_banded_assembly_s"] = profile.get("block_banded_assembly_s", 0.0) + assembly_elapsed
    return gradient.reshape(-1), system, block_plan.free_start, block_plan.free_end


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
    block_plan: BlockAssemblyPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    free_mask = ~prescribed_mask
    positions = unpack(q, free_mask, prescribed)
    if with_jacobian:
        block_plan = block_plan or _fallback_block_assembly_plan(positions, prescribed_mask)
        gradient, system, free_start, free_end = assemble_elastic_block_penta_v3(
            positions, rest_lengths, prescribed_mask, args.EA, args.EI, block_plan, args
        )
        if getattr(args, "compute_kernels", "optimized") == "reference":
            contact_force, contact_blocks = assemble_contact_batched(
                positions, integrals, previous_errors, block_plan, args
            )
        else:
            contact_force, contact_blocks = assemble_contact_analytic(
                positions, integrals, previous_errors, block_plan, args
            )
    else:
        # Residual-only trials retain the established global elastic gradient:
        # on this v3 bending kernel it is faster than local first-order vmap.
        q_grad = q.detach().requires_grad_(True)
        _, _, energy = elastic_energy(
            unpack(q_grad, free_mask, prescribed), rest_lengths, args.EA, args.EI
        )
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
    diagonal_add = (free_masses / args.dt_active**2 + args.damping / args.dt_active).reshape(-1, 3)
    system.diagonal.diagonal(dim1=1, dim2=2).add_(diagonal_add)
    system.diagonal[:, 1:, 1:].add_(-contact_blocks[free_start:free_end])
    return residual, system, positions


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
    step_record = getattr(args, "_current_dt_record", None)
    breakdown = step_record["breakdown"] if step_record is not None else None

    def add_time(key: str, started: float) -> None:
        if breakdown is not None:
            breakdown[key] = breakdown.get(key, 0.0) + (time.perf_counter() - started)

    prescribed_mask, prescribed = make_prescribed_state(initial_positions, tip_constrained, tip_displacement)
    free_mask = ~prescribed_mask
    block_plan = get_block_assembly_plan(str(positions.device), len(positions), tip_constrained)
    previous_radius = torch.linalg.norm(positions[:, 1:], dim=1)
    previous_errors = F.softplus(
        previous_radius - (args.tube_radius - args.safe_distance), beta=1.0 / CONTACT_SMOOTH_LENGTH
    )
    predictor = positions + args.dt_active * velocities
    predictor[prescribed_mask] = prescribed[prescribed_mask]
    q = predictor[free_mask].detach()
    first_residual_started = time.perf_counter()
    current_residual, _, _ = residual_and_jacobian(
        q, positions, velocities, rest_lengths, masses, integrals, previous_errors, pulse_forces,
        prescribed_mask, prescribed, args, False, block_plan
    )
    add_time("first_residual_s", first_residual_started)
    residual_norm = float(torch.linalg.norm(current_residual))
    line_search_failures = 0
    step_norm = 0.0
    converged = residual_norm < args.residual_tol
    iteration = 0
    stagnation_count = 0
    newton_loop_started = time.perf_counter()
    for iteration in range(1, args.max_newton_iters + 1):
        if converged:
            break
        jacobian_started = time.perf_counter()
        residual, jacobian, _ = residual_and_jacobian(
            q, positions, velocities, rest_lengths, masses, integrals, previous_errors, pulse_forces,
            prescribed_mask, prescribed, args, True, block_plan
        )
        add_time("jacobian_assembly_s", jacobian_started)
        # 连续数次残差只获得很小改善时，绕过带状解并用一次稠密解救援。
        force_dense_rescue = stagnation_count >= 3
        solve_started = time.perf_counter()
        delta, used_dense_fallback = jacobian.solve(-residual, force_dense=force_dense_rescue)
        add_time("linear_solve_s", solve_started)
        args.banded_numerical_fallback_count = getattr(args, "banded_numerical_fallback_count", 0) + int(used_dense_fallback)
        if force_dense_rescue:
            stagnation_count = 0
        step_norm = float(torch.linalg.norm(delta))
        if step_norm < args.step_tol:
            converged = True
            break
        base_norm = float(torch.linalg.norm(residual))
        alpha = 1.0
        accepted = False
        for _ in range(12):
            trial = (q + alpha * delta).detach()
            line_search_started = time.perf_counter()
            trial_residual, _, _ = residual_and_jacobian(
                trial, positions, velocities, rest_lengths, masses, integrals, previous_errors, pulse_forces,
                prescribed_mask, prescribed, args, False, block_plan
            )
            add_time("line_search_s", line_search_started)
            trial_norm = float(torch.linalg.norm(trial_residual))
            if np.isfinite(trial_norm) and trial_norm < base_norm:
                q, residual_norm, accepted = trial, trial_norm, True
                break
            alpha *= 0.5
        if not accepted:
            line_search_failures += 1
            q = (q + 0.05 * delta).detach()
            line_search_started = time.perf_counter()
            trial_residual, _, _ = residual_and_jacobian(
                q, positions, velocities, rest_lengths, masses, integrals, previous_errors, pulse_forces,
                prescribed_mask, prescribed, args, False, block_plan
            )
            add_time("line_search_s", line_search_started)
            residual_norm = float(torch.linalg.norm(trial_residual))
        stagnation_count = stagnation_count + 1 if residual_norm >= 0.95 * base_norm else 0
        converged = residual_norm < args.residual_tol
    add_time("newton_loop_s", newton_loop_started)

    new_positions = unpack(q, free_mask, prescribed).detach()
    new_velocities = (new_positions - positions) / args.dt_active
    new_velocities[:2] = 0.0
    if tip_constrained:
        new_velocities[-1] = (prescribed[-1] - positions[-1]) / args.dt_active
    contact_started = time.perf_counter()
    with torch.no_grad():
        force, new_integrals, active, weights, errors, distances = contact_quantities(
            new_positions, integrals, previous_errors, args
        )
    add_time("contact_s", contact_started)
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
    # per-dt 计时：每个隐式 Euler 步(=一次 dt 推进)都记录墙钟时间、阶段、子步长与
    # 内部时间去向，用于诊断真实世界中的每步计算耗时。
    args.performance_profile = getattr(args, "performance_profile", None) or {}
    args.dt_timings: list[dict] = []
    args._dt_phase = "pulse"
    args._current_dt_record: dict | None = None
    time_value = 0.0
    abort_reason: str | None = None
    completed_steps = 0
    last_force = torch.zeros_like(positions)
    last_active = torch.zeros(args.num_nodes, dtype=torch.bool, device=device)
    last_info = NewtonInfo(True, 0, 0.0, 0.0, 0)
    total_newton_iterations = 0
    adaptive_accepted_steps = 0
    adaptive_rejected_steps = 0
    adaptive_increment_sum = 0.0
    adaptive_micro_relaxation_steps = 0
    adaptive_velocity_rejections = 0
    fixed_ramp_steps_total = 0
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
        extra_history_hook = getattr(args, "extra_history_record_hook", None)
        if extra_history_hook is not None:
            extra_history_hook(history, positions)

    def advance(
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        applied_forces: torch.Tensor,
        tip_constrained: bool,
        displacement: float,
        dt_step: float,
    ):
        p, v, i = state
        # 所有 dt 步进（脉冲、ramp、continuation、micro、relax 及其子步重试）都经过这里。
        step_record: dict = {
            "phase": args._dt_phase,
            "dt_step": float(dt_step),
            "wall_s": 0.0,
            "iterations": 0,
            "converged": False,
            "breakdown": {},
        }
        args._current_dt_record = step_record
        sync_device(p)
        started = time.perf_counter()
        try:
            result = implicit_euler_step(
                p, v, rest_lengths, masses, i, initial_positions, applied_forces,
                tip_constrained, displacement, dt_step, args,
            )
        finally:
            sync_device(p)
            step_record["wall_s"] = time.perf_counter() - started
            step_record["realtime_ratio"] = step_record["wall_s"] / max(float(dt_step), 1.0e-30)
            args._current_dt_record = None
        info = result[8]
        step_record["iterations"] = info.iterations
        step_record["converged"] = bool(info.converged)
        dt_timings = getattr(args, "dt_timings", None)
        if dt_timings is not None:
            dt_timings.append(step_record)
        return result

    def snapshot_mesh_cache():
        """Snapshot mutable mesh-search state so a failed trial is transactional."""
        cache = getattr(args, "mesh_contact_cache", None)
        if cache is None:
            return None
        return (
            cache.triangle_local.clone(), cache.last_positions.clone(),
            getattr(args, "mesh_accepted_step_count", 0),
        )

    def restore_mesh_cache(snapshot, force_uniform_refresh: bool = False) -> None:
        cache = getattr(args, "mesh_contact_cache", None)
        if cache is not None and snapshot is not None:
            cache.triangle_local.copy_(snapshot[0])
            cache.last_positions.copy_(snapshot[1])
            args.mesh_accepted_step_count = snapshot[2]
            if force_uniform_refresh:
                cache.triangle_local.fill_(-1)
                cache.last_positions.fill_(float("nan"))
        # Frozen/diagnostic details belong to the abandoned trial as well.
        args._frozen_mesh_topology = None
        args._last_mesh_evaluation = None

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
        cache_checkpoint = snapshot_mesh_cache()
        previous_attempt_failed = False
        while True:
            for subdivision_count in RAMP_RETRY_SUBDIVISIONS:
                # Every retry starts from exactly the accepted checkpoint.  Once
                # any attempt failed, distrust its face cache and rebuild it with
                # the uniform grid on the next attempt.
                restore_mesh_cache(cache_checkpoint, force_uniform_refresh=previous_attempt_failed)
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
                        previous_attempt_failed = True
                        last_failed = latest[8]
                        break
                    state = latest[:3]
                if not failed:
                    return True, latest, subdivision_count, None
            next_scale = effective_ki_scale * KI_ADAPTIVE_REDUCTION_FACTOR
            if args.Ki <= 0.0 or next_scale < KI_ADAPTIVE_MIN_SCALE - 1.0e-15:
                restore_mesh_cache(cache_checkpoint)
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
        f"increment={args.tip_displacement_step:g} m, ramp_mode={args.ramp_mode}, "
        f"ramp_steps={args.tip_ramp_steps}, "
        f"contact_tolerance={args.contact_tolerance:g} m"
    )
    if args.ramp_mode == "adaptive":
        print(
            f"adaptive continuation: micro_relax_steps<={args.adaptive_relax_steps}, "
            f"grow if v<{args.adaptive_speed_grow_factor:g}*settle_tol, "
            f"rollback/shrink if v>{args.adaptive_speed_shrink_factor:g}*settle_tol",
            flush=True,
        )
    if getattr(args, "pulse_nodes", None) is not None:
        pulse_kind = "explicit"
        seed_note = "n/a (fixed nodes & forces)"
    else:
        pulse_kind = "random"
        seed_note = f"seed={args.pulse_seed}"
    print(
        f"initial {pulse_kind} yz-plane pulses: count={len(pulse_nodes)}, nodes={pulse_nodes.tolist()}, "
        f"magnitudes={np.array2string(pulse_magnitudes, precision=4)}, {seed_note}, "
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
        macro_increment = target - previous
        displacement = previous
        internal_step = 0
        if args.ramp_mode == "fixed":
            # 与 v2 相同的三次平滑固定 ramp，便于复现旧算法和进行耗时/结果对照。
            for ramp in range(1, args.tip_ramp_steps + 1):
                fraction = ramp / args.tip_ramp_steps
                smooth = fraction * fraction * (3.0 - 2.0 * fraction)
                displacement = previous + macro_increment * smooth
                previous_fraction = (ramp - 1) / args.tip_ramp_steps
                previous_smooth = previous_fraction * previous_fraction * (3.0 - 2.0 * previous_fraction)
                ramp_start = previous + macro_increment * previous_smooth
                args._dt_phase = "ramp"
                continuation_started = time.perf_counter()
                accepted, result, subdivisions, failed_info = retry_step(
                    (positions, velocities, integrals), zero_forces, True, ramp_start, displacement,
                    phase_label=f"load step {load_step}/{args.num_load_steps}, fixed ramp {ramp}/{args.tip_ramp_steps}",
                )
                profile = getattr(args, "performance_profile", None)
                if profile is not None:
                    profile["fixed_continuation_s"] = profile.get("fixed_continuation_s", 0.0) + time.perf_counter() - continuation_started
                if not accepted:
                    abort_reason = (
                        f"Newton failure during fixed ramp at load step {load_step}, ramp step {ramp}: "
                        f"residual={failed_info.residual_norm:.3e}" if failed_info else
                        f"Newton failure during fixed ramp at load step {load_step}, ramp step {ramp}"
                    )
                    print(f"EARLY ABORT: {abort_reason}")
                    break
                (
                    positions, velocities, integrals, last_force, last_active, weights, _errors, _distances,
                    last_info, prescribed_mask,
                ) = result
                fixed_ramp_steps_total += 1
                internal_step = ramp
                time_value += args.dt
                current_max_speed = max_free_speed(velocities, prescribed_mask)
                max_speed_seen = max(max_speed_seen, current_max_speed)
                ramp_protection = compute_protection_error(positions, args.tube_radius, args.safe_distance)
                max_contact_error_seen = max(max_contact_error_seen, float(torch.max(ramp_protection)))
                if ramp % DIAGNOSTIC_INTERVAL == 0 or ramp == args.tip_ramp_steps:
                    print(
                        f"[RAMP  {load_step:02d}/{args.num_load_steps:02d} | "
                        f"{ramp:4d}/{args.tip_ramp_steps} | {100.0 * fraction:5.1f}%] "
                        f"tip={displacement * 1e3:.4f} mm | v={current_max_speed:.2e} | "
                        f"Newton={last_info.iterations} | R={last_info.residual_norm:.2e} | "
                        f"contacts={int(torch.count_nonzero(last_active))}", flush=True,
                    )
                if ramp % args.history_save_interval == 0 or ramp == args.tip_ramp_steps:
                    record(load_step, displacement, last_force, last_active, weights, last_info, True)
        else:
            minimum_increment = macro_increment / args.tip_ramp_steps
            adaptive_increment = macro_increment / 2.0
            grow_speed_limit = args.adaptive_speed_grow_factor * args.settle_speed_tol
            shrink_speed_limit = args.adaptive_speed_shrink_factor * args.settle_speed_tol
            while displacement < target - 1e-15:
                internal_step += 1
                trial_displacement = min(displacement + adaptive_increment, target)
                proposed_increment = trial_displacement - displacement
                checkpoint = (positions, velocities, integrals)
                checkpoint_ki_scale = effective_ki_scale
                checkpoint_ki = effective_ki
                mesh_cache = getattr(args, "mesh_contact_cache", None)
                adaptive_cache_checkpoint = snapshot_mesh_cache()
                args._dt_phase = "continuation"
                continuation_started = time.perf_counter()
                accepted, result, subdivisions, failed_info = retry_step(
                    checkpoint, zero_forces, True, displacement, trial_displacement,
                    phase_label=f"load step {load_step}/{args.num_load_steps}, adaptive step {internal_step}",
                )
                profile = getattr(args, "performance_profile", None)
                if profile is not None:
                    profile["adaptive_continuation_s"] = profile.get("adaptive_continuation_s", 0.0) + time.perf_counter() - continuation_started
                if not accepted:
                    adaptive_rejected_steps += 1
                    restore_mesh_cache(adaptive_cache_checkpoint)
                    if adaptive_increment > minimum_increment * (1.0 + 1e-12):
                        adaptive_increment = max(0.5 * adaptive_increment, minimum_increment)
                        continue
                    abort_reason = f"adaptive continuation failed at load step {load_step}, tip={displacement:.6e} m"
                    print(f"EARLY ABORT: {abort_reason}")
                    break

                ramp_info = result[8]
                ramp_subdivisions = subdivisions
                trial_state = result[:3]
                trial_result = result
                micro_steps = 0
                micro_failed = False
                current_max_speed = max_free_speed(result[1], result[9])

                # 固定刚到达的末端位置做短松弛；速度足够低即可提前结束，不会人为清零速度。
                micro_started = time.perf_counter()
                while micro_steps < args.adaptive_relax_steps and (
                    micro_steps == 0 or current_max_speed > shrink_speed_limit
                ):
                    micro_steps += 1
                    args._dt_phase = "micro"
                    micro_ok, micro_result, _micro_subdivisions, micro_failed_info = retry_step(
                        trial_state, zero_forces, True, trial_displacement, trial_displacement,
                        phase_label=(
                            f"load step {load_step}/{args.num_load_steps}, adaptive step {internal_step}, "
                            f"micro relaxation {micro_steps}/{args.adaptive_relax_steps}"
                        ),
                    )
                    if not micro_ok:
                        failed_info = micro_failed_info
                        micro_failed = True
                        break
                    trial_result = micro_result
                    trial_state = micro_result[:3]
                    current_max_speed = max_free_speed(micro_result[1], micro_result[9])
                if profile is not None:
                    profile["adaptive_micro_relaxation_s"] = profile.get("adaptive_micro_relaxation_s", 0.0) + time.perf_counter() - micro_started

                velocity_too_high = current_max_speed > shrink_speed_limit
                can_shrink = adaptive_increment > minimum_increment * (1.0 + 1e-12)
                if (micro_failed or velocity_too_high) and can_shrink:
                    # 不接受该动态性过强的试探步，恢复位置、速度、积分状态及本次试探前的 Ki。
                    adaptive_rejected_steps += 1
                    if velocity_too_high:
                        adaptive_velocity_rejections += 1
                    effective_ki_scale = checkpoint_ki_scale
                    effective_ki = checkpoint_ki
                    args.ki_active = effective_ki
                    restore_mesh_cache(adaptive_cache_checkpoint)
                    adaptive_increment = max(0.5 * adaptive_increment, minimum_increment)
                    reason = "micro relaxation failed" if micro_failed else f"v={current_max_speed:.2e} m/s"
                    print(
                        f"[ADAPT ROLLBACK {load_step:02d}/{args.num_load_steps:02d} | {internal_step:3d}] "
                        f"tip trial={trial_displacement * 1e3:.4f} mm | ds={proposed_increment * 1e3:.4f} mm | "
                        f"{reason}; retry ds={adaptive_increment * 1e3:.4f} mm",
                        flush=True,
                    )
                    continue
                if micro_failed:
                    abort_reason = f"adaptive micro relaxation failed at minimum increment, load step {load_step}"
                    print(f"EARLY ABORT: {abort_reason}")
                    break

                (
                    positions, velocities, integrals, last_force, last_active, weights, _errors, _distances,
                    last_info, prescribed_mask,
                ) = trial_result
                displacement = trial_displacement
                adaptive_accepted_steps += 1
                adaptive_micro_relaxation_steps += micro_steps
                adaptive_increment_sum += proposed_increment
                time_value += args.dt * (1 + micro_steps)
                max_speed_seen = max(max_speed_seen, current_max_speed)
                ramp_protection = compute_protection_error(positions, args.tube_radius, args.safe_distance)
                max_contact_error_seen = max(max_contact_error_seen, float(torch.max(ramp_protection)))
                print(
                    f"[ADAPT {load_step:02d}/{args.num_load_steps:02d} | {internal_step:3d}] "
                    f"tip={displacement * 1e3:.4f} mm | ds={proposed_increment * 1e3:.4f} mm | "
                    f"v={current_max_speed:.2e} | micro={micro_steps} | "
                    f"Newton={ramp_info.iterations} | R={last_info.residual_norm:.2e} | "
                    f"contacts={int(torch.count_nonzero(last_active))}", flush=True,
                )
                if internal_step % args.history_save_interval == 0 or displacement >= target - 1e-15:
                    record(load_step, displacement, last_force, last_active, weights, last_info, True)

                easy_step = (
                    ramp_info.iterations <= 5 and ramp_info.line_search_failures == 0
                    and ramp_subdivisions == 1 and current_max_speed <= grow_speed_limit
                )
                difficult_step = (
                    ramp_info.iterations > 10 or ramp_info.line_search_failures > 1
                    or ramp_subdivisions > 1 or current_max_speed > shrink_speed_limit
                )
                if easy_step and displacement < target:
                    adaptive_increment = min(1.75 * adaptive_increment, target - displacement)
                elif difficult_step:
                    adaptive_increment = max(0.5 * adaptive_increment, minimum_increment)
        if abort_reason:
            break
        print(
            f"[RAMP COMPLETE] increment={args.tip_displacement_step * 1e3:.3f} mm "
            f"over {internal_step} {args.ramp_mode} steps; holding tip for relaxation",
            flush=True,
        )
        stable = 0
        settled = False
        max_speed_seen = 0.0
        max_contact_error_seen = 0.0
        max_pi_force_seen = 0.0
        for relaxation in range(1, args.max_settle_steps + 1):
            args._dt_phase = "relax"
            relaxation_started = time.perf_counter()
            accepted, result, subdivisions, failed_info = retry_step(
                (positions, velocities, integrals), zero_forces, True, target, target,
                phase_label=f"load step {load_step}/{args.num_load_steps}, relaxation step {relaxation}",
            )
            profile = getattr(args, "performance_profile", None)
            if profile is not None:
                profile["relaxation_s"] = profile.get("relaxation_s", 0.0) + time.perf_counter() - relaxation_started
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
                    f"peak_v={max_speed_seen:.2e} m/s | final_v={max_speed:.2e} m/s | "
                    f"contact={max_contact_error_seen * 1e3:.4f} mm",
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
        "adaptive_continuation_accepted_steps": np.asarray(adaptive_accepted_steps),
        "adaptive_continuation_rejected_steps": np.asarray(adaptive_rejected_steps),
        "adaptive_continuation_average_increment_m": np.asarray(
            adaptive_increment_sum / max(adaptive_accepted_steps, 1)
        ),
        "ramp_mode": np.asarray(args.ramp_mode),
        "adaptive_micro_relaxation_steps": np.asarray(adaptive_micro_relaxation_steps),
        "adaptive_velocity_rejections": np.asarray(adaptive_velocity_rejections),
        "fixed_ramp_steps_total": np.asarray(fixed_ramp_steps_total),
        "adaptive_relax_steps_limit": np.asarray(args.adaptive_relax_steps),
        "adaptive_speed_grow_limit_m_per_s": np.asarray(args.adaptive_speed_grow_factor * args.settle_speed_tol),
        "adaptive_speed_shrink_limit_m_per_s": np.asarray(args.adaptive_speed_shrink_factor * args.settle_speed_tol),
        "banded_numerical_fallback_count": np.asarray(getattr(args, "banded_numerical_fallback_count", 0)),
    })
    dt_timings = getattr(args, "dt_timings", None) or []
    if dt_timings:
        dt_phases = np.asarray([rec["phase"] for rec in dt_timings])
        dt_step_values = np.asarray([rec["dt_step"] for rec in dt_timings])
        dt_walls = np.asarray([rec["wall_s"] for rec in dt_timings])
        dt_iters = np.asarray([rec["iterations"] for rec in dt_timings])
        dt_converged = np.asarray([rec["converged"] for rec in dt_timings])
        history_np["dt_phase"] = dt_phases
        history_np["dt_step_simulated_s"] = dt_step_values
        history_np["dt_wall_seconds"] = dt_walls
        history_np["dt_realtime_ratio"] = np.asarray([
            rec.get("realtime_ratio", rec["wall_s"] / max(rec["dt_step"], 1.0e-30))
            for rec in dt_timings
        ])
        history_np["dt_newton_iterations"] = dt_iters
        history_np["dt_converged"] = dt_converged
        for key in ("first_residual_s", "jacobian_assembly_s", "linear_solve_s",
                    "line_search_s", "contact_s", "newton_loop_s"):
            history_np[f"dt_{key}"] = np.asarray([rec["breakdown"].get(key, 0.0) for rec in dt_timings])
    performance_profile = getattr(args, "performance_profile", {})
    for profile_name, profile_seconds in performance_profile.items():
        history_np[f"performance_{profile_name}"] = np.asarray(profile_seconds)
    if performance_profile:
        print("PERFORMANCE SUMMARY")
        for profile_name, profile_seconds in sorted(performance_profile.items(), key=lambda item: item[1], reverse=True):
            print(f"  {profile_name}: {profile_seconds:.6f} s")
        print(
            f"  ramp mode={args.ramp_mode} | adaptive accepted/rejected: "
            f"{adaptive_accepted_steps}/{adaptive_rejected_steps} | "
            f"average increment={adaptive_increment_sum / max(adaptive_accepted_steps, 1):.6e} m | "
            f"micro relax={adaptive_micro_relaxation_steps} | velocity rollbacks={adaptive_velocity_rejections} | "
            f"fixed ramp steps={fixed_ramp_steps_total} | "
            f"banded fallback={getattr(args, 'banded_numerical_fallback_count', 0)}"
        )
        mesh_cache = getattr(args, "mesh_contact_cache", None)
        if mesh_cache is not None:
            cache_queries = mesh_cache.hit_count + mesh_cache.miss_count
            print(
                f"  mesh cache hits/misses: {mesh_cache.hit_count}/{mesh_cache.miss_count} | "
                f"hit rate={mesh_cache.hit_count / max(cache_queries, 1):.2%} | "
                f"uniform-grid fallbacks={mesh_cache.uniform_fallback_count} | "
                f"periodic global refreshes={getattr(mesh_cache, 'global_refresh_count', 0)}"
            )
        exact_hits = getattr(args, "mesh_exact_reuse_hit_count", 0)
        exact_misses = getattr(args, "mesh_exact_reuse_miss_count", 0)
        if exact_hits or exact_misses:
            print(
                f"  mesh exact-state reuse hits/misses: {exact_hits}/{exact_misses} | "
                f"hit rate={exact_hits / max(exact_hits + exact_misses, 1):.2%}"
            )
    if dt_timings:
        print("=" * 76)
        print(f"PER-DT WALL-CLOCK PROFILE ({len(dt_timings)} implicit Euler dt steps, all phases)")
        print(
            f"  per-dt wall: mean={dt_walls.mean() * 1e3:.3f} ms | median={np.median(dt_walls) * 1e3:.3f} ms | "
            f"p90={np.percentile(dt_walls, 90) * 1e3:.3f} ms | p95={np.percentile(dt_walls, 95) * 1e3:.3f} ms | "
            f"min={dt_walls.min() * 1e3:.3f} ms | max={dt_walls.max() * 1e3:.3f} ms | "
            f"total dt wall={dt_walls.sum():.3f} s"
        )
        total_simulated = float(dt_step_values.sum())
        if total_simulated > 0:
            print(
                f"  wall/simulated-dt ratio = {dt_walls.sum() / total_simulated:.6f} real-seconds "
                f"per simulated second (模拟 dt={args.dt:g}s 的一步实际耗时)"
            )
            realtime = history_np["dt_realtime_ratio"]
            print(
                f"  realtime ratio wall/dt: mean={realtime.mean():.6f} | "
                f"median={np.median(realtime):.6f} | p95={np.percentile(realtime, 95):.6f} | "
                f"max={realtime.max():.6f} | pass={np.mean(realtime <= 1.0):.2%}"
            )
        print("  per-phase dt breakdown:")
        for phase_name in ("pulse", "ramp", "continuation", "micro", "relax"):
            mask = dt_phases == phase_name
            count = int(mask.sum())
            if count == 0:
                continue
            pwalls = dt_walls[mask]
            piters = dt_iters[mask]
            print(
                f"    [{phase_name:13s}] n={count:6d} | mean={pwalls.mean() * 1e3:9.3f} ms | "
                f"median={np.median(pwalls) * 1e3:9.3f} ms | max={pwalls.max() * 1e3:9.3f} ms | "
                f"total={pwalls.sum():7.3f} s | mean Newton={piters.mean():6.2f}"
            )
        breakdown_keys = ("first_residual_s", "jacobian_assembly_s", "linear_solve_s", "line_search_s", "contact_s")
        print("  average within-dt time fraction:")
        for key in breakdown_keys:
            fraction = float(np.mean([
                rec["breakdown"].get(key, 0.0) / max(rec["wall_s"], 1.0e-15) for rec in dt_timings
            ]))
            print(f"    {key:20s} {fraction:7.1%}")
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
    # Explicit pulse mode: nodes and (Fy, Fz) must round-trip exactly, and
    # misconfigured runs must fail fast instead of silently falling back to random.
    args.pulse_nodes = "5,10,20"
    args.pulse_fyfz = "0.05,0.0,0.02,0.04,0.01,0.03"
    nodes, magnitudes, _angles, directions, profile = make_initial_pulse_profile(args)
    assert nodes.tolist() == [5, 10, 20]
    assert np.allclose(profile[nodes, 1:], [[0.05, 0.0], [0.02, 0.04], [0.01, 0.03]])
    assert np.allclose(magnitudes, [0.05, np.hypot(0.02, 0.04), np.hypot(0.01, 0.03)])
    assert np.allclose(directions[:, 0], 0.0)
    assert np.allclose(profile[nodes], magnitudes[:, None] * directions)
    args.pulse_nodes = "0,1"
    try:
        make_initial_pulse_profile(args)
        raise AssertionError("fixed/boundary nodes must be rejected")
    except ValueError:
        pass
    args.pulse_nodes = "5,10"
    args.pulse_fyfz = "0.05,0.0"
    try:
        make_initial_pulse_profile(args)
        raise AssertionError("mismatched node/force counts must be rejected")
    except ValueError:
        pass
    args.pulse_nodes = "5"
    args.pulse_fyfz = None
    try:
        make_initial_pulse_profile(args)
        raise AssertionError("half-configured explicit mode must be rejected")
    except ValueError:
        pass
    args.pulse_nodes = None
    args.pulse_fyfz = None
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
    parser.add_argument(
        "--pulse-nodes", type=str, default=PULSE_NODES,
        help=(
            "comma-separated node indices for a fixed perturbation, e.g. '5,10,20' "
            "(must lie in the free range [2, num_nodes-2]); must be given together "
            "with --pulse-fyfz, omit both for fully-random pulses"
        ),
    )
    parser.add_argument(
        "--pulse-fyfz", type=str, default=PULSE_FYFZ,
        help=(
            "comma-separated per-node force components (fy,fz), e.g. '0.05,0.0,0.02,0.04' "
            "(2 values per node, in node order); must be given together with --pulse-nodes, "
            "omit both for fully-random pulses"
        ),
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
    parser.add_argument(
        "--ramp-mode", choices=("adaptive", "fixed"), default=RAMP_MODE,
        help="adaptive uses velocity-controlled continuation; fixed reproduces the original smooth ramp",
    )
    parser.add_argument(
        "--tip-ramp-steps", type=int, default=TIP_RAMP_STEPS,
        help="fixed ramp step count; in adaptive mode this defines the minimum displacement increment",
    )
    parser.add_argument(
        "--adaptive-relax-steps", type=int, default=ADAPTIVE_RELAX_STEPS,
        help="maximum held-tip micro-relaxation steps after each adaptive displacement increment",
    )
    parser.add_argument(
        "--adaptive-speed-grow-factor", type=float, default=ADAPTIVE_SPEED_GROW_FACTOR,
        help="grow an adaptive increment only below this factor times settle-speed-tol",
    )
    parser.add_argument(
        "--adaptive-speed-shrink-factor", type=float, default=ADAPTIVE_SPEED_SHRINK_FACTOR,
        help="rollback and halve an adaptive trial above this factor times settle-speed-tol",
    )
    parser.add_argument("--history-save-interval", type=int, default=HISTORY_SAVE_INTERVAL)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default=DEVICE_DEFAULT,
        help="calculation device; the default comes from DEVICE_DEFAULT in the parameter section",
    )
    parser.add_argument(
        "--compute-kernels", choices=("reference", "optimized"), default="optimized",
        help="contact/Jacobian implementation; reference keeps autograd for A/B validation",
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
    for name in ("dt", "max_newton_iters", "residual_tol", "step_tol", "settle_speed_tol", "settle_residual_tol", "settle_confirm_steps", "max_settle_steps", "tip_total_displacement", "num_load_steps", "tip_ramp_steps", "history_save_interval", "adaptive_speed_grow_factor", "adaptive_speed_shrink_factor"):
        if getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.adaptive_relax_steps < 0:
        parser.error("adaptive-relax-steps must be non-negative")
    if args.adaptive_speed_shrink_factor <= args.adaptive_speed_grow_factor:
        parser.error("adaptive-speed-shrink-factor must be greater than adaptive-speed-grow-factor")
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


@dataclass
class BlockPentaSystem:
    """3x3 block pentadiagonal Newton matrix used by the v3 mesh solver."""

    diagonal: torch.Tensor       # (B,3,3)
    upper1: torch.Tensor         # (B-1,3,3)
    upper2: torch.Tensor         # (B-2,3,3)
    lower1: torch.Tensor         # (B-1,3,3)
    lower2: torch.Tensor         # (B-2,3,3)

    @classmethod
    def zeros(cls, blocks: int, dtype: torch.dtype, device: torch.device):
        z = lambda count: torch.zeros((max(count, 0), 3, 3), dtype=dtype, device=device)
        return cls(z(blocks), z(blocks - 1), z(blocks - 2), z(blocks - 1), z(blocks - 2))

    def dense_reference(self) -> torch.Tensor:
        blocks = len(self.diagonal)
        dense = torch.zeros((3 * blocks, 3 * blocks), dtype=self.diagonal.dtype, device=self.diagonal.device)
        for index in range(blocks):
            dense[3 * index:3 * index + 3, 3 * index:3 * index + 3] = self.diagonal[index]
        for index in range(blocks - 1):
            dense[3 * index:3 * index + 3, 3 * (index + 1):3 * (index + 2)] = self.upper1[index]
            dense[3 * (index + 1):3 * (index + 2), 3 * index:3 * index + 3] = self.lower1[index]
        for index in range(blocks - 2):
            dense[3 * index:3 * index + 3, 3 * (index + 2):3 * (index + 3)] = self.upper2[index]
            dense[3 * (index + 2):3 * (index + 3), 3 * index:3 * index + 3] = self.lower2[index]
        return dense

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        """Matrix-vector product directly from the five block diagonals."""
        blocks = len(self.diagonal)
        x = vector.reshape(blocks, 3)
        value = torch.bmm(self.diagonal, x.unsqueeze(-1)).squeeze(-1)
        if blocks > 1:
            value[:-1] += torch.bmm(self.upper1, x[1:].unsqueeze(-1)).squeeze(-1)
            value[1:] += torch.bmm(self.lower1, x[:-1].unsqueeze(-1)).squeeze(-1)
        if blocks > 2:
            value[:-2] += torch.bmm(self.upper2, x[2:].unsqueeze(-1)).squeeze(-1)
            value[2:] += torch.bmm(self.lower2, x[:-2].unsqueeze(-1)).squeeze(-1)
        return value.reshape(-1)

    def _dense_solve(self, rhs: torch.Tensor) -> torch.Tensor:
        """Reliable consolidated dense rescue, used only after a numerical warning."""
        dense = self.dense_reference()
        solution, info = torch.linalg.solve_ex(dense, rhs)
        good = int(info) == 0 and bool(torch.isfinite(solution).all())
        if good:
            relative = torch.linalg.vector_norm(dense @ solution - rhs) / torch.clamp(
                torch.linalg.vector_norm(rhs), min=1.0e-30
            )
            good = bool(relative <= 1.0e-10)
        if not good:
            solution = torch.linalg.lstsq(dense, rhs).solution
        return solution

    def _solve_cpu_block_lu(self, rhs: torch.Tensor) -> torch.Tensor:
        """Original serial block-pentadiagonal LU path, efficient for CPU."""
        blocks = len(self.diagonal)
        diagonal = self.diagonal.clone()
        upper1, upper2 = self.upper1.clone(), self.upper2.clone()
        lower1, lower2 = self.lower1.clone(), self.lower2.clone()
        vector = rhs.reshape(blocks, 3).clone()
        for index in range(blocks):
            if not bool(torch.isfinite(diagonal[index]).all()):
                raise RuntimeError("non-finite block pivot")
            if index + 1 < blocks:
                factor1, info = torch.linalg.solve_ex(diagonal[index].T, lower1[index].T)
                if int(info) != 0:
                    raise RuntimeError("singular block pivot")
                factor1 = factor1.T
                diagonal[index + 1] -= factor1 @ upper1[index]
                vector[index + 1] -= factor1 @ vector[index]
                if index + 2 < blocks:
                    upper1[index + 1] -= factor1 @ upper2[index]
            if index + 2 < blocks:
                factor2, info = torch.linalg.solve_ex(diagonal[index].T, lower2[index].T)
                if int(info) != 0:
                    raise RuntimeError("singular block pivot")
                factor2 = factor2.T
                lower1[index + 1] -= factor2 @ upper1[index]
                diagonal[index + 2] -= factor2 @ upper2[index]
                vector[index + 2] -= factor2 @ vector[index]
        solution = torch.empty_like(vector)
        for index in range(blocks - 1, -1, -1):
            value = vector[index]
            if index + 1 < blocks:
                value = value - upper1[index] @ solution[index + 1]
            if index + 2 < blocks:
                value = value - upper2[index] @ solution[index + 2]
            solution[index], info = torch.linalg.solve_ex(diagonal[index], value)
            if int(info) != 0:
                raise RuntimeError("singular block during back solve")
        return solution.reshape(-1)

    def _solve_cuda_pcr(self, rhs: torch.Tensor) -> torch.Tensor:
        """Parallel cyclic reduction on paired 6x6 blocks for CUDA.

        Pairing consecutive 3x3 rows turns the block-pentadiagonal system into
        a 6x6 block-tridiagonal system.  PCR then needs O(log N) batched solves
        instead of hundreds of serial 3x3 CUDA kernel launches.
        """
        blocks = len(self.diagonal)
        groups = (blocks + 1) // 2
        dtype, device = self.diagonal.dtype, self.diagonal.device
        left = torch.zeros((groups, 6, 6), dtype=dtype, device=device)
        centre = torch.zeros_like(left)
        right = torch.zeros_like(left)
        vector = torch.zeros((groups, 6), dtype=dtype, device=device)

        even_count = len(self.diagonal[0::2])
        odd_count = len(self.diagonal[1::2])
        centre[:even_count, :3, :3] = self.diagonal[0::2]
        vector[:even_count, :3] = rhs.reshape(blocks, 3)[0::2]
        if odd_count:
            centre[:odd_count, 3:, 3:] = self.diagonal[1::2]
            centre[:odd_count, :3, 3:] = self.upper1[0::2][:odd_count]
            centre[:odd_count, 3:, :3] = self.lower1[0::2][:odd_count]
            vector[:odd_count, 3:] = rhs.reshape(blocks, 3)[1::2]
        if blocks % 2:
            centre[-1, 3:, 3:] = torch.eye(3, dtype=dtype, device=device)

        if groups > 1:
            left[1:, :3, :3] = self.lower2[0::2]
            left[1:, :3, 3:] = self.lower1[1::2]
            odd_left2 = self.lower2[1::2]
            if len(odd_left2):
                left[1:1 + len(odd_left2), 3:, 3:] = odd_left2
            right[:-1, :3, :3] = self.upper2[0::2]
            right[:-1, 3:, :3] = self.upper1[1::2]
            odd_right2 = self.upper2[1::2]
            if len(odd_right2):
                right[:len(odd_right2), 3:, 3:] = odd_right2

        stride = 1
        while stride < groups:
            new_left = torch.zeros_like(left)
            new_right = torch.zeros_like(right)
            new_centre = centre.clone()
            new_vector = vector.clone()
            # Python already knows the valid row ranges from ``stride`` and
            # ``groups``.  Continuous slices avoid GPU mask construction,
            # any-reduction and the associated CPU synchronization.
            alpha = torch.linalg.solve(
                centre[:-stride].transpose(1, 2), left[stride:].transpose(1, 2)
            ).transpose(1, 2)
            new_centre[stride:] -= alpha @ right[:-stride]
            new_vector[stride:] -= (alpha @ vector[:-stride, :, None]).squeeze(-1)
            new_left[stride:] = -alpha @ left[:-stride]

            gamma = torch.linalg.solve(
                centre[stride:].transpose(1, 2), right[:-stride].transpose(1, 2)
            ).transpose(1, 2)
            new_centre[:-stride] -= gamma @ left[stride:]
            new_vector[:-stride] -= (gamma @ vector[stride:, :, None]).squeeze(-1)
            new_right[:-stride] = -gamma @ right[stride:]
            left, centre, right, vector = new_left, new_centre, new_right, new_vector
            stride *= 2
        grouped_solution = torch.linalg.solve(centre, vector)
        solution = torch.empty((blocks, 3), dtype=dtype, device=device)
        solution[0::2] = grouped_solution[:even_count, :3]
        if odd_count:
            solution[1::2] = grouped_solution[:odd_count, 3:]
        return solution.reshape(-1)

    def solve(self, rhs: torch.Tensor, force_dense: bool = False) -> tuple[torch.Tensor, bool]:
        """Solve with CUDA PCR or CPU block LU, with verified dense fallback."""
        if force_dense:
            return self._dense_solve(rhs), True
        try:
            solution = self._solve_cuda_pcr(rhs) if self.diagonal.is_cuda else self._solve_cpu_block_lu(rhs)
            rhs_norm = torch.clamp(torch.linalg.vector_norm(rhs), min=1.0e-30)
            relative = torch.linalg.vector_norm(self.matvec(solution) - rhs) / rhs_norm
            # PCR 对强刚度/接触造成的病态系统可能损失若干位精度；先用同一
            # 带状算子做至多两次迭代精化，仍不可靠时才构造稠密矩阵。
            if self.diagonal.is_cuda:
                for _ in range(2):
                    if bool(relative <= 1.0e-8):
                        break
                    correction_rhs = rhs - self.matvec(solution)
                    correction = self._solve_cuda_pcr(correction_rhs)
                    solution = solution + correction
                    relative = torch.linalg.vector_norm(self.matvec(solution) - rhs) / rhs_norm
            if not bool(relative <= 1.0e-8):
                raise RuntimeError(f"banded relative residual too large: {float(relative):.3e}")
            return solution, False
        except RuntimeError:
            return self._dense_solve(rhs), True


def main(argv=None) -> None:
    with cli_arguments(argv):
        parsed = parse_args()
        if parsed.self_test:
            self_test(parsed)
        else:
            run_simulation(parsed)


if __name__ == "__main__":
    main()
