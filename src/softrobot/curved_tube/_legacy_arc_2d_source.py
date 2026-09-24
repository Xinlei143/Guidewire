"""二维圆弧管内导丝准静态压缩：DER + 隐式 Euler/Newton + PI 软接触。

几何朝向与之前确认的图一致：导丝初始从左下角 (0, 0) 水平进入，沿
逆时针 90 度圆弧弯至右上端；圆心位于 (0, R)。管壁不是直线，而是与
中心线同心的内、外两条圆弧。管道两端不额外设置端盖接触。

第 0、1 个节点在整个仿真中固定。初始随机短脉冲结束后，最后一个
节点按用户给定的全局固定方向作分步位移控制；例如默认 (0,-1) 即
竖直向下压缩。该方向不会随圆弧切线改变。

Example (small reproducible smoke case)::

    python rod_soft_contact_PI_dynamic_quasi_static_2d_arc.py --no-show \
      --pulse-seed 7 --num-nodes 24 --tip-total-displacement 0.002 \
      --num-load-steps 2 --tip-ramp-steps 3 --max-settle-steps 40

全部单位统一为 SI：m、kg、s、N。运行时可通过
--tip-direction-x / --tip-direction-y 指定竖直或任意斜向压缩。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import functools
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


DTYPE = torch.float64  # 全部使用双精度，保证曲率、接触和 Newton 的精度。

# ============================================================================
# 用户参数区：通常只需修改这一段，命令行同名参数可覆盖这些默认值。
# ============================================================================
# --- 导丝材料与离散（沿用二维直管的物理尺度）---
TOTAL_LENGTH = 1.5             # 导丝初始总长度，m；此处是沿圆弧中心线的离散弦长总和。
DIAMETER = 1.0e-3                # 导丝直径，m。
MATERIAL_DENSITY = 7850.0        # 钢的密度，kg/m^3。
YOUNG_MODULUS = 200.0e9          # 杨氏模量，Pa。
DAMPING_RATIO = 1.0              # 按等效刚度估计的临界阻尼比。
NUM_NODES = 200                   # DER 节点数；越大越精细，但全局 Newton Jacobian 更耗时。

# --- 圆弧管几何 ---
ARC_ANGLE_DEG = 90.0             # 中心线圆弧转角，deg；90° 时起点切线水平、终点切线竖直。
# ARC_RADIUS 不设全局常量：未在命令行给定时由长度和转角自动计算。
TUBE_HALF_WIDTH = 50.0e-3        # 中心线到内/外实体壁的法向距离，m（即管腔半宽）。
SAFE_DISTANCE = 10.0e-3          # 距实体壁小于此值才进入 PI 保护层，m。

# --- 节点 PI 软接触：不做投影或硬碰撞，仅施加指向中心线的反力 ---
KP = 3500.0                      # 比例增益，N/m。
KI = 100.0                       # 积分增益，N/(m*s)；设为 0 即纯比例接触。
MAX_PI_FORCE = 20.0              # 单节点 PI 接触力上限，N，避免数值发散。
MAX_INTEGRAL_STATE = 5.0e-2      # 单节点积分状态上限，m*s，抗积分饱和。
INTEGRAL_LEAK_RATE = 5.0         # 节点离开接触区后积分量的泄漏速率，1/s。
CONTACT_SMOOTH_LENGTH = 2.0e-5   # softplus 接触切换平滑长度，m。
DIRECTION_EPSILON = 1.0e-6       # 圆弧中心线上平滑符号函数的正则化长度，m。

# --- 隐式动力学松弛与每个载荷步的平衡判据 ---
DT = 5.0e-3                      # 隐式 Euler 时间步，s；只用于加载步内部的动态松弛。
MAX_NEWTON_ITERS = 35            # 每一个隐式时间步允许的最大 Newton 迭代次数。
NEWTON_RESIDUAL_TOL = 1.0e-7     # 单个隐式步 Newton 力残差收敛阈值，N。
NEWTON_STEP_TOL = 1.0e-10        # Newton 位移修正量阈值，m。
SETTLE_SPEED_TOL = 1.0e-3        # 载荷步平衡时自由节点最大速度阈值，m/s。
SETTLE_RESIDUAL_TOL = 1.0e-6     # 平衡时 Newton 残差阈值，N。
CONTACT_TOLERANCE = 5.0e-4       # 平衡时保护层几何误差容限，m。
SETTLE_CONFIRM_STEPS = 4         # 上述条件连续满足的步数，防止瞬时误判平衡。
MAX_SETTLE_STEPS = 1600          # 单个载荷增量允许的最多保持/松弛步数。
# κ0=0 时初始圆弧不是平衡态。先在零内缩、两端固定条件下松弛，
# 得到“已装入管道但尚未压缩”的静力平衡构型。
PREBALANCE_ENABLED = True
RAMP_RETRY_SUBDIVISIONS = (2, 4)        # 仅在 Ki 已减为 0 后才启用的子步重试数。
KI_ADAPTIVE_REDUCTION_FACTOR = 0.5      # 当前载荷步失败时，Ki 的递减系数：Ki -> Ki/2。
KI_ADAPTIVE_ZERO_THRESHOLD = 1.0        # Ki 已降至该值以下时直接置 0，随后再启用 2/4 子步。
KP_CONTACT_GROW_FACTOR = 1.5            # 仅接触误差超限时，Kp 的自动放大系数。
KP_CONTACT_MAX_FACTOR = 8.0             # 有效 Kp 不超过用户初始 Kp 的该倍数。

# --- 末端位移控制：方向在命令行通过 tip-direction-x/y 指定，默认 (0,-1) ---
TIP_TOTAL_DISPLACEMENT = 120.0e-3 # 最后节点总位移大小，m。
NUM_LOAD_STEPS = 40              # 总位移分成的载荷步数。
TIP_RAMP_STEPS = 25              # 每个载荷步中用平滑 S 曲线施加位移的隐式步数。

# --- 初始随机扰动：在自由内部节点沿局部圆弧法向施加正/负短时脉冲 ---
PULSE_COUNT_MIN = 4              # 每次运行随机抽取脉冲节点数的下限。
PULSE_COUNT_MAX = 8              # 每次运行随机抽取脉冲节点数的上限。
PULSE_FORCE_MIN = 2.0e-2         # 每个脉冲力幅值下限，N。
PULSE_FORCE_MAX = 7.0e-2         # 每个脉冲力幅值上限，N。
PULSE_DURATION = 2.0e-2          # 脉冲持续时间，s；此阶段末端固定在初始圆弧终点。

# --- 导丝自然曲率 ---
# 本算例模拟“原本笔直的导丝被装入圆弧管道”。因此自然曲率严格取 κ0=0，
# 初始圆弧只是一种被管道和端部约束保持的初始构型，具有弯曲势能和恢复力。
# 这与“制造后天然 90° 预弯、初始圆弧零势能”的模型不同，但与 Wang et al.
# 的直梁-圆弧管基准、以及已有二维/三维直管导丝算法保持一致。
NATURAL_CURVATURE = 0.0

# --- 插入圆弧管道时的可选几何扰动（当前关闭）---
# 在 κ0=0 的直杆模型中，该项不是“零势能自然预弯”，而是人为指定的
# 初始装入构型偏差。默认 0，仅依赖脉冲和压缩路径选择屈曲分支。
INITIAL_NORMAL_IMPERFECTION_AMPLITUDE = 0.0     # 初始局部法向偏移最大幅值，m。
INITIAL_IMPERFECTION_MODE_COUNT = 3            # 正弦模态个数；3 对应交替内/外初始偏差。
INITIAL_IMPERFECTION_SEED = None                # None=每次随机；整数=可复现。
TIP_MOTION_MODE = "arc-follow"                   # 默认沿圆弧中心线回退；fixed-direction 可模拟固定竖直/斜向加载。


@dataclass
class NewtonInfo:
    converged: bool
    iterations: int
    residual_norm: float
    step_norm: float
    line_search_failures: int


def resolve_device(name: str) -> torch.device:
    """将 --device 的文字参数解析为 PyTorch 设备；auto 优先使用 CUDA。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return torch.device(name)


def rod_properties(length: float, diameter: float, density: float) -> tuple[float, float, float, float]:
    """由圆截面与材料参数计算总质量、EA、EI 和等效临界阻尼。"""
    area = math.pi * diameter**2 / 4.0
    mass = density * area * length
    ea = YOUNG_MODULUS * area
    ei = YOUNG_MODULUS * math.pi * diameter**4 / 64.0
    damping = 2.0 * DAMPING_RATIO * math.sqrt((3.0 * ei / length**3) * mass)
    return mass, ea, ei, damping


def arc_points(radius: float, angle_rad: float, count: int) -> np.ndarray:
    """生成中心线离散点。

    圆心固定为 C=(0,R)，参数方程为：
        r(theta) = (R sin(theta), R[1-cos(theta)])，theta ∈ [0, arc_angle]
    因而 theta=0 的位置为 (0,0)、切线方向为 +x；90° 时终点切线为 +y。
    """
    theta = np.linspace(0.0, angle_rad, count)
    return np.column_stack((radius * np.sin(theta), radius * (1.0 - np.cos(theta))))


def make_natural_precurved_shape(args: argparse.Namespace, radius: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """生成装入圆弧管道时的初始构型，可选地叠加多波法向偏差。

    缺陷沿初始圆弧的局部法向叠加，端点和前两个固定节点保持为零偏移。
    rest_lengths 从该构型计算，所以它没有初始拉伸；但本脚本的 κ0=0，
    因而曲线构型仍有弯曲势能，并非导丝的无应力自然形状。
    """
    radius = args.arc_radius if radius is None else radius
    base = arc_points(radius, args.arc_angle_rad, args.num_nodes)
    if args.initial_normal_imperfection_amplitude == 0.0:
        return base, np.zeros(args.num_nodes, dtype=np.float64)
    rng = np.random.default_rng(args.initial_imperfection_seed)
    # 用移位参数 u，使节点 0、1 和最后一个节点的缺陷严格为零：这些节点
    # 分别为夹持端与位移端，不应因“制造缺陷”改变位置。
    u = np.clip((np.arange(args.num_nodes, dtype=np.float64) - 1.0) / (args.num_nodes - 2), 0.0, 1.0)
    # Σ a_m sin(mπu)，天然在 u=0、1 为零，适合生成交替的局部内/外波。
    # 最高模态保证至少出现 mode_count 个交替的内/外波；较小的低阶随机成分
    # 只改变波形细节而不抹去符号交替。随机种子决定整体内外翻转与低阶细节。
    dominant_sign = rng.choice(np.array([-1.0, 1.0]))
    profile = dominant_sign * np.sin(args.initial_imperfection_mode_count * math.pi * u)
    if args.initial_imperfection_mode_count > 1:
        lower_modes = rng.uniform(-0.12, 0.12, size=args.initial_imperfection_mode_count - 1)
        profile += sum(coefficient * np.sin(mode * math.pi * u) for mode, coefficient in enumerate(lower_modes, start=1))
    peak = np.max(np.abs(profile))
    profile = np.zeros_like(profile) if peak <= 1.0e-15 else profile * (args.initial_normal_imperfection_amplitude / peak)
    profile[:2] = 0.0  # 前两个夹持节点必须与名义圆弧完全重合。
    profile[-1] = 0.0  # 位移端也从名义圆弧终点开始。
    normals = circle_normal_numpy(base, radius)
    return base + profile[:, None] * normals, profile


def initialize_rod(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按圆弧中心线建立无初始拉伸的导丝、质量、速度和 PI 积分状态。"""
    # 几何缺陷使弦长总和略大于名义圆弧；二分微调 R，使“带偏差装入的导丝”的
    # 离散总长度仍严格等于 --total-length，管道圆弧也同步采用该半径。
    if args.initial_normal_imperfection_amplitude > 0.0:
        requested_radius = args.arc_radius
        lower, upper = 1.0e-8, requested_radius
        for _ in range(60):
            midpoint = 0.5 * (lower + upper)
            candidate, _ = make_natural_precurved_shape(args, midpoint)
            candidate_length = np.linalg.norm(np.diff(candidate, axis=0), axis=1).sum()
            if candidate_length < args.total_length:
                lower = midpoint
            else:
                upper = midpoint
        args.arc_radius = 0.5 * (lower + upper)
    positions, imperfection_profile = make_natural_precurved_shape(args)
    # 每条离散 DER 边的当前弦长就是其静息长度，故初始伸长能为 0。
    rest = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    mass, _, _, _ = rod_properties(args.total_length, args.diameter, args.density)
    masses = np.full(args.num_nodes, mass / (args.num_nodes - 1), dtype=np.float64)
    masses[[0, -1]] *= 0.5  # 两端节点只拥有半段杆件质量。
    args.initial_imperfection_profile = imperfection_profile
    return positions, np.zeros_like(positions), rest, masses, np.zeros(args.num_nodes)


def circle_normal_numpy(points: np.ndarray, radius: float) -> np.ndarray:
    """返回从圆心向外的单位径向法向；外壁侧为正、内壁侧为负。"""
    rel = points - np.array([0.0, radius])
    norm = np.linalg.norm(rel, axis=1, keepdims=True)
    return rel / np.maximum(norm, 1.0e-14)


def make_random_normal_pulses(args: argparse.Namespace, initial_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """随机构造局部法向短脉冲，而不改变初始圆弧几何。

    前两个节点固定、最后一个节点留给之后的位移控制，因此脉冲仅施加在
    节点 2 至 num_nodes-2。每个脉冲的正负号独立随机：正向指向外圆壁，
    负向指向内圆壁。指定 --pulse-seed 后可完全复现实验。
    """
    rng = np.random.default_rng(args.pulse_seed)
    candidates = np.arange(2, args.num_nodes - 1, dtype=int)  # 不影响固定端和位移端。
    count = int(rng.integers(args.pulse_count_min, args.pulse_count_max + 1))
    nodes = np.sort(rng.choice(candidates, size=count, replace=False))
    signs = rng.choice(np.array([-1.0, 1.0]), size=count)
    magnitudes = rng.uniform(args.pulse_force_min, args.pulse_force_max, size=count)
    directions = signs[:, None] * circle_normal_numpy(initial_positions[nodes], args.arc_radius)
    profile = np.zeros_like(initial_positions)
    profile[nodes] = magnitudes[:, None] * directions
    return nodes, magnitudes, directions, profile


def discrete_curvature(positions: torch.Tensor) -> torch.Tensor:
    """返回每个内部节点的二维离散曲率 κ（长度为节点数减 2）。"""
    edges = positions[1:] - positions[:-1]
    lengths = torch.linalg.vector_norm(edges, dim=1).clamp_min(1.0e-12)
    tangents = edges / lengths[:, None]
    cross = tangents[:-1, 0] * tangents[1:, 1] - tangents[:-1, 1] * tangents[1:, 0]
    dot = torch.sum(tangents[:-1] * tangents[1:], dim=1)
    return 2.0 * cross / (1.0 + dot).clamp_min(1.0e-12)


def elastic_energy(
    positions: torch.Tensor, rest: torch.Tensor, ea: float, ei: float,
    reference_curvature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """二维离散弹性杆（DER）的拉伸能、曲率双法向弯曲能及总能量。"""
    edges = positions[1:] - positions[:-1]
    lengths = torch.linalg.vector_norm(edges, dim=1).clamp_min(1.0e-12)
    tangents = edges / lengths[:, None]
    # E_s = 1/2 EA Σ (l_i-l0_i)^2/l0_i
    stretch = 0.5 * ea * torch.sum((lengths - rest).square() / rest)
    # 弯曲参考态是笔直导丝：κ0=0。初始圆弧由管道和两端约束强制形成，
    # 因而具有弯曲势能并会产生恢复力；这正是直导丝塞入圆弧管的物理含义。
    kappa = discrete_curvature(positions)
    voronoi = 0.5 * (rest[:-1] + rest[1:])
    bend = 0.5 * ei * torch.sum((kappa - reference_curvature).square() / voronoi)
    return stretch, bend, stretch + bend


def local_bending_energy(
    points: torch.Tensor, rest_left: torch.Tensor, rest_right: torch.Tensor,
    reference_curvature: torch.Tensor, ei: float,
) -> torch.Tensor:
    """单个三节点 DER 顶点的二维弯曲能，专供局部 Hessian 批量组装。"""
    left = points[1] - points[0]
    right = points[2] - points[1]
    left_length = torch.linalg.vector_norm(left).clamp_min(1.0e-12)
    right_length = torch.linalg.vector_norm(right).clamp_min(1.0e-12)
    tangent_left = left / left_length
    tangent_right = right / right_length
    cross = tangent_left[0] * tangent_right[1] - tangent_left[1] * tangent_right[0]
    dot = torch.dot(tangent_left, tangent_right)
    kappa = 2.0 * cross / (1.0 + dot).clamp_min(1.0e-12)
    voronoi = 0.5 * (rest_left + rest_right)
    return 0.5 * ei * (kappa - reference_curvature).square() / voronoi


def stretch_grad_hessian_analytic(positions: torch.Tensor, rest: torch.Tensor, ea: float) -> tuple[torch.Tensor, torch.Tensor]:
    """解析计算所有两节点拉伸单元的梯度与 Hessian，避免全局 autograd Jacobian。"""
    delta = positions[1:] - positions[:-1]
    length = torch.linalg.vector_norm(delta, dim=1).clamp_min(1.0e-12)
    stiffness = ea / rest
    coeff = 1.0 - rest / length
    gradient_delta = stiffness[:, None] * coeff[:, None] * delta
    gradient = torch.cat((-gradient_delta, gradient_delta), dim=1)  # 每条边的 (x0,y0,x1,y1) 梯度。
    eye = torch.eye(2, dtype=positions.dtype, device=positions.device)
    local_k = stiffness[:, None, None] * (
        coeff[:, None, None] * eye
        + (rest / length**3)[:, None, None] * delta[:, :, None] * delta[:, None, :]
    )
    hessian = torch.empty((len(delta), 4, 4), dtype=positions.dtype, device=positions.device)
    hessian[:, :2, :2] = local_k
    hessian[:, :2, 2:] = -local_k
    hessian[:, 2:, :2] = -local_k
    hessian[:, 2:, 2:] = local_k
    return gradient, hessian


@dataclass(frozen=True)
class ArcBlockPlan:
    """圆弧二维问题的固定拓扑：节点块宽度为 2，耦合带宽为 ±2 节点。"""
    free_start: int
    free_end: int
    count: int


@functools.lru_cache(maxsize=8)
def get_block_plan(num_nodes: int, tip_constrained: bool) -> ArcBlockPlan:
    """缓存两种边界拓扑（末端自由/末端已位移约束），避免每个时间步重建索引。"""
    free_start = 2
    free_end = num_nodes - int(tip_constrained)
    return ArcBlockPlan(free_start, free_end, free_end - free_start)


@dataclass
class BlockPenta2D:
    """二维节点块（2×2）的五对角 Newton 系数矩阵。

    DER 拉伸只耦合相邻节点，弯曲只耦合相隔不超过两个节点，故完整
    Jacobian 天然只有这五条块带。CUDA 路径把相邻 2×2 块配对为 4×4
    块三对角系统，再用并行循环消元（PCR）求解。
    """
    diagonal: torch.Tensor
    upper1: torch.Tensor
    upper2: torch.Tensor
    lower1: torch.Tensor
    lower2: torch.Tensor

    @classmethod
    def zeros(cls, blocks: int, dtype: torch.dtype, device: torch.device) -> "BlockPenta2D":
        zero = lambda n: torch.zeros((max(n, 0), 2, 2), dtype=dtype, device=device)
        return cls(zero(blocks), zero(blocks - 1), zero(blocks - 2), zero(blocks - 1), zero(blocks - 2))

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        x = vector.reshape(-1, 2)
        result = torch.bmm(self.diagonal, x.unsqueeze(-1)).squeeze(-1)
        if len(x) > 1:
            result[:-1] += torch.bmm(self.upper1, x[1:].unsqueeze(-1)).squeeze(-1)
            result[1:] += torch.bmm(self.lower1, x[:-1].unsqueeze(-1)).squeeze(-1)
        if len(x) > 2:
            result[:-2] += torch.bmm(self.upper2, x[2:].unsqueeze(-1)).squeeze(-1)
            result[2:] += torch.bmm(self.lower2, x[:-2].unsqueeze(-1)).squeeze(-1)
        return result.reshape(-1)

    def dense(self) -> torch.Tensor:
        blocks = len(self.diagonal)
        value = torch.zeros((2 * blocks, 2 * blocks), dtype=self.diagonal.dtype, device=self.diagonal.device)
        for index in range(blocks):
            value[2 * index:2 * index + 2, 2 * index:2 * index + 2] = self.diagonal[index]
        for index in range(blocks - 1):
            value[2 * index:2 * index + 2, 2 * (index + 1):2 * (index + 2)] = self.upper1[index]
            value[2 * (index + 1):2 * (index + 2), 2 * index:2 * index + 2] = self.lower1[index]
        for index in range(blocks - 2):
            value[2 * index:2 * index + 2, 2 * (index + 2):2 * (index + 3)] = self.upper2[index]
            value[2 * (index + 2):2 * (index + 3), 2 * index:2 * index + 2] = self.lower2[index]
        return value

    def _solve_cpu(self, rhs: torch.Tensor) -> torch.Tensor:
        """CPU 用串行块 LU；250 节点仅需要约 250 个很小的 2×2 求解。"""
        diagonal, upper1, upper2 = self.diagonal.clone(), self.upper1.clone(), self.upper2.clone()
        lower1, lower2 = self.lower1.clone(), self.lower2.clone()
        vector = rhs.reshape(-1, 2).clone()
        for row in range(len(diagonal)):
            if row + 1 < len(diagonal):
                factor = torch.linalg.solve(diagonal[row].T, lower1[row].T).T
                diagonal[row + 1] -= factor @ upper1[row]
                vector[row + 1] -= factor @ vector[row]
                if row + 2 < len(diagonal): upper1[row + 1] -= factor @ upper2[row]
            if row + 2 < len(diagonal):
                factor = torch.linalg.solve(diagonal[row].T, lower2[row].T).T
                lower1[row + 1] -= factor @ upper1[row]
                diagonal[row + 2] -= factor @ upper2[row]
                vector[row + 2] -= factor @ vector[row]
        solution = torch.empty_like(vector)
        for row in range(len(diagonal) - 1, -1, -1):
            value = vector[row]
            if row + 1 < len(diagonal): value -= upper1[row] @ solution[row + 1]
            if row + 2 < len(diagonal): value -= upper2[row] @ solution[row + 2]
            solution[row] = torch.linalg.solve(diagonal[row], value)
        return solution.reshape(-1)

    def _solve_cuda_pcr(self, rhs: torch.Tensor) -> torch.Tensor:
        """CUDA 并行循环消元：2×2 块配对成 4×4 块三对角系统。"""
        blocks = len(self.diagonal)
        groups = (blocks + 1) // 2
        dtype, device = self.diagonal.dtype, self.diagonal.device
        left = torch.zeros((groups, 4, 4), dtype=dtype, device=device)
        centre, right = torch.zeros_like(left), torch.zeros_like(left)
        vector = torch.zeros((groups, 4), dtype=dtype, device=device)
        even_count, odd_count = len(self.diagonal[0::2]), len(self.diagonal[1::2])
        centre[:even_count, :2, :2] = self.diagonal[0::2]
        vector[:even_count, :2] = rhs.reshape(blocks, 2)[0::2]
        if odd_count:
            centre[:odd_count, 2:, 2:] = self.diagonal[1::2]
            centre[:odd_count, :2, 2:] = self.upper1[0::2][:odd_count]
            centre[:odd_count, 2:, :2] = self.lower1[0::2][:odd_count]
            vector[:odd_count, 2:] = rhs.reshape(blocks, 2)[1::2]
        if blocks % 2: centre[-1, 2:, 2:] = torch.eye(2, dtype=dtype, device=device)
        if groups > 1:
            left[1:, :2, :2] = self.lower2[0::2]
            left[1:, :2, 2:] = self.lower1[1::2]
            if len(self.lower2[1::2]): left[1:1 + len(self.lower2[1::2]), 2:, 2:] = self.lower2[1::2]
            right[:-1, :2, :2] = self.upper2[0::2]
            right[:-1, 2:, :2] = self.upper1[1::2]
            if len(self.upper2[1::2]): right[:len(self.upper2[1::2]), 2:, 2:] = self.upper2[1::2]
        rows, stride = torch.arange(groups, device=device), 1
        while stride < groups:
            new_left, new_right = torch.zeros_like(left), torch.zeros_like(right)
            new_centre, new_vector = centre.clone(), vector.clone()
            has_left, has_right = rows >= stride, rows + stride < groups
            if bool(torch.any(has_left)):
                active, neighbour = rows[has_left], rows[has_left] - stride
                alpha = torch.linalg.solve(centre[neighbour].transpose(1, 2), left[active].transpose(1, 2)).transpose(1, 2)
                new_centre[active] -= alpha @ right[neighbour]
                new_vector[active] -= (alpha @ vector[neighbour, :, None]).squeeze(-1)
                new_left[active] = -alpha @ left[neighbour]
            if bool(torch.any(has_right)):
                active, neighbour = rows[has_right], rows[has_right] + stride
                gamma = torch.linalg.solve(centre[neighbour].transpose(1, 2), right[active].transpose(1, 2)).transpose(1, 2)
                new_centre[active] -= gamma @ left[neighbour]
                new_vector[active] -= (gamma @ vector[neighbour, :, None]).squeeze(-1)
                new_right[active] = -gamma @ right[neighbour]
            left, centre, right, vector = new_left, new_centre, new_right, new_vector
            stride *= 2
        groups_solution = torch.linalg.solve(centre, vector)
        solution = torch.empty((blocks, 2), dtype=dtype, device=device)
        solution[0::2] = groups_solution[:even_count, :2]
        if odd_count: solution[1::2] = groups_solution[:odd_count, 2:]
        return solution.reshape(-1)

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        """优先带状求解，数值退化时才退回稠密解，以保证可靠性。"""
        try:
            result = self._solve_cuda_pcr(rhs) if self.diagonal.is_cuda else self._solve_cpu(rhs)
            relative = torch.linalg.vector_norm(self.matvec(result) - rhs) / torch.clamp(torch.linalg.vector_norm(rhs), min=1.0e-30)
            if self.diagonal.is_cuda:
                for _ in range(2):
                    if bool(relative <= 1.0e-8): break
                    result += self._solve_cuda_pcr(rhs - self.matvec(result))
                    relative = torch.linalg.vector_norm(self.matvec(result) - rhs) / torch.clamp(torch.linalg.vector_norm(rhs), min=1.0e-30)
            if not bool(torch.isfinite(result).all()) or not bool(relative <= 1.0e-8): raise RuntimeError("inaccurate band solution")
            return result
        except RuntimeError:
            return torch.linalg.solve(self.dense(), rhs)


def contact_quantities(
    positions: torch.Tensor, old_integrals: torch.Tensor, previous_errors: torch.Tensor,
    args: argparse.Namespace, dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算同心圆弧内、外壁的可微节点 PI 软接触力。

    对节点 x，rho=||x-C||，有符号法向偏移 d=rho-R：d>0 位于外壁侧，
    d<0 位于内壁侧。到实体壁的净间隙为 h-|d|。当 |d| 超过
    h-safe_distance 时进入保护层，接触力沿 -sign(d)*n 推向圆心线。
    """
    center = torch.tensor([0.0, args.arc_radius], dtype=positions.dtype, device=positions.device)
    radial = positions - center
    rho = torch.linalg.vector_norm(radial, dim=1).clamp_min(1.0e-12)
    normal = radial / rho[:, None]                 # n：由圆心向外的局部单位法向。
    signed_distance = rho - args.arc_radius        # d：相对中心线的有符号法向偏移。
    clearance = args.tube_half_width - torch.abs(signed_distance)  # 到实体壁的有符号间隙。
    z = torch.abs(signed_distance) - (args.tube_half_width - args.safe_distance)  # z>0 表示已进入保护层。
    beta = 1.0 / args.contact_smooth_length
    error = F.softplus(z, beta=beta)
    weight = torch.sigmoid(beta * z)
    # 进入接触层：I_{n+1}=I_n+e dt；离开后按指数规律泄漏，不突然清零。
    candidate = weight * (old_integrals + error * dt) + (1.0 - weight) * old_integrals * math.exp(-args.integral_leak_rate * dt)
    candidate = torch.clamp(candidate, 0.0, args.max_integral_state)
    # ki_active 允许困难载荷步临时减小积分增益；未设置时等于用户的 Ki。
    ki = getattr(args, "ki_active", args.Ki)
    kp = getattr(args, "kp_active", args.Kp)
    unsaturated = kp * error + ki * candidate
    # Anti-windup：力饱和或误差正在减小时，不再继续累计积分项。
    blocked = (ki > 0.0) & (unsaturated >= args.max_pi_force) & (candidate > old_integrals)
    decreasing = error < previous_errors
    blocked |= (ki > 0.0) & decreasing & (ki * old_integrals > kp * error) & (candidate > old_integrals)
    integrals = torch.where(blocked, old_integrals, candidate)
    magnitude = torch.clamp(kp * error + ki * integrals, 0.0, args.max_pi_force)
    # 平滑 sign(d)：中心线上得到严格零接触力，避免不可导符号函数。
    direction = -signed_distance[:, None] * normal / torch.sqrt(signed_distance.square()[:, None] + args.direction_epsilon**2)
    force = magnitude[:, None] * direction
    active = z > 0.0
    return force, integrals, active, weight, error, clearance


def assemble_elastic_banded(
    positions: torch.Tensor, rest: torch.Tensor, reference_curvature: torch.Tensor,
    plan: ArcBlockPlan, ea: float, ei: float,
) -> tuple[torch.Tensor, BlockPenta2D]:
    """批量计算局部 DER 导数，并只装配五条 2×2 块带（v3 同一加速思路）。"""
    gradient = torch.zeros((plan.count, 2), dtype=positions.dtype, device=positions.device)
    system = BlockPenta2D.zeros(plan.count, positions.dtype, positions.device)

    def scatter(local_gradient: torch.Tensor, local_hessian: torch.Tensor, first_node: int) -> None:
        local_nodes = local_gradient.shape[1]
        element_count = local_gradient.shape[0]
        for local_row in range(local_nodes):
            nodes = first_node + torch.arange(element_count, device=positions.device) + local_row
            valid_row = (nodes >= plan.free_start) & (nodes < plan.free_end)
            if bool(torch.any(valid_row)):
                gradient.index_add_(0, (nodes[valid_row] - plan.free_start), local_gradient[valid_row, local_row])
            for local_col in range(local_nodes):
                offset = local_col - local_row
                if offset < -2 or offset > 2: continue
                column_nodes = first_node + torch.arange(element_count, device=positions.device) + local_col
                valid = valid_row & (column_nodes >= plan.free_start) & (column_nodes < plan.free_end)
                if not bool(torch.any(valid)): continue
                # 对角线上目标为行节点；下带的存储下标按列节点，与 matvec 的定义一致。
                destination = nodes[valid] - plan.free_start if offset >= 0 else column_nodes[valid] - plan.free_start
                band = {0: system.diagonal, 1: system.upper1, -1: system.lower1, 2: system.upper2, -2: system.lower2}[offset]
                band.index_add_(0, destination, local_hessian[valid, local_row, :, local_col, :])

    stretch_gradient, stretch_hessian = stretch_grad_hessian_analytic(positions, rest, ea)
    scatter(stretch_gradient.reshape(-1, 2, 2), stretch_hessian.reshape(-1, 2, 2, 2, 2), 0)

    def bend_gradient_aux(flat: torch.Tensor, left: torch.Tensor, right: torch.Tensor, kappa0: torch.Tensor):
        value = torch.func.grad(lambda z: local_bending_energy(z.reshape(3, 2), left, right, kappa0, ei))(flat)
        return value, value

    bend_dofs = torch.cat((positions[:-2], positions[1:-1], positions[2:]), dim=1)
    bend_hessian, bend_gradient = torch.func.vmap(torch.func.jacrev(bend_gradient_aux, argnums=0, has_aux=True))(
        bend_dofs, rest[:-1], rest[1:], reference_curvature
    )
    scatter(bend_gradient.reshape(-1, 3, 2), bend_hessian.reshape(-1, 3, 2, 3, 2), 0)
    return gradient.reshape(-1), system


def assemble_contact_banded(positions: torch.Tensor, integrals: torch.Tensor, previous_errors: torch.Tensor, args: argparse.Namespace, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """一次 vmap 得到所有节点 PI 力与 2×2 接触 Jacobian 对角块。"""
    def force_aux(point: torch.Tensor, old_integral: torch.Tensor, previous_error: torch.Tensor):
        value = contact_quantities(point[None], old_integral[None], previous_error[None], args, dt)[0][0]
        return value, value
    blocks, forces = torch.func.vmap(torch.func.jacrev(force_aux, argnums=0, has_aux=True))(positions, integrals, previous_errors)
    return forces, blocks


def unpack_free(q: torch.Tensor, plan: ArcBlockPlan, prescribed: torch.Tensor) -> torch.Tensor:
    """将连续自由节点块 q 写回完整二维节点阵列。"""
    result = prescribed.clone()
    result[plan.free_start:plan.free_end] = q.reshape(plan.count, 2)
    return result


def residual_banded(q: torch.Tensor, old_p: torch.Tensor, old_v: torch.Tensor, rest: torch.Tensor, reference_curvature: torch.Tensor, masses: torch.Tensor, integrals: torch.Tensor, previous_errors: torch.Tensor, pulse: torch.Tensor, prescribed: torch.Tensor, plan: ArcBlockPlan, args: argparse.Namespace, dt: float, with_jacobian: bool) -> tuple[torch.Tensor, BlockPenta2D | None, torch.Tensor]:
    """计算 Newton 残差；需要 Jacobian 时装配二维块五对角系统。"""
    p = unpack_free(q, plan, prescribed)
    if with_jacobian:
        gradient, system = assemble_elastic_banded(p, rest, reference_curvature, plan, args.EA, args.EI)
        contact_force, contact_blocks = assemble_contact_banded(p, integrals, previous_errors, args, dt)
    else:
        p_grad = p.detach().clone().requires_grad_(True)
        _, _, energy = elastic_energy(p_grad, rest, args.EA, args.EI, reference_curvature)
        gradient = torch.autograd.grad(energy, p_grad)[0][plan.free_start:plan.free_end].reshape(-1)
        contact_force = contact_quantities(p, integrals, previous_errors, args, dt)[0]
        system = None
    old_q = old_p[plan.free_start:plan.free_end].reshape(-1)
    old_velocity = old_v[plan.free_start:plan.free_end].reshape(-1)
    mass = masses[plan.free_start:plan.free_end, None].expand(-1, 2).reshape(-1)
    residual = mass * (q - old_q - dt * old_velocity) / dt**2 + gradient + args.damping * (q - old_q) / dt - contact_force[plan.free_start:plan.free_end].reshape(-1) - pulse[plan.free_start:plan.free_end].reshape(-1)
    if with_jacobian:
        diagonal = (mass / dt**2 + args.damping / dt).reshape(-1, 2)
        system.diagonal.diagonal(dim1=1, dim2=2).add_(diagonal)
        system.diagonal.add_(-contact_blocks[plan.free_start:plan.free_end])
    return residual, system, p


def geometric_protection_error(positions: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """返回精确几何保护层误差，仅用于判断某个加载步是否已平衡。"""
    center = torch.tensor([0.0, args.arc_radius], dtype=positions.dtype, device=positions.device)
    signed = torch.linalg.vector_norm(positions - center, dim=1) - args.arc_radius
    return torch.clamp(torch.abs(signed) - (args.tube_half_width - args.safe_distance), min=0.0)


def prescribed_state(initial: torch.Tensor, tip_displacement: float, constrained_tip: bool, direction: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    """建立本隐式步的位移边界。

    节点 0、1 的 x/y 始终固定。加载阶段最后一个节点两个坐标均固定为
    x_tip=x_tip0+s*d_hat，因而它严格沿指定的全局直线运动，垂直分量不自由。
    """
    mask = torch.zeros_like(initial, dtype=torch.bool)
    mask[:2] = True  # 下端两个节点完全固定。
    if constrained_tip:
        mask[-1] = True
    values = initial.clone()
    if constrained_tip:
        if args.tip_motion_mode == "fixed-direction":
            # 用户指定的全局直线位移，例如 (0,-1) 是竖直下压。
            values[-1] = initial[-1] + tip_displacement * direction
        else:
            # 沿初始圆弧中心线向入口回退 s：θ=θ_end-s/R。它不会人为把末端
            # 推到外圆侧，适合研究预弯导丝压缩触发的内/外交替屈曲。
            theta = max(0.0, args.arc_angle_rad - tip_displacement / args.arc_radius)
            values[-1] = torch.as_tensor(
                [args.arc_radius * math.sin(theta), args.arc_radius * (1.0 - math.cos(theta))],
                dtype=initial.dtype, device=initial.device,
            )
    return mask, values


def implicit_euler_step(
    old_p: torch.Tensor, old_v: torch.Tensor, rest: torch.Tensor, reference_curvature: torch.Tensor, masses: torch.Tensor,
    old_integrals: torch.Tensor, previous_errors: torch.Tensor, pulse: torch.Tensor,
    mask: torch.Tensor, prescribed: torch.Tensor, args: argparse.Namespace, dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, NewtonInfo]:
    """完成一个仅对自由自由度求解的隐式 Euler / Newton 动力学步。

    对每个自由坐标求解残差：
        M(x_{n+1}-x_n-dt*v_n)/dt^2 + C(x_{n+1}-x_n)/dt - F(x_{n+1}) = 0。
    其中 F 包括 DER 弹性内力、圆弧壁 PI 力和（仅初始阶段存在的）随机脉冲。
    """
    # 两种固定拓扑与 v3 一样预先缓存；不再对约 500 个自由度构造稠密 Jacobian。
    tip_constrained = bool(torch.all(mask[-1]).item())
    plan = get_block_plan(len(old_p), tip_constrained)
    z = (old_p[plan.free_start:plan.free_end] + dt * old_v[plan.free_start:plan.free_end]).reshape(-1).detach()
    failures, residual_norm, step_norm, converged = 0, math.inf, math.inf, False
    for iteration in range(1, args.max_newton_iters + 1):
        current, _, _ = residual_banded(z, old_p, old_v, rest, reference_curvature, masses, old_integrals, previous_errors, pulse, prescribed, plan, args, dt, False)
        residual_norm = float(torch.linalg.vector_norm(current).item())
        if residual_norm < args.residual_tol:
            converged, step_norm = True, 0.0
            break
        residual, system, _ = residual_banded(z, old_p, old_v, rest, reference_curvature, masses, old_integrals, previous_errors, pulse, prescribed, plan, args, dt, True)
        try:
            delta = system.solve(-residual)
        except RuntimeError:
            failures += 1
            delta = torch.linalg.solve(system.dense(), -residual)
        step_norm = float(torch.linalg.vector_norm(delta).item())
        base_norm, accepted = residual_norm, False
        # 保留 v3 同类回溯线搜索；试算阶段只计算残差，不构造 Jacobian。
        for exponent in range(10):
            candidate = z + (0.5**exponent) * delta
            candidate_residual, _, _ = residual_banded(candidate, old_p, old_v, rest, reference_curvature, masses, old_integrals, previous_errors, pulse, prescribed, plan, args, dt, False)
            candidate_norm = float(torch.linalg.vector_norm(candidate_residual).item())
            if np.isfinite(candidate_norm) and candidate_norm < base_norm:
                z, residual_norm, accepted = candidate.detach(), candidate_norm, True
                break
        if not accepted:
            failures += 1
            z = (z + 0.05 * delta).detach()
        if step_norm < args.step_tol:
            converged = residual_norm < 10.0 * args.residual_tol
            break
    p_new = unpack_free(z, plan, prescribed).detach()
    v_new = ((p_new - old_p) / dt).detach()
    v_new[mask] = 0.0
    force, integrals, active, weight, errors, _ = contact_quantities(p_new, old_integrals, previous_errors, args, dt)
    return p_new, v_new, integrals.detach(), force.detach(), active.detach(), weight.detach(), errors.detach(), NewtonInfo(converged, iteration, residual_norm, step_norm, failures)


def check_equilibrium(velocities: torch.Tensor, mask: torch.Tensor, info: NewtonInfo, errors: torch.Tensor, args: argparse.Namespace, prior: int) -> tuple[bool, float, float, int]:
    """检查自由节点速度、Newton 残差、保护层误差是否连续满足平衡条件。"""
    free_nodes = torch.any(~mask, dim=1)
    speed = float(torch.linalg.vector_norm(velocities[free_nodes], dim=1).max().item())
    protection = float(errors.max().item())
    ok = speed < args.settle_speed_tol and info.residual_norm < args.settle_residual_tol and protection < args.contact_tolerance
    count = prior + 1 if ok else 0
    return count >= args.settle_confirm_steps, speed, protection, count


def append_state(history: dict[str, list], p: torch.Tensor, v: torch.Tensor, pi: torch.Tensor, integ: torch.Tensor, active: torch.Tensor, info: NewtonInfo, load_step: int, displacement: float, time_value: float, errors: torch.Tensor) -> None:
    """向 NPZ 历史字典写入一个完整状态帧，供后处理与绘图复现。"""
    history["time"].append(time_value)
    history["load_step"].append(load_step)
    history["tip_displacement"].append(displacement)
    history["positions"].append(p.cpu().numpy().copy())
    history["velocities"].append(v.cpu().numpy().copy())
    history["pi_forces"].append(pi.cpu().numpy().copy())
    history["pi_integrals"].append(integ.cpu().numpy().copy())
    history["contact_active"].append(active.cpu().numpy().copy())
    history["max_protection_error"].append(float(errors.max().item()))
    history["max_pi_force"].append(float(torch.linalg.vector_norm(pi, dim=1).max().item()))
    history["newton_iterations"].append(info.iterations)
    history["newton_residual"].append(info.residual_norm)


def snapshot_state(p: torch.Tensor, v: torch.Tensor, integrals: torch.Tensor, previous_errors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """保存一个已接受平衡态；失败子步会从此状态无损回滚。"""
    return p.clone(), v.clone(), integrals.clone(), previous_errors.clone()


def try_load_increment(
    state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], initial: torch.Tensor,
    rest: torch.Tensor, reference_curvature: torch.Tensor, masses: torch.Tensor, direction: torch.Tensor, start_displacement: float,
    end_displacement: float, subdivision_count: int, load_step: int, time_value: float,
    args: argparse.Namespace,
) -> tuple[bool, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float, list[tuple], dict]:
    """从一个已收敛状态尝试完成一个位移增量。

    增量先使用 C1 平滑 ramp，随后保持末端松弛。该函数不写入全局历史，
    所以失败时丢弃中间状态即可实现事务式回滚。
    """
    p, v, integrals, previous_errors = (value.clone() for value in state)
    staged: list[tuple] = []
    last = {"speed": math.inf, "protection": math.inf, "residual": math.inf, "relax": 0}
    for substep in range(1, subdivision_count + 1):
        sub_start = start_displacement + (end_displacement - start_displacement) * (substep - 1) / subdivision_count
        sub_end = start_displacement + (end_displacement - start_displacement) * substep / subdivision_count
        # 子步内部仍用 smoothstep 平滑移动末端，避免回滚重试时引入加载冲击。
        for ramp in range(1, args.tip_ramp_steps + 1):
            fraction = ramp / args.tip_ramp_steps
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            displacement = sub_start + (sub_end - sub_start) * smooth
            mask, values = prescribed_state(initial, displacement, True, direction, args)
            p, v, integrals, pi, active, _weight, previous_errors, info = implicit_euler_step(
                p, v, rest, reference_curvature, masses, integrals, previous_errors, torch.zeros_like(p), mask, values, args, args.dt
            )
            time_value += args.dt
            errors = geometric_protection_error(p, args)
            staged.append((p, v, pi, integrals, active, info, load_step, displacement, time_value, errors))
            # Ramp 阶段是有意施加位移的瞬态过程，单个隐式步 Newton 未达到
            # 严格容限并不等于载荷步失败；只在数值非有限时立刻回滚，随后由
            # 持位松弛阶段的平衡判据决定该子步是否真正接受。
            if not torch.isfinite(p).all() or not torch.isfinite(v).all() or not math.isfinite(info.residual_norm):
                last.update(speed=math.inf, protection=float(errors.max().item()), residual=info.residual_norm)
                return False, state, time_value, staged, last
        stable_count = 0
        for relax in range(1, args.max_settle_steps + 1):
            mask, values = prescribed_state(initial, sub_end, True, direction, args)
            p, v, integrals, pi, active, _weight, previous_errors, info = implicit_euler_step(
                p, v, rest, reference_curvature, masses, integrals, previous_errors, torch.zeros_like(p), mask, values, args, args.dt
            )
            time_value += args.dt
            errors = geometric_protection_error(p, args)
            settled, speed, protection, stable_count = check_equilibrium(v, mask, info, errors, args, stable_count)
            last.update(speed=speed, protection=protection, residual=info.residual_norm, relax=relax)
            if relax % args.history_save_interval == 0 or settled:
                staged.append((p, v, pi, integrals, active, info, load_step, sub_end, time_value, errors))
            if settled:
                break
        else:
            return False, state, time_value, staged, last
    return True, (p, v, integrals, previous_errors), time_value, staged, last


def prebalance_zero_displacement(
    p: torch.Tensor, v: torch.Tensor, rest: torch.Tensor, reference_curvature: torch.Tensor,
    masses: torch.Tensor, integrals: torch.Tensor, previous_errors: torch.Tensor,
    initial: torch.Tensor, direction: torch.Tensor, time_value: float,
    history: dict[str, list], args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """在零内缩下求初始静力平衡。

    这是直杆自然态（κ0=0）圆弧管算例特有的准备阶段：节点 0、1 与末端
    均固定在初始圆弧位置，且不施加随机脉冲或末端压缩。导丝仍保持相同
    rest_lengths；它只能通过弯曲、接触及允许的极小轴向弹性应变寻找平衡。
    历史中以 load_step=-1 标识此阶段，方便和后续脉冲/压缩分开观察。
    """
    if not args.prebalance:
        return p, v, integrals, previous_errors, time_value

    print("[PREBALANCE] zero tip displacement; relaxing the straight-rod restoring force")
    mask, values = prescribed_state(initial, 0.0, True, direction, args)
    stable_count = 0
    last_speed, last_residual, last_protection = math.inf, math.inf, math.inf
    for relax in range(1, args.prebalance_max_steps + 1):
        p, v, integrals, pi, active, _weight, previous_errors, info = implicit_euler_step(
            p, v, rest, reference_curvature, masses, integrals, previous_errors,
            torch.zeros_like(p), mask, values, args, args.dt,
        )
        time_value += args.dt
        errors = geometric_protection_error(p, args)
        settled, last_speed, last_protection, stable_count = check_equilibrium(
            v, mask, info, errors, args, stable_count,
        )
        last_residual = info.residual_norm
        if relax % args.history_save_interval == 0 or settled:
            append_state(history, p, v, pi, integrals, active, info, -1, 0.0, time_value, errors)
        if settled:
            print(
                f"  settled; relax={relax}, speed={last_speed:.2e} m/s, "
                f"residual={last_residual:.2e} N, protection error={last_protection * 1e3:.4f} mm"
            )
            return p, v, integrals, previous_errors, time_value

    raise RuntimeError(
        f"prebalance did not settle in {args.prebalance_max_steps} steps. "
        f"Last: speed={last_speed:.3e} m/s, Newton residual={last_residual:.3e} N, "
        f"protection error={last_protection * 1e3:.4f} mm. "
        "Increase --prebalance-max-steps, reduce --dt, or inspect the initial tube/rod parameters."
    )


def run_simulation(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """依次执行：零内缩预平衡 → 随机脉冲 → 分级位移加载与动态松弛。"""
    np_p, np_v, np_rest, np_mass, np_int = initialize_rod(args)
    device = args.torch_device
    p = torch.as_tensor(np_p, dtype=DTYPE, device=device)
    v = torch.as_tensor(np_v, dtype=DTYPE, device=device)
    rest = torch.as_tensor(np_rest, dtype=DTYPE, device=device)
    # 本模型的导丝自然状态是直杆，初始圆弧不是无应力参考形状。
    # 所有内节点 κ0=0，从而初始圆弧有弯曲势能和恢复力。
    reference_curvature = torch.full((args.num_nodes - 2,), NATURAL_CURVATURE, dtype=DTYPE, device=device)
    masses = torch.as_tensor(np_mass, dtype=DTYPE, device=device)
    integrals = torch.as_tensor(np_int, dtype=DTYPE, device=device)
    initial = p.clone()
    direction = torch.as_tensor(args.tip_direction, dtype=DTYPE, device=device)
    nodes, pulse_magnitudes, pulse_directions, pulse_np = make_random_normal_pulses(args, np_p)
    pulse = torch.as_tensor(pulse_np, dtype=DTYPE, device=device)
    previous_errors = torch.zeros(args.num_nodes, dtype=DTYPE, device=device)
    args.ki_active = args.Ki  # 困难载荷步中可能由自动重试机制临时下调。
    args.kp_active = args.Kp  # 仅接触保护层误差超限时，自动提高而不改变 Ki。
    zero = torch.zeros_like(p)
    history: dict[str, list] = {key: [] for key in (
        "time", "load_step", "tip_displacement", "positions", "velocities", "pi_forces", "pi_integrals", "contact_active", "max_protection_error", "max_pi_force", "newton_iterations", "newton_residual"
    )}
    initial_info = NewtonInfo(True, 0, 0.0, 0.0, 0)
    append_state(history, p, v, zero, integrals, torch.zeros(args.num_nodes, dtype=torch.bool, device=device), initial_info, 0, 0.0, 0.0, previous_errors)
    time_value = 0.0
    print(f"2-D circular arc: R={args.arc_radius:.6g} m, angle={args.arc_angle_deg:g} deg, tube half-width={args.tube_half_width:.6g} m")
    print(f"random normal pulses: seed={args.pulse_seed}, nodes={nodes.tolist()}, magnitudes_N={np.round(pulse_magnitudes, 6).tolist()}")

    # 阶段 0：末端仍在初始位置，先释放“直杆被弯成圆弧”带来的恢复力。
    # 这一步结束后才生成可解释的零内缩起点；它不会修改导丝的总静息长度。
    p, v, integrals, previous_errors, time_value = prebalance_zero_displacement(
        p, v, rest, reference_curvature, masses, integrals, previous_errors,
        initial, direction, time_value, history, args,
    )
    prebalanced_positions = p.detach().cpu().numpy().copy()

    # 阶段 1：在预平衡构型上施加随机法向短脉冲，末端保持在初始圆弧终点。
    # pulse_elapsed 独立计时；time_value 已包含预平衡时间，不能拿来判断脉冲持续时间。
    pulse_elapsed = 0.0
    while pulse_elapsed < args.pulse_duration - 1.0e-14:
        step = min(args.dt, args.pulse_duration - pulse_elapsed)
        # Wang 论文的右端是沿中心线移动的铰接端；在脉冲阶段它应留在初始
        # 圆弧终点，而不能自由回弹，否则尚未开始压缩就已改变基准问题。
        mask, values = prescribed_state(initial, 0.0, True, direction, args)
        p, v, integrals, pi, active, _weight, previous_errors, info = implicit_euler_step(p, v, rest, reference_curvature, masses, integrals, previous_errors, pulse, mask, values, args, step)
        time_value += step
        pulse_elapsed += step
        append_state(history, p, v, pi, integrals, active, info, 0, 0.0, time_value, geometric_protection_error(p, args))
        if not info.converged:
            print(f"warning: pulse Newton step residual={info.residual_norm:.3e}")

    for load_step in range(1, args.num_load_steps + 1):
        target = load_step * args.tip_displacement_step
        previous = (load_step - 1) * args.tip_displacement_step
        print(f"[LOAD {load_step:02d}/{args.num_load_steps:02d}] target displacement={target * 1e3:.3f} mm")
        # 困难载荷步策略：
        # - 若速度和残差已收敛、仅保护层误差稍超限，则提高 Kp；
        # - 真正的动态/积分振荡型失败，才执行 Ki -> Ki/2；
        # - Ki=0 后仍失败，最后才用 2/4 个子步。
        checkpoint = snapshot_state(p, v, integrals, previous_errors)
        accepted = False
        last_diagnostic: dict = {}
        while not accepted:
            subdivision_options = (1,) if args.ki_active > 0.0 else RAMP_RETRY_SUBDIVISIONS
            for subdivision_count in subdivision_options:
                success, state, trial_time, staged, diagnostic = try_load_increment(
                    checkpoint, initial, rest, reference_curvature, masses, direction, previous, target, subdivision_count,
                    load_step, time_value, args,
                )
                last_diagnostic = diagnostic
                if success:
                    p, v, integrals, previous_errors = state
                    time_value = trial_time
                    for record in staged:
                        append_state(history, *record)
                    print(
                        f"  settled; subdivisions={subdivision_count}, relax={diagnostic['relax']}, "
                        f"speed={diagnostic['speed']:.2e} m/s, protection error={diagnostic['protection'] * 1e3:.4f} mm, "
                        f"Kp_eff={args.kp_active:g}, Ki_eff={args.ki_active:g}"
                    )
                    accepted = True
                    break
                print(
                    f"  retry: Kp_eff={args.kp_active:g}, Ki_eff={args.ki_active:g}, subdivisions={subdivision_count} failed; v={diagnostic['speed']:.2e} m/s, "
                    f"residual={diagnostic['residual']:.2e} N, protection={diagnostic['protection'] * 1e3:.4f} mm"
                )
            if accepted:
                break
            only_contact_error = (
                last_diagnostic.get("speed", math.inf) < args.settle_speed_tol
                and last_diagnostic.get("residual", math.inf) < args.settle_residual_tol
                and last_diagnostic.get("protection", 0.0) >= args.contact_tolerance
            )
            max_kp = args.Kp * KP_CONTACT_MAX_FACTOR
            if only_contact_error and args.kp_active < max_kp * (1.0 - 1.0e-12):
                old_kp = args.kp_active
                args.kp_active = min(args.kp_active * KP_CONTACT_GROW_FACTOR, max_kp)
                print(
                    f"  KP ADAPT: velocity/residual converged but protection error is high; "
                    f"Kp_eff {old_kp:g} -> {args.kp_active:g} N/m, retrying from accepted state"
                )
                continue
            if args.ki_active > 0.0:
                # 允许继续减半至精确的 Ki=0；然后才开始 2/4 子步尝试。
                half_ki = args.ki_active * KI_ADAPTIVE_REDUCTION_FACTOR
                next_ki = 0.0 if half_ki < KI_ADAPTIVE_ZERO_THRESHOLD else half_ki
                old_ki, args.ki_active = args.ki_active, next_ki
                print(f"  KI ADAPT: full increment failed; Ki_eff {old_ki:g} -> {args.ki_active:g} N/(m*s), retrying from accepted state")
                continue
            else:
                raise RuntimeError(
                    f"load step {load_step} did not settle with Ki=0 after subdivisions {RAMP_RETRY_SUBDIVISIONS}. "
                    f"Last: speed={last_diagnostic.get('speed', math.nan):.3e} m/s, "
                    f"Newton residual={last_diagnostic.get('residual', math.nan):.3e} N, "
                    f"protection error={last_diagnostic.get('protection', math.nan) * 1e3:.4f} mm. "
                    "Reduce --tip-total-displacement or increase --num-load-steps."
                )

    result = {key: np.asarray(value) for key, value in history.items()}
    result.update({
        "initial_positions": np_p, "arc_radius": np.asarray(args.arc_radius), "arc_angle_deg": np.asarray(args.arc_angle_deg),
        "prebalanced_positions": prebalanced_positions, "prebalance_enabled": np.asarray(args.prebalance),
        "prebalance_max_steps": np.asarray(args.prebalance_max_steps),
        "tube_half_width": np.asarray(args.tube_half_width), "safe_distance": np.asarray(args.safe_distance),
        "tip_direction": args.tip_direction, "pulse_nodes": nodes, "pulse_magnitudes": pulse_magnitudes,
        "pulse_directions": pulse_directions, "pulse_profile": pulse_np,
        "tip_motion_mode": np.asarray(args.tip_motion_mode),
        "initial_normal_imperfection_profile": np.asarray(args.initial_imperfection_profile),
        "initial_normal_imperfection_amplitude": np.asarray(args.initial_normal_imperfection_amplitude),
        "initial_imperfection_mode_count": np.asarray(args.initial_imperfection_mode_count),
        "initial_imperfection_seed": np.asarray(-1 if args.initial_imperfection_seed is None else args.initial_imperfection_seed),
    })
    stretch, bend, total = elastic_energy(p, rest, args.EA, args.EI, reference_curvature)
    result["final_elastic_stretch_energy_j"] = np.asarray(float(stretch.item()))
    result["final_elastic_bend_energy_j"] = np.asarray(float(bend.item()))
    result["final_elastic_energy_j"] = np.asarray(float(total.item()))
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prefix.with_suffix(".npz"), **result)
    print(f"history saved: {prefix.with_suffix('.npz')}")
    print("FINAL GUIDEWIRE ELASTIC ENERGY (PI contact is non-conservative and is excluded)")
    print(f"  stretching energy : {float(stretch.item()):.6e} J")
    print(f"  bending energy    : {float(bend.item()):.6e} J")
    print(f"  total elastic energy (stretch + bend): {float(total.item()):.6e} J")
    return result


def sample_boundaries(args: argparse.Namespace, count: int = 600) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """采样中心线、实体内外壁和两条保护层边界，仅供绘图。"""
    theta = np.linspace(0.0, args.arc_angle_rad, count)
    center = np.array([0.0, args.arc_radius])
    radial = np.column_stack((np.sin(theta), -np.cos(theta)))
    centerline = center + args.arc_radius * radial
    outer = center + (args.arc_radius + args.tube_half_width) * radial  # rho=R+h，外侧实体壁。
    inner = center + (args.arc_radius - args.tube_half_width) * radial  # rho=R-h，内侧实体壁。
    po = center + (args.arc_radius + args.tube_half_width - args.safe_distance) * radial
    pi = center + (args.arc_radius - args.tube_half_width + args.safe_distance) * radial
    return centerline, outer, inner, np.stack((po, pi))


def visualize_results(history: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    """先播放完整变形动画，再显示初始/中间/最终构型与求解器历史。"""
    if not args.show:
        return
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    centerline, outer, inner, protection = sample_boundaries(args)
    positions = history["positions"]
    time_values = history["time"]

    # 动画使用全部已保存的历史帧：包括随机脉冲、平滑加载和每个载荷步的松弛过程。
    if not args.skip_animation:
        figure, axis = plt.subplots(figsize=(8.5, 7.0))
        axis.plot(outer[:, 0], outer[:, 1], "k", lw=1.4, label="tube walls")
        axis.plot(inner[:, 0], inner[:, 1], "k", lw=1.4)
        axis.plot(protection[0, :, 0], protection[0, :, 1], color="tab:red", ls=":", lw=1.0, label="protection layer")
        axis.plot(protection[1, :, 0], protection[1, :, 1], color="tab:red", ls=":", lw=1.0)
        axis.plot(centerline[:, 0], centerline[:, 1], color="0.75", ls="--", lw=1.0, label="centerline")
        rod_line, = axis.plot([], [], "b-o", ms=2.5, lw=1.4, label="guidewire")
        fixed_points = axis.scatter([], [], marker="s", color="red", s=42, label="fixed nodes", zorder=5)
        contact_points = axis.scatter([], [], color="magenta", s=25, label="PI contact", zorder=6)
        title = axis.set_title("")
        all_points = np.vstack((outer, inner, positions.reshape(-1, 2)))
        margin = 0.08 * args.total_length
        axis.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        axis.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        axis.set_aspect("equal", adjustable="box")
        axis.set(xlabel="x (m)", ylabel="y (m)")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
        frame_indices = np.arange(0, len(positions), args.animation_stride, dtype=int)
        if frame_indices[-1] != len(positions) - 1:
            frame_indices = np.append(frame_indices, len(positions) - 1)

        def update(frame_number: int):
            frame = int(frame_indices[frame_number])
            point_set = positions[frame]
            rod_line.set_data(point_set[:, 0], point_set[:, 1])
            fixed_points.set_offsets(point_set[:2])
            active = history["contact_active"][frame]
            contact_points.set_offsets(point_set[active] if np.any(active) else np.empty((0, 2)))
            title.set_text(
                f"2-D arc guidewire | t={time_values[frame]:.3f} s | "
                f"load step={history['load_step'][frame]} | tip displacement={history['tip_displacement'][frame] * 1e3:.3f} mm"
            )
            return rod_line, fixed_points, contact_points, title

        animation = FuncAnimation(
            figure, update, frames=len(frame_indices), interval=1000.0 / args.animation_fps,
            blit=False, repeat=args.animation_repeat,
        )
        # 必须保留引用，避免 Matplotlib 在窗口显示前回收动画对象。
        figure._guidewire_animation = animation
        plt.show()
        plt.close(figure)

    bend_index = int(np.argmax(np.ptp(positions[:, :, 0], axis=1) + np.ptp(positions[:, :, 1], axis=1)))
    fig, (ax, hist) = plt.subplots(2, 1, figsize=(9, 10))
    ax.plot(outer[:, 0], outer[:, 1], "k", lw=1.4, label="tube walls")
    ax.plot(inner[:, 0], inner[:, 1], "k", lw=1.4)
    ax.plot(protection[0, :, 0], protection[0, :, 1], color="tab:red", ls=":", lw=1.0, label="protection layer")
    ax.plot(protection[1, :, 0], protection[1, :, 1], color="tab:red", ls=":", lw=1.0)
    ax.plot(centerline[:, 0], centerline[:, 1], color="0.75", ls="--", lw=1.0, label="centerline")
    ax.plot(positions[0, :, 0], positions[0, :, 1], "0.4", ls="--", label="initial")
    ax.plot(positions[bend_index, :, 0], positions[bend_index, :, 1], color="tab:orange", label="maximum span")
    ax.plot(positions[-1, :, 0], positions[-1, :, 1], "b-o", ms=2.5, label="final")
    ax.scatter(positions[-1, :2, 0], positions[-1, :2, 1], marker="s", color="red", label="fixed nodes", zorder=5)
    contact = history["contact_active"][-1]
    if np.any(contact):
        ax.scatter(positions[-1, contact, 0], positions[-1, contact, 1], color="magenta", s=25, label="PI contact", zorder=6)
    tip = positions[-1, -1]
    arrow = 0.12 * args.total_length * args.tip_direction  # 绿色箭头仅表示用户指定的固定加载方向。
    ax.arrow(tip[0], tip[1], arrow[0], arrow[1], color="green", width=0.0015, head_width=0.018, length_includes_head=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="x (m)", ylabel="y (m)", title="2-D guidewire in a circular arc tube")
    ax.grid(alpha=0.25); ax.legend(loc="best")
    hist.plot(history["time"], history["max_protection_error"] * 1e3, label="max protection error (mm)")
    hist.plot(history["time"], history["max_pi_force"], label="max PI force (N)")
    hist.plot(history["time"], history["newton_iterations"], label="Newton iterations")
    hist.grid(alpha=0.25); hist.legend(); hist.set(xlabel="relaxation time (s)", title="Contact and solver history")
    fig.tight_layout(); plt.show()


def run_self_test(args: argparse.Namespace) -> None:
    """不进行完整仿真，只验证圆弧几何、接触零点及壁面反力方向。"""
    p, _, rest, _, integrals = initialize_rod(args)
    # The automatic radius is chord-corrected, so the *discrete* rest length
    # exactly equals the requested guidewire length while retaining zero
    # initial strain.  (The continuous arc length differs by O(ds^2).)
    assert np.isclose(rest.sum(), args.total_length, rtol=1e-12), "arc rest length mismatch"
    assert np.allclose(p[0], (0.0, 0.0), atol=1e-12)
    # 最后节点无缺陷，故仍落在已修正半径对应的名义圆弧终点。
    assert np.allclose(p[-1], (args.arc_radius * math.sin(args.arc_angle_rad), args.arc_radius * (1.0 - math.cos(args.arc_angle_rad))), atol=1e-12)
    n0 = circle_normal_numpy(p[:1], args.arc_radius)[0]
    assert np.allclose(n0, (0.0, -1.0), atol=1e-12), "start radial normal mismatch"
    # 关键物理检查：导丝的自然状态为直杆。故初始圆弧没有拉伸应变，
    # 但必然储存正的弯曲势能；这份势能正是它试图恢复平直的驱动力。
    p_torch = torch.as_tensor(p, dtype=DTYPE)
    rest_torch = torch.as_tensor(rest, dtype=DTYPE)
    kappa0 = torch.zeros(args.num_nodes - 2, dtype=DTYPE)
    stretch0, bend0, total0 = elastic_energy(p_torch, rest_torch, args.EA, args.EI, kappa0)
    assert float(abs(stretch0)) < 1.0e-18 and float(bend0) > 0.0 and float(total0) > 0.0, "initial circular state must carry straight-rod bending energy"
    device = torch.device("cpu")
    # 接触零点测试使用名义圆弧中心线，而不是带缺陷的自然导丝节点。
    nominal_center = arc_points(args.arc_radius, args.arc_angle_rad, args.num_nodes)
    center = torch.as_tensor(nominal_center[args.num_nodes // 2:args.num_nodes // 2 + 1], dtype=DTYPE, device=device)
    force, *_ = contact_quantities(center, torch.zeros(1, dtype=DTYPE), torch.zeros(1, dtype=DTYPE), args, args.dt)
    assert torch.allclose(force, torch.zeros_like(force), atol=1e-12), "centerline contact must be zero"
    outer = center + torch.as_tensor(circle_normal_numpy(center.numpy(), args.arc_radius), dtype=DTYPE) * (args.tube_half_width - 0.5 * args.safe_distance)
    outer_force, *_ = contact_quantities(outer, torch.zeros(1, dtype=DTYPE), torch.zeros(1, dtype=DTYPE), args, args.dt)
    assert float(torch.sum(outer_force * (outer - torch.tensor([0.0, args.arc_radius]))).item()) < 0.0, "outer-wall force must point inward"
    print("self-test passed: circular geometry, straight-rod restoring energy, zero centerline contact, and inward wall force")


def parse_args() -> argparse.Namespace:
    """定义命令行接口、计算派生参数并检查物理/数值参数是否合法。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--total-length", type=float, default=TOTAL_LENGTH)
    parser.add_argument("--diameter", type=float, default=DIAMETER)
    parser.add_argument("--density", type=float, default=MATERIAL_DENSITY)
    parser.add_argument("--num-nodes", type=int, default=NUM_NODES)
    parser.add_argument("--arc-angle-deg", type=float, default=ARC_ANGLE_DEG)
    parser.add_argument("--arc-radius", type=float, default=None, help="圆心线半径(m)；省略时由导丝长度和圆弧角自动计算")
    parser.add_argument("--tube-half-width", type=float, default=TUBE_HALF_WIDTH)
    parser.add_argument("--safe-distance", type=float, default=SAFE_DISTANCE)
    parser.add_argument("--Kp", type=float, default=KP); parser.add_argument("--Ki", type=float, default=KI)
    parser.add_argument("--max-pi-force", type=float, default=MAX_PI_FORCE)
    parser.add_argument("--max-integral-state", type=float, default=MAX_INTEGRAL_STATE)
    parser.add_argument("--integral-leak-rate", type=float, default=INTEGRAL_LEAK_RATE)
    parser.add_argument("--contact-smooth-length", type=float, default=CONTACT_SMOOTH_LENGTH)
    parser.add_argument("--direction-epsilon", type=float, default=DIRECTION_EPSILON)
    # 例如 (0,-1) 为竖直下压；(-1,-1) 为左下斜压。程序会自动归一化。
    parser.add_argument("--tip-direction-x", type=float, default=0.0, help="末端全局固定加载方向的 x 分量")
    parser.add_argument("--tip-direction-y", type=float, default=-1.0, help="末端全局固定加载方向的 y 分量")
    parser.add_argument(
        "--tip-motion-mode", choices=("fixed-direction", "arc-follow"), default=TIP_MOTION_MODE,
        help="末端位移路径：fixed-direction 沿固定全局直线；arc-follow 沿初始圆弧中心线向入口回退",
    )
    parser.add_argument("--tip-total-displacement", type=float, default=TIP_TOTAL_DISPLACEMENT)
    parser.add_argument("--num-load-steps", type=int, default=NUM_LOAD_STEPS); parser.add_argument("--tip-ramp-steps", type=int, default=TIP_RAMP_STEPS)
    parser.add_argument("--pulse-count-min", type=int, default=PULSE_COUNT_MIN); parser.add_argument("--pulse-count-max", type=int, default=PULSE_COUNT_MAX)
    parser.add_argument("--pulse-force-min", type=float, default=PULSE_FORCE_MIN); parser.add_argument("--pulse-force-max", type=float, default=PULSE_FORCE_MAX)
    parser.add_argument("--pulse-duration", type=float, default=PULSE_DURATION)
    parser.add_argument("--pulse-seed", type=int, default=None, help="随机种子；省略则每次运行产生新扰动")
    parser.add_argument(
        "--initial-normal-imperfection-amplitude", type=float,
        default=INITIAL_NORMAL_IMPERFECTION_AMPLITUDE,
        help="作为自然预弯形状写入的随机法向几何缺陷最大幅值(m)；0 表示关闭",
    )
    parser.add_argument(
        "--initial-imperfection-mode-count", type=int,
        default=INITIAL_IMPERFECTION_MODE_COUNT,
        help="初始法向缺陷叠加的正弦模态数；3 可形成上下交替的多波初始形状",
    )
    parser.add_argument(
        "--initial-imperfection-seed", type=int, default=INITIAL_IMPERFECTION_SEED,
        help="自然初始几何缺陷随机种子；省略则每次随机",
    )
    parser.add_argument("--dt", type=float, default=DT); parser.add_argument("--max-newton-iters", type=int, default=MAX_NEWTON_ITERS)
    parser.add_argument("--residual-tol", type=float, default=NEWTON_RESIDUAL_TOL); parser.add_argument("--step-tol", type=float, default=NEWTON_STEP_TOL)
    parser.add_argument("--settle-speed-tol", type=float, default=SETTLE_SPEED_TOL); parser.add_argument("--settle-residual-tol", type=float, default=SETTLE_RESIDUAL_TOL)
    parser.add_argument("--contact-tolerance", type=float, default=CONTACT_TOLERANCE); parser.add_argument("--settle-confirm-steps", type=int, default=SETTLE_CONFIRM_STEPS)
    parser.add_argument("--max-settle-steps", type=int, default=MAX_SETTLE_STEPS); parser.add_argument("--history-save-interval", type=int, default=20)
    parser.add_argument("--prebalance", action=argparse.BooleanOptionalAction, default=PREBALANCE_ENABLED,
                        help="在随机脉冲与压缩前，以零内缩、两端固定条件松弛直杆恢复力")
    parser.add_argument("--prebalance-max-steps", type=int, default=MAX_SETTLE_STEPS,
                        help="零内缩预平衡阶段允许的最大松弛步数")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-prefix", default="2d_npz/rod_soft_contact_PI_quasi_static_2d_arc")
    parser.add_argument("--animation-stride", type=int, default=1, help="动画每隔多少保存帧显示一次；1 表示完整过程")
    parser.add_argument("--animation-fps", type=float, default=30.0, help="动画播放帧率")
    parser.add_argument("--animation-repeat", action="store_true", help="动画播放结束后循环")
    parser.add_argument("--skip-animation", action="store_true", help="不播放过程动画，只显示最终汇总图")
    parser.add_argument("--no-show", dest="show", action="store_false", default=True); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    args.arc_angle_rad = math.radians(args.arc_angle_deg)  # 后续三角函数/几何计算统一使用弧度。
    if args.arc_radius is None:
        # DER 静息长度是离散弦长而非连续圆弧长度。这里作弦长修正，使
        # sum(rest_lengths) 恰等于 total_length，且默认 R≈0.63666 m。
        dtheta = args.arc_angle_rad / (args.num_nodes - 1)
        args.arc_radius = args.total_length / (2.0 * (args.num_nodes - 1) * math.sin(0.5 * dtheta))
    direction = np.array((args.tip_direction_x, args.tip_direction_y), dtype=float)
    if np.linalg.norm(direction) <= 0.0: parser.error("tip direction must be nonzero")
    args.tip_direction = direction / np.linalg.norm(direction)  # 只保留方向，位移大小由 tip-total-displacement 控制。
    for name in ("total_length", "diameter", "density", "arc_radius", "tube_half_width", "safe_distance", "dt", "max_pi_force", "max_integral_state", "contact_smooth_length", "direction_epsilon", "tip_total_displacement", "num_load_steps", "tip_ramp_steps", "max_newton_iters", "residual_tol", "step_tol", "settle_speed_tol", "settle_residual_tol", "settle_confirm_steps", "max_settle_steps", "prebalance_max_steps", "history_save_interval"):
        if getattr(args, name) <= 0: parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_nodes < 6: parser.error("--num-nodes must be at least 6")
    if args.animation_stride <= 0 or args.animation_fps <= 0: parser.error("animation-stride and animation-fps must be positive")
    if not 0.0 < args.safe_distance < args.tube_half_width: parser.error("safe-distance must lie between zero and tube-half-width")
    if not 0.0 < args.arc_angle_rad < 2.0 * math.pi: parser.error("arc-angle-deg must lie between 0 and 360")
    if args.pulse_count_min < 1 or args.pulse_count_max < args.pulse_count_min or args.pulse_count_max > args.num_nodes - 3: parser.error("invalid pulse-count range")
    if args.pulse_force_min < 0 or args.pulse_force_max < args.pulse_force_min or args.pulse_duration < 0: parser.error("invalid pulse force/duration")
    if args.initial_normal_imperfection_amplitude < 0 or args.initial_normal_imperfection_amplitude >= args.tube_half_width - args.safe_distance:
        parser.error("initial-normal-imperfection-amplitude must be non-negative and below the protection-layer boundary")
    if args.initial_imperfection_mode_count < 1:
        parser.error("initial-imperfection-mode-count must be at least 1")
    if args.tip_motion_mode == "arc-follow" and args.tip_total_displacement >= args.arc_radius * args.arc_angle_rad:
        parser.error("arc-follow tip-total-displacement must be smaller than the initial arc length")
    _, args.EA, args.EI, args.damping = rod_properties(args.total_length, args.diameter, args.density)
    args.tip_displacement_step = args.tip_total_displacement / args.num_load_steps  # 每个载荷步的累计位移增量。
    args.torch_device = resolve_device(args.device)
    return args


if __name__ == "__main__":
    parsed = parse_args()
    run_self_test(parsed) if parsed.self_test else None
    if not parsed.self_test:
        started = time.perf_counter()
        archive = run_simulation(parsed)
        print(f"completed in {time.perf_counter() - started:.2f} s")
        visualize_results(archive, parsed)
