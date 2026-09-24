"""Pure-Python 3-D DER guidewire in an analytic 90-degree circular-arc tube.

This is deliberately independent of the OBJ/C++ paths.  It combines the
three-dimensional, twist-free DER/Newton structure used by the successful
straight-tube Python reference with the circular-arc geometry and PI-contact
semantics of the 2-D arc reference.
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

from . import arc_mesh_contact


# ============================================================================
# 可调参数区（命令行中的同名 --参数 会覆盖这里的默认值；单位一律为 SI）
# ============================================================================
# --- 数值精度、导丝材料与离散 ---
DTYPE = torch.float64                 # 双精度：曲率、接触 Jacobian 与 Newton 迭代更稳定。
TOTAL_LENGTH = 1.5                    # 导丝沿初始圆弧中心线的离散总长度，m。
DIAMETER = 1.0e-3                     # 导丝圆截面直径，m。
DENSITY = 7850.0                      # 导丝材料密度，kg/m^3；钢的典型值。
YOUNG_MODULUS = 200.0e9               # 杨氏模量，Pa；增大后导丝更难拉伸/弯曲。
DAMPING_RATIO = 1.0                   # 等效临界阻尼比；增大可更快抑制动力学振荡，但过大可能变慢。
NUM_NODES = 250                        # 与当前测试目标一致的 DER 节点数；越大越精细，但纯 Python 局部 Hessian/Newton 更耗时。

# --- 解析 90° 圆弧管几何 ---
ARC_ANGLE_DEG = 90.0                  # 圆弧转角，deg；起点切线沿 +x，90° 时终点切线沿 +y。
# 圆心线半径默认由 TOTAL_LENGTH、ARC_ANGLE_DEG、NUM_NODES 的离散弦长自动计算。
# 如需固定半径，请用 --arc-radius R（m）覆盖。
CONTACT_RADIUS = 40.0e-3              # PI 接触面半径，m；节点到此半径即开始受到反推力。
TUBE_RADIUS = 50.0e-3                 # 仅供绘图显示的实体外壳半径，m；不直接参与 PI 力计算。
SAFE_DISTANCE = 10.0e-3               # 外壳与 PI 接触面的间距，m；默认满足 TUBE_RADIUS-CONTACT_RADIUS。

# --- 节点 PI 软接触（不投影、不硬碰撞；反力总是指回圆弧中心线）---
KP, KI = 3500.0, 100.0                # 比例/积分增益，单位分别为 N/m、N/(m*s)。
MAX_PI_FORCE = 20.0                   # 单节点 PI 力上限，N；避免刚接触时数值爆炸。
MAX_INTEGRAL_STATE = 5.0e-2           # 单节点积分状态上限，m*s；防积分饱和。
INTEGRAL_LEAK_RATE = 5.0              # 离开保护层后积分项指数泄漏速度，1/s。
CONTACT_SMOOTH_LENGTH = 2.0e-5        # 平移 Softplus/连续激活的接触切换尺度，m；越小越接近硬切换。
DIRECTION_EPSILON = 1.0e-6            # 节点正好在中心线时的力方向正则化尺度，m。

# --- 隐式 Euler / Newton 动力学松弛 ---
DT = 5.0e-3                           # 单个隐式时间步，s；只用于数值松弛，不代表真实临床时间。
MAX_NEWTON_ITERS = 35                 # 每个隐式步允许的最大 Newton 迭代次数。
RESIDUAL_TOL, STEP_TOL = 1.0e-7, 1.0e-10  # Newton 力残差阈值 N、位移修正阈值 m。
SETTLE_SPEED_TOL = 1.0e-3             # 与二维弧形管一致：准静态平衡的最大自由节点速度，m/s。
SETTLE_RESIDUAL_TOL = 1.0e-6          # 载荷步平衡时的 Newton 残差阈值，N。
CONTACT_TOLERANCE = 5.0e-4            # 平衡时允许的最大保护层超限，m。
SETTLE_CONFIRM_STEPS = 4              # 上述平衡条件连续成立次数，避免瞬时误判。
MAX_SETTLE_STEPS = 1600               # 与二维弧形管一致：每个压缩载荷步最多松弛步数。
PREBALANCE_MAX_STEPS = 1600           # 与二维弧形管一致：零位移预平衡最多步数。
SETTLE_DIAGNOSTIC_INTERVAL = 100      # 松弛过程每隔多少个隐式步输出一次速度/残差/接触诊断；0 表示关闭。

# --- 末端压缩：末端只沿解析圆弧中心线向入口回退 ---
TIP_TOTAL_DISPLACEMENT = 0.12         # 与二维弧形管一致：末端沿中心线累计回退距离，m。
NUM_LOAD_STEPS = 20                   # 与二维弧形管一致：总压缩量分成的宏观载荷步数。
TIP_RAMP_STEPS = 25                   # 与二维弧形管一致：每个宏观载荷步内的平滑位移子步数。

# --- 严格 OBJ 面片接触的数值续算（不改变二维物理参数）---
# 面片边缘不可微，因此用自适应位移增量跨越面片；这些仅决定求解路径和耗时。
RAMP_MODE = "adaptive"                # adaptive=速度回滚续算；fixed=旧的 25 步平滑 ramp 对照。
ADAPTIVE_RELAX_STEPS = 8              # 每个试探位移后保持末端进行的最多微松弛步数。
ADAPTIVE_SPEED_GROW_FACTOR = 2.0      # 速度低于该倍数×平衡阈值时，下一试探位移可增大。
ADAPTIVE_SPEED_SHRINK_FACTOR = 10.0   # 与直圆管 v4 一致：速度高于该倍数×平衡阈值时，回滚并减半位移增量；最终平衡仍使用严格二维阈值。

# --- 初始随机三维脉冲：只在压缩前短时施加，用来选择三维屈曲分支 ---
PULSE_COUNT_MIN, PULSE_COUNT_MAX = 4, 8  # 随机脉冲节点数量范围；节点始终在自由内部节点中抽取。
PULSE_FORCE_MIN, PULSE_FORCE_MAX = 2.0e-2, 7.0e-2  # 每个脉冲力幅值范围，N。
PULSE_DURATION = 2.0e-2               # 脉冲持续时间，s；结束后撤除，再开始正式压缩。
PULSE_SEED = 20260907                 # 固定随机种子；保证 Python 基线与后续 C++ 版本使用同一组脉冲。

# --- 结果回放与文件输出 ---
ANIMATION_STRIDE = 5                   # GIF 每隔多少个已保存状态取一帧；1 表示逐帧完整回放。
ANIMATION_FPS = 20.0                   # GIF/交互动画播放帧率。
HISTORY_SAVE_INTERVAL = 10             # 预平衡/松弛每隔多少个隐式步保存一个动画与诊断状态。


@dataclass
class NewtonInfo:
    converged: bool
    iterations: int
    residual_norm: float
    step_norm: float
    line_search_failures: int


def rod_properties(length: float, diameter: float, density: float) -> tuple[float, float, float, float]:
    area = math.pi * diameter**2 / 4.0
    mass = density * area * length
    ea = YOUNG_MODULUS * area
    ei = YOUNG_MODULUS * math.pi * diameter**4 / 64.0
    damping = 2.0 * DAMPING_RATIO * math.sqrt((3.0 * ei / length**3) * mass)
    return mass, ea, ei, damping


def arc_radius(length: float, angle: float, nodes: int) -> float:
    """Radius whose *discrete chord* lengths sum exactly to ``length``."""
    dtheta = angle / (nodes - 1)
    return length / (2.0 * (nodes - 1) * math.sin(0.5 * dtheta))


def arc_points(radius: float, angle: float, count: int) -> np.ndarray:
    theta = np.linspace(0.0, angle, count)
    return np.column_stack((radius * np.sin(theta), radius * (1.0 - np.cos(theta)), np.zeros(count)))


def initialize_rod(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = arc_points(args.arc_radius, args.arc_angle_rad, args.num_nodes)
    rest = np.linalg.norm(np.diff(p, axis=0), axis=1)
    mass, _, _, _ = rod_properties(args.total_length, args.diameter, args.density)
    masses = np.full(args.num_nodes, mass / (args.num_nodes - 1), dtype=np.float64)
    masses[[0, -1]] *= 0.5
    return p, np.zeros_like(p), rest, masses, np.zeros(args.num_nodes)


def local_frames_numpy(points: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.array((0.0, radius, 0.0))
    radial = points - center
    radial[:, 2] = 0.0
    normal = radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1.0e-14)
    tangent = np.column_stack((-normal[:, 1], normal[:, 0], np.zeros(len(points))))
    binormal = np.tile(np.array((0.0, 0.0, 1.0)), (len(points), 1))
    return tangent, normal, binormal


def make_random_pulse(args: argparse.Namespace, initial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random free-node pulses in the full local normal/binormal cross-section."""
    rng = np.random.default_rng(args.pulse_seed)
    candidates = np.arange(2, args.num_nodes - 1, dtype=int)
    count = int(rng.integers(args.pulse_count_min, args.pulse_count_max + 1))
    nodes = np.sort(rng.choice(candidates, size=count, replace=False))
    magnitudes = rng.uniform(args.pulse_force_min, args.pulse_force_max, size=count)
    angles = rng.uniform(0.0, 2.0 * math.pi, size=count)
    _, normal, binormal = local_frames_numpy(initial, args.arc_radius)
    directions = np.cos(angles)[:, None] * normal[nodes] + np.sin(angles)[:, None] * binormal[nodes]
    profile = np.zeros_like(initial)
    profile[nodes] = magnitudes[:, None] * directions
    return nodes, magnitudes, directions, profile


def local_bending_energy(points: torch.Tensor, rest_left: torch.Tensor, rest_right: torch.Tensor, ei: float) -> torch.Tensor:
    left, right = points[1] - points[0], points[2] - points[1]
    tl = left / torch.linalg.vector_norm(left).clamp_min(1.0e-12)
    tr = right / torch.linalg.vector_norm(right).clamp_min(1.0e-12)
    kb = 2.0 * torch.linalg.cross(tl, tr) / (1.0 + torch.dot(tl, tr)).clamp_min(1.0e-12)
    return 0.5 * ei * torch.dot(kb, kb) / (0.5 * (rest_left + rest_right))


def elastic_energy(positions: torch.Tensor, rest: torch.Tensor, ea: float, ei: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edges = positions[1:] - positions[:-1]
    lengths = torch.linalg.vector_norm(edges, dim=1).clamp_min(1.0e-12)
    tangents = edges / lengths[:, None]
    kb = 2.0 * torch.linalg.cross(tangents[:-1], tangents[1:], dim=1) / (1.0 + torch.sum(tangents[:-1] * tangents[1:], dim=1)).clamp_min(1.0e-12)[:, None]
    stretch = 0.5 * ea * torch.sum((lengths - rest).square() / rest)
    bend = 0.5 * ei * torch.sum(torch.sum(kb.square(), dim=1) / (0.5 * (rest[:-1] + rest[1:])))
    return stretch, bend, stretch + bend


def stretch_grad_hessian(positions: torch.Tensor, rest: torch.Tensor, ea: float) -> tuple[torch.Tensor, torch.Tensor]:
    delta = positions[1:] - positions[:-1]
    length = torch.linalg.vector_norm(delta, dim=1).clamp_min(1.0e-12)
    stiffness = ea / rest
    coeff = 1.0 - rest / length
    grad_delta = stiffness[:, None] * coeff[:, None] * delta
    gradient = torch.cat((-grad_delta, grad_delta), dim=1)
    eye = torch.eye(3, dtype=positions.dtype, device=positions.device)
    local_k = stiffness[:, None, None] * (coeff[:, None, None] * eye + (rest / length**3)[:, None, None] * delta[:, :, None] * delta[:, None, :])
    hessian = torch.empty((len(delta), 6, 6), dtype=positions.dtype, device=positions.device)
    hessian[:, :3, :3], hessian[:, :3, 3:] = local_k, -local_k
    hessian[:, 3:, :3], hessian[:, 3:, 3:] = -local_k, local_k
    return gradient, hessian


@dataclass
class BlockPenta3D:
    diagonal: torch.Tensor
    upper1: torch.Tensor
    upper2: torch.Tensor
    lower1: torch.Tensor
    lower2: torch.Tensor

    @classmethod
    def zeros(cls, count: int, dtype: torch.dtype, device: torch.device) -> "BlockPenta3D":
        z = lambda n: torch.zeros((max(n, 0), 3, 3), dtype=dtype, device=device)
        return cls(z(count), z(count - 1), z(count - 2), z(count - 1), z(count - 2))

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        x = vector.reshape(-1, 3)
        y = torch.bmm(self.diagonal, x.unsqueeze(-1)).squeeze(-1)
        if len(x) > 1:
            y[:-1] += torch.bmm(self.upper1, x[1:].unsqueeze(-1)).squeeze(-1)
            y[1:] += torch.bmm(self.lower1, x[:-1].unsqueeze(-1)).squeeze(-1)
        if len(x) > 2:
            y[:-2] += torch.bmm(self.upper2, x[2:].unsqueeze(-1)).squeeze(-1)
            y[2:] += torch.bmm(self.lower2, x[:-2].unsqueeze(-1)).squeeze(-1)
        return y.reshape(-1)

    def dense(self) -> torch.Tensor:
        n = len(self.diagonal)
        value = torch.zeros((3 * n, 3 * n), dtype=self.diagonal.dtype, device=self.diagonal.device)
        for i in range(n): value[3*i:3*i+3, 3*i:3*i+3] = self.diagonal[i]
        for i in range(n - 1):
            value[3*i:3*i+3, 3*(i+1):3*(i+2)] = self.upper1[i]
            value[3*(i+1):3*(i+2), 3*i:3*i+3] = self.lower1[i]
        for i in range(n - 2):
            value[3*i:3*i+3, 3*(i+2):3*(i+3)] = self.upper2[i]
            value[3*(i+2):3*(i+3), 3*i:3*i+3] = self.lower2[i]
        return value

    def _solve_cpu_block_lu(self, rhs: torch.Tensor) -> torch.Tensor:
        """Serial 3x3 block LU: efficient on CPU, but intentionally not on CUDA."""
        d, u1, u2 = self.diagonal.clone(), self.upper1.clone(), self.upper2.clone()
        l1, l2, b = self.lower1.clone(), self.lower2.clone(), rhs.reshape(-1, 3).clone()
        try:
            for i in range(len(d)):
                if i + 1 < len(d):
                    f = torch.linalg.solve(d[i].T, l1[i].T).T
                    d[i+1] -= f @ u1[i]; b[i+1] -= f @ b[i]
                    if i + 2 < len(d): u1[i+1] -= f @ u2[i]
                if i + 2 < len(d):
                    f = torch.linalg.solve(d[i].T, l2[i].T).T
                    l1[i+1] -= f @ u1[i]; d[i+2] -= f @ u2[i]; b[i+2] -= f @ b[i]
            x = torch.empty_like(b)
            for i in range(len(d)-1, -1, -1):
                v = b[i]
                if i + 1 < len(d): v -= u1[i] @ x[i+1]
                if i + 2 < len(d): v -= u2[i] @ x[i+2]
                x[i] = torch.linalg.solve(d[i], v)
            x = x.reshape(-1)
            rel = torch.linalg.vector_norm(self.matvec(x) - rhs) / torch.clamp(torch.linalg.vector_norm(rhs), min=1.0e-30)
            if bool(torch.isfinite(x).all()) and bool(rel < 1.0e-8): return x
        except RuntimeError:
            pass
        return torch.linalg.solve(self.dense(), rhs)

    def _solve_cuda_pcr(self, rhs: torch.Tensor) -> torch.Tensor:
        """GPU parallel cyclic reduction (PCR) on paired 6x6 block rows.

        Pairing two adjacent 3x3 rows transforms the pentadiagonal system into
        a block-tridiagonal 6x6 system.  PCR then reduces it in O(log N)
        batched GPU solves instead of launching O(N) serial 3x3 solves.
        """
        blocks = len(self.diagonal)
        groups = (blocks + 1) // 2
        dtype, device = self.diagonal.dtype, self.diagonal.device
        left = torch.zeros((groups, 6, 6), dtype=dtype, device=device)
        centre, right = torch.zeros_like(left), torch.zeros_like(left)
        vector = torch.zeros((groups, 6), dtype=dtype, device=device)
        even_count, odd_count = len(self.diagonal[0::2]), len(self.diagonal[1::2])
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
            if len(self.lower2[1::2]):
                left[1:1 + len(self.lower2[1::2]), 3:, 3:] = self.lower2[1::2]
            right[:-1, :3, :3] = self.upper2[0::2]
            right[:-1, 3:, :3] = self.upper1[1::2]
            if len(self.upper2[1::2]):
                right[:len(self.upper2[1::2]), 3:, 3:] = self.upper2[1::2]
        stride = 1
        while stride < groups:
            new_left, new_right = torch.zeros_like(left), torch.zeros_like(right)
            new_centre, new_vector = centre.clone(), vector.clone()
            alpha = torch.linalg.solve(centre[:-stride].transpose(1, 2), left[stride:].transpose(1, 2)).transpose(1, 2)
            new_centre[stride:] -= alpha @ right[:-stride]
            new_vector[stride:] -= (alpha @ vector[:-stride, :, None]).squeeze(-1)
            new_left[stride:] = -alpha @ left[:-stride]
            gamma = torch.linalg.solve(centre[stride:].transpose(1, 2), right[:-stride].transpose(1, 2)).transpose(1, 2)
            new_centre[:-stride] -= gamma @ left[stride:]
            new_vector[:-stride] -= (gamma @ vector[stride:, :, None]).squeeze(-1)
            new_right[:-stride] = -gamma @ right[stride:]
            left, centre, right, vector = new_left, new_centre, new_right, new_vector
            stride *= 2
        grouped_solution = torch.linalg.solve(centre, vector)
        solution = torch.empty((blocks, 3), dtype=dtype, device=device)
        solution[0::2] = grouped_solution[:even_count, :3]
        if odd_count: solution[1::2] = grouped_solution[:odd_count, 3:]
        return solution.reshape(-1)

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        """Use CUDA PCR or CPU block LU, then verify and densely rescue if needed."""
        try:
            result = self._solve_cuda_pcr(rhs) if self.diagonal.is_cuda else self._solve_cpu_block_lu(rhs)
            relative = torch.linalg.vector_norm(self.matvec(result) - rhs) / torch.clamp(torch.linalg.vector_norm(rhs), min=1.0e-30)
            if self.diagonal.is_cuda:
                for _ in range(2):
                    if bool(relative <= 1.0e-8): break
                    result += self._solve_cuda_pcr(rhs - self.matvec(result))
                    relative = torch.linalg.vector_norm(self.matvec(result) - rhs) / torch.clamp(torch.linalg.vector_norm(rhs), min=1.0e-30)
            if bool(torch.isfinite(result).all()) and bool(relative <= 1.0e-8): return result
        except RuntimeError:
            pass
        return torch.linalg.solve(self.dense(), rhs)


@dataclass(frozen=True)
class BlockPlan:
    free_start: int
    free_end: int
    count: int


@functools.lru_cache(maxsize=8)
def block_plan(nodes: int, tip_constrained: bool) -> BlockPlan:
    start, end = 2, nodes - int(tip_constrained)
    return BlockPlan(start, end, end - start)


def assemble_elastic_banded(positions: torch.Tensor, rest: torch.Tensor, plan: BlockPlan, ea: float, ei: float) -> tuple[torch.Tensor, BlockPenta3D]:
    grad = torch.zeros((plan.count, 3), dtype=positions.dtype, device=positions.device)
    system = BlockPenta3D.zeros(plan.count, positions.dtype, positions.device)

    def scatter(g: torch.Tensor, h: torch.Tensor, first: int) -> None:
        local_nodes, elements = g.shape[1], g.shape[0]
        h = h.reshape(elements, local_nodes, 3, local_nodes, 3)
        for row in range(local_nodes):
            rn = first + torch.arange(elements, device=positions.device) + row
            valid_row = (rn >= plan.free_start) & (rn < plan.free_end)
            if bool(valid_row.any()): grad.index_add_(0, rn[valid_row] - plan.free_start, g[valid_row, row])
            for col in range(local_nodes):
                cn = first + torch.arange(elements, device=positions.device) + col
                valid = valid_row & (cn >= plan.free_start) & (cn < plan.free_end)
                offset = col - row
                if not bool(valid.any()) or offset not in (-2, -1, 0, 1, 2): continue
                band = {0: system.diagonal, 1: system.upper1, -1: system.lower1, 2: system.upper2, -2: system.lower2}[offset]
                dst = rn[valid] - plan.free_start if offset >= 0 else cn[valid] - plan.free_start
                band.index_add_(0, dst, h[valid, row, :, col, :])

    sg, sh = stretch_grad_hessian(positions, rest, ea)
    scatter(sg.reshape(-1, 2, 3), sh, 0)
    def bend_grad_aux(flat: torch.Tensor, left: torch.Tensor, right: torch.Tensor):
        value = torch.func.grad(lambda q: local_bending_energy(q.reshape(3, 3), left, right, ei))(flat)
        return value, value
    dofs = torch.cat((positions[:-2], positions[1:-1], positions[2:]), dim=1)
    bh, bg = torch.func.vmap(torch.func.jacrev(bend_grad_aux, argnums=0, has_aux=True))(dofs, rest[:-1], rest[1:])
    scatter(bg.reshape(-1, 3, 3), bh, 0)
    return grad.reshape(-1), system


def arc_contact(positions: torch.Tensor, integrals: torch.Tensor, previous: torch.Tensor, args: argparse.Namespace, dt: float):
    """Analytic, open-ended arc-tube PI contact; no mesh and no end-cap force."""
    x, y = positions[:, 0], positions[:, 1]
    theta = torch.atan2(x, args.arc_radius - y)
    valid = (theta >= 0.0) & (theta <= args.arc_angle_rad)
    centerline = torch.stack((args.arc_radius * torch.sin(theta), args.arc_radius * (1.0 - torch.cos(theta)), torch.zeros_like(theta)), dim=1)
    offset = positions - centerline
    distance = torch.linalg.vector_norm(offset, dim=1).clamp_min(1.0e-12)
    signed = distance - args.contact_radius
    beta = 1.0 / args.contact_smooth_length
    # Match the OBJ PI law exactly.  This keeps analytic-vs-mesh comparisons
    # about the wall geometry rather than a different force regularization:
    # strictly zero inside/on the wall and continuous growth after crossing.
    active = valid & (signed > 0.0)
    shifted = F.softplus(signed, beta=beta) - math.log(2.0) / beta
    error = torch.where(active, shifted, torch.zeros_like(signed))
    penetration = torch.clamp(signed, min=0.0)
    weight = torch.where(active, -torch.expm1(-penetration / args.contact_smooth_length), torch.zeros_like(signed))
    candidate = weight * (integrals + error * dt) + (1.0 - weight) * integrals * math.exp(-args.integral_leak_rate * dt)
    candidate = torch.clamp(candidate, 0.0, args.max_integral_state)
    unsaturated = args.Kp * error + args.Ki * candidate
    blocked = (args.Ki > 0.0) & (unsaturated >= args.max_pi_force) & (candidate > integrals)
    blocked |= (args.Ki > 0.0) & (error < previous) & (args.Ki * integrals > args.Kp * error) & (candidate > integrals)
    new_integrals = torch.where(blocked, integrals, candidate)
    magnitude = weight * torch.clamp(args.Kp * error + args.Ki * new_integrals, 0.0, args.max_pi_force)
    force = -offset * magnitude[:, None] / torch.sqrt(torch.sum(offset.square(), dim=1, keepdim=True) + args.direction_epsilon**2)
    force = torch.where(active[:, None], force, torch.zeros_like(force))
    clearance = -signed
    return force, new_integrals, active, weight, error, clearance, centerline, valid


def mesh_geometry(args: argparse.Namespace) -> arc_mesh_contact.ArcMeshGeometry:
    """Lazy OBJ load: mesh details stay in arc_mesh_contact.py, not this solver."""
    geometry = getattr(args, "mesh_geometry", None)
    if geometry is None:
        geometry = arc_mesh_contact.ArcMeshGeometry.from_obj(
            args.mesh_obj, arc_radius=args.arc_radius, arc_angle_rad=args.arc_angle_rad,
            contact_radius=args.contact_radius,
        )
        args.mesh_geometry = geometry
    return geometry


def mesh_contact_kwargs(args: argparse.Namespace, dt: float) -> dict[str, float]:
    return {
        "dt": dt, "kp": args.Kp, "ki": args.Ki, "max_force": args.max_pi_force,
        "max_integral": args.max_integral_state, "leak_rate": args.integral_leak_rate,
        "smooth_length": args.contact_smooth_length,
    }


def contact_response(positions: torch.Tensor, integrals: torch.Tensor, previous: torch.Tensor,
                     args: argparse.Namespace, dt: float,
                     frozen_faces: arc_mesh_contact.MeshFaces | None = None):
    """Common analytic/OBJ PI interface used by all solver stages."""
    if args.contact_type == "analytic":
        return arc_contact(positions, integrals, previous, args, dt)
    faces = frozen_faces or mesh_geometry(args).nearest_faces(positions, args.mesh_normal_mode)
    force, new_i, active, weight, error, clearance = arc_mesh_contact.pi_contact(
        positions, faces, integrals, previous, **mesh_contact_kwargs(args, dt),
    )
    return force, new_i, active, weight, error, clearance, faces.reference, faces.valid_arc


def assemble_contact_banded(positions: torch.Tensor, integrals: torch.Tensor, previous: torch.Tensor,
                            args: argparse.Namespace, dt: float,
                            frozen_faces: arc_mesh_contact.MeshFaces | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if args.contact_type == "mesh":
        faces = frozen_faces or mesh_geometry(args).nearest_faces(positions, args.mesh_normal_mode)
        _force, blocks, values = arc_mesh_contact.contact_with_blocks(
            positions, faces, integrals, previous, **mesh_contact_kwargs(args, dt),
        )
        return values[0], blocks
    def force_aux(point: torch.Tensor, integral: torch.Tensor, prior: torch.Tensor):
        value = arc_contact(point[None], integral[None], prior[None], args, dt)[0][0]
        return value, value
    blocks, forces = torch.func.vmap(torch.func.jacrev(force_aux, argnums=0, has_aux=True))(positions, integrals, previous)
    return forces, blocks


def prescribed_state(initial: torch.Tensor, displacement: float, constrained: bool, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros_like(initial, dtype=torch.bool); mask[:2] = True
    values = initial.clone()
    if constrained:
        mask[-1] = True
        theta = args.arc_angle_rad - displacement / args.arc_radius
        values[-1] = torch.tensor((args.arc_radius * math.sin(theta), args.arc_radius * (1.0 - math.cos(theta)), 0.0), dtype=initial.dtype, device=initial.device)
    return mask, values


def unpack(q: torch.Tensor, plan: BlockPlan, prescribed: torch.Tensor) -> torch.Tensor:
    p = prescribed.clone(); p[plan.free_start:plan.free_end] = q.reshape(plan.count, 3); return p


def residual(q: torch.Tensor, old_p: torch.Tensor, old_v: torch.Tensor, rest: torch.Tensor, masses: torch.Tensor, integrals: torch.Tensor, previous: torch.Tensor, pulse: torch.Tensor, prescribed: torch.Tensor, plan: BlockPlan, args: argparse.Namespace, dt: float, jacobian: bool, frozen_faces: arc_mesh_contact.MeshFaces | None = None, static_equilibrium: bool = False):
    p = unpack(q, plan, prescribed)
    if jacobian:
        gradient, system = assemble_elastic_banded(p, rest, plan, args.EA, args.EI)
        contact, blocks = assemble_contact_banded(p, integrals, previous, args, dt, frozen_faces)
    else:
        qg = q.detach().clone().requires_grad_(True)
        pg = unpack(qg, plan, prescribed)
        gradient = torch.autograd.grad(elastic_energy(pg, rest, args.EA, args.EI)[2], qg)[0]
        contact = contact_response(p, integrals, previous, args, dt, frozen_faces)[0]; system = None; blocks = None
    if static_equilibrium:
        # Final quasi-static corrector: no pseudo-time inertia or damping.
        r = gradient - contact[plan.free_start:plan.free_end].reshape(-1) - pulse[plan.free_start:plan.free_end].reshape(-1)
    else:
        old_q, old_vel = old_p[plan.free_start:plan.free_end].reshape(-1), old_v[plan.free_start:plan.free_end].reshape(-1)
        mass = masses[plan.free_start:plan.free_end, None].expand(-1, 3).reshape(-1)
        r = mass * (q - old_q - dt * old_vel) / dt**2 + gradient + args.damping * (q - old_q) / dt - contact[plan.free_start:plan.free_end].reshape(-1) - pulse[plan.free_start:plan.free_end].reshape(-1)
    if jacobian:
        if not static_equilibrium:
            add = (mass / dt**2 + args.damping / dt).reshape(-1, 3)
            system.diagonal.diagonal(dim1=1, dim2=2).add_(add)
        system.diagonal.add_(-blocks[plan.free_start:plan.free_end])
    return r, system, p


def implicit_step(old_p: torch.Tensor, old_v: torch.Tensor, rest: torch.Tensor, masses: torch.Tensor, integrals: torch.Tensor, pulse: torch.Tensor, constrained: bool, displacement: float, args: argparse.Namespace, dt: float):
    mask, prescribed = prescribed_state(args.initial_positions, displacement, constrained, args)
    plan = block_plan(len(old_p), constrained)
    with torch.no_grad(): previous = contact_response(old_p, integrals, torch.zeros_like(integrals), args, dt)[4]
    q = (old_p[plan.free_start:plan.free_end] + dt * old_v[plan.free_start:plan.free_end]).reshape(-1).detach()
    frozen_faces = mesh_geometry(args).nearest_faces(unpack(q, plan, prescribed), args.mesh_normal_mode) if args.contact_type == "mesh" else None
    failures, converged, residual_norm, step_norm = 0, False, math.inf, math.inf
    for iteration in range(1, args.max_newton_iters + 1):
        current, _, _ = residual(q, old_p, old_v, rest, masses, integrals, previous, pulse, prescribed, plan, args, dt, False, frozen_faces)
        residual_norm = float(torch.linalg.vector_norm(current))
        if residual_norm < args.residual_tol: converged, step_norm = True, 0.0; break
        r, system, _ = residual(q, old_p, old_v, rest, masses, integrals, previous, pulse, prescribed, plan, args, dt, True, frozen_faces)
        delta = system.solve(-r); step_norm = float(torch.linalg.vector_norm(delta))
        base, accepted = residual_norm, False
        for exponent in range(10):
            candidate = (q + 0.5**exponent * delta).detach()
            trial, _, _ = residual(candidate, old_p, old_v, rest, masses, integrals, previous, pulse, prescribed, plan, args, dt, False, frozen_faces)
            norm = float(torch.linalg.vector_norm(trial))
            if np.isfinite(norm) and norm < base: q, residual_norm, accepted = candidate, norm, True; break
        if not accepted: failures += 1; q = (q + 0.05 * delta).detach()
        if step_norm < args.step_tol: converged = residual_norm < 10.0 * args.residual_tol; break
    p = unpack(q, plan, prescribed).detach(); v = ((p - old_p) / dt).detach(); v[mask] = 0.0
    force, new_i, active, weights, errors, clearance, centerline, valid = contact_response(p, integrals, previous, args, dt)
    return p, v, new_i.detach(), force.detach(), active.detach(), weights.detach(), errors.detach(), clearance.detach(), NewtonInfo(converged, iteration, residual_norm, step_norm, failures), mask


def static_contact_corrector(state, rest: torch.Tensor, masses: torch.Tensor, constrained: bool,
                             displacement: float, args: argparse.Namespace):
    """Finish a quasi-static load step with ``elastic force - contact force = 0``.

    Dynamic implicit Euler is retained to select the physical buckling branch.
    A faceted, frictionless OBJ can then keep a tiny velocity while its PI
    force balances damping.  This active-set Newton corrector removes only
    inertia/damping, freezes one selected face set per Newton solve, and
    refreshes it between a few outer iterations.  It never replaces a failed
    Newton/contact solve with a manual velocity reset.
    """
    p, _v, integrals = state
    mask, prescribed = prescribed_state(args.initial_positions, displacement, constrained, args)
    plan = block_plan(len(p), constrained)
    zero_velocity, zero_pulse = torch.zeros_like(p), torch.zeros_like(p)
    q = p[plan.free_start:plan.free_end].reshape(-1).detach()
    last = None
    for _outer in range(6):
        with torch.no_grad():
            previous = contact_response(p, integrals, torch.zeros_like(integrals), args, 0.0)[4]
        frozen_faces = mesh_geometry(args).nearest_faces(unpack(q, plan, prescribed), args.mesh_normal_mode) if args.contact_type == "mesh" else None
        converged, residual_norm, step_norm, failures = False, math.inf, math.inf, 0
        for iteration in range(1, args.max_newton_iters + 1):
            current, _, _ = residual(q, p, zero_velocity, rest, masses, integrals, previous, zero_pulse, prescribed, plan, args, 0.0, False, frozen_faces, True)
            residual_norm = float(torch.linalg.vector_norm(current))
            if residual_norm < args.residual_tol:
                converged, step_norm = True, 0.0
                break
            r, system, _ = residual(q, p, zero_velocity, rest, masses, integrals, previous, zero_pulse, prescribed, plan, args, 0.0, True, frozen_faces, True)
            delta = system.solve(-r)
            step_norm, base, accepted = float(torch.linalg.vector_norm(delta)), residual_norm, False
            for exponent in range(12):
                candidate = (q + 0.5 ** exponent * delta).detach()
                trial, _, _ = residual(candidate, p, zero_velocity, rest, masses, integrals, previous, zero_pulse, prescribed, plan, args, 0.0, False, frozen_faces, True)
                trial_norm = float(torch.linalg.vector_norm(trial))
                if np.isfinite(trial_norm) and trial_norm < base:
                    q, residual_norm, accepted = candidate, trial_norm, True
                    break
            if not accepted:
                failures += 1
                q = (q + 0.05 * delta).detach()
            if step_norm < args.step_tol:
                converged = residual_norm < 10.0 * args.residual_tol
                break
        p = unpack(q, plan, prescribed).detach()
        with torch.no_grad():
            previous = contact_response(p, integrals, torch.zeros_like(integrals), args, 0.0)[4]
            force, new_i, active, weights, errors, clearance, _reference, _valid = contact_response(p, integrals, previous, args, 0.0)
        # Re-evaluate with refreshed faces.  If an edge changed active face,
        # another outer solve starts from this point with the new topology.
        refreshed, _, _ = residual(q, p, zero_velocity, rest, masses, integrals, previous, zero_pulse, prescribed, plan, args, 0.0, False, None, True)
        refreshed_norm = float(torch.linalg.vector_norm(refreshed))
        info = NewtonInfo(converged and refreshed_norm < args.settle_residual_tol, iteration, refreshed_norm, step_norm, failures)
        last = (p, torch.zeros_like(p), new_i.detach(), force.detach(), active.detach(), weights.detach(), errors.detach(), clearance.detach(), info, mask)
        if info.converged and float(protection_error(p, args).max()) < args.contact_tolerance:
            return (p, torch.zeros_like(p), new_i.detach()), last, True
        q = p[plan.free_start:plan.free_end].reshape(-1).detach()
    return (p, torch.zeros_like(p), new_i.detach()), last, False


def max_speed(v: torch.Tensor, mask: torch.Tensor) -> float:
    free = torch.any(~mask, dim=1)
    return float(torch.linalg.vector_norm(v[free], dim=1).max())


def protection_error(p: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    force, _, _, _, _, clearance, _, valid = contact_response(p, torch.zeros(len(p), dtype=p.dtype, device=p.device), torch.zeros(len(p), dtype=p.dtype, device=p.device), args, args.dt)
    del force
    return torch.clamp(-clearance, min=0.0) * valid


def settle(state, rest, masses, pulse, constrained, displacement, args, limit: int, label: str, recorder=None):
    p, v, i = state; stable = 0; last = None
    stalled_checks, prior_stall_speed = 0, None
    for step in range(1, limit + 1):
        last = implicit_step(p, v, rest, masses, i, pulse, constrained, displacement, args, args.dt)
        p, v, i = last[:3]
        info, mask = last[8], last[9]
        if recorder is not None and (step % args.history_save_interval == 0): recorder((p, v, i), last, step)
        okay = info.converged and max_speed(v, mask) < args.settle_speed_tol and info.residual_norm < args.settle_residual_tol and float(protection_error(p, args).max()) < args.contact_tolerance
        stable = stable + 1 if okay else 0
        if args.settle_diagnostic_interval and step % args.settle_diagnostic_interval == 0:
            contact_count = int(last[4].sum())
            print(
                f"[RELAX {label} | {step:4d}/{limit}] v={max_speed(v, mask):.2e} m/s | "
                f"R={info.residual_norm:.2e} N | wall_error={float(protection_error(p, args).max()) * 1e3:.4f} mm | "
                f"contacts={contact_count} | Newton={info.iterations} | stable={stable}",
                flush=True,
            )
        # Do not spend 1600 pseudo-time steps in a face-normal limit cycle.
        # Once force/contact tolerances are met but velocity has stopped
        # decaying for three diagnostics, the caller invokes the static
        # active-set corrector rather than weakening the physical criterion.
        stall_interval = max(100, args.settle_diagnostic_interval or 100)
        strict_mesh = args.contact_type == "mesh" and args.mesh_normal_mode == "face"
        current_speed = max_speed(v, mask)
        if strict_mesh and step % stall_interval == 0 and info.converged and info.residual_norm < args.settle_residual_tol and float(protection_error(p, args).max()) < args.contact_tolerance and current_speed >= args.settle_speed_tol:
            if prior_stall_speed is not None and abs(current_speed - prior_stall_speed) <= 0.02 * max(prior_stall_speed, args.settle_speed_tol):
                stalled_checks += 1
            else:
                stalled_checks = 0
            prior_stall_speed = current_speed
            if stalled_checks >= 2:
                print(f"[RELAX {label}] face-contact velocity plateau at {current_speed:.2e} m/s; switching to static corrector", flush=True)
                return (p, v, i), last, False, step
        elif step % stall_interval == 0:
            stalled_checks, prior_stall_speed = 0, current_speed
        if stable >= args.settle_confirm_steps: return (p, v, i), last, True, step
    return (p, v, i), last, False, limit


def advance_with_retries(state, rest, masses, pulse, constrained, start_displacement: float,
                         end_displacement: float, args):
    """Advance one physical ``args.dt`` with transactional 1/2/4/8 substeps.

    A retry always starts from the same accepted state.  Thus a Newton failure
    while a node crosses a triangle edge cannot leave a partially-updated PI
    integral or velocity in the next attempt.
    """
    last = None
    for substeps in (1, 2, 4, 8):
        trial, failed = state, False
        for index in range(1, substeps + 1):
            fraction = index / substeps
            displacement = start_displacement + (end_displacement - start_displacement) * fraction
            last = implicit_step(
                trial[0], trial[1], rest, masses, trial[2], pulse, constrained,
                displacement, args, args.dt / substeps,
            )
            if not last[8].converged:
                failed = True
                break
            trial = last[:3]
        if not failed:
            return True, trial, last, substeps
    return False, state, last, 0


def append_history(history: dict[str, list], phase: str, time_s: float, load_step: int, displacement: float, state, last, rest, args):
    p, v, i = state; force, active, weights, errors, clearance, info, mask = last[3], last[4], last[5], last[6], last[7], last[8], last[9]
    stretch, bend, total = elastic_energy(p, rest, args.EA, args.EI)
    for key, value in {
        "phase": phase, "time": time_s, "load_step": load_step, "tip_displacement": displacement,
        "positions": p.cpu().numpy().copy(), "velocities": v.cpu().numpy().copy(), "pi_forces": force.cpu().numpy().copy(),
        "pi_integrals": i.cpu().numpy().copy(), "contact_active": active.cpu().numpy().copy(), "contact_weight": weights.cpu().numpy().copy(),
        "contact_error": errors.cpu().numpy().copy(), "max_protection_error": float(protection_error(p, args).max()),
        "max_pi_force": float(torch.linalg.vector_norm(force, dim=1).max()), "newton_iterations": info.iterations,
        "newton_residual": info.residual_norm, "newton_converged": info.converged,
        "elastic_stretch_energy_j": float(stretch), "elastic_bend_energy_j": float(bend), "elastic_total_energy_j": float(total),
    }.items(): history[key].append(value)


def run_simulation(args: argparse.Namespace) -> dict[str, np.ndarray]:
    raw_p, raw_v, raw_rest, raw_masses, raw_i = initialize_rod(args)
    device = args.device
    p, v = torch.as_tensor(raw_p, dtype=DTYPE, device=device), torch.as_tensor(raw_v, dtype=DTYPE, device=device)
    rest, masses, integrals = (torch.as_tensor(x, dtype=DTYPE, device=device) for x in (raw_rest, raw_masses, raw_i))
    args.initial_positions = p.clone()
    print(
        f"[CONFIG] device={device}, dtype={DTYPE}, nodes={args.num_nodes}, "
        f"R={args.arc_radius:.6g} m, angle={args.arc_angle_deg:g} deg, "
        f"contact={args.contact_type}, PI contact radius={args.contact_radius:.6g} m, ramp={args.ramp_mode}",
        flush=True,
    )
    if args.contact_type == "mesh":
        geometry = mesh_geometry(args)
        print(f"[MESH] obj={geometry.path}; vertices={len(geometry.vertices)}; triangles={len(geometry.triangles)}; normal_mode={args.mesh_normal_mode}; open ends=yes", flush=True)
    nodes, magnitudes, directions, raw_pulse = make_random_pulse(args, raw_p)
    start_theta = args.arc_angle_rad
    end_theta = args.arc_angle_rad - args.tip_total_displacement / args.arc_radius
    start_direction = np.array((-math.cos(start_theta), -math.sin(start_theta), 0.0))
    end_direction = np.array((-math.cos(end_theta), -math.sin(end_theta), 0.0))
    print(f"[TIP COMPRESSION] arc-follow toward inlet; retreat={args.tip_total_displacement * 1e3:.3f} mm; direction rotates from {np.round(start_direction, 6).tolist()} to {np.round(end_direction, 6).tolist()}", flush=True)
    print(f"[INITIAL 3-D PULSES] seed={args.pulse_seed}; nodes={nodes.tolist()}; magnitudes_N={np.round(magnitudes, 6).tolist()}", flush=True)
    for node, magnitude, direction, vector in zip(nodes, magnitudes, directions, raw_pulse[nodes]):
        print(f"  node={int(node):3d} | |F|={magnitude:.6e} N | unit_direction={np.round(direction, 6).tolist()} | force_N={np.round(vector, 6).tolist()}", flush=True)
    pulse, zero = torch.as_tensor(raw_pulse, dtype=DTYPE, device=device), torch.zeros_like(p)
    keys = ("phase", "time", "load_step", "tip_displacement", "positions", "velocities", "pi_forces", "pi_integrals", "contact_active", "contact_weight", "contact_error", "max_protection_error", "max_pi_force", "newton_iterations", "newton_residual", "newton_converged", "elastic_stretch_energy_j", "elastic_bend_energy_j", "elastic_total_energy_j")
    history = {k: [] for k in keys}; time_s = 0.0; state = (p, v, integrals)
    started = time.perf_counter()
    phase_wall = {"prebalance": 0.0, "pulse": 0.0, "ramp": 0.0, "relax": 0.0}
    # The initially curved configuration has zero stretch but positive bending energy.
    initial_last = (p, v, integrals, zero, torch.zeros(args.num_nodes, dtype=torch.bool, device=device), torch.zeros(args.num_nodes, dtype=DTYPE, device=device), torch.zeros(args.num_nodes, dtype=DTYPE, device=device), torch.full((args.num_nodes,), args.tube_radius, dtype=DTYPE, device=device), NewtonInfo(True, 0, 0.0, 0.0, 0), torch.zeros_like(p, dtype=torch.bool))
    append_history(history, "initial", time_s, 0, 0.0, state, initial_last, rest, args)
    initial_stretch, initial_bend, initial_total = elastic_energy(p, rest, args.EA, args.EI)
    phase_started = time.perf_counter()
    prebalance_time = time_s
    state, last, ok, steps = settle(state, rest, masses, zero, True, 0.0, args, args.prebalance_max_steps, "prebalance", lambda recorded_state, recorded_last, step: append_history(history, "prebalance", prebalance_time + step * args.dt, 0, 0.0, recorded_state, recorded_last, rest, args))
    phase_wall["prebalance"] += time.perf_counter() - phase_started
    if not ok: raise RuntimeError(f"prebalance did not settle after {steps} steps")
    time_s += steps * args.dt
    append_history(history, "prebalance", time_s, 0, 0.0, state, last, rest, args)
    # Pulse has a fixed tip so it only selects a three-dimensional buckling branch.
    elapsed = 0.0
    phase_started = time.perf_counter()
    while elapsed < args.pulse_duration - 1.0e-15:
        dt = min(args.dt, args.pulse_duration - elapsed)
        # The pulse may be much stiffer than a held displacement increment.
        # Keep its physical duration unchanged but retry it with local substeps.
        accepted = False
        for substeps in (1, 2, 4, 8):
            trial, trial_last, failed = state, None, False
            for _ in range(substeps):
                trial_last = implicit_step(trial[0], trial[1], rest, masses, trial[2], pulse, True, 0.0, args, dt / substeps)
                if not trial_last[8].converged:
                    failed = True
                    break
                trial = trial_last[:3]
            if not failed:
                state, last, accepted = trial, trial_last, True
                break
        if not accepted:
            raise RuntimeError(f"Newton failed during pulse: {trial_last[8].residual_norm:.3e}")
        elapsed += dt; time_s += dt
        append_history(history, "pulse", time_s, 0, 0.0, state, last, rest, args)
    phase_wall["pulse"] += time.perf_counter() - phase_started
    completed = 0
    adaptive_accepted = adaptive_rejected = adaptive_micro_steps = 0
    for load in range(1, args.num_load_steps + 1):
        target = load * args.tip_displacement_step; previous = (load - 1) * args.tip_displacement_step
        phase_started = time.perf_counter()
        if args.ramp_mode == "fixed":
            for ramp in range(1, args.tip_ramp_steps + 1):
                f0, f1 = (ramp - 1) / args.tip_ramp_steps, ramp / args.tip_ramp_steps
                smooth = lambda f: f*f*(3.0 - 2.0*f)
                start, end = previous + (target - previous) * smooth(f0), previous + (target - previous) * smooth(f1)
                accepted, trial, trial_last, _ = advance_with_retries(state, rest, masses, zero, True, start, end, args)
                if not accepted: raise RuntimeError(f"Newton failed at load {load}, fixed ramp {ramp}")
                state, last = trial, trial_last
                time_s += args.dt
                append_history(history, "load_ramp", time_s, load, end, state, last, rest, args)
        else:
            # This is the straight-tube v4 continuation strategy adapted to
            # the arc-following endpoint: high velocity or a failed Newton
            # trial rolls the entire state back and halves the displacement.
            minimum_increment = (target - previous) / args.tip_ramp_steps
            increment, displacement, internal = 0.5 * (target - previous), previous, 0
            grow_limit = args.adaptive_speed_grow_factor * args.settle_speed_tol
            shrink_limit = args.adaptive_speed_shrink_factor * args.settle_speed_tol
            while displacement < target - 1.0e-15:
                internal += 1
                proposed = min(displacement + increment, target)
                checkpoint = state
                accepted, trial_state, trial_last, subdivisions = advance_with_retries(
                    checkpoint, rest, masses, zero, True, displacement, proposed, args,
                )
                can_shrink = increment > minimum_increment * (1.0 + 1.0e-12)
                if not accepted:
                    adaptive_rejected += 1
                    if can_shrink:
                        increment = max(0.5 * increment, minimum_increment)
                        print(f"[ADAPT ROLLBACK {load:02d}/{args.num_load_steps:02d}] Newton retry failed; next ds={increment * 1e3:.4f} mm", flush=True)
                        continue
                    raise RuntimeError(f"Newton failed at minimum adaptive increment in load {load}")
                trial_speed = max_speed(trial_last[1], trial_last[9])
                micro_steps, micro_failed = 0, False
                # Hold the just-reached tip location before deciding whether
                # the trial is dynamically gentle enough to accept.
                while micro_steps < args.adaptive_relax_steps and (micro_steps == 0 or trial_speed > shrink_limit):
                    micro_steps += 1
                    micro_ok, micro_state, micro_last, _ = advance_with_retries(
                        trial_state, rest, masses, zero, True, proposed, proposed, args,
                    )
                    if not micro_ok:
                        micro_failed = True
                        break
                    trial_state, trial_last = micro_state, micro_last
                    trial_speed = max_speed(trial_last[1], trial_last[9])
                too_dynamic = trial_speed > shrink_limit
                if (micro_failed or too_dynamic) and can_shrink:
                    adaptive_rejected += 1
                    increment = max(0.5 * increment, minimum_increment)
                    reason = "micro relaxation failed" if micro_failed else f"v={trial_speed:.2e} m/s"
                    print(f"[ADAPT ROLLBACK {load:02d}/{args.num_load_steps:02d}] ds={(proposed - displacement) * 1e3:.4f} mm, {reason}; next ds={increment * 1e3:.4f} mm", flush=True)
                    continue
                if micro_failed:
                    raise RuntimeError(f"adaptive micro relaxation failed at minimum increment in load {load}")
                accepted_delta = proposed - displacement
                state, last, displacement = trial_state, trial_last, proposed
                adaptive_accepted += 1; adaptive_micro_steps += micro_steps
                time_s += args.dt * (1 + micro_steps)
                append_history(history, "load_adaptive", time_s, load, displacement, state, last, rest, args)
                print(f"[ADAPT {load:02d}/{args.num_load_steps:02d}] tip={displacement * 1e3:.3f} mm | ds={accepted_delta * 1e3:.3f} mm | v={trial_speed:.2e} m/s | micro={micro_steps} | contacts={int(last[4].sum())}", flush=True)
                if trial_speed < grow_limit and displacement < target:
                    increment = min(1.75 * increment, target - displacement)
                elif too_dynamic:
                    increment = max(0.5 * increment, minimum_increment)
        phase_wall["ramp"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        relaxation_time = time_s
        state, last, ok, steps = settle(state, rest, masses, zero, True, target, args, args.max_settle_steps, f"load {load}", lambda recorded_state, recorded_last, step: append_history(history, "load_relax", relaxation_time + step * args.dt, load, target, recorded_state, recorded_last, rest, args))
        phase_wall["relax"] += time.perf_counter() - phase_started
        if not ok and args.contact_type == "mesh" and args.mesh_normal_mode == "face":
            # A strict faceted wall may have already met force/contact
            # tolerances but remain in a damping-balanced micro-slide.  Solve
            # the static active-set equations instead of loosening the 2-D
            # speed criterion.
            correction_started = time.perf_counter()
            state, last, ok = static_contact_corrector(state, rest, masses, True, target, args)
            phase_wall["relax"] += time.perf_counter() - correction_started
            if ok:
                append_history(history, "load_static", time_s, load, target, state, last, rest, args)
                print(f"[STATIC CORRECTOR {load:02d}/{args.num_load_steps}] static residual={last[8].residual_norm:.2e} N; contacts={int(last[4].sum())}", flush=True)
        if not ok: raise RuntimeError(f"load {load} did not settle after {steps} steps and static correction")
        time_s += steps * args.dt
        completed += 1; append_history(history, "load_relax", time_s, load, target, state, last, rest, args)
        print(f"[LOAD {load:02d}/{args.num_load_steps:02d}] settled in {steps} steps; contacts={int(last[4].sum())}", flush=True)
    output = {key: np.asarray(value) for key, value in history.items()}
    final_stretch, final_bend, final_total = elastic_energy(state[0], rest, args.EA, args.EI)
    final_force, final_active, final_info, final_mask = last[3], last[4], last[8], last[9]
    mesh_meta = mesh_geometry(args).stats() if args.contact_type == "mesh" else {}
    final_kinetic = 0.5 * torch.sum(masses[:, None] * state[1].square())
    output.update({"initial_positions": raw_p, "arc_radius": np.asarray(args.arc_radius), "arc_angle_deg": np.asarray(args.arc_angle_deg), "contact_radius": np.asarray(args.contact_radius), "tube_radius": np.asarray(args.tube_radius), "safe_distance": np.asarray(args.safe_distance), "contact_type": np.asarray(args.contact_type), "mesh_obj": np.asarray(str(args.mesh_obj) if args.mesh_obj else ""), "pulse_seed": np.asarray(args.pulse_seed), "initial_pulse_nodes": nodes, "initial_pulse_magnitudes_n": magnitudes, "initial_pulse_unit_directions": directions, "initial_pulse_force_vectors_n": raw_pulse[nodes], "completed_load_steps": np.asarray(completed), "wall_clock_seconds": np.asarray(time.perf_counter() - started), "phase_prebalance_wall_seconds": np.asarray(phase_wall["prebalance"]), "phase_pulse_wall_seconds": np.asarray(phase_wall["pulse"]), "phase_ramp_wall_seconds": np.asarray(phase_wall["ramp"]), "phase_relax_wall_seconds": np.asarray(phase_wall["relax"]), "initial_elastic_stretch_energy_j": np.asarray(float(initial_stretch)), "initial_elastic_bend_energy_j": np.asarray(float(initial_bend)), "initial_elastic_energy_j": np.asarray(float(initial_total)), "final_elastic_stretch_energy_j": np.asarray(float(final_stretch)), "final_elastic_bend_energy_j": np.asarray(float(final_bend)), "final_elastic_energy_j": np.asarray(float(final_total)), "final_kinetic_energy_j": np.asarray(float(final_kinetic)), "final_kinetic_to_elastic_ratio": np.asarray(float(final_kinetic / final_total.clamp_min(1.0e-30))), "final_active_contact_nodes": np.asarray(int(final_active.sum())), "final_max_pi_force_n": np.asarray(float(torch.linalg.vector_norm(final_force, dim=1).max())), "final_max_protection_error_m": np.asarray(float(protection_error(state[0], args).max())), "final_newton_residual_n": np.asarray(final_info.residual_norm), "final_free_node_speed_mps": np.asarray(max_speed(state[1], final_mask)), **{key: np.asarray(value) for key, value in mesh_meta.items()}})
    output.update({
        "mesh_normal_mode": np.asarray(args.mesh_normal_mode), "ramp_mode": np.asarray(args.ramp_mode),
        "adaptive_accepted_steps": np.asarray(adaptive_accepted), "adaptive_rejected_steps": np.asarray(adaptive_rejected),
        "adaptive_micro_relaxation_steps": np.asarray(adaptive_micro_steps),
    })
    path = save_results(output, args); output["npz_path"] = np.asarray(str(path))
    print(f"simulation complete: completed_load_steps={completed}/{args.num_load_steps}, wall_time={float(output['wall_clock_seconds']):.2f}s, npz={path}", flush=True)
    return output


def output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = Path(args.output_prefix) if args.output_prefix else Path(__file__).parent / "outputs" / "arc3d"
    return root.with_suffix(".npz"), root.with_suffix(".png"), root.with_suffix(".gif")


def save_results(result: dict[str, np.ndarray], args: argparse.Namespace) -> Path:
    path, _, _ = output_paths(args); path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **result); return path


def print_final_diagnostics(result: dict[str, np.ndarray], history: dict[str, list], args: argparse.Namespace, npz_path: Path) -> None:
    """Terminal summary mirrors the 2-D arc diagnostics and exposes bottlenecks."""
    iterations, residuals = np.asarray(history["newton_iterations"]), np.asarray(history["newton_residual"])
    print("\n========== FINAL 3-D ARC GUIDEWIRE DIAGNOSTICS =========="); print(f"history saved : {npz_path}")
    print(f"completed load steps : {int(result['completed_load_steps'])}/{args.num_load_steps}")
    print(f"wall-clock time      : {float(result['wall_clock_seconds']):.3f} s")
    print(f"contact type         : {args.contact_type}")
    if args.contact_type == "mesh":
        print(f"mesh OBJ             : {args.mesh_obj}")
        print(f"mesh normal mode     : {str(result['mesh_normal_mode'].item())}")
        print(f"mesh vertices/triangles : {int(result['mesh_vertex_count'])}/{int(result['mesh_triangle_count'])} | cache hit/miss={int(result['mesh_cache_hits'])}/{int(result['mesh_cache_misses'])}")
    print(f"ramp mode            : {str(result['ramp_mode'].item())} | adaptive accepted/rejected/micro={int(result['adaptive_accepted_steps'])}/{int(result['adaptive_rejected_steps'])}/{int(result['adaptive_micro_relaxation_steps'])}")
    print("PHASE WALL TIME (where the run spent its time)")
    print(f"  prebalance={float(result['phase_prebalance_wall_seconds']):.3f} s | pulse={float(result['phase_pulse_wall_seconds']):.3f} s | ramp={float(result['phase_ramp_wall_seconds']):.3f} s | relaxation={float(result['phase_relax_wall_seconds']):.3f} s")
    print("ELASTIC ENERGY (PI contact is non-conservative and excluded)")
    print(f"  initial stretch / bend / total : {float(result['initial_elastic_stretch_energy_j']):.6e} / {float(result['initial_elastic_bend_energy_j']):.6e} / {float(result['initial_elastic_energy_j']):.6e} J")
    print(f"  final   stretch / bend / total : {float(result['final_elastic_stretch_energy_j']):.6e} / {float(result['final_elastic_bend_energy_j']):.6e} / {float(result['final_elastic_energy_j']):.6e} J")
    print(f"  elastic-energy change           : {float(result['final_elastic_energy_j'] - result['initial_elastic_energy_j']):.6e} J")
    print(f"  final kinetic energy / elastic  : {float(result['final_kinetic_energy_j']):.6e} J / {float(result['final_kinetic_to_elastic_ratio']):.3e}")
    print("FINAL EQUILIBRIUM / CONTACT")
    print(f"  Newton residual={float(result['final_newton_residual_n']):.3e} N | free-node speed={float(result['final_free_node_speed_mps']):.3e} m/s")
    print(f"  protection error={float(result['final_max_protection_error_m']) * 1e3:.4f} mm | active contact nodes={int(result['final_active_contact_nodes'])} | max PI force={float(result['final_max_pi_force_n']):.3e} N")
    print("SOLVER HISTORY (saved animation frames)")
    print(f"  frames={len(iterations)} | Newton iterations mean/max={iterations.mean():.2f}/{iterations.max()} | residual max={residuals.max():.3e} N | non-converged frames={int((~np.asarray(history['newton_converged'], dtype=bool)).sum())}")
    print("===========================================================\n")


def tube_surface(args: argparse.Namespace, radius: float, theta_count: int = 50, phi_count: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta, phi = np.linspace(0.0, args.arc_angle_rad, theta_count), np.linspace(0.0, 2.0 * math.pi, phi_count)
    tt, pp = np.meshgrid(theta, phi)
    cx, cy = args.arc_radius * np.sin(tt), args.arc_radius * (1.0 - np.cos(tt))
    nx, ny = np.sin(tt), -np.cos(tt)
    return cx + radius * np.cos(pp) * nx, cy + radius * np.cos(pp) * ny, radius * np.sin(pp)


def draw_tube(axis, args: argparse.Namespace) -> None:
    """Draw exactly the contact geometry selected by the solver."""
    if args.contact_type == "mesh":
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        geometry = mesh_geometry(args)
        surface = Poly3DCollection(
            geometry.vertices[geometry.faces], facecolor="darkorange", edgecolor="none", alpha=.24,
        )
        axis.add_collection3d(surface)
        axis.auto_scale_xyz(geometry.vertices[:, 0], geometry.vertices[:, 1], geometry.vertices[:, 2])
        axis.plot([], [], [], color="darkorange", lw=5, alpha=.7,
                  label=f"OBJ PI surface: {args.contact_radius * 1e3:.0f} mm")
        return
    X, Y, Z = tube_surface(args, args.tube_radius)
    PX, PY, PZ = tube_surface(args, args.contact_radius)
    axis.plot_surface(X, Y, Z, color="lightsteelblue", alpha=.12, linewidth=0)
    axis.plot_surface(PX, PY, PZ, color="darkorange", alpha=.22, linewidth=0)
    axis.plot([], [], [], color="darkorange", lw=5, alpha=.7,
              label=f"PI contact surface: {args.contact_radius * 1e3:.0f} mm")
    axis.plot([], [], [], color="lightsteelblue", lw=5, alpha=.7,
              label=f"visible shell: {args.tube_radius * 1e3:.0f} mm")


def create_animation(history: dict[str, np.ndarray], args: argparse.Namespace) -> Path:
    """Save a GIF, then (unless --no-show) play it until the user closes its window."""
    import matplotlib
    if args.no_show: matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    _, _, gif_path = output_paths(args); gif_path.parent.mkdir(parents=True, exist_ok=True)
    positions, times = history["positions"], history["time"]
    frames = np.arange(0, len(positions), args.animation_stride, dtype=int)
    if frames[-1] != len(positions) - 1: frames = np.append(frames, len(positions) - 1)
    figure = plt.figure(figsize=(10, 8), dpi=120); axis = figure.add_subplot(111, projection="3d")
    draw_tube(axis, args)
    axis.plot(*positions[0].T, "k--", lw=1.0, label="initial")
    rod, = axis.plot([], [], [], color="tab:blue", lw=1.8, label="guidewire")
    fixed = axis.scatter([], [], [], marker="s", color="red", s=28, label="fixed inlet")
    contacts = axis.scatter([], [], [], color="magenta", s=12, label="PI contact")
    title = axis.set_title("")
    axis.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)"); axis.set_box_aspect((args.arc_radius, args.arc_radius, .4 * args.arc_radius)); axis.legend(loc="upper left")

    def update(frame_number: int):
        frame = int(frames[frame_number]); points = positions[frame]
        rod.set_data(points[:, 0], points[:, 1]); rod.set_3d_properties(points[:, 2])
        fixed._offsets3d = (points[:2, 0], points[:2, 1], points[:2, 2])
        active = history["contact_active"][frame]
        selected = points[active]
        contacts._offsets3d = (selected[:, 0], selected[:, 1], selected[:, 2]) if len(selected) else ([], [], [])
        title.set_text(f"3-D arc guidewire | phase={history['phase'][frame]} | t={times[frame]:.3f} s | load={history['load_step'][frame]} | tip retreat={history['tip_displacement'][frame] * 1e3:.2f} mm")
        return rod, fixed, contacts, title

    animation = FuncAnimation(figure, update, frames=len(frames), interval=1000.0 / args.animation_fps, blit=False, repeat=args.animation_repeat)
    animation.save(gif_path, writer=PillowWriter(fps=args.animation_fps), dpi=100)
    print(f"animation saved: {gif_path} ({len(frames)} frames, stride={args.animation_stride})", flush=True)
    if not args.no_show:
        # User closes this replay window; only then is the final static summary generated.
        figure._guidewire_animation = animation
        plt.show()
    plt.close(figure)
    return gif_path


def visualize_results(history: dict[str, np.ndarray], args: argparse.Namespace) -> Path:
    """Generate the final summary only after the animation window has closed."""
    import matplotlib
    if args.no_show: matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    _, image, _ = output_paths(args); image.parent.mkdir(parents=True, exist_ok=True)
    positions = history["positions"]
    fig = plt.figure(figsize=(15, 9), dpi=160); ax = fig.add_subplot(221, projection="3d")
    draw_tube(ax, args)
    ax.plot(*positions[0].T, "k--", lw=1, label="initial"); ax.plot(*positions[-1].T, color="tab:blue", lw=1.8, label="final")
    active = history["contact_active"][-1]
    if active.any(): ax.scatter(*positions[-1, active].T, color="magenta", s=18, label="PI contact")
    ax.set(title=f"3-D guidewire in {args.contact_type} circular-arc tube", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)"); ax.legend()
    ax.set_box_aspect((args.arc_radius, args.arc_radius, .4*args.arc_radius))
    for index, (key, label) in enumerate((("elastic_total_energy_j", "elastic energy (J)"), ("newton_residual", "Newton residual (N)"), ("max_protection_error", "protection error (m)")), start=2):
        a = fig.add_subplot(2, 2, index); a.plot(history["time"], history[key]); a.set(xlabel="simulation time (s)", ylabel=label); a.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(image); plt.close(fig); return image


def self_test(args: argparse.Namespace) -> None:
    raw_p, _, raw_rest, _, _ = initialize_rod(args)
    assert np.isclose(raw_rest.sum(), args.total_length, rtol=1e-12)
    p = torch.as_tensor(raw_p, dtype=DTYPE); rest = torch.as_tensor(raw_rest, dtype=DTYPE)
    stretch, bend, _ = elastic_energy(p, rest, args.EA, args.EI)
    assert abs(float(stretch)) < 1.0e-12 and float(bend) > 0.0
    mid = args.num_nodes // 2; _, normal, binormal = local_frames_numpy(raw_p, args.arc_radius)
    original_type = args.contact_type
    for contact_type in ("analytic", "mesh"):
        args.contact_type = contact_type
        zero = torch.zeros(args.num_nodes, dtype=DTYPE)
        force, *_ = contact_response(p, zero, zero, args, args.dt)
        assert torch.allclose(force, torch.zeros_like(force), atol=1e-12)
        for direction in (normal[mid], binormal[mid]):
            q = p[mid:mid+1].clone() + torch.as_tensor(direction[None] * (args.contact_radius + 5.0e-3), dtype=DTYPE)
            f, *_ = contact_response(q, torch.zeros(1, dtype=DTYPE), torch.zeros(1, dtype=DTYPE), args, args.dt)
            assert float(torch.sum(f * torch.as_tensor(direction, dtype=DTYPE))) < 0.0, f"{contact_type} contact does not point inward"
        outside = torch.tensor([[-.1, 0., 0.]], dtype=DTYPE)
        f, *_ = contact_response(outside, torch.zeros(1, dtype=DTYPE), torch.zeros(1, dtype=DTYPE), args, args.dt)
        assert torch.allclose(f, torch.zeros_like(f)), f"{contact_type} open end must not add a cap force"
        if contact_type == "mesh":
            # The OBJ path must not create a pre-contact force merely because
            # a node lies exactly on a selected triangle plane.  Also check
            # the new closed-form rank-one block against autodiff once.
            faces = mesh_geometry(args).nearest_faces(p[mid:mid + 1], args.mesh_normal_mode)
            on_face = faces.reference.clone()
            f_on, *_ = arc_mesh_contact.pi_contact(on_face, faces, torch.zeros(1, dtype=DTYPE), torch.zeros(1, dtype=DTYPE), **mesh_contact_kwargs(args, args.dt))
            assert torch.allclose(f_on, torch.zeros_like(f_on), atol=1e-14), "mesh face must have zero force before crossing"
            crossed = on_face - 1.0e-4 * faces.inward_normal
            _, blocks, _ = arc_mesh_contact.contact_with_blocks(crossed, faces, torch.zeros(1, dtype=DTYPE), torch.zeros(1, dtype=DTYPE), **mesh_contact_kwargs(args, args.dt))
            auto = torch.func.jacrev(lambda point: arc_mesh_contact.pi_contact(point, faces, torch.zeros(1, dtype=DTYPE), torch.zeros(1, dtype=DTYPE), **mesh_contact_kwargs(args, args.dt))[0])(crossed)
            assert torch.allclose(blocks[0], auto[0, :, 0, :], rtol=1e-10, atol=1e-9), "mesh analytic contact Jacobian disagrees with autodiff"
    args.contact_type = original_type
    # 自检临时改为另一固定种子，避免依赖正式算例使用的随机序列。
    original_seed, args.pulse_seed = args.pulse_seed, 7
    a = make_random_pulse(args, raw_p); b = make_random_pulse(args, raw_p)
    args.pulse_seed = original_seed
    assert np.array_equal(a[0], b[0]) and np.allclose(a[3], b[3])
    assert np.any(np.abs(a[2][:, 2]) > 1e-12), "pulse must include out-of-plane directions"
    print("self-test passed: analytic/OBJ arc geometry, 3-D wall directions, open ends, and reproducible 3-D pulses")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-length", type=float, default=TOTAL_LENGTH); parser.add_argument("--diameter", type=float, default=DIAMETER); parser.add_argument("--density", type=float, default=DENSITY)
    parser.add_argument("--num-nodes", type=int, default=NUM_NODES); parser.add_argument("--arc-angle-deg", type=float, default=ARC_ANGLE_DEG); parser.add_argument("--arc-radius", type=float, default=None)
    parser.add_argument("--contact-radius", type=float, default=CONTACT_RADIUS, help="PI 接触面半径；节点达到此半径开始受力")
    parser.add_argument("--tube-radius", type=float, default=TUBE_RADIUS, help="仅绘图显示的实体外壳半径")
    parser.add_argument("--safe-distance", type=float, default=SAFE_DISTANCE, help="外壳与 PI 接触面间距的记录参数")
    parser.add_argument("--contact-type", choices=("mesh", "analytic"), default="mesh", help="默认 mesh 使用 OBJ 三角面；analytic 保留为对照基准")
    parser.add_argument("--mesh-obj", type=Path, default=None, help="OBJ 三角面文件；省略时自动生成本目录 40 mm 圆弧管")
    parser.add_argument("--mesh-normal-mode", choices=("face", "smooth-radial"), default="face", help="face=严格 OBJ 三角面法向（默认）；smooth-radial=仅用于已知圆弧管的平滑对照")
    parser.add_argument("--Kp", type=float, default=KP); parser.add_argument("--Ki", type=float, default=KI); parser.add_argument("--max-pi-force", type=float, default=MAX_PI_FORCE); parser.add_argument("--max-integral-state", type=float, default=MAX_INTEGRAL_STATE); parser.add_argument("--integral-leak-rate", type=float, default=INTEGRAL_LEAK_RATE); parser.add_argument("--contact-smooth-length", type=float, default=CONTACT_SMOOTH_LENGTH); parser.add_argument("--direction-epsilon", type=float, default=DIRECTION_EPSILON)
    parser.add_argument("--dt", type=float, default=DT); parser.add_argument("--max-newton-iters", type=int, default=MAX_NEWTON_ITERS); parser.add_argument("--residual-tol", type=float, default=RESIDUAL_TOL); parser.add_argument("--step-tol", type=float, default=STEP_TOL)
    parser.add_argument("--settle-speed-tol", type=float, default=SETTLE_SPEED_TOL); parser.add_argument("--settle-residual-tol", type=float, default=SETTLE_RESIDUAL_TOL); parser.add_argument("--contact-tolerance", type=float, default=CONTACT_TOLERANCE); parser.add_argument("--settle-confirm-steps", type=int, default=SETTLE_CONFIRM_STEPS); parser.add_argument("--max-settle-steps", type=int, default=MAX_SETTLE_STEPS); parser.add_argument("--prebalance-max-steps", type=int, default=PREBALANCE_MAX_STEPS)
    parser.add_argument("--settle-diagnostic-interval", type=int, default=SETTLE_DIAGNOSTIC_INTERVAL, help="松弛期间速度/残差/接触输出间隔；0=关闭")
    parser.add_argument("--tip-total-displacement", type=float, default=TIP_TOTAL_DISPLACEMENT); parser.add_argument("--num-load-steps", type=int, default=NUM_LOAD_STEPS); parser.add_argument("--tip-ramp-steps", type=int, default=TIP_RAMP_STEPS)
    parser.add_argument("--ramp-mode", choices=("adaptive", "fixed"), default=RAMP_MODE, help="adaptive=面片跨越时速度回滚续算；fixed=二维原有平滑 ramp")
    parser.add_argument("--adaptive-relax-steps", type=int, default=ADAPTIVE_RELAX_STEPS, help="每个自适应试探位移后的最多保持末端微松弛步数")
    parser.add_argument("--adaptive-speed-grow-factor", type=float, default=ADAPTIVE_SPEED_GROW_FACTOR, help="低于该倍数×速度阈值时放大下一自适应增量")
    parser.add_argument("--adaptive-speed-shrink-factor", type=float, default=ADAPTIVE_SPEED_SHRINK_FACTOR, help="高于该倍数×速度阈值时回滚并减半增量")
    parser.add_argument("--pulse-count-min", type=int, default=PULSE_COUNT_MIN); parser.add_argument("--pulse-count-max", type=int, default=PULSE_COUNT_MAX); parser.add_argument("--pulse-force-min", type=float, default=PULSE_FORCE_MIN); parser.add_argument("--pulse-force-max", type=float, default=PULSE_FORCE_MAX); parser.add_argument("--pulse-duration", type=float, default=PULSE_DURATION); parser.add_argument("--pulse-seed", type=int, default=PULSE_SEED, help="随机脉冲种子；默认固定为 20260907，传入其他整数可替换")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda", help="默认 CUDA；可显式指定 cpu 或 auto。OBJ 最近三角形选择仍在 Python 中完成，固定面 PI/Jacobian/DER 在所选设备运行")
    parser.add_argument("--output-prefix", type=str, default=None); parser.add_argument("--history-save-interval", type=int, default=HISTORY_SAVE_INTERVAL, help="预平衡/松弛每隔多少隐式步保存一帧"); parser.add_argument("--animation-stride", type=int, default=ANIMATION_STRIDE, help="GIF 每隔多少个历史状态采样一帧；1=逐帧"); parser.add_argument("--animation-fps", type=float, default=ANIMATION_FPS, help="GIF 和交互动画帧率"); parser.add_argument("--animation-repeat", action="store_true", help="交互动画结束后循环播放")
    parser.add_argument("--skip-animation", action="store_true", help="不生成 GIF，也不播放动画；仍生成最终 PNG"); parser.add_argument("--self-test", action="store_true"); parser.add_argument("--no-show", action="store_true", help="不弹出动画窗口；仍生成 GIF 和最终 PNG")
    args = parser.parse_args(); args.arc_angle_rad = math.radians(args.arc_angle_deg)
    if args.device == "auto": args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: args.device = torch.device(args.device)
    if args.device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA was requested but is unavailable")
    if args.num_nodes < 6 or args.total_length <= 0.0 or not 0.0 < args.arc_angle_rad < 2.0*math.pi: parser.error("invalid length, node count, or arc angle")
    if args.arc_radius is None: args.arc_radius = arc_radius(args.total_length, args.arc_angle_rad, args.num_nodes)
    if not 0.0 < args.contact_radius < args.tube_radius: parser.error("contact-radius must lie strictly between zero and tube-radius")
    if args.tip_total_displacement >= args.arc_radius * args.arc_angle_rad: parser.error("tip displacement must be shorter than arc length")
    if not (1 <= args.pulse_count_min <= args.pulse_count_max <= args.num_nodes - 3): parser.error("invalid pulse count range")
    if args.pulse_force_min < 0.0 or args.pulse_force_max < args.pulse_force_min: parser.error("invalid pulse force range")
    if args.history_save_interval <= 0 or args.animation_stride <= 0 or args.animation_fps <= 0.0 or args.settle_diagnostic_interval < 0 or args.adaptive_relax_steps < 0 or args.adaptive_speed_grow_factor <= 0.0 or args.adaptive_speed_shrink_factor <= 0.0: parser.error("invalid history, animation, or adaptive-continuation parameter")
    if args.mesh_obj is None:
        args.mesh_obj = arc_mesh_contact.default_obj_path()
        if args.contact_type == "mesh":
            args.mesh_obj = arc_mesh_contact.ensure_default_obj(
                args.mesh_obj, arc_radius=args.arc_radius, arc_angle_rad=args.arc_angle_rad,
                contact_radius=args.contact_radius,
            )
    else:
        args.mesh_obj = args.mesh_obj.expanduser().resolve()
        if args.contact_type == "mesh" and not args.mesh_obj.is_file(): parser.error(f"mesh OBJ does not exist: {args.mesh_obj}")
    _, args.EA, args.EI, args.damping = rod_properties(args.total_length, args.diameter, args.density); args.tip_displacement_step = args.tip_total_displacement / args.num_load_steps
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test: self_test(parsed)
    else:
        result = run_simulation(parsed)
        if not parsed.skip_animation:
            animation_path = create_animation(result, parsed)
            print(f"animation={animation_path}")
        image = visualize_results(result, parsed)
        print(f"figure={image}")
        # Keep numerical diagnosis as the terminal's final block, after replay and final-image generation.
        print_final_diagnostics(result, result, parsed, Path(str(result["npz_path"].item())))
