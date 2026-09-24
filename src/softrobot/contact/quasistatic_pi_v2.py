"""
Two-dimensional discrete elastic rod with node-wise PI soft contact.

The model is a 1 m long, 1 mm diameter steel/iron guidewire between two
horizontal walls.  Elastic stretching and curvature-binormal bending energies
are differentiated by PyTorch autograd.  Every time step is solved with an
implicit-Euler residual and damped Newton iterations.

Contact is force-only: there is no projection, position correction, bounce, or
hard contact constraint.  After a short transverse pulse, the right-hand tip
is moved left by a fixed displacement increment at every accepted time step.
Once its total prescribed displacement is reached, the tip remains fixed.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


DTYPE = torch.float64  # 全部计算使用双精度，保证 Newton/接触计算精度

# 几何与材料参数
TOTAL_LENGTH = 1.0  # 导丝初始总长度，m
DIAMETER = 0.001  # 导丝圆截面直径，m
MATERIAL_DENSITY = 7850.0  # 钢丝材料密度，kg/m^3
YOUNG_MODULUS = 200.0e9  # 杨氏模量，Pa
DAMPING_RATIO = 1.0  # 阻尼比；1.0 表示按估计刚度采用临界阻尼

def compute_rod_properties(
    total_length: float,
    diameter: float,
    density: float,
    young_modulus: float = YOUNG_MODULUS,
    damping_ratio: float = DAMPING_RATIO,
) -> tuple[float, float, float, float, float, float, float]:
    """Compute circular-section rod properties and critical damping."""
    area = math.pi * diameter**2 / 4.0
    total_mass = density * area * total_length
    axial_rigidity = young_modulus * area
    second_moment = math.pi * diameter**4 / 64.0
    bending_rigidity = young_modulus * second_moment
    effective_stiffness = 3.0 * bending_rigidity / total_length**3
    damping = 2.0 * damping_ratio * math.sqrt(effective_stiffness * total_mass)
    return (
        area, total_mass, axial_rigidity, second_moment,
        bending_rigidity, effective_stiffness, damping,
    )


(
    CROSS_SECTION_AREA,
    MTOTAL,
    EA,
    SECOND_MOMENT_OF_AREA,
    EI,
    K_EFFECTIVE,
    DAMPING,
) = compute_rod_properties(TOTAL_LENGTH, DIAMETER, MATERIAL_DENSITY)

NUM_NODES = 100  # 离散杆节点总数；增大可提高空间分辨率，但会显著增加 Jacobian 计算量
INITIAL_SINE_IMPERFECTION = 2.9e-3  # 初始一阶正弦几何扰动幅值，m；用于选择屈曲方向

# 血管壁面与逐节点 PI 接触参数
WALL_HALF_GAP = 60e-3             # 血管上下真实壁面位置：y = +/- WALL_HALF_GAP，m
SAFE_DISTANCE = 10.0e-3             # 节点距真实壁面小于该值时进入原始保护层，m
KP = 3000.0  # 比例接触增益，N/m；误差越大，法向排斥力越大
KI = 400  # 积分接触增益，N/(m*s)；设为 0 即纯比例接触
MAX_PI_FORCE = 20.0                 # 单个节点允许的最大接触力，N
MAX_INTEGRAL_STATE = 5.0e-2  # 单节点积分状态上限，m*s；anti-windup 限制
INTEGRAL_LEAK_RATE = 5.0            # 脱离接触区域后积分量的泄漏速率，1/s
CONTACT_SMOOTH_LENGTH = 2e-5  # Softplus 接触过渡长度，m；越大越平滑，但接触会更早起作用
DIRECTION_EPSILON = 1.0e-6          # 接触方向正则化长度，避免 y=0 处 sign 函数不可导，m

# Quasi-static displacement-control settings.  Time is used only for the
# implicit dynamic relaxation inside a pulse or a single load step; it is not
# a global termination condition.
DT = 5.0e-3  # 隐式 Euler 时间步长，s；只用于每个载荷步内的动力学松弛
MAX_NEWTON_ITERS = 100  # 每个隐式时间步允许的 Newton 最大迭代次数
NEWTON_RESIDUAL_TOL = 1.0e-7  # 单个隐式步 Newton 求解残差阈值，N
NEWTON_STEP_TOL = 1.0e-10  # 单个隐式步 Newton 位移修正量阈值，m
SETTLE_SPEED_TOL = 1.0e-3  # 载荷步平衡时自由节点允许的最大速度，m/s
SETTLE_RESIDUAL_TOL = 1.0e-6  # 载荷步平衡时允许的 Newton 残差，N
PULSE_FORCE = 0.0  # 可选横向脉冲力，N；当前为 0，未启用脉冲
PULSE_DURATION = 0.0  # 横向脉冲持续时间，s；当前为 0，未启用脉冲
TIP_TOTAL_DISPLACEMENT = 35e-3  # 右端末节点总向左压缩位移，m
NUM_LOAD_STEPS = 25  # 总压缩位移划分的载荷步数
TIP_DISPLACEMENT_STEP = TIP_TOTAL_DISPLACEMENT / NUM_LOAD_STEPS  # 每一载荷步的压缩位移，m
TIP_RAMP_STEPS = 800  # 每一载荷步平滑施加 tip 位移所用的隐式时间步数
CONTACT_TOLERANCE = 1.0e-3  # 载荷步平衡时允许的最大保护层几何误差，m
SETTLE_CONFIRM_STEPS = 5  # 速度、残差和接触误差连续满足条件的确认步数
MAX_SETTLE_STEPS_PER_LOAD = 3000  # 单一载荷步允许的最大 relaxation 隐式步数
HISTORY_SAVE_INTERVAL = 50  # 每隔多少隐式步保存一次完整历史状态
DIAGNOSTIC_INTERVAL = 50  # 每隔多少 relaxation 步打印一次诊断信息
DIAGNOSTIC_WINDOW_SIZE = 10  # 非收敛诊断使用的最近记录窗口长度
EARLY_ABORT_SETTLE_STEPS = 800  # relaxation 超过此步数后开始检查早停条件
EXTENDED_SETTLE_STEPS = 1600  # 满足单调收敛尾部条件时允许的最长 relaxation 步数
MAX_TUNING_TRIALS = 12  # 历史自动调参逻辑的最大试验次数；当前单次运行入口不会使用
TUNING_LOG_FILENAME = "quasi_static_tuning_log.txt"  # 历史调参日志文件名；当前单次运行入口不会写入


@dataclass
class NewtonInfo:
    converged: bool
    iterations: int
    residual_norm: float
    step_norm: float
    line_search_failures: int


@dataclass(frozen=True)
class TrialParameters:
    kp: float
    ki: float
    damping: float
    dt: float


@dataclass
class TrialResult:
    trial_number: int
    parameters: TrialParameters
    completed_load_steps: int
    blocked_load_step: int | None
    load_step_implicit_steps: list[int]
    max_speed: float
    max_contact_error: float
    max_newton_residual: float
    max_pi_force: float
    max_integral_state: float
    elapsed_seconds: float
    abort_reason: str | None
    next_adjustment_reason: str
    history: dict | None = None


class RelaxationAbort(RuntimeError):
    """Intentional early stop for a non-convergent tuning trial."""

    def __init__(self, reason: str, next_adjustment_reason: str):
        super().__init__(reason)
        self.reason = reason
        self.next_adjustment_reason = next_adjustment_reason


def initialize_rod(
    total_length: float = TOTAL_LENGTH,
    num_nodes: int = NUM_NODES,
    diameter: float = DIAMETER,
    density: float = MATERIAL_DENSITY,
    sine_imperfection: float = INITIAL_SINE_IMPERFECTION,
):
    """Create a lightly imperfect 2-D rod and lump physical steel mass to nodes."""
    if num_nodes < 4:
        raise ValueError("num_nodes must be at least 4")
    positions = np.zeros((num_nodes, 2), dtype=np.float64)
    positions[:, 0] = np.linspace(0.0, total_length, num_nodes)
    # A smooth, sub-protection-layer geometric imperfection selects the
    # physically relevant first buckling mode without changing contact nodes.
    positions[:, 1] = sine_imperfection * np.sin(np.pi * positions[:, 0] / total_length)
    rest_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    _, total_mass, _, _, _, _, _ = compute_rod_properties(
        total_length, diameter, density
    )
    segment_mass = total_mass / (num_nodes - 1)
    node_masses = np.full(num_nodes, segment_mass, dtype=np.float64)
    node_masses[[0, -1]] *= 0.5
    velocities = np.zeros_like(positions)
    integral_states = np.zeros(num_nodes, dtype=np.float64)
    return positions, velocities, rest_lengths, node_masses, integral_states


def _positions_3d(positions_2d: torch.Tensor) -> torch.Tensor:
    """Embed a planar rod in 3-D so the DER curvature-binormal formula applies."""
    zeros = torch.zeros((positions_2d.shape[0], 1), dtype=positions_2d.dtype)
    return torch.cat((positions_2d, zeros), dim=1)


def compute_elastic_energy_torch(
    positions_2d: torch.Tensor,
    rest_lengths: torch.Tensor,
    EA_value: float,
    EI_value: float,
):
    """Return stretching, bending, and total DER energy for a planar rod."""
    positions = _positions_3d(positions_2d)
    edges = positions[1:] - positions[:-1]
    lengths = torch.linalg.norm(edges, dim=1).clamp_min(1.0e-12)
    tangents = edges / lengths[:, None]

    stretch = 0.5 * EA_value * torch.sum((lengths - rest_lengths) ** 2 / rest_lengths)
    cross = torch.linalg.cross(tangents[:-1], tangents[1:], dim=1)
    dot = torch.sum(tangents[:-1] * tangents[1:], dim=1)
    kappa_binormal = 2.0 * cross / (1.0 + dot).clamp_min(1.0e-12)[:, None]
    voronoi_lengths = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
    bend = 0.5 * EI_value * torch.sum(
        torch.sum(kappa_binormal**2, dim=1) / voronoi_lengths
    )
    return stretch, bend, stretch + bend


def _contact_quantities_torch(
    positions: torch.Tensor,
    old_integrals: torch.Tensor,
    dt: float,
    wall_half_gap: float,
    safe_distance: float,
    kp: float,
    ki: float,
    max_force: float,
    leak_rate: float,
    max_integral_state: float = MAX_INTEGRAL_STATE,
    previous_errors: torch.Tensor | None = None,
):
    """平滑、可微的逐节点 PI 血管接触力。"""
    y = positions[:, 1]
    # Signed inward clearance: positive inside the channel and negative after
    # crossing a real wall.  This makes error continue growing on penetration.
    distances = wall_half_gap - torch.abs(y)

    # 相对原始保护层内边界的位置：z < 0 表示保护层外，z > 0 表示保护层内。
    contact_boundary = wall_half_gap - safe_distance
    z = torch.abs(y) - contact_boundary
    beta = 1.0 / CONTACT_SMOOTH_LENGTH

    # 用 Softplus 平滑近似 max(z, 0)，避免在保护层边界产生力学折角。
    errors = F.softplus(z, beta=beta)
    # 连续接触权重仅用于积分状态的平滑切换，不参与最终力的额外缩放。
    contact_weight = torch.sigmoid(beta * z)

    integral_contact = old_integrals + errors * dt
    integral_leak = old_integrals * math.exp(-leak_rate * dt)
    candidate_integrals = (
        contact_weight * integral_contact
        + (1.0 - contact_weight) * integral_leak
    )
    candidate_integrals = torch.clamp(candidate_integrals, 0.0, max_integral_state)

    # Anti-windup: never add integral action while the PI force is already at
    # its cap.  During an error decrease, do not grow an integral term that is
    # already larger than the proportional contribution (overshoot guard).
    unsaturated_magnitudes = kp * errors + ki * candidate_integrals
    saturation_blocks_growth = (
        (ki > 0.0)
        & (unsaturated_magnitudes >= max_force)
        & (candidate_integrals > old_integrals)
    )
    if previous_errors is None:
        error_is_decreasing = torch.zeros_like(errors, dtype=torch.bool)
    else:
        error_is_decreasing = errors < previous_errors
    overshoot_blocks_growth = (
        (ki > 0.0)
        & error_is_decreasing
        & (ki * old_integrals > kp * errors)
        & (candidate_integrals > old_integrals)
    )
    new_integrals = torch.where(
        saturation_blocks_growth | overshoot_blocks_growth,
        old_integrals,
        candidate_integrals,
    )
    magnitudes = torch.clamp(kp * errors + ki * new_integrals, 0.0, max_force)
    directions_y = -y / torch.sqrt(y**2 + DIRECTION_EPSILON**2)
    forces = torch.zeros_like(positions)
    forces[:, 1] = directions_y * magnitudes

    # 仅用于接触事件、动画和日志统计；不得用于力或积分方程的硬切换。
    active = z > 0.0
    return forces, new_integrals, active, contact_weight, errors, distances


def compute_PI_contact_force(
    positions: np.ndarray,
    integral_states: np.ndarray,
    dt: float,
    wall_half_gap: float = WALL_HALF_GAP,
    safe_distance: float = SAFE_DISTANCE,
    kp: float = KP,
    ki: float = KI,
    max_force: float = MAX_PI_FORCE,
    leak_rate: float = INTEGRAL_LEAK_RATE,
    max_integral_state: float = MAX_INTEGRAL_STATE,
    previous_positions: np.ndarray | None = None,
    update_integral: bool = True,
):
    """计算平滑 PI 法向接触力及其诊断状态。"""
    p = torch.as_tensor(positions, dtype=DTYPE)
    integrals = torch.as_tensor(integral_states, dtype=DTYPE)
    with torch.no_grad():
        previous_errors = None
        if previous_positions is not None:
            previous_y = torch.as_tensor(previous_positions, dtype=DTYPE)[:, 1]
            previous_z = torch.abs(previous_y) - (wall_half_gap - safe_distance)
            previous_errors = F.softplus(previous_z, beta=1.0 / CONTACT_SMOOTH_LENGTH)
        force, new_i, active, contact_weight, error, distance = _contact_quantities_torch(
            p, integrals, dt, wall_half_gap, safe_distance,
            kp, ki, max_force, leak_rate, max_integral_state, previous_errors,
        )
    result_integrals = new_i.numpy() if update_integral else integral_states.copy()
    return (
        force.numpy(), result_integrals, active.numpy(), contact_weight.numpy(),
        error.numpy(), distance.numpy(),
    )


def apply_tip_step_displacement_boundary(
    prescribed_values: np.ndarray,
    initial_tip_position: np.ndarray,
    cumulative_displacement: float,
):
    """Set the fully constrained tip position for one cumulative load step."""
    prescribed_values = prescribed_values.copy()
    prescribed_values[-1] = initial_tip_position + np.array([-cumulative_displacement, 0.0])
    return prescribed_values


def detect_contact_event(active_mask: np.ndarray) -> bool:
    """Return True when any node is inside either protection layer."""
    return bool(np.any(active_mask))


def compute_protection_layer_error(
    positions: np.ndarray,
    wall_half_gap: float,
    safe_distance: float,
) -> np.ndarray:
    """Geometric protection-layer penetration used only for convergence checks.

    The PI model itself continues to use its existing smooth Softplus error.
    This diagnostic error is exactly zero outside the protection layer.
    """
    protection_boundary = wall_half_gap - safe_distance
    return np.maximum(np.abs(positions[:, 1]) - protection_boundary, 0.0)


def check_equilibrium_state(
    velocities: np.ndarray,
    prescribed_mask: np.ndarray,
    newton_info: NewtonInfo,
    protection_errors: np.ndarray,
    settle_speed_tol: float,
    settle_residual_tol: float,
    contact_tolerance: float,
    consecutive_satisfied_steps: int,
    settle_confirm_steps: int,
) -> tuple[bool, float, float, int]:
    """Check equilibrium and require consecutive stable relaxation steps."""
    free_nodes = np.any(~prescribed_mask, axis=1)
    max_free_speed = (
        float(np.max(np.linalg.norm(velocities[free_nodes], axis=1)))
        if np.any(free_nodes) else 0.0
    )
    max_protection_error = float(np.max(protection_errors))
    instantaneous_equilibrium = (
        max_free_speed < settle_speed_tol
        and newton_info.residual_norm < settle_residual_tol
        and max_protection_error < contact_tolerance
    )
    confirmed_steps = consecutive_satisfied_steps + 1 if instantaneous_equilibrium else 0
    is_equilibrium = confirmed_steps >= settle_confirm_steps
    return is_equilibrium, max_free_speed, max_protection_error, confirmed_steps


def detect_nonconvergent_relaxation(
    diagnostics: deque[dict],
    args: argparse.Namespace,
    relaxation_step: int,
) -> tuple[str, str] | None:
    """Classify a persistent relaxation failure from the rolling print window."""
    records = list(diagnostics)
    if relaxation_step >= EARLY_ABORT_SETTLE_STEPS:
        # The 800-step limit is an early-stop guard, not an acceptance rule.
        # When the last diagnostics show a genuinely monotone, near-equilibrium
        # tail, allow a bounded extra hold period rather than discarding a
        # converging quasi-static step.
        recent_speeds = np.asarray([record["max_free_speed"] for record in records[-5:]])
        recent_errors = np.asarray([record["contact_error"] for record in records[-5:]])
        recent_residuals = np.asarray([record["newton_residual"] for record in records[-5:]])
        clear_settling_tail = (
            len(records) >= 5
            and np.all(np.diff(recent_speeds) < 0.0)
            and recent_speeds[-1] <= 5.0 * args.settle_speed_tol
            and np.all(recent_errors <= args.contact_tolerance)
            and np.all(recent_residuals <= args.settle_residual_tol)
        )
        if not (clear_settling_tail and relaxation_step < EXTENDED_SETTLE_STEPS):
            return (
                "E: load step exceeded permitted relaxation without confirmed equilibrium",
                "Stop this trial early; use the strongest applicable diagnostic before increasing work.",
            )
    if len(records) < 5 or any(record["stable_count"] != 0 for record in records[-5:]):
        return None

    recent5 = records[-5:]
    errors = np.asarray([record["contact_error"] for record in recent5])
    speeds = np.asarray([record["max_free_speed"] for record in recent5])
    error_plateau = (
        np.all(errors > args.contact_tolerance)
        and (np.max(errors) - np.min(errors)) / max(np.max(errors), 1.0e-15) < 0.05
        and speeds[-1] < 0.5 * max(speeds[0], 1.0e-15)
    )
    if error_plateau:
        return (
            "A: contact error plateaued above tolerance after velocity decayed",
            "Increase Kp by exactly one configured level; proportional contact is too soft.",
        )

    if args.Ki > 0.0:
        integrals = np.asarray([record["max_integral_state"] for record in recent5])
        forces = np.asarray([record["max_pi_force"] for record in recent5])
        integral_growing = np.all(np.diff(integrals) >= -1.0e-12)
        error_not_reducing = errors[-1] >= 0.95 * errors[0]
        speed_spread = np.max(speeds) / max(np.min(speeds), 1.0e-15)
        has_actual_protection_contact = any(
            record["contact_node_count"] > 0 or record["contact_error"] > 0.0
            for record in recent5
        )
        if (
            has_actual_protection_contact
            and integral_growing
            and error_not_reducing
            and speed_spread > 3.0
        ) or np.any(forces >= 0.98 * args.max_pi_force):
            return (
                "C: PI integral saturation or integral-driven overshoot",
                "Reduce Ki to the previous stable level; retain anti-windup and do not raise Kp yet.",
            )

    if len(records) >= 6:
        recent6 = records[-6:]
        speeds6 = np.asarray([record["max_free_speed"] for record in recent6])
        rises = np.count_nonzero(speeds6[1:] > 1.4 * speeds6[:-1])
        drops = np.count_nonzero(speeds6[1:] < 0.85 * speeds6[:-1])
        # A first impact can accelerate monotonically while the soft layer
        # comes into load.  That is not an oscillation.  Require repeated
        # decays followed by a later rebound, as specified for case B.
        drop_indices = np.flatnonzero(speeds6[1:] < 0.85 * speeds6[:-1])
        rise_indices = np.flatnonzero(speeds6[1:] > 1.4 * speeds6[:-1])
        rebound_after_drop = bool(
            len(drop_indices) and len(rise_indices)
            and np.any(rise_indices[:, None] > drop_indices[None, :])
        )
        speed_spread = np.max(speeds6) / max(np.min(speeds6), 1.0e-15)
        no_decay = speeds6[-1] >= 0.8 * speeds6[0]
        if drops >= 2 and rises >= 2 and rebound_after_drop and speed_spread > 3.0 and no_decay:
            return (
                "B: sustained dynamic contact-speed oscillation",
                "For nonzero Ki lower Ki first; otherwise increase damping before reducing dt.",
            )
    return None


def _format_diagnostic(record: dict, load_step: int, total_steps: int) -> str:
    return (
        f"load step {load_step:2d}/{total_steps}, relaxation step {record['relaxation_step']:4d}, "
        f"speed={record['max_free_speed']:.3e}, residual={record['newton_residual']:.3e}, "
        f"contact_error={record['contact_error']:.3e}, max_PI_force={record['max_pi_force']:.3e}, "
        f"stable_count={record['stable_count']}, contact_node={record['max_contact_node']}, "
        f"max_integral={record['max_integral_state']:.3e}, contacts={record['contact_node_count']}, "
        f"newton_ok={record['newton_converged']}, ls_failures={record['line_search_failures']}"
    )


def append_trial_log(log_path: Path, result: TrialResult) -> None:
    """Append a complete, human-readable immutable summary for one tuning run."""
    status = "SUCCESS" if result.abort_reason is None else "ABORTED"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n" + "=" * 76 + "\n")
        log_file.write(f"trial={result.trial_number} status={status}\n")
        log_file.write(
            f"Kp={result.parameters.kp:g}, Ki={result.parameters.ki:g}, "
            f"DAMPING={result.parameters.damping:g}, DT={result.parameters.dt:g}\n"
        )
        log_file.write(
            f"settle_speed_tol={SETTLE_SPEED_TOL:.3e}, settle_residual_tol={SETTLE_RESIDUAL_TOL:.3e}, "
            f"contact_tolerance={CONTACT_TOLERANCE:.3e}\n"
        )
        log_file.write(
            f"completed_load_steps={result.completed_load_steps}, blocked_load_step={result.blocked_load_step}, "
            f"load_step_relaxation_steps={result.load_step_implicit_steps}\n"
        )
        log_file.write(
            f"max_speed={result.max_speed:.6e}, max_contact_error={result.max_contact_error:.6e}, "
            f"max_newton_residual={result.max_newton_residual:.6e}, "
            f"max_PI_force={result.max_pi_force:.6e}, max_integral_state={result.max_integral_state:.6e}\n"
        )
        log_file.write(f"elapsed_seconds={result.elapsed_seconds:.3f}\n")
        log_file.write(f"interruption_reason={result.abort_reason or 'none'}\n")
        log_file.write(f"next_adjustment_reason={result.next_adjustment_reason}\n")


def _pack_dofs(values: torch.Tensor, free_mask: torch.Tensor) -> torch.Tensor:
    return values[free_mask]


def _unpack_dofs(
    q: torch.Tensor,
    template: torch.Tensor,
    free_mask: torch.Tensor,
    prescribed_values: torch.Tensor,
):
    positions = prescribed_values.clone()
    positions[free_mask] = q
    return positions


def _implicit_residual(
    q: torch.Tensor,
    old_positions: torch.Tensor,
    rest_lengths: torch.Tensor,
    pulse_forces: torch.Tensor,
    old_integrals: torch.Tensor,
    free_mask: torch.Tensor,
    prescribed_values: torch.Tensor,
    q_old: torch.Tensor,
    v_old: torch.Tensor,
    free_node_masses: torch.Tensor,
    previous_errors: torch.Tensor,
    dt: float,
    damping: float,
    EA_value: float,
    EI_value: float,
    wall_half_gap: float,
    safe_distance: float,
    kp: float,
    ki: float,
    max_pi_force: float,
    integral_leak_rate: float,
    max_integral_state: float,
    create_graph: bool,
):
    positions_new = _unpack_dofs(q, old_positions, free_mask, prescribed_values)
    _, _, energy = compute_elastic_energy_torch(
        positions_new, rest_lengths, EA_value, EI_value
    )
    elastic_gradient = torch.autograd.grad(
        energy, q, create_graph=create_graph, retain_graph=True
    )[0]

    pi_forces, _, _, _, _, _ = _contact_quantities_torch(
        positions_new, old_integrals, dt, wall_half_gap, safe_distance,
        kp, ki, max_pi_force, integral_leak_rate, max_integral_state, previous_errors,
    )
    external_forces = pulse_forces + pi_forces
    v_new = (q - q_old) / dt
    inertia = free_node_masses * (q - q_old - dt * v_old) / dt**2
    # M*a + elastic_gradient + damping*v - external_force = 0
    return inertia + elastic_gradient + damping * v_new - external_forces[free_mask]


def _residual_and_jacobian(q: torch.Tensor, residual_args):
    q_work = q.detach().clone().requires_grad_(True)
    residual = _implicit_residual(q_work, *residual_args, create_graph=True)
    # A batched vector-Jacobian product produces the same dense Newton matrix
    # as one autograd call per residual component, without the Python loop.
    identity = torch.eye(residual.numel(), dtype=residual.dtype, device=residual.device)
    try:
        (jacobian,) = torch.autograd.grad(
            residual,
            q_work,
            grad_outputs=identity,
            is_grads_batched=True,
            retain_graph=True,
        )
    except RuntimeError:
        # Preserve compatibility with PyTorch builds that lack a batching rule
        # for an operation used by the DER residual.
        rows = []
        for residual_i in residual:
            (row,) = torch.autograd.grad(residual_i, q_work, retain_graph=True)
            rows.append(row)
        jacobian = torch.stack(rows)
    return residual.detach(), jacobian.detach()


def _residual_norm(q: torch.Tensor, residual_args) -> float:
    q_work = q.detach().clone().requires_grad_(True)
    residual = _implicit_residual(q_work, *residual_args, create_graph=False)
    return float(torch.linalg.norm(residual.detach()))


def implicit_euler_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    rest_lengths: np.ndarray,
    node_masses: np.ndarray,
    pulse_forces: np.ndarray,
    integral_states: np.ndarray,
    prescribed_mask: np.ndarray,
    prescribed_values: np.ndarray,
    dt: float,
    damping: float,
    EA_value: float = EA,
    EI_value: float = EI,
    wall_half_gap: float = WALL_HALF_GAP,
    safe_distance: float = SAFE_DISTANCE,
    kp: float = KP,
    ki: float = KI,
    max_pi_force: float = MAX_PI_FORCE,
    integral_leak_rate: float = INTEGRAL_LEAK_RATE,
    max_integral_state: float = MAX_INTEGRAL_STATE,
    max_newton_iters: int = MAX_NEWTON_ITERS,
    residual_tol: float = NEWTON_RESIDUAL_TOL,
    step_tol: float = NEWTON_STEP_TOL,
):
    """Solve one implicit-Euler step with Newton and residual line search."""
    old_p = torch.as_tensor(positions, dtype=DTYPE)
    old_v = torch.as_tensor(velocities, dtype=DTYPE)
    rest = torch.as_tensor(rest_lengths, dtype=DTYPE)
    masses = torch.as_tensor(node_masses, dtype=DTYPE)
    pulse = torch.as_tensor(pulse_forces, dtype=DTYPE)
    integrals = torch.as_tensor(integral_states, dtype=DTYPE)
    prescribed = torch.as_tensor(prescribed_values, dtype=DTYPE)
    prescribed_mask_t = torch.as_tensor(prescribed_mask, dtype=torch.bool)
    free_mask = ~prescribed_mask_t

    # Implicit-Euler predictor is particularly important when the tip has just
    # been released: its current velocity is retained in this initial guess.
    predictor = old_p + dt * old_v
    predictor[prescribed_mask_t] = prescribed[prescribed_mask_t]
    q = _pack_dofs(predictor, free_mask).detach()
    q_old = _pack_dofs(old_p, free_mask)
    v_old = _pack_dofs(old_v, free_mask)
    free_node_masses = masses[:, None].expand_as(old_p)[free_mask]
    previous_z = torch.abs(old_p[:, 1]) - (wall_half_gap - safe_distance)
    previous_errors = F.softplus(previous_z, beta=1.0 / CONTACT_SMOOTH_LENGTH)
    residual_args = (
        old_p, rest, pulse, integrals, free_mask, prescribed,
        q_old, v_old, free_node_masses, previous_errors,
        dt, damping, EA_value, EI_value, wall_half_gap, safe_distance,
        kp, ki, max_pi_force, integral_leak_rate, max_integral_state,
    )

    residual_norm = _residual_norm(q, residual_args)
    converged = residual_norm < residual_tol
    step_norm = 0.0
    iterations = 0
    line_search_failures = 0
    for iteration in range(1, max_newton_iters + 1):
        if converged:
            break
        residual, jacobian = _residual_and_jacobian(q, residual_args)
        try:
            delta = torch.linalg.solve(jacobian, -residual)
        except RuntimeError:
            delta = torch.linalg.lstsq(jacobian, -residual).solution
        step_norm = float(torch.linalg.norm(delta))
        iterations = iteration
        if step_norm < step_tol:
            converged = True
            break

        current_norm = float(torch.linalg.norm(residual))
        accepted = False
        alpha = 1.0
        for _ in range(12):
            trial = (q + alpha * delta).detach()
            trial_norm = _residual_norm(trial, residual_args)
            if np.isfinite(trial_norm) and trial_norm < current_norm:
                q, residual_norm, accepted = trial, trial_norm, True
                break
            alpha *= 0.5
        if not accepted:
            line_search_failures += 1
            q = (q + 0.05 * delta).detach()
            residual_norm = _residual_norm(q, residual_args)
        converged = residual_norm < residual_tol

    new_p_t = _unpack_dofs(q, old_p, free_mask, prescribed).detach()
    new_v_t = (new_p_t - old_p) / dt
    # Fully fixed base nodes have zero velocity.  The prescribed tip velocity
    # is the finite-difference velocity of its imposed step displacement.
    new_v_t[0:2] = 0.0
    new_p = new_p_t.numpy()
    new_v = new_v_t.numpy()
    # Keep the post-solve contact update in Torch, avoiding a NumPy -> Torch
    # -> NumPy round trip after every implicit step.
    with torch.no_grad():
        (
            pi_force_t,
            new_integrals_t,
            active_t,
            contact_weight_t,
            error_t,
            distance_t,
        ) = _contact_quantities_torch(
            new_p_t, integrals, dt, wall_half_gap, safe_distance,
            kp, ki, max_pi_force, integral_leak_rate, max_integral_state,
            previous_errors,
        )
    pi_force = pi_force_t.numpy()
    new_integrals = new_integrals_t.numpy()
    active = active_t.numpy()
    contact_weight = contact_weight_t.numpy()
    error = error_t.numpy()
    distance = distance_t.numpy()
    info = NewtonInfo(
        converged, iterations, residual_norm, step_norm, line_search_failures
    )
    return (
        new_p, new_v, new_integrals, pi_force, active, contact_weight,
        error, distance, info,
    )


def _make_prescribed_state(initial_positions: np.ndarray, tip_constrained: bool):
    mask = np.zeros_like(initial_positions, dtype=bool)
    mask[0:2, :] = True
    if tip_constrained:
        mask[-1, :] = True
    return mask, initial_positions.copy()


def run_single_trial(
    args: argparse.Namespace,
    trial_number: int,
    visualize: bool = False,
) -> TrialResult:
    positions, velocities, rest_lengths, masses, integrals = initialize_rod(
        args.total_length, args.num_nodes, args.diameter, args.density,
        args.initial_sine_imperfection,
    )
    initial_positions = positions.copy()
    pulse_node = args.pulse_node if args.pulse_node is not None else args.num_nodes // 2
    contact_event_time = np.nan
    time_value = 0.0

    history = {
        "time": [0.0],
        "time_step": [0.0],
        "load_step": [0],
        "tip_displacement": [0.0],
        "positions": [positions.copy()],
        "velocities": [velocities.copy()],
        "pi_forces": [np.zeros_like(positions)],
        "pi_integrals": [integrals.copy()],
        "contact_active": [np.zeros(args.num_nodes, dtype=bool)],
        "contact_weight": [np.zeros(args.num_nodes, dtype=np.float64)],
        "max_protection_error": [0.0],
        "max_pi_force": [0.0],
        "newton_iterations": [0],
        "newton_residual": [0.0],
        "newton_converged": [True],
        "tip_position_constraint_active": [False],
    }

    def record_state(
        dt_step: float,
        load_step: int,
        tip_displacement: float,
        pi_forces: np.ndarray,
        active: np.ndarray,
        contact_weights: np.ndarray,
        pi_errors: np.ndarray,
        prescribed_mask: np.ndarray,
        info: NewtonInfo,
        tip_constrained: bool,
    ):
        protection_errors = compute_protection_layer_error(
            positions, args.wall_half_gap, args.safe_distance
        )
        history["time"].append(time_value)
        history["time_step"].append(dt_step)
        history["load_step"].append(load_step)
        history["tip_displacement"].append(tip_displacement)
        history["positions"].append(positions.copy())
        history["velocities"].append(velocities.copy())
        history["pi_forces"].append(pi_forces.copy())
        history["pi_integrals"].append(integrals.copy())
        history["contact_active"].append(active.copy())
        history["contact_weight"].append(contact_weights.copy())
        # Keep the smooth PI error available as well as the exact geometric
        # protection-layer error used by the quasi-static criterion.
        history["max_protection_error"].append(float(np.max(protection_errors)))
        history["max_pi_force"].append(float(np.max(np.linalg.norm(pi_forces, axis=1))))
        history["newton_iterations"].append(info.iterations)
        history["newton_residual"].append(info.residual_norm)
        history["newton_converged"].append(info.converged)
        history["tip_position_constraint_active"].append(tip_constrained)
        return protection_errors

    def advance_one_implicit_step(
        pulse_forces: np.ndarray,
        tip_constrained: bool,
        cumulative_tip_displacement: float,
        dt_step: float,
    ):
        prescribed_mask, prescribed_values = _make_prescribed_state(
            initial_positions, tip_constrained
        )
        prescribed_values[0:2] = initial_positions[0:2]
        if tip_constrained:
            prescribed_values = apply_tip_step_displacement_boundary(
                prescribed_values,
                initial_positions[-1],
                cumulative_tip_displacement,
            )
        return (
            *implicit_euler_step(
                positions, velocities, rest_lengths, masses, pulse_forces, integrals,
                prescribed_mask, prescribed_values, dt_step, args.damping,
                args.EA, args.EI, args.wall_half_gap, args.safe_distance,
                args.Kp, args.Ki, args.max_pi_force, args.integral_leak_rate,
                args.max_integral_state,
                args.max_newton_iters, args.residual_tol, args.step_tol,
            ),
            prescribed_mask,
        )

    print("=" * 76)
    print(f"TUNING TRIAL {trial_number}")
    print("2-D DER + implicit Euler/Newton + node-wise PI soft contact (quasi-static displacement control)")
    print("=" * 76)
    print(
        f"nodes={args.num_nodes}, length={args.total_length:g} m, diameter={args.diameter:g} m, "
        f"mass={np.sum(masses):.6g} kg"
    )
    print(
        f"mtotal={args.mtotal:.6e} kg, EA={args.EA:g} N, EI={args.EI:g} N*m^2, "
        f"k_eff={args.k_eff:.6e} N/m, damping={args.damping:.6e} N*s/m, "
        f"relaxation_dt={args.dt:g} s"
    )
    print(
        f"walls=+/-{args.wall_half_gap:g} m, safe_distance={args.safe_distance:g} m, "
        f"Kp={args.Kp:g}, Ki={args.Ki:g}, cap={args.max_pi_force:g} N"
    )
    print(
        f"tip_total_displacement={args.tip_total_displacement:g} m, load_steps={args.num_load_steps}, "
        f"increment={args.tip_displacement_step:g} m, ramp_steps={args.tip_ramp_steps}, "
        f"contact_tolerance={args.contact_tolerance:g} m"
    )

    failed_steps = 0
    total_newton_iterations = 0
    load_step_implicit_steps: list[int] = []
    wall_start = time.perf_counter()
    max_speed_seen = 0.0
    max_contact_error_seen = 0.0
    max_newton_residual_seen = 0.0
    max_pi_force_seen = 0.0
    max_integral_state_seen = 0.0
    abort_reason: str | None = None
    next_adjustment_reason = "All load steps completed under the required steady-state criteria."
    blocked_load_step: int | None = None
    # Pulse phase: the tip is free.  This is only a symmetry-breaking transient
    # and is completed before any displacement-controlled loading begins.
    while time_value < args.pulse_duration - 1.0e-15:
        dt_step = min(args.dt, args.pulse_duration - time_value)
        pulse_forces = np.zeros_like(positions)
        pulse_forces[pulse_node, 1] = args.pulse_force
        (
            positions, velocities, integrals, pi_forces, active, contact_weights,
            pi_errors, _distances, info, prescribed_mask,
        ) = advance_one_implicit_step(pulse_forces, False, 0.0, dt_step)
        time_value += dt_step
        if not info.converged:
            failed_steps += 1
        total_newton_iterations += info.iterations
        record_state(
            dt_step, 0, 0.0, pi_forces, active, contact_weights, pi_errors,
            prescribed_mask, info, False,
        )

    # Each 0.2 mm load increment is first applied by a smooth C1 displacement
    # ramp, then the right tip is held at the exact increment target while the
    # free nodes relax.  This retains displacement control without injecting an
    # artificial one-step impulse into the implicit dynamics.
    for load_step in range(1, args.num_load_steps + 1):
        cumulative_tip_displacement = load_step * args.tip_displacement_step
        previous_tip_displacement = (load_step - 1) * args.tip_displacement_step
        pulse_forces = np.zeros_like(positions)
        settled = False
        consecutive_stable_steps = 0
        diagnostics: deque[dict] = deque(maxlen=DIAGNOSTIC_WINDOW_SIZE)
        newton_failure_streak = 0
        for ramp_step in range(1, args.tip_ramp_steps + 1):
            ramp_fraction = ramp_step / args.tip_ramp_steps
            smooth_fraction = ramp_fraction * ramp_fraction * (3.0 - 2.0 * ramp_fraction)
            ramp_tip_displacement = previous_tip_displacement + (
                cumulative_tip_displacement - previous_tip_displacement
            ) * smooth_fraction
            (
                positions, velocities, integrals, pi_forces, active, contact_weights,
                pi_errors, _distances, info, prescribed_mask,
            ) = advance_one_implicit_step(
                pulse_forces, True, ramp_tip_displacement, args.dt
            )
            time_value += args.dt
            failed_steps += int(not info.converged)
            total_newton_iterations += info.iterations
            protection_errors = compute_protection_layer_error(
                positions, args.wall_half_gap, args.safe_distance
            )
            if detect_contact_event(active) and not np.isfinite(contact_event_time):
                contact_event_time = time_value
                print(f"first protection-layer contact at t={time_value:.6f} s")
            free_nodes = ~np.all(prescribed_mask, axis=1)
            free_speeds = np.linalg.norm(velocities[free_nodes], axis=1)
            max_speed_seen = max(max_speed_seen, float(np.max(free_speeds)))
            max_contact_error_seen = max(max_contact_error_seen, float(np.max(protection_errors)))
            max_newton_residual_seen = max(max_newton_residual_seen, info.residual_norm)
            max_pi_force_seen = max(max_pi_force_seen, float(np.max(np.linalg.norm(pi_forces, axis=1))))
            max_integral_state_seen = max(max_integral_state_seen, float(np.max(integrals)))
            if ramp_step % args.history_save_interval == 0 or ramp_step == args.tip_ramp_steps:
                record_state(
                    args.dt, load_step, ramp_tip_displacement,
                    pi_forces, active, contact_weights, pi_errors, prescribed_mask,
                    info, True,
                )
            if not info.converged and info.residual_norm >= args.residual_tol:
                abort_reason = "D: Newton failure during smooth prescribed-tip ramp"
                next_adjustment_reason = "Reduce DT before changing contact gains."
                blocked_load_step = load_step
                print(f"EARLY ABORT: {abort_reason}")
                break
        if abort_reason is not None:
            break
        print(
            f"load step {load_step:2d}/{args.num_load_steps}: smoothly advanced "
            f"{args.tip_displacement_step * 1e3:.3f} mm over {args.tip_ramp_steps} implicit steps; "
            "holding final tip position for relaxation"
        )
        for relaxation_step in range(1, args.max_settle_steps + 1):
            (
                positions, velocities, integrals, pi_forces, active, contact_weights,
                pi_errors, _distances, info, prescribed_mask,
            ) = advance_one_implicit_step(
                pulse_forces, True, cumulative_tip_displacement, args.dt
            )
            time_value += args.dt
            failed_steps += int(not info.converged)
            total_newton_iterations += info.iterations
            protection_errors = compute_protection_layer_error(
                positions, args.wall_half_gap, args.safe_distance
            )
            if detect_contact_event(active) and not np.isfinite(contact_event_time):
                contact_event_time = time_value
                print(f"first protection-layer contact at t={time_value:.6f} s")

            settled, max_speed, max_contact_error, consecutive_stable_steps = check_equilibrium_state(
                velocities, prescribed_mask, info, protection_errors,
                args.settle_speed_tol, args.settle_residual_tol,
                args.contact_tolerance, consecutive_stable_steps,
                args.settle_confirm_steps,
            )
            max_force = float(np.max(np.linalg.norm(pi_forces, axis=1)))
            max_integral = float(np.max(integrals))
            max_speed_seen = max(max_speed_seen, max_speed)
            max_contact_error_seen = max(max_contact_error_seen, max_contact_error)
            max_newton_residual_seen = max(max_newton_residual_seen, info.residual_norm)
            max_pi_force_seen = max(max_pi_force_seen, max_force)
            max_integral_state_seen = max(max_integral_state_seen, max_integral)
            # A large transient residual alone is not a Newton failure: D is
            # reserved for repeated non-convergence with an above-tolerance
            # residual, as required by the diagnostic specification.
            newton_failure_streak = (
                newton_failure_streak + 1
                if (not info.converged and info.residual_norm >= args.residual_tol)
                else 0
            )

            save_interval_state = relaxation_step % args.history_save_interval == 0
            diagnostic_due = relaxation_step % DIAGNOSTIC_INTERVAL == 0 or settled
            if save_interval_state or settled:
                record_state(
                    args.dt, load_step, cumulative_tip_displacement,
                    pi_forces, active, contact_weights, pi_errors, prescribed_mask,
                    info, True,
                )
            if diagnostic_due:
                record = {
                    "relaxation_step": relaxation_step,
                    "max_free_speed": max_speed,
                    "newton_residual": info.residual_norm,
                    "contact_error": max_contact_error,
                    "max_pi_force": max_force,
                    "stable_count": consecutive_stable_steps,
                    "max_contact_node": int(np.argmax(protection_errors)),
                    "max_integral_state": max_integral,
                    "contact_node_count": int(np.count_nonzero(active)),
                    "newton_converged": info.converged,
                    "line_search_failures": info.line_search_failures,
                }
                diagnostics.append(record)
                print(_format_diagnostic(record, load_step, args.num_load_steps))
                decision = detect_nonconvergent_relaxation(diagnostics, args, relaxation_step)
                if decision is not None:
                    abort_reason, next_adjustment_reason = decision
                    blocked_load_step = load_step
                    print(f"EARLY ABORT: {abort_reason}")
                    break
            if newton_failure_streak >= 3:
                abort_reason = "D: three consecutive implicit steps failed the Newton residual criterion"
                next_adjustment_reason = "Reduce DT before changing Newton tolerances or contact parameters."
                blocked_load_step = load_step
                print(f"EARLY ABORT: {abort_reason}")
                break
            if settled:
                load_step_implicit_steps.append(relaxation_step)
                print(
                    f"load step {load_step:2d}/{args.num_load_steps}: tip displacement="
                    f"{cumulative_tip_displacement * 1e3:.3f} mm, relaxed in "
                    f"{relaxation_step} steps (confirmed for {consecutive_stable_steps} steps), "
                    f"max_speed={max_speed:.3e}, max_contact_error={max_contact_error * 1e3:.4f} mm"
                )
                break

        if abort_reason is not None:
            break
        if not settled:
            blocked_load_step = load_step
            abort_reason = (
                f"Load step {load_step} did not reach equilibrium within "
                f"{args.max_settle_steps} implicit-Euler steps. Last criteria: "
                f"max_speed={max_speed:.3e}, residual={info.residual_norm:.3e}, "
                f"max_contact_error={max_contact_error:.3e}."
            )
            next_adjustment_reason = "Inspect the rolling diagnostics before attempting another parameter set."
            print(f"EARLY ABORT: {abort_reason}")
            break

    elapsed = time.perf_counter() - wall_start
    completed_load_steps = len(load_step_implicit_steps)
    successful = abort_reason is None and completed_load_steps == args.num_load_steps
    print(
        f"trial {trial_number}: completed_load_steps={completed_load_steps}/{args.num_load_steps}, "
        f"wall_time={elapsed:.2f} s, unconverged_implicit_steps={failed_steps}"
    )
    if successful:
        for key in history:
            history[key] = np.asarray(history[key])
        history["initial_positions"] = initial_positions
        history["contact_event_time"] = np.asarray(contact_event_time)
        history["load_step_implicit_steps"] = np.asarray(load_step_implicit_steps, dtype=int)
        history["total_newton_iterations"] = np.asarray(total_newton_iterations)
        output_prefix = Path(args.output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        npz_path = output_prefix.with_suffix(".npz")
        np.savez_compressed(npz_path, **history)
        print(f"history saved: {npz_path}")
        if visualize:
            visualize_results(history, args)
    return TrialResult(
        trial_number=trial_number,
        parameters=TrialParameters(args.Kp, args.Ki, args.damping, args.dt),
        completed_load_steps=completed_load_steps,
        blocked_load_step=blocked_load_step,
        load_step_implicit_steps=load_step_implicit_steps,
        max_speed=max_speed_seen,
        max_contact_error=max_contact_error_seen,
        max_newton_residual=max_newton_residual_seen,
        max_pi_force=max_pi_force_seen,
        max_integral_state=max_integral_state_seen,
        elapsed_seconds=elapsed,
        abort_reason=None if successful else abort_reason,
        next_adjustment_reason=next_adjustment_reason,
        history=history if successful else None,
    )


def choose_next_trial_parameters(
    result: TrialResult,
    attempted: set[TrialParameters],
    args: argparse.Namespace,
) -> tuple[TrialParameters | None, str]:
    """Apply the prescribed Kp/Ki/damping/dt priority order without duplicates."""
    current = result.parameters
    reason = result.abort_reason or ""
    kp_levels = (4000.0, 6000.0, 8000.0, 10000.0, 15000.0, 20000.0)
    ki_levels = (0.0, 10.0, 25.0, 50.0, 100.0, 200.0)

    def unused(candidate: TrialParameters) -> TrialParameters | None:
        return candidate if candidate not in attempted else None

    if reason.startswith("D:"):
        if current.dt > 2.5e-3:
            return unused(TrialParameters(current.kp, current.ki, current.damping, 2.5e-3)), (
                "Newton instability: reduce DT to 2.5e-3 before relaxing solver tolerances."
            )
        return None, "Newton instability persisted at the minimum permitted DT."

    if reason.startswith("C:") or (reason.startswith("B:") and current.ki > 0.0):
        smaller = [value for value in ki_levels if value < current.ki]
        if smaller:
            return unused(TrialParameters(current.kp, smaller[-1], current.damping, current.dt)), (
                "Integral saturation/oscillation: return to the previous lower Ki level."
            )
        return None, "Oscillation remains with Ki=0; no integral gain remains to remove."

    if reason.startswith("B:"):
        if current.damping < 1.5:
            return unused(TrialParameters(current.kp, current.ki, 1.5, current.dt)), (
                "Persistent oscillation: increase DAMPING from 1.0 to 1.5."
            )
        if current.damping < 2.0:
            return unused(TrialParameters(current.kp, current.ki, 2.0, current.dt)), (
                "Persistent oscillation: increase DAMPING from 1.5 to 2.0."
            )
        if current.dt > 2.5e-3:
            return unused(TrialParameters(current.kp, current.ki, current.damping, 2.5e-3)), (
                "Oscillation persisted with damping 2.0: reduce DT to 2.5e-3."
            )
        return None, "Oscillation persisted after all permitted damping and DT adjustments."

    needs_more_contact_stiffness = (
        reason.startswith("A:")
        or (reason.startswith("E:") and result.max_contact_error > args.contact_tolerance)
    )
    if needs_more_contact_stiffness and current.kp < kp_levels[-1]:
        next_kp = next(value for value in kp_levels if value > current.kp)
        return unused(TrialParameters(next_kp, 0.0, current.damping, current.dt)), (
            f"Contact error remains above {args.contact_tolerance:.1e} m after velocity decay: raise Kp one level."
        )

    if needs_more_contact_stiffness and current.kp >= kp_levels[-1]:
        next_ki = next((value for value in ki_levels[1:] if value > current.ki), None)
        if next_ki is not None:
            return unused(TrialParameters(current.kp, next_ki, current.damping, current.dt)), (
                "Kp reached 20000 with a stable error plateau: test the next small Ki level with anti-windup."
            )

    if reason.startswith("E:") and result.max_speed > args.settle_speed_tol:
        if current.damping < 1.5:
            return unused(TrialParameters(current.kp, current.ki, 1.5, current.dt)), (
                "800-step limit reached with excessive speed: increase damping before changing contact gains."
            )
        if current.damping < 2.0:
            return unused(TrialParameters(current.kp, current.ki, 2.0, current.dt)), (
                "800-step limit persisted at damping 1.5: increase damping to 2.0."
            )
        if current.dt > 2.5e-3:
            return unused(TrialParameters(current.kp, current.ki, current.damping, 2.5e-3)), (
                "800-step limit persisted at normal damping: reduce DT to 2.5e-3."
            )
    return None, "No permitted next parameter change is justified by the recorded diagnostics."


def run_simulation(args: argparse.Namespace):
    """Run one simulation with the requested fixed physical and contact parameters."""
    result = run_single_trial(args, trial_number=1, visualize=False)
    if result.abort_reason is not None:
        raise RuntimeError(
            f"Simulation did not complete: {result.abort_reason}"
        )
    # Keep the existing visualization after a successful fixed-parameter run.
    visualize_results(result.history, args)
    return result.history


def visualize_results(history, args: argparse.Namespace):
    """Show the live animation, then summary and final-frame figures; save none."""
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if not args.show:
        return
    positions = history["positions"]
    time_values = history["time"]
    max_bend_index = int(np.argmax(np.ptp(positions[:, :, 1], axis=1)))
    protection_y = args.wall_half_gap - args.safe_distance

    if not args.skip_animation:
        fig_anim, ax = plt.subplots(figsize=(12, 3.8))
        rod_line, = ax.plot([], [], "b-o", ms=3, lw=1.5)
        fixed_scatter = ax.scatter([], [], marker="s", c="red", s=45, zorder=4)
        contact_scatter = ax.scatter([], [], c="magenta", s=32, zorder=5)
        title = ax.set_title("")
        for wall in (-args.wall_half_gap, args.wall_half_gap):
            ax.axhline(wall, color="black", lw=1.3)
        for layer in (-protection_y, protection_y):
            ax.axhline(layer, color="tab:red", ls=":", lw=1.1)
        ax.set_xlim(-0.02, args.total_length + 0.02)
        ax.set_ylim(-1.35 * args.wall_half_gap, 1.35 * args.wall_half_gap)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(alpha=0.2)

        frame_indices = np.arange(0, len(time_values), max(1, args.animation_stride))
        if frame_indices[-1] != len(time_values) - 1:
            frame_indices = np.append(frame_indices, len(time_values) - 1)

        def update(frame_number):
            index = int(frame_indices[frame_number])
            p = positions[index]
            rod_line.set_data(p[:, 0], p[:, 1])
            fixed_scatter.set_offsets(p[:2])
            contact_scatter.set_offsets(p[history["contact_active"][index]])
            title.set_text(f"Node-wise PI soft contact, t={time_values[index]:.3f} s")
            return rod_line, fixed_scatter, contact_scatter, title

        animation = FuncAnimation(
            fig_anim,
            update,
            frames=len(frame_indices),
            interval=1000.0 / args.animation_fps,
            blit=False,
        )
        plt.show()
        plt.close(fig_anim)

    fig, (ax_shape, ax_hist) = plt.subplots(2, 1, figsize=(12, 8))
    ax_shape.plot(positions[0, :, 0], positions[0, :, 1], "k--", label="initial")
    ax_shape.plot(
        positions[max_bend_index, :, 0], positions[max_bend_index, :, 1],
        color="tab:orange", label=f"maximum bend (t={time_values[max_bend_index]:.3f}s)",
    )
    ax_shape.plot(positions[-1, :, 0], positions[-1, :, 1], "b-o", ms=2.5, label="final")
    ax_shape.scatter(positions[-1, :2, 0], positions[-1, :2, 1], marker="s", c="red", label="fixed")
    for wall in (-args.wall_half_gap, args.wall_half_gap):
        ax_shape.axhline(wall, color="black", lw=1.4)
    for layer in (-protection_y, protection_y):
        ax_shape.axhline(layer, color="tab:red", ls=":", lw=1.1)
    ax_shape.set(xlabel="x (m)", ylabel="y (m)", title="Initial, maximum-bending, and final states")
    ax_shape.set_ylim(-1.35 * args.wall_half_gap, 1.35 * args.wall_half_gap)
    ax_shape.grid(alpha=0.25)
    ax_shape.legend(loc="upper right", ncol=3)

    ax_hist.plot(time_values, history["max_protection_error"] * 1e3, label="max error (mm)")
    ax_hist.plot(time_values, history["max_pi_force"], label="max PI force (N)")
    ax_hist.plot(time_values, history["newton_iterations"], label="Newton iterations")
    ax_hist.set(xlabel="time (s)", title="Contact and solver histories")
    ax_hist.grid(alpha=0.25)
    ax_hist.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    final_displacement = np.linalg.norm(positions[-1] - positions[0], axis=1)
    fig_final, ax_final = plt.subplots(figsize=(12, 3.8))
    ax_final.plot(positions[-1, :, 0], positions[-1, :, 1], "b-o", ms=3, lw=1.5)
    ax_final.scatter(positions[-1, :2, 0], positions[-1, :2, 1], marker="s", c="red", s=45, label="fixed")
    for wall in (-args.wall_half_gap, args.wall_half_gap):
        ax_final.axhline(wall, color="black", lw=1.3)
    for layer in (-protection_y, protection_y):
        ax_final.axhline(layer, color="tab:red", ls=":", lw=1.1)
    ax_final.set(
        xlim=(-0.02, args.total_length + 0.02),
        ylim=(-1.35 * args.wall_half_gap, 1.35 * args.wall_half_gap),
        xlabel="x (m)",
        ylabel="y (m)",
        title=(
            f"Final frame: t={time_values[-1]:.3f} s, "
            f"max displacement={np.max(final_displacement) * 1e3:.3f} mm"
        ),
    )
    ax_final.grid(alpha=0.2)
    ax_final.legend()
    fig_final.tight_layout()
    plt.show()
    plt.close(fig_final)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-length", type=float, default=TOTAL_LENGTH)
    parser.add_argument("--diameter", type=float, default=DIAMETER)
    parser.add_argument("--density", type=float, default=MATERIAL_DENSITY)
    parser.add_argument("--num-nodes", type=int, default=NUM_NODES)
    parser.add_argument(
        "--initial-sine-imperfection", type=float, default=INITIAL_SINE_IMPERFECTION,
        help="first-mode initial rod imperfection used to select a smooth buckling branch (m)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DT,
        help="implicit-Euler relaxation time step within pulse and each load step",
    )
    parser.add_argument("--pulse-force", type=float, default=PULSE_FORCE)
    parser.add_argument("--pulse-duration", type=float, default=PULSE_DURATION)
    parser.add_argument("--pulse-node", type=int)
    parser.add_argument("--wall-half-gap", type=float, default=WALL_HALF_GAP)
    parser.add_argument("--safe-distance", type=float, default=SAFE_DISTANCE)
    parser.add_argument("--Kp", type=float, default=KP)
    parser.add_argument("--Ki", type=float, default=KI)
    parser.add_argument("--max-pi-force", type=float, default=MAX_PI_FORCE)
    parser.add_argument(
        "--max-integral-state", type=float, default=MAX_INTEGRAL_STATE,
        help="per-node PI integral-state anti-windup limit (m*s)",
    )
    parser.add_argument("--integral-leak-rate", type=float, default=INTEGRAL_LEAK_RATE)
    parser.add_argument(
        "--max-newton-iters", type=int, default=MAX_NEWTON_ITERS,
        help="单个隐式时间步允许的 Newton 最大迭代次数",
    )
    parser.add_argument(
        "--residual-tol", type=float, default=NEWTON_RESIDUAL_TOL,
        help="单个隐式时间步的 Newton 残差收敛阈值（N）",
    )
    parser.add_argument(
        "--step-tol", type=float, default=NEWTON_STEP_TOL,
        help="单个隐式时间步的 Newton 位移修正量阈值（m）",
    )
    parser.add_argument(
        "--settle-speed-tol",
        type=float,
        default=SETTLE_SPEED_TOL,
        help="maximum free-node speed for a load step to be in equilibrium (m/s)",
    )
    parser.add_argument(
        "--settle-residual-tol",
        type=float,
        default=SETTLE_RESIDUAL_TOL,
        help="Newton residual threshold for a load step to be in equilibrium (N)",
    )
    parser.add_argument(
        "--contact-tolerance",
        type=float,
        default=CONTACT_TOLERANCE,
        help="allowed geometric protection-layer error at equilibrium (m)",
    )
    parser.add_argument(
        "--tip-total-displacement",
        type=float,
        default=TIP_TOTAL_DISPLACEMENT,
        help="total imposed leftward tip displacement (m)",
    )
    parser.add_argument(
        "--num-load-steps",
        type=int,
        default=NUM_LOAD_STEPS,
        help="number of equal displacement-controlled loading increments",
    )
    parser.add_argument(
        "--tip-ramp-steps",
        type=int,
        default=TIP_RAMP_STEPS,
        help="implicit steps used to smoothly apply one fixed 0.2 mm load increment before holding",
    )
    parser.add_argument(
        "--max-settle-steps",
        type=int,
        default=MAX_SETTLE_STEPS_PER_LOAD,
        help="safety limit on implicit relaxation steps for one load increment",
    )
    parser.add_argument(
        "--settle-confirm-steps",
        type=int,
        default=SETTLE_CONFIRM_STEPS,
        help="consecutive steps that must satisfy all equilibrium criteria",
    )
    parser.add_argument(
        "--history-save-interval",
        type=int,
        default=HISTORY_SAVE_INTERVAL,
        help="save one transient relaxation state every N implicit steps",
    )
    parser.add_argument("--output-prefix", default="rod_soft_contact_PI_quasi_static")
    parser.add_argument(
        "--animation-stride",
        type=int,
        default=1,
        help="show one frame every N time steps; default 1 keeps every step",
    )
    parser.add_argument(
        "--animation-fps",
        type=float,
        default=30.0,
        help="live animation speed in frames per second",
    )
    parser.add_argument("--skip-animation", action="store_true")
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        default=True,
        help="run without opening the animation or figures",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args(argv)
    if args.num_nodes < 4:
        parser.error("num_nodes must be at least 4")
    protection_boundary = args.wall_half_gap - args.safe_distance
    if args.initial_sine_imperfection < 0 or args.initial_sine_imperfection >= protection_boundary:
        parser.error("initial-sine-imperfection must be non-negative and below the protection-layer boundary")
    if args.total_length <= 0 or args.diameter <= 0 or args.density <= 0:
        parser.error("total-length, diameter, and density must be positive")
    if args.dt <= 0:
        parser.error("dt must be positive")
    if args.max_pi_force <= 0 or args.max_integral_state <= 0:
        parser.error("max-pi-force and max-integral-state must be positive")
    if args.settle_speed_tol <= 0 or args.settle_residual_tol <= 0 or args.contact_tolerance < 0:
        parser.error("equilibrium tolerances must be non-negative, with speed and residual tolerances positive")
    if (
        args.tip_total_displacement <= 0
        or args.num_load_steps <= 0
        or args.tip_ramp_steps <= 0
        or args.max_settle_steps <= 0
        or args.settle_confirm_steps <= 0
        or args.history_save_interval <= 0
    ):
        parser.error(
            "tip_total_displacement, num_load_steps, tip_ramp_steps, max_settle_steps, "
            "settle_confirm_steps, and history_save_interval must be positive"
        )
    if not 0 < args.safe_distance < args.wall_half_gap:
        parser.error("safe_distance must be between zero and wall_half_gap")
    if args.animation_stride <= 0 or args.animation_fps <= 0:
        parser.error("animation_stride and animation_fps must be positive")
    if args.pulse_node is not None and not 2 <= args.pulse_node < args.num_nodes:
        parser.error("pulse_node must be a free node from 2 through num_nodes-1")
    (
        args.area,
        args.mtotal,
        args.EA,
        args.second_moment_of_area,
        args.EI,
        args.k_eff,
        args.damping,
    ) = compute_rod_properties(args.total_length, args.diameter, args.density)
    args.tip_displacement_step = args.tip_total_displacement / args.num_load_steps
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke_test:
        import sys
        from ._smoke import run_contact_smoke

        run_contact_smoke(
            sys.modules[__name__], 'contact-2d-quasistatic-pi-v2', args.output_dir,
            quasistatic=True,
        )
        return 0
    run_simulation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
