"""
二维离散弹性杆大变形解析解

基于 Euler-Bernoulli 梁的大变形理论（Euler Elastica）
支持任意节点施加任意方向的集中力
支持两种边界条件：理想固支 或 固定两个节点

适用场景：
    - 验证 PyTorch 动态仿真代码的静态平衡结果
    - 大变形情况（挠度 > 梁厚的5倍）
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.optimize import root_scalar
from scipy.special import ellipk, ellipe


# ============================================================================
# 参数设置区（用户可修改）
# ============================================================================

# --- 梁的几何参数 ---
L = 2.0                      # 梁的原始长度 (m)
num_segments = 20            # 分段数（节点数 = num_segments + 1）

# --- 材料参数 ---
EA = 1000.0                  # 拉伸刚度 (N)
EI = 5.0                     # 弯曲刚度 (N·m²)

# --- 载荷参数 ---
force_node = 20              # 施力节点索引 (0 到 num_segments)
force_magnitude = 10.0       # 力大小 (N)
force_angle_deg = 45       # 力方向角度 (度)，0°=沿X正向，90°=沿Y正向

# --- 边界条件设置 ---
fix_two_nodes = False        # True=固定两个节点（匹配PyTorch），False=理想固支

# --- 求解参数 ---
n_elements = 20             # 离散点数（用于输出节点位置）
tol = 1e-10                  # 收敛容差
plot_shape = True            # 是否绘制变形图
save_plot = False            # 是否保存图片
plot_path = "large_deflection_shape.png"


# ============================================================================
# 大变形解析解类
# ============================================================================

@dataclass
class ForceComponents:
    """力的分量"""
    axial: float
    transverse: float


class LargeDeflectionAnalytic:
    """二维离散弹性杆大变形解析解"""

    def __init__(self, L: float, EA: float, EI: float,
                 num_segments: int, force_node: int,
                 force_magnitude: float, force_angle_deg: float,
                 fix_two_nodes: bool = False):
        """
        初始化大变形解析解求解器

        参数:
            L: 梁长度 (m)
            EA: 拉伸刚度 (N)
            EI: 弯曲刚度 (N·m²)
            num_segments: 分段数
            force_node: 施力节点索引 (0 到 num_segments)
            force_magnitude: 力大小 (N)
            force_angle_deg: 力方向角度 (度)
            fix_two_nodes: 是否固定两个节点
        """
        self.L = L
        self.EA = EA
        self.EI = EI
        self.num_segments = num_segments
        self.num_nodes = num_segments + 1
        self.force_node = force_node
        self.force_magnitude = force_magnitude
        self.force_angle_deg = force_angle_deg
        self.fix_two_nodes = fix_two_nodes

        # 计算施力点距离
        segment_length = L / num_segments
        self.segment_length = segment_length

        if fix_two_nodes and force_node == num_segments:
            # 固定两个节点且力在末端：有效长度从节点1开始
            self.force_distance = (force_node - 1) * segment_length
            self.fixed_offset = segment_length
        else:
            self.force_distance = force_node * segment_length
            self.fixed_offset = 0.0

        # 结果存储
        self.force_components = None
        self.delta_stretch = None
        self.end_x = None
        self.end_y = None
        self.final_positions = None
        self.initial_positions = None
        self.n_elements = n_elements

    def _decompose_force(self) -> ForceComponents:
        """将力分解为轴向和横向分量"""
        force_angle_rad = np.radians(self.force_angle_deg)

        axial_force = self.force_magnitude * np.cos(force_angle_rad)
        transverse_force = self.force_magnitude * np.sin(force_angle_rad)

        self.force_components = ForceComponents(
            axial=axial_force,
            transverse=transverse_force
        )

        return self.force_components

    def _solve_stretching(self) -> float:
        """轴向拉伸的大变形解（对数应变）"""
        F_axial = self.force_components.axial
        if abs(F_axial) < 1e-12 or self.EA == 0:
            return 0.0

        if F_axial > 0:
            true_strain = F_axial / self.EA
            stretch_ratio = np.exp(true_strain)
            delta = self.force_distance * (stretch_ratio - 1)
        else:
            true_strain = -F_axial / self.EA
            stretch_ratio = np.exp(-true_strain)
            delta = self.force_distance * (1 - 1/stretch_ratio)

        return delta

    def _solve_euler_elastica(self, F: float, L_eff: float):
        """
        Euler Elastica 纯弯曲解析解
        返回: x_end, y_end, s_vals, x_vals, y_vals, theta_vals
        """
        if abs(F) < 1e-12 or self.EI == 0:
            return L_eff, 0.0, np.array([]), np.array([]), np.array([]), np.array([])

        def find_phi(phi):
            if phi <= 0 or phi >= np.pi:
                return 1e10
            k = np.sin(phi / 2)
            if k >= 1:
                k = 0.999999
            K_k = ellipk(k**2)
            L_calc = 2 * np.sqrt(self.EI / F) * K_k
            return L_calc - L_eff

        try:
            phi_guess = min(F * L_eff**2 / (2 * self.EI), np.pi/2)
            result = root_scalar(find_phi, bracket=[0.01, np.pi/1.5], method='bisect', rtol=tol)
            phi = result.root
        except:
            phi = F * L_eff**2 / (2 * self.EI)
            phi = min(phi, np.pi/2)

        k = np.sin(phi / 2)
        if k >= 1:
            k = 0.999999

        K_k = ellipk(k**2)
        E_k = ellipe(k**2)

        x_end = L_eff * (2 * E_k / K_k - 1)
        y_end = L_eff * (2 * k / K_k)

        if F < 0:
            y_end = -y_end

        # 生成形状
        n_points = self.n_elements
        s_vals = np.linspace(0, L_eff, n_points)

        theta_vals = np.zeros(n_points)
        x_vals = np.zeros(n_points)
        y_vals = np.zeros(n_points)

        for i, s in enumerate(s_vals):
            theta_vals[i] = phi * (s / L_eff)
            if i == 0:
                x_vals[i] = 0
                y_vals[i] = 0
            else:
                ds = s_vals[i] - s_vals[i-1]
                theta_mid = (theta_vals[i-1] + theta_vals[i]) / 2
                x_vals[i] = x_vals[i-1] + np.cos(theta_mid) * ds
                y_vals[i] = y_vals[i-1] + np.sin(theta_mid) * ds

        scale_x = x_end / x_vals[-1] if x_vals[-1] != 0 else 1
        scale_y = y_end / y_vals[-1] if y_vals[-1] != 0 else 1
        x_vals = x_vals * scale_x
        y_vals = y_vals * scale_y

        if F < 0:
            y_vals = -y_vals

        return x_end, y_end, s_vals, x_vals, y_vals, theta_vals

    def solve(self) -> Dict:
        """求解大变形解析解"""

        # 1. 力分解
        self._decompose_force()

        F_axial = self.force_components.axial
        F_trans = self.force_components.transverse

        print(f"\n力分解结果:")
        print(f"  轴向力 F_axial = {F_axial:+.4f} N")
        print(f"  横向力 F_trans = {F_trans:+.4f} N")

        if self.fix_two_nodes:
            print(f"  边界条件       = 固定两个节点")
            print(f"  有效长度 L_eff = {self.force_distance:.4f} m")
            print(f"  固定点偏移     = {self.fixed_offset:.4f} m")
        else:
            print(f"  边界条件       = 理想固支")
            print(f"  有效长度 L_eff = {self.force_distance:.4f} m")

        # 2. 拉伸变形
        delta_stretch = self._solve_stretching()
        self.delta_stretch = delta_stretch

        # 3. 弯曲变形
        L_eff_bending = self.force_distance

        x_bend, y_bend, s_vals, x_shape, y_shape, theta_shape = self._solve_euler_elastica(
            abs(F_trans), L_eff_bending
        )

        if F_trans < 0:
            y_bend = -y_bend

        # 4. 施加拉伸效应
        stretch_ratio = (L_eff_bending + delta_stretch) / L_eff_bending if L_eff_bending > 0 else 1
        end_x_bend = x_bend * stretch_ratio
        end_y_bend = y_bend

        # 应用偏移
        self.end_x = self.fixed_offset + end_x_bend
        self.end_y = end_y_bend

        # 5. 生成节点位置
        self._generate_node_positions(x_shape * stretch_ratio, y_shape, s_vals)

        # 计算总位移
        initial_force_x = self.force_node * self.segment_length
        total_disp_x = self.end_x - initial_force_x
        total_disp_y = self.end_y

        return {
            'boundary_condition': 'fixed_two_nodes' if self.fix_two_nodes else 'ideal_clamped',
            'force_components': {
                'axial': F_axial,
                'transverse': F_trans
            },
            'stretching': {
                'delta': delta_stretch,
                'strain': delta_stretch / self.force_distance if self.force_distance > 0 else 0
            },
            'bending_x': x_bend - L_eff_bending,
            'bending_y': y_bend,
            'total_displacement': {
                'x': total_disp_x,
                'y': total_disp_y,
                'magnitude': np.sqrt(total_disp_x**2 + total_disp_y**2)
            },
            'force_point_position': np.array([self.end_x, self.end_y])
        }

    def _generate_node_positions(self, x_shape: np.ndarray, y_shape: np.ndarray, s_vals: np.ndarray):
        """生成所有节点的最终位置"""

        # 节点原始坐标
        self.initial_positions = np.zeros((self.num_nodes, 2))
        for i in range(self.num_nodes):
            self.initial_positions[i] = [i * self.segment_length, 0.0]

        self.final_positions = self.initial_positions.copy()

        force_pos_initial = self.force_node * self.segment_length

        if self.fix_two_nodes:
            # 节点0和节点1固定
            self.final_positions[0] = [0.0, 0.0]
            self.final_positions[1] = [self.segment_length, 0.0]
            start_idx = 1
            effective_force_pos = force_pos_initial - self.segment_length
        else:
            start_idx = 0
            effective_force_pos = force_pos_initial

        # 生成形状插值
        if len(x_shape) > 0 and len(y_shape) > 0:
            n_shape = len(x_shape)
            for i in range(start_idx, self.num_nodes):
                s = i * self.segment_length

                if s < force_pos_initial - 1e-10:
                    # 施力点之前：从形状插值
                    if effective_force_pos > 1e-10:
                        if self.fix_two_nodes:
                            effective_s = s - self.segment_length
                        else:
                            effective_s = s
                        ratio = effective_s / effective_force_pos
                        ratio = max(0, min(1, ratio))
                        idx = int(ratio * (n_shape - 1))
                        idx = min(idx, n_shape - 1)

                        if self.fix_two_nodes:
                            self.final_positions[i, 0] = self.segment_length + x_shape[idx]
                        else:
                            self.final_positions[i, 0] = x_shape[idx]
                        self.final_positions[i, 1] = y_shape[idx]
                    else:
                        self.final_positions[i, 0] = s
                        self.final_positions[i, 1] = 0

                elif abs(s - force_pos_initial) < 1e-10:
                    # 施力点
                    self.final_positions[i, 0] = self.end_x
                    self.final_positions[i, 1] = self.end_y

                else:
                    # 施力点之后：沿切线方向延伸
                    if len(x_shape) > 1:
                        dx = x_shape[-1] - x_shape[-2]
                        dy = y_shape[-1] - y_shape[-2]
                        tangent = np.array([dx, dy])
                        norm = np.linalg.norm(tangent)
                        if norm > 0:
                            tangent = tangent / norm
                        else:
                            tangent = np.array([1.0, 0.0])
                    else:
                        tangent = np.array([1.0, 0.0])

                    extension = s - force_pos_initial
                    self.final_positions[i, 0] = self.end_x + extension * tangent[0]
                    self.final_positions[i, 1] = self.end_y + extension * tangent[1]
        else:
            # 无形状数据，线性插值
            for i in range(self.num_nodes):
                s = i * self.segment_length
                if s <= force_pos_initial:
                    if effective_force_pos > 0:
                        ratio = max(0, min(1, (s - start_idx * self.segment_length) / effective_force_pos))
                    else:
                        ratio = 0

                    if self.fix_two_nodes and i >= start_idx:
                        self.final_positions[i, 0] = self.segment_length + ratio * (self.end_x - self.segment_length)
                    else:
                        self.final_positions[i, 0] = ratio * self.end_x
                    self.final_positions[i, 1] = ratio * self.end_y
                else:
                    self.final_positions[i, 0] = self.end_x + (s - force_pos_initial)
                    self.final_positions[i, 1] = self.end_y

        # 确保固定点正确
        if self.fix_two_nodes:
            self.final_positions[0] = [0.0, 0.0]
            self.final_positions[1] = [self.segment_length, 0.0]

        # 确保施力点准确
        self.final_positions[self.force_node, 0] = self.end_x
        self.final_positions[self.force_node, 1] = self.end_y

    def get_node_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.initial_positions, self.final_positions


# ============================================================================
# 可视化函数
# ============================================================================

def plot_large_deflection(solver: LargeDeflectionAnalytic, result: Dict,
                          save_path: Optional[str] = None):
    """绘制大变形形状图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 获取节点坐标
    initial_pos, final_pos = solver.get_node_positions()

    # 子图1：变形形状对比
    ax1 = axes[0]

    # 绘制初始形状
    ax1.plot(initial_pos[:, 0], initial_pos[:, 1], 'b--', linewidth=2,
             label='Initial shape', alpha=0.7)

    # 绘制变形后形状
    ax1.plot(final_pos[:, 0], final_pos[:, 1], 'r-', linewidth=2,
             label='Deformed shape')

    # 标记节点
    ax1.scatter(initial_pos[:, 0], initial_pos[:, 1], c='blue', s=20, alpha=0.5)
    ax1.scatter(final_pos[:, 0], final_pos[:, 1], c='red', s=30, alpha=0.7)

    # 标记固定端
    ax1.scatter(0, 0, c='black', s=100, marker='s', label='Fixed end', zorder=5)

    if solver.fix_two_nodes:
        ax1.scatter(solver.segment_length, 0, c='black', s=80, marker='s',
                    label='Second fixed node', zorder=5, alpha=0.7)

    # 标记施力点
    force_pos = result['force_point_position']
    ax1.scatter(force_pos[0], force_pos[1],
                c='green', s=100, marker='^', label=f'Force point (node {solver.force_node})', zorder=5)

    # 绘制力矢量
    force_angle_rad = np.radians(solver.force_angle_deg)
    force_vector = np.array([np.cos(force_angle_rad), np.sin(force_angle_rad)])
    force_vector = force_vector * solver.force_magnitude * 0.05
    ax1.arrow(force_pos[0], force_pos[1],
              force_vector[0], force_vector[1],
              head_width=0.05, head_length=0.05, fc='green', ec='green',
              label=f'F = {solver.force_magnitude} N, {solver.force_angle_deg}°')

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    boundary_label = "Fixed Two Nodes" if solver.fix_two_nodes else "Ideal Clamped"
    ax1.set_title(f'Large Deflection ({boundary_label})\nL={solver.L}m, EA={solver.EA:.0f}N, EI={solver.EI:.2f}N·m²')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 设置坐标轴范围
    all_x = np.concatenate([initial_pos[:, 0], final_pos[:, 0], [force_pos[0]]])
    all_y = np.concatenate([initial_pos[:, 1], final_pos[:, 1], [force_pos[1]]])
    x_min = min(np.min(all_x), 0) - 0.2
    x_max = max(np.max(all_x), solver.L) + 0.2
    y_min = min(np.min(all_y), -0.2) - 0.2
    y_max = max(np.max(all_y), 2.0) + 0.2
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    # 子图2：位移分布
    ax2 = axes[1]

    displacements = final_pos - initial_pos
    x_coords = initial_pos[:, 0]

    ax2.plot(x_coords, displacements[:, 0], 'b-o', linewidth=2, markersize=4, label='Displacement X')
    ax2.plot(x_coords, displacements[:, 1], 'r-o', linewidth=2, markersize=4, label='Displacement Y')
    ax2.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    ax2.axvline(x=initial_pos[solver.force_node, 0], color='green', linestyle='--', alpha=0.5,
                label=f'Force point at x={initial_pos[solver.force_node, 0]:.2f}m')

    ax2.set_xlabel('Initial x position (m)')
    ax2.set_ylabel('Displacement (m)')
    ax2.set_title('Node Displacement Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")

    plt.show()


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数：计算并输出大变形解析解结果"""

    print("\n" + "=" * 70)
    print("二维离散弹性杆大变形解析解")
    print("基于 Euler Elastica 理论 + 对数应变")
    print("=" * 70)
    print(f"梁长度 L           = {L} m")
    print(f"分段数             = {num_segments}")
    print(f"节点数             = {num_segments + 1}")
    print(f"拉伸刚度 EA        = {EA:.2e} N")
    print(f"弯曲刚度 EI        = {EI:.2e} N·m²")
    print(f"施力节点           = {force_node}")
    print(f"力大小 F           = {force_magnitude} N")
    print(f"力方向角度         = {force_angle_deg}°")
    print(f"边界条件           = {'固定两个节点' if fix_two_nodes else '理想固支'}")
    print("-" * 70)

    # 创建求解器
    solver = LargeDeflectionAnalytic(
        L=L, EA=EA, EI=EI,
        num_segments=num_segments,
        force_node=force_node,
        force_magnitude=force_magnitude,
        force_angle_deg=force_angle_deg,
        fix_two_nodes=fix_two_nodes
    )

    solver.n_elements = n_elements

    # 求解
    result = solver.solve()

    print("\n【结果】")
    print("-" * 50)
    print(f"  弯曲 X 方向位移 = {result['bending_x']:+.6e} m")
    print(f"  弯曲 Y 方向位移 = {result['bending_y']:+.6e} m")
    print(f"  总 X 方向位移   = {result['total_displacement']['x']:+.6e} m")
    print(f"  总 Y 方向位移   = {result['total_displacement']['y']:+.6e} m")
    print(f"  总位移大小      = {result['total_displacement']['magnitude']:.6e} m")

    force_pos = result['force_point_position']
    print(f"\n【施力点 (节点 {force_node}) 最终位置】")
    print("-" * 50)
    print(f"  x = {force_pos[0]:.6f} m")
    print(f"  y = {force_pos[1]:.6f} m")
    print("=" * 70)

    # 输出节点坐标表
    initial_pos, final_pos = solver.get_node_positions()
    displacements = final_pos - initial_pos

    print("\n【节点坐标列表】")
    print("-" * 80)
    print(f"{'节点':>6} | {'x_initial (m)':>14} | {'x_final (m)':>14} | {'y_final (m)':>14} | {'u_x (m)':>12} | {'u_y (m)':>12}")
    print("-" * 80)

    for i in range(len(initial_pos)):
        print(f"{i:6d} | {initial_pos[i,0]:14.6f} | {final_pos[i,0]:14.6f} | {final_pos[i,1]:14.6f} | {displacements[i,0]:12.6f} | {displacements[i,1]:12.6f}")

    print("-" * 80)

    # 绘制形状
    if plot_shape:
        plot_large_deflection(solver, result, save_path=plot_path if save_plot else None)

    return result


if __name__ == "__main__":
    result = main()