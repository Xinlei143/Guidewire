import numpy as np


def compute_internal_forces(positions, rest_lengths, EA, num_segments, num_nodes):
    # 1. 初始化一个全为0的力数组，形状和节点位置数组一样（比如10个节点就是(10, 2)）
    # 这个数组用来存放最终累加出来的、每个节点受到的总内力
    forces = np.zeros_like(positions)

    # 2. 遍历所有的杆段（比如你的例子里，j 会从 0 循环到 14，共15段）
    for j in range(num_segments):

        # 【第一阶段：确定当前这段杆(j)的两头连着谁，以及它们的坐标在哪】
        if j < num_nodes:
            # --- 情况A：常规杆段（第0段到第9段）---
            if j == 0:
                # 第0段比较特殊，它的起点是墙壁（固定端），终点是第0个节点
                p1 = np.array([0.0, 0.0])  # 固定端坐标永远是(0,0)
                p2 = positions[0]  # 第0个节点的当前坐标
                node_idx_1, node_idx_2 = -1, 0  # 记录两头的身份：-1代表墙壁，0代表第0号节点
            else:
                # 第1段到第9段，连接的是相邻的两个节点（比如第1段连着节点0和节点1）
                p1 = positions[j - 1]
                p2 = positions[j]
                node_idx_1, node_idx_2 = j - 1, j
        else:
            # --- 情况B：多出来的尾巴杆段（第10段到第14段）---
            # 这些杆段没有对应的节点了，它们全部挤在最后一个节点（第9号）的屁股后面
            p1 = positions[num_nodes - 1]  # 尾巴的起点是最后一个节点
            p2 = positions[num_nodes - 1]  # 尾巴的终点是自由端（坐标暂时和起点重合）
            node_idx_1, node_idx_2 = num_nodes - 1, -2  # 身份：-2代表自由

        # 【第二阶段：根据两头的坐标，算出这段杆产生了多大的弹力】
        edge_vec = p2 - p1  # 算出从起点指向终点的向量
        current_length = np.linalg.norm(edge_vec)  # 算出这段杆当前的实际长度
        safe_length = max(current_length, 1e-8)  # 防崩溃保护：如果长度算出来是0，就强制当成极小值，防止后面除以0报错
        direction = edge_vec / safe_length  # 算出单位方向向量（告诉程序弹力是往左拉还是往右拉）

        delta_length = current_length - rest_lengths[j]  # 算出伸长量（当前长度 - 初始长度）
        f_mag = (EA / rest_lengths[j]) * delta_length  # 核心胡克定律：算出弹力的大小 (刚度 k * 伸长量)
        f_vector = f_mag * direction  # 把大小和方向结合，得到弹力的向量形式

        # 【第三阶段：把算出来的弹力，分配到具体的节点上（牛顿第三定律）】
        # 杆段会对它连接的两头产生大小相等、方向相反的力。
        # 我们只关心有质量的“真实节点”（索引 >= 0），墙壁(-1)和空气(-2)不需要记录受力。

        if node_idx_1 >= 0:
            # 如果起点是真实节点，就给它加上这个力（比如杆被拉长，起点会受到向后的拉力）
            forces[node_idx_1] += f_vector

        if node_idx_2 >= 0:
            # 如果终点是真实节点，就给它减去这个力（反作用力，方向相反）
            # 注意：对于尾巴段，node_idx_2是-2，这行代码不会执行，说明尾巴的自由端不产生反作用力
            forces[node_idx_2] -= f_vector

    # 循环结束后，forces 数组里就装满了10个节点各自受到的总内力，返回给主程序
    return forces


# =============== 修改重点：积分函数加入重力 ===============
def integrate_semi_implicit_euler(positions, velocities, forces, external_forces,
                                  mass, dt, damping_c, gravity_g=9.81):
    """
    半隐式欧拉积分器
    新增了 gravity_g 参数，默认为 9.81 (m/s^2)，方向沿 Y 轴负方向
    """
    if isinstance(mass, (int, float)):
        mass = np.full(positions.shape[0], mass)

    # 1. 计算重力 (每个节点都受到向下的重力 G = m * g)
    # 形状与 positions 相同，X方向为0，Y方向为 -m*g
    gravity_forces = np.zeros_like(positions)
    gravity_forces[:, 1] = -mass * gravity_g

    # 2. 计算总合力 = 内力 + 外力(拉力) + 重力 - 阻尼力
    total_forces = forces + external_forces + gravity_forces - damping_c * velocities

    # 3. 计算加速度 a = F_total / m
    accelerations = total_forces / mass[:, np.newaxis]

    # 4. 更新速度和位置
    new_velocities = velocities + dt * accelerations
    new_positions = positions + dt * new_velocities

    return new_positions, new_velocities


def calculate_critical_dt(mass_per_node, segment_length, EA, num_segments, num_nodes):
    """
    根据柯朗-弗里德里希斯-列维条件计算最大允许时间步长。
    只考虑拉伸刚度项: dt <= sqrt(M * L / EA)

    参数:
    mass_per_node: 单个节点的质量 (float)
    segment_length: 每段杆的长度 (float)
    EA: 拉伸刚度 (float)
    num_segments: 杆段总数 (例如 15)
    num_nodes: 节点总数 (例如 10)
    """
    dt_candidates = []

    # 我们需要检查每一段杆的稳定性
    for j in range(num_segments):
        # 1. 确定这一段杆两端的有效质量
        # 在离散元/DER中，通常认为杆段连接的质量是两端节点质量的一半
        # 对于中间杆段，有效质量 = 0.5*m + 0.5*m = m
        # 对于边界杆段（固定端或自由端），有效质量 = 0.5*m (或者根据具体离散化方案)

        # 为了保守起见（保证绝对稳定），我们通常取这一段关联的最小质量。
        # 这里简化处理：
        # 如果是中间的杆段（连接两个节点），关联质量就是 mass_per_node
        # 如果是首尾杆段（连接节点和端点），关联质量约为 0.5 * mass_per_node

        if j == 0:
            # 第0段：连接 固定端(无质量) 和 节点0
            # 有效质量取节点质量的一半
            effective_mass = 0.5 * mass_per_node
        elif j == num_segments - 1:
            # 最后一段：连接 最后一个节点 和 自由端
            # 如果尾巴没有节点，只连着最后一个节点
            effective_mass = 0.5 * mass_per_node
        else:
            # 中间段：连接 节点(j-1) 和 节点(j)
            # 只要 j < num_nodes，就是标准段
            if j < num_nodes:
                effective_mass = mass_per_node
            else:
                # 这种情况理论上不会发生，除非杆段数远大于节点数+1
                # 比如第11段杆，既没连节点9也没连节点10（不存在），它是孤立的
                # 但根据之前的逻辑，第11段连的是节点9和尾巴。
                effective_mass = 0.5 * mass_per_node

        # 2. 计算这一段杆的临界时间步长
        # 公式: dt = sqrt( M_effective * L / EA )
        # 注意：这里M取的是该段关联的总质量概念，或者直接用节点质量公式推导
        # 标准推导通常是: dt <= L / c, 其中 c = sqrt(E/rho)
        # 对应离散形式就是 sqrt(m*L/EA)

        dt_limit = np.sqrt((effective_mass * segment_length) / EA)
        dt_candidates.append(dt_limit)

    # 3. 返回所有段中最严格（最小）的那个限制
    return min(dt_candidates)