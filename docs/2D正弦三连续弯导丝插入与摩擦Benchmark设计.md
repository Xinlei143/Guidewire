# 2-D 正弦三连续弯导丝插入与摩擦 Benchmark 设计

## 1. 用途与证据边界

本 benchmark 用于检查一个自然直、平面离散弹性杆在恒宽刚性正弦管内接受近端
位移送入时的接触、摩擦、反力与离散收敛行为。它是数值算法 benchmark，不是
导丝、血管或润滑层的材料标定。默认摩擦系数 `mu=0.10` 只是名义仿真系数；默认
接触与切向线刚度来自旧 Arc2D 节点参数在 2.5 m/800 节点网格上的尺度换算。

该阶段不包含 3-D、mesh、血流、柔性血管、twist、J-tip、自接触、分段材料、
黏附、JKR 或水化润滑。因此，无论求解是否收敛，结果都不能外推为实验性能、
器械安全或临床有效性证据。导丝有限元的近端加载背景可参考
[Guidewire FE study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9937453/)；水凝胶
导丝的黏附与水化润滑属于后续模型，可参考
[Hydrogel guidewire study](https://www.sciencedirect.com/science/article/pii/S0927775725033126)。

## 2. 几何、单位与初始构型

全部长度单位为 m，力为 N，时间为 s，能量为 J。默认中轴线为

\[
y_c(x)=A\cos(3\pi x/L_x),\qquad A=0.225,\quad L_x=1.50.
\]

管半宽为 \(h=0.05\)，上下壁由单位法向 \(\mathbf n(x)\) 严格偏移：

\[
\mathbf w_\pm(x)=\mathbf c(x)\pm h\mathbf n(x).
\]

通用几何合法性要求最小曲率半径 \(R_{\min}>h\)，以避免法向偏移壁在曲率
中心处产生 cusp，并额外拒绝单壁自交和上下壁互交。默认几何还必须通过更强的
\(R_{\min}\approx0.1126\ {\rm m}>2h=0.10\ {\rm m}\) 余量检查；这一 \(2h\) 不等式是
默认 benchmark 的额外验收量，不是所有无 cusp 自定义管的必要条件。默认中心线长度
约 2.086 m。最近点投影先使用缓存参数进行有保护 Newton；若驻点、
区间或距离下降检查失败，则枚举全域粗采样的局部候选、分别细化，并与两个端点
比较。Python 与 C++ 使用同一选择规则。

初始导丝总长 2.50 m，其中 0.12 m 沿管中心线，其余部分位于入口左侧的半无限
feeder：

\[
x<0,\qquad y=A.
\]

feeder 在接触数学中没有有限长度，图中截断仅为显示。直线与中心线在入口位置和
切线连续。rest length 从该离散初态逐段建立，所以初始 stretching energy 为零；
自然曲率保持为零。

## 3. 边界、法向接触与 Voronoi 缩放

节点 0 和 1 始终同步沿 \(+x\) 移动，保持 \(y=A\) 及初始相对间距；远端 tip
完全自由。没有 outlet end-cap 或远端位移边界。解析配置要求

\[
L_{\rm inserted,0}+\Delta L_{\rm insert}
<L_{\rm vessel}-L_{\rm outlet\ margin}.
\]

运行中还按 tip 最近点的中心线弧长检查出口余量，越界即停止并记录失败原因。

血管法向接触保留 softplus activation、PI integral、leak、anti-windup、force cap
和 Kp/Ki 自适应。`safe_distance=0.01 m` 是壁面 contact layer。节点力参数由
线密度乘节点 Voronoi 控制域 \(l_i^V\)：内部节点使用完整控制域，两端节点使用
半控制域。默认值为：

| 参数 | 默认值 | 单位 |
|---|---:|---|
| `contact_kp_line` | 1,118,600 | N/m² |
| `contact_ki_line` | 31,960 | N/(m²·s) |
| `contact_force_cap_line` | 6,392 | N/m |
| `friction_kt_line` | 1,118,600 | N/m² |

feeder 使用独立、无积分的平滑法向 penalty，默认半宽 0.002 m、接触层 0.0005 m。
它只恢复 y 偏差，因而对任意状态必须逐节点严格满足
\(F_{\rm feeder,x}=0\)。`feeder_kp_line` 可单独调整，虽然默认数值与
`contact_kp_line` 相同，但二者没有永久绑定。

## 4. Coulomb stick-slip 与状态提交

节点切向刚度为 \(k_{t,i}=k_{t,\ell}l_i^V\)。对增量开始时已经提交的切向弹性
滑移 \(\xi_i^c\)，trial 状态为

\[
\xi_i^{\rm trial}=\xi_i^c+\Delta u_{t,i},\qquad
F_{t,i}^{\rm trial}=-k_{t,i}\xi_i^{\rm trial}.
\]

若 \(|F_{t,i}^{\rm trial}|\le\mu N_i\)，节点 stick；否则通过精确半光滑回映射得到

\[
F_{t,i}=-\mu N_i\operatorname{sign}(\xi_i^{\rm trial}).
\]

settled 后才提交 \(\xi_i^{new}=-F_{t,i}/k_{t,i}\)。始终检查
\(|F_t|\le\mu N\) 及离散弹簧能量/塑性耗散一致性。需要注意：弹性 stick 发生
反向卸载时，瞬时 \(F_t\Delta u_t\) 不一定非正，不能用这一条点态不等式错误地
否定保守弹簧回能。摩擦仅在 vessel contact active 时存在；失去血管接触或进入
feeder 后，摩擦历史清零。

每个 Newton iteration 有两层不同状态：

1. 几何 linearization state：按当前 iterate 投影得到 closest point、tangent 和
   normal；这一 frame 在该 iteration 的 Jacobian、全部 line-search candidate 与
   残差比较中冻结。接受 step 后，下一 iteration 才可重新投影。
2. 本构 committed history：friction anchor、committed projection、committed
   contact status 和 \(\xi^c\) 在整个 load increment 内只读，只能在该增量 settled
   后提交。

若节点在增量开始时未接触而在本增量首次接触，本增量固定使用 \(\xi^c=0\)，anchor
由增量开始状态确定，绝不由 Newton 或 line-search trial 重新定义。重试细分增量在
局部事务中逐段提交；任一后续段失败时，位置、速度、PI、时间、frame、anchor、
contact status 和 \(\xi\) 一起回滚。prebalance 只打开 feeder 与血管法向 PI，关闭
摩擦；平衡接受后才初始化摩擦历史。随机 pulse 默认关闭。

## 5. 反力、力平衡与能量

唯一 insertion force 定义为近端两个约束节点的静态 actuator-on-wire force：

\[
F_{\rm insert}=-\left(\mathbf R_0+\mathbf R_1\right)\cdot(1,0).
\]

实现中 `proximal_reaction` 是导丝对 actuator 的反力，
`actuator_force_on_wire=-proximal_reaction`，因此保存的 scalar 为
`insertion_force=actuator_force_on_wire_x`。远端法向或摩擦合力不会再次加到该
scalar 上。另存 normal、friction、feeder 的全局 x/y 合力，以及 inertia/damping
remainder。

settled 静态平衡使用作用在导丝上的 actuator force：

\[
\mathbf e_F=\mathbf F_{\rm actuator\ on\ wire}
+\sum_i(\mathbf F_{n,i}+\mathbf F_{t,i}+\mathbf F_{feeder,i}),
\]

\[
e_{\rm balance}=\|\mathbf e_F\|,\qquad
e_{\rm balance,rel}=\frac{\|\mathbf e_F\|}
{\max(\|\mathbf F_{\rm actuator\ on\ wire}\|,10^{-12}\ {\rm N})}.
\]

动态平衡另用明确命名的 dynamic actuator force 与 inertia/damping remainder，不与
静态插入力混用。导出 settled 证据前强制要求 feeder x 合力严格为零，并要求
inertia/damping remainder 与 actuator force 的范数比不超过 5%（分母下限
(10^{-12}\,\mathrm N)）。不满足时拒绝导出为 settled benchmark 产物。elastic
energy 只等于 stretching 与 bending；PI 和摩擦始终标为非保守力。

## 6. 单次复现与产物

从仓库根目录运行 nominal C++ 算例：

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/softrobot-mplconfig \
conda run --no-capture-output -n pendulum-rl python -u \
  rod_soft_contact_PI_dynamic_quasi_static_2d_sine_insert.py \
  --backend cpp --output-dir 2d_npz --run-name sine_nominal_cpp
```

每次运行保存到独立目录：

- `config.json`：全部配置、中心线长度和最小曲率半径；
- `settled.tsv`：settled step 的反力、合力、平衡、tip、能量和 Newton 诊断；
- `history.npz`：节点位置、速度、PI、摩擦、接触和全局量的历史；
- `configurations.png`；
- `insertion_force_vs_insertion.png`；
- `normal_friction_resultants_vs_insertion.png`；
- `elastic_energy_vs_insertion.png`。

目录 `2d_npz/` 默认由 Git 忽略，运行器不会自动 commit 或 push。
一个 case 只有在 `config`、`settled`、`history`、`configurations`、
`insertion_force_plot`、`normal_friction_plot` 和 `elastic_energy_plot` 七个标准
文件键全部指向 `run_dir` 内的非空普通文件，且 `run_dir` 本身是预期目录时，
才有资格进入聚合证据。

## 7. 收敛与摩擦扫描

正式 campaign 使用一个全新目录，禁止覆盖已有目录：

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/softrobot-mplconfig \
conda run --no-capture-output -n pendulum-rl python -u \
  sine2d_benchmark_campaign.py --backend cpp --study all \
  --output-dir 2d_npz --campaign-name sine2d_formal_YYYYMMDD
```

空间组为 400/800/1200 节点、60 步；时间组为 800 节点、60/120 步。只有未通过
门槛的维度才追加 1600 节点或 240 步。所有力曲线先限制到共同插入区间，再取
默认 241 点的均匀网格与所有源曲线 knot 的有序并集。这样即使减少均匀采样数，也不会
遗漏仅出现在某条源曲线内部 knot 上的尖峰。细网格/细时间步作为 reference，门槛为：

| 指标 | 门槛 |
|---|---:|
| insertion-force 曲线相对 L2 | ≤ 5% |
| insertion-force 绝对峰值相对差 | ≤ 5% |
| insertion-force 末值相对差 | ≤ 5% |
| 最终 tip centerline s 差 | ≤ 5 mm |
| 全程最大穿透差 | ≤ 0.1 mm |

摩擦扫描固定为 `mu=0,0.05,0.10,0.20`。`mu_sweep_unified.tsv` 保存共同网格上的
四条插入力曲线，`mu_sweep_insertion_force_overlay.png` 保存叠加图。全部 case、
运行状态、产物相对路径、门槛和 pairwise 指标写入 `campaign_manifest.json`。

manifest 的 `evidence_status` 只有在所需数值对照实际满足门槛、且 `all`
campaign 的摩擦扫描也全部完成时，才是 `converged` 或
`converged_after_refinement`。收敛细化仍失败保留为 `not_converged`；
收敛通过但 `all` 中的摩擦 case 失败则是 `campaign_incomplete`。只运行摩擦
扫描时收敛状态是 `not_assessed`，扫描本身另标为 `complete` 或 `incomplete`。
仅数值求解已返回失败状态才属于 `not_converged`；运行器异常、导出失败、载荷步数不全或
未完整覆盖每个 \(i\Delta L_{\rm total}/N_{\rm load}\) settled 载荷目标都属于
`campaign_incomplete`。允许额外的预平衡零插入记录和重试细分增量的 settled knot，
但这些额外 knot 不能替代任一标准载荷目标；仅有末点而缺少中间目标的稀疏曲线
不具备证据资格。这些不完整状态不会被错写成数值不收敛，也不会触发无意义的细化。
失败产物应保留用于诊断，但不能改写为定量验证完成。

## 8. 测试与限制

轻量单元测试不会启动 400–1600 节点的正式计算：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
conda run --no-capture-output -n pendulum-rl python -m pytest -q \
  tests/test_sine2d_geometry.py \
  tests/test_sine2d_insert_contact.py \
  tests/test_sine2d_friction.py \
  tests/test_sine2d_insertion.py \
  tests/test_sine2d_outputs.py \
  tests/test_sine2d_cpp_backend.py \
  tests/test_sine2d_campaign.py
```

通过单元测试说明公式、状态机、输出 schema 和后端 parity 满足规定容差，不替代
正式收敛 campaign。完成 campaign 后仍须核对所有 case 的 `success`、
`completed_load_steps`、settled 条件、static/full balance、feeder x 合力和 manifest
状态，再决定能够支持何种数值结论。
