# SoftRobot：v4 三维 DER + PI 接触仿真

当前 README 以 v4 为主。v4 是独立的三维导丝准静态仿真入口，包含：

- 无扭转离散弹性杆（DER）的拉伸、弯曲和隐式 Euler/Newton 求解；
- 解析直圆管接触或 OBJ/STL 三角网格接触；
- 初始随机横向脉冲、末端位移续加载和自适应松弛；
- Python/PyTorch 参考后端与 C++ CPU 后端；
- NPZ 历史、最终 PNG 和 TSV 汇总输出。

所有长度单位为 m，力为 N，时间为 s。v4 的主入口是
`rod_soft_contact_PI_dynamic_quasi_static_3d_v4.py`；v3、旧的
`Stretching_bending/` 示例和其他脚本是独立的历史入口，不是 v4 的运行依赖。

## 1. 快速开始

从仓库根目录执行：

```bash
cd /home/xinlei/Data/SoftRobot
```

推荐先运行 C++ + 解析圆管组合：

```bash
conda run --no-capture-output -n pendulum-rl python -u \
  rod_soft_contact_PI_dynamic_quasi_static_3d_v4.py \
  --backend cpp --contact-type tube --device cpu \
  --pulse-seed 42 --no-show
```

这里的 `--no-capture-output` 和 `python -u` 用于实时看到 `[LOAD ...]`、
`[RELAX ...]` 等进度信息。普通 `conda run` 可能会缓存子进程 stdout，导致仿真
正在运行时终端看起来没有任何输出。

## 2. v4 的 backend × contact-type

| backend | contact-type | 适用场景 | 设备限制 |
|---|---|---|---|
| `cpp` | `tube` | 默认推荐；CPU 上运行 DER/Newton 热路径 | 仅 CPU、`float64` |
| `python` | `tube` | Python 参考实现、结果 A/B 对照 | CPU 或 CUDA |
| `cpp` | `curved-tube` | 解析光滑 90° 圆弧血管，无需 OBJ | 仅 CPU、`float64` |
| `python` | `curved-tube` | 光滑曲管参考实现与 Jacobian 对照 | CPU 或 CUDA |
| `cpp` | `mesh` | OBJ 三角网格接触的 CPU 运行 | 仅 CPU、`float64` |
| `python` | `mesh` | OBJ 网格参考实现或 CUDA 运行 | CPU 或 CUDA |

### 2.1 解析直圆管接触

```bash
# Python backend（参考路径）
conda run --no-capture-output -n pendulum-rl python -u \
  rod_soft_contact_PI_dynamic_quasi_static_3d_v4.py \
  --backend python --contact-type tube --device cpu \
  --pulse-seed 42 --no-show

# C++ backend（推荐的 CPU 路径）
conda run --no-capture-output -n pendulum-rl python -u \
  rod_soft_contact_PI_dynamic_quasi_static_3d_v4.py \
  --backend cpp --contact-type tube --device cpu \
  --pulse-seed 42 --no-show
```

### 2.2 OBJ/STL 三角网格接触

下面的 OBJ 是仓库中与 v4 参数匹配的保护管网格。`mesh` 模式支持 OBJ 以及
ASCII/binary STL；OBJ 必须包含名称含 `protection` 的面组，STL 会将全部三角形
作为 protection surface。

网格初始化时，v4 对 protection surface 建立 edge adjacency，在每个 connected
component 内用 BFS 恢复一致 triangle winding，提取所有 boundary loops，并只在
方向判定期间用 centroid fan 临时封口后计算 signed volume。临时 cap 不会进入
runtime contact mesh、C++ `MeshBVH` 或 Newton contact search，因此不依赖全局 x/y/z
轴，也不要求额外的 centerline。无法处理的 non-manifold、严重非平面/自交边界、
退化三角形或近零 signed volume 会以 geometry validation error 拒绝。

网格表示的是已经编码在 OBJ/STL 中的 protection surface。其语义是：

```text
mesh contact operates on the protection surface encoded in the mesh;
safe_distance is not applied a second time during runtime mesh contact.
```

初始化会记录 `mesh_parse_ms`、`mesh_orientation_ms`、`mesh_cap_ms` 和
`mesh_preprocess_total_ms`，并在 mesh simulation 启动时打印。它们只属于一次性
mesh initialization，不计入后续每个 Dt / Newton step 的 `wall_time`。

```bash
# Python backend
conda run --no-capture-output -n pendulum-rl python -u \
  rod_soft_contact_PI_dynamic_quasi_static_3d_v4.py \
  --backend python --contact-type mesh \
  --mesh-obj tube_obj/straight_tube_L1p5m_boundaryR50mm_shellR60mm_32sides_axial2.obj \
  --device cpu --pulse-seed 42 --no-show

# C++ backend
conda run --no-capture-output -n pendulum-rl python -u \
  rod_soft_contact_PI_dynamic_quasi_static_3d_v4.py \
  --backend cpp --contact-type mesh \
  --mesh-obj tube_obj/straight_tube_L1p5m_boundaryR50mm_shellR60mm_32sides_axial2.obj \
  --device cpu --pulse-seed 42 --no-show
```

Python 网格路径也可以使用 CUDA：

```bash
conda run --no-capture-output -n pendulum-rl python -u \
  rod_soft_contact_PI_dynamic_quasi_static_3d_v4.py \
  --backend python --contact-type mesh \
  --mesh-obj tube_obj/straight_tube_L1p5m_boundaryR50mm_shellR60mm_32sides_axial2.obj \
  --device cuda --pulse-seed 42 --no-show
```

### 2.3 与 v4 默认截面一致的 90° 三维圆弧血管

仓库提供一对由同一确定性脚本生成的几何输入：

```text
tube_obj/arc90_L1p5m_wallR60mm_protectionR50mm_64sides_axial256.obj
tube_obj/arc90_L1p5m_centerline_800.csv
```

CSV 必须使用 `x,y,z` 表头，单位为 m，点序从入口到出口。中轴线继续严格采用
2-D 默认算例：长度为 1.5 m、转角为 90°、弦长修正半径为
`0.9549298123336835 m`。截面改用 v4 默认尺寸，实体管腔
半径为 60 mm；OBJ 中名称含 `protection` 的接触面半径为 50 mm，与 v4 默认
直管截面一致，并且已经编码了
10 mm 安全距离。因此 mesh runtime 不会再将 `--safe-distance` 偏移第二次。

曲管模式由 `--centerline-csv` 显式启用，不改变无参数运行的直管默认行为。导丝
沿中轴线初始化，先在零内缩、两端固定条件下预平衡，然后在 Bishop 局部法平面
施加三维随机脉冲；末端压缩位置沿中轴线弧长回退。

300 节点、与 2-D 参数对齐的验证命令为：

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/softrobot-mplconfig \
conda run --no-capture-output -n pendulum-rl python -u \
  rod_soft_contact_PI_dynamic_quasi_static_3d_v4.py \
  --backend cpp --device cpu --contact-type mesh \
  --mesh-obj tube_obj/arc90_L1p5m_wallR60mm_protectionR50mm_64sides_axial256.obj \
  --centerline-csv tube_obj/arc90_L1p5m_centerline_800.csv \
  --num-nodes 300 --tube-radius 0.06 --safe-distance 0.01 \
  --Kp 3500 --Ki 100 --dt 0.005 \
  --tip-total-displacement 0.12 --num-load-steps 20 \
  --pulse-seed 7 --prebalance --skip-animation --no-show
```

需要重新生成资产时执行：

```bash
conda run -n pendulum-rl python tools/generate_arc_vessel_3d.py
```

### 2.4 不使用 OBJ 的解析光滑曲管

`--contact-type curved-tube` 直接使用与 2-D 弯曲血管脚本相同的默认参数，并将该圆弧嵌入全局 xz 平面（y=0）：
1.5 m 离散弦长、90° 圆弧、800 点弦长修正半径
`0.9549298123336835 m`、实体管腔半径 50 mm、10 mm 安全层和 40 mm
接触保护半径。接触距离是节点到圆弧中轴线的三维解析距离，法向和完整 3×3
Jacobian 均连续，不读取 `--mesh-obj`，也不受三角面数量或法向跳变影响。

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/softrobot-mplconfig \
conda run --no-capture-output -n pendulum-rl python -u \
  rod_soft_contact_PI_dynamic_quasi_static_3d_v4.py \
  --backend cpp --device cpu --contact-type curved-tube \
  --num-nodes 300 --Kp 3500 --Ki 100 --dt 0.005 \
  --tip-total-displacement 0.12 --num-load-steps 20 \
  --pulse-seed 7 --prebalance --skip-animation --no-show
```

若显式传入 `--tube-radius` 或 `--safe-distance`，可覆盖上述截面默认值；不传时
`curved-tube` 使用 2-D 的 50/10 mm，而原有直管 `tube` 仍使用 v4 的 60/10 mm。

省略 `--backend` 和 `--contact-type` 时，默认分别为 `cpp` 和 `tube`。C++ 后端
不能配合 `--device cuda`；需要 GPU 时使用 Python 后端。

## 3. 构建与自检

C++ 扩展会通过 `torch.utils.cpp_extension.load` 延迟编译到
`/tmp/softrobot_der_cpp`。首次运行时也会触发编译，建议先显式做一次构建检查：

```bash
conda run --no-capture-output -n pendulum-rl python -u \
  cpp/build_der_core.py --nodes 8
```

只检查 v4 几何、能量、接触和脉冲配置，不启动完整加载过程：

```bash
conda run --no-capture-output -n pendulum-rl python -u \
  rod_soft_contact_PI_dynamic_quasi_static_3d_v4.py --self-test
```

运行 v4 相关测试：

```bash
conda run --no-capture-output -n pendulum-rl python -m pytest -q \
  tests/test_v4_mesh_contact.py \
  tests/test_v4_cpp_backend.py \
  tests/test_v3_phase1.py
```

如果本机 pytest 自动加载了不兼容的外部插件，可在命令前增加
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。

## 4. 常用参数

先用默认参数确认流程，再按实验目的覆盖参数。完整列表以
`--help` 为准：

```bash
conda run --no-capture-output -n pendulum-rl python -u \
  rod_soft_contact_PI_dynamic_quasi_static_3d_v4.py --help
```

| 类别 | 参数 | 默认值 | 含义 |
|---|---|---:|---|
| 后端 | `--backend` | `cpp` | `cpp` 或 `python` |
| 接触 | `--contact-type` | `tube` | 解析圆管或 OBJ 网格 |
| 设备 | `--device` | `auto` | `auto`、`cpu`、`cuda`；C++ 最终固定为 CPU |
| 几何 | `--total-length` | `1.5` | 导丝初始长度 |
| 几何 | `--diameter` | `0.001` | 导丝直径 |
| 离散 | `--num-nodes` | `250` | 杆节点数，越大越精细也越慢 |
| 材料 | `--density` | `7850` | 材料密度，kg/m³ |
| 圆管 | `--tube-radius` | `0.06` | 管内半径 |
| 圆管/网格 | `--safe-distance` | `0.01` | 接触保护层厚度 |
| PI | `--Kp` / `--Ki` | `3000` / `100` | 接触比例、积分增益 |
| 时间步 | `--dt` | `0.05` | 隐式 Euler/松弛步长，不等同于真实加载时间 |
| 续加载 | `--tip-total-displacement` | `0.09` | 末端总压缩位移，沿 `-x` |
| 续加载 | `--num-load-steps` | `20` | 总位移分成的载荷步数 |
| 续加载 | `--ramp-mode` | `adaptive` | 自适应或固定末端 ramp |
| 脉冲 | `--pulse-count-min/max` | `4` / `8` | 随机脉冲节点数量范围 |
| 脉冲 | `--pulse-force-min/max` | `0.02` / `0.07` | 随机脉冲力幅值范围，N |
| 脉冲 | `--pulse-duration` | `0.02` | 初始脉冲持续时间 |
| 复现 | `--pulse-seed` | 未设置 | 设置整数可复现实验；省略则每次重新随机 |
| 输出 | `--output-prefix` | 自动命名 | 指定 NPZ 文件路径前缀 |
| 显示 | `--no-show` | 否 | 不弹出窗口，但仍保存最终 PNG |

曲管模式另外使用 `--centerline-csv`、`--prebalance/--no-prebalance` 和
`--prebalance-max-steps`。提供中轴线时预平衡默认开启；直管模式默认不开启。

脉冲有两种互斥用法：

- 默认模式：不提供 `--pulse-nodes` 和 `--pulse-fyfz`，节点、幅值和 y/z 方向
  自动随机；可用 `--pulse-seed` 固定随机序列。
- 显式模式：两个参数必须同时提供。例如
  `--pulse-nodes 5,10,20 --pulse-fyfz 0.05,0.0,0.02,0.04,0.01,0.03`，其中每个
  节点对应一对 `(Fy,Fz)` 分量。

## 5. 输出与结果检查

默认输出位置：

```text
3d_npz/<run_stem>.npz                         完整历史和诊断量
3d_picture/<run_stem>_final.png              最终构型、脉冲、接触和能量图
3d_picture/random_pulse_experiment_summary_elastic_energy.tsv
                                               每次运行一行的汇总表
```

`--output-prefix results/test_run` 会把历史写入
`results/test_run.npz`；最终 PNG 和汇总表仍使用 `3d_picture/`，并沿用同一个
`run_stem` 关联。

终端出现 `completed_load_steps` 或 `all load steps settled` 后，仍应检查：

1. `completed_load_steps` 是否等于 `num_load_steps`；
2. Newton 残差、最大自由节点速度和接触误差是否达到阈值；
3. NPZ 是否成功保存，能量/接触历史是否完整；
4. 性能比较时查看每个隐式步的 `wall_time` 与 `dt`，不要只依据终端是否返回。

## 6. 常见问题

### 运行时没有打印进度

使用下面的形式关闭 `conda run` 的输出捕获，并让 Python 不缓冲 stdout：

```bash
conda run --no-capture-output -n pendulum-rl python -u \
  rod_soft_contact_PI_dynamic_quasi_static_3d_v4.py \
  --backend python --contact-type tube --device cpu \
  --pulse-seed 42 --no-show
```

每一行续行符必须是行尾的单个反斜杠 `\`；反斜杠后不要再放空格或其他字符。

### C++ 后端提示 CUDA 不可用或设备错误

这是预期限制：C++ 后端只接受 CPU、`float64` 数组。改用
`--backend python --device cuda`，或显式指定 `--device cpu`。

### mesh 模式找不到网格

确认当前工作目录是仓库根目录，或者给 `--mesh-obj` 传绝对路径；OBJ 需要存在
名称包含 `protection` 的面组，STL 则应包含完整三角形 facets。若网格存在
non-manifold edge、无法闭合的 boundary loop、严重非平面/自交 boundary 或近零
signed volume，初始化会明确报错而不会猜测 lumen 方向。

### 单个载荷步失败

先保存失败输出和终端诊断，再尝试减小 `--tip-total-displacement`、增加
`--max-settle-steps`，或使用 `--ramp-mode fixed` 做对照。不要把“进程结束”直接
等同于物理任务成功。

## 7. 代码布局

```text
rod_soft_contact_PI_dynamic_quasi_static_3d_v4.py  v4 主入口、实验管理和可视化
mesh_contact_v4.py                                 OBJ/STL 网格解析与接触几何
der_cpp_backend.py                                 C++ 扩展的 Python 边界
cpp/                                                Eigen/C++ DER 核心和构建脚本
tube_obj/                                           v4 网格接触用 OBJ
rod_soft_contact_PI_dynamic_quasi_static_3d_v3.py  v3 独立入口
Stretching_bending/                                 旧的 Torch cantilever 示例
```

旧的 `Stretching_bending/` 示例仍可单独运行，但它不包含 v4 的 PI 接触、OBJ
网格、续加载或 C++ backend，不应与本 README 的 v4 命令混用。
