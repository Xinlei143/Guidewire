# SoftRobot 导丝仿真模型目录

这是面向课程/研究复核的 CPU 代码包。19 个独立模型按物理任务及版本列在第 3 节；
v4 只是直管、弯管任务中的部分实现，不代替此前的模型。代码目录与任务对应：

```text
├── src/softrobot/
│   ├── foundations/       基础杆：伸长、弯曲、隐式 Newton、cantilever
│   ├── contact/           基础 PI 软接触：动态与准静态
│   ├── straight_tube/     直管：2-D、3-D v2/v3、mesh、当前 v4
│   ├── curved_tube/       弯管：2-D、旧 3-D analytic/mesh、当前 v4
│   ├── common/            中轴线与共享命令行工具
│   └── backends/          C++ 扩展、网格接触和 Python/C++ 边界
└── assets/geometry/        运行所需的规范 OBJ/CSV 几何（仓库根目录）
```

老师版 ZIP 不含报告、原始实验数据、历史输出、媒体、缓存、第三方源码、正弦插入和
自由插入研究线；当前合并后的研究工作区仍保留后两者（见第 6 节）。老师版模型的
完整来源和内容哈希见 `docs/model_lineage.md`。

## 1. CPU 环境与安装

最低验收基线为 Linux、Python 3.10、CPU 版 PyTorch、NumPy、SciPy、Matplotlib、
Pillow、pytest、setuptools，以及 g++/Ninja。CUDA 不是依赖，也没有作为验收条件。
精确验收版本与新环境命令见 `requirements-cpu.lock` 和 `docs/environment.md`。
本次解释器是 Python 3.10.20；已安装的 `torch.__version__` 为 `2.11.0+cu130`（CUDA-enabled distribution），
所有实际模型与 smoke 显式使用 CPU，没有执行或验收 CUDA 算例。完整 pytest 中 `torch.cuda.is_available()`
仅作为 CUDA-only 测试的 availability gate；由此产生的 skip 不代表 CPU 功能缺陷。CPU-only wheel 未单独验收。

已有依赖时可离线安装：

```bash
python -m pip install --no-deps --no-build-isolation .
softrobot models
```

C++ 后端第一次运行会由 PyTorch 扩展机制调用 Ninja/g++ 构建，并把编译缓存写到临时
目录；构建物不会写入或提交到源码树。

## 2. 统一运行接口

```bash
# 列出模型
softrobot models

# 不同任务、不同版本的单模型 CPU smoke；输出只写到指定目录
softrobot run foundation-2d-stretch-numpy --smoke-test --output-dir /tmp/foundation-smoke
softrobot run contact-2d-quasistatic-pi-v1 --smoke-test --output-dir /tmp/contact-v1-smoke
softrobot run straight-tube-v2-3d --smoke-test --output-dir /tmp/straight-v2-smoke
softrobot run curved-tube-3d-legacy-mesh --smoke-test --output-dir /tmp/curved-legacy-smoke

# 对全部 19 个模型执行 smoke
softrobot smoke --all --output-dir /tmp/softrobot-smoke
```

每个模型也可按模块独立运行。以下示例分别覆盖基础杆、接触、直管和弯管；
全部模块入口见第 3 节：

```bash
python -m softrobot.foundations.implicit_newton_3d --smoke-test --output-dir /tmp/newton3d
python -m softrobot.contact.quasistatic_pi_v2 --smoke-test --output-dir /tmp/contact-v2
python -m softrobot.straight_tube.v3 --smoke-test --output-dir /tmp/straight-v3
python -m softrobot.curved_tube.arc_2d --smoke-test --output-dir /tmp/arc2d
```

注册表中的每个 Model ID 对应一个独立模块入口；共用底层求解器的不同接触配置也有各自
的入口。不带 `--smoke-test` 时保留历史模型原有默认参数和算例语义。smoke 模式才会缩小节点
数、步数和输出规模，并固定为 CPU、Matplotlib Agg 无头后端和确定性随机种子。

## 3. 按任务与版本查找模型

下表每行都是可单独运行的 Model ID。统一入口为 `softrobot run <Model ID>`；“模块入口”
列可接在 `python -m` 后独立运行。表中“历史”表示保留的独立物理/数值实现，
“当前”表示整理时的研究实现，**不表示历史模型已被替代**。smoke 只检查小参数 CPU
端到端运行，不证明物理精度；来源、内容哈希及更细的验证证据见
`docs/model_lineage.md`。

### 3.1 基础杆模型（5）

| Model ID | 版本/分支 | 维度 | 任务与接触 | 后端 | 模块入口（`python -m`） | CPU smoke |
|---|---|---:|---|---|---|---|
| `foundation-2d-stretch-numpy` | 历史：伸长 | 2-D | 无接触杆伸长 | NumPy | `softrobot.foundations.stretch_2d_numpy` | 已覆盖 |
| `foundation-2d-stretch-bend-torch` | 历史：伸长＋弯曲 | 2-D | 无接触杆弹性 | Torch | `softrobot.foundations.stretch_bend_2d_torch` | 已覆盖 |
| `foundation-3d-stretch-bend-torch` | 历史：3-D 弹性 | 3-D | 无接触杆伸长＋弯曲 | Torch | `softrobot.foundations.stretch_bend_3d_torch` | 已覆盖 |
| `foundation-3d-implicit-newton` | 历史：隐式积分 | 3-D | 无接触杆隐式 Euler/Newton | Torch | `softrobot.foundations.implicit_newton_3d` | 已覆盖 |
| `foundation-cantilever-der` | 历史：最终悬臂杆 | 3-D | 固定端、点载荷下的无接触杆 | Torch | `softrobot.foundations.cantilever.run_cantilever_demo` | 已覆盖 |

### 3.2 基础 PI 接触（3）

| Model ID | 版本/分支 | 维度 | 任务与接触 | 后端 | 模块入口（`python -m`） | CPU smoke |
|---|---|---:|---|---|---|---|
| `contact-2d-dynamic-pi` | 历史：动态 | 2-D | 平行边界／PI 软接触 | Torch | `softrobot.contact.dynamic_pi_2d` | 已覆盖 |
| `contact-2d-quasistatic-pi-v1` | 历史：准静态 v1 | 2-D | 平行边界／PI 框架，积分力项关闭 | Torch | `softrobot.contact.quasistatic_pi_v1` | 已覆盖 |
| `contact-2d-quasistatic-pi-v2` | 历史：准静态 v2 | 2-D | 平行边界／PI 软接触，积分力项启用 | Torch | `softrobot.contact.quasistatic_pi_v2` | 已覆盖 |

### 3.3 直管任务（5）

| Model ID | 版本/分支 | 维度 | 任务与接触 | 后端 | 模块入口（`python -m`） | CPU smoke |
|---|---|---:|---|---|---|---|
| `straight-tube-2d-pi` | 历史：2-D | 2-D | 直管／PI 软壁 | Torch | `softrobot.straight_tube.legacy_pi_2d` | 已覆盖 |
| `straight-tube-v2-3d` | 历史：v2 | 3-D | 直管／PI 软壁，准静态 | Torch | `softrobot.straight_tube.v2` | 已覆盖 |
| `straight-tube-v3` | 当前：v3 | 3-D | 直管／PI 节点接触 | Torch | `softrobot.straight_tube.v3` | 已覆盖 |
| `straight-tube-mesh-3d` | 历史：独立网格分支 | 3-D | 直管／三角网格接触 | Torch | `softrobot.straight_tube.quasi_static_mesh_3d` | 已覆盖 |
| `straight-tube-v4` | 当前：v4 | 3-D | 直管／解析管或网格接触 | Python＋C++ | `softrobot.straight_tube.v4` | 4 种配置 |

`straight-tube-v4` 的正常运行另支持续加载、预平衡、解析直管和 OBJ 网格；
4 种已执行的 CPU smoke 配置为 Python/C++ × 解析直管/直管网格。
它与历史 `straight-tube-mesh-3d` 是不同实现，后者仍保留独立入口。
参数见 `python -m softrobot.straight_tube.v4 --help`。

### 3.4 弯管任务（6）

| Model ID | 版本/分支 | 维度 | 任务与接触 | 后端 | 模块入口（`python -m`） | CPU smoke |
|---|---|---:|---|---|---|---|
| `curved-tube-2d-legacy-python` | 历史：2-D Python | 2-D | 圆弧管／PI 软壁 | Torch | `softrobot.curved_tube.legacy_arc_2d` | 已覆盖 |
| `curved-tube-2d-arc` | 当前：2-D 加速分支 | 2-D | 圆弧管／PI 软壁 | Python＋C++ | `softrobot.curved_tube.arc_2d` | 2 种后端 |
| `curved-tube-3d-legacy-analytic` | 历史：3-D 解析分支 | 3-D | 圆弧管／解析接触 | Torch | `softrobot.curved_tube.legacy_arc_3d_analytic` | 已覆盖 |
| `curved-tube-3d-legacy-mesh` | 历史：3-D 网格分支 | 3-D | 圆弧管／三角网格接触 | Torch | `softrobot.curved_tube.legacy_arc_3d_mesh` | 已覆盖 |
| `curved-tube-v4-analytic` | 当前：v4 解析配置 | 3-D | 曲线中轴管／解析接触 | Python＋C++ | `softrobot.curved_tube.v4_analytic` | 2 种后端 |
| `curved-tube-v4-mesh` | 当前：v4 网格配置 | 3-D | 曲管／三角网格接触 | Python＋C++ | `softrobot.curved_tube.v4_mesh` | 2 种后端 |

历史 3-D 圆弧管的解析/网格入口与当前 v4 曲管的解析/网格入口分别预置接触配置，
共用底层实现的配置不重复保存源码。`curved-tube-v4-mesh` 默认配对
`arc90_L1p5m_xz_wallR60mm_protectionR50mm_64sides_axial256.obj` 与
`arc90_L1p5m_xz_centerline_800.csv`；CSV 列为 `x,y,z`、`y=0`，曲线位于 xz 平面。
用户显式提供的同名长选项可覆盖预置资产或接触配置。

## 4. 旧路径迁移

以下 7 个根目录入口保留为薄兼容脚本，实际实现统一位于 `src/softrobot/`：

| 旧根路径 | 新模块 |
|---|---|
| `centerline_3d.py` | `softrobot.common.centerline_3d` |
| `der_cpp_backend.py` | `softrobot.backends.der_cpp_backend` |
| `mesh_contact_v4.py` | `softrobot.backends.mesh_contact_v4` |
| `quasi_static_mesh_3d.py` | `softrobot.straight_tube.quasi_static_mesh_3d` |
| `rod_soft_contact_PI_dynamic_quasi_static_2d_arc.py` | `softrobot.curved_tube.arc_2d` |
| `rod_soft_contact_PI_dynamic_quasi_static_3d_v3.py` | `softrobot.straight_tube.v3` |
| `rod_soft_contact_PI_dynamic_quasi_static_3d_v4.py` | `softrobot.straight_tube.v4` |

上述 7 个入口是老师版的薄兼容脚本。当前研究工作区另保留正弦插入和自由插入的
根目录研究入口，它们不属于老师版的 19 个 Model ID，也不在已交付 ZIP 中。
历史目录名只用于 `docs/model_lineage.md` 的来源核对，不作为老师版运行依赖。

## 5. 输出与测试

统一 CLI 的 `--output-dir` 指定本次运行的输出目录。smoke 会写入严格 JSON 结果并检查
有限状态、完成步数/收敛、真实接触证据（适用时）和非零状态增量；测试产物应放在
`/tmp` 或其他仓库外目录。

```bash
python -m pytest -q
python -m compileall -q src
softrobot smoke --all --output-dir /tmp/softrobot-all-smoke
```

smoke test 不是物理验证，也不能支持精度、稳定性、性能、硬件安全或临床有效性结论。
它只证明在所列 CPU 环境中，小参数端到端路径能结束并产生结构上有效的结果。正式结果
仍需独立的网格/步长收敛研究、基准对照和任务级验证。

## 6. 当前研究工作区保留的后续实验

合并后的工作分支另外保留正弦管插入入口
`rod_soft_contact_PI_dynamic_quasi_static_2d_sine_insert.py`、设计与证据说明
`docs/2D正弦三连续弯导丝插入与摩擦Benchmark设计.md`，以及直管自由插入入口
`rod_soft_contact_PI_dynamic_quasi_static_3d_straight_free_insert.py`。研究线的 C++
源码随共同后端放在 `src/softrobot/backends/cpp/`，根目录 `der_cpp_backend.py` 仍是
兼容导入；它们不增加老师版的模型 ID。自由插入的 360 mm 参数与可视化修改是备份
提交 `5ac40fb` 中的本地研究状态，不应当被解释为已经完成正式工况验证。

`wire_report/` 和 `血管导丝/` 是本机保留、Git 忽略的材料目录，不进入远端备份或
老师版 ZIP。
