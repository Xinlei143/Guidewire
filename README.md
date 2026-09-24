# SoftRobot 导丝仿真代码包

这是面向课程/研究复核的 CPU 代码包。代码按物理任务组织，而不是按文件产生时间堆放：

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

报告、原始实验数据、历史输出、媒体、缓存、第三方源码、正弦插入和自由插入研究线不在
本代码包中。完整来源和内容哈希见 `docs/model_lineage.md`。

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

# 单模型小参数 CPU 验收；输出只写到指定目录
softrobot run straight-tube-v4 --smoke-test --output-dir /tmp/straight-v4-smoke

# 对全部 19 个模型执行 smoke
softrobot smoke --all --output-dir /tmp/softrobot-smoke
```

每个模型也可按模块独立运行，例如：

```bash
python -m softrobot.straight_tube.v4 --smoke-test --output-dir /tmp/v4
python -m softrobot.curved_tube.arc_2d --smoke-test --output-dir /tmp/arc2d
python -m softrobot.curved_tube.legacy_arc_3d_analytic --smoke-test --output-dir /tmp/legacy-arc-analytic
python -m softrobot.curved_tube.legacy_arc_3d_mesh --smoke-test --output-dir /tmp/legacy-arc-mesh
python -m softrobot.curved_tube.v4_analytic --smoke-test --output-dir /tmp/v4-arc-analytic
python -m softrobot.curved_tube.v4_mesh --smoke-test --output-dir /tmp/v4-arc-mesh
python -m softrobot.foundations.implicit_newton_3d --smoke-test --output-dir /tmp/newton3d
```

registry 中 19 个 Model ID 各自对应唯一模块入口。曲管 analytic/mesh 模块各自预置正确
接触类型和规范资产；用户显式提供的同名长选项会覆盖预置。`softrobot.curved_tube.v4_mesh`
默认配对 `arc90_L1p5m_xz_wallR60mm_protectionR50mm_64sides_axial256.obj` 与
`arc90_L1p5m_xz_centerline_800.csv`；CSV 列为 `x,y,z`、`y=0`，曲线位于 xz 平面。不带 `--smoke-test` 时保留历史模型原有默认参数和算例语义。smoke 模式才会缩小节点
数、步数和输出规模，并固定为 CPU、Matplotlib Agg 无头后端和确定性随机种子。

## 3. 模型状态与 smoke 覆盖

| Model ID | 状态 | 维度 | 几何/接触 | 后端 | smoke |
|---|---|---:|---|---|---|
| `foundation-2d-stretch-numpy` | historical | 2-D | 无接触 | NumPy CPU | yes |
| `foundation-2d-stretch-bend-torch` | historical | 2-D | 无接触 | Torch CPU | yes |
| `foundation-3d-stretch-bend-torch` | historical | 3-D | 无接触 | Torch CPU | yes |
| `foundation-3d-implicit-newton` | historical | 3-D | 无接触 | Torch CPU | yes |
| `foundation-cantilever-der` | historical final | 3-D | cantilever | Torch CPU | yes |
| `contact-2d-dynamic-pi` | historical | 2-D | parallel walls / PI | Torch CPU | yes |
| `contact-2d-quasistatic-pi-v1` | historical | 2-D | parallel walls / PI | Torch CPU | yes |
| `contact-2d-quasistatic-pi-v2` | historical | 2-D | parallel walls / PI | Torch CPU | yes |
| `straight-tube-2d-pi` | historical | 2-D | straight / PI | Torch CPU | yes |
| `straight-tube-v2-3d` | historical v2 | 3-D | straight / PI | Torch CPU | yes |
| `straight-tube-v3` | current v3 | 3-D | straight / PI | Torch CPU | yes |
| `straight-tube-mesh-3d` | historical mesh | 3-D | straight / triangle mesh | Torch CPU | yes |
| `straight-tube-v4` | current | 3-D | straight analytic + mesh | Python + C++ CPU | 4 variants |
| `curved-tube-2d-legacy-python` | historical | 2-D | circular arc / PI | Torch CPU | yes |
| `curved-tube-2d-arc` | current | 2-D | circular arc / PI | Python + C++ CPU | 2 variants |
| `curved-tube-3d-legacy-analytic` | historical | 3-D | circular arc / analytic | Torch CPU | yes |
| `curved-tube-3d-legacy-mesh` | historical | 3-D | circular arc / mesh | Torch CPU | yes |
| `curved-tube-v4-analytic` | current preset | 3-D | curved centerline / analytic | Python + C++ CPU | 2 variants |
| `curved-tube-v4-mesh` | current preset | 3-D | curved / triangle mesh | Python + C++ CPU | 2 variants |

当前 v4 正常运行还支持续加载、预平衡、解析直管、解析曲管、OBJ 网格、Python 参考路径
和 C++ CPU 路径；使用 `python -m softrobot.straight_tube.v4 --help` 查看全部参数。

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

本分支不提供正弦插入兼容入口，也不提供自由插入兼容入口；这两条研究线晚于本次老师版
基线。历史目录名只用于 `docs/model_lineage.md` 的来源核对，不作为运行依赖。

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

## 6. 可复现老师版 ZIP

发布构建器只读取指定 Git commit，不读取未跟踪文件；tracked 工作树不干净时会拒绝
构建。归档按固定排序、时间戳和权限写入，含覆盖其他文件的 `MANIFEST.sha256`，并在
ZIP 外生成整体 `.sha256`：

```bash
python tools/build_teacher_release.py \
  --output ../SoftRobot_teacher_code_20260924.zip \
  --release-date 20260924
```

构建器会排除报告、portfolio exporter、benchmark、历史输出、缓存、编译对象、媒体、
原始数据、第三方源码、重复仓库和不在交付基线内的研究线。
