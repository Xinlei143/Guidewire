# 无接触悬臂杆位移验证

现有模型 ID `foundation-cantilever-der` 是无接触、左端节点 0 和 1 固定的 3-D 无扭转 DER 杆。此页记录其指定节点点力下的静态位移检查；验证工具不注册为新的模型 ID，因此模型总数仍为 19。

## 方法与运行

`tools/validate_cantilever_displacement.py` 使用模型已有的 `total_elastic_energy`，减去受力节点处的外力做功，以 SciPy `L-BFGS-B` 求自由节点静态平衡。输出 JSON 包含受力节点和末端位移、优化迭代数，以及自由节点净力分量的最大绝对值（残差，单位 N）。残差必须有限、不超过 `--residual-tolerance` 且优化器成功，否则命令失败。此静态能量求解与模型正常运行时的动态 relaxation 路径不同。

Euler–Bernoulli（EB）参考采用左端节点 1 为夹持位置，横向点力在指定节点施加：载荷前位移形函数为 $s^2(3a-s)/(6B)$，载荷后为 $a^2(3s-a)/(6B)$；$a$ 为夹持点到受力节点距离。适用条件是直杆、小挠度、无重力、无轴向载荷。相对误差定义为 DER 与 EB 的末端横向 `yz` 位移向量之差的范数除以 EB 末端横向位移向量范数；参考为零时结果为 `null`。有轴向力的算例不计算 EB 误差。

已有依赖安装后，在仓库根目录运行；输出位置放在仓库外：

```bash
python tools/validate_cantilever_displacement.py --num-links 6 --force-node 4 --output-dir /tmp/cantilever-interior
python tools/validate_cantilever_displacement.py --num-links 10 --output-dir /tmp/cantilever-n10
python tools/validate_cantilever_displacement.py --num-links 30 --force-x 2 --force-z 2 --output-dir /tmp/cantilever-n30
```

默认值：长度 1 m、`EA=1000 N`、`B=0.5 N·m²`、无重力、横向末端力 `Fz=0.005 N`；负的 `--force-node -1` 表示末端。JSON 写至输出目录的 `cantilever_displacement.json`。

## 小载荷 EB 对照

下表为本分支 CPU 实跑：`F=(0,0,0.005) N`，力施于末端，长度/刚度为上述默认值，残差门槛 `1e-5 N`。`n` 是杆段数；误差只比较末端横向 `yz` 位移。

| n | DER 末端 z 位移 (m) | EB 末端 z 位移 (m) | 相对误差 | L-BFGS-B 迭代 | 残差 (N) |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.002546274507 | 0.001929012346 | 31.999% | 32 | 6.339e-9 |
| 10 | 0.002849973367 | 0.002430000000 | 17.283% | 60 | 2.476e-7 |
| 20 | 0.003087516405 | 0.002857916667 | 8.034% | 189 | 1.800e-6 |

在这三个网格中误差随杆段数增加而下降；这不是系统性的收敛阶或任意载荷精度证明。可逐项复现：

```bash
python tools/validate_cantilever_displacement.py --num-links 6 --output-dir /tmp/cantilever-n6
python tools/validate_cantilever_displacement.py --num-links 10 --output-dir /tmp/cantilever-n10
python tools/validate_cantilever_displacement.py --num-links 20 --output-dir /tmp/cantilever-n20
```

## 匹配参数的 Cosserat 对照

`validation/cosserat/CantileverRodStretchBendCompare.m` 是本地 Git 祖先 `eb33bf34658a472d393a2775fb35e3517f875cdd` 跟踪的匹配参数脚本，源文件 SHA-256 为 `19f2c1d7799df500ab79cdec131a88f6dac14d76ba47d4ba45303eaa1e5a36ff`。方法背景参见 [Till、Aloi、Rucker 的 Cosserat 杆论文](https://journals.sagepub.com/doi/10.1177/0278364919842269)。这里不归属该脚本的个人作者，也不收录疑似第三方原始示例 `untitled.m`。

可选 MATLAB 复现命令（需要 MATLAB Optimization Toolbox 的 `fsolve`，不属于 Python CPU 最低依赖）：

```bash
matlab -batch "set(0,'DefaultFigureVisible','off'); addpath('validation/cosserat'); CantileverRodStretchBendCompare"
```

两者采用 `n=30`、`L=1 m`、`EA=1000 N`、`B=0.5 N·m²`、无重力、末端力 `(Fx,Fy,Fz)=(2,0,2) N`。本轮无头实跑结果：

| 求解 | 末端 ux (m) | 末端 uz (m) | 说明 |
|---|---:|---:|---|
| DER 静态 | -0.111322857927 | 0.425090719333 | L-BFGS-B 530 次，残差 3.749e-6 N |
| MATLAB Cosserat | -0.115692363 | 0.437051073 | MATLAB `fsolve` 输出 |

按 MATLAB 分量绝对值为分母，`ux` 差 3.777%，`uz` 差 2.737%；`(ux,uz)` 向量差相对于 MATLAB 向量范数为 2.817%。Cosserat 脚本包含高刚度剪切/扭转惩罚，而 DER 模型不包含剪切/扭转自由度；边界离散和静态求解形式也不同。这是参数匹配的单算例数值对照，不表示两种杆理论等价，也不足以宣称模型已全面验证。
