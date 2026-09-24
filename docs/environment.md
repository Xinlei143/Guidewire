# CPU 环境与依赖基线

老师代码包的最低运行目标是 Linux、Python 3.10 和以 CPU device 运行的 PyTorch，另需 g++ 与 Ninja。
`requirements-cpu.lock` 记录本次实际验收的 Python distributions：解释器为 Python 3.10.20；
安装包元数据为 torch 2.11.0，而 `torch.__version__` 是 `2.11.0+cu130`，即 CUDA-enabled distribution。
本轮所有命令均显式走 CPU 路径，没有调用 CUDA 检查，也没有验收 CUDA 或单独验收 CPU-only wheel。
系统工具不属于 Python distributions，也不写入 lock：本次 `g++ --version` 首行是
`g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0`，`ninja --version` 输出是
`1.13.0.git.kitware.jobserver-pipe-1`。

推荐在全新虚拟环境安装。联网配置环境时，先从 PyTorch 官方 CPU wheel 索引安装锁定
版本，再安装其余锁定依赖和本项目：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
python -m pip install -r requirements-cpu.lock
python -m pip install --no-deps --no-build-isolation .
```

已有依赖的离线机器可使用 `python -m pip install --no-deps --no-build-isolation .`。C++ 后端第一次运行时
由 `torch.utils.cpp_extension` 调用 Ninja/g++ 编译，缓存写入临时目录，不进入源码包。
锁定文件是本次验收快照，不代表更早版本一定不兼容；`pyproject.toml` 只声明运行依赖
和 Python 下限，避免把单台机器的精确版本误作算法要求。
