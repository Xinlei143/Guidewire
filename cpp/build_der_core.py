#!/usr/bin/env python3
"""Build and smoke-test the cached DER C++ extension."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from der_cpp_backend import DERCore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=8)
    args = parser.parse_args()
    if args.nodes < 4:
        parser.error("--nodes must be at least 4")
    rest = np.full(args.nodes - 1, 0.1, dtype=np.float64)
    masses = np.full(args.nodes, 0.01, dtype=np.float64)
    core = DERCore(rest, masses, 10.0, 2.0, 0.1, 0.05, 0.001,
                   1.0, 1.0, 1.0, 0.1, 1.0)
    positions = np.zeros((args.nodes, 3), dtype=np.float64)
    positions[:, 0] = np.arange(args.nodes, dtype=np.float64) * 0.1
    gradient, hessian = core.stretch_kernel(positions, rest)
    zeros = np.zeros_like(positions)
    result = core.implicit_euler_step(
        positions, zeros, rest, masses, np.zeros(args.nodes, dtype=np.float64),
        positions, zeros, False, 0.0, 0.1, 8, 1e-7, 1e-10, 1.0,
    )
    print(f"DER C++ extension ready: nodes={args.nodes}, "
          f"stretch_gradient_shape={gradient.shape}, "
          f"stretch_hessian_shape={hessian.shape}, "
          f"implicit_converged={bool(result['converged'])}, "
          f"full_assemblies={int(result['full_assembly_count'])}, "
          f"residual_only={int(result['residual_only_count'])}")


if __name__ == "__main__":
    main()
