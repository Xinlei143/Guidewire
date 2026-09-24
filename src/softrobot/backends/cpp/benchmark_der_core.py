#!/usr/bin/env python3
"""Benchmark the standalone v4 Python and C++ implicit-Euler backends.

The harness reports wall time per step, simulated-dt ratio, Newton and
line-search counters, dense-rescue counters, and the C++ assembly/solve
profile. ``--snapshot`` accepts an NPZ containing a v4 state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from softrobot.straight_tube import v4
from softrobot.backends.der_cpp_backend import DERCore


def _case_state(case: str, nodes: int, length: float) -> dict[str, np.ndarray]:
    rest = np.full(nodes - 1, length / (nodes - 1), dtype=np.float64)
    masses = np.full(nodes, 0.01, dtype=np.float64)
    positions = np.zeros((nodes, 3), dtype=np.float64)
    positions[:, 0] = np.linspace(0.0, length, nodes)
    if case == "bent":
        phase = np.linspace(0.0, 1.8 * np.pi, nodes)
        positions[:, 1] = 0.008 * np.sin(phase)
        positions[:, 2] = 0.005 * np.cos(0.7 * phase)
    elif case == "contact":
        phase = np.linspace(0.0, 2.0 * np.pi, nodes)
        radius = v4.TUBE_RADIUS - 0.5 * v4.SAFE_DISTANCE
        positions[:, 1] = radius * np.cos(phase)
        positions[:, 2] = radius * np.sin(phase)
    elif case != "straight":
        raise ValueError(f"unknown case: {case}")
    zeros = np.zeros_like(positions)
    return {
        "positions": positions,
        "velocities": zeros.copy(),
        "rest_lengths": rest,
        "masses": masses,
        "integrals": np.zeros(nodes, dtype=np.float64),
        "initial_positions": positions.copy(),
        "pulse_forces": zeros,
    }


def _load_snapshot(path: str) -> dict[str, np.ndarray]:
    data = np.load(path)
    missing = [key for key in ("positions",) if key not in data]
    if missing:
        raise ValueError(f"snapshot is missing: {', '.join(missing)}")
    positions = np.asarray(data["positions"], dtype=np.float64)
    if positions.ndim == 3:
        positions = positions[-1]
    zeros = np.zeros_like(positions)
    velocities = data["velocities"] if "velocities" in data else zeros
    if np.asarray(velocities).ndim == 3:
        velocities = np.asarray(velocities)[-1]
    integrals = data["integrals"] if "integrals" in data else data["pi_integrals"] if "pi_integrals" in data else np.zeros(len(positions))
    if np.asarray(integrals).ndim == 2:
        integrals = np.asarray(integrals)[-1]
    reference_positions = np.asarray(
        data["initial_positions"] if "initial_positions" in data else positions,
        dtype=np.float64,
    )
    if reference_positions.ndim == 3:
        reference_positions = reference_positions[-1]
    rest_lengths = (
        np.asarray(data["rest_lengths"], dtype=np.float64)
        if "rest_lengths" in data
        else np.linalg.norm(np.diff(reference_positions, axis=0), axis=1)
    )
    masses = (
        np.asarray(data["masses"], dtype=np.float64)
        if "masses" in data
        else np.full(len(positions), 0.01, dtype=np.float64)
    )
    return {
        "positions": np.ascontiguousarray(positions),
        "velocities": np.ascontiguousarray(velocities, dtype=np.float64),
        "rest_lengths": np.ascontiguousarray(rest_lengths),
        "masses": np.ascontiguousarray(masses),
        "integrals": np.ascontiguousarray(integrals, dtype=np.float64),
        "initial_positions": np.ascontiguousarray(reference_positions, dtype=np.float64),
        "pulse_forces": np.ascontiguousarray(data["pulse_forces"] if "pulse_forces" in data else zeros, dtype=np.float64),
    }


def _args(dt: float, newton_iters: int, length: float = 1.0) -> argparse.Namespace:
    _, ea, ei, damping = v4.rod_properties(length, v4.DIAMETER, v4.MATERIAL_DENSITY)
    return argparse.Namespace(
        EA=ea, EI=ei, damping=damping, dt_active=dt,
        tube_radius=v4.TUBE_RADIUS, safe_distance=v4.SAFE_DISTANCE,
        Kp=v4.KP, Ki=v4.KI, ki_active=v4.KI,
        max_pi_force=v4.MAX_PI_FORCE,
        integral_leak_rate=v4.INTEGRAL_LEAK_RATE,
        max_integral_state=v4.MAX_INTEGRAL_STATE,
        max_newton_iters=newton_iters, residual_tol=1e-7, step_tol=1e-10,
        performance_profile={}, backend="python",
    )


def _cpp_runner(state: dict[str, np.ndarray], dt: float, newton_iters: int):
    params = _args(dt, newton_iters, float(state["rest_lengths"].sum()))
    core = DERCore(
        state["rest_lengths"], state["masses"], params.EA, params.EI,
        params.damping, v4.TUBE_RADIUS, v4.SAFE_DISTANCE, v4.KP, v4.KI,
        v4.MAX_PI_FORCE, v4.INTEGRAL_LEAK_RATE, v4.MAX_INTEGRAL_STATE,
        v4.CONTACT_SMOOTH_LENGTH, v4.DIRECTION_EPSILON,
    )

    def run():
        return core.implicit_euler_step_cached(
            state["positions"], state["velocities"], state["integrals"], state["initial_positions"],
            state["pulse_forces"], False, 0.0, dt, newton_iters, 1e-7, 1e-10,
            v4.KI,
        )
    return run


def _python_runner(state: dict[str, np.ndarray], dt: float, newton_iters: int):
    args = _args(dt, newton_iters, float(state["rest_lengths"].sum()))
    tensors = {key: torch.as_tensor(value, dtype=v4.DTYPE) for key, value in state.items()}

    def run():
        return v4._implicit_euler_step_python(
            tensors["positions"], tensors["velocities"], tensors["rest_lengths"],
            tensors["masses"], tensors["integrals"], tensors["initial_positions"],
            tensors["pulse_forces"], False, 0.0, dt, args,
        )
    return run


def _measure(name: str, run, warmup: int, repeat: int, dt: float) -> dict:
    for _ in range(warmup):
        run()
    samples = []
    result = None
    for _ in range(repeat):
        started = time.perf_counter()
        result = run()
        samples.append(time.perf_counter() - started)
    values = np.asarray(samples, dtype=np.float64)
    if isinstance(result, dict):
        info = {
            "converged": bool(result["converged"]),
            "iterations": int(result["iterations"]),
            "residual_norm": float(result["residual_norm"]),
            "line_search_failures": int(result["line_search_failures"]),
        }
        for key in ("full_assembly_count", "residual_only_count", "line_search_evaluation_count",
                    "dense_build_count", "dense_fallback_count", "banded_solve_count"):
            if key in result:
                info[key] = int(result[key])
        for key in ("assembly_seconds", "solve_seconds", "line_search_seconds", "wall_seconds"):
            if key in result:
                info[key] = float(result[key])
    else:
        info_obj = result[8]
        info = {
            "converged": bool(info_obj.converged),
            "iterations": int(info_obj.iterations),
            "residual_norm": float(info_obj.residual_norm),
            "line_search_failures": int(info_obj.line_search_failures),
        }
    info.update({
        "backend": name,
        "wall_mean_s": float(values.mean()),
        "wall_median_s": float(np.median(values)),
        "wall_p95_s": float(np.percentile(values, 95)),
        "simulated_dt_s": dt,
        "realtime_ratio": float(values.mean() / dt),
    })
    return info


def _numeric_fields(result) -> dict[str, np.ndarray]:
    if isinstance(result, dict):
        return {
            "positions": np.asarray(result["positions"]),
            "velocities": np.asarray(result["velocities"]),
            "integrals": np.asarray(result["integrals"]),
            "contact_force": np.asarray(result["contact_force"]),
        }
    return {
        "positions": result[0].detach().cpu().numpy(),
        "velocities": result[1].detach().cpu().numpy(),
        "integrals": result[2].detach().cpu().numpy(),
        "contact_force": result[3].detach().cpu().numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("straight", "bent", "contact", "npz"), default="straight")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--nodes", type=int, default=250)
    parser.add_argument("--length", type=float, default=1.5)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--newton-iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--backend", choices=("cpp", "python", "both"), default="both")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.case == "npz":
        if args.snapshot is None:
            parser.error("--snapshot is required for --case npz")
        state = _load_snapshot(str(args.snapshot))
    else:
        if args.nodes < 4 or args.length <= 0.0:
            parser.error("nodes must be >= 4 and length must be positive")
        state = _case_state(args.case, args.nodes, args.length)
    if args.dt <= 0.0 or args.warmup < 0 or args.repeat <= 0:
        parser.error("dt > 0, warmup >= 0 and repeat > 0 are required")
    results = []
    cpp_run = python_run = None
    if args.backend in ("cpp", "both"):
        cpp_run = _cpp_runner(state, args.dt, args.newton_iters)
        results.append(_measure("cpp", cpp_run, args.warmup, args.repeat, args.dt))
    if args.backend in ("python", "both"):
        python_run = _python_runner(state, args.dt, args.newton_iters)
        results.append(_measure("python", python_run, args.warmup, args.repeat, args.dt))
    payload = {"case": args.case, "nodes": len(state["positions"]), "results": results}
    if cpp_run is not None and python_run is not None:
        cpp_fields = _numeric_fields(cpp_run())
        python_fields = _numeric_fields(python_run())
        payload["cpp_python_max_abs_difference"] = {
            key: float(np.max(np.abs(cpp_fields[key] - python_fields[key])))
            for key in cpp_fields
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json:
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
