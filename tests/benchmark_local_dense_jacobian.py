"""Reproducible correctness and timing benchmark for Newton Jacobian modes."""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

import rod_soft_contact_PI_dynamic_quasi_static as rod


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_contact_state(node_count: int, device: torch.device):
    positions, velocities, rest, masses, integrals = rod.initialize_rod(num_nodes=node_count)
    old_positions = torch.as_tensor(positions, dtype=rod.DTYPE, device=device)
    old_positions = old_positions.clone()
    old_positions[:, 1] = 0.056 * torch.sin(
        torch.linspace(0.0, math.pi, node_count, dtype=rod.DTYPE, device=device)
    )
    old_velocities = torch.as_tensor(velocities, dtype=rod.DTYPE, device=device)
    rest_lengths = torch.as_tensor(rest, dtype=rod.DTYPE, device=device)
    node_masses = torch.as_tensor(masses, dtype=rod.DTYPE, device=device)
    old_integrals = torch.linspace(
        0.0, 0.02, node_count, dtype=rod.DTYPE, device=device
    )
    prescribed_mask, prescribed_values = rod._make_prescribed_state_device(
        old_positions, True, 2.0e-5
    )
    free_mask = ~prescribed_mask
    q_old = rod._pack_dofs(old_positions, free_mask)
    q = q_old.clone()
    q[1::2] += 1.0e-7
    v_old = rod._pack_dofs(old_velocities, free_mask)
    free_node_masses = node_masses[:, None].expand_as(old_positions)[free_mask]
    previous_z = torch.abs(old_positions[:, 1]) - (rod.WALL_HALF_GAP - rod.SAFE_DISTANCE)
    previous_errors = F.softplus(previous_z, beta=1.0 / rod.CONTACT_SMOOTH_LENGTH)
    residual_args = (
        old_positions, rest_lengths, torch.zeros_like(old_positions), old_integrals,
        free_mask, prescribed_values, q_old, v_old, free_node_masses, previous_errors,
        rod.DT, rod.DAMPING, rod.EA, rod.EI, rod.WALL_HALF_GAP, rod.SAFE_DISTANCE,
        rod.KP, rod.KI, rod.MAX_PI_FORCE, rod.INTEGRAL_LEAK_RATE,
        rod.MAX_INTEGRAL_STATE,
    )
    return q, residual_args


def timed_call(function, device: torch.device):
    synchronize(device)
    started = time.perf_counter()
    output = function()
    synchronize(device)
    return output, time.perf_counter() - started


def median_timing(function, device: torch.device, repeats: int) -> float:
    timed_call(function, device)  # unmeasured warm-up
    samples = [timed_call(function, device)[1] for _ in range(repeats)]
    return float(torch.tensor(samples).median().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--nodes", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    q, residual_args = make_contact_state(args.nodes, device)
    full = lambda: rod._residual_and_jacobian_full_autograd(q, residual_args)
    local = lambda: rod._residual_and_jacobian_local_dense(q, residual_args)
    batched = lambda: rod._residual_and_jacobian_local_batched(q, residual_args)
    (residual_full, jacobian_full), _ = timed_call(full, device)
    (residual_local, jacobian_local), _ = timed_call(local, device)
    (residual_batched, jacobian_batched), _ = timed_call(batched, device)
    _, full_build_seconds = timed_call(full, device)
    _, local_build_seconds = timed_call(local, device)
    full_build_median = median_timing(full, device, args.repeats)
    local_build_median = median_timing(local, device, args.repeats)
    batched_build_median = median_timing(batched, device, args.repeats)
    _, solve_full_seconds = timed_call(
        lambda: torch.linalg.solve(jacobian_full, -residual_full), device
    )
    _, solve_local_seconds = timed_call(
        lambda: torch.linalg.solve(jacobian_local, -residual_local), device
    )
    delta_full = torch.linalg.solve(jacobian_full, -residual_full)
    delta_local = torch.linalg.solve(jacobian_local, -residual_local)
    delta_batched = torch.linalg.solve(jacobian_batched, -residual_batched)
    result = {
        "device": str(device),
        "node_count": args.nodes,
        "free_dofs": int(q.numel()),
        "residual_max_abs_error": float(torch.max(torch.abs(residual_local - residual_full)).item()),
        "jacobian_max_abs_error": float(torch.max(torch.abs(jacobian_local - jacobian_full)).item()),
        "jacobian_relative_fro_error": float(
            (torch.linalg.norm(jacobian_local - jacobian_full)
             / torch.linalg.norm(jacobian_full).clamp_min(1.0e-30)).item()
        ),
        "newton_delta_relative_error": float(
            (torch.linalg.norm(delta_local - delta_full)
             / torch.linalg.norm(delta_full).clamp_min(1.0e-30)).item()
        ),
        "batched_residual_max_abs_error": float(
            torch.max(torch.abs(residual_batched - residual_full)).item()
        ),
        "batched_jacobian_max_abs_error": float(
            torch.max(torch.abs(jacobian_batched - jacobian_full)).item()
        ),
        "batched_jacobian_relative_fro_error": float(
            (torch.linalg.norm(jacobian_batched - jacobian_full)
             / torch.linalg.norm(jacobian_full).clamp_min(1.0e-30)).item()
        ),
        "batched_newton_delta_relative_error": float(
            (torch.linalg.norm(delta_batched - delta_full)
             / torch.linalg.norm(delta_full).clamp_min(1.0e-30)).item()
        ),
        "full_autograd_build_seconds_median": full_build_median,
        "local_dense_build_seconds_median": local_build_median,
        "local_batched_build_seconds_median": batched_build_median,
        "full_autograd_build_seconds_sample": full_build_seconds,
        "local_dense_build_seconds_sample": local_build_seconds,
        "linear_solve_full_seconds": solve_full_seconds,
        "linear_solve_local_seconds": solve_local_seconds,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
