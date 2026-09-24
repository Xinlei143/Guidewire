"""A/B benchmark for the first v3 mathematical-equivalence optimizations."""

from __future__ import annotations

import argparse
import json
import math
import time
import types

import torch

from softrobot.straight_tube import v3 as rod


def timed(function, device: torch.device, repeats: int) -> float:
    rod.sync_device(device)
    function()
    rod.sync_device(device)
    samples = []
    for _ in range(repeats):
        rod.sync_device(device)
        started = time.perf_counter()
        function()
        rod.sync_device(device)
        samples.append(time.perf_counter() - started)
    samples.sort()
    return samples[len(samples) // 2]


def make_state(nodes: int, device: torch.device):
    dtype = rod.DTYPE
    positions = torch.zeros((nodes, 3), dtype=dtype, device=device)
    positions[:, 0] = torch.linspace(0.0, 0.7, nodes, dtype=dtype, device=device)
    phase = torch.linspace(0.0, 2.0 * math.pi, nodes, dtype=dtype, device=device)
    positions[:, 1] = 0.052 * torch.sin(phase)
    positions[:, 2] = 0.052 * torch.cos(phase)
    velocities = torch.zeros_like(positions)
    rest = torch.full((nodes - 1,), 0.7 / (nodes - 1), dtype=dtype, device=device)
    masses = torch.full((nodes,), 0.01, dtype=dtype, device=device)
    integrals = torch.linspace(0.0, 0.02, nodes, dtype=dtype, device=device)
    previous_errors = torch.zeros(nodes, dtype=dtype, device=device)
    pulse = torch.zeros_like(positions)
    prescribed_mask = torch.zeros_like(positions, dtype=torch.bool)
    prescribed_mask[:2] = True
    prescribed = positions.clone()
    q = positions[~prescribed_mask].clone()
    block_plan = rod.get_block_assembly_plan(str(device), nodes, False)
    _, ea, ei, damping = rod.rod_properties(0.7, rod.DIAMETER, rod.MATERIAL_DENSITY)
    common = dict(
        dt_active=0.1,
        tube_radius=rod.TUBE_RADIUS,
        safe_distance=rod.SAFE_DISTANCE,
        Kp=rod.KP,
        ki_active=rod.KI,
        max_pi_force=rod.MAX_PI_FORCE,
        integral_leak_rate=rod.INTEGRAL_LEAK_RATE,
        max_integral_state=rod.MAX_INTEGRAL_STATE,
        EA=ea,
        EI=ei,
        damping=damping,
        performance_profile={},
    )
    return (
        positions, velocities, rest, masses, integrals, previous_errors, pulse,
        prescribed_mask, prescribed, q, block_plan, types.SimpleNamespace(**common),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--nodes", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    (
        positions, velocities, rest, masses, integrals, previous_errors, pulse,
        prescribed_mask, prescribed, q, block_plan, common,
    ) = make_state(args.nodes, device)
    pattern = rod.get_assembly_pattern(torch.zeros_like(positions, dtype=torch.bool))

    reference_contact = lambda: rod.assemble_contact_batched(
        positions, integrals, previous_errors, pattern, common
    )
    optimized_contact = lambda: rod.assemble_contact_analytic(
        positions, integrals, previous_errors, pattern, common
    )
    reference_args = types.SimpleNamespace(**vars(common), compute_kernels="reference")
    optimized_args = types.SimpleNamespace(**vars(common), compute_kernels="optimized")

    def reference_residual():
        return rod.residual_and_jacobian(
            q, positions, velocities, rest, masses, integrals, previous_errors, pulse,
            prescribed_mask, prescribed, reference_args, False, block_plan,
        )

    def optimized_residual():
        return rod.residual_and_jacobian(
            q, positions, velocities, rest, masses, integrals, previous_errors, pulse,
            prescribed_mask, prescribed, optimized_args, False, block_plan,
        )

    def reference_jacobian():
        return rod.residual_and_jacobian(
            q, positions, velocities, rest, masses, integrals, previous_errors, pulse,
            prescribed_mask, prescribed, reference_args, True, block_plan,
        )

    def optimized_jacobian():
        return rod.residual_and_jacobian(
            q, positions, velocities, rest, masses, integrals, previous_errors, pulse,
            prescribed_mask, prescribed, optimized_args, True, block_plan,
        )

    force_ref, jac_ref = reference_contact()
    force_opt, jac_opt = optimized_contact()
    residual_ref, _, _ = reference_residual()
    residual_opt, _, _ = optimized_residual()
    _, system_ref, _ = reference_jacobian()
    _, system_opt, _ = optimized_jacobian()
    probe = torch.linspace(-1.0, 1.0, q.numel(), dtype=q.dtype, device=device)
    result = {
        "device": str(device),
        "nodes": args.nodes,
        "contact_reference_ms": 1e3 * timed(reference_contact, device, args.repeats),
        "contact_optimized_ms": 1e3 * timed(optimized_contact, device, args.repeats),
        "residual_reference_ms": 1e3 * timed(reference_residual, device, args.repeats),
        "residual_optimized_ms": 1e3 * timed(optimized_residual, device, args.repeats),
        "jacobian_reference_ms": 1e3 * timed(reference_jacobian, device, args.repeats),
        "jacobian_optimized_ms": 1e3 * timed(optimized_jacobian, device, args.repeats),
        "contact_force_max_abs_error": float(torch.max(torch.abs(force_opt - force_ref)).item()),
        "contact_jacobian_max_abs_error": float(torch.max(torch.abs(jac_opt - jac_ref)).item()),
        "residual_max_abs_error": float(torch.max(torch.abs(residual_opt - residual_ref)).item()),
        "jacobian_matvec_max_abs_error": float(
            torch.max(torch.abs(system_opt.matvec(probe) - system_ref.matvec(probe))).item()
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
