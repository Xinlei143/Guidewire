"""One-step smoke driver shared by the historical planar PI models."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import numpy as np

from softrobot._smoke import configure_cpu_runtime, write_smoke_result


def run_contact_smoke(
    module: ModuleType,
    model_id: str,
    output_dir: str | Path,
    *,
    quasistatic: bool,
) -> None:
    """Evaluate one genuine implicit-Euler/Newton residual step on CPU."""
    configure_cpu_runtime()
    if quasistatic:
        state = module.initialize_rod(
            module.TOTAL_LENGTH, 6, module.DIAMETER, module.MATERIAL_DENSITY, 0.0
        )
    else:
        state = module.initialize_rod(
            module.TOTAL_LENGTH, 6, module.DIAMETER, module.MATERIAL_DENSITY
        )
    positions, velocities, rest_lengths, masses, integrals = state
    prescribed_mask, prescribed_values = module._make_prescribed_state(
        positions, False
    )
    keyword_arguments = dict(
        EA_value=module.EA,
        EI_value=module.EI,
        wall_half_gap=module.WALL_HALF_GAP,
        safe_distance=module.SAFE_DISTANCE,
        kp=module.KP,
        ki=module.KI,
        max_pi_force=module.MAX_PI_FORCE,
        integral_leak_rate=module.INTEGRAL_LEAK_RATE,
        max_newton_iters=2,
        residual_tol=1.0e-7,
        step_tol=1.0e-10,
    )
    if quasistatic:
        keyword_arguments["max_integral_state"] = module.MAX_INTEGRAL_STATE
    pulse_forces = np.zeros_like(positions)
    pulse_forces[-1, 1] = 1.0e-6
    result = module.implicit_euler_step(
        positions,
        velocities,
        rest_lengths,
        masses,
        pulse_forces,
        integrals,
        prescribed_mask,
        prescribed_values,
        module.DT,
        module.DAMPING,
        **keyword_arguments,
    )
    new_positions = result[0]
    info = result[-1]
    write_smoke_result(
        output_dir,
        model_id,
        new_positions,
        steps=1,
        newton_iterations=int(info.iterations),
        residual_norm=float(info.residual_norm),
    )
