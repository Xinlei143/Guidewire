"""One-step smoke driver shared by the historical planar PI models."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import numpy as np

from .._smoke import cpu_smoke_runtime, write_smoke_result


@cpu_smoke_runtime()
def run_contact_smoke(
    module: ModuleType,
    model_id: str,
    output_dir: str | Path,
    *,
    quasistatic: bool,
) -> None:
    """Evaluate one genuine implicit-Euler/Newton residual step on CPU."""
    result_path = Path(output_dir) / "smoke_result.json"
    result_path.unlink(missing_ok=True)
    if quasistatic:
        state = module.initialize_rod(
            module.TOTAL_LENGTH, 6, module.DIAMETER, module.MATERIAL_DENSITY, 0.0
        )
    else:
        state = module.initialize_rod(
            module.TOTAL_LENGTH, 6, module.DIAMETER, module.MATERIAL_DENSITY
        )
    positions, velocities, rest_lengths, masses, integrals = state
    contact_boundary = module.WALL_HALF_GAP - module.SAFE_DISTANCE
    positions[:, 1] = contact_boundary + 1.0e-6
    initial_positions = positions.copy()
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
        max_newton_iters=50,
        residual_tol=1.0e-7,
        step_tol=1.0e-10,
    )
    if quasistatic:
        keyword_arguments["max_integral_state"] = module.MAX_INTEGRAL_STATE
    pulse_forces = np.zeros_like(positions)
    pulse_forces[-1, 1] = -1.0e-6
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
    new_integrals = np.asarray(result[2])
    contact_force = np.asarray(result[3])
    active = np.asarray(result[4])
    info = result[-1]
    converged = bool(info.converged)
    if not converged:
        raise RuntimeError(
            f"{model_id} implicit smoke step did not converge: "
            f"residual={info.residual_norm:.6e}"
        )
    state_delta_norm = float(np.linalg.norm(new_positions - initial_positions))
    contact_force_norm = float(np.linalg.norm(contact_force))
    active_count = int(np.count_nonzero(active))
    integral_state_norm = float(np.linalg.norm(new_integrals))
    integral_term_norm = float(abs(module.KI) * integral_state_norm)
    residual = float(info.residual_norm)
    iterations = int(info.iterations)
    numeric_evidence = (
        np.asarray(initial_positions), np.asarray(new_positions), contact_force,
        new_integrals, np.asarray((state_delta_norm, contact_force_norm,
                                  integral_state_norm, integral_term_norm, residual)),
    )
    if not all(np.isfinite(value).all() for value in numeric_evidence):
        raise RuntimeError(f"{model_id} contact smoke produced non-finite evidence")
    if residual < 0.0:
        raise RuntimeError(f"{model_id} contact smoke reported a negative residual")
    if iterations < 1:
        raise RuntimeError(f"{model_id} contact smoke reported no Newton iteration")
    if state_delta_norm <= 0.0:
        raise RuntimeError(f"{model_id} contact smoke state did not move")
    if active_count <= 0 or contact_force_norm <= 0.0:
        raise RuntimeError(f"{model_id} contact smoke did not exercise contact")
    if integral_state_norm <= 0.0:
        raise RuntimeError(f"{model_id} contact smoke did not update PI state")
    if module.KI > 0.0 and integral_term_norm <= 0.0:
        raise RuntimeError(f"{model_id} contact smoke did not exercise integral force")
    variant = {
        "name": "python+pi-contact",
        "backend": "python",
        "finite": True,
        "converged": converged,
        "iterations": iterations,
        "residual": residual,
        "state_delta_norm": state_delta_norm,
        "contact_force_norm": contact_force_norm,
        "contact_count": active_count,
        "ki": float(module.KI),
        "integral_state_norm": integral_state_norm,
        "integral_term_norm": integral_term_norm,
    }
    write_smoke_result(
        output_dir,
        model_id,
        new_positions,
        steps=1,
        converged=converged,
        newton_iterations=iterations,
        residual=residual,
        residual_norm=residual,
        active_count=active_count,
        contact_force_norm=contact_force_norm,
        state_delta_norm=state_delta_norm,
        ki=float(module.KI),
        integral_state_norm=integral_state_norm,
        integral_term_norm=integral_term_norm,
        variants=[variant],
    )
