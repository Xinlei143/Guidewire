"""CPU static displacement validation for the no-twist DER cantilever.

Run with `python tools/validate_cantilever_displacement.py --output-dir DIR`.
The EB reference assumes small transverse deflection and zero gravity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from softrobot.foundations.cantilever.der_no_twist_model import (
    RodParams, make_straight_rest_state, make_uniform_node_masses, total_elastic_energy,
)
from softrobot.foundations.cantilever.eb_beam_validation import (
    eb_tip_load_skip_reason, euler_bernoulli_point_load_positions,
)


def _transverse_relative_error(measured: np.ndarray, reference: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(reference[1:3]))
    if denominator == 0.0:
        return None
    numerator = float(np.linalg.norm(measured[1:3] - reference[1:3]))
    return numerator / denominator


def run_validation(
    *,
    num_links: int = 10,
    total_length: float = 1.0,
    EA: float = 1000.0,
    B: float = 0.5,
    force_node: int = -1,
    force_vector: tuple[float, float, float] = (0.0, 0.0, 0.005),
    output_dir: str | Path,
    residual_tolerance: float = 1e-5,
    max_iterations: int = 10000,
) -> dict:
    """Minimize the existing DER elastic energy minus point-load work.

    Nodes 0 and 1 remain fixed. The reported residual is the maximum absolute
    net-force component on free nodes, in newtons.
    """
    if num_links < 2 or total_length <= 0 or EA <= 0 or B <= 0:
        raise ValueError("Require >=2 links and positive length, EA, and B.")
    if not math.isfinite(residual_tolerance) or residual_tolerance <= 0:
        raise ValueError("Require a finite positive residual tolerance.")
    if max_iterations < 1:
        raise ValueError("Require positive max iterations.")
    if not all(math.isfinite(v) for v in (*force_vector, total_length, EA, B)):
        raise ValueError("All physical inputs must be finite.")
    num_nodes = num_links + 1
    if force_node < 0:
        force_node += num_nodes
    if force_node < 2 or force_node >= num_nodes:
        raise ValueError("force_node must be a free node 2..num_links.")

    torch.set_num_threads(1)
    rest = make_straight_rest_state(num_links, total_length, device="cpu")
    initial = rest.positions.detach()
    params = RodParams(
        EA=EA, B=B,
        node_masses=make_uniform_node_masses(num_nodes, 0.1, (0, 1), device="cpu"),
    )
    forces = torch.zeros_like(initial)
    forces[force_node] = torch.tensor(force_vector, dtype=torch.float64)

    eb_reason = eb_tip_load_skip_reason(
        force_node=force_node, num_nodes=num_nodes, force_vector=forces[force_node],
        B=B, gravity=0.0,
    )
    eb = None
    if eb_reason is None:
        eb = euler_bernoulli_point_load_positions(
            initial, forces[force_node], B, force_node=force_node,
        )
    trial = (eb if eb is not None else initial).clone()

    def objective(free_flat: np.ndarray) -> tuple[float, np.ndarray]:
        free = torch.tensor(free_flat.reshape(-1, 3), dtype=torch.float64, requires_grad=True)
        positions = torch.cat((initial[:2], free), dim=0)
        energy = total_elastic_energy(positions, rest, params)["elastic_energy"]
        potential = energy - torch.sum(forces * positions)
        (gradient,) = torch.autograd.grad(potential, free)
        return float(potential.detach()), gradient.detach().numpy().reshape(-1)

    result = minimize(
        objective, trial[2:].numpy().reshape(-1), jac=True, method="L-BFGS-B",
        options={"maxiter": max_iterations, "gtol": min(residual_tolerance * 0.01, 1e-10),
                 "ftol": 1e-15, "maxls": 50, "maxcor": 30},
    )
    _, gradient = objective(result.x)
    residual = float(np.max(np.abs(gradient)))
    if not result.success or not math.isfinite(residual) or residual > residual_tolerance:
        raise RuntimeError(
            f"Static DER equilibrium failed: success={result.success}, "
            f"residual={residual:.6g} N, message={result.message}"
        )
    final = np.vstack((initial[:2].numpy(), result.x.reshape(-1, 3)))
    displacement = final - initial.numpy()
    force_disp = displacement[force_node].tolist()
    tip_disp = displacement[-1].tolist()
    reference = None
    if eb is not None:
        eb_disp = (eb - initial).numpy()
        force_error = _transverse_relative_error(displacement[force_node], eb_disp[force_node])
        tip_error = _transverse_relative_error(displacement[-1], eb_disp[-1])
        reference = {
            "comparison_quantity": "transverse_yz_displacement_vector",
            "relative_error_status": ("defined" if force_error is not None and tip_error is not None
                                      else "undefined_zero_reference"),
            "force_node_displacement_m": eb_disp[force_node].tolist(),
            "tip_displacement_m": eb_disp[-1].tolist(),
            "force_node_relative_error": force_error,
            "tip_relative_error": tip_error,
        }

    report = {
        "model_id": "foundation-cantilever-der",
        "case": {
            "num_links": num_links, "total_length_m": total_length,
            "EA_n": EA, "B_nm2": B, "force_vector_n": list(force_vector),
            "gravity_m_s2": 0.0, "fixed_nodes": [0, 1],
        },
        "force_node": force_node,
        "force_node_displacement_m": force_disp,
        "tip_displacement_m": tip_disp,
        "solver": {
            "method": "L-BFGS-B", "success": True,
            "iterations": int(result.nit),
            "free_force_residual_inf_n": residual,
        },
        "eb_reference": reference,
        "eb_skip_reason": eb_reason,
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "cantilever_displacement.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-links", type=int, default=10)
    parser.add_argument("--total-length", type=float, default=1.0)
    parser.add_argument("--EA", type=float, default=1000.0)
    parser.add_argument("--bend-stiffness", type=float, default=0.5)
    parser.add_argument("--force-node", type=int, default=-1)
    parser.add_argument("--force-x", type=float, default=0.0)
    parser.add_argument("--force-y", type=float, default=0.0)
    parser.add_argument("--force-z", type=float, default=0.005)
    parser.add_argument("--residual-tolerance", type=float, default=1e-5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = run_validation(
        num_links=args.num_links, total_length=args.total_length,
        EA=args.EA, B=args.bend_stiffness, force_node=args.force_node,
        force_vector=(args.force_x, args.force_y, args.force_z),
        output_dir=args.output_dir, residual_tolerance=args.residual_tolerance,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
