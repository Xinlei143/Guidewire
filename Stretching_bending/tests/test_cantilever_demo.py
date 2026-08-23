"""Smoke checks for cantilever-style dynamic relaxation setup."""

from __future__ import annotations

import unittest

import torch

from der_no_twist_model import RodParams, make_straight_rest_state, make_uniform_node_masses
from run_cantilever_demo import resolve_device
from semi_implicit_solver import (
    BoundaryCondition,
    ExternalLoads,
    SolverOptions,
    build_external_forces,
    run_dynamic_relaxation,
)


class TestCantileverDemo(unittest.TestCase):
    def test_resolve_device_can_force_cpu(self) -> None:
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    def test_solver_progress_is_disabled_by_default(self) -> None:
        options = SolverOptions(dt=2e-4, steps=2, damping=0.5, convergence_tol=1e-5)
        self.assertFalse(options.show_progress)

    def test_short_b_zero_relaxation_runs(self) -> None:
        fixed_indices = (0, 1)
        rest = make_straight_rest_state(num_links=5, total_length=1.0)
        node_masses = make_uniform_node_masses(
            num_nodes=rest.positions.shape[0],
            value=0.1,
            fixed_indices=fixed_indices,
        )
        external_forces = build_external_forces(
            num_nodes=rest.positions.shape[0],
            force_node=-1,
            force_vector=torch.tensor([0.0, 0.0, -0.01], dtype=torch.float64),
            fixed_indices=fixed_indices,
        )

        result = run_dynamic_relaxation(
            initial_positions=rest.positions.clone(),
            rest=rest,
            params=RodParams(EA=1000.0, B=0.0, node_masses=node_masses),
            bc=BoundaryCondition(fixed_indices=fixed_indices),
            loads=ExternalLoads(nodal_forces=external_forces, gravity=0.0),
            options=SolverOptions(
                dt=2e-4,
                steps=2,
                damping=0.5,
                convergence_tol=1e-5,
                save_every=1,
            ),
        )

        self.assertFalse(result.status.unstable)
        self.assertEqual(len(result.history.time), 3)
        self.assertEqual(result.final_positions.shape, rest.positions.shape)
        self.assertIsInstance(result.final_positions, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
