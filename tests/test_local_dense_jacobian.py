"""Regression tests for the local dense Newton Jacobian implementation."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F

import rod_soft_contact_PI_dynamic_quasi_static as rod


class LocalDenseJacobianTests(unittest.TestCase):
    device = torch.device("cpu")

    def _state(self, num_nodes: int, contact: bool):
        positions, velocities, rest, masses, integrals = rod.initialize_rod(num_nodes=num_nodes)
        old_positions = torch.as_tensor(positions, dtype=rod.DTYPE, device=self.device)
        if contact:
            phase = torch.linspace(0.0, math.pi, num_nodes, dtype=rod.DTYPE)
            old_positions = old_positions.clone()
            old_positions[:, 1] = 0.056 * torch.sin(phase)
            integrals = torch.linspace(0.0, 0.02, num_nodes, dtype=rod.DTYPE).numpy()
        old_velocities = torch.as_tensor(velocities, dtype=rod.DTYPE, device=self.device)
        rest_lengths = torch.as_tensor(rest, dtype=rod.DTYPE, device=self.device)
        node_masses = torch.as_tensor(masses, dtype=rod.DTYPE, device=self.device)
        old_integrals = torch.as_tensor(integrals, dtype=rod.DTYPE, device=self.device)
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
        pulse = torch.zeros_like(old_positions)
        residual_args = (
            old_positions, rest_lengths, pulse, old_integrals, free_mask, prescribed_values,
            q_old, v_old, free_node_masses, previous_errors,
            rod.DT, rod.DAMPING, rod.EA, rod.EI, rod.WALL_HALF_GAP, rod.SAFE_DISTANCE,
            rod.KP, rod.KI, rod.MAX_PI_FORCE, rod.INTEGRAL_LEAK_RATE,
            rod.MAX_INTEGRAL_STATE,
        )
        return q, residual_args

    def test_local_energy_sums_match_global_energy(self):
        positions, _, rest, _, _ = rod.initialize_rod(num_nodes=17)
        positions = torch.as_tensor(positions, dtype=rod.DTYPE, device=self.device)
        positions = positions.clone()
        positions[:, 1] += 0.004 * torch.sin(torch.linspace(0.0, 4.0, 17, dtype=rod.DTYPE))
        rest = torch.as_tensor(rest, dtype=rod.DTYPE, device=self.device)
        stretch_global, bend_global, _ = rod.compute_elastic_energy_torch(
            positions, rest, rod.EA, rod.EI
        )
        positions_3d = torch.cat((positions, torch.zeros((17, 1), dtype=rod.DTYPE)), dim=1)
        edges_3d = positions_3d[1:] - positions_3d[:-1]
        lengths_3d = torch.linalg.norm(edges_3d, dim=1).clamp_min(1.0e-12)
        tangents_3d = edges_3d / lengths_3d[:, None]
        legacy_stretch = 0.5 * rod.EA * torch.sum((lengths_3d - rest) ** 2 / rest)
        legacy_cross = torch.linalg.cross(tangents_3d[:-1], tangents_3d[1:], dim=1)
        legacy_dot = torch.sum(tangents_3d[:-1] * tangents_3d[1:], dim=1)
        legacy_kappa = 2.0 * legacy_cross / (1.0 + legacy_dot).clamp_min(1.0e-12)[:, None]
        legacy_bend = 0.5 * rod.EI * torch.sum(
            torch.sum(legacy_kappa**2, dim=1) / (0.5 * (rest[:-1] + rest[1:]))
        )
        stretch_local = torch.stack([
            rod.local_stretch_energy_2d(positions[i:i + 2], rest[i], rod.EA)
            for i in range(positions.shape[0] - 1)
        ]).sum()
        bend_local = torch.stack([
            rod.local_bending_energy_2d(positions[i - 1:i + 2], rest[i - 1], rest[i], rod.EI)
            for i in range(1, positions.shape[0] - 1)
        ]).sum()
        torch.testing.assert_close(stretch_local, stretch_global, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(bend_local, bend_global, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(stretch_global, legacy_stretch, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(bend_global, legacy_bend, rtol=1e-12, atol=1e-12)

    def test_implicit_step_does_not_mutate_input_state(self):
        positions, velocities, rest, masses, integrals = rod.initialize_rod(num_nodes=12)
        positions_before = positions.copy()
        velocities_before = velocities.copy()
        integrals_before = integrals.copy()
        prescribed_mask, prescribed_values = rod._make_prescribed_state(positions, True)
        prescribed_values[-1, 0] -= 2.0e-5
        rod.implicit_euler_step(
            positions, velocities, rest, masses, np.zeros_like(positions), integrals,
            prescribed_mask, prescribed_values, rod.DT, rod.DAMPING,
            max_newton_iters=3, jacobian_mode="local-batched", device=self.device,
        )
        np.testing.assert_array_equal(positions, positions_before)
        np.testing.assert_array_equal(velocities, velocities_before)
        np.testing.assert_array_equal(integrals, integrals_before)

    def test_dof_mapping_matches_boolean_packing(self):
        positions = torch.arange(10, dtype=rod.DTYPE, device=self.device).reshape(5, 2)
        mask = torch.zeros_like(positions, dtype=torch.bool)
        mask[0, :] = True
        mask[3, 1] = True
        _, free_full_indices = rod.build_dof_mapping(mask)
        torch.testing.assert_close(positions[~mask], positions.reshape(-1)[free_full_indices])

    def test_scalar_contact_matches_vectorized_contact(self):
        y = torch.tensor([-0.070, -0.055, -0.050, -0.049, 0.0, 0.049, 0.050, 0.055, 0.070], dtype=rod.DTYPE)
        positions = torch.stack((torch.zeros_like(y), y), dim=1)
        integrals = torch.linspace(0.0, rod.MAX_INTEGRAL_STATE, y.numel(), dtype=rod.DTYPE)
        previous_errors = torch.linspace(0.0, 0.01, y.numel(), dtype=rod.DTYPE)
        vector = rod._contact_quantities_torch(
            positions, integrals, rod.DT, rod.WALL_HALF_GAP, rod.SAFE_DISTANCE,
            rod.KP, rod.KI, rod.MAX_PI_FORCE, rod.INTEGRAL_LEAK_RATE,
            rod.MAX_INTEGRAL_STATE, previous_errors,
        )
        scalar = [rod.contact_quantities_single_node(
            y[i], integrals[i], previous_errors[i], rod.DT, rod.WALL_HALF_GAP,
            rod.SAFE_DISTANCE, rod.KP, rod.KI, rod.MAX_PI_FORCE,
            rod.INTEGRAL_LEAK_RATE, rod.MAX_INTEGRAL_STATE,
        ) for i in range(y.numel())]
        torch.testing.assert_close(torch.stack([value[0] for value in scalar]), vector[0][:, 1])
        torch.testing.assert_close(torch.stack([value[1] for value in scalar]), vector[1])
        torch.testing.assert_close(torch.stack([value[2] for value in scalar]), vector[4])
        torch.testing.assert_close(torch.stack([value[3] for value in scalar]), vector[3])
        torch.testing.assert_close(torch.stack([value[4] for value in scalar]), vector[2])
        torch.testing.assert_close(torch.stack([value[5] for value in scalar]), vector[5])

    def test_local_dense_matches_full_without_contact(self):
        q, args = self._state(num_nodes=12, contact=False)
        residual_full, jacobian_full = rod._residual_and_jacobian_full_autograd(q, args)
        residual_local, jacobian_local = rod._residual_and_jacobian_local_dense(q, args)
        residual_batched, jacobian_batched = rod._residual_and_jacobian_local_batched(q, args)
        torch.testing.assert_close(residual_local, residual_full, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(jacobian_local, jacobian_full, rtol=1e-9, atol=1e-7)
        torch.testing.assert_close(residual_batched, residual_local, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(jacobian_batched, jacobian_local, rtol=1e-9, atol=1e-7)
        torch.testing.assert_close(
            torch.linalg.solve(jacobian_local, -residual_local),
            torch.linalg.solve(jacobian_full, -residual_full),
            rtol=1e-9, atol=1e-10,
        )

    def test_local_dense_matches_full_with_pi_contact(self):
        q, args = self._state(num_nodes=12, contact=True)
        residual_full, jacobian_full = rod._residual_and_jacobian_full_autograd(q, args)
        residual_local, jacobian_local = rod._residual_and_jacobian_local_dense(q, args)
        residual_batched, jacobian_batched = rod._residual_and_jacobian_local_batched(q, args)
        torch.testing.assert_close(residual_local, residual_full, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(jacobian_local, jacobian_full, rtol=1e-9, atol=1e-7)
        torch.testing.assert_close(residual_batched, residual_local, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(jacobian_batched, jacobian_local, rtol=1e-9, atol=1e-7)

    def test_one_implicit_step_matches_full_autograd(self):
        positions, velocities, rest, masses, integrals = rod.initialize_rod(num_nodes=12)
        prescribed_mask, prescribed_values = rod._make_prescribed_state(positions, True)
        prescribed_values[-1, 0] -= 2.0e-5
        pulse = torch.zeros((12, 2), dtype=rod.DTYPE)
        common = dict(
            dt=rod.DT, damping=rod.DAMPING, max_newton_iters=6,
            device=self.device,
        )
        full = rod.implicit_euler_step(
            positions, velocities, rest, masses, pulse, integrals,
            prescribed_mask, prescribed_values, jacobian_mode="full-autograd", **common,
        )
        local = rod.implicit_euler_step(
            positions, velocities, rest, masses, pulse, integrals,
            prescribed_mask, prescribed_values, jacobian_mode="local-dense", **common,
        )
        batched = rod.implicit_euler_step(
            positions, velocities, rest, masses, pulse, integrals,
            prescribed_mask, prescribed_values, jacobian_mode="local-batched", **common,
        )
        for full_value, local_value in zip(full[:8], local[:8]):
            torch.testing.assert_close(
                torch.as_tensor(local_value), torch.as_tensor(full_value),
                rtol=1e-9, atol=1e-10,
            )
        self.assertEqual(local[-1].iterations, full[-1].iterations)
        self.assertEqual(local[-1].converged, full[-1].converged)
        for full_value, batched_value in zip(full[:8], batched[:8]):
            torch.testing.assert_close(
                torch.as_tensor(batched_value), torch.as_tensor(full_value),
                rtol=1e-9, atol=1e-10,
            )
        self.assertEqual(batched[-1].iterations, full[-1].iterations)
        self.assertEqual(batched[-1].converged, full[-1].converged)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_one_implicit_step_matches_full_autograd_on_cuda(self):
        positions, velocities, rest, masses, integrals = rod.initialize_rod(num_nodes=12)
        prescribed_mask, prescribed_values = rod._make_prescribed_state(positions, True)
        prescribed_values[-1, 0] -= 2.0e-5
        pulse = torch.zeros((12, 2), dtype=rod.DTYPE)
        common = dict(
            dt=rod.DT, damping=rod.DAMPING, max_newton_iters=6,
            device=torch.device("cuda"),
        )
        full = rod.implicit_euler_step(
            positions, velocities, rest, masses, pulse, integrals,
            prescribed_mask, prescribed_values, jacobian_mode="full-autograd", **common,
        )
        local = rod.implicit_euler_step(
            positions, velocities, rest, masses, pulse, integrals,
            prescribed_mask, prescribed_values, jacobian_mode="local-dense", **common,
        )
        batched = rod.implicit_euler_step(
            positions, velocities, rest, masses, pulse, integrals,
            prescribed_mask, prescribed_values, jacobian_mode="local-batched", **common,
        )
        for full_value, local_value in zip(full[:8], local[:8]):
            torch.testing.assert_close(
                torch.as_tensor(local_value), torch.as_tensor(full_value),
                rtol=1e-9, atol=1e-10,
            )
        self.assertEqual(local[-1].iterations, full[-1].iterations)
        self.assertEqual(local[-1].converged, full[-1].converged)
        for full_value, batched_value in zip(full[:8], batched[:8]):
            torch.testing.assert_close(
                torch.as_tensor(batched_value), torch.as_tensor(full_value),
                rtol=1e-9, atol=1e-10,
            )
        self.assertEqual(batched[-1].iterations, full[-1].iterations)
        self.assertEqual(batched[-1].converged, full[-1].converged)


if __name__ == "__main__":
    unittest.main(verbosity=2)
