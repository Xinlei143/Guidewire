import math
import inspect
import types
import unittest

import torch

import rod_soft_contact_PI_dynamic_quasi_static_3d_v3 as rod


def contact_args(**overrides):
    values = dict(
        dt_active=0.1,
        tube_radius=0.05,
        safe_distance=0.001,
        Kp=120.0,
        ki_active=8.0,
        max_pi_force=20.0,
        integral_leak_rate=0.4,
        max_integral_state=10.0,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


class AnalyticContactJacobianTests(unittest.TestCase):
    def test_matches_batched_autograd_across_contact_branches(self):
        dtype = rod.DTYPE
        positions = torch.zeros((8, 3), dtype=dtype)
        yz = torch.tensor(
            [
                [0.0, 0.0],
                [0.0480, 0.0],
                [0.0492, 0.0006],
                [0.0500, 0.0],
                [0.0510, 0.0010],
                [0.0350, 0.0350],
                [0.0, 0.0520],
                [0.0520, 0.0],
            ],
            dtype=dtype,
        )
        positions[:, 1:] = yz
        integrals = torch.tensor([0.0, 0.1, 0.5, 1.0, 2.0, 0.2, 3.0, 9.9], dtype=dtype)
        previous_errors = torch.tensor([0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.8, 0.0], dtype=dtype)
        args = contact_args()
        pattern = rod.get_assembly_pattern(torch.zeros_like(positions, dtype=torch.bool))

        force_reference, jac_reference = rod.assemble_contact_batched(
            positions, integrals, previous_errors, pattern, args
        )
        force_analytic, jac_analytic = rod.assemble_contact_analytic(
            positions, integrals, previous_errors, pattern, args
        )

        self.assertTrue(torch.allclose(force_analytic, force_reference, rtol=1e-10, atol=1e-10))
        self.assertTrue(torch.allclose(jac_analytic, jac_reference, rtol=1e-9, atol=1e-9))

    def test_analytic_jacobian_has_yz_coupling(self):
        dtype = rod.DTYPE
        positions = torch.zeros((4, 3), dtype=dtype)
        positions[0, 1:] = torch.tensor([0.04, 0.04], dtype=dtype)
        args = contact_args()
        pattern = rod.get_assembly_pattern(torch.zeros_like(positions, dtype=torch.bool))
        _, jacobian = rod.assemble_contact_analytic(
            positions, torch.zeros(4, dtype=dtype), torch.zeros(4, dtype=dtype), pattern, args
        )
        self.assertNotEqual(float(jacobian[0, 0, 1]), 0.0)


class GradientOnlyAssemblyTests(unittest.TestCase):
    def test_optimized_residual_matches_reference_residual(self):
        dtype = rod.DTYPE
        node_count = 8
        positions = torch.zeros((node_count, 3), dtype=dtype)
        positions[:, 0] = torch.linspace(0.0, 0.7, node_count, dtype=dtype)
        positions[:, 1] = 0.012 * torch.sin(torch.linspace(0.0, 1.7 * math.pi, node_count, dtype=dtype))
        positions[:, 2] = 0.009 * torch.cos(torch.linspace(0.0, 1.3 * math.pi, node_count, dtype=dtype))
        velocities = torch.zeros_like(positions)
        rest = torch.full((node_count - 1,), 0.1, dtype=dtype)
        masses = torch.full((node_count,), 0.01, dtype=dtype)
        integrals = torch.linspace(0.0, 0.02, node_count, dtype=dtype)
        previous_errors = torch.zeros(node_count, dtype=dtype)
        pulse = torch.zeros_like(positions)
        prescribed_mask = torch.zeros_like(positions, dtype=torch.bool)
        prescribed_mask[:2] = True
        prescribed = positions.clone()
        q = positions[~prescribed_mask].clone()
        _, ea, ei, damping = rod.rod_properties(0.7, rod.DIAMETER, rod.MATERIAL_DENSITY)
        base = dict(
            dt_active=0.1,
            tube_radius=0.06,
            safe_distance=0.01,
            Kp=3000.0,
            ki_active=100.0,
            max_pi_force=20.0,
            integral_leak_rate=5.0,
            max_integral_state=0.05,
            EA=ea,
            EI=ei,
            damping=damping,
            compute_kernels="reference",
        )
        reference_args = types.SimpleNamespace(**base)
        optimized_args = types.SimpleNamespace(**{**base, "compute_kernels": "optimized"})
        block_plan = rod.get_block_assembly_plan("cpu", node_count, False)
        residual_reference, _, _ = rod.residual_and_jacobian(
            q, positions, velocities, rest, masses, integrals, previous_errors, pulse,
            prescribed_mask, prescribed, reference_args, False, block_plan,
        )
        residual_optimized, _, _ = rod.residual_and_jacobian(
            q, positions, velocities, rest, masses, integrals, previous_errors, pulse,
            prescribed_mask, prescribed, optimized_args, False, block_plan,
        )
        self.assertTrue(torch.allclose(residual_optimized, residual_reference, rtol=1e-9, atol=1e-9))


class PCRImplementationTests(unittest.TestCase):
    def test_cuda_pcr_uses_stride_slices_without_gpu_mask_any(self):
        source = inspect.getsource(rod.BlockPentaSystem._solve_cuda_pcr)
        self.assertNotIn("torch.any(has_left)", source)
        self.assertNotIn("torch.any(has_right)", source)
        solve_source = inspect.getsource(rod.BlockPentaSystem.solve)
        self.assertNotIn("torch.isfinite(solution)", solve_source)


if __name__ == "__main__":
    unittest.main()
