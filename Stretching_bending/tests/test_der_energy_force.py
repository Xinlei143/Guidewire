"""Energy and force checks for the phase-1 no-twist rod."""

from __future__ import annotations

import unittest

import torch

from der_no_twist_model import (
    RodParams,
    bending_energy,
    elastic_internal_forces,
    make_straight_rest_state,
    total_elastic_energy,
)


class TestDerEnergyForce(unittest.TestCase):
    def test_rest_energy_is_zero(self) -> None:
        rest = make_straight_rest_state(num_links=6, total_length=1.0)
        params = RodParams(EA=1000.0, B=0.5, node_masses=torch.ones(7, dtype=torch.float64))

        energies = total_elastic_energy(rest.positions, rest, params)

        self.assertAlmostEqual(energies["stretching_energy"].item(), 0.0, places=24)
        self.assertAlmostEqual(energies["bending_energy"].item(), 0.0, places=24)

    def test_translation_invariance(self) -> None:
        rest = make_straight_rest_state(num_links=6, total_length=1.0)
        params = RodParams(EA=1000.0, B=0.5, node_masses=torch.ones(7, dtype=torch.float64))
        shifted = rest.positions + torch.tensor([2.0, -1.0, 0.25], dtype=torch.float64)

        reference = total_elastic_energy(rest.positions, rest, params)
        translated = total_elastic_energy(shifted, rest, params)

        self.assertAlmostEqual(
            translated["elastic_energy"].item(),
            reference["elastic_energy"].item(),
            places=20,
        )

    def test_rest_stretching_force_is_zero(self) -> None:
        rest = make_straight_rest_state(num_links=6, total_length=1.0)
        params = RodParams(EA=1000.0, B=0.5, node_masses=torch.ones(7, dtype=torch.float64))
        forces = elastic_internal_forces(rest.positions, rest, params)

        torch.testing.assert_close(forces, torch.zeros_like(forces), atol=1e-12, rtol=0.0)

    def test_bending_force_matches_energy_gradient(self) -> None:
        rest = make_straight_rest_state(num_links=4, total_length=1.0)
        positions = rest.positions.clone()
        positions[2, 2] = 0.02
        stiffness = 0.5
        params = RodParams(EA=0.0, B=stiffness, node_masses=torch.ones(5, dtype=torch.float64))

        forces = elastic_internal_forces(positions, rest, params)
        epsilon = 1e-6
        plus = positions.clone()
        minus = positions.clone()
        plus[2, 2] += epsilon
        minus[2, 2] -= epsilon
        finite_difference = (
            bending_energy(plus, rest, stiffness) - bending_energy(minus, rest, stiffness)
        ) / (2.0 * epsilon)

        self.assertAlmostEqual(-forces[2, 2].item(), finite_difference.item(), places=6)


if __name__ == "__main__":
    unittest.main()
