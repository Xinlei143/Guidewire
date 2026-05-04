"""Geometry checks for the phase-1 no-twist rod."""

from __future__ import annotations

import unittest

import torch

from der_no_twist_model import make_straight_rest_state
from soft_robot_common import (
    compute_curvature_binormal_magnitudes,
    compute_edge_geometry,
    compute_vertex_arc_positions,
)


class TestDerGeometry(unittest.TestCase):
    def test_straight_rest_lengths_are_uniform(self) -> None:
        rest = make_straight_rest_state(num_links=5, total_length=2.0)
        _, lengths, tangents = compute_edge_geometry(rest.positions)

        torch.testing.assert_close(lengths, torch.full((5,), 0.4, dtype=torch.float64))
        torch.testing.assert_close(tangents[:, 0], torch.ones(5, dtype=torch.float64))
        torch.testing.assert_close(tangents[:, 1:], torch.zeros((5, 2), dtype=torch.float64))

    def test_straight_rest_curvature_is_zero(self) -> None:
        rest = make_straight_rest_state(num_links=5, total_length=2.0)
        torch.testing.assert_close(
            compute_curvature_binormal_magnitudes(rest.positions),
            torch.zeros(4, dtype=torch.float64),
        )

    def test_vertex_arc_positions_are_interior_nodes(self) -> None:
        rest = make_straight_rest_state(num_links=4, total_length=1.0)
        torch.testing.assert_close(
            compute_vertex_arc_positions(rest.lengths),
            torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64),
        )


if __name__ == "__main__":
    unittest.main()
