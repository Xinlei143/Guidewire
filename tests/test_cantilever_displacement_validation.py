import json

import pytest
import torch

from softrobot.foundations.cantilever.eb_beam_validation import (
    euler_bernoulli_point_load_positions,
    euler_bernoulli_tip_load_positions,
)


def test_interior_point_load_is_piecewise_and_tip_api_is_unchanged():
    positions = torch.stack((torch.arange(5, dtype=torch.float64), torch.zeros(5), torch.zeros(5)), dim=1)
    force = torch.tensor([0.0, 0.0, 6.0], dtype=torch.float64)
    interior = euler_bernoulli_point_load_positions(positions, force, 2.0, force_node=3)
    # Clamp x=1, load x=3 (a=2), tip x=4 (s=3).
    assert interior[2, 2].item() == pytest.approx(2.5)
    assert interior[3, 2].item() == pytest.approx(8.0)
    assert interior[4, 2].item() == pytest.approx(14.0)
    assert torch.allclose(
        euler_bernoulli_point_load_positions(positions, force, 2.0, force_node=4),
        euler_bernoulli_tip_load_positions(positions, force, 2.0),
    )


def test_static_validation_writes_residual_and_displacements(tmp_path):
    from tools.validate_cantilever_displacement import run_validation

    result = run_validation(
        num_links=6, total_length=1.0, EA=1000.0, B=0.5,
        force_node=4, force_vector=(0.0, 0.0, 0.005),
        output_dir=tmp_path, residual_tolerance=1e-5,
    )
    assert result["solver"]["success"]
    assert result["solver"]["free_force_residual_inf_n"] <= 1e-5
    assert result["force_node"] == 4
    assert result["force_node_displacement_m"][2] > 0
    assert result["tip_displacement_m"][2] > result["force_node_displacement_m"][2]
    assert result["eb_reference"]["tip_relative_error"] >= 0
    assert json.loads((tmp_path / "cantilever_displacement.json").read_text()) == result
