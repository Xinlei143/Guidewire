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


@pytest.mark.parametrize("force_vector", [(0.0, 0.005, 0.0), (0.0, 0.003, 0.004)])
def test_transverse_error_uses_yz_vector_for_any_direction(tmp_path, force_vector):
    from tools.validate_cantilever_displacement import run_validation

    report = run_validation(
        num_links=6, force_vector=force_vector, output_dir=tmp_path,
    )
    reference = report["eb_reference"]
    assert reference["comparison_quantity"] == "transverse_yz_displacement_vector"
    for key in ("force_node", "tip"):
        measured = torch.tensor(report[f"{key}_displacement_m"][1:3])
        expected = torch.tensor(reference[f"{key}_displacement_m"][1:3])
        ratio = torch.linalg.vector_norm(measured - expected) / torch.linalg.vector_norm(expected)
        assert reference[f"{key}_relative_error"] == pytest.approx(float(ratio))
        assert reference[f"{key}_relative_error"] is not None


def test_zero_load_has_undefined_relative_error_without_false_comparison(tmp_path):
    from tools.validate_cantilever_displacement import run_validation

    report = run_validation(num_links=6, force_vector=(0.0, 0.0, 0.0), output_dir=tmp_path)
    assert report["solver"]["success"]
    assert report["eb_reference"]["comparison_quantity"] == "transverse_yz_displacement_vector"
    assert report["eb_reference"]["relative_error_status"] == "undefined_zero_reference"
    assert report["eb_reference"]["force_node_relative_error"] is None
    assert report["eb_reference"]["tip_relative_error"] is None


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_residual_tolerance_is_rejected(tmp_path, tolerance):
    from tools.validate_cantilever_displacement import run_validation

    with pytest.raises(ValueError, match="residual tolerance"):
        run_validation(num_links=6, output_dir=tmp_path, residual_tolerance=tolerance)
