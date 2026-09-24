import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _api():
    import rod_soft_contact_PI_dynamic_quasi_static_3d_straight_free_insert as module

    return module


def _smoke_config(tmp_path: Path, **overrides):
    api = _api()
    values = dict(
        num_nodes=12,
        young_modulus=2.0e6,
        contact_kp=100.0,
        contact_ki=0.0,
        friction_kt_line=500.0,
        dt=0.01,
        max_newton_iters=4,
        residual_tol=1.0e3,
        step_tol=1.0e-10,
        settle_speed_tol=1.0e9,
        settle_residual_tol=1.0e9,
        contact_tolerance=1.0,
        settle_confirm_steps=1,
        max_settle_steps=1,
        prebalance_max_steps=1,
        total_insertion=1.0e-5,
        num_load_steps=1,
        ramp_steps=1,
        retry_subdivisions=(1,),
        output_dir=str(tmp_path),
        run_name="smoke",
        write_animation=False,
    )
    values.update(overrides)
    return api.StraightFreeInsertConfig(**values)


def test_single_run_commits_once_per_settled_increment_and_writes_outputs(tmp_path):
    api = _api()
    config = _smoke_config(tmp_path, friction_mu=0.0)

    result = api.run_simulation(config)

    assert result.completed_load_steps == 1
    assert result.friction_commit_count == 1
    assert result.config.friction_mu == 0.0
    for path in (result.config_path, result.settled_path, result.history_path, result.figure_path):
        assert path.is_file()
    payload = json.loads(result.config_path.read_text())
    assert payload["friction_mu"] == 0.0
    assert np.all(result.history["friction_resultant_n"] == 0.0)
    assert np.all(result.history["friction_force_n"] == 0.0)
    assert result.history["positions"].shape[1:] == (config.num_nodes, 3)
    assert result.history["free_tip_advance_m"][0] == 0.0


def test_comparison_runs_mu_zero_and_one_and_creates_joint_figure(tmp_path):
    api = _api()
    config = _smoke_config(tmp_path)

    comparison = api.run_comparison(config, mus=(0.0, 1.0))

    assert set(comparison.results) == {0.0, 1.0}
    assert comparison.figure_path.is_file()
    assert comparison.results[0.0].history["friction_resultant_n"][-1] == 0.0
    for mu, result in comparison.results.items():
        assert result.completed_load_steps == config.num_load_steps
        assert result.history["actuator_displacement_m"][-1] == config.total_insertion
        assert result.history["free_tip_advance_m"].shape == (config.num_load_steps + 1,)


def test_default_campaign_is_360_mm_in_80_four_point_five_mm_steps():
    api = _api()

    config = api.StraightFreeInsertConfig()
    args = api.parse_args([])

    assert config.total_insertion == pytest.approx(360.0e-3)
    assert config.num_load_steps == 80
    assert config.total_insertion / config.num_load_steps == pytest.approx(4.5e-3)
    assert args.total_insertion == pytest.approx(360.0e-3)
    assert args.num_load_steps == 80


def test_finite_display_vessel_and_tracking_camera_cover_entire_guidewire():
    api = _api()
    config = api.StraightFreeInsertConfig()
    initial = api.initialize_rod(config).positions
    final = initial.copy()
    final[:, 0] -= config.total_insertion
    history = {"positions": np.stack((initial, final))}

    vessel_start, vessel_end = api.display_vessel_axial_bounds(config)
    api.validate_display_vessel_history(history, config)
    camera_start = api.guidewire_tracking_camera_bounds(initial, config)
    camera_end = api.guidewire_tracking_camera_bounds(final, config)

    assert (vessel_start, vessel_end) == pytest.approx((-0.410, 1.550))
    for positions, camera in ((initial, camera_start), (final, camera_end)):
        assert np.min(positions[:, 0]) >= vessel_start
        assert np.max(positions[:, 0]) <= vessel_end
        assert np.min(positions[:, 0]) >= camera[0]
        assert np.max(positions[:, 0]) <= camera[1]
    assert camera_start[1] - camera_start[0] == pytest.approx(
        camera_end[1] - camera_end[0]
    )


def test_tracking_view_keeps_physical_limits_but_uses_a_legible_display_aspect():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    api = _api()
    config = api.StraightFreeInsertConfig()
    positions = api.initialize_rod(config).positions
    figure = plt.figure()
    axis = figure.add_subplot(projection="3d")
    api._configure_tracking_axis(axis, positions, config)

    x_start, x_end, radial_limit = api.guidewire_tracking_camera_bounds(positions, config)
    np.testing.assert_allclose(axis.get_xlim(), (x_start, x_end))
    np.testing.assert_allclose(axis.get_ylim(), (-radial_limit, radial_limit))
    assert axis.get_box_aspect()[1] / axis.get_box_aspect()[0] >= 0.24
    plt.close(figure)


def test_display_vessel_validation_rejects_a_wire_outside_its_finite_range():
    api = _api()
    config = api.StraightFreeInsertConfig()
    positions = api.initialize_rod(config).positions
    positions[0, 0] = -config.total_insertion - 0.051

    with pytest.raises(ValueError, match="outside the finite display vessel"):
        api.validate_display_vessel_history({"positions": positions[None, ...]}, config)


def test_single_run_writes_a_following_guidewire_gif(tmp_path):
    api = _api()
    config = _smoke_config(tmp_path, write_animation=True)

    result = api.run_simulation(config)

    assert result.animation_path is not None
    assert result.animation_path.is_file()
    with Image.open(result.animation_path) as image:
        assert image.n_frames == config.num_load_steps + 1


def test_comparison_writes_a_two_mu_following_guidewire_gif(tmp_path):
    api = _api()
    config = _smoke_config(tmp_path, write_animation=True)

    comparison = api.run_comparison(config, mus=(0.0, 1.0))

    assert comparison.animation_path is not None
    assert comparison.animation_path.is_file()
    with Image.open(comparison.animation_path) as image:
        assert image.n_frames == config.num_load_steps + 1


def test_settled_history_contains_reaction_contact_energy_and_dynamic_diagnostics(tmp_path):
    api = _api()
    result = api.run_simulation(_smoke_config(tmp_path, friction_mu=1.0))

    required = {
        "actuator_reaction_n",
        "push_force_n",
        "normal_resultant_n",
        "friction_resultant_n",
        "friction_axial_n",
        "contact_count",
        "free_contact_start",
        "free_contact_end",
        "stick_count",
        "slip_count",
        "bending_energy_j",
        "dynamic_remainder_n",
        "dynamic_to_push_ratio",
    }
    assert required <= set(result.history)
    assert np.all(np.isfinite(result.history["dynamic_remainder_n"]))
    assert result.history["contact_count"][0] >= 1


def test_progress_output_uses_v4_style_phase_and_settling_labels(tmp_path, capsys):
    api = _api()

    api.run_simulation(_smoke_config(tmp_path, progress=True))

    output = capsys.readouterr().out
    for label in ("[PREBALANCE", "[PREBALANCE COMPLETE]", "[LOAD 01/01]", "[RAMP", "[RELAX", "[LOAD COMPLETE"):
        assert label in output


def test_high_dynamic_to_push_ratio_is_diagnostic_only_for_settled_increment(
    tmp_path, monkeypatch
):
    """A vanishing actuator reaction must not reject an otherwise settled step."""
    api = _api()
    original_diagnostics = api._diagnostics

    def high_ratio_diagnostics(*args, **kwargs):
        diagnostics = original_diagnostics(*args, **kwargs)
        diagnostics["dynamic_remainder_n"] = 1.0
        diagnostics["push_force_n"] = 1.0e-8
        diagnostics["dynamic_to_push_ratio"] = 1.0e8
        return diagnostics

    def forbidden_gate(*_args, **_kwargs):
        pytest.fail("dynamic balance gate must not participate in load-step acceptance")

    monkeypatch.setattr(api, "_diagnostics", high_ratio_diagnostics)
    monkeypatch.setattr(api, "dynamic_balance_acceptable", forbidden_gate)

    result = api.run_simulation(_smoke_config(tmp_path, friction_mu=0.0))

    assert result.completed_load_steps == 1
    assert result.friction_commit_count == 1
    assert result.history["dynamic_to_push_ratio"][-1] > 0.05


def test_small_mu_zero_free_insertion_settles_with_physical_tolerances(tmp_path, monkeypatch):
    """A small true-parameter μ=0 run must translate freely and commit once."""
    api = _api()
    default = api.StraightFreeInsertConfig()
    config = api.StraightFreeInsertConfig(
        num_nodes=16,
        total_insertion=2.0e-4,
        num_load_steps=1,
        ramp_steps=3,
        max_settle_steps=500,
        prebalance_max_steps=500,
        friction_mu=0.0,
        output_dir=str(tmp_path),
        run_name="mu_zero_physical",
        progress=False,
    )
    settled_results = []
    original_settled = api._settled

    def capture_settled(result, candidate_config):
        accepted = original_settled(result, candidate_config)
        if accepted:
            settled_results.append(result)
        return accepted

    monkeypatch.setattr(api, "_settled", capture_settled)
    result = api.run_simulation(config)

    assert config.settle_speed_tol == default.settle_speed_tol
    assert config.settle_residual_tol == default.settle_residual_tol
    assert config.contact_tolerance == default.contact_tolerance
    assert result.completed_load_steps == 1
    assert result.friction_commit_count == 1
    assert settled_results[-1].info.converged
    assert np.all(result.history["friction_force_n"] == 0.0)
    assert result.history["friction_resultant_n"][-1] == 0.0
    assert result.history["free_tip_advance_m"][-1] > 0.0
