import json
from pathlib import Path

import numpy as np


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
