import argparse
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

import rod_soft_contact_PI_dynamic_quasi_static_2d_arc as rod


def test_final_perturbation_cli_defaults_and_rejects_negative_scale():
    with patch.object(sys, "argv", ["rod"]):
        args = rod.parse_args()
    assert args.final_perturbation_scale == 0.0
    assert args.final_perturbation_seed == 0

    with patch.object(sys, "argv", ["rod", "--final-perturbation-scale", "-1e-3"]):
        with pytest.raises(SystemExit):
            rod.parse_args()
    for invalid_scale in ("nan", "inf"):
        with patch.object(sys, "argv", ["rod", "--final-perturbation-scale", invalid_scale]):
            with pytest.raises(SystemExit):
                rod.parse_args()


def _pulse_args(scale: float) -> argparse.Namespace:
    return argparse.Namespace(
        num_nodes=12,
        pulse_count_min=2,
        pulse_count_max=4,
        pulse_force_min=0.2,
        pulse_force_max=0.7,
        arc_radius=0.5,
        final_perturbation_seed=17,
        final_perturbation_scale=scale,
    )


def test_final_perturbation_seed_fixes_base_pulse_while_scale_only_changes_force():
    positions = rod.arc_points(0.5, np.pi / 2.0, 12)
    low = rod.make_final_perturbation_pulse(_pulse_args(0.25), positions)
    high = rod.make_final_perturbation_pulse(_pulse_args(1.5), positions)
    repeat = rod.make_final_perturbation_pulse(_pulse_args(0.25), positions)

    low_nodes, low_base, low_magnitudes, low_directions, low_profile = low
    high_nodes, high_base, high_magnitudes, high_directions, high_profile = high
    repeat_nodes, repeat_base, repeat_magnitudes, repeat_directions, repeat_profile = repeat
    assert np.array_equal(low_nodes, high_nodes)
    assert np.array_equal(low_nodes, repeat_nodes)
    assert np.allclose(low_base, high_base)
    assert np.allclose(low_base, repeat_base)
    assert np.allclose(low_directions, high_directions)
    assert np.allclose(low_directions, repeat_directions)
    assert np.allclose(low_magnitudes, 0.25 * low_base)
    assert np.allclose(high_magnitudes, 1.5 * high_base)
    assert np.allclose(low_profile, 0.25 / 1.5 * high_profile)
    assert np.allclose(low_profile, repeat_profile)
    assert np.allclose(low_magnitudes, repeat_magnitudes)


def test_configuration_change_requires_settling_and_threshold_exceedance():
    assert not rod.classify_final_perturbation_configuration_change(False, 1.0)
    assert not rod.classify_final_perturbation_configuration_change(True, 1.0e-4)
    assert rod.classify_final_perturbation_configuration_change(True, 1.0001e-4)


def _diagnostic_args() -> argparse.Namespace:
    return argparse.Namespace(
        num_nodes=6,
        arc_radius=0.5,
        arc_angle_rad=np.pi / 2.0,
        tube_half_width=0.1,
        safe_distance=0.01,
        pulse_count_min=1,
        pulse_count_max=1,
        pulse_force_min=0.1,
        pulse_force_max=0.1,
        pulse_duration=0.1,
        final_perturbation_scale=1.0,
        final_perturbation_seed=3,
        tip_total_displacement=0.001,
        tip_motion_mode="fixed-direction",
        dt=0.1,
        max_settle_steps=1,
        settle_speed_tol=1.0e-3,
        settle_residual_tol=1.0e-6,
        contact_tolerance=1.0e-3,
        settle_confirm_steps=1,
        EA=1.0,
        EI=1.0,
        Kp=1.0,
        Ki=0.0,
        kp_active=1.0,
        ki_active=0.0,
        max_pi_force=1.0,
        max_integral_state=1.0,
        integral_leak_rate=1.0,
        contact_smooth_length=1.0e-5,
        direction_epsilon=1.0e-6,
    )


def test_unsettled_finite_diagnostic_keeps_post_state_metrics(monkeypatch):
    args = _diagnostic_args()
    positions = torch.as_tensor(rod.arc_points(args.arc_radius, args.arc_angle_rad, args.num_nodes), dtype=rod.DTYPE)
    rest = torch.linalg.vector_norm(positions[1:] - positions[:-1], dim=1)
    state = (positions, torch.zeros_like(positions), torch.zeros(args.num_nodes, dtype=rod.DTYPE), torch.zeros(args.num_nodes, dtype=rod.DTYPE))

    def finite_unsettled_step(p, v, rest, reference_curvature, masses, integrals, previous_errors, pulse, mask, values, args, dt):
        moved = p.clone()
        moved[2, 0] += 2.0e-4
        velocities = torch.zeros_like(p)
        velocities[2, 0] = 1.0e-2
        return moved, velocities, integrals.clone(), torch.zeros_like(p), torch.zeros(args.num_nodes, dtype=torch.bool), torch.zeros(args.num_nodes), previous_errors.clone(), rod.NewtonInfo(True, 1, 0.0, 0.0, 0)

    monkeypatch.setattr(rod, "implicit_euler_step", finite_unsettled_step)
    diagnostic = rod.run_final_perturbation_diagnostic(
        state, positions, rest, torch.zeros(args.num_nodes - 2, dtype=rod.DTYPE),
        torch.ones(args.num_nodes, dtype=rod.DTYPE), torch.tensor([0.0, -1.0], dtype=rod.DTYPE),
        (0.0, 0.0, 0.0), args,
    )

    assert not diagnostic["final_perturbation_post_settled"].item()
    assert diagnostic["final_perturbation_failed"].item()
    assert np.isfinite(diagnostic["final_perturbation_post_positions"]).all()
    assert diagnostic["final_perturbation_rms_position_difference"].item() > 0.0
    assert diagnostic["final_perturbation_max_position_difference"].item() > 0.0
    # A finite attempted state is useful for geometry diagnostics, but it is
    # not a reported steady-state energy until the relaxation criterion holds.
    assert np.isnan(diagnostic["final_perturbation_post_elastic_energy_j"]).item()
    assert np.isnan(diagnostic["final_perturbation_post_elastic_stretch_energy_j"]).item()
    assert np.isnan(diagnostic["final_perturbation_post_elastic_bend_energy_j"]).item()
    assert np.isnan(diagnostic["final_perturbation_delta_elastic_stretch_energy_j"]).item()
    assert np.isnan(diagnostic["final_perturbation_delta_elastic_bend_energy_j"]).item()
    assert np.isnan(diagnostic["final_perturbation_delta_elastic_energy_j"]).item()
    assert np.isnan(diagnostic["final_perturbation_post_contact_total_force_magnitude_sum"]).item()
    assert np.isnan(diagnostic["final_perturbation_post_contact_net_resultant"]).all()
    assert [record[-1] for record in diagnostic["final_perturbation_history_records"]] == [
        "final_perturb_pulse", "final_perturb_relax",
    ]
    assert not diagnostic["final_perturbation_configuration_changed"].item()


def test_finite_diagnostic_does_not_require_numpy_vector_norm(monkeypatch):
    args = _diagnostic_args()
    positions = torch.as_tensor(rod.arc_points(args.arc_radius, args.arc_angle_rad, args.num_nodes), dtype=rod.DTYPE)
    rest = torch.linalg.vector_norm(positions[1:] - positions[:-1], dim=1)
    state = (positions, torch.zeros_like(positions), torch.zeros(args.num_nodes, dtype=rod.DTYPE), torch.zeros(args.num_nodes, dtype=rod.DTYPE))

    def finite_settling_step(p, v, rest, reference_curvature, masses, integrals, previous_errors, pulse, mask, values, args, dt):
        moved = p.clone()
        moved[2, 0] += 2.0e-4
        return moved, torch.zeros_like(p), integrals.clone(), torch.zeros_like(p), torch.zeros(args.num_nodes, dtype=torch.bool), torch.zeros(args.num_nodes), previous_errors.clone(), rod.NewtonInfo(True, 1, 0.0, 0.0, 0)

    monkeypatch.setattr(rod, "implicit_euler_step", finite_settling_step)
    monkeypatch.delattr(np.linalg, "vector_norm", raising=False)
    diagnostic = rod.run_final_perturbation_diagnostic(
        state, positions, rest, torch.zeros(args.num_nodes - 2, dtype=rod.DTYPE),
        torch.ones(args.num_nodes, dtype=rod.DTYPE), torch.tensor([0.0, -1.0], dtype=rod.DTYPE),
        (0.0, 0.0, 0.0), args,
    )

    assert diagnostic["final_perturbation_post_settled"].item()
    assert np.isfinite(diagnostic["final_perturbation_rms_position_difference"]).item()
    assert np.isfinite(diagnostic["final_perturbation_max_position_difference"]).item()


def test_enabled_diagnostic_does_not_mutate_primary_state_or_history_inputs(monkeypatch):
    args = _diagnostic_args()
    positions = torch.as_tensor(rod.arc_points(args.arc_radius, args.arc_angle_rad, args.num_nodes), dtype=rod.DTYPE)
    rest = torch.linalg.vector_norm(positions[1:] - positions[:-1], dim=1)
    state = (
        positions.clone(), torch.full_like(positions, 3.0e-4),
        torch.linspace(0.0, 5.0e-3, args.num_nodes, dtype=rod.DTYPE),
        torch.linspace(5.0e-3, 0.0, args.num_nodes, dtype=rod.DTYPE),
    )
    snapshots = tuple(value.clone() for value in state)

    def finite_settling_step(p, v, rest, reference_curvature, masses, integrals, previous_errors, pulse, mask, values, args, dt):
        moved = p.clone()
        moved[2, 1] += 2.0e-4
        return moved, torch.zeros_like(p), integrals.clone(), torch.zeros_like(p), torch.zeros(args.num_nodes, dtype=torch.bool), torch.zeros(args.num_nodes), previous_errors.clone(), rod.NewtonInfo(True, 1, 0.0, 0.0, 0)

    monkeypatch.setattr(rod, "implicit_euler_step", finite_settling_step)
    diagnostic = rod.run_final_perturbation_diagnostic(
        state, positions, rest, torch.zeros(args.num_nodes - 2, dtype=rod.DTYPE),
        torch.ones(args.num_nodes, dtype=rod.DTYPE), torch.tensor([0.0, -1.0], dtype=rod.DTYPE),
        (0.0, 0.0, 0.0), args,
    )

    assert diagnostic["final_perturbation_post_settled"].item()
    for before, after in zip(snapshots, state):
        assert torch.equal(before, after)
    assert np.array_equal(diagnostic["final_perturbation_baseline_positions"], snapshots[0].numpy())
    assert diagnostic["final_perturbation_baseline_positions"].shape == diagnostic["final_perturbation_post_positions"].shape == tuple(snapshots[0].shape)


def test_diagnostic_exception_preserves_and_saves_primary_result(monkeypatch, tmp_path):
    args = _diagnostic_args()
    args.torch_device = torch.device("cpu")
    args.Kp = 1.0
    args.Ki = 0.0
    args.pulse_duration = 0.0
    args.tip_direction = np.array([0.0, -1.0])
    args.arc_angle_deg = 90.0
    args.tip_displacement_step = args.tip_total_displacement
    args.num_load_steps = 1
    args.prebalance = False
    args.prebalance_max_steps = 1
    args.history_save_interval = 1
    args.initial_imperfection_profile = np.zeros(args.num_nodes)
    args.initial_normal_imperfection_amplitude = 0.0
    args.initial_imperfection_mode_count = 1
    args.initial_imperfection_seed = None
    args.pulse_seed = 0
    args.output_prefix = str(tmp_path / "primary")
    positions = rod.arc_points(args.arc_radius, args.arc_angle_rad, args.num_nodes)
    rest = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    masses = np.ones(args.num_nodes)

    monkeypatch.setattr(rod, "initialize_rod", lambda _args: (positions, np.zeros_like(positions), rest, masses, np.zeros(args.num_nodes)))
    monkeypatch.setattr(rod, "make_random_normal_pulses", lambda _args, _positions: (np.array([2]), np.array([0.0]), np.array([[1.0, 0.0]]), np.zeros_like(positions)))
    monkeypatch.setattr(rod, "prebalance_zero_displacement", lambda p, v, *unused: (p, v, unused[3], unused[4], 0.0))
    monkeypatch.setattr(rod, "try_load_increment", lambda state, *unused: (True, state, 0.0, [], {"relax": 1, "speed": 0.0, "protection": 0.0, "residual": 0.0}))

    def fail_diagnostic(*unused):
        raise RuntimeError("injected final diagnostic failure")

    monkeypatch.setattr(rod, "run_final_perturbation_diagnostic", fail_diagnostic)
    result = rod.run_simulation(args)
    saved = np.load(tmp_path / "primary.npz")
    assert result["positions"].shape[0] >= 1
    assert saved["final_elastic_energy_j"].item() == result["final_elastic_energy_j"].item()
    assert result["final_perturbation_enabled"].item()
    assert result["final_perturbation_failed"].item()
    assert not result["final_perturbation_post_settled"].item()
    assert not result["final_perturbation_configuration_changed"].item()
    for key in (
        "final_perturbation_post_elastic_stretch_energy_j",
        "final_perturbation_post_elastic_bend_energy_j",
        "final_perturbation_post_elastic_energy_j",
        "final_perturbation_delta_elastic_stretch_energy_j",
        "final_perturbation_delta_elastic_bend_energy_j",
        "final_perturbation_delta_elastic_energy_j",
    ):
        assert np.isnan(result[key]).item()
        assert np.isnan(saved[key]).item()


def test_enabled_diagnostic_emits_pulse_and_relax_history_records(monkeypatch):
    args = _diagnostic_args()
    positions = torch.as_tensor(rod.arc_points(args.arc_radius, args.arc_angle_rad, args.num_nodes), dtype=rod.DTYPE)
    rest = torch.linalg.vector_norm(positions[1:] - positions[:-1], dim=1)
    state = (positions, torch.zeros_like(positions), torch.zeros(args.num_nodes, dtype=rod.DTYPE), torch.zeros(args.num_nodes, dtype=rod.DTYPE))

    def settling_step(p, v, rest, reference_curvature, masses, integrals, previous_errors, pulse, mask, values, args, dt):
        moved = p.clone()
        moved[2, 0] += 2.0e-4
        return moved, torch.zeros_like(p), integrals.clone(), torch.zeros_like(p), torch.zeros(args.num_nodes, dtype=torch.bool), torch.zeros(args.num_nodes), previous_errors.clone(), rod.NewtonInfo(True, 1, 0.0, 0.0, 0)

    monkeypatch.setattr(rod, "implicit_euler_step", settling_step)
    diagnostic = rod.run_final_perturbation_diagnostic(
        state, positions, rest, torch.zeros(args.num_nodes - 2, dtype=rod.DTYPE),
        torch.ones(args.num_nodes, dtype=rod.DTYPE), torch.tensor([0.0, -1.0], dtype=rod.DTYPE),
        (0.0, 0.0, 0.0), args,
    )

    records = diagnostic["final_perturbation_history_records"]
    assert [record[-1] for record in records] == ["final_perturb_pulse", "final_perturb_relax"]
    assert diagnostic["final_perturbation_post_settled"].item()
    assert np.isfinite(diagnostic["final_perturbation_post_elastic_energy_j"]).item()
    assert np.isfinite(diagnostic["final_perturbation_baseline_contact_total_force_magnitude_sum"]).item()
    assert np.isfinite(diagnostic["final_perturbation_baseline_contact_net_resultant"]).all()
    assert np.isfinite(diagnostic["final_perturbation_post_contact_total_force_magnitude_sum"]).item()
    assert np.isfinite(diagnostic["final_perturbation_post_contact_net_resultant"]).all()


def test_disabled_diagnostic_emits_no_history_records():
    args = _diagnostic_args()
    args.final_perturbation_scale = 0.0
    positions = torch.as_tensor(rod.arc_points(args.arc_radius, args.arc_angle_rad, args.num_nodes), dtype=rod.DTYPE)
    rest = torch.linalg.vector_norm(positions[1:] - positions[:-1], dim=1)
    state = (positions, torch.zeros_like(positions), torch.zeros(args.num_nodes, dtype=rod.DTYPE), torch.zeros(args.num_nodes, dtype=rod.DTYPE))
    diagnostic = rod.run_final_perturbation_diagnostic(
        state, positions, rest, torch.zeros(args.num_nodes - 2, dtype=rod.DTYPE),
        torch.ones(args.num_nodes, dtype=rod.DTYPE), torch.tensor([0.0, -1.0], dtype=rod.DTYPE),
        (0.0, 0.0, 0.0), args,
    )
    assert diagnostic["final_perturbation_history_records"] == []


def test_contact_segment_summary_groups_contiguous_active_nodes():
    summary = rod.summarize_contact_segments(
        np.array([False, False, True, True, False, True, False]),
        np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 2.0], [-0.5, 1.0], [0.0, 0.0], [3.0, 4.0], [0.0, 0.0]]),
    )
    assert summary["start_nodes"].tolist() == [2, 5]
    assert summary["end_nodes"].tolist() == [3, 5]
    assert summary["counts"].tolist() == [2, 1]
    np.testing.assert_allclose(summary["resultants"], [[0.5, 3.0], [3.0, 4.0]])
    np.testing.assert_allclose(summary["force_magnitude_sums"], [np.sqrt(5.0) + np.sqrt(1.25), 5.0])


def test_final_energy_bar_data_groups_primary_and_settled_post_values():
    labels, primary, post = rod.final_energy_bar_data({
        "final_elastic_stretch_energy_j": np.asarray(1.0),
        "final_elastic_bend_energy_j": np.asarray(2.0),
        "final_elastic_energy_j": np.asarray(3.0),
        "final_perturbation_post_settled": np.asarray(True),
        "final_perturbation_post_elastic_stretch_energy_j": np.asarray(4.0),
        "final_perturbation_post_elastic_bend_energy_j": np.asarray(5.0),
        "final_perturbation_post_elastic_energy_j": np.asarray(6.0),
    })
    assert labels == ("stretch", "bend", "total")
    np.testing.assert_allclose(primary, (1.0, 2.0, 3.0))
    np.testing.assert_allclose(post, (4.0, 5.0, 6.0))


def test_run_simulation_appends_diagnostic_history_and_reports_steady_energy(monkeypatch, tmp_path, capsys):
    args = _diagnostic_args()
    args.torch_device = torch.device("cpu")
    args.backend = "python"
    args.prebalance = False
    args.prebalance_max_steps = 1
    args.history_save_interval = 1
    args.initial_imperfection_profile = np.zeros(args.num_nodes)
    args.initial_normal_imperfection_amplitude = 0.0
    args.initial_imperfection_mode_count = 1
    args.initial_imperfection_seed = None
    args.pulse_seed = 0
    args.output_prefix = str(tmp_path / "with_history")
    args.tip_direction = np.array([0.0, -1.0])
    args.arc_angle_deg = 90.0
    args.tip_displacement_step = args.tip_total_displacement
    args.num_load_steps = 1
    positions = rod.arc_points(args.arc_radius, args.arc_angle_rad, args.num_nodes)
    rest = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    monkeypatch.setattr(rod, "initialize_rod", lambda _args: (positions, np.zeros_like(positions), rest, np.ones(args.num_nodes), np.zeros(args.num_nodes)))
    monkeypatch.setattr(rod, "make_random_normal_pulses", lambda _args, _positions: (np.array([2]), np.array([0.0]), np.array([[1.0, 0.0]]), np.zeros_like(positions)))
    monkeypatch.setattr(rod, "try_load_increment", lambda state, *unused: (True, state, 0.0, [], {"relax": 1, "speed": 0.0, "protection": 0.0, "residual": 0.0}))
    def settling_step(p, v, rest, reference_curvature, masses, integrals, previous_errors, pulse, mask, values, args, dt):
        moved = p.clone()
        moved[2, 0] += 2.0e-4
        return moved, torch.zeros_like(p), integrals.clone(), torch.zeros_like(p), torch.zeros(args.num_nodes, dtype=torch.bool), torch.zeros(args.num_nodes), previous_errors.clone(), rod.NewtonInfo(True, 1, 0.0, 0.0, 0)

    monkeypatch.setattr(rod, "implicit_euler_step", settling_step)
    result = rod.run_simulation(args)
    output = capsys.readouterr().out
    assert result["phase"].tolist()[0] == "initial"
    assert result["phase"].tolist()[-2:] == ["final_perturb_pulse", "final_perturb_relax"]
    assert result["time"].tolist() == sorted(result["time"].tolist())
    assert result["final_elastic_energy_j"].item() == result["final_perturbation_baseline_elastic_energy_j"].item()
    assert np.isfinite(result["final_perturbation_post_elastic_energy_j"]).item()
    assert "[PERTURB PULSE]" in output
    assert "[PERTURB RELAX]" in output
    assert "[ENERGY BASELINE]" in output
    assert "[ENERGY POST STEADY]" in output
    saved = np.load(tmp_path / "with_history.npz")
    assert saved["phase"].tolist() == result["phase"].tolist()
    assert "final_perturbation_post_contact_segment_start_nodes" in saved
    assert "final_perturbation_baseline_contact_total_force_magnitude_sum" in saved
    assert "final_perturbation_post_contact_net_resultant" in saved


def test_scale_zero_run_has_no_final_perturbation_phase_in_result_or_npz(monkeypatch, tmp_path):
    args = _diagnostic_args()
    args.final_perturbation_scale = 0.0
    args.torch_device = torch.device("cpu")
    args.backend = "python"
    args.prebalance = False
    args.prebalance_max_steps = 1
    args.history_save_interval = 1
    args.initial_imperfection_profile = np.zeros(args.num_nodes)
    args.initial_normal_imperfection_amplitude = 0.0
    args.initial_imperfection_mode_count = 1
    args.initial_imperfection_seed = None
    args.pulse_seed = 0
    args.output_prefix = str(tmp_path / "disabled")
    args.tip_direction = np.array([0.0, -1.0])
    args.arc_angle_deg = 90.0
    args.tip_displacement_step = args.tip_total_displacement
    args.num_load_steps = 1
    positions = rod.arc_points(args.arc_radius, args.arc_angle_rad, args.num_nodes)
    rest = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    monkeypatch.setattr(rod, "initialize_rod", lambda _args: (positions, np.zeros_like(positions), rest, np.ones(args.num_nodes), np.zeros(args.num_nodes)))
    monkeypatch.setattr(rod, "make_random_normal_pulses", lambda _args, _positions: (np.array([2]), np.array([0.0]), np.array([[1.0, 0.0]]), np.zeros_like(positions)))
    monkeypatch.setattr(rod, "try_load_increment", lambda state, *unused: (True, state, 0.0, [], {"relax": 1, "speed": 0.0, "protection": 0.0, "residual": 0.0}))
    monkeypatch.setattr(
        rod, "implicit_euler_step",
        lambda p, v, rest, reference_curvature, masses, integrals, previous_errors, pulse, mask, values, args, dt: (
            p.clone(), torch.zeros_like(p), integrals.clone(), torch.zeros_like(p),
            torch.zeros(args.num_nodes, dtype=torch.bool), torch.zeros(args.num_nodes), previous_errors.clone(),
            rod.NewtonInfo(True, 1, 0.0, 0.0, 0),
        ),
    )
    result = rod.run_simulation(args)
    saved = np.load(tmp_path / "disabled.npz")
    for payload in (result, saved):
        assert not np.isin(payload["phase"], ["final_perturb_pulse", "final_perturb_relax"]).any()


def test_mid_diagnostic_exception_preserves_completed_pulse_history(monkeypatch, tmp_path):
    args = _diagnostic_args()
    args.torch_device = torch.device("cpu")
    args.backend = "python"
    args.prebalance = False
    args.prebalance_max_steps = 1
    args.history_save_interval = 1
    args.initial_imperfection_profile = np.zeros(args.num_nodes)
    args.initial_normal_imperfection_amplitude = 0.0
    args.initial_imperfection_mode_count = 1
    args.initial_imperfection_seed = None
    args.pulse_seed = 0
    args.output_prefix = str(tmp_path / "partial_exception")
    args.tip_direction = np.array([0.0, -1.0])
    args.arc_angle_deg = 90.0
    args.tip_displacement_step = args.tip_total_displacement
    args.num_load_steps = 1
    positions = rod.arc_points(args.arc_radius, args.arc_angle_rad, args.num_nodes)
    rest = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    monkeypatch.setattr(rod, "initialize_rod", lambda _args: (positions, np.zeros_like(positions), rest, np.ones(args.num_nodes), np.zeros(args.num_nodes)))
    monkeypatch.setattr(rod, "make_random_normal_pulses", lambda _args, _positions: (np.array([2]), np.array([0.0]), np.array([[1.0, 0.0]]), np.zeros_like(positions)))
    monkeypatch.setattr(rod, "try_load_increment", lambda state, *unused: (True, state, 0.0, [], {"relax": 1, "speed": 0.0, "protection": 0.0, "residual": 0.0}))
    completed_final_pulse = False

    def pulse_then_raise(p, v, rest, reference_curvature, masses, integrals, previous_errors, pulse, mask, values, args, dt):
        nonlocal completed_final_pulse
        if torch.any(pulse):
            completed_final_pulse = True
        elif completed_final_pulse:
            raise RuntimeError("injected diagnostic relaxation failure")
        return p.clone(), torch.zeros_like(p), integrals.clone(), torch.zeros_like(p), torch.zeros(args.num_nodes, dtype=torch.bool), torch.zeros(args.num_nodes), previous_errors.clone(), rod.NewtonInfo(True, 1, 0.0, 0.0, 0)

    monkeypatch.setattr(rod, "implicit_euler_step", pulse_then_raise)
    result = rod.run_simulation(args)
    saved = np.load(tmp_path / "partial_exception.npz")
    for payload in (result, saved):
        assert "final_perturb_pulse" in payload["phase"].tolist()
        assert payload["final_perturbation_failed"].item()
        assert "injected diagnostic relaxation failure" in payload["final_perturbation_failure_reason"].item()
        assert payload["final_elastic_energy_j"].item() == payload["final_perturbation_baseline_elastic_energy_j"].item()
        for key in (
            "final_perturbation_post_elastic_stretch_energy_j",
            "final_perturbation_post_elastic_bend_energy_j",
            "final_perturbation_post_elastic_energy_j",
            "final_perturbation_delta_elastic_stretch_energy_j",
            "final_perturbation_delta_elastic_bend_energy_j",
            "final_perturbation_delta_elastic_energy_j",
        ):
            assert np.isnan(payload[key]).item()


def test_v4_style_perturbation_logging_includes_periodic_step_twenty(monkeypatch, tmp_path, capsys):
    args = _diagnostic_args()
    args.torch_device = torch.device("cpu")
    args.backend = "python"
    args.prebalance = True
    args.prebalance_max_steps = 1
    args.history_save_interval = 1
    args.initial_imperfection_profile = np.zeros(args.num_nodes)
    args.initial_normal_imperfection_amplitude = 0.0
    args.initial_imperfection_mode_count = 1
    args.initial_imperfection_seed = None
    args.pulse_seed = 7
    args.pulse_duration = 2.0
    args.dt = 0.1
    args.max_settle_steps = 20
    args.output_prefix = str(tmp_path / "periodic")
    args.tip_direction = np.array([0.0, -1.0])
    args.arc_angle_deg = 90.0
    args.tip_displacement_step = args.tip_total_displacement
    args.num_load_steps = 1
    positions = rod.arc_points(args.arc_radius, args.arc_angle_rad, args.num_nodes)
    rest = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    monkeypatch.setattr(rod, "initialize_rod", lambda _args: (positions, np.zeros_like(positions), rest, np.ones(args.num_nodes), np.zeros(args.num_nodes)))
    monkeypatch.setattr(rod, "make_random_normal_pulses", lambda _args, _positions: (np.array([2]), np.array([0.0]), np.array([[1.0, 0.0]]), np.zeros_like(positions)))
    monkeypatch.setattr(rod, "try_load_increment", lambda state, *unused: (True, state, 0.0, [], {"relax": 1, "speed": 0.0, "protection": 0.0, "residual": 0.0}))
    final_relax_calls = 0
    initial_tip = torch.as_tensor(positions[-1], dtype=rod.DTYPE)

    def controlled_step(p, v, rest, reference_curvature, masses, integrals, previous_errors, pulse, mask, values, args, dt):
        nonlocal final_relax_calls
        final_relax = not torch.any(pulse) and float(torch.linalg.vector_norm(values[-1] - initial_tip)) > 1.0e-12
        if final_relax:
            final_relax_calls += 1
            velocity = torch.zeros_like(p) if final_relax_calls == 20 else torch.full_like(p, 1.0e-2)
        else:
            velocity = torch.zeros_like(p)
        return p.clone(), velocity, integrals.clone(), torch.zeros_like(p), torch.zeros(args.num_nodes, dtype=torch.bool), torch.zeros(args.num_nodes), previous_errors.clone(), rod.NewtonInfo(True, 1, 0.0, 0.0, 0)

    monkeypatch.setattr(rod, "implicit_euler_step", controlled_step)
    rod.run_simulation(args)
    output = capsys.readouterr().out
    for marker in (
        "[CONFIG]", "[INITIAL PULSE]", "[PREBALANCE]", "[LOAD 01/01]",
        "[PERTURB PULSE]", "[PERTURB RELAX]", "BASELINE CONTACT-SEGMENT",
        "POST STEADY CONTACT-SEGMENT", "[ENERGY BASELINE]", "[ENERGY POST STEADY]", "[OUTPUT]",
    ):
        assert marker in output
    for value in ("seed=3", "scale=1", "nodes=", "base_N=", "actual_N=", "directions=", "step=20"):
        assert value in output
