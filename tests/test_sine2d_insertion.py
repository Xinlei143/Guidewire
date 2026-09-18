import dataclasses

import numpy as np
import pytest
import torch


def _api():
    import rod_soft_contact_PI_dynamic_quasi_static_2d_sine_insert as module

    return module


def _tiny_config(**overrides):
    api = _api()
    values = dict(
        backend="python",
        num_nodes=12,
        wire_length=0.30,
        initial_inserted_length=0.05,
        total_insertion=2.0e-4,
        num_load_steps=1,
        young_modulus=2.0e6,
        contact_kp_line=2.0e3,
        contact_ki_line=0.0,
        feeder_kp_line=2.0e3,
        friction_kt_line=2.0e3,
        dt=0.02,
        ramp_steps=2,
        max_newton_iters=10,
        residual_tol=1.0e-8,
        step_tol=1.0e-11,
        settle_speed_tol=0.2,
        settle_residual_tol=1.0e-6,
        contact_tolerance=2.0e-3,
        settle_confirm_steps=1,
        max_settle_steps=30,
        prebalance=True,
        prebalance_max_steps=30,
        history_save_interval=5,
        retry_subdivisions=(),
    )
    values.update(overrides)
    return api.SineInsertConfig(**values)


def _step_inputs(config):
    api = _api()
    tube = api.SineTube2D(config.amplitude, config.x_span, config.tube_half_width)
    initial = api.initialize_rod(config, tube)
    p = torch.as_tensor(initial.positions, dtype=torch.float64)
    v = torch.as_tensor(initial.velocities, dtype=torch.float64)
    integrals = torch.as_tensor(initial.integrals, dtype=torch.float64)
    errors = torch.as_tensor(initial.previous_errors, dtype=torch.float64)
    rest = torch.as_tensor(initial.rest_lengths, dtype=torch.float64)
    reference = torch.as_tensor(initial.reference_curvature, dtype=torch.float64)
    masses = torch.as_tensor(initial.masses, dtype=torch.float64)
    nodal = torch.as_tensor(initial.nodal_lengths, dtype=torch.float64)
    frame = api.freeze_contact_frames(p, tube)
    normal = api.contact_quantities(p, integrals, errors, nodal, frame, config, config.dt)
    history = api.initialize_friction_history(p, frame, normal.active)
    context = api.begin_friction_increment(history, p)
    prescribed_np = api.proximal_prescribed_state(initial.positions, 1.0e-5, config.amplitude)
    mask = torch.as_tensor(prescribed_np.mask, dtype=torch.bool)
    prescribed = torch.as_tensor(prescribed_np.positions, dtype=torch.float64)
    return api, tube, initial, p, v, integrals, errors, rest, reference, masses, nodal, history, context, mask, prescribed


def test_tracking_projection_uses_trusted_warm_solution_and_global_fallback():
    api = _api()
    tube = api.SineTube2D()
    accepted_q = np.array([0.3, 0.7], dtype=np.float64)
    nearby = tube.centerline(accepted_q) + 1.0e-5 * tube.normal(accepted_q)

    trusted = tube.project_tracked(nearby, accepted_q, trust_radius=0.02)

    np.testing.assert_allclose(trusted.q, accepted_q, atol=2.0e-10)
    assert not np.any(trusted.used_global_fallback)

    adversarial_q = 1.27
    adversarial = tube.centerline(adversarial_q) + 0.006 * tube.normal(adversarial_q)
    recovered = tube.project_tracked(adversarial, 0.05, trust_radius=0.02)

    assert recovered.used_global_fallback
    assert recovered.q == pytest.approx(adversarial_q, abs=2.0e-9)


def test_public_snapshot_tensors_cannot_mutate_frame_history_or_context():
    api = _api()
    config = _tiny_config(friction_mu=10.0)
    tube = api.SineTube2D(config.amplitude, config.x_span, config.tube_half_width)
    q = np.array([0.2, 0.4], dtype=np.float64)
    positions = torch.as_tensor(
        tube.centerline(q) + 0.045 * tube.normal(q), dtype=torch.float64
    )
    frame = api.freeze_contact_frames(positions, tube)
    lengths = torch.full((2,), 0.1, dtype=torch.float64)
    zeros = torch.zeros(2, dtype=torch.float64)
    baseline_contact = api.contact_quantities(
        positions, zeros, zeros, lengths, frame, config, config.dt
    )
    history = api.initialize_friction_history(positions, frame, baseline_contact.active)
    context = api.begin_friction_increment(history, positions)
    baseline_friction = api.evaluate_friction(
        positions + 1.0e-5 * frame.tangent,
        frame,
        baseline_contact.vessel_normal_force,
        baseline_contact.active,
        lengths,
        context,
        config,
    )
    expected_q = frame.q
    expected_slip = history.elastic_slip
    expected_anchors = context.anchors

    frame.q.add_(9.0)
    frame.tangent.zero_()
    history.elastic_slip.add_(7.0)
    history.anchors.zero_()
    context.elastic_slip.add_(5.0)
    context.anchors.add_(4.0)

    torch.testing.assert_close(frame.q, expected_q, rtol=0.0, atol=0.0)
    torch.testing.assert_close(history.elastic_slip, expected_slip, rtol=0.0, atol=0.0)
    torch.testing.assert_close(context.anchors, expected_anchors, rtol=0.0, atol=0.0)
    repeated_contact = api.contact_quantities(
        positions, zeros, zeros, lengths, frame, config, config.dt
    )
    repeated_friction = api.evaluate_friction(
        positions + 1.0e-5 * frame.tangent,
        frame,
        repeated_contact.vessel_normal_force,
        repeated_contact.active,
        lengths,
        context,
        config,
    )
    torch.testing.assert_close(
        repeated_contact.total_force, baseline_contact.total_force, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        repeated_friction.force, baseline_friction.force, rtol=0.0, atol=0.0
    )


def test_contact_frame_constructor_rejects_invalid_shapes_types_and_basis():
    api = _api()
    valid = dict(
        q=torch.tensor([0.2, 0.4], dtype=torch.float64),
        center=torch.zeros((2, 2), dtype=torch.float64),
        tangent=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        normal=torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64),
        region=torch.tensor([True, False]),
    )
    invalid = (
        ({**valid, "q": valid["q"].float()}, TypeError, "float64"),
        ({**valid, "region": valid["region"].to(torch.float64)}, TypeError, "bool"),
        ({**valid, "center": torch.zeros((2, 3), dtype=torch.float64)}, ValueError, "shape"),
        ({**valid, "q": torch.tensor([0.2, np.nan], dtype=torch.float64)}, ValueError, "finite"),
        ({**valid, "tangent": 2.0 * valid["tangent"]}, ValueError, "unit"),
        ({**valid, "normal": 2.0 * valid["normal"]}, ValueError, "unit"),
        ({**valid, "normal": valid["tangent"].clone()}, ValueError, "orthogonal"),
    )

    for values, error, message in invalid:
        with pytest.raises(error, match=message):
            api.SineContactFrame(**values)


def test_external_force_blocks_match_direct_force_and_finite_difference():
    api = _api()
    config = _tiny_config(friction_mu=100.0, friction_kt_line=500.0)
    tube = api.SineTube2D(config.amplitude, config.x_span, config.tube_half_width)
    q = np.array([0.25, 0.65], dtype=np.float64)
    positions = torch.as_tensor(
        tube.centerline(q) + 0.045 * tube.normal(q), dtype=torch.float64
    )
    frame = api.freeze_contact_frames(positions, tube)
    lengths = torch.tensor([0.08, 0.12], dtype=torch.float64)
    zeros = torch.zeros(2, dtype=torch.float64)
    contact = api.contact_quantities(
        positions, zeros, zeros, lengths, frame, config, config.dt
    )
    history = api.initialize_friction_history(positions, frame, contact.active)
    context = api.begin_friction_increment(
        history, positions - 1.0e-5 * frame.tangent
    )

    force, blocks = api._external_force_and_blocks(
        positions, frame, zeros, zeros, lengths, context, config, config.dt, True
    )
    friction = api.evaluate_friction(
        positions, frame, contact.vessel_normal_force, contact.active,
        lengths, context, config,
    )
    torch.testing.assert_close(
        force, contact.total_force + friction.force, rtol=2.0e-12, atol=2.0e-12
    )

    step = 1.0e-7
    finite_difference = torch.empty_like(blocks)
    for node in range(2):
        for coordinate in range(2):
            offset = torch.zeros_like(positions)
            offset[node, coordinate] = step
            plus = api._external_force_and_blocks(
                positions + offset, frame, zeros, zeros, lengths,
                context, config, config.dt, True,
            )[0][node]
            minus = api._external_force_and_blocks(
                positions - offset, frame, zeros, zeros, lengths,
                context, config, config.dt, True,
            )[0][node]
            finite_difference[node, :, coordinate] = (plus - minus) / (2.0 * step)
    torch.testing.assert_close(blocks, finite_difference, rtol=3.0e-5, atol=2.0e-5)


def test_solver_defaults_validation_cli_and_cpp_boundary():
    api = _api()
    defaults = api.SineInsertConfig()

    assert defaults.backend == "cpp"
    assert defaults.damping_ratio == pytest.approx(1.0)
    assert defaults.max_newton_iters == 35
    assert defaults.residual_tol == pytest.approx(1.0e-7)
    assert defaults.step_tol == pytest.approx(1.0e-10)
    assert defaults.settle_speed_tol == pytest.approx(1.0e-3)
    assert defaults.settle_residual_tol == pytest.approx(1.0e-6)
    assert defaults.contact_tolerance == pytest.approx(5.0e-4)
    assert defaults.settle_confirm_steps == 4
    assert defaults.max_settle_steps == 1600
    assert defaults.prebalance
    assert defaults.prebalance_max_steps == 1600
    assert defaults.ramp_steps == 25
    assert defaults.history_save_interval == 20
    parsed = api.parse_args(["--ramp-steps", "3", "--no-prebalance", "--retry-subdivisions", "2", "3"])
    assert parsed.ramp_steps == 3
    assert not parsed.prebalance
    assert parsed.retry_subdivisions == (2, 3)

    for kwargs in (
        {"wire_length": True},
        {"contact_kp_line": np.bool_(True)},
        {"damping_ratio": 0.0},
        {"max_newton_iters": 0},
        {"settle_confirm_steps": True},
        {"retry_subdivisions": (1,)},
        {"kp_grow_factor": 1.0},
        {"ki_reduction_factor": 1.0},
    ):
        with pytest.raises(ValueError):
            api.SineInsertConfig(**kwargs)

    cpp_result = api.run_simulation(dataclasses.replace(
        _tiny_config(), backend="cpp", prebalance=False,
        total_insertion=1.0e-5, max_settle_steps=3,
    ))
    assert cpp_result.config.backend == "cpp"


def test_smooth_proximal_ramp_is_exact_and_tip_remains_free():
    api = _api()
    initial = np.array(
        [[-0.30, 0.225], [-0.20, 0.225], [-0.10, 0.225], [0.0, 0.225], [0.02, 0.22]],
        dtype=np.float64,
    )

    start = api.proximal_ramp_state(initial, 0.03, 0.09, 0, 4, amplitude=0.225)
    middle = api.proximal_ramp_state(initial, 0.03, 0.09, 2, 4, amplitude=0.225)
    end = api.proximal_ramp_state(initial, 0.03, 0.09, 4, 4, amplitude=0.225)

    np.testing.assert_allclose(start.positions[:2], initial[:2] + [0.03, 0.0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(middle.positions[:2], initial[:2] + [0.06, 0.0], atol=1.0e-15)
    np.testing.assert_allclose(end.positions[:2], initial[:2] + [0.09, 0.0], rtol=0.0, atol=0.0)
    assert not end.mask[-1].any()
    np.testing.assert_allclose(end.positions[-1], initial[-1], rtol=0.0, atol=0.0)


def test_newton_iteration_reuses_one_frozen_frame_for_jacobian_and_line_search():
    config = _tiny_config(max_newton_iters=1, residual_tol=1.0e-16, prebalance=False)
    (
        api, tube, initial, p, v, integrals, errors, rest, reference, masses, nodal,
        history, context, mask, prescribed,
    ) = _step_inputs(config)
    events = []

    result = api.implicit_euler_step(
        p, v, rest, reference, masses, integrals, errors, nodal, mask, prescribed,
        tube, config, context, friction_enabled=True,
        evaluation_observer=lambda iteration, stage, frame: events.append((iteration, stage, frame)),
    )

    iteration_events = [event for event in events if event[0] == 1]
    assert {event[1] for event in iteration_events} >= {"jacobian", "line_search"}
    assert len({id(event[2]) for event in iteration_events}) == 1
    assert result.info.iterations == 1


def test_implicit_step_does_not_mutate_committed_friction_or_increment_context():
    config = _tiny_config(max_newton_iters=2, residual_tol=1.0e-12, prebalance=False)
    (
        api, tube, initial, p, v, integrals, errors, rest, reference, masses, nodal,
        history, context, mask, prescribed,
    ) = _step_inputs(config)
    history_before = tuple(value.clone() for value in dataclasses.astuple(history))
    context_before = tuple(value.clone() for value in dataclasses.astuple(context))

    api.implicit_euler_step(
        p, v, rest, reference, masses, integrals, errors, nodal, mask, prescribed,
        tube, config, context, friction_enabled=True,
    )

    for actual, expected in zip(dataclasses.astuple(history), history_before):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    for actual, expected in zip(dataclasses.astuple(context), context_before):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_prebalance_is_frictionless_then_initializes_zero_slip_history():
    api = _api()
    result = api.run_simulation(
        _tiny_config(total_insertion=1.0e-5, friction_mu=0.3, settle_speed_tol=1.0)
    )

    prebalance = [record for record in result.history if record.phase == "prebalance"]
    assert prebalance
    assert prebalance[-1].settled
    torch.testing.assert_close(
        prebalance[-1].friction_force,
        torch.zeros_like(prebalance[-1].friction_force),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.prebalanced_state.friction.elastic_slip,
        torch.zeros_like(result.prebalanced_state.friction.elastic_slip),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.prebalanced_state.friction.anchors,
        result.prebalanced_state.positions,
        rtol=0.0,
        atol=0.0,
    )


def test_short_mu_zero_run_settles_advances_tip_records_complete_finite_state():
    api = _api()
    result = api.run_simulation(_tiny_config(friction_mu=0.0, settle_speed_tol=1.0))

    assert result.success
    assert result.completed_load_steps == 1
    assert len(result.settled_records) >= 2  # accepted prebalance and load states
    assert result.final_state.insertion == pytest.approx(2.0e-4)
    assert result.final_state.positions[-1, 0] > result.prebalanced_state.positions[-1, 0]
    assert result.history[-1].settled
    assert result.history[-1].phase == "settled"
    for record in result.history:
        torch.testing.assert_close(
            record.friction_force, torch.zeros_like(record.friction_force), rtol=0.0, atol=0.0
        )
        for value in (
            record.positions, record.velocities, record.normal_force, record.feeder_force,
            record.friction_force, record.integrals, record.xi,
        ):
            assert torch.isfinite(value).all()
        assert np.isfinite(record.tip_s)
        assert np.isfinite(record.elastic_total)


@pytest.mark.parametrize("friction_mu", [0.0, 0.05, 0.10, 0.20])
def test_python_reference_100_node_short_run_settles_for_mu_sweep(friction_mu):
    """M5 gate: all four campaign coefficients settle on a 100-node short run."""
    api = _api()
    result = api.run_simulation(
        _tiny_config(
            backend="python",
            num_nodes=100,
            total_insertion=6.0e-5,
            friction_mu=friction_mu,
            prebalance=False,
            ramp_steps=1,
            max_settle_steps=3,
            settle_speed_tol=1.0,
        )
    )

    assert result.success
    assert result.completed_load_steps == 1
    assert len(result.settled_records) == 1


def test_failed_increment_rolls_back_all_state_time_history_and_friction_exactly():
    api = _api()
    result = api.run_simulation(
        _tiny_config(
            max_settle_steps=1,
            settle_confirm_steps=2,
            retry_subdivisions=(),
            ki_reduction_factor=0.5,
            max_ki_retries=0,
            max_kp_retries=0,
        )
    )

    assert not result.success
    assert result.completed_load_steps == 0
    assert result.abort_reason
    assert result.history == result.history[: result.prebalance_history_length]
    assert result.final_state.time == result.prebalanced_state.time
    assert result.final_state.insertion == result.prebalanced_state.insertion == 0.0
    for name in ("positions", "velocities", "integrals", "previous_errors"):
        torch.testing.assert_close(
            getattr(result.final_state, name), getattr(result.prebalanced_state, name), rtol=0.0, atol=0.0
        )
    for actual, expected in zip(
        dataclasses.astuple(result.final_state.friction),
        dataclasses.astuple(result.prebalanced_state.friction),
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_failed_prebalance_rolls_back_to_initial_state_and_discards_records():
    api = _api()
    config = _tiny_config(prebalance_max_steps=1, settle_confirm_steps=2)
    tube = api.SineTube2D(config.amplitude, config.x_span, config.tube_half_width)
    initial = api.initialize_rod(config, tube)

    result = api.run_simulation(config)

    assert not result.success
    assert result.completed_load_steps == 0
    assert result.final_state.time == 0.0
    assert result.final_state.insertion == 0.0
    assert result.history == ()
    torch.testing.assert_close(
        result.final_state.positions,
        torch.as_tensor(initial.positions, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.final_state.friction.anchors,
        result.final_state.positions,
        rtol=0.0,
        atol=0.0,
    )


def test_runtime_tip_outlet_guard_uses_centerline_arclength():
    api = _api()
    config = _tiny_config()
    tube = api.SineTube2D(config.amplitude, config.x_span, config.tube_half_width)
    q = tube.parameter_at_arclength(tube.vessel_length - 0.5 * config.outlet_margin)
    positions = torch.as_tensor(
        np.vstack((tube.centerline(0.0), tube.centerline(q))), dtype=torch.float64
    )

    with pytest.raises(RuntimeError, match="outlet"):
        api.guard_tip_before_outlet(positions, tube, config)
