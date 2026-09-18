from dataclasses import replace

import numpy as np
import pytest
import torch

import der_cpp_backend as backend
import rod_soft_contact_PI_dynamic_quasi_static_2d_sine_insert as sine


def _config(backend_name="python", **overrides):
    values = dict(
        backend=backend_name,
        num_nodes=12,
        wire_length=0.30,
        initial_inserted_length=0.05,
        total_insertion=2.0e-5,
        num_load_steps=1,
        young_modulus=2.0e6,
        contact_kp_line=2.0e3,
        contact_ki_line=0.0,
        feeder_kp_line=2.0e3,
        friction_mu=0.15,
        friction_kt_line=2.0e3,
        dt=0.02,
        ramp_steps=1,
        max_newton_iters=12,
        residual_tol=1.0e-9,
        step_tol=1.0e-12,
        settle_speed_tol=1.0,
        settle_residual_tol=1.0e-6,
        contact_tolerance=2.0e-3,
        settle_confirm_steps=1,
        max_settle_steps=2,
        prebalance=False,
        history_save_interval=1,
        max_kp_retries=0,
        max_ki_retries=0,
        retry_subdivisions=(),
    )
    values.update(overrides)
    return sine.SineInsertConfig(**values)


def _inputs(config):
    tube = sine.SineTube2D(
        config.amplitude, config.x_span, config.tube_half_width
    )
    initial = sine.initialize_rod(config, tube)
    tensors = {
        name: torch.as_tensor(getattr(initial, name), dtype=torch.float64)
        for name in (
            "positions", "velocities", "rest_lengths", "reference_curvature",
            "masses", "integrals", "previous_errors", "nodal_lengths",
        )
    }
    frame = sine.freeze_contact_frames(tensors["positions"], tube)
    contact = sine.contact_quantities(
        tensors["positions"], tensors["integrals"], tensors["previous_errors"],
        tensors["nodal_lengths"], frame, config, config.dt,
    )
    history = sine.initialize_friction_history(
        tensors["positions"], frame, contact.active
    )
    context = sine.begin_friction_increment(history, tensors["positions"])
    prescription = sine.proximal_prescribed_state(
        initial.positions, 1.0e-5, config.amplitude
    )
    return (
        tube, initial, tensors, frame, context,
        torch.as_tensor(prescription.mask, dtype=torch.bool),
        torch.as_tensor(prescription.positions, dtype=torch.float64),
    )


def test_native_cpp_implicit_step_matches_python_reference():
    python_config = _config("python")
    tube, initial, values, frame, context, mask, prescribed = _inputs(python_config)
    expected = sine.implicit_euler_step(
        values["positions"], values["velocities"], values["rest_lengths"],
        values["reference_curvature"], values["masses"], values["integrals"],
        values["previous_errors"], values["nodal_lengths"], mask, prescribed,
        tube, python_config, context, friction_enabled=True,
        accepted_frame=frame,
    )

    cpp_config = replace(python_config, backend="cpp")
    actual = sine.implicit_euler_step(
        values["positions"], values["velocities"], values["rest_lengths"],
        values["reference_curvature"], values["masses"], values["integrals"],
        values["previous_errors"], values["nodal_lengths"], mask, prescribed,
        tube, cpp_config, context, friction_enabled=True,
        accepted_frame=frame,
    )

    torch.testing.assert_close(actual.positions, expected.positions, rtol=0.0, atol=1e-9)
    for field in (
        "velocities", "integrals", "previous_errors", "normal_force",
        "feeder_force", "friction_force", "dynamic_force",
    ):
        torch.testing.assert_close(
            getattr(actual, field), getattr(expected, field), rtol=1e-8, atol=1e-10
        )
    torch.testing.assert_close(actual.frame.q, expected.frame.q, rtol=0.0, atol=1e-10)
    torch.testing.assert_close(
        actual.friction_result.candidate_slip,
        expected.friction_result.candidate_slip,
        rtol=1e-8,
        atol=1e-12,
    )
    assert actual.info.converged == expected.info.converged
    assert actual.info.iterations == expected.info.iterations


def test_cpp_frozen_frame_force_jacobian_matches_python_autodiff():
    config = _config("python", friction_mu=10.0, friction_kt_line=700.0)
    tube, initial, values, frame, context, _mask, _prescribed = _inputs(config)
    positions = values["positions"].clone()
    q = frame.q[-1].item()
    positions[-1] = torch.as_tensor(
        tube.centerline(q) + 0.045 * tube.normal(q), dtype=torch.float64
    )
    frame = sine.freeze_contact_frames(positions, tube)
    context = sine.FrictionIncrementContext(
        elastic_slip=torch.zeros(config.num_nodes, dtype=torch.float64),
        anchors=positions - 1.0e-5 * frame.tangent,
        q=frame.q,
        contact_active=torch.ones(config.num_nodes, dtype=torch.bool),
    )
    expected_force, expected_jacobian = sine._external_force_and_blocks(
        positions, frame, values["integrals"], values["previous_errors"],
        values["nodal_lengths"], context, config, config.dt, True,
    )
    ea = config.young_modulus * np.pi * config.diameter**2 / 4.0
    ei = config.young_modulus * np.pi * config.diameter**4 / 64.0
    total_mass = float(values["masses"].sum().item())
    damping = 2.0 * config.damping_ratio * np.sqrt(
        (3.0 * ei / config.wire_length**3) * total_mass
    )
    core = backend.Sine2DCore(
        values["rest_lengths"], values["masses"], ea, ei, damping,
        config.amplitude, config.x_span, config.tube_half_width,
        config.safe_distance, config.feeder_half_width,
        config.feeder_contact_layer, config.feeder_kp_line,
        config.contact_kp_line, config.contact_ki_line,
        config.contact_force_cap_line, config.friction_mu,
        config.friction_kt_line, config.max_integral_state,
        config.integral_leak_rate, config.contact_smooth_length,
        config.direction_epsilon,
    )
    actual = core.contact_kernel(
        positions, values["integrals"], values["previous_errors"],
        values["nodal_lengths"], context.elastic_slip, context.anchors,
        context.contact_active, config.dt, True, initial_q=frame.q,
    )

    np.testing.assert_allclose(
        actual["total_contact"], expected_force.numpy(), rtol=1e-8, atol=1e-11
    )
    np.testing.assert_allclose(
        actual["force_jacobian"], expected_jacobian.numpy(), rtol=1e-8, atol=1e-9
    )


def test_cpp_backend_runs_a_real_short_simulation_without_python_newton(monkeypatch):
    config = _config("cpp", friction_mu=0.0)
    monkeypatch.setattr(
        sine, "_residual",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("C++ backend called the Python Newton residual")
        ),
    )
    result = sine.run_simulation(config)

    assert result.config.backend == "cpp"
    assert result.success
    assert result.completed_load_steps == 1
    assert result.settled_records


def test_short_run_python_cpp_insertion_force_curve_relative_l2():
    common = dict(
        friction_mu=0.10, num_load_steps=3, total_insertion=6.0e-5,
        max_settle_steps=5,
    )
    python = sine.run_simulation(_config("python", **common))
    cpp = sine.run_simulation(_config("cpp", **common))
    assert python.success and cpp.success
    python_force = np.array(
        [record.reaction.insertion_force for record in python.settled_records]
    )
    cpp_force = np.array(
        [record.reaction.insertion_force for record in cpp.settled_records]
    )
    assert python_force.shape == cpp_force.shape == (3,)
    relative_l2 = np.linalg.norm(cpp_force - python_force) / max(
        np.linalg.norm(python_force), 1.0e-30
    )
    assert relative_l2 <= 1.0e-6


def test_cpp_facade_rejects_non_proximal_prescription():
    config = _config("cpp")
    tube, initial, values, frame, context, mask, prescribed = _inputs(config)
    bad_mask = mask.clone()
    bad_mask[-1] = True

    with pytest.raises(ValueError, match="proximal"):
        sine.implicit_euler_step(
            values["positions"], values["velocities"], values["rest_lengths"],
            values["reference_curvature"], values["masses"], values["integrals"],
            values["previous_errors"], values["nodal_lengths"], bad_mask,
            prescribed, tube, config, context, friction_enabled=True,
            accepted_frame=frame,
        )


def test_failed_line_search_recomputes_forced_step_residual_before_convergence(monkeypatch):
    config = _config(
        "python", max_newton_iters=1, residual_tol=1.0e-8,
        step_tol=1.0e-3,
    )
    tube, initial, values, frame, context, mask, prescribed = _inputs(config)

    class FakeSystem:
        def solve(self, right_hand_side):
            return torch.full_like(right_hand_side, 1.0e-6)

        def dense(self):
            raise AssertionError("dense fallback should not be used")

    evaluations = []

    def fake_residual(free, *args, **kwargs):
        with_jacobian = bool(args[-1])
        evaluations.append((with_jacobian, free.clone()))
        norm = 5.0e-8 if with_jacobian else 1.2e-7
        residual = torch.full_like(free, norm / np.sqrt(free.numel()))
        return residual, FakeSystem() if with_jacobian else None, prescribed

    monkeypatch.setattr(sine, "_residual", fake_residual)
    step = sine.implicit_euler_step(
        values["positions"], values["velocities"], values["rest_lengths"],
        values["reference_curvature"], values["masses"], values["integrals"],
        values["previous_errors"], values["nodal_lengths"], mask, prescribed,
        tube, config, context, friction_enabled=True, accepted_frame=frame,
    )

    # Ten rejected trial scales plus one explicit evaluation of the forced 5% step.
    assert [flag for flag, _ in evaluations].count(False) == 11
    assert not step.info.converged
    assert step.info.residual_norm == pytest.approx(1.2e-7)
    assert step.info.line_search_failures == 1


def test_cpp_diagnostics_keep_dense_fallback_separate_from_line_search_failures():
    config = _config("cpp")
    tube, initial, values, frame, context, _mask, prescribed = _inputs(config)
    ea = config.young_modulus * np.pi * config.diameter**2 / 4.0
    ei = config.young_modulus * np.pi * config.diameter**4 / 64.0
    damping = 2.0 * config.damping_ratio * np.sqrt(
        (3.0 * ei / config.wire_length**3) * float(values["masses"].sum())
    )
    core = backend.Sine2DCore(
        values["rest_lengths"], values["masses"], ea, ei, damping,
        config.amplitude, config.x_span, config.tube_half_width,
        config.safe_distance, config.feeder_half_width,
        config.feeder_contact_layer, config.feeder_kp_line,
        config.contact_kp_line, config.contact_ki_line,
        config.contact_force_cap_line, config.friction_mu,
        config.friction_kt_line, config.max_integral_state,
        config.integral_leak_rate, config.contact_smooth_length,
        config.direction_epsilon,
    )
    raw = core.implicit_euler_step(
        values["positions"], values["velocities"], values["integrals"],
        values["previous_errors"], values["nodal_lengths"], prescribed,
        values["reference_curvature"], context.elastic_slip, context.anchors,
        context.contact_active, True, frame.q, config.dt,
        config.max_newton_iters, config.residual_tol, config.step_tol,
        config.projection_trust_radius,
    )

    assert raw["line_search_failures"] >= 0
    assert raw["dense_fallback_count"] >= 0
    assert set(("line_search_failures", "dense_fallback_count")) <= set(raw)


def test_python_dense_solve_fallback_is_not_reported_as_line_search_failure(monkeypatch):
    config = _config(
        "python", max_newton_iters=1, residual_tol=1.0e-12,
        step_tol=1.0e-3,
    )
    tube, initial, values, frame, context, mask, prescribed = _inputs(config)

    class FallbackSystem:
        def solve(self, right_hand_side):
            raise RuntimeError("synthetic banded solve failure")

        def dense(self):
            size = 2 * (config.num_nodes - 2)
            return torch.eye(size, dtype=torch.float64)

    def fake_residual(free, *args, **kwargs):
        with_jacobian = bool(args[-1])
        norm = 1.0e-8 if with_jacobian else 1.0e-13
        residual = torch.full_like(free, norm / np.sqrt(free.numel()))
        return residual, FallbackSystem() if with_jacobian else None, prescribed

    monkeypatch.setattr(sine, "_residual", fake_residual)
    step = sine.implicit_euler_step(
        values["positions"], values["velocities"], values["rest_lengths"],
        values["reference_curvature"], values["masses"], values["integrals"],
        values["previous_errors"], values["nodal_lengths"], mask, prescribed,
        tube, config, context, friction_enabled=True, accepted_frame=frame,
    )

    assert step.info.line_search_failures == 0
    assert step.info.dense_fallback_count == 1


def test_native_cpp_failed_line_search_uses_forced_candidate_residual():
    positions = np.array(
        [[-0.2, 0.225], [-0.1, 0.225],
         [0.76898479501885, 0.6702776414844283]], dtype=np.float64,
    )
    anchors = np.array(
        [[-0.2, 0.225], [-0.1, 0.225],
         [1.1078950719393643, 0.920779419877624]], dtype=np.float64,
    )
    velocities = np.array(
        [[0.0, 0.0], [0.0, 0.0],
         [41.37877193355476, -64.38925514938686]], dtype=np.float64,
    )
    integrals = np.array(
        [0.023487660478199907, 0.004156289071513482,
         0.0011905765245629782], dtype=np.float64,
    )
    errors = np.array(
        [0.012158081091462396, 0.006566656001910643,
         0.014113105550284763], dtype=np.float64,
    )
    committed_slip = np.array(
        [0.015422604748851237, 0.08482737365546261,
         -0.021695742164277804], dtype=np.float64,
    )
    committed_contact = np.array([True, False, True])
    accepted_q = np.array([0.0, 0.0, 0.9124374096558872], dtype=np.float64)
    rest = np.array([0.1, 0.1], dtype=np.float64)
    masses = np.full(3, 0.01, dtype=np.float64)
    nodal = np.full(3, 0.1, dtype=np.float64)
    reference = np.zeros(1, dtype=np.float64)
    dt = 0.17634597738548513
    diameter = np.sqrt(0.016)
    young_modulus = 40.0 / (np.pi * diameter**2)
    config = replace(
        sine.SineInsertConfig(), backend="python", diameter=diameter,
        young_modulus=young_modulus, contact_kp_line=1.0e9,
        contact_ki_line=3.0e4, contact_force_cap_line=1.0e10,
        friction_mu=2.0, friction_kt_line=1.0e9,
        feeder_kp_line=1.0e9, dt=dt,
    )
    tube = sine.SineTube2D()
    previous_frame = sine.SineContactFrame(
        q=torch.as_tensor(accepted_q),
        center=torch.as_tensor(tube.centerline(accepted_q)),
        tangent=torch.as_tensor(tube.tangent(accepted_q)),
        normal=torch.as_tensor(tube.normal(accepted_q)),
        region=torch.as_tensor(positions[:, 0] >= 0.0),
    )
    predicted = torch.as_tensor(positions + dt * velocities)
    predicted[:2] = torch.as_tensor(positions[:2])
    frozen_frame = sine._tracked_frame(predicted, tube, previous_frame, config)
    context = sine.FrictionIncrementContext(
        elastic_slip=torch.as_tensor(committed_slip),
        anchors=torch.as_tensor(anchors), q=torch.as_tensor(accepted_q),
        contact_active=torch.as_tensor(committed_contact),
    )
    core = backend.Sine2DCore(
        rest, masses, 10.0, 0.01, 0.1, config.amplitude, config.x_span,
        config.tube_half_width, config.safe_distance,
        config.feeder_half_width, config.feeder_contact_layer, 1.0e9,
        1.0e9, 3.0e4, 1.0e10, 2.0, 1.0e9,
        config.max_integral_state, config.integral_leak_rate,
        config.contact_smooth_length, config.direction_epsilon,
    )
    raw = core.implicit_euler_step(
        positions, velocities, integrals, errors, nodal, positions, reference,
        committed_slip, anchors, committed_contact, True, accepted_q, dt, 1,
        1.23e8, 1.0e9, config.projection_trust_radius,
    )
    reported_positions = torch.as_tensor(raw["positions"])
    reported_residual, _, _ = sine._residual(
        reported_positions[2:].reshape(-1), torch.as_tensor(positions),
        torch.as_tensor(velocities), torch.as_tensor(rest),
        torch.as_tensor(reference), torch.as_tensor(masses),
        torch.as_tensor(integrals), torch.as_tensor(errors),
        torch.as_tensor(nodal), torch.as_tensor(positions), frozen_frame,
        context, config, dt, 0.1, True, False,
    )

    assert raw["line_search_failures"] == 1
    assert raw["dense_fallback_count"] == 0
    assert not raw["converged"]
    assert raw["residual_norm"] == pytest.approx(
        torch.linalg.vector_norm(reported_residual).item(), rel=1.0e-12
    )
