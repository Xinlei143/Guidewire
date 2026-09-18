from dataclasses import replace

import numpy as np
import pytest
import torch

import der_cpp_backend as backend
import rod_soft_contact_PI_dynamic_quasi_static_2d_sine_insert as sine
from sine_tube_2d import SineTube2D


def test_sine2d_cpp_backend_exposes_core_wrapper():
    assert hasattr(backend, "Sine2DCore")


def _state(config=None, nodes=7):
    if nodes < 3:
        raise ValueError("test cores require at least three nodes")
    config = sine.SineInsertConfig() if config is None else config
    rest = np.linspace(0.08, 0.14, nodes - 1, dtype=np.float64)
    masses = np.linspace(0.01, 0.025, nodes, dtype=np.float64)
    return config, rest, masses


def _core(config=None, nodes=7):
    config, rest, masses = _state(config, nodes)
    core = backend.Sine2DCore(
        rest,
        masses,
        4.0,
        0.015,
        0.12,
        config.amplitude,
        config.x_span,
        config.tube_half_width,
        config.safe_distance,
        config.feeder_half_width,
        config.feeder_contact_layer,
        config.feeder_kp_line,
        config.contact_kp_line,
        config.contact_ki_line,
        config.contact_force_cap_line,
        config.friction_mu,
        config.friction_kt_line,
        config.max_integral_state,
        config.integral_leak_rate,
        config.contact_smooth_length,
        config.direction_epsilon,
    )
    return core, rest, masses


def _geometry_reference(points, config, initial_q=None):
    tube = SineTube2D(config.amplitude, config.x_span, config.tube_half_width)
    projection = tube.project(points, initial_q=initial_q)
    return {
        "centerline": np.asarray(projection.point),
        "tangent": np.asarray(projection.tangent),
        "normal": np.asarray(projection.normal),
        "curvature": np.asarray(tube.curvature(projection.q)),
        "q": np.asarray(projection.q),
        "s": np.asarray(tube.arclength(projection.q)),
        "signed_distance": np.asarray(projection.signed_distance),
        "region": np.asarray(points)[:, 0] >= 0.0,
    }


def _assert_geometry(actual, expected):
    for key, value in expected.items():
        if np.asarray(value).dtype == np.bool_:
            assert np.array_equal(actual[key], value)
        else:
            np.testing.assert_allclose(actual[key], value, rtol=0.0, atol=1.0e-10)


@pytest.mark.parametrize(
    "points,initial_q",
    [
        (
            np.array(
                [[0.0, 0.225], [0.32, -0.03], [0.75, 0.0], [1.5, -0.225]],
                dtype=np.float64,
            ),
            None,
        ),
        (
            np.array(
                [
                    [-0.4, 0.8],
                    [0.01, -0.9],
                    [0.37, 0.62],
                    [0.76, -0.62],
                    [1.12, 0.60],
                    [1.49, -0.9],
                    [1.9, 0.8],
                ],
                dtype=np.float64,
            ),
            np.array([1.5, 0.0, 1.48, 0.02, 0.02, 0.0, 1.5], dtype=np.float64),
        ),
    ],
)
def test_geometry_kernel_matches_global_python_projection_for_defaults_and_adversarial_starts(
    points, initial_q
):
    config = sine.SineInsertConfig()
    core, *_ = _core(config, nodes=len(points))

    actual = core.geometry_kernel(points, initial_q=initial_q)
    expected = _geometry_reference(points, config, initial_q=initial_q)

    _assert_geometry(actual, expected)
    np.testing.assert_array_equal(actual["center"], actual["centerline"])


def test_geometry_kernel_random_batch_matches_python_to_ten_decimal_places():
    config = sine.SineInsertConfig()
    rng = np.random.default_rng(20260908)
    points = rng.uniform([-0.3, -0.8], [1.8, 0.8], size=(31, 2)).astype(np.float64)
    starts = rng.uniform(-2.0, 3.0, size=len(points)).astype(np.float64)
    core, *_ = _core(config, nodes=len(points))

    actual = core.geometry_kernel(points, initial_q=starts)
    expected = _geometry_reference(points, config, initial_q=starts)

    _assert_geometry(actual, expected)


def test_cpp_tracked_projection_rejects_stationary_but_non_global_cached_lobe():
    config = sine.SineInsertConfig()
    core, *_ = _core(config, nodes=3)
    point = np.repeat(
        np.array([[0.772988341563213, -0.29987806787660176]], dtype=np.float64),
        3,
        axis=0,
    )
    wrong_lobe_q = np.full(3, 1.2312006705233192, dtype=np.float64)
    tube = SineTube2D(config.amplitude, config.x_span, config.tube_half_width)
    expected = tube.project(point, initial_q=wrong_lobe_q)

    actual = core.tracked_geometry_kernel(point, wrong_lobe_q, 0.02)

    assert np.all(actual["used_global_fallback"])
    np.testing.assert_allclose(actual["q"], expected.q, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(
        actual["centerline"], expected.point, rtol=0.0, atol=1.0e-10
    )
    np.testing.assert_allclose(
        actual["tangent"], expected.tangent, rtol=0.0, atol=1.0e-10
    )
    np.testing.assert_allclose(
        actual["normal"], expected.normal, rtol=0.0, atol=1.0e-10
    )


def test_sine2d_wrapper_and_constructor_reject_invalid_arrays_and_physics():
    config, rest, masses = _state()
    with pytest.raises((TypeError, ValueError), match="float64"):
        backend.Sine2DCore(
            rest.astype(np.float32), masses, 4.0, 0.015, 0.12,
            config.amplitude, config.x_span, config.tube_half_width,
            config.safe_distance, config.feeder_half_width,
            config.feeder_contact_layer, config.feeder_kp_line,
            config.contact_kp_line, config.contact_ki_line,
            config.contact_force_cap_line, config.friction_mu,
            config.friction_kt_line, config.max_integral_state,
            config.integral_leak_rate, config.contact_smooth_length,
            config.direction_epsilon,
        )
    with pytest.raises(ValueError, match="positive"):
        bad = rest.copy()
        bad[1] = 0.0
        backend.Sine2DCore(
            bad, masses, 4.0, 0.015, 0.12, config.amplitude,
            config.x_span, config.tube_half_width, config.safe_distance,
            config.feeder_half_width, config.feeder_contact_layer,
            config.feeder_kp_line, config.contact_kp_line,
            config.contact_ki_line, config.contact_force_cap_line,
            config.friction_mu, config.friction_kt_line,
            config.max_integral_state, config.integral_leak_rate,
            config.contact_smooth_length, config.direction_epsilon,
        )
    with pytest.raises(ValueError, match="non-negative"):
        _core(replace(config, friction_mu=-0.1))

    core, *_ = _core(config)
    good = np.zeros((len(masses), 2), dtype=np.float64)
    with pytest.raises(ValueError, match="C-contiguous"):
        core.geometry_kernel(good[::-1])
    with pytest.raises(TypeError, match="float64"):
        core.geometry_kernel(good.astype(np.float32))
    with pytest.raises(ValueError, match="finite"):
        bad_points = good.copy()
        bad_points[0, 0] = np.nan
        core.geometry_kernel(bad_points)


def _python_contact(
    positions,
    old_integrals,
    previous_errors,
    nodal_lengths,
    committed_slip,
    increment_start_anchors,
    committed_contact,
    config,
    dt,
    friction_enabled,
    initial_q=None,
):
    tube = SineTube2D(config.amplitude, config.x_span, config.tube_half_width)
    projection = tube.project(positions, initial_q=initial_q)

    def tensor(value):
        return torch.as_tensor(np.asarray(value), dtype=torch.float64)

    frame = sine.SineContactFrame(
        q=tensor(projection.q),
        center=tensor(projection.point),
        tangent=tensor(projection.tangent),
        normal=tensor(projection.normal),
        region=torch.as_tensor(positions[:, 0] >= 0.0, dtype=torch.bool),
    )
    contact = sine.contact_quantities(
        tensor(positions), tensor(old_integrals), tensor(previous_errors),
        tensor(nodal_lengths), frame, config, dt,
    )
    context = sine.FrictionIncrementContext(
        elastic_slip=tensor(committed_slip),
        anchors=tensor(increment_start_anchors),
        q=tensor(projection.q),
        contact_active=torch.as_tensor(committed_contact, dtype=torch.bool),
    )
    friction = sine.evaluate_friction(
        tensor(positions), frame, contact.vessel_normal_force, contact.active,
        tensor(nodal_lengths), context, config, enabled=friction_enabled,
    )
    return {
        "vessel_normal_force": contact.vessel_normal_force.numpy(),
        "feeder_force": contact.feeder_force.numpy(),
        "total_normal": contact.total_force.numpy(),
        "friction": friction.friction_force.numpy(),
        "total_contact": (contact.total_force + friction.friction_force).numpy(),
        "new_integrals": contact.new_integrals.numpy(),
        "active": contact.active.numpy(),
        "weight": contact.weight.numpy(),
        "error": contact.error.numpy(),
        "clearance": contact.clearance.numpy(),
        "q": contact.q.numpy(),
        "center": np.asarray(projection.point),
        "tangent": contact.tangent.numpy(),
        "normal": contact.normal.numpy(),
        "region": contact.region.numpy(),
        "signed_distance": contact.signed_distance.numpy(),
        "trial_slip": friction.trial_slip.numpy(),
        "candidate_slip": friction.candidate_slip.numpy(),
        "traction": friction.tangent_traction.numpy(),
        "stick": friction.stick.numpy(),
        "slip": friction.slip.numpy(),
        "limit": friction.coulomb_limit.numpy(),
    }


def _cpp_contact(
    core,
    positions,
    old_integrals,
    previous_errors,
    nodal_lengths,
    committed_slip,
    increment_start_anchors,
    committed_contact,
    config,
    *,
    friction_enabled=True,
    kp_active=None,
    ki_active=None,
    initial_q=None,
):
    return core.contact_kernel(
        positions,
        old_integrals,
        previous_errors,
        nodal_lengths,
        committed_slip,
        increment_start_anchors,
        committed_contact,
        config.dt,
        friction_enabled,
        kp_active=kp_active,
        ki_active=ki_active,
        initial_q=initial_q,
    )


def _assert_contact(actual, expected):
    for key, value in expected.items():
        if np.asarray(value).dtype == np.bool_:
            assert np.array_equal(actual[key], value), key
        else:
            np.testing.assert_allclose(actual[key], value, rtol=0.0, atol=1.0e-9, err_msg=key)


def test_contact_kernel_matches_current_python_normal_feeder_pi_and_friction_quantities():
    config = replace(sine.SineInsertConfig(), friction_mu=0.4, friction_kt_line=600.0)
    tube = SineTube2D(config.amplitude, config.x_span, config.tube_half_width)
    q = np.array([0.22, 0.48, 0.84, 1.18, 1.42], dtype=np.float64)
    center = tube.centerline(q)
    normal = tube.normal(q)
    tangent = tube.tangent(q)
    vessel = center + np.array([0.044, -0.043, 0.049, -0.047, 0.0])[:, None] * normal
    positions = np.vstack((np.array([[-0.3, config.amplitude + 0.00225]]), vessel))
    nodal = np.array([0.04, 0.05, 0.10, 0.10, 0.05, 0.04], dtype=np.float64)
    integrals = np.array([0.01, 0.0, 0.003, 0.002, 0.0, 0.004], dtype=np.float64)
    previous = np.array([0.0, 0.003, 0.0, 0.009, 0.0, 0.0], dtype=np.float64)
    committed_slip = np.array([0.3, 1e-4, -3e-4, 0.02, -0.02, 0.0], dtype=np.float64)
    anchors = positions.copy()
    anchors[1:] -= np.array([2e-4, -1e-4, 4e-3, -3e-3, 2e-4])[:, None] * tangent
    committed_contact = np.array([True, True, True, True, True, False])
    starts = np.linspace(1.5, 0.0, len(positions), dtype=np.float64)
    core, *_ = _core(config, nodes=len(positions))

    expected = _python_contact(
        positions, integrals, previous, nodal, committed_slip, anchors,
        committed_contact, config, config.dt, True, initial_q=starts,
    )
    actual = _cpp_contact(
        core, positions, integrals, previous, nodal, committed_slip, anchors,
        committed_contact, config, initial_q=starts,
    )

    _assert_contact(actual, expected)
    assert np.all(actual["feeder_force"][:, 0] == 0.0)
    np.testing.assert_array_equal(actual["total_normal"], actual["vessel_normal_force"] + actual["feeder_force"])
    np.testing.assert_array_equal(actual["total_contact"], actual["total_normal"] + actual["friction"])


def test_contact_pi_matches_both_antiwindup_branches_and_active_overrides():
    base = replace(
        sine.SineInsertConfig(), contact_kp_line=1.0,
        contact_ki_line=5.0, contact_force_cap_line=20.0,
        friction_mu=0.0,
    )
    tube = SineTube2D(base.amplitude, base.x_span, base.tube_half_width)
    q = np.array([0.3, 0.7, 1.1], dtype=np.float64)
    positions = tube.centerline(q) + 0.049 * tube.normal(q)
    nodal = np.array([0.1, 0.1, 0.1], dtype=np.float64)
    anchors = positions.copy()
    committed = np.ones(3, dtype=bool)

    cases = [
        (replace(base, contact_force_cap_line=0.009), np.zeros(3), np.zeros(3), None, None),
        (base, np.array([0.02, 0.02, 0.02]), np.array([1.0, 0.0, 0.0]), 1.0, 5.0),
    ]
    for config, integrals, previous, kp_active, ki_active in cases:
        core, *_ = _core(config, nodes=3)
        expected_config = config if kp_active is None else replace(
            config, contact_kp_line=kp_active, contact_ki_line=ki_active
        )
        expected = _python_contact(
            positions, integrals, previous, nodal, np.zeros(3), anchors,
            committed, expected_config, config.dt, False,
        )
        actual = _cpp_contact(
            core, positions, integrals, previous, nodal, np.zeros(3), anchors,
            committed, config, friction_enabled=False,
            kp_active=kp_active, ki_active=ki_active,
        )
        _assert_contact(actual, expected)
        assert actual["new_integrals"][0] == integrals[0]


def test_line_scaling_uses_endpoint_voronoi_lengths_and_feeder_stiffness_is_independent():
    config = replace(sine.SineInsertConfig(), contact_ki_line=0.0, friction_mu=0.0)
    tube = SineTube2D(config.amplitude, config.x_span, config.tube_half_width)
    q = np.array([0.35, 0.65, 0.95], dtype=np.float64)
    vessel = tube.centerline(q) + 0.045 * tube.normal(q)
    nodal = np.array([0.05, 0.10, 0.05], dtype=np.float64)
    zeros = np.zeros(3, dtype=np.float64)
    core, *_ = _core(config, nodes=3)
    actual = _cpp_contact(core, vessel, zeros, zeros, nodal, zeros, vessel, np.ones(3, dtype=bool), config)
    magnitude = np.linalg.norm(actual["vessel_normal_force"], axis=1)
    assert magnitude[0] + magnitude[2] == pytest.approx(magnitude[1], rel=2e-12)

    feeder = np.array([[-3.0, config.amplitude + 0.00225]] * 3, dtype=np.float64)
    length = np.full(3, 0.1, dtype=np.float64)
    base_core, *_ = _core(replace(config, feeder_kp_line=1000.0, contact_kp_line=50.0), nodes=3)
    vessel_core, *_ = _core(replace(config, feeder_kp_line=1000.0, contact_kp_line=50000.0), nodes=3)
    double_core, *_ = _core(replace(config, feeder_kp_line=2000.0, contact_kp_line=50.0), nodes=3)
    args = (feeder, np.zeros(3), np.zeros(3), length, np.zeros(3), feeder, np.ones(3, dtype=bool))
    first = _cpp_contact(base_core, *args, replace(config, feeder_kp_line=1000.0, contact_kp_line=50.0))
    changed = _cpp_contact(vessel_core, *args, replace(config, feeder_kp_line=1000.0, contact_kp_line=50000.0))
    doubled = _cpp_contact(double_core, *args, replace(config, feeder_kp_line=2000.0, contact_kp_line=50.0))
    assert first["feeder_force"][0, 0] == 0.0
    np.testing.assert_array_equal(changed["feeder_force"], first["feeder_force"])
    np.testing.assert_allclose(doubled["feeder_force"], 2.0 * first["feeder_force"], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "mu,kt,committed_slip,committed_contact,delta,friction_enabled",
    [
        (0.0, 1000.0, 0.02, True, 0.01, True),
        (10.0, 1000.0, 0.0, True, 1.0e-4, True),
        (0.2, 1000.0, 0.0, True, 0.1, True),
        (10.0, 100.0, 0.02, True, -0.005, True),
        (10.0, 1000.0, 0.02, False, 0.01, True),
        (10.0, 1000.0, 0.02, True, 0.01, False),
    ],
)
def test_friction_return_mapping_matches_mu0_stick_slip_reversal_first_contact_and_disabled(
    mu, kt, committed_slip, committed_contact, delta, friction_enabled
):
    config = replace(
        sine.SineInsertConfig(), friction_mu=mu, friction_kt_line=kt,
        contact_kp_line=1000.0, contact_ki_line=0.0,
    )
    tube = SineTube2D(config.amplitude, config.x_span, config.tube_half_width)
    q = 0.6
    point = tube.centerline(q) + 0.049 * tube.normal(q)
    tangent = tube.tangent(q)
    position = np.vstack((point, tube.centerline(0.9), tube.centerline(1.2)))
    anchor = position.copy()
    anchor[0] -= delta * tangent
    length = np.full(3, 0.1, dtype=np.float64)
    zeros = np.zeros(3, dtype=np.float64)
    core, *_ = _core(config, nodes=3)
    expected = _python_contact(
        position, zeros, zeros, length, np.array([committed_slip, 0.0, 0.0]), anchor,
        np.array([committed_contact, False, False]), config, config.dt, friction_enabled,
    )
    actual = _cpp_contact(
        core, position, zeros, zeros, length, np.array([committed_slip, 0.0, 0.0]), anchor,
        np.array([committed_contact, False, False]), config, friction_enabled=friction_enabled,
    )
    _assert_contact(actual, expected)


def test_contact_kernel_rejects_bad_state_without_mutating_inputs():
    config = sine.SineInsertConfig()
    core, _rest, masses = _core(config)
    count = len(masses)
    positions = np.zeros((count, 2), dtype=np.float64)
    vectors = [np.zeros(count, dtype=np.float64) for _ in range(4)]
    nodal = np.ones(count, dtype=np.float64)
    anchors = positions.copy()
    committed = np.zeros(count, dtype=bool)
    snapshots = [value.copy() for value in (positions, *vectors, nodal, anchors, committed)]

    with pytest.raises(ValueError, match="shape"):
        core.contact_kernel(
            positions, vectors[0][:-1], vectors[1], nodal, vectors[2], anchors,
            committed, config.dt, True,
        )
    with pytest.raises(TypeError, match="bool"):
        core.contact_kernel(
            positions, vectors[0], vectors[1], nodal, vectors[2], anchors,
            committed.astype(np.float64), config.dt, True,
        )
    with pytest.raises(ValueError, match="positive"):
        bad_nodal = nodal.copy()
        bad_nodal[0] = 0.0
        core.contact_kernel(
            positions, vectors[0], vectors[1], bad_nodal, vectors[2], anchors,
            committed, config.dt, True,
        )

    for value, snapshot in zip((positions, *vectors, nodal, anchors, committed), snapshots):
        np.testing.assert_array_equal(value, snapshot)
