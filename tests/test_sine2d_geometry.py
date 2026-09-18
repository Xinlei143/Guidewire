import numpy as np
import pytest


def _geometry(*args, **kwargs):
    from sine_tube_2d import SineTube2D

    return SineTube2D(*args, **kwargs)


def test_centerline_extrema_and_entry_frame():
    tube = _geometry()
    q = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)

    np.testing.assert_allclose(
        tube.centerline(q),
        [[0.0, 0.225], [0.5, -0.225], [1.0, 0.225], [1.5, -0.225]],
        atol=1.0e-14,
    )
    np.testing.assert_allclose(tube.derivative(0.0), [1.0, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(tube.tangent(0.0), [1.0, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(tube.normal(0.0), [0.0, 1.0], atol=1.0e-14)


def test_walls_are_consistent_normal_offsets_with_separation_point_one():
    tube = _geometry()
    upper, lower = tube.walls(41)
    q = np.linspace(0.0, tube.x_span, 41, dtype=np.float64)
    center = tube.centerline(q)

    np.testing.assert_allclose(np.linalg.norm(upper - center, axis=1), 0.05, atol=2e-14)
    np.testing.assert_allclose(np.linalg.norm(lower - center, axis=1), 0.05, atol=2e-14)
    np.testing.assert_allclose(np.linalg.norm(upper - lower, axis=1), 0.1, atol=4e-14)


def test_default_radius_and_arclength_match_analytic_geometry():
    tube = _geometry()

    assert tube.minimum_curvature_radius() == pytest.approx(0.11257909293593087, rel=2e-12)
    assert tube.minimum_curvature_radius() > 2.0 * tube.tube_half_width
    assert tube.arclength(tube.x_span) == pytest.approx(2.085644924350019, rel=2e-11)


def test_arclength_parameter_inverse_round_trip():
    tube = _geometry()
    q = np.linspace(0.0, tube.x_span, 39, dtype=np.float64)
    s = tube.arclength(q)

    np.testing.assert_allclose(tube.parameter_at_arclength(s), q, atol=3e-11)


def test_project_recovers_endpoint_and_normal_offset():
    tube = _geometry()
    point = tube.centerline(0.0) + 0.02 * tube.normal(0.0)

    result = tube.project(point)

    assert result.q == pytest.approx(0.0, abs=1e-13)
    np.testing.assert_allclose(result.point, [0.0, 0.225], atol=1e-13)
    np.testing.assert_allclose(result.tangent, [1.0, 0.0], atol=1e-13)
    np.testing.assert_allclose(result.normal, [0.0, 1.0], atol=1e-13)
    assert result.signed_distance == pytest.approx(0.02, abs=1e-13)


def test_project_global_fallback_beats_wrong_warm_start_lobe():
    tube = _geometry()
    expected_q = 1.27
    point = tube.centerline(expected_q) + 0.006 * tube.normal(expected_q)

    result = tube.project(point, initial_q=0.05)

    assert result.used_global_fallback
    assert result.q == pytest.approx(expected_q, abs=2e-9)
    assert result.signed_distance == pytest.approx(0.006, abs=2e-9)


def test_tracked_projection_rejects_stationary_but_non_global_cached_lobe():
    tube = _geometry()
    point = np.array(
        [0.772988341563213, -0.29987806787660176], dtype=np.float64
    )
    wrong_lobe_q = 1.2312006705233192

    expected = tube.project(point, initial_q=wrong_lobe_q)
    tracked = tube.project_tracked(
        point, accepted_q=wrong_lobe_q, trust_radius=0.02
    )

    assert tracked.used_global_fallback
    assert tracked.q == pytest.approx(expected.q, abs=1.0e-10)
    np.testing.assert_allclose(tracked.point, expected.point, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(
        tracked.tangent, expected.tangent, rtol=0.0, atol=1.0e-10
    )
    np.testing.assert_allclose(
        tracked.normal, expected.normal, rtol=0.0, atol=1.0e-10
    )


def test_project_terminates_and_keeps_global_result_after_stalled_backtracking_case():
    tube = _geometry()
    point = np.array([1.4029132475837969, -0.13920803384868274], dtype=np.float64)

    result = tube.project(point, initial_q=-0.53536376358091542)

    dense_q = np.linspace(0.0, tube.x_span, 4097, dtype=np.float64)
    dense_distance_squared = np.sum((tube.centerline(dense_q) - point) ** 2, axis=1)
    result_distance_squared = float(np.sum((result.point - point) ** 2))
    assert np.isfinite(result.q)
    assert np.all(np.isfinite(result.point))
    assert result_distance_squared <= float(np.min(dense_distance_squared)) + 2.0e-9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"amplitude": 0.0},
        {"amplitude": np.inf},
        {"x_span": -1.0},
        {"tube_half_width": np.nan},
        {"tube_half_width": 0.12},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _geometry(**kwargs)


def test_high_amplitude_narrow_span_is_rejected_by_clearance_validation():
    with pytest.raises(ValueError, match="minimum curvature radius"):
        _geometry(amplitude=0.45, x_span=0.5, tube_half_width=0.05)
