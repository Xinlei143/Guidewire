import numpy as np
import pytest
import torch


def _api():
    import rod_soft_contact_PI_dynamic_quasi_static_3d_straight_free_insert as module

    return module


def test_only_last_two_nodes_are_prescribed_and_translate_together():
    api = _api()
    config = api.StraightFreeInsertConfig(num_nodes=20)
    initial = api.initialize_rod(config)

    state = api.make_prescribed_state(initial.positions, 0.012)

    assert not state.mask[:-2].any()
    assert state.mask[-2:].all()
    np.testing.assert_allclose(
        state.positions[-2:, 0], initial.positions[-2:, 0] - 0.012
    )
    np.testing.assert_allclose(
        state.positions[-2:, 1:], initial.positions[-2:, 1:]
    )


def test_block_plan_covers_free_interval_from_zero_to_n_minus_two():
    api = _api()

    plan = api.get_block_plan("cpu", 20)

    assert plan.free_start == 0
    assert plan.free_end == 18
    assert plan.block_count == 18
    torch.testing.assert_close(
        plan.free_indices,
        torch.arange(18 * 3, dtype=torch.long),
        rtol=0.0,
        atol=0.0,
    )


def test_deterministic_imperfection_is_confined_to_final_quarter_and_is_spatial():
    api = _api()
    config = api.StraightFreeInsertConfig(num_nodes=41)

    first = api.initialize_rod(config)
    second = api.initialize_rod(config)

    np.testing.assert_array_equal(first.positions, second.positions)
    start = int(np.ceil(0.75 * (config.num_nodes - 1)))
    np.testing.assert_allclose(first.positions[:start, 1:], 0.0, atol=1.0e-15)
    assert np.max(np.abs(first.positions[start:, 1])) > 0.0
    assert np.max(np.abs(first.positions[start:, 2])) > 0.0
    expected_radius = config.tube_radius - config.safe_distance + config.imperfection_penetration
    np.testing.assert_allclose(
        np.linalg.norm(first.positions[-2:, 1:], axis=1),
        expected_radius,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(first.positions[-1, 1:], first.positions[-2, 1:])


def test_invalid_geometry_parameters_are_rejected():
    api = _api()

    with pytest.raises(ValueError, match="safe_distance"):
        api.StraightFreeInsertConfig(tube_radius=0.01, safe_distance=0.01)
