import numpy as np
import pytest
import torch


def _api():
    import rod_soft_contact_PI_dynamic_quasi_static_2d_sine_insert as module

    return module


def test_initialize_rod_joins_feeder_to_centerline_without_initial_stretch():
    api = _api()
    config = api.SineInsertConfig(num_nodes=101)
    tube = api.SineTube2D(
        amplitude=config.amplitude,
        x_span=config.x_span,
        tube_half_width=config.tube_half_width,
    )

    initial = api.initialize_rod(config, tube)

    assert initial.positions.dtype == np.float64
    np.testing.assert_allclose(initial.positions[0], [-2.38, config.amplitude], atol=2.0e-12)
    tip_q = tube.parameter_at_arclength(config.initial_inserted_length)
    np.testing.assert_allclose(initial.positions[-1], tube.centerline(tip_q), atol=2.0e-12)
    entry = initial.positions[np.argmin(np.linalg.norm(initial.positions - [0.0, config.amplitude], axis=1))]
    assert entry[1] == pytest.approx(config.amplitude, abs=2.0e-5)
    np.testing.assert_allclose(
        initial.rest_lengths,
        np.linalg.norm(np.diff(initial.positions, axis=0), axis=1),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(initial.reference_curvature, np.zeros(config.num_nodes - 2))
    np.testing.assert_array_equal(initial.integrals, np.zeros(config.num_nodes))
    np.testing.assert_array_equal(initial.previous_errors, np.zeros(config.num_nodes))


def test_nodal_voronoi_lengths_use_half_edges_at_endpoints():
    api = _api()

    lengths = api.nodal_voronoi_lengths(np.full(4, 0.2, dtype=np.float64))

    np.testing.assert_allclose(lengths, [0.1, 0.2, 0.2, 0.2, 0.1])
    assert lengths[0] + lengths[-1] == pytest.approx(lengths[2])


def test_proximal_prescription_moves_only_first_two_nodes_and_preserves_spacing():
    api = _api()
    initial = np.array(
        [[-2.38, 0.225], [-2.37, 0.225], [-0.2, 0.225], [0.05, 0.20]],
        dtype=np.float64,
    )

    prescribed = api.proximal_prescribed_state(initial, 0.3, amplitude=0.225)

    expected_mask = np.zeros_like(initial, dtype=bool)
    expected_mask[:2] = True
    np.testing.assert_array_equal(prescribed.mask, expected_mask)
    np.testing.assert_allclose(prescribed.positions[:2], initial[:2] + [0.3, 0.0])
    assert np.linalg.norm(np.diff(prescribed.positions[:2], axis=0)[0]) == pytest.approx(
        np.linalg.norm(np.diff(initial[:2], axis=0)[0])
    )
    assert not prescribed.mask[-1].any()
    np.testing.assert_allclose(prescribed.positions[2:], initial[2:])


def test_proximal_prescription_infers_custom_feeder_height_from_initial_state():
    api = _api()
    initial = np.array(
        [[-1.0, 0.2], [-0.9, 0.2], [-0.1, 0.2], [0.1, 0.15]],
        dtype=np.float64,
    )

    prescribed = api.proximal_prescribed_state(initial, 0.1)

    np.testing.assert_allclose(prescribed.positions[:2, 1], 0.2)


def _contact(api, positions, lengths, config, *, frame=None, dt=0.005):
    positions = torch.as_tensor(positions, dtype=torch.float64)
    count = len(positions)
    tube = api.SineTube2D(config.amplitude, config.x_span, config.tube_half_width)
    frame = api.freeze_contact_frames(positions, tube) if frame is None else frame
    return api.contact_quantities(
        positions,
        torch.zeros(count, dtype=torch.float64),
        torch.zeros(count, dtype=torch.float64),
        torch.as_tensor(lengths, dtype=torch.float64),
        frame,
        config,
        dt,
    )


def test_infinite_feeder_at_negative_x_has_exact_zero_x_force_and_independent_stiffness():
    api = _api()
    position = np.array([[-3.0, 0.22725]], dtype=np.float64)
    base = api.SineInsertConfig(feeder_kp_line=1000.0, contact_kp_line=50.0)

    first = _contact(api, position, [0.1], base)
    vessel_changed = _contact(
        api,
        position,
        [0.1],
        api.SineInsertConfig(feeder_kp_line=1000.0, contact_kp_line=50000.0),
    )
    feeder_doubled = _contact(
        api,
        position,
        [0.1],
        api.SineInsertConfig(feeder_kp_line=2000.0, contact_kp_line=50.0),
    )

    assert first.feeder_force[0, 0].item() == 0.0
    assert first.feeder_force[0, 1].item() < 0.0
    torch.testing.assert_close(first.total_force, first.feeder_force)
    torch.testing.assert_close(vessel_changed.feeder_force, first.feeder_force)
    torch.testing.assert_close(feeder_doubled.feeder_force, 2.0 * first.feeder_force)


def test_vessel_centerline_is_inactive_and_both_wall_forces_point_inward():
    api = _api()
    config = api.SineInsertConfig()
    tube = api.SineTube2D(config.amplitude, config.x_span, config.tube_half_width)
    q = np.array([0.4, 0.7, 1.1])
    center = tube.centerline(q)
    normal = tube.normal(q)
    positions = np.vstack((center[0], center[1] + 0.045 * normal[1], center[2] - 0.045 * normal[2]))

    result = _contact(api, positions, np.ones(3), config)

    assert not result.active[0].item()
    torch.testing.assert_close(result.vessel_normal_force[0], torch.zeros(2, dtype=torch.float64), atol=1e-15, rtol=0.0)
    assert torch.dot(result.vessel_normal_force[1], torch.as_tensor(normal[1])).item() < 0.0
    assert torch.dot(result.vessel_normal_force[2], torch.as_tensor(normal[2])).item() > 0.0
    torch.testing.assert_close(result.feeder_force, torch.zeros_like(result.feeder_force))


def test_line_contact_scaling_makes_two_half_nodes_equal_one_interior_node():
    api = _api()
    config = api.SineInsertConfig(contact_ki_line=0.0)
    tube = api.SineTube2D(config.amplitude, config.x_span, config.tube_half_width)
    q = np.array([0.35, 0.65, 0.95])
    positions = tube.centerline(q) + 0.045 * tube.normal(q)

    result = _contact(api, positions, [0.05, 0.1, 0.05], config)
    magnitude = torch.linalg.vector_norm(result.vessel_normal_force, dim=1)

    assert magnitude[0] + magnitude[2] == pytest.approx(magnitude[1].item(), rel=2.0e-12)


def test_result_metadata_mutation_cannot_change_frozen_frame_or_repeated_evaluation():
    api = _api()
    config = api.SineInsertConfig(contact_ki_line=0.0)
    tube = api.SineTube2D(config.amplitude, config.x_span, config.tube_half_width)
    accepted = torch.as_tensor(tube.centerline(np.array([0.25, 0.75])), dtype=torch.float64)
    frame = api.freeze_contact_frames(accepted, tube)
    trial = accepted + torch.tensor([[0.003, 0.044], [-0.004, -0.044]], dtype=torch.float64)

    first = _contact(api, trial, [0.1, 0.1], config, frame=frame)
    expected_force = first.total_force.clone()
    expected_q = frame.q.clone()
    expected_tangent = frame.tangent.clone()
    expected_normal = frame.normal.clone()
    expected_region = frame.region.clone()

    first.q.add_(10.0)
    first.tangent.zero_()
    first.normal.neg_()
    first.region.logical_not_()

    torch.testing.assert_close(frame.q, expected_q, rtol=0.0, atol=0.0)
    torch.testing.assert_close(frame.tangent, expected_tangent, rtol=0.0, atol=0.0)
    torch.testing.assert_close(frame.normal, expected_normal, rtol=0.0, atol=0.0)
    torch.testing.assert_close(frame.region, expected_region, rtol=0.0, atol=0.0)
    second = _contact(api, trial.clone(), [0.1, 0.1], config, frame=frame)
    torch.testing.assert_close(second.total_force, expected_force, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"amplitude": np.nan},
        {"wire_length": 0.0},
        {"num_nodes": 3},
        {"safe_distance": 0.05},
        {"feeder_contact_layer": 0.002},
        {"integral_leak_rate": -1.0},
        {"contact_force_cap_line": np.inf},
        {"initial_inserted_length": 2.0},
        {"total_insertion": 2.0},
    ],
)
def test_config_rejects_invalid_values_and_outlet_budget(kwargs):
    api = _api()

    with pytest.raises(ValueError):
        api.SineInsertConfig(**kwargs)


def test_parse_args_exposes_geometry_contact_feeder_insertion_output_and_backend_options():
    api = _api()

    config = api.parse_args(
        [
            "--amplitude", "0.2", "--initial-inserted-length", "0.1",
            "--contact-kp-line", "1200", "--feeder-kp-line", "800",
            "--total-insertion", "0.7", "--output-dir", "custom", "--run-name", "case-a",
            "--backend", "python",
        ]
    )

    assert config.amplitude == pytest.approx(0.2)
    assert config.initial_inserted_length == pytest.approx(0.1)
    assert config.contact_kp_line == pytest.approx(1200.0)
    assert config.feeder_kp_line == pytest.approx(800.0)
    assert config.total_insertion == pytest.approx(0.7)
    assert config.output_dir == "custom"
    assert config.run_name == "case-a"
    assert config.backend == "python"
