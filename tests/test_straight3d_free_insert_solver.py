import dataclasses

import numpy as np
import pytest
import torch


def _api():
    import rod_soft_contact_PI_dynamic_quasi_static_3d_straight_free_insert as module

    return module


def _tiny_config(**overrides):
    api = _api()
    values = dict(
        num_nodes=8,
        young_modulus=2.0e6,
        contact_kp=200.0,
        contact_ki=0.0,
        friction_kt_line=500.0,
        dt=0.01,
        max_newton_iters=12,
        residual_tol=1.0e-7,
        step_tol=1.0e-11,
        total_insertion=1.0e-4,
        num_load_steps=1,
        ramp_steps=1,
        settle_confirm_steps=1,
        max_settle_steps=3,
        prebalance_max_steps=3,
    )
    values.update(overrides)
    return api.StraightFreeInsertConfig(**values)


def test_default_retry_ladder_reaches_stiff_material_time_scale():
    api = _api()

    config = api.StraightFreeInsertConfig()

    assert config.retry_subdivisions[-1] >= 128


def test_contact_ki_reduction_matches_the_v4_retry_ladder():
    api = _api()
    config = api.StraightFreeInsertConfig(contact_ki=100.0)

    reduced = api.reduced_contact_ki_config(config, 0.5)

    assert reduced.contact_ki == pytest.approx(50.0)
    assert reduced.friction_mu == config.friction_mu
    assert api.reduced_contact_ki_config(config, 0.125).contact_ki == pytest.approx(12.5)


def _inputs(config):
    api = _api()
    initial = api.initialize_rod(config)
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float64)
    positions = tensor(initial.positions)
    zeros = torch.zeros(config.num_nodes, dtype=torch.float64)
    normal_only_history = api.FrictionHistory3D(
        torch.zeros_like(positions), positions, torch.zeros(config.num_nodes, dtype=torch.bool)
    )
    context = api.begin_friction_increment(normal_only_history, positions)
    prescribed_np = api.make_prescribed_state(initial.positions, 0.0)
    return api, initial, positions, zeros, context, tensor(prescribed_np.positions)


def test_prescribed_velocity_update_uses_mask_not_left_node_indices():
    api = _api()
    old = torch.zeros((6, 3), dtype=torch.float64)
    new = old.clone()
    new[-2:, 0] = -0.02
    mask = torch.zeros_like(old, dtype=torch.bool)
    mask[-2:] = True

    velocity = api.update_velocities(new, old, mask, 0.1)

    torch.testing.assert_close(velocity[:-2], torch.zeros_like(velocity[:-2]))
    torch.testing.assert_close(
        velocity[-2:, 0], torch.full((2,), -0.2, dtype=torch.float64)
    )


def test_residual_and_jacobian_use_all_free_nodes_before_last_two():
    config = _tiny_config()
    api, initial, positions, zeros, context, prescribed = _inputs(config)
    free = positions[:-2].reshape(-1)

    residual, system, reconstructed = api.residual_and_jacobian(
        free,
        positions,
        torch.zeros_like(positions),
        torch.as_tensor(initial.rest_lengths, dtype=torch.float64),
        torch.as_tensor(initial.masses, dtype=torch.float64),
        zeros,
        zeros,
        torch.as_tensor(initial.nodal_lengths, dtype=torch.float64),
        prescribed,
        context,
        config,
        config.dt,
        False,
        True,
    )

    assert residual.shape == (3 * (config.num_nodes - 2),)
    assert system.diagonal.shape == (config.num_nodes - 2, 3, 3)
    torch.testing.assert_close(reconstructed[:-2], positions[:-2])


def test_implicit_step_does_not_mutate_friction_context():
    config = _tiny_config()
    api, initial, positions, zeros, context, prescribed = _inputs(config)
    before = tuple(value.clone() for value in dataclasses.astuple(context))
    mask = torch.zeros_like(positions, dtype=torch.bool)
    mask[-2:] = True

    api.implicit_euler_step(
        positions,
        torch.zeros_like(positions),
        torch.as_tensor(initial.rest_lengths, dtype=torch.float64),
        torch.as_tensor(initial.masses, dtype=torch.float64),
        zeros,
        zeros,
        torch.as_tensor(initial.nodal_lengths, dtype=torch.float64),
        mask,
        prescribed,
        context,
        config,
        config.dt,
        friction_enabled=False,
    )

    for actual, expected in zip(dataclasses.astuple(context), before):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_actuator_reaction_is_zero_for_unloaded_straight_rest_state():
    api = _api()
    config = _tiny_config()
    positions = torch.zeros((config.num_nodes, 3), dtype=torch.float64)
    positions[:, 0] = torch.linspace(0.0, config.total_length, config.num_nodes)
    rest = torch.linalg.vector_norm(positions[1:] - positions[:-1], dim=1)
    nodal = torch.empty(config.num_nodes, dtype=torch.float64)
    nodal[0] = 0.5 * rest[0]
    nodal[-1] = 0.5 * rest[-1]
    nodal[1:-1] = 0.5 * (rest[:-1] + rest[1:])
    area = np.pi * config.diameter**2 / 4.0
    masses = config.density * area * nodal
    zeros = torch.zeros(config.num_nodes, dtype=torch.float64)
    history = api.FrictionHistory3D(
        torch.zeros_like(positions), positions, torch.zeros(config.num_nodes, dtype=torch.bool)
    )
    context = api.begin_friction_increment(history, positions)

    reaction, dynamic = api.compute_actuator_reaction(
        positions, positions, torch.zeros_like(positions), rest, masses, zeros,
        zeros, nodal, context, config, config.dt, False,
    )

    torch.testing.assert_close(reaction, torch.zeros((2, 3), dtype=torch.float64), atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(dynamic, torch.zeros_like(positions), atol=0.0, rtol=0.0)


def test_dynamic_balance_gate_uses_five_percent_or_absolute_residual_floor():
    api = _api()

    assert api.dynamic_balance_acceptable(0.04, 1.0, 1.0e-6)
    assert not api.dynamic_balance_acceptable(0.06, 1.0, 1.0e-6)
    assert api.dynamic_balance_acceptable(5.0e-7, 1.0e-8, 1.0e-6)
    assert not api.dynamic_balance_acceptable(2.0e-6, 1.0e-8, 1.0e-6)


def test_free_tip_advance_is_measured_from_prebalanced_reference():
    api = _api()
    reference = torch.tensor([[0.004, 0.0, 0.0]], dtype=torch.float64)
    current = torch.tensor([[0.003, 0.1, 0.2]], dtype=torch.float64)

    assert api.free_tip_advance(current, reference) == pytest.approx(0.001)
