import dataclasses

import pytest
import torch


def _api():
    import rod_soft_contact_PI_dynamic_quasi_static_3d_straight_free_insert as module

    return module


def _tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def _context(api, anchors, active, slip=None):
    anchors = _tensor(anchors)
    count = len(anchors)
    history = api.FrictionHistory3D(
        elastic_slip=(
            torch.zeros((count, 3), dtype=torch.float64)
            if slip is None
            else _tensor(slip)
        ),
        anchors=anchors,
        contact_active=torch.tensor(active, dtype=torch.bool),
    )
    return history, api.begin_friction_increment(history, anchors)


def test_noncontact_and_mu_zero_friction_are_exactly_zero():
    api = _api()
    positions = _tensor([[0.0, 0.0, 0.0], [0.0, 0.0502, 0.0]])
    lengths = _tensor([0.1, 0.1])
    zeros = torch.zeros(2, dtype=torch.float64)
    _, context = _context(api, [[-0.01, 0.0, 0.0], [-0.01, 0.0502, 0.0]], (False, True))

    result = api.evaluate_straight_tube_contact_friction(
        positions, zeros, zeros, lengths, context,
        api.StraightFreeInsertConfig(num_nodes=6, friction_mu=0.0), 0.01,
    )

    torch.testing.assert_close(result.friction_force, torch.zeros_like(positions), rtol=0.0, atol=0.0)
    assert not result.active[0]
    assert result.active[1]


def test_contact_activity_uses_the_documented_smooth_pi_transition():
    api = _api()
    position = _tensor([[0.0, 0.04995, 0.0]])
    zeros = torch.zeros(1, dtype=torch.float64)
    _, context = _context(api, position.tolist(), (False,))
    config = api.StraightFreeInsertConfig(
        num_nodes=6, friction_mu=1.0, contact_active_weight=0.01
    )

    result = api.evaluate_straight_tube_contact_friction(
        position, zeros, zeros, _tensor([0.1]), context, config, 0.01
    )

    assert result.weight.item() > config.contact_active_weight
    assert result.active.item()


def test_friction_lies_in_tangent_plane_and_obeys_coulomb_cone():
    api = _api()
    positions = _tensor([[0.001, 0.0502, 0.001]])
    zeros = torch.zeros(1, dtype=torch.float64)
    lengths = _tensor([0.1])
    _, context = _context(api, [[0.0, 0.0502, 0.001]], (True,))
    config = api.StraightFreeInsertConfig(num_nodes=6, friction_mu=0.5)

    result = api.evaluate_straight_tube_contact_friction(
        positions, zeros, zeros, lengths, context, config, 0.01
    )

    tangent_dot = torch.sum(result.friction_force * result.normal, dim=1)
    torch.testing.assert_close(tangent_dot, torch.zeros_like(tangent_dot), atol=1.0e-12, rtol=0.0)
    friction_norm = torch.linalg.vector_norm(result.friction_force, dim=1)
    normal_norm = torch.linalg.vector_norm(result.normal_force, dim=1)
    assert torch.all(friction_norm <= config.friction_mu * normal_norm + 1.0e-12)
    assert result.slip.item()


def test_commit_is_explicit_and_inactive_nodes_lose_slip_history():
    api = _api()
    history, context = _context(
        api,
        [[0.0, 0.0502, 0.0], [0.0, 0.0502, 0.0]],
        (True, True),
        [[0.001, 0.0, 0.0], [0.002, 0.0, 0.0]],
    )
    before = tuple(value.clone() for value in dataclasses.astuple(history))
    candidate = _tensor([[0.003, 0.0, 0.0], [0.004, 0.0, 0.0]])
    settled = _tensor([[0.1, 0.0502, 0.0], [0.2, 0.0, 0.0]])

    committed = api.commit_friction_increment(
        context, settled, torch.tensor([True, False]), candidate
    )

    torch.testing.assert_close(committed.elastic_slip[0], candidate[0])
    torch.testing.assert_close(committed.elastic_slip[1], torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(committed.anchors, settled)
    for actual, expected in zip(dataclasses.astuple(history), before):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_full_contact_block_matches_finite_difference_with_xyz_coupling():
    api = _api()
    positions = _tensor([[0.00001, 0.0502, 0.001]])
    zeros = torch.zeros(1, dtype=torch.float64)
    lengths = _tensor([0.1])
    _, context = _context(api, [[0.0, 0.0502, 0.001]], (True,))
    config = api.StraightFreeInsertConfig(
        num_nodes=6, friction_mu=0.5, friction_kt_line=500_000.0
    )

    force, blocks = api.external_force_and_blocks(
        positions, zeros, zeros, lengths, context, config, 0.01, True
    )
    direct = api.evaluate_straight_tube_contact_friction(
        positions, zeros, zeros, lengths, context, config, 0.01
    ).total_force
    torch.testing.assert_close(force, direct, atol=1.0e-12, rtol=1.0e-12)

    step = 1.0e-8
    finite_difference = torch.empty_like(blocks)
    for coordinate in range(3):
        offset = torch.zeros_like(positions)
        offset[0, coordinate] = step
        plus = api.evaluate_straight_tube_contact_friction(
            positions + offset, zeros, zeros, lengths, context, config, 0.01
        ).total_force[0]
        minus = api.evaluate_straight_tube_contact_friction(
            positions - offset, zeros, zeros, lengths, context, config, 0.01
        ).total_force[0]
        finite_difference[0, :, coordinate] = (plus - minus) / (2.0 * step)
    torch.testing.assert_close(blocks, finite_difference, atol=3.0e-5, rtol=3.0e-5)
    assert abs(float(blocks[0, 0, 1])) > 0.0


def test_history_constructor_and_properties_break_tensor_aliases():
    api = _api()
    slip = _tensor([[0.1, 0.0, 0.0]])
    anchors = _tensor([[0.0, 0.0502, 0.0]])
    active = torch.tensor([True])
    history = api.FrictionHistory3D(slip, anchors, active)
    slip.zero_()
    anchors.zero_()
    active.logical_not_()

    torch.testing.assert_close(history.elastic_slip, _tensor([[0.1, 0.0, 0.0]]))
    torch.testing.assert_close(history.anchors, _tensor([[0.0, 0.0502, 0.0]]))
    assert history.contact_active.item()
    returned = history.elastic_slip
    returned.zero_()
    torch.testing.assert_close(history.elastic_slip, _tensor([[0.1, 0.0, 0.0]]))
