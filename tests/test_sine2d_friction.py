import dataclasses

import numpy as np
import pytest
import torch


def _api():
    import rod_soft_contact_PI_dynamic_quasi_static_2d_sine_insert as module

    return module


def _tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def _frame(api, tangent=((1.0, 0.0),), *, q=None, vessel=None):
    tangent = _tensor(tangent)
    count = len(tangent)
    normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=1)
    return api.SineContactFrame(
        q=_tensor(np.arange(count) if q is None else q),
        center=torch.zeros((count, 2), dtype=torch.float64),
        tangent=tangent,
        normal=normal,
        region=torch.ones(count, dtype=torch.bool) if vessel is None else torch.as_tensor(vessel, dtype=torch.bool),
    )


def _history(api, positions, frame, active):
    return api.initialize_friction_history(
        _tensor(positions), frame, torch.as_tensor(active, dtype=torch.bool)
    )


def _evaluate(api, history, start, current, frame, normal_force, active, lengths, config, *, enabled=True):
    context = api.begin_friction_increment(history, _tensor(start))
    result = api.evaluate_friction(
        _tensor(current),
        frame,
        _tensor(normal_force),
        torch.as_tensor(active, dtype=torch.bool),
        _tensor(lengths),
        context,
        config,
        enabled=enabled,
    )
    return context, result


def test_friction_config_defaults_validation_and_cli():
    api = _api()

    defaults = api.SineInsertConfig()
    parsed = api.parse_args(["--friction-mu", "0.25", "--friction-kt-line", "3210"])

    assert defaults.friction_mu == pytest.approx(0.10)
    assert defaults.friction_kt_line == pytest.approx(1_118_600.0)
    assert parsed.friction_mu == pytest.approx(0.25)
    assert parsed.friction_kt_line == pytest.approx(3210.0)
    for kwargs in (
        {"friction_mu": -1.0},
        {"friction_mu": np.inf},
        {"friction_mu": True},
        {"friction_kt_line": 0.0},
        {"friction_kt_line": np.nan},
        {"friction_kt_line": False},
    ):
        with pytest.raises(ValueError):
            api.SineInsertConfig(**kwargs)


def test_mu_zero_disabled_inactive_and_feeder_are_exactly_zero():
    api = _api()
    positions = [[0.0, 0.0], [0.0, 0.0]]
    current = [[0.02, 0.0], [0.02, 0.0]]
    frame = _frame(api, tangent=((1.0, 0.0), (1.0, 0.0)), vessel=(True, False))
    history = _history(api, positions, frame, (True, True))

    _, mu_zero = _evaluate(
        api, history, positions, current, frame, [[0.0, 4.0], [0.0, 4.0]], (True, True), (0.1, 0.1),
        api.SineInsertConfig(friction_mu=0.0),
    )
    _, inactive = _evaluate(
        api, history, positions, current, frame, [[0.0, 4.0], [0.0, 4.0]], (False, True), (0.1, 0.1),
        api.SineInsertConfig(),
    )
    _, disabled = _evaluate(
        api, history, positions, current, frame, [[0.0, 4.0], [0.0, 4.0]], (True, True), (0.1, 0.1),
        api.SineInsertConfig(), enabled=False,
    )

    for result in (mu_zero, inactive, disabled):
        torch.testing.assert_close(result.force, torch.zeros_like(result.force), rtol=0.0, atol=0.0)
        torch.testing.assert_close(result.traction, torch.zeros_like(result.traction), rtol=0.0, atol=0.0)
        torch.testing.assert_close(result.candidate_slip, torch.zeros_like(result.candidate_slip), rtol=0.0, atol=0.0)
        assert not result.stick.any()
        assert not result.slip.any()
    torch.testing.assert_close(mu_zero.current_active, torch.tensor([True, False]))
    torch.testing.assert_close(inactive.current_active, torch.tensor([False, False]))
    torch.testing.assert_close(disabled.current_active, torch.tensor([True, False]))


def test_stick_force_direction_candidate_slip_and_line_scaling():
    api = _api()
    frame = _frame(api, tangent=((1.0, 0.0), (1.0, 0.0), (1.0, 0.0)))
    start = np.zeros((3, 2))
    current = [[1.0e-4, 0.0], [-1.0e-4, 0.0], [1.0e-4, 0.0]]
    history = _history(api, start, frame, (True, True, True))
    config = api.SineInsertConfig(friction_mu=1.0, friction_kt_line=1000.0)

    _, result = _evaluate(
        api, history, start, current, frame, [[0.0, 10.0]] * 3, (True, True, True),
        (0.05, 0.10, 0.05), config,
    )

    torch.testing.assert_close(result.trial_slip, _tensor([1.0e-4, -1.0e-4, 1.0e-4]))
    torch.testing.assert_close(result.candidate_slip, result.trial_slip)
    torch.testing.assert_close(result.traction, _tensor([-0.005, 0.010, -0.005]))
    torch.testing.assert_close(result.force[:, 1], torch.zeros(3, dtype=torch.float64))
    assert result.stick.all()
    assert not result.slip.any()
    assert torch.abs(result.force[0, 0]) + torch.abs(result.force[2, 0]) == pytest.approx(
        torch.abs(result.force[1, 0]).item()
    )


def test_slip_is_capped_and_candidate_is_consistent_with_capped_traction():
    api = _api()
    frame = _frame(api, tangent=((1.0, 0.0), (1.0, 0.0)))
    start = np.zeros((2, 2))
    history = _history(api, start, frame, (True, True))
    config = api.SineInsertConfig(friction_mu=0.2, friction_kt_line=1000.0)

    _, result = _evaluate(
        api, history, start, [[0.1, 0.0], [-0.1, 0.0]], frame,
        [[0.0, 3.0], [0.0, -4.0]], (True, True), (0.1, 0.1), config,
    )

    torch.testing.assert_close(result.coulomb_limit, _tensor([0.6, 0.8]))
    torch.testing.assert_close(result.traction, _tensor([-0.6, 0.8]))
    torch.testing.assert_close(result.force[:, 0], result.traction)
    torch.testing.assert_close(result.candidate_slip, _tensor([0.006, -0.008]))
    assert result.slip.all()
    assert not result.stick.any()


def test_first_contact_uses_zero_history_and_fixed_increment_anchor_for_every_trial():
    api = _api()
    frame = _frame(api)
    initial = [[0.0, 0.0]]
    history = _history(api, initial, frame, (False,))
    context = api.begin_friction_increment(history, _tensor([[0.01, 0.0]]))
    config = api.SineInsertConfig(friction_mu=10.0, friction_kt_line=1000.0)

    far = api.evaluate_friction(
        _tensor([[0.04, 0.0]]), frame, _tensor([[0.0, 10.0]]), torch.tensor([True]),
        _tensor([0.1]), context, config,
    )
    near = api.evaluate_friction(
        _tensor([[0.02, 0.0]]), frame, _tensor([[0.0, 10.0]]), torch.tensor([True]),
        _tensor([0.1]), context, config,
    )

    assert far.trial_slip.item() == pytest.approx(0.03)
    assert near.trial_slip.item() == pytest.approx(0.01)
    torch.testing.assert_close(context.anchors, _tensor([[0.01, 0.0]]), rtol=0.0, atol=0.0)
    torch.testing.assert_close(history.anchors, _tensor(initial), rtol=0.0, atol=0.0)
    assert history.elastic_slip.item() == 0.0


def test_frame_changes_between_trials_do_not_mutate_history_or_increment_context():
    api = _api()
    frame_x = _frame(api, q=(0.3,))
    frame_y = _frame(api, tangent=((0.0, 1.0),), q=(0.7,))
    history = api.FrictionHistory(
        elastic_slip=_tensor([0.02]), anchors=_tensor([[0.0, 0.0]]),
        q=_tensor([0.3]), contact_active=torch.tensor([True]),
    )
    context = api.begin_friction_increment(history, _tensor([[0.1, 0.2]]))
    before_history = tuple(value.clone() for value in dataclasses.astuple(history))
    before_context = tuple(value.clone() for value in dataclasses.astuple(context))
    config = api.SineInsertConfig(friction_mu=100.0, friction_kt_line=10.0)

    first = api.evaluate_friction(
        _tensor([[0.11, 0.24]]), frame_x, _tensor([[0.0, 1.0]]), torch.tensor([True]),
        _tensor([0.1]), context, config,
    )
    second = api.evaluate_friction(
        _tensor([[0.11, 0.24]]), frame_y, _tensor([[1.0, 0.0]]), torch.tensor([True]),
        _tensor([0.1]), context, config,
    )

    assert first.trial_slip.item() == pytest.approx(0.03)
    assert second.trial_slip.item() == pytest.approx(0.06)
    for actual, expected in zip(dataclasses.astuple(history), before_history):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    for actual, expected in zip(dataclasses.astuple(context), before_context):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_commit_is_explicit_and_lost_contact_clears_history_while_rollback_is_exact():
    api = _api()
    old_frame = _frame(api, tangent=((1.0, 0.0), (1.0, 0.0)), q=(0.2, 0.4))
    history = api.FrictionHistory(
        elastic_slip=_tensor([0.01, -0.02]),
        anchors=_tensor([[1.0, 1.0], [2.0, 2.0]]),
        q=_tensor([0.2, 0.4]),
        contact_active=torch.tensor([True, True]),
    )
    history_before = tuple(value.clone() for value in dataclasses.astuple(history))
    context = api.begin_friction_increment(history, _tensor([[0.0, 0.0], [0.0, 0.0]]))
    config = api.SineInsertConfig(friction_mu=100.0, friction_kt_line=100.0)
    trial = api.evaluate_friction(
        _tensor([[0.01, 0.0], [-0.03, 0.0]]), old_frame, _tensor([[0.0, 1.0], [0.0, 1.0]]),
        torch.tensor([True, False]), _tensor([0.1, 0.1]), context, config,
    )

    for actual, expected in zip(dataclasses.astuple(history), history_before):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    settled = _tensor([[0.011, 0.001], [-0.031, 0.002]])
    new_frame = _frame(api, tangent=((0.8, 0.6), (0.6, 0.8)), q=(0.25, 0.45))
    committed = api.commit_friction_increment(
        context, settled, new_frame, torch.tensor([True, False]), trial.candidate_slip
    )

    torch.testing.assert_close(committed.elastic_slip, _tensor([0.02, 0.0]))
    torch.testing.assert_close(committed.anchors, settled)
    torch.testing.assert_close(committed.q, _tensor([0.25, 0.45]))
    torch.testing.assert_close(committed.contact_active, torch.tensor([True, False]))
    for actual, expected in zip(dataclasses.astuple(history), history_before):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_returned_metadata_and_constructor_inputs_cannot_alias_history_or_context():
    api = _api()
    positions = _tensor([[0.0, 0.0]])
    frame = _frame(api, q=(0.4,))
    active = torch.tensor([True])
    history = api.initialize_friction_history(positions, frame, active)
    positions.add_(9.0)
    frame.q.add_(9.0)
    active.logical_not_()
    torch.testing.assert_close(history.anchors, _tensor([[0.0, 0.0]]), rtol=0.0, atol=0.0)
    torch.testing.assert_close(history.q, _tensor([0.4]), rtol=0.0, atol=0.0)
    assert history.contact_active.item()

    context = api.begin_friction_increment(history, _tensor([[0.1, 0.0]]))
    result = api.evaluate_friction(
        _tensor([[0.11, 0.0]]), _frame(api, q=(0.5,)), _tensor([[0.0, 1.0]]),
        torch.tensor([True]), _tensor([0.1]), context,
        api.SineInsertConfig(friction_mu=10.0, friction_kt_line=100.0),
    )
    saved_context = tuple(value.clone() for value in dataclasses.astuple(context))

    result.q.add_(100.0)
    result.tangent.zero_()
    result.current_active.logical_not_()
    result.candidate_slip.add_(100.0)
    for actual, expected in zip(dataclasses.astuple(context), saved_context):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(history.q, _tensor([0.4]), rtol=0.0, atol=0.0)


def test_discrete_friction_work_accounts_for_elastic_energy_and_plastic_dissipation():
    """Raw traction times displacement need not be dissipative during reversal."""
    api = _api()
    frame = _frame(api, tangent=((1.0, 0.0), (1.0, 0.0)))
    history = api.FrictionHistory(
        elastic_slip=_tensor([0.02, 0.02]), anchors=torch.zeros((2, 2), dtype=torch.float64),
        q=_tensor([0.2, 0.3]), contact_active=torch.tensor([True, True]),
    )
    start = np.zeros((2, 2))
    delta_u = _tensor([-0.005, 0.08])
    config = api.SineInsertConfig(friction_mu=0.2, friction_kt_line=100.0)
    context, result = _evaluate(
        api, history, start, torch.stack((delta_u, torch.zeros_like(delta_u)), dim=1), frame,
        [[0.0, 100.0], [0.0, 1.0]], (True, True), (1.0, 1.0), config,
    )
    stiffness = config.friction_kt_line * _tensor([1.0, 1.0])
    base = context.elastic_slip
    contact_work = result.traction * delta_u
    elastic_change = 0.5 * stiffness * (result.candidate_slip.square() - base.square())
    dissipation = -(contact_work + elastic_change)

    assert result.stick[0] and result.slip[1]
    assert contact_work[0] > 0.0  # reversal can return stored elastic energy
    assert contact_work[0] + elastic_change[0] == pytest.approx(
        -0.5 * stiffness[0].item() * delta_u[0].item() ** 2
    )
    assert dissipation[1] >= 0.0


def test_stick_jacobian_matches_frozen_tangent_stiffness_away_from_branch_boundary():
    api = _api()
    tangent = _tensor([[0.6, 0.8]])
    frame = _frame(api, tangent=tangent)
    history = _history(api, [[0.0, 0.0]], frame, (True,))
    context = api.begin_friction_increment(history, _tensor([[0.0, 0.0]]))
    config = api.SineInsertConfig(friction_mu=100.0, friction_kt_line=250.0)

    def force(position):
        return api.evaluate_friction(
            position.reshape(1, 2), frame, _tensor([[0.0, 10.0]]), torch.tensor([True]),
            _tensor([0.2]), context, config,
        ).force.reshape(2)

    position = _tensor([0.001, -0.002]).requires_grad_(True)
    jacobian = torch.func.jacrev(force)(position)
    expected = -(config.friction_kt_line * 0.2) * torch.outer(tangent[0], tangent[0])

    torch.testing.assert_close(jacobian, expected, rtol=2.0e-12, atol=2.0e-12)
