import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import der_cpp_backend as backend
import rod_soft_contact_PI_dynamic_quasi_static_2d_arc as arc


def _args(**overrides):
    fields = dict(
        EA=4.0,
        EI=0.015,
        damping=0.12,
        arc_radius=0.5,
        tube_half_width=0.03,
        safe_distance=0.01,
        Kp=8.0,
        Ki=1.5,
        kp_active=6.5,
        ki_active=0.7,
        max_pi_force=2.0,
        max_integral_state=0.08,
        integral_leak_rate=1.7,
        contact_smooth_length=2.0e-5,
        direction_epsilon=1.0e-6,
        max_newton_iters=30,
        residual_tol=1.0e-10,
        step_tol=1.0e-12,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _state(nodes=7, args=None):
    args = _args() if args is None else args
    initial = arc.arc_points(args.arc_radius, np.pi / 5.0, nodes).astype(np.float64)
    positions = initial.copy()
    center = np.array([0.0, args.arc_radius])
    # Move an interior node into the PI protection layer while retaining a
    # smooth, well-conditioned local configuration.
    radial = positions[3] - center
    positions[3] += 0.028 * radial / np.linalg.norm(radial)
    rest = np.linalg.norm(np.diff(initial, axis=0), axis=1).astype(np.float64)
    masses = np.full(nodes, 0.02, dtype=np.float64)
    velocities = np.zeros_like(positions)
    velocities[2] = np.array([0.002, -0.001], dtype=np.float64)
    integrals = np.linspace(0.0, 0.006, nodes, dtype=np.float64)
    previous_errors = np.linspace(0.0, 0.002, nodes, dtype=np.float64)
    pulse = np.zeros_like(positions)
    pulse[4] = np.array([-0.003, 0.004], dtype=np.float64)
    prescribed = initial.copy()
    prescribed[-1] += np.array([-0.002, -0.001], dtype=np.float64)
    reference_curvature = arc.discrete_curvature(
        torch.as_tensor(initial, dtype=arc.DTYPE)
    ).numpy()
    return (
        args,
        positions,
        velocities,
        rest,
        masses,
        integrals,
        previous_errors,
        prescribed,
        pulse,
        reference_curvature,
    )


def _core(rest, masses, args):
    return backend.Arc2DCore(
        rest,
        masses,
        args.EA,
        args.EI,
        args.damping,
        args.arc_radius,
        args.tube_half_width,
        args.safe_distance,
        args.Kp,
        args.Ki,
        args.max_pi_force,
        args.max_integral_state,
        args.integral_leak_rate,
        args.contact_smooth_length,
        args.direction_epsilon,
    )


def _reference_step(args, positions, velocities, rest, masses, integrals,
                    previous_errors, prescribed, pulse, reference_curvature,
                    tip_constrained=True):
    mask = torch.zeros_like(torch.as_tensor(positions), dtype=torch.bool)
    mask[:2] = True
    if tip_constrained:
        mask[-1] = True
    return arc.implicit_euler_step(
        torch.as_tensor(positions, dtype=arc.DTYPE),
        torch.as_tensor(velocities, dtype=arc.DTYPE),
        torch.as_tensor(rest, dtype=arc.DTYPE),
        torch.as_tensor(reference_curvature, dtype=arc.DTYPE),
        torch.as_tensor(masses, dtype=arc.DTYPE),
        torch.as_tensor(integrals, dtype=arc.DTYPE),
        torch.as_tensor(previous_errors, dtype=arc.DTYPE),
        torch.as_tensor(pulse, dtype=arc.DTYPE),
        mask,
        torch.as_tensor(prescribed, dtype=arc.DTYPE),
        args,
        0.025,
    )


def test_arc2d_wrapper_rejects_non_2d_float64_contiguous_positions():
    args, positions, _velocities, rest, masses, integrals, previous, *_ = _state()
    core = _core(rest, masses, args)
    with pytest.raises(ValueError, match="shape"):
        core.contact_kernel(np.zeros((len(positions), 3), dtype=np.float64), integrals, previous, 0.025)
    with pytest.raises(ValueError, match="C-contiguous"):
        core.contact_kernel(positions[::-1], integrals, previous, 0.025)
    with pytest.raises(TypeError, match="float64"):
        core.contact_kernel(positions.astype(np.float32), integrals, previous, 0.025)


def test_arc2d_rejects_nonfinite_and_nonphysical_core_inputs():
    args, positions, _velocities, rest, masses, integrals, previous, *_ = _state()
    for invalid_args in (
        _args(contact_smooth_length=0.0),
        _args(direction_epsilon=0.0),
    ):
        with pytest.raises(ValueError, match="positive"):
            _core(rest, masses, invalid_args)
    for invalid_rest, invalid_masses in (
        (np.array([rest[0], 0.0, *rest[2:]], dtype=np.float64), masses),
        (rest, np.array([*masses[:-1], 0.0], dtype=np.float64)),
    ):
        with pytest.raises(ValueError, match="positive"):
            _core(invalid_rest, invalid_masses, args)

    core = _core(rest, masses, args)
    nonfinite = positions.copy()
    nonfinite[3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        core.contact_kernel(nonfinite, integrals, previous, 0.025)
    for invoke in (
        lambda: core.contact_kernel(positions, integrals, previous, np.nan),
        lambda: core.contact_kernel(
            positions, integrals, previous, 0.025, kp_active=np.nan,
        ),
        lambda: core.contact_kernel(
            positions, integrals, previous, 0.025, ki_active=np.inf,
        ),
    ):
        with pytest.raises(ValueError, match="finite"):
            invoke()


def test_arc2d_contact_and_constrained_implicit_step_match_python_reference():
    (args, positions, velocities, rest, masses, integrals, previous, prescribed,
     pulse, reference_curvature) = _state()
    core = _core(rest, masses, args)
    expected_contact = arc.contact_quantities(
        torch.as_tensor(positions, dtype=arc.DTYPE),
        torch.as_tensor(integrals, dtype=arc.DTYPE),
        torch.as_tensor(previous, dtype=arc.DTYPE),
        args,
        0.025,
    )
    actual_contact = core.contact_kernel(
        positions, integrals, previous, 0.025, args.kp_active, args.ki_active,
    )
    assert expected_contact[2][3].item()
    assert torch.linalg.vector_norm(expected_contact[0][3]).item() > 0.0
    assert expected_contact[4][3].item() > 0.0
    for key, expected in (
        ("force", expected_contact[0]),
        ("new_integrals", expected_contact[1]),
        ("active", expected_contact[2]),
        ("weights", expected_contact[3]),
        ("errors", expected_contact[4]),
        ("clearance", expected_contact[5]),
    ):
        if expected.dtype == torch.bool:
            assert np.array_equal(actual_contact[key], expected.numpy())
        else:
            np.testing.assert_allclose(actual_contact[key], expected.numpy(), rtol=0.0, atol=1e-9)

    expected = _reference_step(
        args, positions, velocities, rest, masses, integrals, previous,
        prescribed, pulse, reference_curvature,
    )
    actual = core.implicit_euler_step(
        positions, velocities, integrals, previous, prescribed, pulse,
        reference_curvature, True, 0.025, args.max_newton_iters,
        args.residual_tol, args.step_tol, args.kp_active, args.ki_active,
    )
    for key, expected_value in (
        ("positions", expected[0]),
        ("velocities", expected[1]),
        ("integrals", expected[2]),
        ("contact_force", expected[3]),
        ("contact_weight", expected[5]),
        ("contact_error", expected[6]),
        ("previous_errors", expected[6]),
    ):
        np.testing.assert_allclose(actual[key], expected_value.numpy(), rtol=0.0, atol=1e-9)
    assert np.array_equal(actual["contact_active"], expected[4].numpy())
    assert bool(actual["converged"]) is expected[7].converged
    assert float(actual["residual_norm"]) == pytest.approx(expected[7].residual_norm, rel=0.0, abs=1e-9)


def test_arc2d_contact_matches_python_for_both_antiwindup_branches():
    args, positions, _velocities, rest, masses, _integrals, _previous, *_ = _state()

    saturation_args = _args(max_pi_force=0.01)
    saturation_integrals = np.zeros(len(positions), dtype=np.float64)
    saturation_previous = np.zeros(len(positions), dtype=np.float64)
    saturation_reference = arc.contact_quantities(
        torch.as_tensor(positions, dtype=arc.DTYPE),
        torch.as_tensor(saturation_integrals, dtype=arc.DTYPE),
        torch.as_tensor(saturation_previous, dtype=arc.DTYPE),
        saturation_args,
        0.025,
    )
    saturation_actual = _core(rest, masses, saturation_args).contact_kernel(
        positions, saturation_integrals, saturation_previous, 0.025,
        saturation_args.kp_active, saturation_args.ki_active,
    )
    assert saturation_reference[2][3].item()
    assert saturation_reference[1][3].item() == saturation_integrals[3]
    np.testing.assert_allclose(
        saturation_actual["new_integrals"], saturation_reference[1].numpy(),
        rtol=0.0, atol=1e-9,
    )
    np.testing.assert_allclose(
        saturation_actual["force"], saturation_reference[0].numpy(),
        rtol=0.0, atol=1e-9,
    )

    decreasing_args = _args(Kp=1.0, Ki=5.0, kp_active=1.0, ki_active=5.0)
    decreasing_integrals = np.zeros(len(positions), dtype=np.float64)
    decreasing_integrals[3] = 0.02
    decreasing_previous = np.zeros(len(positions), dtype=np.float64)
    decreasing_previous[3] = 1.0
    decreasing_reference = arc.contact_quantities(
        torch.as_tensor(positions, dtype=arc.DTYPE),
        torch.as_tensor(decreasing_integrals, dtype=arc.DTYPE),
        torch.as_tensor(decreasing_previous, dtype=arc.DTYPE),
        decreasing_args,
        0.025,
    )
    decreasing_actual = _core(rest, masses, decreasing_args).contact_kernel(
        positions, decreasing_integrals, decreasing_previous, 0.025,
        decreasing_args.kp_active, decreasing_args.ki_active,
    )
    assert decreasing_reference[2][3].item()
    assert decreasing_reference[1][3].item() == decreasing_integrals[3]
    np.testing.assert_allclose(
        decreasing_actual["new_integrals"], decreasing_reference[1].numpy(),
        rtol=0.0, atol=1e-9,
    )
    np.testing.assert_allclose(
        decreasing_actual["force"], decreasing_reference[0].numpy(),
        rtol=0.0, atol=1e-9,
    )


def test_arc2d_unconstrained_implicit_step_matches_python_reference():
    (args, positions, velocities, rest, masses, integrals, previous, prescribed,
     pulse, reference_curvature) = _state()
    expected = _reference_step(
        args, positions, velocities, rest, masses, integrals, previous,
        prescribed, pulse, reference_curvature, tip_constrained=False,
    )
    actual = _core(rest, masses, args).implicit_euler_step(
        positions, velocities, integrals, previous, prescribed, pulse,
        reference_curvature, False, 0.025, args.max_newton_iters,
        args.residual_tol, args.step_tol, args.kp_active, args.ki_active,
    )
    for key, expected_value in (
        ("positions", expected[0]),
        ("velocities", expected[1]),
        ("integrals", expected[2]),
        ("contact_force", expected[3]),
        ("contact_weight", expected[5]),
        ("contact_error", expected[6]),
        ("previous_errors", expected[6]),
    ):
        np.testing.assert_allclose(actual[key], expected_value.numpy(), rtol=0.0, atol=1e-9)
    assert np.array_equal(actual["contact_active"], expected[4].numpy())
    assert bool(actual["converged"]) is expected[7].converged
    assert float(actual["residual_norm"]) == pytest.approx(expected[7].residual_norm, rel=0.0, abs=1e-9)


def test_arc2d_nonzero_reference_curvature_changes_bending_consistently_with_python():
    (args, positions, velocities, rest, masses, integrals, previous, prescribed,
     pulse, reference_curvature) = _state()
    zero_reference = np.zeros_like(reference_curvature)
    core = _core(rest, masses, args)
    expected_zero = _reference_step(
        args, positions, velocities, rest, masses, integrals, previous,
        prescribed, pulse, zero_reference,
    )
    expected_curved = _reference_step(
        args, positions, velocities, rest, masses, integrals, previous,
        prescribed, pulse, reference_curvature,
    )
    actual_zero = core.implicit_euler_step(
        positions, velocities, integrals, previous, prescribed, pulse,
        zero_reference, True, 0.025, args.max_newton_iters, args.residual_tol,
        args.step_tol, args.kp_active, args.ki_active,
    )
    actual_curved = core.implicit_euler_step(
        positions, velocities, integrals, previous, prescribed, pulse,
        reference_curvature, True, 0.025, args.max_newton_iters,
        args.residual_tol, args.step_tol, args.kp_active, args.ki_active,
    )
    np.testing.assert_allclose(actual_zero["positions"], expected_zero[0].numpy(), rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(actual_curved["positions"], expected_curved[0].numpy(), rtol=0.0, atol=1e-9)
    assert np.linalg.norm(expected_zero[0].numpy() - expected_curved[0].numpy()) > 1.0e-10


def _parse_arc_args(monkeypatch, *options):
    monkeypatch.setattr(sys, "argv", ["rod_soft_contact_PI_dynamic_quasi_static_2d_arc.py", *options])
    return arc.parse_args()


def _workflow_args(monkeypatch, tmp_path, backend_name):
    return _parse_arc_args(
        monkeypatch,
        "--backend", backend_name,
        "--device", "cpu",
        "--num-nodes", "6",
        "--total-length", "0.12",
        "--arc-angle-deg", "30",
        "--tube-half-width", "0.04",
        "--safe-distance", "0.01",
        "--Kp", "0",
        "--Ki", "0",
        "--max-pi-force", "1",
        "--max-integral-state", "0.02",
        "--integral-leak-rate", "1",
        "--pulse-count-min", "1",
        "--pulse-count-max", "1",
        "--pulse-force-min", "1e-5",
        "--pulse-force-max", "1e-5",
        "--pulse-duration", "0.002",
        "--pulse-seed", "7",
        "--tip-total-displacement", "1e-4",
        "--num-load-steps", "1",
        "--tip-ramp-steps", "1",
        "--dt", "0.001",
        "--max-newton-iters", "20",
        "--residual-tol", "1e-10",
        "--step-tol", "1e-12",
        "--prebalance",
        "--prebalance-max-steps", "1",
        "--max-settle-steps", "1",
        "--history-save-interval", "1",
        "--settle-speed-tol", "1e9",
        "--settle-residual-tol", "1e9",
        "--contact-tolerance", "1e9",
        "--settle-confirm-steps", "1",
        "--final-perturbation-scale", "1",
        "--final-perturbation-seed", "31",
        "--output-prefix", str(tmp_path / f"workflow_{backend_name}"),
    )


def _assert_workflow_values_match(python_result, cpp_result):
    expected_history = (
        "time", "phase", "load_step", "tip_displacement", "positions", "velocities",
        "pi_forces", "pi_integrals", "previous_errors", "contact_active", "max_protection_error",
        "max_pi_force", "newton_converged", "newton_iterations",
        "newton_residual", "newton_step_norm", "newton_line_search_failures",
    )
    for key in expected_history:
        assert key in python_result and key in cpp_result
        expected, actual = np.asarray(python_result[key]), np.asarray(cpp_result[key])
        assert expected.shape == actual.shape, key
        if expected.dtype.kind in "bOUS":
            assert np.array_equal(actual, expected), key
        elif np.issubdtype(expected.dtype, np.integer):
            assert np.array_equal(actual, expected), key
        else:
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-9, err_msg=key)

    for key, expected in python_result.items():
        if not key.startswith("final_perturbation_"):
            continue
        actual = cpp_result[key]
        expected_array, actual_array = np.asarray(expected), np.asarray(actual)
        assert expected_array.shape == actual_array.shape, key
        if expected_array.dtype.kind in "bOUS":
            assert np.array_equal(actual_array, expected_array), key
        else:
            np.testing.assert_allclose(actual_array, expected_array, rtol=0.0, atol=1e-9, err_msg=key)


def test_arc2d_parser_defaults_to_cpp_and_enforces_cpu_boundary(monkeypatch):
    default = _parse_arc_args(monkeypatch)
    assert default.backend == "cpp"
    assert default.torch_device.type == "cpu"

    python_cpu = _parse_arc_args(monkeypatch, "--backend", "python", "--device", "cpu")
    assert python_cpu.backend == "python"
    assert python_cpu.torch_device.type == "cpu"

    monkeypatch.setattr(arc.torch.cuda, "is_available", lambda: True)
    cpp_auto = _parse_arc_args(monkeypatch, "--backend", "cpp", "--device", "auto")
    assert cpp_auto.torch_device.type == "cpu"
    python_auto = _parse_arc_args(monkeypatch, "--backend", "python", "--device", "auto")
    assert python_auto.torch_device.type == "cuda"
    with pytest.raises(SystemExit):
        _parse_arc_args(monkeypatch, "--backend", "cpp", "--device", "cuda")


def test_arc2d_cpp_whole_workflow_matches_python_and_reuses_one_core(monkeypatch, tmp_path):
    python_result = arc.run_simulation(_workflow_args(monkeypatch, tmp_path, "python"))

    construction_count = 0
    cpp_step_count = 0
    real_core = backend.Arc2DCore
    real_cpp_step = arc._implicit_euler_step_cpp

    def counted_core(*args, **kwargs):
        nonlocal construction_count
        construction_count += 1
        return real_core(*args, **kwargs)

    def counted_cpp_step(*args, **kwargs):
        nonlocal cpp_step_count
        cpp_step_count += 1
        return real_cpp_step(*args, **kwargs)

    monkeypatch.setattr(arc.der_cpp_backend, "Arc2DCore", counted_core)
    monkeypatch.setattr(arc, "_implicit_euler_step_cpp", counted_cpp_step)
    cpp_result = arc.run_simulation(_workflow_args(monkeypatch, tmp_path, "cpp"))
    assert construction_count == 1
    # Ki=0 deliberately selects the subdivided load route: 1 prebalance +
    # 2 initial-pulse + 4 load (two substeps, each ramp/relax) +
    # 3 final-diagnostic (pulse/pulse/relax) calls all use the dispatcher.
    assert cpp_step_count == 10
    _assert_workflow_values_match(python_result, cpp_result)
