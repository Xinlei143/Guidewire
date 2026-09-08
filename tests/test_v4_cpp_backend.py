import numpy as np
import pytest
import der_cpp_backend as backend
import torch
from pathlib import Path

import rod_soft_contact_PI_dynamic_quasi_static_3d_v3 as rod
import rod_soft_contact_PI_dynamic_quasi_static_3d_v4 as v4


def test_cpp_implicit_step_reuses_a_per_call_solver_workspace():
    root = Path(backend.__file__).resolve().parent
    source = (root / "cpp" / "der_core.cpp").read_text()
    header = (root / "cpp" / "der_core.h").read_text()

    assert "struct SolverWorkspace" in source
    step_source = source.split("py::dict DERCore::implicit_euler_step(", 1)[1]
    step_source = step_source.split("py::dict DERCore::implicit_euler_step_cached(", 1)[0]
    assert step_source.count("SolverWorkspace workspace(") == 1
    assert "workspace.jacobian.reset();" in step_source
    assert "*system = BlockPentaSystem" not in step_source
    assert "std::vector<Mat3> diagonal = system.diagonal" not in source
    assert "std::vector<Mat3> upper1 = system.upper1" not in source
    assert "mutable std::vector<int> mesh_frozen_faces_" not in header


def test_cpp_mesh_bvh_sources_are_compiled_and_hashed():
    root = Path(backend.__file__).resolve().parent
    assert (root / "cpp" / "mesh_bvh.h").is_file()
    assert (root / "cpp" / "mesh_bvh.cpp").is_file()
    loader_source = Path(backend.__file__).read_text()
    assert '"mesh_bvh.cpp"' in loader_source
    assert '"mesh_bvh.h"' in loader_source


def test_cpp_mesh_bvh_query_does_not_allocate_a_dynamic_stack():
    source = (Path(backend.__file__).resolve().parent / "cpp" / "mesh_bvh.cpp").read_text()
    assert "std::vector<int> pending" not in source


def _mesh_core():
    return backend.DERCore(
        np.array([0.1], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
        10.0, 2.0, 0.1, 0.05, 0.001, 4.0, 1.5, 20.0, 0.1, 1.0,
        2.0e-5, 1.0e-6,
    )


def _planar_grid_mesh(cells=10):
    triangles = []
    for row in range(cells):
        for col in range(cells):
            x0, x1 = float(col), float(col + 1)
            z0, z1 = float(row), float(row + 1)
            triangles.extend((
                ((x0, 0.0, z0), (x1, 0.0, z0), (x1, 0.0, z1)),
                ((x0, 0.0, z0), (x1, 0.0, z1), (x0, 0.0, z1)),
            ))
    triangles = np.asarray(triangles, dtype=np.float64)
    # An exact duplicate verifies original first-face tie semantics after BVH reordering.
    triangles = np.concatenate((triangles[:1], triangles), axis=0)
    normals = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float64), (len(triangles), 1))
    ids = np.arange(700, 700 + len(triangles), dtype=np.int64)
    return triangles, normals, ids


def _closest_point_triangle(point, a, b, c):
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = point - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + d1 / max(d1 - d3, 1.0e-30) * ab
    cp = point - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + d2 / max(d2 - d6, 1.0e-30) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        return b + (d4 - d3) / max((d4 - d3) + (d5 - d6), 1.0e-30) * (c - b)
    return a + vb / max(va + vb + vc, 1.0e-30) * ab + vc / max(va + vb + vc, 1.0e-30) * ac


def _exhaustive_face(point, triangles):
    best_distance2, best = float("inf"), -1
    for face, triangle in enumerate(triangles):
        closest = _closest_point_triangle(point, *triangle)
        distance2 = float(np.sum((point - closest) ** 2))
        if distance2 < best_distance2:
            best_distance2, best = distance2, face
    return best


def test_cpp_mesh_accepted_state_uses_predictor_frozen_face():
    triangles = np.array([
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.0, 0.2]],
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.2], [0.0, 0.0, 0.2]],
        [[0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.4, 0.0, 0.2]],
        [[0.2, 0.0, 0.0], [0.4, 0.0, 0.2], [0.2, 0.0, 0.2]],
    ], dtype=np.float64)
    normals = np.array([
        [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    nodes = 4
    core = backend.DERCore(
        np.full(nodes - 1, 0.1, dtype=np.float64),
        np.full(nodes, 0.01, dtype=np.float64),
        100.0, 1.0, 0.1, 0.05, 0.001, 3500.0, 0.0, 20.0, 5.0, 0.05,
    )
    core.configure_mesh(triangles, normals, np.arange(len(triangles), dtype=np.int64))
    positions = np.array([
        [0.0, -0.002, 0.1], [0.1, -0.002, 0.1],
        [0.195, -0.002, 0.1], [0.3, -0.002, 0.1],
    ], dtype=np.float64)
    velocities = np.zeros_like(positions)
    velocities[2, 0] = 0.1
    predictor = positions + 0.1 * velocities

    result = core.implicit_euler_step_prescribed_cached(
        positions, velocities, np.zeros(nodes, dtype=np.float64), positions.copy(),
        np.zeros_like(positions), True, 0.1, 20, 1.0e-8, 1.0e-12, 0.0,
    )
    final_positions = np.asarray(result["positions"])

    assert _exhaustive_face(predictor[2], triangles) == 3
    assert _exhaustive_face(final_positions[2], triangles) == 0
    np.testing.assert_allclose(result["contact_force"][2], 0.0, atol=1.0e-12)


def test_cpp_mesh_contact_matches_exhaustive_face_force_and_jacobian():
    triangles, normals, triangle_ids = _planar_grid_mesh()
    core = _mesh_core()
    core.configure_mesh(triangles, normals, triangle_ids)
    rng = np.random.default_rng(20260825)
    points = [
        np.array((0.0, -0.002, 0.0)),       # vertex and duplicate-face tie
        np.array((1.0, -0.003, 0.5)),       # shared edge
        np.array((5.0, -0.001, 5.0)),       # grid vertex
        *(np.array((rng.uniform(-0.5, 10.5), -rng.uniform(0.0001, 0.004),
                    rng.uniform(-0.5, 10.5))) for _ in range(16)),
    ]
    for point in points:
        face = _exhaustive_face(point, triangles)
        expected = _mesh_core()
        expected.configure_mesh(
            np.ascontiguousarray(triangles[face:face + 1]),
            np.ascontiguousarray(normals[face:face + 1]),
            np.ascontiguousarray(triangle_ids[face:face + 1]),
        )
        positions = np.ascontiguousarray(point[None, :], dtype=np.float64)
        integrals = np.array([0.02], dtype=np.float64)
        previous_errors = np.array([0.001], dtype=np.float64)
        actual = core.contact_kernel(positions, integrals, previous_errors, 0.02)
        reference = expected.contact_kernel(positions, integrals, previous_errors, 0.02)
        assert int(actual["triangle_ids"][0]) == int(triangle_ids[face])
        np.testing.assert_allclose(actual["force"], reference["force"], rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(actual["jacobian"], reference["jacobian"], rtol=0.0, atol=1e-12)


def test_cpp_mesh_contact_keeps_no_face_when_finite_coordinates_overflow_distance2():
    core = _mesh_core()
    triangles = np.array([
        ((1.0e200, 0.0, 0.0), (1.0e200, 1.0e190, 0.0), (1.0e200, 0.0, 1.0e190)),
        ((-1.0e200, 0.0, 0.0), (-1.0e200, 1.0e190, 0.0), (-1.0e200, 0.0, 1.0e190)),
    ], dtype=np.float64)
    normals = np.array(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float64)
    core.configure_mesh(triangles, normals, np.array((11, 22), dtype=np.int64))
    result = core.contact_kernel(
        np.zeros((1, 3), dtype=np.float64), np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64), 0.02,
    )
    assert int(result["triangle_ids"][0]) == -1


def test_cpp_backend_exports_der_core_and_computes_stretch_kernel():
    core = backend.DERCore(
        np.array([0.1], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
        10.0,
        2.0,
        0.1,
        0.05,
        0.001,
        1.0,
        1.0,
        1.0,
        0.1,
        1.0,
    )
    gradient, hessian = core.stretch_kernel(
        np.array([[0.0, 0.0, 0.0], [0.11, 0.0, 0.0]], dtype=np.float64),
        np.array([0.1], dtype=np.float64),
    )
    assert gradient.shape == (1, 6)
    assert hessian.shape == (1, 6, 6)
    np.testing.assert_allclose(gradient[0, 0], -1.0, rtol=1e-12, atol=1e-12)


def test_cpp_backend_rejects_non_float64_kernel_input():
    core = backend.DERCore(
        np.array([0.1], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
        10.0,
        2.0,
        0.1,
        0.05,
        0.001,
        1.0,
        1.0,
        1.0,
        0.1,
        1.0,
    )
    with pytest.raises((TypeError, ValueError)):
        core.stretch_kernel(
            np.zeros((2, 3), dtype=np.float32),
            np.array([0.1], dtype=np.float64),
        )


def _core(nodes=8):
    _, ea, ei, damping = rod.rod_properties(0.7, rod.DIAMETER, rod.MATERIAL_DENSITY)
    return backend.DERCore(
        np.full(nodes - 1, 0.7 / (nodes - 1), dtype=np.float64),
        np.full(nodes, 0.01, dtype=np.float64),
        ea, ei, damping, rod.TUBE_RADIUS, rod.SAFE_DISTANCE, rod.KP, rod.KI,
        rod.MAX_PI_FORCE, rod.INTEGRAL_LEAK_RATE, rod.MAX_INTEGRAL_STATE,
    )


def test_cpp_bending_kernel_matches_v3_autograd_oracle():
    nodes = 8
    rest = np.full(nodes - 1, 0.7 / (nodes - 1), dtype=np.float64)
    phase = np.linspace(0.0, 1.7 * np.pi, nodes)
    positions = np.column_stack((
        np.linspace(0.0, 0.7, nodes),
        0.012 * np.sin(phase),
        0.009 * np.cos(1.3 * phase),
    )).astype(np.float64)
    cpp_gradient, cpp_hessian = _core(nodes).bending_kernel(positions, rest)
    flat = torch.tensor(positions[:3].reshape(-1), dtype=rod.DTYPE, requires_grad=True)
    left = torch.tensor(rest[0], dtype=rod.DTYPE)
    right = torch.tensor(rest[1], dtype=rod.DTYPE)
    ei = rod.rod_properties(0.7, rod.DIAMETER, rod.MATERIAL_DENSITY)[2]
    energy_fn = lambda x: rod.local_bending_energy(x.reshape(3, 3), left, right, ei)
    gradient = torch.autograd.functional.jacobian(energy_fn, flat)
    hessian = torch.autograd.functional.hessian(energy_fn, flat)
    np.testing.assert_allclose(cpp_gradient[0], gradient.detach().numpy(), rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(cpp_hessian[0], hessian.detach().numpy(), rtol=1e-6, atol=1e-7)


def test_cpp_contact_kernel_matches_v3_analytic_contact():
    nodes = 8
    positions = np.zeros((nodes, 3), dtype=np.float64)
    positions[:, 0] = np.linspace(0.0, 0.7, nodes)
    positions[:, 1:] = np.array([[0.052, 0.0], [0.049, 0.003]] * 4)
    integrals = np.linspace(0.0, 0.02, nodes, dtype=np.float64)
    previous = np.zeros(nodes, dtype=np.float64)
    args = type("Args", (), dict(
        dt_active=0.1, tube_radius=rod.TUBE_RADIUS, safe_distance=rod.SAFE_DISTANCE,
        Kp=rod.KP, ki_active=rod.KI, max_pi_force=rod.MAX_PI_FORCE,
        integral_leak_rate=rod.INTEGRAL_LEAK_RATE,
        max_integral_state=rod.MAX_INTEGRAL_STATE,
    ))()
    pattern = rod.get_assembly_pattern(torch.zeros((nodes, 3), dtype=torch.bool))
    _, reference_jac = rod.assemble_contact_analytic(
        torch.tensor(positions, dtype=rod.DTYPE), torch.tensor(integrals, dtype=rod.DTYPE),
        torch.tensor(previous, dtype=rod.DTYPE), pattern, args,
    )
    reference_force, reference_integrals, *_ = rod.contact_quantities(
        torch.tensor(positions, dtype=rod.DTYPE), torch.tensor(integrals, dtype=rod.DTYPE),
        torch.tensor(previous, dtype=rod.DTYPE), args,
    )
    actual = _core(nodes).contact_kernel(positions, integrals, previous, 0.1)
    np.testing.assert_allclose(actual["force"], reference_force.numpy(), rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(actual["jacobian"][:, :, :], reference_jac.numpy(), rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(actual["new_integrals"], reference_integrals.numpy(), rtol=1e-9, atol=1e-10)


def test_cpp_contact_kernel_uses_active_ki_and_custom_constants():
    nodes = 8
    positions = np.zeros((nodes, 3), dtype=np.float64)
    positions[:, 0] = np.linspace(0.0, 0.7, nodes)
    positions[:, 1:] = np.array([[0.052, 0.0], [0.049, 0.003]] * 4)
    integrals = np.linspace(0.0, 0.02, nodes, dtype=np.float64)
    previous = np.zeros(nodes, dtype=np.float64)
    smooth_length = 1.1e-5
    direction_epsilon = 2.5e-6
    core = backend.DERCore(
        np.full(nodes - 1, 0.7 / (nodes - 1), dtype=np.float64),
        np.full(nodes, 0.01, dtype=np.float64),
        rod.rod_properties(0.7, rod.DIAMETER, rod.MATERIAL_DENSITY)[1],
        rod.rod_properties(0.7, rod.DIAMETER, rod.MATERIAL_DENSITY)[2],
        rod.rod_properties(0.7, rod.DIAMETER, rod.MATERIAL_DENSITY)[3],
        rod.TUBE_RADIUS, rod.SAFE_DISTANCE, rod.KP, rod.KI, rod.MAX_PI_FORCE,
        rod.INTEGRAL_LEAK_RATE, rod.MAX_INTEGRAL_STATE, smooth_length,
        direction_epsilon,
    )
    args = type("Args", (), dict(
        dt_active=0.1, tube_radius=rod.TUBE_RADIUS, safe_distance=rod.SAFE_DISTANCE,
        Kp=rod.KP, ki_active=0.0, max_pi_force=rod.MAX_PI_FORCE,
        integral_leak_rate=rod.INTEGRAL_LEAK_RATE,
        max_integral_state=rod.MAX_INTEGRAL_STATE,
    ))()
    reference_force, reference_integrals, *_ = rod.contact_quantities(
        torch.tensor(positions, dtype=rod.DTYPE), torch.tensor(integrals, dtype=rod.DTYPE),
        torch.tensor(previous, dtype=rod.DTYPE), args,
    )
    actual = core.contact_kernel(positions, integrals, previous, 0.1, 0.0)
    np.testing.assert_allclose(actual["force"], reference_force.numpy(), rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(actual["new_integrals"], reference_integrals.numpy(), rtol=1e-9, atol=1e-10)


def test_cpp_curved_tube_contact_matches_python_force_and_jacobian():
    nodes = 6
    radius = 0.9549298123336835
    core = backend.DERCore(
        np.full(nodes - 1, 1.5 / (nodes - 1), dtype=np.float64),
        np.full(nodes, 0.01, dtype=np.float64),
        100.0, 0.01, 0.02, 0.05, 0.01, 3500.0, 100.0, 20.0, 5.0, 0.05,
        v4.CONTACT_SMOOTH_LENGTH, v4.DIRECTION_EPSILON,
    )
    core.configure_curved_tube(radius)
    theta = np.linspace(0.1, 1.4, nodes)
    centers = np.column_stack((
        radius * np.sin(theta), np.zeros(nodes), radius * (1.0 - np.cos(theta)),
    ))
    outward = np.column_stack((np.sin(theta), np.zeros(nodes), -np.cos(theta)))
    positions = np.ascontiguousarray(centers + outward * np.linspace(0.0398, 0.041, nodes)[:, None])
    positions[:, 1] = np.linspace(-0.002, 0.002, nodes)
    integrals = np.linspace(0.0, 2.0e-5, nodes, dtype=np.float64)
    previous = np.linspace(0.0, 3.0e-4, nodes, dtype=np.float64)
    args = type("Args", (), dict(
        arc_radius=radius, tube_radius=0.05, safe_distance=0.01,
        dt_active=0.005, Kp=3500.0, ki_active=100.0, max_pi_force=20.0,
        integral_leak_rate=5.0, max_integral_state=0.05,
    ))()
    pattern = v4.get_assembly_pattern(torch.zeros((nodes, 3), dtype=torch.bool))
    reference_force, reference_jac = v4.assemble_contact_batched(
        torch.tensor(positions), torch.tensor(integrals), torch.tensor(previous), pattern,
        type("CurvedArgs", (type(args),), {"contact_type": "curved-tube"})(),
    )
    actual = core.contact_kernel(positions, integrals, previous, 0.005, 100.0)
    np.testing.assert_allclose(actual["force"].reshape(-1), reference_force.numpy(), rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(actual["jacobian"], reference_jac.numpy(), rtol=2e-8, atol=2e-10)


def test_cpp_implicit_step_reports_profile_counters():
    nodes = 8
    positions = np.zeros((nodes, 3), dtype=np.float64)
    positions[:, 0] = np.linspace(0.0, 0.7, nodes)
    rest = np.full(nodes - 1, 0.7 / (nodes - 1), dtype=np.float64)
    masses = np.full(nodes, 0.01, dtype=np.float64)
    zeros = np.zeros_like(positions)
    result = _core(nodes).implicit_euler_step(
        positions, zeros, rest, masses, np.zeros(nodes), positions, zeros,
        False, 0.0, 0.1, 20, 1e-7, 1e-10, 100.0,
    )
    assert int(result["full_assembly_count"]) >= 1
    assert int(result["residual_only_count"]) >= 0
    assert int(result["line_search_evaluation_count"]) >= 0
    assert int(result["dense_build_count"]) == int(result["dense_fallback_count"])


def test_cpp_implicit_step_matches_python_reference_on_straight_state():
    nodes = 8
    dtype = rod.DTYPE
    positions = torch.zeros((nodes, 3), dtype=dtype)
    positions[:, 0] = torch.linspace(0.0, 0.7, nodes, dtype=dtype)
    velocities = torch.zeros_like(positions)
    rest = torch.full((nodes - 1,), 0.7 / (nodes - 1), dtype=dtype)
    masses = torch.full((nodes,), 0.01, dtype=dtype)
    integrals = torch.zeros(nodes, dtype=dtype)
    pulse = torch.zeros_like(positions)
    _, ea, ei, damping = rod.rod_properties(0.7, rod.DIAMETER, rod.MATERIAL_DENSITY)
    fields = dict(
        EA=ea, EI=ei, damping=damping, dt_active=0.1, tube_radius=rod.TUBE_RADIUS,
        safe_distance=rod.SAFE_DISTANCE, Kp=rod.KP, Ki=rod.KI,
        ki_active=rod.KI, max_pi_force=rod.MAX_PI_FORCE,
        integral_leak_rate=rod.INTEGRAL_LEAK_RATE,
        max_integral_state=rod.MAX_INTEGRAL_STATE, max_newton_iters=20,
        residual_tol=1e-7, step_tol=1e-10, performance_profile={},
    )
    cpp_args = type("Args", (), fields)()
    py_args = type("Args", (), fields.copy())()
    cpp = v4._cpp_implicit_euler_step(
        positions, velocities, rest, masses, integrals, positions.clone(), pulse,
        False, 0.0, 0.1, cpp_args,
    )
    reference = rod.implicit_euler_step(
        positions, velocities, rest, masses, integrals, positions.clone(), pulse,
        False, 0.0, 0.1, py_args,
    )
    for actual, expected in zip(cpp[:8], reference[:8]):
        if expected.dtype == torch.bool:
            assert torch.equal(actual, expected)
        else:
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-8)


def test_cpp_cached_and_legacy_steps_remain_equivalent_over_repeated_steps():
    nodes = 8
    length = 0.7
    rest = np.full(nodes - 1, length / (nodes - 1), dtype=np.float64)
    masses = np.full(nodes, 0.01, dtype=np.float64)
    phase = np.linspace(0.0, 1.4 * np.pi, nodes)
    initial = np.column_stack((
        np.linspace(0.0, length, nodes),
        0.003 * np.sin(phase),
        0.002 * np.cos(0.8 * phase),
    )).astype(np.float64)
    pulse = np.zeros_like(initial)
    pulse[-2, 1] = 2.0e-4
    _, ea, ei, damping = rod.rod_properties(length, rod.DIAMETER, rod.MATERIAL_DENSITY)

    def make_core():
        return backend.DERCore(
            rest, masses, ea, ei, damping, rod.TUBE_RADIUS, rod.SAFE_DISTANCE,
            rod.KP, rod.KI, rod.MAX_PI_FORCE, rod.INTEGRAL_LEAK_RATE,
            rod.MAX_INTEGRAL_STATE,
        )

    legacy_core = make_core()
    cached_core = make_core()
    legacy_state = (initial.copy(), np.zeros_like(initial), np.zeros(nodes))
    cached_state = tuple(value.copy() for value in legacy_state)
    numeric_keys = (
        "positions", "velocities", "integrals", "contact_force",
        "contact_weight", "contact_error", "contact_distance",
    )
    scalar_keys = (
        "converged", "iterations", "residual_norm", "step_norm",
        "line_search_failures", "dense_fallback_count", "dense_build_count",
        "full_assembly_count", "residual_only_count",
        "line_search_evaluation_count", "banded_solve_count",
    )

    for _ in range(3):
        legacy = legacy_core.implicit_euler_step(
            legacy_state[0], legacy_state[1], rest, masses, legacy_state[2],
            initial, pulse, False, 0.0, 0.05, 20, 1e-7, 1e-10, rod.KI,
        )
        cached = cached_core.implicit_euler_step_cached(
            cached_state[0], cached_state[1], cached_state[2], initial, pulse,
            False, 0.0, 0.05, 20, 1e-7, 1e-10, rod.KI,
        )
        for key in numeric_keys:
            np.testing.assert_array_equal(np.asarray(cached[key]), np.asarray(legacy[key]))
        np.testing.assert_array_equal(
            np.asarray(cached["contact_active"]), np.asarray(legacy["contact_active"])
        )
        for key in scalar_keys:
            assert cached[key] == legacy[key]
        legacy_state = tuple(np.asarray(legacy[key]).copy() for key in (
            "positions", "velocities", "integrals"
        ))
        cached_state = tuple(np.asarray(cached[key]).copy() for key in (
            "positions", "velocities", "integrals"
        ))


def test_cpp_prescribed_cached_accepts_full_3d_tip_target():
    nodes = 8
    length = 0.7
    rest = np.full(nodes - 1, length / (nodes - 1), dtype=np.float64)
    masses = np.full(nodes, 0.01, dtype=np.float64)
    initial = np.zeros((nodes, 3), dtype=np.float64)
    initial[:, 0] = np.linspace(0.0, length, nodes)
    prescribed = initial.copy()
    prescribed[-1] += np.array((-0.001, 0.002, 0.003))
    core = _core(nodes)
    result = core.implicit_euler_step_prescribed_cached(
        initial, np.zeros_like(initial), np.zeros(nodes), prescribed,
        np.zeros_like(initial), True, 0.05, 20, 1e-7, 1e-10, 100.0,
    )
    np.testing.assert_allclose(result["positions"][-1], prescribed[-1], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        result["velocities"][-1], (prescribed[-1] - initial[-1]) / 0.05,
        atol=1e-15, rtol=1e-15,
    )


def test_cpp_legacy_scalar_and_full_prescribed_interfaces_match():
    nodes = 8
    length = 0.7
    rest = np.full(nodes - 1, length / (nodes - 1), dtype=np.float64)
    masses = np.full(nodes, 0.01, dtype=np.float64)
    initial = np.zeros((nodes, 3), dtype=np.float64)
    initial[:, 0] = np.linspace(0.0, length, nodes)
    pulse = np.zeros_like(initial)
    displacement = 0.001
    prescribed = initial.copy()
    prescribed[-1, 0] -= displacement
    legacy = _core(nodes).implicit_euler_step_cached(
        initial, np.zeros_like(initial), np.zeros(nodes), initial, pulse,
        True, displacement, 0.05, 20, 1e-7, 1e-10, 100.0,
    )
    generalized = _core(nodes).implicit_euler_step_prescribed_cached(
        initial, np.zeros_like(initial), np.zeros(nodes), prescribed, pulse,
        True, 0.05, 20, 1e-7, 1e-10, 100.0,
    )
    for key in (
        "positions", "velocities", "integrals", "contact_force",
        "contact_weight", "contact_error", "contact_distance",
    ):
        np.testing.assert_array_equal(np.asarray(generalized[key]), np.asarray(legacy[key]))


def test_v4_no_argument_defaults_to_auto_device_for_cpp(monkeypatch):
    monkeypatch.setattr(v4.sys, "argv", ["rod_soft_contact_PI_dynamic_quasi_static_3d_v4.py"])
    parsed = v4._parse_args_with_backend()
    assert parsed.backend == "cpp"
    assert parsed.device == "auto"
    assert parsed.torch_device.type == "cpu"


def test_v4_front_parameters_drive_parser_defaults(monkeypatch):
    monkeypatch.setattr(v4, "NUM_NODES", 17)
    monkeypatch.setattr(v4, "TIP_TOTAL_DISPLACEMENT", 0.012)
    monkeypatch.setattr(v4.sys, "argv", ["rod_soft_contact_PI_dynamic_quasi_static_3d_v4.py"])
    parsed = v4._parse_args_with_backend()
    assert parsed.num_nodes == 17
    assert parsed.tip_total_displacement == 0.012


def test_v4_import_does_not_load_v3_module():
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    code = (
        "import sys; import rod_soft_contact_PI_dynamic_quasi_static_3d_v4; "
        "assert 'rod_soft_contact_PI_dynamic_quasi_static_3d_v3' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=os.getcwd(), env=env,
    )
    assert result.returncode == 0, result.stderr


def test_cpp_micro_relaxation_keeps_absolute_tip_target():
    nodes = 8
    dtype = rod.DTYPE
    positions = torch.zeros((nodes, 3), dtype=dtype)
    positions[:, 0] = torch.linspace(0.0, 0.7, nodes, dtype=dtype)
    velocities = torch.zeros_like(positions)
    rest = torch.full((nodes - 1,), 0.7 / (nodes - 1), dtype=dtype)
    masses = torch.full((nodes,), 0.01, dtype=dtype)
    integrals = torch.zeros(nodes, dtype=dtype)
    pulse = torch.zeros_like(positions)
    _, ea, ei, damping = rod.rod_properties(0.7, rod.DIAMETER, rod.MATERIAL_DENSITY)
    args = type("Args", (), dict(
        EA=ea, EI=ei, damping=damping, dt_active=0.2,
        tube_radius=rod.TUBE_RADIUS, safe_distance=rod.SAFE_DISTANCE,
        Kp=rod.KP, Ki=rod.KI, ki_active=rod.KI,
        max_pi_force=rod.MAX_PI_FORCE, integral_leak_rate=rod.INTEGRAL_LEAK_RATE,
        max_integral_state=rod.MAX_INTEGRAL_STATE, max_newton_iters=20,
        residual_tol=1e-7, step_tol=1e-10, performance_profile={},
    ))()
    first = v4._cpp_implicit_euler_step(
        positions, velocities, rest, masses, integrals, positions.clone(), pulse,
        True, 0.0045, 0.2, args,
    )
    second = v4._cpp_implicit_euler_step(
        first[0], first[1], rest, masses, first[2], positions.clone(), pulse,
        True, 0.0045, 0.2, args,
    )
    torch.testing.assert_close(second[0][-1], first[0][-1], rtol=0.0, atol=1e-12)
