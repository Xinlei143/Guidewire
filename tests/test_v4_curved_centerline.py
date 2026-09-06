import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from centerline_3d import Centerline3D
import mesh_contact_v4 as mesh
import rod_soft_contact_PI_dynamic_quasi_static_3d_v4 as v4


ROOT = Path(__file__).resolve().parents[1]
OBJ = ROOT / "tube_obj" / "arc90_L1p5m_wallR60mm_protectionR50mm_64sides_axial256.obj"
CENTERLINE = ROOT / "tube_obj" / "arc90_L1p5m_centerline_800.csv"
GENERATOR = ROOT / "tools" / "generate_arc_vessel_3d.py"
RADIUS = 0.9549298123336835


def test_centerline_matches_discrete_2d_arc_geometry():
    line = Centerline3D.from_csv(CENTERLINE)
    assert line.points.shape == (800, 3)
    np.testing.assert_allclose(line.points[0], (0.0, 0.0, 0.0), atol=1e-15)
    np.testing.assert_allclose(line.points[-1], (RADIUS, RADIUS, 0.0), atol=2e-15)
    assert line.length == pytest.approx(1.5, abs=2e-14)

    sampled = line.sample_by_arclength(1.5, 300)
    assert sampled.shape == (300, 3)
    np.testing.assert_allclose(sampled[[0, -1]], line.points[[0, -1]], atol=2e-15)


def test_centerline_rejects_invalid_csv(tmp_path):
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("x,y,z\n0,0,0\n0,0,0\n", encoding="ascii")
    with pytest.raises(ValueError, match="duplicate adjacent"):
        Centerline3D.from_csv(duplicate)

    nonfinite = tmp_path / "nonfinite.csv"
    nonfinite.write_text("x,y,z\n0,0,0\nnan,1,0\n", encoding="ascii")
    with pytest.raises(ValueError, match="finite"):
        Centerline3D.from_csv(nonfinite)


def test_bishop_frames_are_orthonormal_and_continuous():
    line = Centerline3D.from_csv(CENTERLINE)
    s = np.linspace(0.0, line.length, 300)
    tangents, normals, binormals = line.frames_at_s(s)
    for basis in (tangents, normals, binormals):
        np.testing.assert_allclose(np.linalg.norm(basis, axis=1), 1.0, atol=2e-12)
    np.testing.assert_allclose(np.sum(tangents * normals, axis=1), 0.0, atol=2e-12)
    np.testing.assert_allclose(np.sum(tangents * binormals, axis=1), 0.0, atol=2e-12)
    np.testing.assert_allclose(np.sum(normals * binormals, axis=1), 0.0, atol=2e-12)
    assert np.all(np.sum(normals[1:] * normals[:-1], axis=1) > 0.999)
    assert np.all(np.sum(binormals[1:] * binormals[:-1], axis=1) > 0.999)


def test_centerline_projection_recovers_distance_and_arclength():
    line = Centerline3D.from_csv(CENTERLINE)
    query_s = np.array((0.0, 0.25, 0.9, 1.5))
    points = line.point_at_s(query_s)
    projected_s, distance = line.project(points)
    np.testing.assert_allclose(projected_s, query_s, atol=2e-12)
    np.testing.assert_allclose(distance, 0.0, atol=2e-12)


def test_committed_geometry_is_deterministically_generated(tmp_path):
    generated_obj = tmp_path / OBJ.name
    generated_csv = tmp_path / CENTERLINE.name
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output-obj", str(generated_obj),
         "--output-centerline", str(generated_csv)],
        cwd=ROOT, check=True,
    )
    assert generated_obj.read_bytes() == OBJ.read_bytes()
    assert generated_csv.read_bytes() == CENTERLINE.read_bytes()


def test_arc_obj_has_expected_topology_and_clearance():
    vertices, faces, groups = mesh._parse_obj(OBJ)
    assert vertices.shape == (32896, 3)
    assert faces.shape == (65792, 3)
    protection_ids = mesh._select_group(groups, "protection")
    assert protection_ids.shape == (32768,)

    geometry = mesh.MeshGeometry.from_obj(OBJ, device=torch.device("cpu"))
    assert geometry.boundary_loop_count == 2
    line = Centerline3D.from_csv(CENTERLINE)
    sample = torch.as_tensor(line.point_at_s(np.linspace(0.0, line.length, 33)), dtype=torch.float64)
    details = geometry.nearest(sample)
    assert torch.all(details.clearance > 0.0)
    np.testing.assert_allclose(
        details.clearance.numpy(), 0.05 * np.cos(np.pi / 64.0),
        atol=8.0e-5,
    )


def _curved_args(monkeypatch, *extra):
    argv = [
        "v4", "--backend", "python", "--device", "cpu",
        "--contact-type", "mesh", "--mesh-obj", str(OBJ),
        "--centerline-csv", str(CENTERLINE), "--num-nodes", "12",
        "--pulse-count-min", "4", "--pulse-count-max", "4",
        "--pulse-seed", "7", *extra,
    ]
    monkeypatch.setattr(v4.sys, "argv", argv)
    return v4._parse_args_with_backend()


def test_parser_enables_opt_in_curved_mode_and_rejects_analytic_contact(monkeypatch):
    args = _curved_args(monkeypatch)
    assert args.centerline_csv == CENTERLINE
    assert isinstance(args.centerline, Centerline3D)
    assert args.prebalance is True

    monkeypatch.setattr(
        v4.sys, "argv", ["v4", "--contact-type", "tube", "--centerline-csv", str(CENTERLINE)]
    )
    with pytest.raises(SystemExit):
        v4._parse_args_with_backend()


def test_curved_initialization_has_zero_stretch_and_centerline_tip(monkeypatch):
    args = _curved_args(monkeypatch, "--tip-total-displacement", "0.12")
    positions, _, rest, _, _ = v4.initialize_rod(
        args.total_length, args.num_nodes, args.diameter, args.density,
        centerline=args.centerline,
    )
    np.testing.assert_allclose(rest, np.linalg.norm(np.diff(positions, axis=0), axis=1))
    np.testing.assert_allclose(positions[-1], args.centerline.point_at_s(args.total_length), atol=2e-15)
    initial = torch.as_tensor(positions, dtype=v4.DTYPE)
    mask, prescribed = v4.make_prescribed_state(initial, True, 0.12, args)
    assert mask[-1].all()
    np.testing.assert_allclose(
        prescribed[-1].numpy(), args.centerline.point_at_s(args.total_length - 0.12), atol=2e-15,
    )


def test_curved_random_pulses_use_local_bishop_normal_plane(monkeypatch):
    args = _curved_args(monkeypatch)
    first = v4.make_initial_pulse_profile(args, centerline=args.centerline)
    second = v4.make_initial_pulse_profile(args, centerline=args.centerline)
    for actual, expected in zip(first, second):
        np.testing.assert_array_equal(actual, expected)
    nodes, _, _, directions, profile = first
    s = np.linspace(0.0, args.total_length, args.num_nodes)
    tangents, normals, binormals = args.centerline.frames_at_s(s)
    np.testing.assert_allclose(np.sum(directions * tangents[nodes], axis=1), 0.0, atol=2e-12)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=2e-12)
    assert np.any(np.abs(profile[nodes, 2]) > 1.0e-8)
    assert np.any(np.abs(profile[nodes, :2]) > 1.0e-8)
    np.testing.assert_allclose(binormals[:, 2], 0.0, atol=2e-12)
    np.testing.assert_allclose(np.abs(normals[:, 2]), 1.0, atol=2e-12)


def test_curved_geometry_validation_reports_positive_clearance(monkeypatch):
    args = _curved_args(monkeypatch)
    geometry = mesh.MeshGeometry.from_obj(OBJ, device=torch.device("cpu"))
    clearances = v4.validate_centerline_mesh(args.centerline, geometry, args.total_length)
    assert clearances.shape == (65,)
    assert float(clearances.min()) > 0.0398


def test_prebalance_holds_tip_until_equilibrium():
    args = type("Args", (), dict(
        prebalance=True, prebalance_max_steps=5, settle_confirm_steps=2,
        settle_speed_tol=1.0e-3, settle_residual_tol=1.0e-6,
        contact_tolerance=5.0e-4, dt=0.01,
    ))()
    positions = torch.zeros((6, 3), dtype=torch.float64)
    state = (positions, torch.zeros_like(positions), torch.zeros(6, dtype=torch.float64))
    calls = []

    def step_once(current, forces, tip_constrained, displacement):
        calls.append((tip_constrained, displacement))
        info = v4.NewtonInfo(True, 1, 1.0e-9, 0.0, 0)
        mask, _ = v4.make_prescribed_state(positions, True, 0.0)
        return (
            current[0], current[1], current[2], torch.zeros_like(positions),
            torch.zeros(6, dtype=torch.bool), torch.zeros(6), torch.zeros(6),
            torch.ones(6), info, mask,
        )

    final_state, result, steps = v4.prebalance_zero_displacement(
        state, torch.zeros_like(positions), args, step_once,
    )
    assert steps == 2
    assert result[8].converged
    assert final_state[0] is positions
    assert calls == [(True, 0.0), (True, 0.0)]


def test_v4_cpp_curved_tip_target_matches_python(monkeypatch):
    common = [
        "--device", "cpu", "--contact-type", "mesh", "--mesh-obj", str(OBJ),
        "--centerline-csv", str(CENTERLINE), "--num-nodes", "8",
        "--pulse-count-min", "4", "--pulse-count-max", "4",
    ]

    def parsed(backend):
        monkeypatch.setattr(v4.sys, "argv", ["v4", "--backend", backend, *common])
        return v4._parse_args_with_backend()

    py_args, cpp_args = parsed("python"), parsed("cpp")
    initial_np, velocity_np, rest_np, masses_np, integral_np = v4.initialize_rod(
        py_args.total_length, py_args.num_nodes, py_args.diameter, py_args.density,
        centerline=py_args.centerline,
    )
    tensor = lambda value: torch.as_tensor(value, dtype=v4.DTYPE)
    initial, velocity = tensor(initial_np), tensor(velocity_np)
    rest, masses, integrals = tensor(rest_np), tensor(masses_np), tensor(integral_np)
    pulse = torch.zeros_like(initial)
    displacement = 1.0e-4
    python_result = v4._python_implicit_euler_step(
        initial, velocity, rest, masses, integrals, initial.clone(), pulse,
        True, displacement, 0.005, py_args,
    )
    cpp_result = v4._cpp_implicit_euler_step(
        initial, velocity, rest, masses, integrals, initial.clone(), pulse,
        True, displacement, 0.005, cpp_args,
    )
    expected_tip = py_args.centerline.point_at_s(py_args.total_length - displacement)
    np.testing.assert_allclose(cpp_result[0][-1].numpy(), expected_tip, atol=2e-15)
    torch.testing.assert_close(cpp_result[0], python_result[0], rtol=2e-8, atol=2e-10)


def test_mesh_visualization_uses_deterministic_obj_triangles():
    geometry = mesh.MeshGeometry.from_obj(OBJ, device=torch.device("cpu"))
    first = v4._mesh_visualization_triangles(geometry, max_faces=12000)
    second = v4._mesh_visualization_triangles(geometry, max_faces=12000)
    assert first is second
    assert first.shape == (12000, 3, 3)
    assert np.isfinite(first).all()
    all_vertices = geometry.vertices.numpy()
    np.testing.assert_allclose(first.min(axis=(0, 1)), all_vertices.min(axis=0), atol=1e-12)
    np.testing.assert_allclose(first.max(axis=(0, 1)), all_vertices.max(axis=0), atol=1e-12)
