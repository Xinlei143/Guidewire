"""Deterministic CPU smoke runners that execute each packaged production solver."""
from __future__ import annotations

import argparse
from importlib import import_module
import math
import os
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np

from ._smoke import cpu_smoke_runtime, write_smoke_result
from .common.cli import cli_arguments

_MODULES = {
    "straight-tube-2d-pi": "softrobot.straight_tube.legacy_pi_2d",
    "straight-tube-v2-3d": "softrobot.straight_tube.v2",
    "straight-tube-v3": "softrobot.straight_tube.v3",
    "straight-tube-mesh-3d": "softrobot.straight_tube.quasi_static_mesh_3d",
    "curved-tube-2d-legacy-python": "softrobot.curved_tube.legacy_arc_2d",
    "curved-tube-2d-arc": "softrobot.curved_tube.arc_2d",
    "curved-tube-3d-legacy-analytic": "softrobot.curved_tube.legacy_arc_3d",
    "curved-tube-3d-legacy-mesh": "softrobot.curved_tube.legacy_arc_3d",
    "straight-tube-v4": "softrobot.straight_tube.v4",
    "curved-tube-v4-analytic": "softrobot.straight_tube.v4",
    "curved-tube-v4-mesh": "softrobot.straight_tube.v4",
}


def _variant(name: str, backend: str, initial, result, force, active, info, started: float, **extra):
    import torch
    state = result.detach().cpu().numpy() if torch.is_tensor(result) else np.asarray(result)
    before = initial.detach().cpu().numpy() if torch.is_tensor(initial) else np.asarray(initial)
    force_array = force.detach().cpu().numpy() if torch.is_tensor(force) else np.asarray(force)
    active_array = active.detach().cpu().numpy() if torch.is_tensor(active) else np.asarray(active)
    if not np.isfinite(before).all():
        raise RuntimeError(f"{name} production initial state is not finite")
    if not np.isfinite(state).all():
        raise RuntimeError(f"{name} production state is not finite")
    if not np.isfinite(force_array).all():
        raise RuntimeError(f"{name} production contact force is not finite")
    with np.errstate(over="ignore", invalid="ignore"):
        delta = state - before
        state_delta_norm = float(np.linalg.norm(delta))
        contact_force_norm = float(np.linalg.norm(force_array))
    if not np.isfinite(delta).all() or not math.isfinite(state_delta_norm):
        raise RuntimeError(f"{name} production state delta is not finite")
    if not math.isfinite(contact_force_norm):
        raise RuntimeError(f"{name} production contact force norm is not finite")
    if isinstance(info.iterations, (bool, np.bool_)) or not isinstance(
        info.iterations, (int, np.integer),
    ):
        raise RuntimeError(f"{name} production Newton iterations are invalid")
    iterations = int(info.iterations)
    if iterations < 1:
        raise RuntimeError(f"{name} production Newton iterations must be at least one")
    residual = float(info.residual_norm)
    if not math.isfinite(residual) or residual < 0.0:
        raise RuntimeError(f"{name} production Newton residual is invalid")
    evidence = {
        "name": name,
        "backend": backend,
        "finite": True,
        "converged": bool(info.converged),
        "iterations": iterations,
        "residual": residual,
        "state_delta_norm": state_delta_norm,
        "contact_force_norm": contact_force_norm,
        "contact_count": int(np.count_nonzero(active_array)),
        "elapsed_s": time.perf_counter() - started,
        **extra,
    }
    if not evidence["converged"]:
        raise RuntimeError(f"{name} production Newton step did not converge: {evidence['residual']:.3e}")
    if evidence["state_delta_norm"] <= 0.0:
        raise RuntimeError(f"{name} production step did not move")
    if evidence["contact_force_norm"] <= 0.0 and evidence["contact_count"] <= 0:
        raise RuntimeError(f"{name} production contact was not exercised")
    return state, evidence


def _straight_2d(module: ModuleType):
    positions, rest, masses = module.initialize_rod(
        module.rod_length, 8, module.initial_angle_deg, module.total_free_mass,
    )
    positions[:, 1] = module.vessel_radius - module.safe_distance + 1.0e-5
    velocities = np.zeros_like(positions)
    started = time.perf_counter()
    output = module.implicit_euler_step(
        positions, velocities, rest, masses, np.zeros(8), np.array((1.0e-4, 0.0)), 1.0e-4,
    )
    new_positions, _new_velocities, _integrals, diagnostics, info = output
    return _variant(
        "python", "python", positions, new_positions, diagnostics["wall_force"],
        diagnostics["active"], info, started,
    )


def _straight_3d(module: ModuleType, name: str = "python"):
    import torch
    with cli_arguments([
        "--num-nodes", "8", "--pulse-count-min", "4", "--pulse-count-max", "4",
        "--device", "cpu", "--max-newton-iters", "50",
    ]):
        args = module.parse_args()
    arrays = module.initialize_rod(args.total_length, args.num_nodes, args.diameter, args.density)
    positions, velocities, rest, masses, integrals = (
        torch.as_tensor(value, dtype=module.DTYPE) for value in arrays
    )
    positions[:, 1] = args.tube_radius - args.safe_distance + 1.0e-5
    initial = positions.clone()
    pulse = torch.zeros_like(positions)
    pulse[4, 1] = -1.0e-4
    started = time.perf_counter()
    output = module.implicit_euler_step(
        positions, velocities, rest, masses, integrals, initial, pulse,
        False, 0.0, 1.0e-4, args,
    )
    return _variant(name, "python", positions, output[0], output[3], output[4], output[8], started)


def _configure_straight_mesh(module: ModuleType):
    import torch
    with cli_arguments([
        "--num-nodes", "8", "--pulse-count-min", "4", "--pulse-count-max", "4",
        "--device", "cpu", "--max-newton-iters", "50",
    ]):
        args = module.v3.parse_args()
    mesh_path = module.DEFAULT_MESH_OBJ
    mesh = module.load_obj_triangles(mesh_path)
    module.validate_mesh_alignment(mesh, args)
    args.visual_mesh = mesh
    args.mesh_plot_face_stride = 4
    args.mesh_collision = module.build_uniform_triangle_grid(mesh, mesh_path, args, 0.0, 2)
    args.performance_profile = {}
    args._last_mesh_evaluation = None
    args.mesh_exact_reuse_hit_count = 0
    args.mesh_exact_reuse_miss_count = 0
    args.mesh_contact_cache = module.MeshContactCache(
        triangle_local=torch.full((args.num_nodes,), -1, dtype=torch.long),
        last_positions=torch.full((args.num_nodes, 3), torch.nan, dtype=module.v3.DTYPE),
    )
    return args, mesh_path, mesh


def _straight_mesh(module: ModuleType):
    import torch
    args, mesh_path, mesh = _configure_straight_mesh(module)
    arrays = module.v3.initialize_rod(args.total_length, args.num_nodes, args.diameter, args.density)
    positions, velocities, rest, masses, integrals = (
        torch.as_tensor(value, dtype=module.v3.DTYPE) for value in arrays
    )
    positions[:, 1] = 0.05001
    initial = positions.clone()
    pulse = torch.zeros_like(positions)
    pulse[4, 1] = -1.0e-4
    started = time.perf_counter()
    output = module.mesh_implicit_euler_step(
        positions, velocities, rest, masses, integrals, initial, pulse,
        False, 0.0, 1.0e-4, args,
    )
    return _variant(
        "python+mesh", "python", positions, output[0], output[3], output[4], output[8], started,
        mesh_obj=str(mesh_path), mesh_face_count=int(len(mesh.triangles)),
    )


def _arc2d(module: ModuleType, backend: str, current: bool):
    import torch
    arguments = [
        "--num-nodes", "8", "--pulse-count-min", "1", "--pulse-count-max", "1",
        "--device", "cpu", "--max-newton-iters", "50",
    ]
    if current:
        arguments += ["--backend", backend, "--no-prebalance"]
    with cli_arguments(arguments):
        args = module.parse_args()
    arrays = module.initialize_rod(args)
    positions, velocities, rest, masses, integrals = (
        torch.as_tensor(value, dtype=module.DTYPE) for value in arrays
    )
    if current and backend == "cpp":
        module.initialize_arc2d_backend(args, rest, masses)
    normal = torch.as_tensor(module.circle_normal_numpy(positions.numpy(), args.arc_radius), dtype=module.DTYPE)
    positions = positions + normal * (args.tube_half_width - args.safe_distance + 1.0e-5)
    previous = torch.zeros(args.num_nodes, dtype=module.DTYPE)
    pulse = torch.zeros_like(positions)
    pulse[4] = normal[4] * 1.0e-4
    mask = torch.zeros_like(positions, dtype=torch.bool)
    mask[:2] = True
    reference = torch.full((args.num_nodes - 2,), module.NATURAL_CURVATURE, dtype=module.DTYPE)
    started = time.perf_counter()
    output = module.implicit_euler_step(
        positions, velocities, rest, reference, masses, integrals, previous,
        pulse, mask, positions.clone(), args, 1.0e-4,
    )
    return _variant(backend, backend, positions, output[0], output[3], output[4], output[7], started)


def _legacy_arc3d(module: ModuleType, contact_type: str):
    import torch
    resource = Path(__file__).resolve().parent / "resources" / "tube_obj" / "arc90_legacy_protectionR40mm_40x80.obj"
    with cli_arguments([
        "--contact-type", contact_type, "--mesh-obj", str(resource), "--device", "cpu",
        "--num-nodes", "8", "--pulse-count-min", "1", "--pulse-count-max", "1",
        "--max-newton-iters", "50",
    ]):
        args = module.parse_args()
    arrays = module.initialize_rod(args)
    positions, velocities, rest, masses, integrals = (
        torch.as_tensor(value, dtype=module.DTYPE) for value in arrays
    )
    _tangent, normal_np, _binormal = module.local_frames_numpy(positions.numpy(), args.arc_radius)
    normal = torch.as_tensor(normal_np, dtype=module.DTYPE)
    positions = positions + normal * (args.contact_radius + 1.0e-5)
    args.initial_positions = positions.clone()
    pulse = torch.zeros_like(positions)
    pulse[4] = normal[4] * 1.0e-4
    started = time.perf_counter()
    output = module.implicit_step(
        positions, velocities, rest, masses, integrals, pulse, False, 0.0, args, 1.0e-4,
    )
    extra = {}
    if contact_type == "mesh":
        stats = args.mesh_geometry.stats()
        extra = {"mesh_obj": str(resource), "mesh_face_count": int(stats["mesh_triangle_count"])}
    return _variant("python", "python", positions, output[0], output[3], output[4], output[8], started, **extra)


def _v4(module: ModuleType, model_id: str, backend: str, mesh: bool, name: str):
    import torch
    resources = Path(module.__file__).resolve().parents[1] / "resources" / "tube_obj"
    curved = model_id.startswith("curved-")
    contact_type = "mesh" if mesh else ("curved-tube" if curved else "tube")
    arguments = [
        "--backend", backend, "--contact-type", contact_type, "--num-nodes", "8",
        "--pulse-count-min", "4", "--pulse-count-max", "4", "--device", "cpu",
        "--max-newton-iters", "50",
    ]
    mesh_path = None
    if mesh:
        mesh_path = resources / (
            "arc90_L1p5m_xz_wallR60mm_protectionR50mm_64sides_axial256.obj"
            if curved else "straight_tube_L1p5m_boundaryR50mm_shellR60mm_32sides_axial2.obj"
        )
        arguments += ["--mesh-obj", str(mesh_path)]
        if curved:
            arguments += ["--centerline-csv", str(resources / "arc90_L1p5m_xz_centerline_800.csv")]
    with cli_arguments(arguments):
        args = module._parse_args_with_backend()
    arrays = module.initialize_rod(
        args.total_length, args.num_nodes, args.diameter, args.density, centerline=args.centerline,
    )
    positions, velocities, rest, masses, integrals = (
        torch.as_tensor(value, dtype=module.DTYPE) for value in arrays
    )
    if curved:
        center = torch.tensor((0.0, 0.0, getattr(args, "arc_radius", module.CURVED_TUBE_RADIUS)), dtype=module.DTYPE)
        normal = positions - center
        normal[:, 1] = 0.0
        normal = normal / torch.linalg.vector_norm(normal, dim=1)[:, None]
    else:
        normal = torch.zeros_like(positions)
        normal[:, 1] = 1.0
    offset = (0.05001 if mesh else args.tube_radius - args.safe_distance + 1.0e-5)
    positions = positions + normal * offset
    initial = positions.clone()
    pulse = torch.zeros_like(positions)
    pulse[4] = -normal[4] * 1.0e-4
    step = module._python_implicit_euler_step if backend == "python" else module._cpp_implicit_euler_step
    started = time.perf_counter()
    output = step(
        positions, velocities, rest, masses, integrals, initial, pulse,
        False, 0.0, 1.0e-4, args,
    )
    extra = {}
    if mesh:
        geometry = args.mesh_geometry
        extra = {"mesh_obj": str(mesh_path), "mesh_face_count": int(len(geometry.triangles))}
    return _variant(name, backend, positions, output[0], output[3], output[4], output[8], started, **extra)


def _run_variants(model_id: str, module: ModuleType):
    if model_id == "straight-tube-2d-pi":
        return [_straight_2d(module)]
    if model_id in {"straight-tube-v2-3d", "straight-tube-v3"}:
        return [_straight_3d(module)]
    if model_id == "straight-tube-mesh-3d":
        return [_straight_mesh(module)]
    if model_id == "curved-tube-2d-legacy-python":
        return [_arc2d(module, "python", False)]
    if model_id == "curved-tube-2d-arc":
        return [_arc2d(module, backend, True) for backend in ("python", "cpp")]
    if model_id.startswith("curved-tube-3d-legacy-"):
        return [_legacy_arc3d(module, "mesh" if model_id.endswith("mesh") else "analytic")]
    if model_id == "straight-tube-v4":
        return [
            _v4(module, model_id, "python", False, "python+tube"),
            _v4(module, model_id, "cpp", False, "cpp+tube"),
            _v4(module, model_id, "python", True, "python+straight-mesh"),
            _v4(module, model_id, "cpp", True, "cpp+straight-mesh"),
        ]
    if model_id == "curved-tube-v4-analytic":
        return [_v4(module, model_id, backend, False, backend) for backend in ("python", "cpp")]
    if model_id == "curved-tube-v4-mesh":
        return [_v4(module, model_id, backend, True, backend) for backend in ("python", "cpp")]
    raise KeyError(model_id)


def run_smoke(argv=None, default_model_id: str | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", default=".")
    known, _remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    model_id = os.environ.get("SOFTROBOT_SMOKE_MODEL_ID", default_model_id)
    if not known.smoke_test or not model_id:
        return -1
    result_path = Path(known.output_dir) / "smoke_result.json"
    result_path.unlink(missing_ok=True)
    module = import_module(_MODULES[model_id])
    with cpu_smoke_runtime():
        pairs = _run_variants(model_id, module)
    states = [state.reshape(-1) for state, _evidence in pairs]
    variants = [evidence for _state, evidence in pairs]
    write_smoke_result(
        known.output_dir, model_id, np.concatenate(states), steps=len(variants), variants=variants,
        solver="production-implicit-newton", cpu=True, seed=0,
    )
    return 0
