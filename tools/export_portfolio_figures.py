#!/usr/bin/env python3
"""Export deterministic, publication-style portfolio figures from settled v4 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "softrobot-mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from softrobot.common.centerline_3d import Centerline3D


REQUIRED_KEYS = (
    "positions",
    "velocities",
    "completed_load_steps",
    "newton_residual",
    "max_protection_error",
    "centerline_points",
    "initial_pulse_seed",
    "initial_pulse_nodes",
    "initial_pulse_magnitudes_n",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate_run(
    path: Path,
    *,
    expected_load_steps: int,
    speed_tol: float,
    residual_tol: float,
    contact_tol: float,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    with np.load(path) as archive:
        missing = [key for key in REQUIRED_KEYS if key not in archive]
        if missing:
            raise ValueError(f"{path} is missing required fields: {', '.join(missing)}")
        run = {key: np.asarray(archive[key]).copy() for key in archive.files}

    for key in REQUIRED_KEYS:
        value = run[key]
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"{path}: {key} contains NaN or Inf")

    positions = run["positions"]
    velocities = run["velocities"]
    if positions.ndim != 3 or positions.shape[-1] != 3 or velocities.shape != positions.shape:
        raise ValueError(f"{path}: positions and velocities must have matching (frames,nodes,3) shape")
    completed = int(run["completed_load_steps"])
    if completed != expected_load_steps:
        raise ValueError(f"{path}: completed {completed}/{expected_load_steps} load steps")
    if "newton_converged" in run and not bool(run["newton_converged"][-1]):
        raise ValueError(f"{path}: final Newton solve did not converge")

    free_velocities = velocities[-1, 2:-1]
    if not len(free_velocities):
        free_velocities = velocities[-1]
    max_speed = float(np.max(np.linalg.norm(free_velocities, axis=1)))
    residual = float(run["newton_residual"][-1])
    contact_error = float(run["max_protection_error"][-1])
    if max_speed >= speed_tol:
        raise ValueError(f"{path}: final free-node speed {max_speed:.6g} exceeds {speed_tol:.6g}")
    if residual >= residual_tol:
        raise ValueError(f"{path}: final Newton residual {residual:.6g} exceeds {residual_tol:.6g}")
    if contact_error >= contact_tol:
        raise ValueError(f"{path}: final contact error {contact_error:.6g} exceeds {contact_tol:.6g}")

    return run, {
        "completed_load_steps": completed,
        "final_max_free_node_speed_m_per_s": max_speed,
        "final_newton_residual_n": residual,
        "final_max_protection_error_m": contact_error,
    }


def parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v" and len(fields) >= 4:
                vertices.append([float(value) for value in fields[1:4]])
            elif fields[0] == "f" and len(fields) >= 4:
                indices = [int(token.split("/")[0]) - 1 for token in fields[1:]]
                for index in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[index], indices[index + 1]])
    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    if vertex_array.ndim != 2 or vertex_array.shape[1:] != (3,) or not len(face_array):
        raise ValueError(f"{path}: OBJ must contain finite vertices and triangular faces")
    if not np.isfinite(vertex_array).all() or face_array.min() < 0 or face_array.max() >= len(vertex_array):
        raise ValueError(f"{path}: OBJ contains invalid geometry")
    return vertex_array, face_array


def tube_surface(centerline_csv: Path, radius: float = 0.06) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line = Centerline3D.from_csv(centerline_csv)
    samples = min(180, max(40, len(line.points)))
    s = np.linspace(0.0, line.length, samples)
    centers = line.point_at_s(s)
    _tangents, normals, binormals = line.frames_at_s(s)
    angles = np.linspace(0.0, 2.0 * np.pi, 30)
    rings = centers[:, None, :] + radius * (
        np.cos(angles)[None, :, None] * normals[:, None, :]
        + np.sin(angles)[None, :, None] * binormals[:, None, :]
    )
    return rings[..., 0], rings[..., 1], rings[..., 2]


def style_axis(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = max(float(np.max(maxs - mins)), 1.0e-6) * 0.54
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=23, azim=-58)
    ax.set_axis_off()


def add_rod(ax, run: dict[str, np.ndarray], *, label: str | None = None) -> None:
    positions = run["positions"][-1]
    ax.plot(
        positions[:, 0], positions[:, 1], positions[:, 2],
        color="#d96b3b", linewidth=3.2, solid_capstyle="round", label=label,
    )
    active = run.get("contact_active")
    if active is not None and active.shape[:2] == run["positions"].shape[:2]:
        nodes = positions[np.asarray(active[-1], dtype=bool)]
        if len(nodes):
            ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=10, color="#7b2f1b", depthshade=False)


def save_webp(figure, path: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".png") as temporary:
        figure.savefig(temporary.name, dpi=100, facecolor="#fbfbfa", bbox_inches=None, pad_inches=0)
        with Image.open(temporary.name) as image:
            image.convert("RGB").save(path, "WEBP", quality=88, method=6)
    plt.close(figure)


def render_hero(run: dict[str, np.ndarray], centerline_csv: Path, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    figure = plt.figure(figsize=(16, 9))
    ax = figure.add_subplot(111, projection="3d")
    x, y, z = tube_surface(centerline_csv)
    ax.plot_surface(x, y, z, color="#b8d2df", alpha=0.20, linewidth=0, shade=False)
    add_rod(ax, run)
    style_axis(ax, np.concatenate((run["centerline_points"], run["positions"][-1]), axis=0))
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # A single 3-D axes otherwise occupies only a centered square inside a 16:9
    # canvas. Enlarge it so the curved guidewire remains legible at card width.
    ax.set_position([0.04, -0.28, 0.92, 1.56])
    save_webp(figure, output)


def render_contact_models(
    analytic: dict[str, np.ndarray],
    mesh: dict[str, np.ndarray],
    mesh_obj: Path,
    centerline_csv: Path,
    output: Path,
) -> None:
    figure = plt.figure(figsize=(16, 9))
    axes = [figure.add_subplot(121, projection="3d"), figure.add_subplot(122, projection="3d")]
    x, y, z = tube_surface(centerline_csv)
    axes[0].plot_surface(x, y, z, color="#b8d2df", alpha=0.22, linewidth=0, shade=False)
    add_rod(axes[0], analytic)
    axes[0].set_title("A  Smooth analytic curved tube", loc="left", color="#1f2933", pad=10, weight="bold")

    vertices, faces = parse_obj(mesh_obj)
    step = max(1, len(faces) // 5000)
    collection = Poly3DCollection(
        vertices[faces[::step]], facecolor="#b8d2df", edgecolor="#6f97aa",
        linewidth=0.18, alpha=0.16,
    )
    axes[1].add_collection3d(collection)
    add_rod(axes[1], mesh)
    axes[1].set_title("B  Triangular-mesh curved vessel", loc="left", color="#1f2933", pad=10, weight="bold")

    extent = np.concatenate((analytic["centerline_points"], analytic["positions"][-1], mesh["positions"][-1]), axis=0)
    for ax in axes:
        style_axis(ax, extent)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.01, top=0.94, wspace=0.02)
    save_webp(figure, output)


def file_record(path: Path) -> dict[str, int | str]:
    with Image.open(path) as image:
        width, height = image.size
    return {"sha256": sha256(path), "bytes": path.stat().st_size, "width": width, "height": height}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic-npz", type=Path, required=True)
    parser.add_argument("--mesh-npz", type=Path, required=True)
    parser.add_argument("--mesh-obj", type=Path, required=True)
    parser.add_argument("--centerline-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-load-steps", type=int, required=True)
    parser.add_argument("--speed-tol", type=float, default=1.0e-3)
    parser.add_argument("--residual-tol", type=float, default=1.0e-6)
    parser.add_argument("--contact-tol", type=float, default=5.0e-4)
    args = parser.parse_args()

    analytic, analytic_metrics = load_and_validate_run(
        args.analytic_npz,
        expected_load_steps=args.expected_load_steps,
        speed_tol=args.speed_tol,
        residual_tol=args.residual_tol,
        contact_tol=args.contact_tol,
    )
    mesh, mesh_metrics = load_and_validate_run(
        args.mesh_npz,
        expected_load_steps=args.expected_load_steps,
        speed_tol=args.speed_tol,
        residual_tol=args.residual_tol,
        contact_tol=args.contact_tol,
    )
    centerline = Centerline3D.from_csv(args.centerline_csv).points
    for label, run in (("analytic", analytic), ("mesh", mesh)):
        if run["positions"].shape[1] != analytic["positions"].shape[1]:
            raise ValueError("analytic and mesh runs must use the same node count")
        if run["centerline_points"].shape != centerline.shape or not np.allclose(
            run["centerline_points"], centerline, rtol=0.0, atol=2.0e-12
        ):
            raise ValueError(f"{label} run does not match --centerline-csv")
    for key in ("initial_pulse_seed", "initial_pulse_nodes", "initial_pulse_magnitudes_n"):
        if not np.array_equal(analytic[key], mesh[key]):
            raise ValueError(f"analytic and mesh runs differ in {key}")
    parse_obj(args.mesh_obj)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hero = args.output_dir / "hero-curved-guidewire.webp"
    comparison = args.output_dir / "contact-models.webp"
    render_hero(analytic, args.centerline_csv, hero)
    render_contact_models(analytic, mesh, args.mesh_obj, args.centerline_csv, comparison)

    metadata = {
        "inputs": {
            "analytic_npz": {"path": str(args.analytic_npz), "sha256": sha256(args.analytic_npz)},
            "mesh_npz": {"path": str(args.mesh_npz), "sha256": sha256(args.mesh_npz)},
            "mesh_obj": {"path": str(args.mesh_obj), "sha256": sha256(args.mesh_obj)},
            "centerline_csv": {"path": str(args.centerline_csv), "sha256": sha256(args.centerline_csv)},
        },
        "thresholds": {
            "expected_load_steps": args.expected_load_steps,
            "speed_tol_m_per_s": args.speed_tol,
            "residual_tol_n": args.residual_tol,
            "contact_tol_m": args.contact_tol,
        },
        "validation": {"analytic": analytic_metrics, "mesh": mesh_metrics},
        "figures": {hero.name: file_record(hero), comparison.name: file_record(comparison)},
    }
    metadata_path = args.output_dir / "figure-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
