#!/usr/bin/env python3
"""Generate the 3-D sweep corresponding to the default 2-D arc vessel."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJ = ROOT / "assets" / "geometry" / "arc90_L1p5m_xz_wallR60mm_protectionR50mm_64sides_axial256.obj"
DEFAULT_CENTERLINE = ROOT / "assets" / "geometry" / "arc90_L1p5m_xz_centerline_800.csv"


def chord_corrected_radius(length: float, angle: float, node_count: int) -> float:
    return length / (2.0 * (node_count - 1) * math.sin(angle / (2.0 * (node_count - 1))))


def arc_centers(radius: float, angle: float, count: int, plane: str = "xz") -> np.ndarray:
    theta = np.linspace(0.0, angle, count)
    bend = radius * (1.0 - np.cos(theta))
    if plane == "xz":
        return np.column_stack((radius * np.sin(theta), np.zeros(count), bend))
    return np.column_stack((radius * np.sin(theta), bend, np.zeros(count)))


def write_centerline(path: Path, centers: np.ndarray) -> None:
    lines = ["x,y,z"]
    lines.extend(",".join(f"{value:.17g}" for value in point) for point in centers)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def rings(
    radius: float, angle: float, count: int, cross_radius: float, sides: int,
    plane: str = "xz",
) -> np.ndarray:
    theta = np.linspace(0.0, angle, count)
    centers = arc_centers(radius, angle, count, plane)
    if plane == "xz":
        outward = np.column_stack((np.sin(theta), np.zeros(count), -np.cos(theta)))
        binormal = np.array((0.0, 1.0, 0.0))
    else:
        outward = np.column_stack((np.sin(theta), -np.cos(theta), np.zeros(count)))
        binormal = np.array((0.0, 0.0, 1.0))
    phi = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    return centers[:, None, :] + cross_radius * (
        np.cos(phi)[None, :, None] * outward[:, None, :]
        + np.sin(phi)[None, :, None] * binormal[None, None, :]
    )


def surface_faces(ring_count: int, sides: int, offset: int, *, reverse: bool) -> list[tuple[int, int, int]]:
    faces = []
    for axial in range(ring_count - 1):
        for side in range(sides):
            following = (side + 1) % sides
            a = offset + axial * sides + side
            b = offset + axial * sides + following
            c = offset + (axial + 1) * sides + following
            d = offset + (axial + 1) * sides + side
            pair = ((a, c, b), (a, d, c)) if reverse else ((a, b, c), (a, c, d))
            faces.extend(pair)
    return faces


def write_obj(
    path: Path, *, length: float, angle: float, node_count: int,
    wall_radius: float, safe_distance: float, axial_segments: int, sides: int,
    plane: str = "xz",
) -> None:
    radius = chord_corrected_radius(length, angle, node_count)
    protection_radius = wall_radius - safe_distance
    ring_count = axial_segments + 1
    protection = rings(radius, angle, ring_count, protection_radius, sides, plane)
    wall = rings(radius, angle, ring_count, wall_radius, sides, plane)
    vertices = np.concatenate((protection.reshape(-1, 3), wall.reshape(-1, 3)), axis=0)
    wall_offset = protection.size // 3
    lines = [
        "# Strict 3-D sweep of the default 2-D circular-arc vessel",
        f"# discrete_rod_length={length:.17g}m angle_deg={math.degrees(angle):.17g} chord_corrected_radius={radius:.17g}m",
        f"# wall_radius={wall_radius:.17g}m safe_distance={safe_distance:.17g}m protection_radius={protection_radius:.17g}m",
        f"# cross_section_sides={sides} axial_segments={axial_segments} centerline_nodes={node_count}",
    ]
    if plane == "xz":
        lines.append(f"# centerline_plane={plane}")
    lines.extend("v " + " ".join(f"{value:.17g}" for value in point) for point in vertices)

    lines.append(f"g protection_boundary_r_{protection_radius:.6f}m")
    for face in surface_faces(ring_count, sides, 0, reverse=False):
        lines.append("f " + " ".join(str(value + 1) for value in face))
    lines.append(f"g vessel_wall_r_{wall_radius:.6f}m")
    for face in surface_faces(ring_count, sides, wall_offset, reverse=True):
        lines.append("f " + " ".join(str(value + 1) for value in face))
    lines.append("g annular_end_rims")
    for axial in (0, ring_count - 1):
        for side in range(sides):
            following = (side + 1) % sides
            p0 = axial * sides + side
            p1 = axial * sides + following
            w0 = wall_offset + axial * sides + side
            w1 = wall_offset + axial * sides + following
            faces = ((p0, p1, w1), (p0, w1, w0)) if axial == 0 else ((p0, w1, p1), (p0, w0, w1))
            for face in faces:
                lines.append("f " + " ".join(str(value + 1) for value in face))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-obj", type=Path, default=DEFAULT_OBJ)
    parser.add_argument("--output-centerline", type=Path, default=DEFAULT_CENTERLINE)
    parser.add_argument("--length", type=float, default=1.5)
    parser.add_argument("--angle-deg", type=float, default=90.0)
    parser.add_argument("--centerline-nodes", type=int, default=800)
    parser.add_argument("--wall-radius", type=float, default=0.06)
    parser.add_argument("--safe-distance", type=float, default=0.01)
    parser.add_argument("--axial-segments", type=int, default=256)
    parser.add_argument("--sides", type=int, default=64)
    parser.add_argument("--plane", choices=("xy", "xz"), default="xz")
    args = parser.parse_args()
    if args.length <= 0.0 or not 0.0 < args.angle_deg < 360.0:
        parser.error("length must be positive and angle-deg must lie in (0, 360)")
    if args.centerline_nodes < 2 or args.axial_segments < 1 or args.sides < 3:
        parser.error("centerline-nodes, axial-segments, and sides are too small")
    if not 0.0 < args.safe_distance < args.wall_radius:
        parser.error("safe-distance must lie in (0, wall-radius)")
    return args


def main() -> None:
    args = parse_args()
    angle = math.radians(args.angle_deg)
    radius = chord_corrected_radius(args.length, angle, args.centerline_nodes)
    write_centerline(
        args.output_centerline,
        arc_centers(radius, angle, args.centerline_nodes, args.plane),
    )
    write_obj(
        args.output_obj, length=args.length, angle=angle,
        node_count=args.centerline_nodes, wall_radius=args.wall_radius,
        safe_distance=args.safe_distance, axial_segments=args.axial_segments,
        sides=args.sides, plane=args.plane,
    )


if __name__ == "__main__":
    main()
