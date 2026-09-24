"""OBJ generation and pure-Python triangle contact for the 3-D arc-tube solver.

This module intentionally contains only tube-mesh responsibilities.  The DER
model, loading protocol, Newton loop and plotting workflow remain in the main
solver.  The generated protection surface is open at both ends: it never adds
an OBJ cap and its contact response is gated by the arc-angle interval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

from ..common.assets import geometry_asset

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_AXIAL_SEGMENTS = 240
DEFAULT_CIRCUMFERENTIAL_SEGMENTS = 192
PROTECTION_GROUP = "protection_arc_r40mm"


@dataclass(frozen=True)
class MeshFaces:
    """One frozen contact triangle per rod node for a Newton step."""

    ids: torch.Tensor
    reference: torch.Tensor
    inward_normal: torch.Tensor
    valid_arc: torch.Tensor
    distance: torch.Tensor


@dataclass
class MeshContactCache:
    face_ids: np.ndarray
    last_positions: np.ndarray
    hits: int = 0
    misses: int = 0

    @classmethod
    def for_nodes(cls, count: int) -> "MeshContactCache":
        return cls(np.full(count, -1, dtype=np.int64), np.full((count, 3), np.nan))


def default_obj_path() -> Path:
    return geometry_asset("arc90_legacy_protectionR40mm_40x80.obj")


def _arc_centerline(radius: float, theta: np.ndarray) -> np.ndarray:
    return np.column_stack((radius * np.sin(theta), radius * (1.0 - np.cos(theta)), np.zeros_like(theta)))


def generate_arc_tube_obj(
    path: Path,
    *,
    arc_radius: float,
    arc_angle_rad: float,
    contact_radius: float,
    axial_segments: int = DEFAULT_AXIAL_SEGMENTS,
    circumferential_segments: int = DEFAULT_CIRCUMFERENTIAL_SEGMENTS,
) -> Path:
    """Write a single, open, uniformly sampled 40-mm protection surface.

    The default 192 x 240 resolution keeps both chord sags well below the 20-µm
    contact smoothing length while remaining a modest, single-layer OBJ.
    """
    if axial_segments < 2 or circumferential_segments < 8:
        raise ValueError("arc OBJ needs at least 2 axial and 8 circumferential segments")
    path = Path(path)
    theta = np.linspace(0.0, arc_angle_rad, axial_segments + 1)
    phi = np.arange(circumferential_segments, dtype=np.float64) * (2.0 * math.pi / circumferential_segments)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    cx, cy = arc_radius * np.sin(tt), arc_radius * (1.0 - np.cos(tt))
    nx, ny = np.sin(tt), -np.cos(tt)  # planar cross-section direction.
    vertices = np.stack((
        cx + contact_radius * np.cos(pp) * nx,
        cy + contact_radius * np.cos(pp) * ny,
        contact_radius * np.sin(pp),
    ), axis=-1).reshape(-1, 3)

    faces: list[tuple[int, int, int]] = []
    # Winding is inward.  The loader also verifies/reorients the normal.
    for i in range(axial_segments):
        for j in range(circumferential_segments):
            a = i * circumferential_segments + j
            b = i * circumferential_segments + (j + 1) % circumferential_segments
            c = (i + 1) * circumferential_segments + (j + 1) % circumferential_segments
            d = (i + 1) * circumferential_segments + j
            faces.extend(((a, c, b), (a, d, c)))

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Single-layer open circular-arc protection surface",
        f"# arc_radius_m={arc_radius:.15g}",
        f"# arc_angle_rad={arc_angle_rad:.15g}",
        f"# contact_radius_m={contact_radius:.15g}",
        f"# axial_segments={axial_segments}",
        f"# circumferential_segments={circumferential_segments}",
        "# no end caps; triangle normals point toward lumen",
    ]
    lines.extend(f"v {x:.12f} {y:.12f} {z:.12f}" for x, y, z in vertices)
    lines.append(f"g {PROTECTION_GROUP}")
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def ensure_default_obj(path: Path, *, arc_radius: float, arc_angle_rad: float, contact_radius: float) -> Path:
    """Generate the standard OBJ once; a custom existing OBJ is never overwritten."""
    path = Path(path)
    if not path.exists():
        generate_arc_tube_obj(
            path, arc_radius=arc_radius, arc_angle_rad=arc_angle_rad,
            contact_radius=contact_radius,
        )
    return path.resolve()


def _parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, str]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    groups: dict[str, list[int]] = {"ungrouped": []}
    metadata: dict[str, str] = {}
    group = "ungrouped"
    for raw in path.read_text(encoding="ascii").splitlines():
        if raw.startswith("#") and "=" in raw:
            key, value = raw[1:].strip().split("=", 1)
            metadata[key.strip()] = value.strip()
            continue
        fields = raw.split()
        if not fields:
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "g":
            group = "_".join(fields[1:]) if len(fields) > 1 else "unnamed_group"
            groups.setdefault(group, [])
        elif fields[0] == "f" and len(fields) >= 4:
            ids = [int(value.split("/")[0]) - 1 for value in fields[1:]]
            for index in range(1, len(ids) - 1):
                faces.append((ids[0], ids[index], ids[index + 1]))
                groups[group].append(len(faces) - 1)
    if not vertices or not faces:
        raise ValueError(f"OBJ has no usable vertices/faces: {path}")
    vertex_array, face_array = np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    if face_array.min() < 0 or face_array.max() >= len(vertex_array):
        raise ValueError(f"OBJ face index is outside its vertex array: {path}")
    return vertex_array, face_array, {name: np.asarray(ids, dtype=np.int64) for name, ids in groups.items() if ids}, metadata


def _closest_point(point: np.ndarray, triangle: np.ndarray) -> tuple[np.ndarray, float]:
    """Ericson closest point on one triangle; executed only for a tiny UV neighbourhood."""
    a, b, c = triangle
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0: return a, float(np.dot(ap, ap))
    bp = point - b; d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3: return b, float(np.dot(bp, bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3); q = a + v * ab; return q, float(np.dot(point - q, point - q))
    cp = point - c; d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6: return c, float(np.dot(cp, cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6); q = a + w * ac; return q, float(np.dot(point - q, point - q))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); q = b + w * (c - b); return q, float(np.dot(point - q, point - q))
    denom = 1.0 / (va + vb + vc); q = a + (vb * denom) * ab + (vc * denom) * ac
    return q, float(np.dot(point - q, point - q))


@dataclass
class ArcMeshGeometry:
    path: Path
    vertices: np.ndarray
    faces: np.ndarray
    triangles: np.ndarray
    inward_normals: np.ndarray
    arc_radius: float
    arc_angle_rad: float
    contact_radius: float
    axial_segments: int | None
    circumferential_segments: int | None
    cache: MeshContactCache | None = None
    _tensor_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    @classmethod
    def from_obj(cls, path: Path, *, arc_radius: float, arc_angle_rad: float, contact_radius: float) -> "ArcMeshGeometry":
        path = Path(path).resolve()
        vertices, faces, groups, metadata = _parse_obj(path)
        protection = [ids for name, ids in groups.items() if "protection" in name.lower()]
        if not protection:
            raise KeyError(f"OBJ needs a protection face group; found {list(groups)}")
        faces = faces[np.concatenate(protection)]
        triangles = vertices[faces]
        raw = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        length = np.linalg.norm(raw, axis=1)
        if np.any(length < 1.0e-12):
            raise ValueError("OBJ protection surface contains degenerate triangles")
        normals = raw / length[:, None]
        centroids = triangles.mean(axis=1)
        theta = np.clip(np.arctan2(centroids[:, 0], arc_radius - centroids[:, 1]), 0.0, arc_angle_rad)
        center = _arc_centerline(arc_radius, theta)
        outward = np.sum(normals * (centroids - center), axis=1) > 0.0
        normals[outward] *= -1.0
        axial = int(metadata["axial_segments"]) if "axial_segments" in metadata else None
        circum = int(metadata["circumferential_segments"]) if "circumferential_segments" in metadata else None
        expected = 2 * axial * circum if axial and circum else -1
        if expected != len(faces): axial = circum = None
        return cls(path, vertices, faces, triangles, normals, arc_radius, arc_angle_rad, contact_radius, axial, circum)

    def ensure_cache(self, count: int) -> MeshContactCache:
        if self.cache is None or len(self.cache.face_ids) != count:
            self.cache = MeshContactCache.for_nodes(count)
        return self.cache

    def _inward_centerline_normal_numpy(self, surface_points: np.ndarray) -> np.ndarray:
        """Smooth lumen direction at points on the OBJ surface.

        The mesh determines *where* the protection surface is, including its
        chord sag.  For a circular-arc tube the prescribed contact direction
        remains toward the arc centreline, rather than jumping between the
        piecewise-planar triangle normals.  This avoids a non-physical
        tangential force whenever a sliding rod crosses a triangle edge.
        """
        theta = np.clip(
            np.arctan2(surface_points[:, 0], self.arc_radius - surface_points[:, 1]),
            0.0, self.arc_angle_rad,
        )
        center = _arc_centerline(self.arc_radius, theta)
        outward = surface_points - center
        return -outward / np.linalg.norm(outward, axis=1, keepdims=True).clip(1.0e-12)

    def _candidates(self, point: np.ndarray, cached: int) -> np.ndarray:
        """Use the generated tube's UV layout; fall back to all faces for external OBJ files."""
        selected: set[int] = {int(cached)} if cached >= 0 else set()
        if self.axial_segments is None or self.circumferential_segments is None:
            selected.update(range(len(self.triangles)))
            return np.fromiter(selected, dtype=np.int64)
        theta = np.clip(math.atan2(float(point[0]), self.arc_radius - float(point[1])), 0.0, self.arc_angle_rad)
        center = _arc_centerline(self.arc_radius, np.asarray([theta]))[0]
        normal = np.array((math.sin(theta), -math.cos(theta), 0.0))
        offset = point - center
        phi = math.atan2(float(offset[2]), float(np.dot(offset, normal))) % (2.0 * math.pi)
        i0 = min(self.axial_segments - 1, max(0, int(theta / self.arc_angle_rad * self.axial_segments)))
        j0 = int(phi / (2.0 * math.pi) * self.circumferential_segments) % self.circumferential_segments
        for i in range(max(0, i0 - 2), min(self.axial_segments, i0 + 3)):
            for j in range(j0 - 2, j0 + 3):
                face0 = 2 * (i * self.circumferential_segments + j % self.circumferential_segments)
                selected.update((face0, face0 + 1))
        return np.fromiter(selected, dtype=np.int64)

    def nearest_faces(self, points: torch.Tensor, normal_mode: str = "face") -> MeshFaces:
        """Return closest triangles and frozen contact directions.

        ``face`` is the strict OBJ model: the inward triangle normal is the
        gradient of the selected face-plane constraint.  ``smooth-radial`` is
        retained only as a circular-arc analytic comparison mode.
        """
        if normal_mode not in {"face", "smooth-radial"}:
            raise ValueError(f"unsupported mesh normal mode: {normal_mode}")
        if points.is_cuda and self.axial_segments is not None and self.circumferential_segments is not None:
            return self._nearest_faces_cuda(points, normal_mode)
        values = points.detach().cpu().numpy()
        cache = self.ensure_cache(len(values))
        ids, references, normals, distances = [], [], [], []
        valid = []
        for index, point in enumerate(values):
            theta = math.atan2(float(point[0]), self.arc_radius - float(point[1]))
            valid.append(0.0 <= theta <= self.arc_angle_rad)
            candidates = self._candidates(point, int(cache.face_ids[index]))
            best_id, best_ref, best_d2 = -1, None, math.inf
            for face_id in candidates:
                reference, distance2 = _closest_point(point, self.triangles[face_id])
                if distance2 < best_d2:
                    best_id, best_ref, best_d2 = int(face_id), reference, distance2
            if best_id == cache.face_ids[index]: cache.hits += 1
            else: cache.misses += 1
            cache.face_ids[index], cache.last_positions[index] = best_id, point
            selected_normal = self.inward_normals[best_id] if normal_mode == "face" else self._inward_centerline_normal_numpy(best_ref[None])[0]
            ids.append(best_id); references.append(best_ref); normals.append(selected_normal); distances.append(math.sqrt(best_d2))
        device, dtype = points.device, points.dtype
        return MeshFaces(
            torch.as_tensor(ids, dtype=torch.long, device=device),
            torch.as_tensor(np.asarray(references), dtype=dtype, device=device),
            torch.as_tensor(np.asarray(normals), dtype=dtype, device=device),
            torch.as_tensor(np.asarray(valid), dtype=torch.bool, device=device),
            torch.as_tensor(np.asarray(distances), dtype=dtype, device=device),
        )

    def _nearest_faces_cuda(self, points: torch.Tensor, normal_mode: str) -> MeshFaces:
        """GPU nearest-face query for this module's regular theta/phi OBJ mesh.

        The query tests the 5x5 neighbouring UV cells (two triangles each),
        then performs exact closest-point tests entirely in PyTorch.  Therefore
        the iterative CUDA solver has no per-step host/device position copy.
        """
        device, dtype = points.device, points.dtype
        key = f"{device.type}:{device.index}:{dtype}"
        cached = self._tensor_cache.get(key)
        if cached is None:
            triangles = torch.as_tensor(self.triangles, dtype=dtype, device=device)
            normals = torch.as_tensor(self.inward_normals, dtype=dtype, device=device)
            self._tensor_cache[key] = (triangles, normals)
        else:
            triangles, normals = cached
        axial, circum = int(self.axial_segments), int(self.circumferential_segments)
        theta_raw = torch.atan2(points[:, 0], self.arc_radius - points[:, 1])
        valid = (theta_raw >= 0.0) & (theta_raw <= self.arc_angle_rad)
        theta = theta_raw.clamp(0.0, self.arc_angle_rad)
        cx, cy = self.arc_radius * torch.sin(theta), self.arc_radius * (1.0 - torch.cos(theta))
        planar = torch.stack((torch.sin(theta), -torch.cos(theta), torch.zeros_like(theta)), dim=1)
        offset = points - torch.stack((cx, cy, torch.zeros_like(cx)), dim=1)
        phi = torch.remainder(torch.atan2(offset[:, 2], torch.sum(offset * planar, dim=1)), 2.0 * math.pi)
        i0 = torch.clamp((theta / self.arc_angle_rad * axial).to(torch.long), 0, axial - 1)
        j0 = torch.remainder((phi / (2.0 * math.pi) * circum).to(torch.long), circum)
        oi, oj = torch.meshgrid(
            torch.arange(-2, 3, device=device), torch.arange(-2, 3, device=device), indexing="ij",
        )
        ii = torch.clamp(i0[:, None] + oi.reshape(1, -1), 0, axial - 1)
        jj = torch.remainder(j0[:, None] + oj.reshape(1, -1), circum)
        quads = ii * circum + jj
        ids = torch.stack((2 * quads, 2 * quads + 1), dim=2).reshape(len(points), -1)
        tri = triangles[ids]
        p = points[:, None, :]
        a, b, c = tri[:, :, 0], tri[:, :, 1], tri[:, :, 2]
        ab, ac, ap = b - a, c - a, p - a
        d1, d2 = torch.sum(ab * ap, dim=-1), torch.sum(ac * ap, dim=-1)
        bp = p - b; d3, d4 = torch.sum(ab * bp, dim=-1), torch.sum(ac * bp, dim=-1)
        cp = p - c; d5, d6 = torch.sum(ab * cp, dim=-1), torch.sum(ac * cp, dim=-1)
        eps = torch.finfo(dtype).eps
        vertex_a = (d1 <= 0) & (d2 <= 0)
        vertex_b = (d3 >= 0) & (d4 <= d3)
        vc = d1 * d4 - d3 * d2
        edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        v_ab = d1 / (d1 - d3).clamp_min(eps)
        vertex_c = (d6 >= 0) & (d5 <= d6)
        vb = d5 * d2 - d1 * d6
        edge_ac = (vb <= 0) & (d1 >= 0) & (d6 <= 0)
        w_ac = d2 / (d2 - d6).clamp_min(eps)
        va = d3 * d6 - d5 * d4
        edge_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(eps)
        denom = (va + vb + vc).clamp_min(eps)
        closest = a + (vb / denom)[..., None] * ab + (vc / denom)[..., None] * ac
        closest = torch.where(vertex_a[..., None], a, closest)
        closest = torch.where(vertex_b[..., None], b, closest)
        closest = torch.where(edge_ab[..., None], a + v_ab[..., None] * ab, closest)
        closest = torch.where(vertex_c[..., None], c, closest)
        closest = torch.where(edge_ac[..., None], a + w_ac[..., None] * ac, closest)
        closest = torch.where(edge_bc[..., None], b + w_bc[..., None] * (c - b), closest)
        distance2 = torch.sum((p - closest).square(), dim=-1)
        local = torch.argmin(distance2, dim=1)
        row = torch.arange(len(points), device=device)
        chosen_ids, reference = ids[row, local], closest[row, local]
        if normal_mode == "face":
            inward = normals[chosen_ids]
        else:
            ref_theta = torch.atan2(reference[:, 0], self.arc_radius - reference[:, 1]).clamp(0.0, self.arc_angle_rad)
            ref_center = torch.stack((
                self.arc_radius * torch.sin(ref_theta),
                self.arc_radius * (1.0 - torch.cos(ref_theta)),
                torch.zeros_like(ref_theta),
            ), dim=1)
            inward = -(reference - ref_center)
            inward = inward / torch.linalg.vector_norm(inward, dim=1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
        return MeshFaces(
            chosen_ids, reference, inward, valid,
            torch.sqrt(distance2[row, local].clamp_min(0.0)),
        )

    def stats(self) -> dict[str, float | int]:
        return {
            "mesh_vertex_count": int(len(self.vertices)), "mesh_triangle_count": int(len(self.triangles)),
            "mesh_axial_segments": int(self.axial_segments or -1), "mesh_circumferential_segments": int(self.circumferential_segments or -1),
            "mesh_cache_hits": int(self.cache.hits if self.cache else 0), "mesh_cache_misses": int(self.cache.misses if self.cache else 0),
        }


def pi_contact(
    positions: torch.Tensor, faces: MeshFaces, integrals: torch.Tensor, previous: torch.Tensor,
    *, dt: float, kp: float, ki: float, max_force: float, max_integral: float,
    leak_rate: float, smooth_length: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PI force on frozen mesh planes; the normal points toward the lumen.

    ``signed`` is positive only after a node has crossed a frozen triangle
    plane.  The shifted Softplus and the explicit activation are copied from
    the mature straight-tube mesh path: a node on, or inside, the triangle
    surface has *exactly* zero contact force, whereas the force starts from
    zero continuously after crossing it.  This matters more for a faceted
    OBJ than for the analytic tube because the selected face can change from
    one implicit step to the next.
    """
    signed = -torch.sum((positions - faces.reference) * faces.inward_normal, dim=1)
    valid = faces.valid_arc
    active = valid & (signed > 0.0)
    beta = 1.0 / smooth_length
    shifted = F.softplus(signed, beta=beta) - math.log(2.0) / beta
    error = torch.where(active, shifted, torch.zeros_like(signed))
    penetration = torch.clamp(signed, min=0.0)
    # ``-expm1`` is accurate for very small penetration.  It simultaneously
    # makes the P force continuous at first contact and controls when the
    # integral state begins to accumulate.
    activation = torch.where(
        active, -torch.expm1(-penetration / smooth_length), torch.zeros_like(signed),
    )
    decay = math.exp(-leak_rate * dt)
    accumulated = integrals + error * dt
    leaked = integrals * decay
    candidate = torch.clamp(
        activation * accumulated + (1.0 - activation) * leaked,
        0.0, max_integral,
    )
    unsaturated = kp * error + ki * candidate
    blocked = (ki > 0.0) & (unsaturated >= max_force) & (candidate > integrals)
    blocked |= (ki > 0.0) & (error < previous) & (ki * integrals > kp * error) & (candidate > integrals)
    new_integrals = torch.where(blocked, integrals, candidate)
    raw_force = torch.clamp(kp * error + ki * new_integrals, 0.0, max_force)
    magnitude = activation * raw_force
    force = torch.where(active[:, None], faces.inward_normal * magnitude[:, None], torch.zeros_like(positions))
    clearance = -signed
    return force, new_integrals, active, activation, error, clearance


def contact_with_blocks(positions: torch.Tensor, faces: MeshFaces, integrals: torch.Tensor, previous: torch.Tensor, **kwargs):
    """Return frozen-face PI force and analytic 3x3 point Jacobians.

    For a frozen triangle the force is ``normal * magnitude(signed)`` and
    ``d signed / d position = -normal``.  Its Jacobian is therefore the
    rank-one matrix ``-d(magnitude)/d(signed) * normal outer normal``.  Using
    this closed form avoids one ``jacrev`` call per node per Newton iteration.
    """
    dt, kp, ki = kwargs["dt"], kwargs["kp"], kwargs["ki"]
    max_force, max_integral = kwargs["max_force"], kwargs["max_integral"]
    leak_rate, smooth_length = kwargs["leak_rate"], kwargs["smooth_length"]
    signed = -torch.sum((positions - faces.reference) * faces.inward_normal, dim=1)
    active = faces.valid_arc & (signed > 0.0)
    beta = 1.0 / smooth_length
    shifted = F.softplus(signed, beta=beta) - math.log(2.0) / beta
    error = torch.where(active, shifted, torch.zeros_like(signed))
    error_prime = torch.where(active, torch.sigmoid(beta * signed), torch.zeros_like(signed))
    penetration = torch.clamp(signed, min=0.0)
    activation = torch.where(active, -torch.expm1(-penetration / smooth_length), torch.zeros_like(signed))
    activation_prime = torch.where(
        active, torch.exp(-penetration / smooth_length) / smooth_length, torch.zeros_like(signed),
    )
    decay = math.exp(-leak_rate * dt)
    accumulated, leaked = integrals + error * dt, integrals * decay
    candidate_raw = activation * accumulated + (1.0 - activation) * leaked
    candidate = torch.clamp(candidate_raw, 0.0, max_integral)
    candidate_prime_raw = activation_prime * (accumulated - leaked) + activation * error_prime * dt
    candidate_prime = torch.where(
        (candidate_raw >= 0.0) & (candidate_raw <= max_integral),
        candidate_prime_raw, torch.zeros_like(candidate_prime_raw),
    )
    unsaturated = kp * error + ki * candidate
    blocked = (ki > 0.0) & (unsaturated >= max_force) & (candidate > integrals)
    blocked |= (ki > 0.0) & (error < previous) & (ki * integrals > kp * error) & (candidate > integrals)
    new_integrals = torch.where(blocked, integrals, candidate)
    new_integral_prime = torch.where(blocked, torch.zeros_like(candidate_prime), candidate_prime)
    raw_force = kp * error + ki * new_integrals
    clipped_force = torch.clamp(raw_force, 0.0, max_force)
    raw_prime = kp * error_prime + ki * new_integral_prime
    magnitude = activation * clipped_force
    magnitude_prime = activation_prime * clipped_force
    magnitude_prime += activation * torch.where(
        (raw_force >= 0.0) & (raw_force <= max_force), raw_prime, torch.zeros_like(raw_prime),
    )
    force = torch.where(active[:, None], faces.inward_normal * magnitude[:, None], torch.zeros_like(positions))
    blocks = -magnitude_prime[:, None, None] * faces.inward_normal[:, :, None] * faces.inward_normal[:, None, :]
    clearance = -signed
    values = (force, new_integrals, active, activation, error, clearance)
    return force, blocks, values


def validate_generated_obj(path: Path, *, arc_radius: float, arc_angle_rad: float, contact_radius: float) -> dict[str, float | int]:
    geometry = ArcMeshGeometry.from_obj(path, arc_radius=arc_radius, arc_angle_rad=arc_angle_rad, contact_radius=contact_radius)
    theta = np.clip(np.arctan2(geometry.vertices[:, 0], arc_radius - geometry.vertices[:, 1]), 0.0, arc_angle_rad)
    center = _arc_centerline(arc_radius, theta)
    radii = np.linalg.norm(geometry.vertices - center, axis=1)
    expected_vertices = (DEFAULT_AXIAL_SEGMENTS + 1) * DEFAULT_CIRCUMFERENTIAL_SEGMENTS
    expected_faces = 2 * DEFAULT_AXIAL_SEGMENTS * DEFAULT_CIRCUMFERENTIAL_SEGMENTS
    if len(geometry.vertices) != expected_vertices or len(geometry.triangles) != expected_faces:
        raise AssertionError("generated OBJ count does not match its single-layer specification")
    if not np.allclose(radii, contact_radius, atol=2.0e-10):
        raise AssertionError("generated OBJ vertices are not on the requested contact radius")
    return geometry.stats()


if __name__ == "__main__":
    # Standalone convenience command: uses the current 250-node 90-degree defaults.
    angle, length, nodes = math.pi / 2.0, 1.5, 250
    radius = length / (2.0 * (nodes - 1) * math.sin(0.5 * angle / (nodes - 1)))
    target = generate_arc_tube_obj(default_obj_path(), arc_radius=radius, arc_angle_rad=angle, contact_radius=0.04)
    print(f"generated {target}")
    print(validate_generated_obj(target, arc_radius=radius, arc_angle_rad=angle, contact_radius=0.04))
