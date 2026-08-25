"""Independent OBJ-triangle contact geometry for the v4 DER solver.

The module deliberately contains no v3 imports.  Geometry/search state is kept
outside the DER state so Newton can freeze and transactionally restore it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MeshContactDetails:
    triangle_local: torch.Tensor
    triangle_obj: torch.Tensor
    normal: torch.Tensor
    reference: torch.Tensor
    clearance: torch.Tensor
    nearest_distance: torch.Tensor
    candidate_count: torch.Tensor
    search_ring: torch.Tensor


@dataclass
class MeshContactCache:
    triangle_local: torch.Tensor
    last_positions: torch.Tensor
    hit_count: int = 0
    miss_count: int = 0
    uniform_fallback_count: int = 0
    global_refresh_count: int = 0

    @classmethod
    def for_nodes(cls, count: int, device: torch.device) -> "MeshContactCache":
        return cls(
            triangle_local=torch.full((count,), -1, dtype=torch.long, device=device),
            last_positions=torch.full((count, 3), float("nan"), dtype=torch.float64, device=device),
        )

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
        return (
            self.triangle_local.clone(), self.last_positions.clone(),
            self.hit_count, self.miss_count, self.uniform_fallback_count,
            self.global_refresh_count,
        )

    def restore(self, snapshot: tuple[torch.Tensor, torch.Tensor, int, int, int, int]) -> None:
        self.triangle_local.copy_(snapshot[0])
        self.last_positions.copy_(snapshot[1])
        self.hit_count, self.miss_count = snapshot[2], snapshot[3]
        self.uniform_fallback_count, self.global_refresh_count = snapshot[4], snapshot[5]


@dataclass(frozen=True)
class MeshContactState:
    new_integrals: torch.Tensor
    active: torch.Tensor
    weights: torch.Tensor
    errors: torch.Tensor
    distances: torch.Tensor


def _closest_points_on_triangles(
    points: torch.Tensor, triangles: torch.Tensor, valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Ericson-region closest points for one point per row and M faces."""
    p = points[:, None, :]
    a, b, c = triangles[..., 0, :], triangles[..., 1, :], triangles[..., 2, :]
    ab, ac = b - a, c - a
    ap = p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    eps = torch.finfo(points.dtype).eps
    vertex_a = (d1 <= 0) & (d2 <= 0)
    vertex_b = (d3 >= 0) & (d4 <= d3)
    vc = d1 * d4 - d3 * d2
    edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v_ab = d1 / (d1 - d3).clamp_min(eps)
    vertex_c = (d6 >= 0) & (d5 <= d6)
    vb = d5 * d2 - d1 * d6
    edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w_ac = d2 / (d2 - d6).clamp_min(eps)
    va = d3 * d6 - d5 * d4
    edge_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(eps)
    denom = (va + vb + vc).clamp_min(eps)
    v_face, w_face = vb / denom, vc / denom
    closest = a
    closest = torch.where(vertex_b[..., None], b, closest)
    closest = torch.where(edge_ab[..., None], a + v_ab[..., None] * ab, closest)
    closest = torch.where(vertex_c[..., None], c, closest)
    closest = torch.where(edge_ac[..., None], a + w_ac[..., None] * ac, closest)
    closest = torch.where(edge_bc[..., None], b + w_bc[..., None] * (c - b), closest)
    face = ~(vertex_a | vertex_b | edge_ab | vertex_c | edge_ac | edge_bc)
    closest = torch.where(face[..., None], a + v_face[..., None] * ab + w_face[..., None] * ac, closest)
    distance2 = ((p - closest) ** 2).sum(-1)
    return closest, torch.where(valid, distance2, torch.full_like(distance2, torch.inf))


def _parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"OBJ mesh does not exist: {path}")
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    groups: dict[str, list[int]] = {"ungrouped": []}
    group = "ungrouped"
    for raw in path.read_text(encoding="ascii").splitlines():
        fields = raw.split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "g":
            group = "_".join(fields[1:]) if len(fields) > 1 else "unnamed_group"
            groups.setdefault(group, [])
        elif fields[0] == "f" and len(fields) >= 4:
            indices = [int(field.split("/")[0]) - 1 for field in fields[1:]]
            for index in range(1, len(indices) - 1):
                faces.append((indices[0], indices[index], indices[index + 1]))
                groups[group].append(len(faces) - 1)
    if not vertices or not faces:
        raise ValueError(f"OBJ contains no usable vertices or faces: {path}")
    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    if face_array.min() < 0 or face_array.max() >= len(vertex_array):
        raise ValueError(f"OBJ face index is outside its vertex array: {path}")
    return vertex_array, face_array, {
        name: np.asarray(ids, dtype=np.int64) for name, ids in groups.items() if ids
    }


def _select_group(groups: dict[str, np.ndarray], keyword: str) -> np.ndarray:
    selected = [ids for name, ids in groups.items() if keyword.lower() in name.lower()]
    if not selected:
        raise KeyError(f"OBJ mesh has no face group containing '{keyword}'. Groups: {list(groups)}")
    return np.concatenate(selected)


@dataclass
class MeshGeometry:
    obj_path: Path
    vertices: torch.Tensor
    face_indices: torch.Tensor
    triangles: torch.Tensor
    normals: torch.Tensor
    triangle_obj_ids: torch.Tensor
    cell_size: float
    origin: torch.Tensor
    dimensions: tuple[int, int, int]
    candidates: torch.Tensor
    local_neighbours: torch.Tensor
    initial_ring: int = 1
    max_ring: int = 8

    @classmethod
    def from_obj(
        cls, path: str | Path, protection_keyword: str = "protection",
        device: torch.device | None = None, cell_size: float = 0.0,
        initial_ring: int = 1,
    ) -> "MeshGeometry":
        device = device or torch.device("cpu")
        path = Path(path).resolve()
        vertices_np, faces_np, groups = _parse_obj(path)
        protection_ids = _select_group(groups, protection_keyword)
        triangles_np = vertices_np[faces_np[protection_ids]]
        raw_normals = np.cross(triangles_np[:, 1] - triangles_np[:, 0], triangles_np[:, 2] - triangles_np[:, 0])
        lengths = np.linalg.norm(raw_normals, axis=1)
        if np.any(lengths < 1e-12):
            raise ValueError("OBJ protection group contains a degenerate triangle")
        normals_np = raw_normals / lengths[:, None]
        centroids = triangles_np.mean(axis=1)
        outward = np.sum(normals_np[:, 1:] * centroids[:, 1:], axis=1) > 0.0
        normals_np[outward] *= -1.0
        if np.any(np.sum(normals_np[:, 1:] * centroids[:, 1:], axis=1) >= -1e-12):
            raise ValueError("could not orient every protection-face normal toward tube lumen")
        edges = np.stack((
            np.linalg.norm(triangles_np[:, 1] - triangles_np[:, 0], axis=1),
            np.linalg.norm(triangles_np[:, 2] - triangles_np[:, 1], axis=1),
            np.linalg.norm(triangles_np[:, 0] - triangles_np[:, 2], axis=1),
        ), axis=1)
        chosen_cell = float(cell_size) if cell_size > 0.0 else float(np.median(np.min(edges, axis=1)))
        if not np.isfinite(chosen_cell) or chosen_cell <= 0.0:
            raise ValueError("mesh cell size must be positive")
        pad = (initial_ring + 2) * chosen_cell
        lower = triangles_np.reshape(-1, 3).min(axis=0) - pad
        upper = triangles_np.reshape(-1, 3).max(axis=0) + pad
        dims_np = np.maximum(np.ceil((upper - lower) / chosen_cell).astype(int), 1)
        dimensions = tuple(int(value) for value in dims_np)
        cells: dict[int, list[int]] = {}
        tri_min, tri_max = triangles_np.min(1), triangles_np.max(1)
        for face_id, (lo_raw, hi_raw) in enumerate(zip(tri_min, tri_max)):
            lo = np.clip(np.floor((lo_raw - lower) / chosen_cell).astype(int), 0, dims_np - 1)
            hi = np.clip(np.floor((hi_raw - lower) / chosen_cell).astype(int), 0, dims_np - 1)
            for cell in product(range(lo[0], hi[0] + 1), range(lo[1], hi[1] + 1), range(lo[2], hi[2] + 1)):
                flat = (cell[0] * dimensions[1] + cell[1]) * dimensions[2] + cell[2]
                cells.setdefault(flat, []).append(face_id)
        max_per_cell = max(len(ids) for ids in cells.values())
        dense = np.full((math.prod(dimensions), max_per_cell), -1, dtype=np.int64)
        for flat, ids in cells.items():
            dense[flat, :len(ids)] = ids

        # Face adjacency is used for cheap cache validation; the grid remains the reliable fallback.
        edge_owners: dict[tuple[int, int], list[int]] = {}
        for face_id, face in enumerate(faces_np[protection_ids]):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge_owners.setdefault(tuple(sorted((int(a), int(b)))), []).append(face_id)
        adjacency = [set([i]) for i in range(len(triangles_np))]
        for owners in edge_owners.values():
            if len(owners) == 2:
                adjacency[owners[0]].add(owners[1])
                adjacency[owners[1]].add(owners[0])
        neighbours = []
        for face_id, direct in enumerate(adjacency):
            expanded = set(direct)
            for other in direct:
                expanded.update(adjacency[other])
            neighbours.append(sorted(expanded))
        max_neighbours = max(map(len, neighbours))
        neighbour_table = np.full((len(neighbours), max_neighbours), -1, dtype=np.int64)
        for index, ids in enumerate(neighbours):
            neighbour_table[index, :len(ids)] = ids
        tensor = lambda value, dtype: torch.as_tensor(value, dtype=dtype, device=device)
        return cls(
            obj_path=path,
            vertices=tensor(vertices_np, torch.float64),
            face_indices=tensor(faces_np, torch.long),
            triangles=tensor(triangles_np, torch.float64),
            normals=tensor(normals_np, torch.float64),
            triangle_obj_ids=tensor(protection_ids, torch.long),
            cell_size=chosen_cell,
            origin=tensor(lower, torch.float64),
            dimensions=dimensions,
            candidates=tensor(dense, torch.long),
            local_neighbours=tensor(neighbour_table, torch.long),
            initial_ring=initial_ring,
            max_ring=max(initial_ring, 8),
        )

    def _grid_candidates(self, points: torch.Tensor, ring: int | None = None) -> torch.Tensor:
        dims = torch.as_tensor(self.dimensions, dtype=torch.long, device=points.device)
        base = torch.floor((points - self.origin) / self.cell_size).to(torch.long)
        ring = self.initial_ring if ring is None else int(ring)
        offsets_1 = torch.stack(torch.meshgrid(
            torch.arange(-ring, ring + 1, device=points.device),
            torch.arange(-ring, ring + 1, device=points.device),
            torch.arange(-ring, ring + 1, device=points.device),
            indexing="ij",
        ), dim=-1).reshape(-1, 3)
        if len(points) == 0:
            return torch.empty(
                (0, len(offsets_1) * self.candidates.shape[1]),
                dtype=torch.long, device=points.device,
            )
        cells = base[:, None, :] + offsets_1[None, :, :]
        valid = torch.all((cells >= 0) & (cells < dims), dim=-1)
        safe = torch.minimum(torch.maximum(cells, torch.zeros_like(cells)), dims - 1)
        flat = (safe[..., 0] * self.dimensions[1] + safe[..., 1]) * self.dimensions[2] + safe[..., 2]
        ids = self.candidates[flat].reshape(len(points), -1)
        valid = valid[..., None].expand(-1, -1, self.candidates.shape[1]).reshape(len(points), -1) & (ids >= 0)
        sentinel = len(self.triangles)
        values = torch.where(valid, ids, torch.full_like(ids, sentinel))
        values = torch.sort(values, dim=1).values
        unique = values < sentinel
        if values.shape[1] > 1:
            unique[:, 1:] &= values[:, 1:] != values[:, :-1]
        return torch.where(unique, values, torch.full_like(values, -1))

    def nearest(self, points: torch.Tensor, cache: MeshContactCache | None = None) -> MeshContactDetails:
        points = points.detach()
        if points.dtype != torch.float64:
            points = points.to(torch.float64)
        count = len(points)
        local = torch.full((count,), -1, dtype=torch.long, device=points.device)
        candidate_count = torch.zeros(count, dtype=torch.long, device=points.device)
        used_cache = torch.zeros(count, dtype=torch.bool, device=points.device)
        if cache is not None and len(cache.triangle_local) == count:
            valid_cache = cache.triangle_local >= 0
            safe = cache.triangle_local.clamp_min(0)
            neighbours = self.local_neighbours[safe]
            valid = valid_cache[:, None] & (neighbours >= 0)
            safe_neighbours = neighbours.clamp_min(0)
            _, d2 = _closest_points_on_triangles(points, self.triangles[safe_neighbours], valid)
            best_d2, best_col = d2.min(dim=1)
            best = safe_neighbours.gather(1, best_col[:, None]).squeeze(1)
            displacement = torch.linalg.vector_norm(points - cache.last_positions, dim=1)
            used_cache = valid_cache & torch.isfinite(best_d2) & (displacement <= 0.25 * self.cell_size)
            local[used_cache] = best[used_cache]
            candidate_count[used_cache] = valid[used_cache].sum(dim=1)
            cache.hit_count += int(used_cache.sum())
            cache.miss_count += int((~used_cache).sum())
        unresolved = ~used_cache
        search_ring = torch.zeros(count, dtype=torch.long, device=points.device)
        for ring in range(self.initial_ring, self.max_ring + 1):
            if not bool(unresolved.any()):
                break
            target = torch.nonzero(unresolved, as_tuple=False).squeeze(1)
            ids = self._grid_candidates(points[target], ring)
            valid = ids >= 0
            if not len(ids):
                continue
            safe_ids = ids.clamp_min(0)
            _, d2 = _closest_points_on_triangles(points[target], self.triangles[safe_ids], valid)
            best_d2, best_col = d2.min(dim=1)
            best = safe_ids.gather(1, best_col[:, None]).squeeze(1)
            has_candidate = valid.any(dim=1)
            dims = torch.as_tensor(self.dimensions, dtype=torch.long, device=points.device)
            base = torch.floor((points[target] - self.origin) / self.cell_size).to(torch.long)
            lower_cell = torch.maximum(base - ring, torch.zeros_like(base))
            upper_cell = torch.minimum(base + ring, dims - 1)
            lower = self.origin + lower_cell.to(points.dtype) * self.cell_size
            upper = self.origin + (upper_cell.to(points.dtype) + 1.0) * self.cell_size
            boundary_distance = torch.minimum(points[target] - lower, upper - points[target]).amin(dim=1).clamp_min(0.0)
            exact = has_candidate & (torch.sqrt(best_d2) <= boundary_distance + 1.0e-12)
            if ring == self.max_ring:
                # If the bounded search cannot certify the nearest face, use the
                # exact all-face fallback below instead of returning a wrong face.
                exact &= has_candidate & (torch.sqrt(best_d2) <= boundary_distance + 1.0e-12)
            local[target[exact]] = best[exact]
            candidate_count[target] = valid.sum(dim=1)
            search_ring[target[exact]] = ring
            unresolved[target[exact]] = False
        # If a point is outside the first grid ring, use a bounded all-face fallback.
        unresolved = local < 0
        if bool(unresolved.any()):
            _, d2 = _closest_points_on_triangles(points[unresolved], self.triangles,
                                                  torch.ones((int(unresolved.sum()), len(self.triangles)), dtype=torch.bool, device=points.device))
            best_d2, best = d2.min(dim=1)
            target = torch.nonzero(unresolved, as_tuple=False).squeeze(1)
            local[target] = best
            candidate_count[target] = len(self.triangles)
            search_ring[target] = self.max_ring
            if cache is not None:
                cache.uniform_fallback_count += int(unresolved.sum())
        safe_local = local.clamp_min(0)
        valid_local = local >= 0
        triangles = self.triangles[safe_local]
        closest, d2 = _closest_points_on_triangles(points, triangles[:, None, :, :], valid_local[:, None])
        reference = triangles[:, 0]
        normal = self.normals[safe_local]
        distance = torch.sqrt(d2[:, 0])
        clearance = torch.sum(normal * (points - reference), dim=1)
        obj_ids = self.triangle_obj_ids[safe_local]
        if cache is not None:
            cache.triangle_local.copy_(local)
            cache.last_positions.copy_(points)
        return MeshContactDetails(
            triangle_local=local,
            triangle_obj=obj_ids,
            normal=normal,
            reference=reference,
            clearance=clearance,
            nearest_distance=distance,
            candidate_count=candidate_count,
            search_ring=search_ring,
        )


def fixed_face_contact(
    positions: torch.Tensor, details: MeshContactDetails, integrals: torch.Tensor,
    previous_errors: torch.Tensor, *, dt: float, kp: float, ki: float,
    safe_distance: float, max_pi_force: float, max_integral_state: float,
    leak_rate: float, smooth_length: float, direction_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, MeshContactState]:
    """Evaluate frozen-face PI contact and its analytic 3x3 force Jacobian."""
    del safe_distance, direction_epsilon  # encoded by the mesh surface; retained for API parity
    valid = details.triangle_local >= 0
    clearance = torch.sum(details.normal * (positions - details.reference), dim=1)
    signed = torch.where(valid, -clearance, torch.full_like(clearance, -1.0))
    active = valid & (signed > 0.0)
    beta = 1.0 / smooth_length
    shifted = F.softplus(signed, beta=beta) - math.log(2.0) / beta
    error = torch.where(active, shifted, torch.zeros_like(signed))
    error_prime = torch.where(active, torch.sigmoid(beta * signed), torch.zeros_like(signed))
    penetration = torch.clamp(signed, min=0.0)
    activation = torch.where(active, -torch.expm1(-penetration / smooth_length), torch.zeros_like(signed))
    activation_prime = torch.where(active, torch.exp(-penetration / smooth_length) / smooth_length, torch.zeros_like(signed))
    decay = math.exp(-leak_rate * dt)
    accumulated = integrals + error * dt
    leaked = integrals * decay
    candidate_raw = activation * accumulated + (1.0 - activation) * leaked
    candidate = torch.clamp(candidate_raw, 0.0, max_integral_state)
    candidate_prime_raw = activation_prime * (accumulated - leaked) + activation * error_prime * dt
    candidate_prime = torch.where((candidate_raw >= 0.0) & (candidate_raw <= max_integral_state), candidate_prime_raw, torch.zeros_like(candidate_raw))
    unsaturated = kp * error + ki * candidate
    stop_growth = ((ki > 0.0) & (unsaturated >= max_pi_force) & (candidate > integrals)) | (
        (ki > 0.0) & (error < previous_errors) & (ki * integrals > kp * error) & (candidate > integrals)
    )
    new_integrals = torch.where(stop_growth, integrals, candidate)
    new_integral_prime = torch.where(stop_growth, torch.zeros_like(candidate_prime), candidate_prime)
    raw_force = kp * error + ki * new_integrals
    magnitude = activation * torch.clamp(raw_force, 0.0, max_pi_force)
    raw_prime = kp * error_prime + ki * new_integral_prime
    magnitude_prime = activation_prime * torch.clamp(raw_force, 0.0, max_pi_force)
    magnitude_prime += activation * torch.where((raw_force >= 0.0) & (raw_force <= max_pi_force), raw_prime, torch.zeros_like(raw_prime))
    force = torch.where(active[:, None], details.normal * magnitude[:, None], torch.zeros_like(positions))
    blocks = -magnitude_prime[:, None, None] * details.normal[:, :, None] * details.normal[:, None, :]
    state = MeshContactState(new_integrals, active, activation, error, clearance)
    return force, blocks, state


class MeshContactBackend:
    """Small stateful adapter consumed by a Newton driver."""

    def __init__(self, obj_file: str | Path, *, safe_distance: float, kp: float, ki: float,
                 device: torch.device, protection_keyword: str = "protection") -> None:
        self.obj_file = Path(obj_file).resolve()
        self.safe_distance = float(safe_distance)
        self.kp = float(kp)
        self.ki = float(ki)
        self.device = device
        self.geometry = MeshGeometry.from_obj(self.obj_file, protection_keyword, device)
        self.cache: MeshContactCache | None = None

    def _cache_for(self, count: int) -> MeshContactCache:
        if self.cache is None or len(self.cache.triangle_local) != count:
            self.cache = MeshContactCache.for_nodes(count, self.device)
        return self.cache

    def evaluate(self, positions: torch.Tensor, integrals: torch.Tensor,
                 previous_errors: torch.Tensor, *, dt: float, ki_active: float,
                 max_pi_force: float, max_integral_state: float,
                 leak_rate: float, smooth_length: float, direction_epsilon: float):
        details = self.geometry.nearest(positions, self._cache_for(len(positions)))
        return fixed_face_contact(
            positions, details, integrals, previous_errors, dt=dt,
            kp=self.kp, ki=ki_active, safe_distance=self.safe_distance,
            max_pi_force=max_pi_force, max_integral_state=max_integral_state,
            leak_rate=leak_rate, smooth_length=smooth_length,
            direction_epsilon=direction_epsilon,
        )

    assemble = evaluate

    def snapshot(self):
        return None if self.cache is None else self.cache.snapshot()

    def restore(self, snapshot) -> None:
        if snapshot is not None and self.cache is not None:
            self.cache.restore(snapshot)

    def invalidate(self) -> None:
        if self.cache is not None:
            self.cache.triangle_local.fill_(-1)
            self.cache.last_positions.fill_(float("nan"))


__all__ = [
    "MeshContactBackend", "MeshContactCache", "MeshContactDetails", "MeshContactState",
    "MeshGeometry", "fixed_face_contact",
]
