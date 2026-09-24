"""3-D quasi-static guidewire with OBJ-triangle PI contact.

This is the mesh-contact front end for the high-performance v3 guidewire solver.
The frozen v2 source remains unchanged.  The 50 mm ``protection`` OBJ
surface supplies the contact geometry; the 60 mm outer shell is visual only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import functools
from itertools import product
from pathlib import Path
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import v3
from ..common.cli import cli_arguments


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESH_OBJ = (
    ROOT / "resources" / "tube_obj" / "straight_tube_L1p5m_boundaryR50mm_shellR60mm_32sides_axial2.obj"
)
CONTACT_SMOOTH_LENGTH = v3.CONTACT_SMOOTH_LENGTH
MESH_CACHE_GLOBAL_REFRESH_INTERVAL = 50  # 每隔多少个已接受隐式步用 uniform grid 全量校验一次


def profile_add(args: argparse.Namespace, key: str, seconds: float) -> None:
    profile = getattr(args, "performance_profile", None)
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + seconds


@dataclass
class ObjMesh:
    vertices: np.ndarray
    triangles: np.ndarray
    groups: dict[str, np.ndarray]


@dataclass
class MeshContactDetails:
    triangle_local: torch.Tensor       # (N,), -1 means the node is far inside the tube
    triangle_obj: torch.Tensor         # (N,), original OBJ triangle number
    normal: torch.Tensor               # (N, 3), inward normal
    reference: torch.Tensor            # (N, 3), a point on the selected face
    clearance: torch.Tensor            # (N,), positive on the lumen side
    nearest_distance: torch.Tensor     # (N,), Euclidean point-to-triangle distance
    candidate_count: torch.Tensor      # (N,), broad-phase candidates inspected
    search_ring: torch.Tensor          # (N,), final uniform-grid search ring


@dataclass
class MeshContactEvaluation:
    """Contact details evaluated for one exact positions Tensor object."""

    positions: torch.Tensor
    details: MeshContactDetails


@dataclass
class UniformTriangleGrid:
    """Static protection-surface grid, stored on the same device as the solver."""

    obj_path: Path
    cell_size: float
    origin: torch.Tensor               # (3,)
    dimensions: tuple[int, int, int]
    candidates: torch.Tensor           # (cell_count, max_per_cell), local face IDs or -1
    triangles: torch.Tensor            # (M, 3, 3)
    normals: torch.Tensor              # (M, 3), all point into the tube lumen
    obj_triangle_ids: torch.Tensor     # (M,)
    initial_ring: int
    max_ring: int
    protection_face_count: int
    max_candidates_per_cell: int
    local_neighbours: torch.Tensor      # self + one/two edge-hop faces, padded with -1


@dataclass
class MeshContactCache:
    triangle_local: torch.Tensor
    last_positions: torch.Tensor
    hit_count: int = 0
    miss_count: int = 0
    uniform_fallback_count: int = 0
    global_refresh_count: int = 0


def load_obj_triangles(path: Path) -> ObjMesh:
    """Read OBJ vertices, faces, and named face groups without extra packages."""
    if not path.is_file():
        raise FileNotFoundError(f"OBJ mesh does not exist: {path}")
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    grouped_faces: dict[str, list[int]] = {"ungrouped": []}
    current_group = "ungrouped"
    for raw_line in path.read_text(encoding="ascii").splitlines():
        tokens = raw_line.split()
        if not tokens or tokens[0].startswith("#"):
            continue
        if tokens[0] == "v" and len(tokens) >= 4:
            vertices.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
        elif tokens[0] == "g":
            current_group = "_".join(tokens[1:]) if len(tokens) > 1 else "unnamed_group"
            grouped_faces.setdefault(current_group, [])
        elif tokens[0] == "f" and len(tokens) >= 4:
            indices = [int(token.split("/")[0]) - 1 for token in tokens[1:]]
            for index in range(1, len(indices) - 1):
                faces.append((indices[0], indices[index], indices[index + 1]))
                grouped_faces[current_group].append(len(faces) - 1)
    if not vertices or not faces:
        raise ValueError(f"OBJ contains no usable vertices or faces: {path}")
    vertex_array = np.asarray(vertices, dtype=np.float64)
    triangle_array = np.asarray(faces, dtype=np.int64)
    if triangle_array.min() < 0 or triangle_array.max() >= len(vertex_array):
        raise ValueError(f"OBJ face index is outside its vertex array: {path}")
    return ObjMesh(
        vertices=vertex_array,
        triangles=triangle_array,
        groups={name: np.asarray(indices, dtype=np.int64) for name, indices in grouped_faces.items() if indices},
    )


def mesh_group_indices(mesh: ObjMesh, keyword: str) -> np.ndarray:
    selected = [indices for name, indices in mesh.groups.items() if keyword.lower() in name.lower()]
    if not selected:
        raise KeyError(f"OBJ mesh has no face group containing '{keyword}'. Groups: {list(mesh.groups)}")
    return np.concatenate(selected)


def validate_mesh_alignment(mesh: ObjMesh, args: argparse.Namespace) -> tuple[float, float]:
    """Validate the supplied two-layer straight tube against the simulation parameters."""
    x_min, x_max = float(mesh.vertices[:, 0].min()), float(mesh.vertices[:, 0].max())
    radial = np.linalg.norm(mesh.vertices[:, 1:], axis=1)
    protection_radius = args.tube_radius - args.safe_distance
    outer_radius = args.tube_radius
    tolerance = 1e-8
    if abs(x_min) > tolerance or abs(x_max - args.total_length) > tolerance:
        raise ValueError(
            f"OBJ x range [{x_min:g}, {x_max:g}] m does not match simulation [0, {args.total_length:g}] m"
        )
    protection_faces = mesh_group_indices(mesh, "protection")
    protection_vertices = mesh.vertices[mesh.triangles[protection_faces].ravel()]
    if np.max(np.abs(np.linalg.norm(protection_vertices[:, 1:], axis=1) - protection_radius)) > tolerance:
        raise ValueError(f"OBJ protection group does not lie on radius {protection_radius:g} m")
    if np.min(np.abs(radial - outer_radius)) > tolerance:
        raise ValueError(f"OBJ has no layer matching outer radius {outer_radius:g} m")
    return protection_radius, outer_radius


def _flat_cell_index(cell: tuple[int, int, int], dimensions: tuple[int, int, int]) -> int:
    return (cell[0] * dimensions[1] + cell[1]) * dimensions[2] + cell[2]


def _auto_grid_cell_size(triangles: np.ndarray) -> float:
    # Long axial elements must not inflate the uniform-grid size.  One local
    # characteristic length per triangle is robust for sparse axial OBJ meshes.
    edges = np.stack((
        np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
        np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
        np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
    ), axis=1)
    return float(np.median(np.min(edges, axis=1)))


def build_uniform_triangle_grid(
    mesh: ObjMesh,
    mesh_path: Path,
    args: argparse.Namespace,
    cell_size_override: float,
    initial_ring: int,
) -> UniformTriangleGrid:
    """Build a sparse CPU grid once, then upload its padded table to CPU/CUDA."""
    protection_obj_ids = mesh_group_indices(mesh, "protection")
    triangles_np = mesh.vertices[mesh.triangles[protection_obj_ids]]
    if len(triangles_np) == 0:
        raise ValueError("OBJ protection group contains no triangles")
    normals_np = np.cross(triangles_np[:, 1] - triangles_np[:, 0], triangles_np[:, 2] - triangles_np[:, 0])
    normal_lengths = np.linalg.norm(normals_np, axis=1)
    if np.any(normal_lengths < 1e-12):
        raise ValueError("OBJ protection group contains a degenerate triangle")
    normals_np /= normal_lengths[:, None]
    centroids = triangles_np.mean(axis=1)
    # The lumen is around the x-axis.  A valid inner-wall normal must face it.
    outward = np.sum(normals_np[:, 1:] * centroids[:, 1:], axis=1) > 0.0
    normals_np[outward] *= -1.0
    if np.any(np.sum(normals_np[:, 1:] * centroids[:, 1:], axis=1) >= -1e-12):
        raise ValueError("could not orient every protection-face normal toward the tube lumen")

    cell_size = float(cell_size_override) if cell_size_override > 0.0 else _auto_grid_cell_size(triangles_np)
    if not np.isfinite(cell_size) or cell_size <= 0.0:
        raise ValueError("mesh grid cell size must be positive")
    # The pad covers the complete 50--60 mm shell and two extra search rings.
    pad = args.safe_distance + (initial_ring + 2) * cell_size
    lower = triangles_np.reshape(-1, 3).min(axis=0) - pad
    upper = triangles_np.reshape(-1, 3).max(axis=0) + pad
    dimensions_array = np.ceil((upper - lower) / cell_size).astype(int)
    dimensions_array = np.maximum(dimensions_array, 1)
    dimensions = tuple(int(value) for value in dimensions_array)

    cells: dict[int, list[int]] = {}
    tri_min = triangles_np.min(axis=1)
    tri_max = triangles_np.max(axis=1)
    for local_id, (bounds_min, bounds_max) in enumerate(zip(tri_min, tri_max)):
        lo = np.floor((bounds_min - lower) / cell_size).astype(int)
        hi = np.floor((bounds_max - lower) / cell_size).astype(int)
        lo = np.clip(lo, 0, dimensions_array - 1)
        hi = np.clip(hi, 0, dimensions_array - 1)
        for cell in product(range(lo[0], hi[0] + 1), range(lo[1], hi[1] + 1), range(lo[2], hi[2] + 1)):
            cells.setdefault(_flat_cell_index(cell, dimensions), []).append(local_id)
    max_per_cell = max(len(ids) for ids in cells.values())
    dense = np.full((math.prod(dimensions), max_per_cell), -1, dtype=np.int64)
    for index, ids in cells.items():
        dense[index, :len(ids)] = ids

    # Static edge adjacency is built once.  Two-hop neighbourhoods exploit
    # temporal coherence when a node crosses a triangle edge.
    edge_owners: dict[tuple[int, int], list[int]] = {}
    protection_vertex_ids = mesh.triangles[protection_obj_ids]
    for face_id, vertices in enumerate(protection_vertex_ids):
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(vertices[first]), int(vertices[second]))))
            edge_owners.setdefault(edge, []).append(face_id)
    adjacency = [set([index]) for index in range(len(triangles_np))]
    for owners in edge_owners.values():
        if len(owners) == 2:
            adjacency[owners[0]].add(owners[1])
            adjacency[owners[1]].add(owners[0])
    two_hop = []
    for face_id, direct in enumerate(adjacency):
        expanded = set(direct)
        for neighbour in direct:
            expanded.update(adjacency[neighbour])
        two_hop.append(sorted(expanded))
    max_neighbours = max(map(len, two_hop))
    neighbour_table = np.full((len(two_hop), max_neighbours), -1, dtype=np.int64)
    for face_id, neighbours in enumerate(two_hop):
        neighbour_table[face_id, :len(neighbours)] = neighbours

    # Eight rings safely cover the tube radius for this 32-sided mesh; no global scan is used at runtime.
    max_ring = max(initial_ring, int(math.ceil((args.tube_radius + args.safe_distance) / cell_size)) + 2)
    device = args.torch_device
    return UniformTriangleGrid(
        obj_path=mesh_path.resolve(),
        cell_size=cell_size,
        origin=torch.as_tensor(lower, dtype=v3.DTYPE, device=device),
        dimensions=dimensions,
        candidates=torch.as_tensor(dense, dtype=torch.long, device=device),
        triangles=torch.as_tensor(triangles_np, dtype=v3.DTYPE, device=device),
        normals=torch.as_tensor(normals_np, dtype=v3.DTYPE, device=device),
        obj_triangle_ids=torch.as_tensor(protection_obj_ids, dtype=torch.long, device=device),
        initial_ring=initial_ring,
        max_ring=max_ring,
        protection_face_count=len(triangles_np),
        max_candidates_per_cell=max_per_cell,
        local_neighbours=torch.as_tensor(neighbour_table, dtype=torch.long, device=device),
    )


@functools.lru_cache(maxsize=32)
def _ring_offsets(ring: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(-ring, ring + 1, dtype=torch.long, device=device)
    return torch.stack(torch.meshgrid(values, values, values, indexing="ij"), dim=-1).reshape(-1, 3)


def _gather_grid_candidates(points: torch.Tensor, grid: UniformTriangleGrid, ring: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather a cube of nearby grid cells; invalid entries are marked -1."""
    dims = torch.as_tensor(grid.dimensions, dtype=torch.long, device=points.device)
    base = torch.floor((points - grid.origin) / grid.cell_size).to(torch.long)
    offsets = _ring_offsets(ring, points.device)
    cells = base[:, None, :] + offsets[None, :, :]
    valid_cells = torch.all((cells >= 0) & (cells < dims), dim=-1)
    safe_cells = torch.minimum(torch.maximum(cells, torch.zeros_like(cells)), dims - 1)
    flat = (safe_cells[..., 0] * grid.dimensions[1] + safe_cells[..., 1]) * grid.dimensions[2] + safe_cells[..., 2]
    ids = grid.candidates[flat].reshape(len(points), -1)
    valid = (ids >= 0) & valid_cells[..., None].expand(-1, -1, grid.max_candidates_per_cell).reshape(len(points), -1)
    # A long triangle can overlap several neighbouring cells.  Sort and mask
    # duplicate IDs per node so it is evaluated only once in the narrow phase.
    sentinel = grid.protection_face_count
    sortable = torch.where(valid, ids, torch.full_like(ids, sentinel))
    sorted_ids = torch.sort(sortable, dim=1).values
    unique = sorted_ids < sentinel
    if sorted_ids.shape[1] > 1:
        unique[:, 1:] &= sorted_ids[:, 1:] != sorted_ids[:, :-1]
    ids = torch.where(unique, sorted_ids, torch.full_like(sorted_ids, -1))
    return ids, unique, base


def _closest_points_on_triangles(points: torch.Tensor, triangles: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact vectorized point-to-triangle closest points (Ericson region tests)."""
    p = points[:, None, :]
    a, b, c = triangles[..., 0, :], triangles[..., 1, :], triangles[..., 2, :]
    ab, ac = b - a, c - a
    ap = p - a
    d1, d2 = (ab * ap).sum(dim=-1), (ac * ap).sum(dim=-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(dim=-1), (ac * bp).sum(dim=-1)
    cp = p - c
    d5, d6 = (ab * cp).sum(dim=-1), (ac * cp).sum(dim=-1)
    eps = torch.finfo(points.dtype).eps

    vertex_a = (d1 <= 0.0) & (d2 <= 0.0)
    vertex_b = (d3 >= 0.0) & (d4 <= d3)
    vc = d1 * d4 - d3 * d2
    edge_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    v_ab = d1 / (d1 - d3).clamp_min(eps)
    vertex_c = (d6 >= 0.0) & (d5 <= d6)
    vb = d5 * d2 - d1 * d6
    edge_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    w_ac = d2 / (d2 - d6).clamp_min(eps)
    va = d3 * d6 - d5 * d4
    edge_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(eps)
    denom = (va + vb + vc).clamp_min(eps)
    v_face, w_face = vb / denom, vc / denom

    closest = a
    closest = torch.where(vertex_b[..., None], b, closest)
    closest = torch.where(edge_ab[..., None], a + v_ab[..., None] * ab, closest)
    closest = torch.where(vertex_c[..., None], c, closest)
    closest = torch.where(edge_ac[..., None], a + w_ac[..., None] * ac, closest)
    closest = torch.where(edge_bc[..., None], b + w_bc[..., None] * (c - b), closest)
    face_region = ~(vertex_a | vertex_b | edge_ab | vertex_c | edge_ac | edge_bc)
    closest = torch.where(face_region[..., None], a + v_face[..., None] * ab + w_face[..., None] * ac, closest)
    distance2 = ((p - closest) ** 2).sum(dim=-1)
    distance2 = torch.where(valid, distance2, torch.full_like(distance2, torch.inf))
    return closest, distance2


def nearest_protection_faces(points: torch.Tensor, grid: UniformTriangleGrid) -> MeshContactDetails:
    """Find exact nearest candidates through expanding uniform-grid rings, never scanning all faces."""
    count = len(points)
    device, dtype = points.device, points.dtype
    local_ids = torch.full((count,), -1, dtype=torch.long, device=device)
    candidate_count = torch.zeros(count, dtype=torch.long, device=device)
    search_ring = torch.zeros(count, dtype=torch.long, device=device)
    nearest_distance = torch.full((count,), torch.inf, dtype=dtype, device=device)
    nearest_point = torch.zeros((count, 3), dtype=dtype, device=device)
    unresolved = torch.ones(count, dtype=torch.bool, device=device)

    for ring in range(grid.initial_ring, grid.max_ring + 1):
        if not bool(torch.any(unresolved)):
            break
        index = torch.nonzero(unresolved, as_tuple=False).squeeze(1)
        # The narrow phase is fully vectorized over query x candidates.  With a
        # dense rod (N=1000) and the full search ring this is ~10^7 point-face
        # pairs, each fp64 temporary ~400 MiB with ~20 alive at once -> OOM on
        # an 8 GiB GPU (the 15 936-face tube mesh is the culprit).  Chunk the
        # unresolved queries so each closest-point batch stays ~4M pairs.
        cube_cells = (2 * ring + 1) ** 3
        candidate_bound = cube_cells * grid.max_candidates_per_cell
        batch = max(32, min(512, 4_000_000 // max(1, candidate_bound)))
        for chunk_start in range(0, len(index), batch):
            chunk_idx = index[chunk_start:chunk_start + batch]
            query = points[chunk_idx]
            ids, valid, base = _gather_grid_candidates(query, grid, ring)
            candidate_count[chunk_idx] = valid.sum(dim=1)
            has_candidate = torch.any(valid, dim=1)
            safe_ids = ids.clamp_min(0)
            triangles = grid.triangles[safe_ids]
            closest, distance2 = _closest_points_on_triangles(query, triangles, valid)
            best_distance2, best_column = distance2.min(dim=1)
            best_ids = safe_ids.gather(1, best_column[:, None]).squeeze(1)
            best_points = closest[torch.arange(len(query), device=device), best_column]

            # Distance from the node to the boundary of the currently searched cube.
            dims = torch.as_tensor(grid.dimensions, dtype=torch.long, device=device)
            lower_cell = torch.maximum(base - ring, torch.zeros_like(base))
            upper_cell = torch.minimum(base + ring, dims - 1)
            lower_bound = grid.origin + lower_cell.to(dtype) * grid.cell_size
            upper_bound = grid.origin + (upper_cell.to(dtype) + 1.0) * grid.cell_size
            outside_lower_bound = torch.minimum(query - lower_bound, upper_bound - query).amin(dim=1).clamp_min(0.0)
            exact = has_candidate & (torch.sqrt(best_distance2) <= outside_lower_bound + 1e-12)
            # At the configured maximum ring, accept the best local candidate and record its ring.
            exact |= has_candidate & (ring == grid.max_ring)
            if bool(torch.any(exact)):
                resolved_index = chunk_idx[exact]
                local_ids[resolved_index] = best_ids[exact]
                nearest_distance[resolved_index] = torch.sqrt(best_distance2[exact])
                nearest_point[resolved_index] = best_points[exact]
                search_ring[resolved_index] = ring
                unresolved[resolved_index] = False

    # A node inside the tube core need not search all the way to the wall.  It is safely inactive.
    # A node outside the padded mesh box is instead a genuine setup/numerical error.
    if bool(torch.any(unresolved)):
        index = torch.nonzero(unresolved, as_tuple=False).squeeze(1)
        position = points[index]
        bounds_min = grid.origin
        bounds_max = grid.origin + torch.as_tensor(grid.dimensions, dtype=dtype, device=device) * grid.cell_size
        outside = torch.any((position < bounds_min) | (position > bounds_max), dim=1)
        if bool(torch.any(outside)):
            raise RuntimeError("a guidewire node left the padded uniform-grid domain; increase mesh search coverage")
        search_ring[index] = grid.max_ring

    valid_face = local_ids >= 0
    safe_local = local_ids.clamp_min(0)
    normals = grid.normals[safe_local] * valid_face[:, None]
    references = grid.triangles[safe_local, 0]
    references = torch.where(valid_face[:, None], references, torch.zeros_like(references))
    clearance = torch.sum(normals * (points - references), dim=1)
    clearance = torch.where(valid_face, clearance, torch.full_like(clearance, torch.inf))
    obj_ids = torch.full_like(local_ids, -1)
    obj_ids[valid_face] = grid.obj_triangle_ids[safe_local[valid_face]]
    return MeshContactDetails(
        triangle_local=local_ids,
        triangle_obj=obj_ids,
        normal=normals,
        reference=references,
        clearance=clearance,
        nearest_distance=nearest_distance,
        candidate_count=candidate_count,
        search_ring=search_ring,
    )


def current_mesh_topology(positions: torch.Tensor, args: argparse.Namespace) -> MeshContactDetails:
    """Use the face topology frozen at a Newton iteration, if one is active."""
    started = time.perf_counter()
    frozen = getattr(args, "_frozen_mesh_topology", None)
    if frozen is not None:
        return frozen
    cache = getattr(args, "mesh_contact_cache", None)
    if cache is None or len(cache.triangle_local) != len(positions):
        result = nearest_protection_faces(positions.detach(), args.mesh_collision)
        profile_add(args, "mesh_uniform_grid_search_s", time.perf_counter() - started)
        return result
    points = positions.detach()
    grid = args.mesh_collision
    cached_valid = cache.triangle_local >= 0
    result = nearest_protection_faces(points, grid) if not bool(torch.any(cached_valid)) else None
    if result is not None:
        cache.triangle_local = result.triangle_local.clone()
        cache.last_positions = points.clone()
        cache.miss_count += len(points)
        cache.uniform_fallback_count += len(points)
        profile_add(args, "mesh_uniform_grid_search_s", time.perf_counter() - started)
        return result

    safe_cached = cache.triangle_local.clamp_min(0)
    local_ids = grid.local_neighbours[safe_cached]
    local_valid = (local_ids >= 0) & cached_valid[:, None]
    safe_local = local_ids.clamp_min(0)
    closest, distance2 = _closest_points_on_triangles(points, grid.triangles[safe_local], local_valid)
    best_d2, best_column = distance2.min(dim=1)
    best_ids = safe_local.gather(1, best_column[:, None]).squeeze(1)

    displacement = torch.linalg.vector_norm(points - cache.last_positions, dim=1)
    # A quarter-cell motion cannot skip the cached face plus its two edge-hop
    # neighbourhood on this manifold mesh.  Larger moves use the reliable grid fallback.
    hit = cached_valid & torch.isfinite(best_d2) & (displacement <= 0.25 * grid.cell_size)
    miss = ~hit
    cache.hit_count += int(torch.count_nonzero(hit))
    cache.miss_count += int(torch.count_nonzero(miss))
    cache.uniform_fallback_count += int(torch.count_nonzero(miss))

    # Construct local-hit details, then replace uncertain nodes with reliable uniform-grid results.
    normals = grid.normals[best_ids] * hit[:, None]
    references = grid.triangles[best_ids, 0]
    clearance = torch.sum(normals * (points - references), dim=1)
    nearest_distance = torch.sqrt(best_d2)
    triangle_local = torch.where(hit, best_ids, torch.full_like(best_ids, -1))
    triangle_obj = torch.where(hit, grid.obj_triangle_ids[best_ids], torch.full_like(best_ids, -1))
    candidate_count = local_valid.sum(dim=1)
    search_ring = torch.zeros_like(best_ids)
    if bool(torch.any(miss)):
        miss_result = nearest_protection_faces(points[miss], grid)
        triangle_local[miss] = miss_result.triangle_local
        triangle_obj[miss] = miss_result.triangle_obj
        normals[miss] = miss_result.normal
        references[miss] = miss_result.reference
        clearance[miss] = miss_result.clearance
        nearest_distance[miss] = miss_result.nearest_distance
        candidate_count[miss] = miss_result.candidate_count
        search_ring[miss] = miss_result.search_ring
    cache.triangle_local = triangle_local.clone()
    cache.last_positions = points.clone()
    profile_add(args, "mesh_cache_validation_s", time.perf_counter() - started)
    return MeshContactDetails(triangle_local, triangle_obj, normals, references, clearance, nearest_distance, candidate_count, search_ring)


def get_mesh_details(
    positions: torch.Tensor,
    args: argparse.Namespace,
    allow_last_evaluation: bool = True,
) -> MeshContactDetails:
    """Reuse details only for the exact same positions Tensor object.

    Frozen Newton topology is handled first and never inserted into this cache:
    its face geometry is reusable across line-search positions, but its stored
    clearance belongs to the position where the topology was selected.
    """
    if getattr(args, "_frozen_mesh_topology", None) is not None:
        return current_mesh_topology(positions, args)

    if allow_last_evaluation:
        cached = getattr(args, "_last_mesh_evaluation", None)
        if cached is not None and cached.positions is positions:
            args.mesh_exact_reuse_hit_count = getattr(args, "mesh_exact_reuse_hit_count", 0) + 1
            return cached.details
        args.mesh_exact_reuse_miss_count = getattr(args, "mesh_exact_reuse_miss_count", 0) + 1

    details = current_mesh_topology(positions, args)
    args._last_mesh_evaluation = MeshContactEvaluation(positions=positions, details=details)
    return details


def _mesh_contact_scalar_response(
    signed_penetration: torch.Tensor,
    has_face: torch.Tensor,
    old_integral: torch.Tensor,
    previous_error: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PI magnitude response to scalar signed penetration."""
    active = has_face & (signed_penetration > 0.0)
    beta = 1.0 / CONTACT_SMOOTH_LENGTH
    # 管腔侧 c>=0 时 PI 力必须严格为零。穿透后使用从零连续起步的
    # shifted-Softplus，避免原实现尚未穿透便产生数 mN 的“提前接触力”。
    shifted_softplus = F.softplus(signed_penetration, beta=beta) - math.log(2.0) / beta
    error = torch.where(active, shifted_softplus, torch.zeros_like(shifted_softplus))
    penetration = torch.clamp(signed_penetration, min=0.0)
    # 严格零力与 Newton 连续性同时满足：腔内 activation=0；穿透侧从 0
    # 连续增长并很快趋近 1。历史积分力也乘此因子，避免跨界瞬间跳变。
    activation = torch.where(
        active, -torch.expm1(-penetration / CONTACT_SMOOTH_LENGTH), torch.zeros_like(signed_penetration)
    )
    weight = activation
    candidate = torch.clamp(
        weight * (old_integral + error * args.dt_active)
        + (1.0 - weight) * old_integral * math.exp(-args.integral_leak_rate * args.dt_active),
        0.0,
        args.max_integral_state,
    )
    unsaturated = args.Kp * error + args.ki_active * candidate
    stop_growth = ((args.ki_active > 0.0) & (unsaturated >= args.max_pi_force) & (candidate > old_integral)) | (
        (args.ki_active > 0.0) & (error < previous_error)
        & (args.ki_active * old_integral > args.Kp * error) & (candidate > old_integral)
    )
    new_integral = torch.where(stop_growth, old_integral, candidate)
    raw_magnitude = torch.clamp(
        args.Kp * error + args.ki_active * new_integral, 0.0, args.max_pi_force
    )
    magnitude = activation * raw_magnitude
    return magnitude, new_integral, error, weight, active


def _mesh_contact_fixed_face(
    position: torch.Tensor,
    normal: torch.Tensor,
    reference: torch.Tensor,
    has_face: torch.Tensor,
    old_integral: torch.Tensor,
    previous_error: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PI update with a fixed selected face; this path is differentiable for Newton."""
    clearance = torch.dot(normal, position - reference)
    signed_penetration = torch.where(has_face, -clearance, torch.full_like(clearance, -1.0))
    magnitude, new_integral, error, weight, active = _mesh_contact_scalar_response(
        signed_penetration, has_face, old_integral, previous_error, args
    )
    force = torch.where(active, normal * magnitude, torch.zeros_like(normal))
    return force, new_integral, error, weight, active, clearance


def _mesh_contact_fixed_face_reference_3d_autograd(
    position: torch.Tensor,
    normal: torch.Tensor,
    reference: torch.Tensor,
    has_face: torch.Tensor,
    old_integral: torch.Tensor,
    previous_error: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frozen pre-optimization contact implementation used only by self-test."""
    clearance = torch.dot(normal, position - reference)
    signed_penetration = torch.where(has_face, -clearance, torch.full_like(clearance, -1.0))
    active = has_face & (signed_penetration > 0.0)
    beta = 1.0 / CONTACT_SMOOTH_LENGTH
    shifted_softplus = F.softplus(signed_penetration, beta=beta) - math.log(2.0) / beta
    error = torch.where(active, shifted_softplus, torch.zeros_like(shifted_softplus))
    penetration = torch.clamp(signed_penetration, min=0.0)
    activation = torch.where(
        active, -torch.expm1(-penetration / CONTACT_SMOOTH_LENGTH), torch.zeros_like(clearance)
    )
    weight = activation
    candidate = torch.clamp(
        weight * (old_integral + error * args.dt_active)
        + (1.0 - weight) * old_integral * math.exp(-args.integral_leak_rate * args.dt_active),
        0.0,
        args.max_integral_state,
    )
    unsaturated = args.Kp * error + args.ki_active * candidate
    stop_growth = ((args.ki_active > 0.0) & (unsaturated >= args.max_pi_force) & (candidate > old_integral)) | (
        (args.ki_active > 0.0) & (error < previous_error)
        & (args.ki_active * old_integral > args.Kp * error) & (candidate > old_integral)
    )
    new_integral = torch.where(stop_growth, old_integral, candidate)
    raw_magnitude = torch.clamp(
        args.Kp * error + args.ki_active * new_integral, 0.0, args.max_pi_force
    )
    magnitude = activation * raw_magnitude
    force = torch.where(active, normal * magnitude, torch.zeros_like(normal))
    return force, new_integral, error, weight, active, clearance


def mesh_contact_quantities(
    positions: torch.Tensor,
    integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    details = get_mesh_details(positions, args)
    values = torch.func.vmap(
        lambda p, n, a, valid, old_i, previous: _mesh_contact_fixed_face(p, n, a, valid, old_i, previous, args)
    )(
        positions, details.normal, details.reference, details.triangle_local >= 0,
        integrals, previous_errors,
    )
    forces, new_integrals, errors, weights, active, clearance = values
    return forces, new_integrals, active, weights, errors, clearance


def _assemble_mesh_contact_scalar_fixed_details(
    positions: torch.Tensor,
    normals: torch.Tensor,
    references: torch.Tensor,
    valid: torch.Tensor,
    integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble force/Jacobian for already-selected fixed mesh faces."""
    clearance = torch.sum(normals * (positions - references), dim=1)
    signed_penetration = torch.where(valid, -clearance, torch.full_like(clearance, -1.0))

    def magnitude_aux(signed: torch.Tensor, has_face: torch.Tensor, old_i: torch.Tensor, previous: torch.Tensor):
        magnitude = _mesh_contact_scalar_response(signed, has_face, old_i, previous, args)[0]
        return magnitude, magnitude

    dm_ds, magnitude = torch.func.vmap(
        torch.func.grad(magnitude_aux, argnums=0, has_aux=True)
    )(signed_penetration, valid, integrals, previous_errors)
    active = valid & (signed_penetration > 0.0)
    forces = torch.where(active[:, None], normals * magnitude[:, None], torch.zeros_like(normals))
    outer_nn = normals[:, :, None] * normals[:, None, :]
    blocks = -dm_ds[:, None, None] * outer_nn
    return forces, blocks


def assemble_mesh_contact_batched(
    positions: torch.Tensor,
    integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    pattern: v3.AssemblyPattern,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return free contact force and complete per-node 3x3 force Jacobian blocks."""
    started = time.perf_counter()
    details = get_mesh_details(positions, args)
    valid = details.triangle_local >= 0
    forces, blocks = _assemble_mesh_contact_scalar_fixed_details(
        positions, details.normal, details.reference, valid, integrals, previous_errors, args
    )
    profile_add(args, "contact_jacobian_s", time.perf_counter() - started)
    return forces.reshape(-1)[pattern.free_indices], blocks


def _assemble_mesh_contact_batched_reference_3d_autograd(
    positions: torch.Tensor,
    normals: torch.Tensor,
    references: torch.Tensor,
    valid: torch.Tensor,
    integrals: torch.Tensor,
    previous_errors: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the old full 3->3 contact Jacobian for regression tests."""
    def force_aux(p: torch.Tensor, n: torch.Tensor, a: torch.Tensor, has_face: torch.Tensor, old_i: torch.Tensor, previous: torch.Tensor):
        force = _mesh_contact_fixed_face_reference_3d_autograd(
            p, n, a, has_face, old_i, previous, args
        )[0]
        return force, force

    blocks, forces = torch.func.vmap(
        torch.func.jacrev(force_aux, argnums=0, has_aux=True)
    )(positions, normals, references, valid, integrals, previous_errors)
    return forces, blocks


def compute_mesh_penetration(positions: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    details = get_mesh_details(positions, args)
    return torch.clamp(-details.clearance, min=0.0)


def assemble_elastic_block_penta(
    positions: torch.Tensor, rest_lengths: torch.Tensor, prescribed_mask: torch.Tensor,
    args: argparse.Namespace, block_plan: v3.BlockAssemblyPlan | None = None,
) -> tuple[torch.Tensor, v3.BlockPentaSystem, int, int]:
    """Use the shared v3 DER block assembly and retain mesh profiling."""
    block_plan = block_plan or v3.get_block_assembly_plan(
        str(positions.device), len(positions), bool(prescribed_mask[-1].all())
    )
    result = v3.assemble_elastic_block_penta_v3(
        positions, rest_lengths, prescribed_mask, args.EA, args.EI, block_plan, args
    )
    return result


def mesh_residual_and_jacobian(
    q: torch.Tensor, old_positions: torch.Tensor, old_velocities: torch.Tensor,
    rest_lengths: torch.Tensor, masses: torch.Tensor, integrals: torch.Tensor,
    previous_errors: torch.Tensor, pulse_forces: torch.Tensor,
    prescribed_mask: torch.Tensor, prescribed: torch.Tensor, args: argparse.Namespace,
    with_jacobian: bool, block_plan: v3.BlockAssemblyPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    free_mask = ~prescribed_mask
    positions = v3.unpack(q, free_mask, prescribed)
    if with_jacobian:
        block_plan = block_plan or v3.get_block_assembly_plan(
            str(positions.device), len(positions), bool(prescribed_mask[-1].all())
        )
        gradient, system, free_start, free_end = assemble_elastic_block_penta(
            positions, rest_lengths, prescribed_mask, args, block_plan
        )
        contact_force, contact_blocks = assemble_mesh_contact_batched(
            positions, integrals, previous_errors, block_plan, args
        )
    else:
        q_grad = q.detach().requires_grad_(True)
        _, _, energy = v3.elastic_energy(v3.unpack(q_grad, free_mask, prescribed), rest_lengths, args.EA, args.EI)
        gradient = torch.autograd.grad(energy, q_grad)[0]
        all_contact, *_ = mesh_contact_quantities(positions, integrals, previous_errors, args)
        contact_force = all_contact[free_mask]
        contact_blocks = None
    q_old, v_old = old_positions[free_mask], old_velocities[free_mask]
    free_masses = masses[:, None].expand_as(old_positions)[free_mask]
    residual = (
        free_masses * (q - q_old - args.dt_active * v_old) / args.dt_active**2
        + gradient + args.damping * (q - q_old) / args.dt_active
        - contact_force - pulse_forces[free_mask]
    )
    if not with_jacobian:
        return residual, None, positions
    mass_damping = (free_masses / args.dt_active**2 + args.damping / args.dt_active).reshape(-1, 3)
    system.diagonal.diagonal(dim1=1, dim2=2).add_(mass_damping)
    system.diagonal.add_(-contact_blocks[free_start:free_end])
    return residual, system, positions


def mesh_implicit_euler_step(
    positions: torch.Tensor, velocities: torch.Tensor, rest_lengths: torch.Tensor, masses: torch.Tensor,
    integrals: torch.Tensor, initial_positions: torch.Tensor, pulse_forces: torch.Tensor,
    tip_constrained: bool, tip_displacement: float, dt_step: float, args: argparse.Namespace,
):
    """v2 implicit Euler/Newton loop, with previous error queried from OBJ faces."""
    args.dt_active = dt_step
    prescribed_mask, prescribed = v3.make_prescribed_state(initial_positions, tip_constrained, tip_displacement)
    free_mask = ~prescribed_mask
    block_plan = v3.get_block_assembly_plan(str(positions.device), len(positions), tip_constrained)
    args._frozen_mesh_topology = None
    accepted_serial = getattr(args, "mesh_accepted_step_count", 0)
    cache = getattr(args, "mesh_contact_cache", None)
    if (
        cache is not None and accepted_serial > 0
        and accepted_serial % MESH_CACHE_GLOBAL_REFRESH_INTERVAL == 0
    ):
        cache.triangle_local.fill_(-1)
        cache.last_positions.fill_(float("nan"))
        cache.global_refresh_count += 1
    with torch.no_grad():
        _, _, _, _, previous_errors, _ = mesh_contact_quantities(
            positions, integrals, torch.zeros_like(integrals), args
        )
    predictor = positions + args.dt_active * velocities
    predictor[prescribed_mask] = prescribed[prescribed_mask]
    q = predictor[free_mask].detach()
    # 一个完整隐式步只选择一次最近面片。Newton 残差、Jacobian 和所有
    # line-search trial 共用同一面片及法向，防止相邻多边形面之间来回切换。
    with torch.no_grad():
        predictor_positions = v3.unpack(q, free_mask, prescribed)
        args._frozen_mesh_topology = current_mesh_topology(predictor_positions, args)
    current_residual, _, _ = mesh_residual_and_jacobian(
        q, positions, velocities, rest_lengths, masses, integrals, previous_errors, pulse_forces,
        prescribed_mask, prescribed, args, False, block_plan,
    )
    residual_norm = float(torch.linalg.norm(current_residual))
    line_search_failures, step_norm = 0, 0.0
    converged, iteration = residual_norm < args.residual_tol, 0
    stagnation_count = 0
    for iteration in range(1, args.max_newton_iters + 1):
        if converged:
            break
        # 面片拓扑在整个隐式步内保持冻结；子步重试会以更小 dt 重新选面。
        residual, jacobian, _ = mesh_residual_and_jacobian(
            q, positions, velocities, rest_lengths, masses, integrals, previous_errors, pulse_forces,
            prescribed_mask, prescribed, args, True, block_plan,
        )
        solve_started = time.perf_counter()
        force_dense_rescue = stagnation_count >= 3
        delta, used_dense_fallback = jacobian.solve(-residual, force_dense=force_dense_rescue)
        profile_add(args, "banded_linear_solve_s", time.perf_counter() - solve_started)
        args.banded_numerical_fallback_count = getattr(args, "banded_numerical_fallback_count", 0) + int(used_dense_fallback)
        if force_dense_rescue:
            stagnation_count = 0
        step_norm = float(torch.linalg.norm(delta))
        if step_norm < args.step_tol:
            converged = True
            break
        base_norm, alpha, accepted = float(torch.linalg.norm(residual)), 1.0, False
        line_search_started = time.perf_counter()
        for _ in range(12):
            trial = (q + alpha * delta).detach()
            trial_residual, _, _ = mesh_residual_and_jacobian(
                trial, positions, velocities, rest_lengths, masses, integrals, previous_errors, pulse_forces,
                prescribed_mask, prescribed, args, False, block_plan,
            )
            trial_norm = float(torch.linalg.norm(trial_residual))
            if np.isfinite(trial_norm) and trial_norm < base_norm:
                q, residual_norm, accepted = trial, trial_norm, True
                break
            alpha *= 0.5
        profile_add(args, "line_search_residual_s", time.perf_counter() - line_search_started)
        if not accepted:
            line_search_failures += 1
            q = (q + 0.05 * delta).detach()
            trial_residual, _, _ = mesh_residual_and_jacobian(
                q, positions, velocities, rest_lengths, masses, integrals, previous_errors, pulse_forces,
                prescribed_mask, prescribed, args, False, block_plan,
            )
            residual_norm = float(torch.linalg.norm(trial_residual))
        stagnation_count = stagnation_count + 1 if residual_norm >= 0.95 * base_norm else 0
        converged = residual_norm < args.residual_tol
    args._frozen_mesh_topology = None
    new_positions = v3.unpack(q, free_mask, prescribed).detach()
    new_velocities = (new_positions - positions) / args.dt_active
    new_velocities[:2] = 0.0
    if tip_constrained:
        new_velocities[-1] = (prescribed[-1] - positions[-1]) / args.dt_active
    with torch.no_grad():
        force, new_integrals, active, weights, errors, distances = mesh_contact_quantities(
            new_positions, integrals, previous_errors, args
        )
    if converged:
        args.mesh_accepted_step_count = accepted_serial + 1
    return (
        new_positions, new_velocities, new_integrals, force, active, weights, errors, distances,
        v3.NewtonInfo(converged, iteration, residual_norm, step_norm, line_search_failures), prescribed_mask,
    )


def add_obj_tube_to_axes(ax, mesh: ObjMesh, face_stride: int) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    definitions = (
        ("protection", "#2f6690", 0.30, "50 mm mesh PI boundary"),
        ("outer_shell", "#526777", 0.42, "60 mm outer shell OBJ"),
    )
    for keyword, color, alpha, label in definitions:
        indices = mesh_group_indices(mesh, keyword)[::face_stride]
        triangles = mesh.vertices[mesh.triangles[indices]]
        ax.add_collection3d(Poly3DCollection(
            triangles, facecolors=color, edgecolors=(0.12, 0.18, 0.25, 0.52),
            linewidths=0.14, alpha=alpha, label=label,
        ))


def configure_scene_axes(ax, args: argparse.Namespace, title: str) -> None:
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title=title)
    ax.set_xlim(0.0, args.total_length)
    ax.set_ylim(-1.15 * args.tube_radius, 1.15 * args.tube_radius)
    ax.set_zlim(-1.15 * args.tube_radius, 1.15 * args.tube_radius)
    ax.set_box_aspect((args.total_length, 6 * args.tube_radius, 6 * args.tube_radius))
    ax.view_init(elev=18, azim=-64)


def _final_mesh_details(history: dict[str, np.ndarray], args: argparse.Namespace) -> MeshContactDetails:
    positions = torch.as_tensor(history["positions"][-1], dtype=v3.DTYPE, device=args.torch_device)
    with torch.no_grad():
        return nearest_protection_faces(positions, args.mesh_collision)


def create_mesh_final_result_figure(history: dict[str, np.ndarray], args: argparse.Namespace):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    positions = history["positions"]
    fig = plt.figure(figsize=(16, 10), dpi=180)
    ax = fig.add_axes([0.02, 0.37, 0.96, 0.60], projection="3d")
    add_obj_tube_to_axes(ax, args.visual_mesh, args.mesh_plot_face_stride)
    ax.plot(*positions[0].T, "k--", lw=1.2, label="initial")
    ax.plot(*positions[-1].T, color="tab:blue", lw=1.9, label="final deformation")
    active = history["contact_active"][-1]
    if np.any(active):
        ax.scatter(*positions[-1, active].T, c="magenta", s=20, label="mesh contact")
        details = _final_mesh_details(history, args)
        normals = details.normal.detach().cpu().numpy()
        active_ids = np.flatnonzero(active)[:12]
        points = positions[-1, active_ids]
        ax.quiver(points[:, 0], points[:, 1], points[:, 2], normals[active_ids, 0], normals[active_ids, 1], normals[active_ids, 2], length=0.025, color="lime", linewidth=1.2, label="inward face normal")
    configure_scene_axes(ax, args, "Final guidewire deformation with OBJ-triangle PI contact")
    ax.legend(loc="upper right")

    load_ax = fig.add_axes([0.03, 0.05, 0.57, 0.23])
    load_ax.axis("off")
    nodes = np.asarray(history["initial_pulse_nodes"], dtype=int)
    x_positions = np.asarray(history["initial_pulse_x_m"], dtype=float)
    magnitudes = np.asarray(history["initial_pulse_magnitudes_n"], dtype=float)
    force_vectors = np.asarray(history["initial_pulse_force_vectors_n"], dtype=float)
    rows = [[str(node), f"{x_value:.4f}", f"{magnitude:.4e}", f"{force[1]:.4e}", f"{force[2]:.4e}"]
            for node, x_value, magnitude, force in zip(nodes, x_positions, magnitudes, force_vectors)]
    table = load_ax.table(cellText=rows, colLabels=["node", "x (m)", "|F| (N)", "Fy (N)", "Fz (N)"], cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.25)
    load_ax.set_title("Random initial pulse loads (all directions are in the y/z plane)", fontsize=10, pad=5)

    energy_ax = fig.add_axes([0.66, 0.07, 0.30, 0.21])
    values = [float(history["elastic_stretch_energy_j"][-1]), float(history["elastic_bend_energy_j"][-1]), float(history["elastic_total_energy_j"][-1])]
    bars = energy_ax.bar(["stretch", "bend", "total"], values, color=["#4c78a8", "#f58518", "#54a24b"])
    for bar, value in zip(bars, values):
        energy_ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2e}", ha="center", va="bottom", fontsize=7)
    energy_ax.set_ylabel("energy (J)")
    energy_ax.set_title("Final system energy", fontsize=10)
    energy_ax.grid(axis="y", alpha=0.25)
    return fig


def visualize_mesh_results(history: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    positions = history["positions"]
    if not args.skip_animation:
        fig_animation = plt.figure(figsize=(13, 5.4), dpi=120)
        ax = fig_animation.add_axes([0.015, 0.07, 0.97, 0.87], projection="3d")
        add_obj_tube_to_axes(ax, args.visual_mesh, args.mesh_plot_face_stride)
        rod_line, = ax.plot([], [], [], color="tab:blue", lw=1.8, label="guidewire")
        contact_scatter = ax.scatter([], [], [], c="magenta", s=18, label="mesh contact")
        configure_scene_axes(ax, args, "3-D guidewire compression animation with OBJ mesh contact")
        ax.legend(loc="upper right")
        frame_indices = np.arange(0, len(positions), max(1, args.animation_stride))
        if frame_indices[-1] != len(positions) - 1:
            frame_indices = np.append(frame_indices, len(positions) - 1)

        def update(frame_number: int):
            frame = int(frame_indices[frame_number])
            current = positions[frame]
            rod_line.set_data(current[:, 0], current[:, 1])
            rod_line.set_3d_properties(current[:, 2])
            contact = current[history["contact_active"][frame]]
            contact_scatter._offsets3d = (contact[:, 0], contact[:, 1], contact[:, 2])
            ax.set_title(f"OBJ-triangle PI contact, t={history['time'][frame]:.3f} s")
            return rod_line, contact_scatter

        animation = FuncAnimation(fig_animation, update, frames=len(frame_indices), interval=1000.0 / args.animation_fps, blit=False)
        fig_animation._guidewire_animation = animation
        plt.show()
        plt.close(fig_animation)
    fig = create_mesh_final_result_figure(history, args)
    plt.show()
    plt.close(fig)


def mesh_resolve_output_paths(args: argparse.Namespace, pulse_nodes: np.ndarray, pulse_magnitudes: np.ndarray):
    """Keep the normal v2 directory layout but never mix mesh and analytic runs."""
    if args.output_prefix is not None:
        return ORIGINAL_RESOLVE_OUTPUT_PATHS(args, pulse_nodes, pulse_magnitudes)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    nodes_tag = "-".join(map(str, pulse_nodes.tolist()))
    run_stem = (
        f"quasi_static_mesh_3d_L{args.total_length:g}m_disp{args.tip_total_displacement * 1e3:g}mm"
        f"_N{args.num_nodes}_pulse{len(pulse_nodes)}_nodes{nodes_tag}"
        f"_F{pulse_magnitudes.min():.3g}-{pulse_magnitudes.max():.3g}_{timestamp}"
    ).replace(".", "p")
    return (
        run_stem,
        v3.NPZ_OUTPUT_DIR / f"{run_stem}.npz",
        v3.PICTURE_OUTPUT_DIR / f"{run_stem}_final.png",
        v3.PICTURE_OUTPUT_DIR / v3.SUMMARY_FILENAME,
    )


def record_mesh_diagnostics(history: dict[str, list], positions: torch.Tensor, args: argparse.Namespace) -> None:
    """Store the already-computed step-end contact topology with each history frame."""
    with torch.no_grad():
        details = get_mesh_details(positions, args)
    clear = details.clearance.detach().cpu().numpy()
    values = {
        "mesh_nearest_triangle_id": details.triangle_obj.detach().cpu().numpy(),
        "mesh_contact_normal": details.normal.detach().cpu().numpy(),
        "mesh_signed_clearance_m": clear,
        "mesh_penetration_m": np.maximum(-clear, 0.0),
        "mesh_candidate_count": details.candidate_count.detach().cpu().numpy(),
        "mesh_search_ring_used": details.search_ring.detach().cpu().numpy(),
    }
    for key, value in values.items():
        history.setdefault(key, []).append(value)


def append_mesh_metadata(history: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    penetration = np.asarray(history["mesh_penetration_m"])
    history.update({
        "max_mesh_penetration_m": np.max(penetration, axis=1),
        "mesh_grid_cell_size_m": np.asarray(args.mesh_collision.cell_size),
        "mesh_grid_dimensions": np.asarray(args.mesh_collision.dimensions, dtype=np.int64),
        "mesh_grid_max_candidates_per_cell": np.asarray(args.mesh_collision.max_candidates_per_cell, dtype=np.int64),
        "mesh_initial_search_ring": np.asarray(args.mesh_collision.initial_ring, dtype=np.int64),
        "mesh_obj_path": np.asarray(str(args.mesh_collision.obj_path)),
        "mesh_protection_face_count": np.asarray(args.mesh_collision.protection_face_count, dtype=np.int64),
        "mesh_cache_hit_count": np.asarray(args.mesh_contact_cache.hit_count, dtype=np.int64),
        "mesh_cache_miss_count": np.asarray(args.mesh_contact_cache.miss_count, dtype=np.int64),
        "mesh_uniform_grid_fallback_count": np.asarray(args.mesh_contact_cache.uniform_fallback_count, dtype=np.int64),
        "mesh_global_refresh_count": np.asarray(args.mesh_contact_cache.global_refresh_count, dtype=np.int64),
        "mesh_exact_reuse_hit_count": np.asarray(
            getattr(args, "mesh_exact_reuse_hit_count", 0), dtype=np.int64
        ),
        "mesh_exact_reuse_miss_count": np.asarray(
            getattr(args, "mesh_exact_reuse_miss_count", 0), dtype=np.int64
        ),
        "mesh_exact_reuse_hit_rate": np.asarray(
            getattr(args, "mesh_exact_reuse_hit_count", 0)
            / max(
                getattr(args, "mesh_exact_reuse_hit_count", 0)
                + getattr(args, "mesh_exact_reuse_miss_count", 0),
                1,
            )
        ),
        "mesh_global_refresh_interval": np.asarray(MESH_CACHE_GLOBAL_REFRESH_INTERVAL, dtype=np.int64),
        "mesh_cache_hit_rate": np.asarray(
            args.mesh_contact_cache.hit_count
            / max(args.mesh_contact_cache.hit_count + args.mesh_contact_cache.miss_count, 1)
        ),
    })
    # Preserve v2's familiar name while making its meaning mesh penetration.
    history["max_protection_error"] = history["max_mesh_penetration_m"].copy()


def mesh_self_test(args: argparse.Namespace) -> None:
    """Geometry, grid-vs-brute-force, force direction, and 3x3-Jacobian checks."""
    grid = args.mesh_collision
    centers = grid.triangles.mean(dim=1)
    assert bool(torch.all(torch.sum(grid.normals[:, 1:] * centers[:, 1:], dim=1) < 0.0))
    x = args.total_length * 0.5
    samples = torch.tensor([
        [x, 0.045, 0.0], [x, 0.051, 0.0], [x, 0.0, 0.051],
        [x, 0.051 / math.sqrt(2), 0.051 / math.sqrt(2)],
    ], dtype=v3.DTYPE, device=args.torch_device)
    zero = torch.zeros(len(samples), dtype=v3.DTYPE, device=args.torch_device)
    args.dt_active = args.dt
    force, _, active, _, _, _ = mesh_contact_quantities(samples, zero, zero, args)
    details = nearest_protection_faces(samples, grid)
    assert not bool(active[0]) and bool(torch.allclose(force[0], torch.zeros(3, dtype=v3.DTYPE, device=args.torch_device)))
    assert bool(torch.all(active[1:]))
    assert bool(torch.all(torch.sum(force[1:] * details.normal[1:], dim=1) > 0.0))
    pattern = v3.get_assembly_pattern(torch.zeros_like(samples, dtype=torch.bool))
    free_force, blocks = assemble_mesh_contact_batched(samples, zero, zero, pattern, args)
    assert bool(torch.isfinite(free_force).all() and torch.isfinite(blocks).all()) and blocks.shape == (len(samples), 3, 3)
    assert abs(float(blocks[3, 1, 2])) > 0.0

    # Exact-state reuse is identity based. A clone must miss even when all
    # values match, and frozen topology must bypass this cache entirely.
    args._last_mesh_evaluation = None
    args.mesh_exact_reuse_hit_count = 0
    args.mesh_exact_reuse_miss_count = 0
    exact_first = get_mesh_details(samples, args)
    exact_second = get_mesh_details(samples, args)
    cloned_samples = samples.clone()
    exact_clone = get_mesh_details(cloned_samples, args)
    assert exact_second is exact_first and exact_clone is not exact_first
    assert args.mesh_exact_reuse_hit_count == 1 and args.mesh_exact_reuse_miss_count == 2
    frozen_hits = args.mesh_exact_reuse_hit_count
    frozen_misses = args.mesh_exact_reuse_miss_count
    args._frozen_mesh_topology = exact_first
    frozen_result = get_mesh_details(samples.clone(), args)
    args._frozen_mesh_topology = None
    assert frozen_result is exact_first
    assert args.mesh_exact_reuse_hit_count == frozen_hits
    assert args.mesh_exact_reuse_miss_count == frozen_misses

    # 紧贴同一面片内外各 1 微米：未穿透必须严格零力，即使保留有历史积分；
    # 穿透侧则必须沿朝内法向产生正恢复力。
    face_center = grid.triangles[0].mean(dim=0)
    face_normal = grid.normals[0]
    near_positions = torch.stack((face_center + 1.0e-6 * face_normal, face_center - 1.0e-6 * face_normal))
    near_normals = face_normal.expand(2, -1)
    near_references = grid.triangles[0, 0].expand(2, -1)
    near_integrals = torch.full((2,), 1.0e-3, dtype=v3.DTYPE, device=args.torch_device)
    near_values = torch.func.vmap(
        lambda p, n, a, old_i: _mesh_contact_fixed_face(
            p, n, a, torch.ones((), dtype=torch.bool, device=p.device), old_i,
            torch.zeros((), dtype=v3.DTYPE, device=p.device), args,
        )
    )(near_positions, near_normals, near_references, near_integrals)
    near_forces, _, _, _, near_active, _ = near_values
    assert not bool(near_active[0]) and bool(torch.equal(near_forces[0], torch.zeros_like(near_forces[0])))
    assert bool(near_active[1]) and float(torch.dot(near_forces[1], face_normal)) > 0.0

    # Production scalar derivative must match the frozen 3->3 autograd
    # reference across inactive, smooth onset, integral, saturation, and
    # anti-windup regimes, including a fully coupled arbitrary face normal.
    signed_cases = torch.tensor(
        [-1.0e-3, -1.0e-6, 1.0e-6, 2.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-4],
        dtype=v3.DTYPE, device=args.torch_device,
    )
    case_count = len(signed_cases)
    arbitrary_normal = torch.tensor(
        [1.0, 1.0, 1.0], dtype=v3.DTYPE, device=args.torch_device
    ) / math.sqrt(3.0)
    case_normals = arbitrary_normal.expand(case_count, -1).clone()
    case_references = torch.zeros((case_count, 3), dtype=v3.DTYPE, device=args.torch_device)
    case_positions = case_references - signed_cases[:, None] * case_normals
    case_valid = torch.tensor(
        [False, True, True, True, True, True, True, True],
        dtype=torch.bool, device=args.torch_device,
    )
    case_integrals = torch.tensor(
        [0.0, 0.0, 0.0, 1.0e-3, 2.0e-2, 4.0e-2, 4.0e-2, 5.0e-2],
        dtype=v3.DTYPE, device=args.torch_device,
    )
    case_previous = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 2.0e-3, 0.0, 0.0, 1.0e-3],
        dtype=v3.DTYPE, device=args.torch_device,
    )
    new_case_values = torch.func.vmap(
        lambda p, n, a, valid, old_i, previous: _mesh_contact_fixed_face(
            p, n, a, valid, old_i, previous, args
        )
    )(case_positions, case_normals, case_references, case_valid, case_integrals, case_previous)
    old_case_values = torch.func.vmap(
        lambda p, n, a, valid, old_i, previous: _mesh_contact_fixed_face_reference_3d_autograd(
            p, n, a, valid, old_i, previous, args
        )
    )(case_positions, case_normals, case_references, case_valid, case_integrals, case_previous)
    for new_value, old_value in zip(new_case_values, old_case_values):
        if new_value.dtype == torch.bool:
            assert bool(torch.equal(new_value, old_value))
        else:
            assert bool(torch.allclose(new_value, old_value, atol=1e-14, rtol=1e-13))
    scalar_forces, scalar_blocks = _assemble_mesh_contact_scalar_fixed_details(
        case_positions, case_normals, case_references, case_valid,
        case_integrals, case_previous, args,
    )
    reference_forces, reference_blocks = _assemble_mesh_contact_batched_reference_3d_autograd(
        case_positions, case_normals, case_references, case_valid,
        case_integrals, case_previous, args,
    )
    force_relative = torch.linalg.norm(scalar_forces - reference_forces) / torch.linalg.norm(
        reference_forces
    ).clamp_min(1e-14)
    jacobian_relative = torch.linalg.norm(scalar_blocks - reference_blocks) / torch.linalg.norm(
        reference_blocks
    ).clamp_min(1e-14)
    assert float(force_relative) < 1e-12
    assert float(jacobian_relative) < 1e-10

    generator = torch.Generator(device=args.torch_device).manual_seed(20260810)
    reference_count = 128
    angles = 2.0 * math.pi * torch.rand(reference_count, generator=generator, dtype=v3.DTYPE, device=args.torch_device)
    radii = 0.045 + 0.020 * torch.rand(reference_count, generator=generator, dtype=v3.DTYPE, device=args.torch_device)
    probes = torch.stack((
        0.1 + 1.3 * torch.rand(reference_count, generator=generator, dtype=v3.DTYPE, device=args.torch_device),
        radii * torch.cos(angles), radii * torch.sin(angles),
    ), dim=1)
    fast = nearest_protection_faces(probes, grid)
    all_triangles = grid.triangles[None].expand(len(probes), -1, -1, -1)
    all_valid = torch.ones(all_triangles.shape[:2], dtype=torch.bool, device=args.torch_device)
    _, brute_distance2 = _closest_points_on_triangles(probes, all_triangles, all_valid)
    brute_distance = torch.sqrt(brute_distance2.min(dim=1).values)
    assert bool(torch.allclose(fast.nearest_distance, brute_distance, atol=2e-10, rtol=2e-10))
    assert float(fast.candidate_count.to(v3.DTYPE).mean()) < grid.protection_face_count * 0.2

    # Seed the temporal cache, move every probe by a sub-threshold amount, and
    # compare the cached result with the brute-force reference.
    original_cache = args.mesh_contact_cache
    args.mesh_contact_cache = MeshContactCache(
        torch.full((reference_count,), -1, dtype=torch.long, device=args.torch_device),
        torch.full((reference_count, 3), torch.nan, dtype=v3.DTYPE, device=args.torch_device),
    )
    _ = current_mesh_topology(probes, args)
    moved = probes + 1e-5 * torch.randn(reference_count, 3, generator=generator, dtype=v3.DTYPE, device=args.torch_device)
    cached = current_mesh_topology(moved, args)
    moved_triangles = grid.triangles[None].expand(reference_count, -1, -1, -1)
    moved_valid = torch.ones(moved_triangles.shape[:2], dtype=torch.bool, device=args.torch_device)
    moved_closest, moved_d2 = _closest_points_on_triangles(moved, moved_triangles, moved_valid)
    reference_d2, reference_column = moved_d2.min(dim=1)
    reference_ids = reference_column
    reference_normals = grid.normals[reference_ids]
    reference_clearance = torch.sum(reference_normals * (moved - grid.triangles[reference_ids, 0]), dim=1)
    assert bool(torch.allclose(cached.nearest_distance, torch.sqrt(reference_d2), atol=1e-9, rtol=0.0))
    assert bool(torch.allclose(cached.clearance, reference_clearance, atol=1e-9, rtol=0.0))
    zero_reference = torch.zeros(reference_count, dtype=v3.DTYPE, device=args.torch_device)
    cached_force = torch.func.vmap(
        lambda p, n, a, old_i, previous: _mesh_contact_fixed_face(
            p, n, a, torch.ones((), dtype=torch.bool, device=p.device), old_i, previous, args
        )[0]
    )(moved, cached.normal, cached.reference, zero_reference, zero_reference)
    reference_force = torch.func.vmap(
        lambda p, n, a, old_i, previous: _mesh_contact_fixed_face(
            p, n, a, torch.ones((), dtype=torch.bool, device=p.device), old_i, previous, args
        )[0]
    )(moved, reference_normals, grid.triangles[reference_ids, 0], zero_reference, zero_reference)
    force_relative_error = torch.linalg.norm(cached_force - reference_force) / torch.linalg.norm(reference_force).clamp_min(1e-14)
    assert float(force_relative_error) < 1e-8
    cache_test_hit_rate = args.mesh_contact_cache.hit_count / max(args.mesh_contact_cache.hit_count + args.mesh_contact_cache.miss_count, 1)
    args.mesh_contact_cache = original_cache

    # Compare direct block-penta assembly/solve with the frozen v3 dense reference.
    test_nodes = 12
    test_positions = torch.zeros((test_nodes, 3), dtype=v3.DTYPE, device=args.torch_device)
    test_positions[:, 0] = torch.linspace(0.0, args.total_length, test_nodes, dtype=v3.DTYPE, device=args.torch_device)
    test_positions[:, 1] = 0.004 * torch.sin(torch.linspace(0.0, 2.0 * math.pi, test_nodes, dtype=v3.DTYPE, device=args.torch_device))
    test_positions[6, 1:] = torch.tensor([0.051 / math.sqrt(2), 0.051 / math.sqrt(2)], dtype=v3.DTYPE, device=args.torch_device)
    test_rest = torch.full((test_nodes - 1,), args.total_length / (test_nodes - 1), dtype=v3.DTYPE, device=args.torch_device)
    matrix_errors, delta_errors = [], []
    original_cache = args.mesh_contact_cache
    args.mesh_contact_cache = MeshContactCache(
        torch.full((test_nodes,), -1, dtype=torch.long, device=args.torch_device),
        torch.full((test_nodes, 3), torch.nan, dtype=v3.DTYPE, device=args.torch_device),
    )
    for tip_constrained in (False, True):
        prescribed_mask, prescribed = v3.make_prescribed_state(test_positions, tip_constrained, 0.0)
        free_mask = ~prescribed_mask
        pattern = v3.get_assembly_pattern(prescribed_mask)
        dense_gradient, dense_elastic = v3.assemble_elastic_batched(test_positions, test_rest, pattern, args.EA, args.EI)
        block_gradient, block_system, free_start, free_end = assemble_elastic_block_penta(test_positions, test_rest, prescribed_mask, args)
        assert torch.allclose(block_gradient, dense_gradient, atol=1e-10, rtol=1e-10)
        zero_state = torch.zeros(test_nodes, dtype=v3.DTYPE, device=args.torch_device)
        contact_force, contact_blocks = assemble_mesh_contact_batched(test_positions, zero_state, zero_state, pattern, args)
        masses = torch.full((test_nodes,), args.mtotal / test_nodes, dtype=v3.DTYPE, device=args.torch_device)
        free_masses = masses[:, None].expand_as(test_positions)[free_mask]
        diagonal_add = free_masses / args.dt**2 + args.damping / args.dt
        block_system.diagonal.diagonal(dim1=1, dim2=2).add_(diagonal_add.reshape(-1, 3))
        block_system.diagonal.add_(-contact_blocks[free_start:free_end])
        dense_reference = dense_elastic.clone()
        dense_reference.diagonal().add_(diagonal_add)
        node_dofs = pattern.full_to_free.reshape(-1, 3)
        rows = node_dofs[:, :, None].expand(-1, -1, 3)
        cols = node_dofs[:, None, :].expand(-1, 3, -1)
        valid_blocks = (rows >= 0) & (cols >= 0)
        dense_reference.index_put_((rows[valid_blocks], cols[valid_blocks]), (-contact_blocks)[valid_blocks], accumulate=True)
        reconstructed = block_system.dense_reference()
        matrix_error = torch.linalg.norm(reconstructed - dense_reference) / torch.linalg.norm(dense_reference).clamp_min(1e-14)
        rhs = torch.linspace(-0.1, 0.1, len(dense_gradient), dtype=v3.DTYPE, device=args.torch_device)
        band_delta, fallback = block_system.solve(rhs)
        dense_delta = torch.linalg.solve(dense_reference, rhs)
        rescue_delta, rescue_used = block_system.solve(rhs, force_dense=True)
        delta_error = torch.linalg.norm(band_delta - dense_delta) / torch.linalg.norm(dense_delta).clamp_min(1e-14)
        rescue_error = torch.linalg.norm(rescue_delta - dense_delta) / torch.linalg.norm(dense_delta).clamp_min(1e-14)
        assert not fallback and rescue_used
        assert float(matrix_error) < 1e-10 and float(delta_error) < 1e-9 and float(rescue_error) < 1e-9
        matrix_errors.append(float(matrix_error))
        delta_errors.append(float(delta_error))
    args.mesh_contact_cache = original_cache
    print(
        f"mesh self-test passed: faces={grid.protection_face_count}, cell={grid.cell_size * 1e3:.3f} mm, "
        f"mean candidates={float(fast.candidate_count.to(v3.DTYPE).mean()):.1f}, device={args.torch_device}, "
        f"cache test hit rate={cache_test_hit_rate:.1%}, force error={float(force_relative_error):.3e}, "
        f"band matrix error={max(matrix_errors):.3e}, "
        f"delta error={max(delta_errors):.3e}"
    )


def parse_mesh_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mesh-obj", type=Path, default=DEFAULT_MESH_OBJ, help="50 mm protection / 60 mm display OBJ")
    parser.add_argument("--mesh-grid-cell-size", type=float, default=0.0, help="uniform-grid cell size in metres; 0 chooses OBJ median edge")
    parser.add_argument("--mesh-search-ring", type=int, default=2, help="initial cube radius in uniform-grid cells")
    parser.add_argument("--mesh-plot-face-stride", type=int, default=4, help="plot every kth OBJ face only; collision always uses all faces")
    options, remaining = parser.parse_known_args()
    if options.mesh_grid_cell_size < 0.0 or options.mesh_search_ring < 0 or options.mesh_plot_face_stride <= 0:
        parser.error("mesh-grid-cell-size must be >=0, mesh-search-ring >=0, and mesh-plot-face-stride >0")
    sys.argv = [sys.argv[0], *remaining]
    return options


ORIGINAL_RESOLVE_OUTPUT_PATHS = v3.resolve_output_paths


def _main() -> None:
    options = parse_mesh_options()
    args = v3.parse_args()
    mesh = load_obj_triangles(options.mesh_obj)
    protection_radius, outer_radius = validate_mesh_alignment(mesh, args)
    args.visual_mesh = mesh
    args.mesh_plot_face_stride = options.mesh_plot_face_stride
    args.mesh_collision = build_uniform_triangle_grid(mesh, options.mesh_obj, args, options.mesh_grid_cell_size, options.mesh_search_ring)
    args.performance_profile = {}
    args._last_mesh_evaluation = None
    args.mesh_exact_reuse_hit_count = 0
    args.mesh_exact_reuse_miss_count = 0
    args.mesh_contact_cache = MeshContactCache(
        triangle_local=torch.full((args.num_nodes,), -1, dtype=torch.long, device=args.torch_device),
        last_positions=torch.full((args.num_nodes, 3), torch.nan, dtype=v3.DTYPE, device=args.torch_device),
    )
    args.extra_history_record_hook = lambda history, positions: record_mesh_diagnostics(history, positions, args)
    print(
        f"mesh PI contact: {options.mesh_obj.resolve()} | protection faces={args.mesh_collision.protection_face_count} | "
        f"boundary={protection_radius * 1e3:.3f} mm | outer shell={outer_radius * 1e3:.3f} mm | "
        f"grid cell={args.mesh_collision.cell_size * 1e3:.3f} mm | dims={args.mesh_collision.dimensions} | device={args.torch_device}"
    )
    # Patch only this wrapper process.  The imported v2 source file is never modified.
    v3.contact_quantities = mesh_contact_quantities
    v3.residual_and_jacobian = mesh_residual_and_jacobian
    v3.implicit_euler_step = mesh_implicit_euler_step
    v3.compute_protection_error = lambda positions, _radius, _safe: compute_mesh_penetration(positions, args)
    v3.create_final_result_figure = create_mesh_final_result_figure
    v3.visualize_results = visualize_mesh_results
    v3.resolve_output_paths = mesh_resolve_output_paths
    if args.self_test:
        mesh_self_test(args)
        return
    history = v3.run_simulation(args)
    append_mesh_metadata(history, args)
    npz_path = Path(str(history["npz_path"].item()))
    np.savez_compressed(npz_path, **history)
    print(f"mesh metadata finalized in: {npz_path}")


def main(argv=None) -> None:
    with cli_arguments(argv):
        _main()


if __name__ == "__main__":
    main()
