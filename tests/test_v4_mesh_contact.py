import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import mesh_contact_v4 as mesh
import rod_soft_contact_PI_dynamic_quasi_static_3d_v4 as v4


ROOT = Path(__file__).resolve().parents[1]
OBJ = ROOT / "tube_obj" / "straight_tube_L1p5m_boundaryR50mm_shellR60mm_32sides_axial2.obj"
CURVED_OBJ = ROOT / "tube_obj" / "arc90_L1p5m_xz_wallR60mm_protectionR50mm_64sides_axial256.obj"


def _write_indexed_obj(path, vertices, faces, flip_faces=()):
    flip_faces = set(flip_faces)
    lines = ["# test protection surface", *(
        f"v {x:.17g} {y:.17g} {z:.17g}" for x, y, z in np.asarray(vertices)
    ), "g protection"]
    for face_id, face in enumerate(np.asarray(faces, dtype=np.int64)):
        values = face[::-1] if face_id in flip_faces else face
        lines.append("f " + " ".join(str(int(value) + 1) for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _swept_tube(centerline, radius=0.1, sides=16):
    centers = np.asarray(centerline, dtype=np.float64)
    rings = []
    for index, center in enumerate(centers):
        if index == 0:
            tangent = centers[1] - centers[0]
        elif index == len(centers) - 1:
            tangent = centers[-1] - centers[-2]
        else:
            tangent = centers[index + 1] - centers[index - 1]
        tangent /= np.linalg.norm(tangent)
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(reference, tangent))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        first = np.cross(reference, tangent)
        first /= np.linalg.norm(first)
        second = np.cross(tangent, first)
        rings.append(np.asarray([
            center + radius * (np.cos(theta) * first + np.sin(theta) * second)
            for theta in np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
        ]))
    vertices = np.concatenate(rings, axis=0)
    faces = []
    for axial in range(len(centers) - 1):
        for side in range(sides):
            next_side = (side + 1) % sides
            a = axial * sides + side
            b = axial * sides + next_side
            c = (axial + 1) * sides + next_side
            d = (axial + 1) * sides + side
            faces.extend(((a, b, c), (a, c, d)))
    return vertices, np.asarray(faces, dtype=np.int64), centers


def _write_transformed_existing_protection(path, rotation):
    vertices, faces, groups = mesh._parse_obj(OBJ)
    protection_ids = mesh._select_group(groups, "protection")
    return _write_indexed_obj(
        path, vertices @ np.asarray(rotation, dtype=np.float64).T,
        faces[protection_ids],
    )


def _write_ascii_stl(path, vertices, faces):
    lines = ["solid protection"]
    for face in np.asarray(faces, dtype=np.int64):
        a, b, c = np.asarray(vertices)[face]
        normal = np.cross(b - a, c - a)
        normal /= np.linalg.norm(normal)
        lines.append(f" facet normal {normal[0]:.17g} {normal[1]:.17g} {normal[2]:.17g}")
        lines.append("  outer loop")
        for point in (a, b, c):
            lines.append(f"   vertex {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}")
        lines.extend(("  endloop", " endfacet"))
    lines.append("endsolid protection")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _write_binary_stl(path, vertices, faces):
    data = bytearray(b"test binary protection STL".ljust(80, b" "))
    data.extend(struct.pack("<I", len(faces)))
    for face in np.asarray(faces, dtype=np.int64):
        a, b, c = np.asarray(vertices)[face]
        normal = np.cross(b - a, c - a)
        normal /= np.linalg.norm(normal)
        data.extend(struct.pack("<12fH", *normal, *a, *b, *c, 0))
    path.write_bytes(data)
    return path


def _branching_three_boundary_surface():
    """A connected box junction with three central outlet openings."""
    half = 1.0
    divisions = 4
    vertices = []
    vertex_ids = {}

    def vertex(point):
        key = tuple(np.asarray(point, dtype=np.float64))
        if key not in vertex_ids:
            vertex_ids[key] = len(vertices)
            vertices.append(key)
        return vertex_ids[key]

    faces = []
    side_specs = (
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])),
    )
    removed_sides = {0, 2, 4}
    for side_id, (normal, first, second) in enumerate(side_specs):
        for row in range(divisions):
            for column in range(divisions):
                if side_id in removed_sides and row == 1 and column == 1:
                    continue
                u0 = -half + 2.0 * row / divisions
                u1 = -half + 2.0 * (row + 1) / divisions
                v0 = -half + 2.0 * column / divisions
                v1 = -half + 2.0 * (column + 1) / divisions
                base = half * normal
                p00 = vertex(base + u0 * first + v0 * second)
                p10 = vertex(base + u1 * first + v0 * second)
                p11 = vertex(base + u1 * first + v1 * second)
                p01 = vertex(base + u0 * first + v1 * second)
                faces.extend(((p00, p10, p11), (p00, p11, p01)))
    return np.asarray(vertices), np.asarray(faces, dtype=np.int64)


def _all_shared_edges_have_opposite_directions(geometry):
    faces = geometry.face_indices[geometry.triangle_obj_ids].cpu().numpy()
    owners = {}
    for face_id, face in enumerate(faces):
        for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            owners.setdefault(tuple(sorted(edge)), []).append((face_id, edge))
    return all(
        len(entries) != 2 or entries[0][1] == entries[1][1][::-1]
        for entries in owners.values()
    )


def _assert_inward_to_centerline(geometry, centers):
    triangles = geometry.triangles.cpu().numpy()
    centroids = triangles.mean(axis=1)
    nearest = np.asarray([
        centers[np.argmin(np.linalg.norm(centers - point, axis=1))]
        for point in centroids
    ])
    radial = centroids - nearest
    assert np.all(np.sum(geometry.normals.cpu().numpy() * radial, axis=1) < 0.0)


def test_mesh_orientation_handles_rotated_swept_tubes(tmp_path):
    import mesh_contact_v4 as mesh

    rotation = np.array(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)))
    straight = _write_transformed_existing_protection(tmp_path / "straight.obj", np.eye(3))
    rotated = _write_transformed_existing_protection(tmp_path / "rotated.obj", rotation)
    a = mesh.MeshGeometry.from_obj(straight)
    b = mesh.MeshGeometry.from_obj(rotated)
    assert a.boundary_loop_count == 2
    assert b.boundary_loop_count == 2
    assert a.connected_component_count == b.connected_component_count == 1
    assert float(a.normals[:, 0].abs().max()) < 1.0e-12
    assert float(b.normals[:, 2].abs().max()) < 1.0e-12


def test_mesh_orientation_handles_curved_and_s_shaped_tubes(tmp_path):
    import mesh_contact_v4 as mesh

    curved = mesh.MeshGeometry.from_obj(CURVED_OBJ, device=torch.device("cpu"))
    curved_vertices, curved_faces, curved_groups = mesh._parse_obj(CURVED_OBJ)
    curved_ids = mesh._select_group(curved_groups, "protection")
    protection_vertex_ids = np.unique(curved_faces[curved_ids])
    curved_centers = curved_vertices[protection_vertex_ids].reshape(-1, 32, 3).mean(axis=1)
    _assert_inward_to_centerline(curved, curved_centers)

    centers = np.array(((0.0, 0.0, 0.0), (0.3, 0.35, 0.0), (0.6, -0.35, 0.0),
                        (0.9, 0.35, 0.0), (1.2, 0.0, 0.0)))
    vertices, faces, _ = _swept_tube(centers, radius=0.08, sides=16)
    s_path = _write_indexed_obj(tmp_path / "s_tube.obj", vertices, faces)
    s_geometry = mesh.MeshGeometry.from_obj(s_path)
    assert s_geometry.boundary_loop_count == 2
    _assert_inward_to_centerline(s_geometry, centers)


def test_mesh_orientation_recovers_random_face_flips_and_reversals(tmp_path):
    import mesh_contact_v4 as mesh

    centers = np.array(((0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (0.8, 0.0, 0.0)))
    vertices, faces, _ = _swept_tube(centers, radius=0.15, sides=12)
    reference = mesh.MeshGeometry.from_obj(_write_indexed_obj(tmp_path / "reference.obj", vertices, faces))
    flipped_path = _write_indexed_obj(
        tmp_path / "randomly_flipped.obj", vertices, faces,
        flip_faces={1, 4, 9, 17, 22},
    )
    recovered = mesh.MeshGeometry.from_obj(flipped_path)
    assert _all_shared_edges_have_opposite_directions(recovered)
    np.testing.assert_allclose(
        np.sort(recovered.triangles.cpu().numpy(), axis=1),
        np.sort(reference.triangles.cpu().numpy(), axis=1),
        rtol=0.0, atol=1.0e-14,
    )

    reversed_geometry = mesh.MeshGeometry.from_obj(
        _write_indexed_obj(tmp_path / "reversed.obj", vertices, faces, flip_faces=range(len(faces)))
    )
    np.testing.assert_allclose(
        np.sort(reversed_geometry.triangles.cpu().numpy(), axis=1),
        np.sort(reference.triangles.cpu().numpy(), axis=1),
        rtol=0.0, atol=1.0e-14,
    )


def test_mesh_orientation_handles_connected_three_outlet_branching_surface(tmp_path):
    import mesh_contact_v4 as mesh

    vertices, faces = _branching_three_boundary_surface()
    geometry = mesh.MeshGeometry.from_obj(_write_indexed_obj(tmp_path / "branching.obj", vertices, faces))
    assert geometry.connected_component_count == 1
    assert geometry.boundary_loop_count == 3
    assert _all_shared_edges_have_opposite_directions(geometry)


def test_mesh_orientation_handles_closed_component_without_caps(tmp_path):
    vertices = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                         (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    faces = np.array(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int64)
    geometry = mesh.MeshGeometry.from_obj(_write_indexed_obj(tmp_path / "closed.obj", vertices, faces))
    assert geometry.connected_component_count == 1
    assert geometry.boundary_loop_count == 0
    assert geometry.mesh_cap_ms >= 0.0
    assert _all_shared_edges_have_opposite_directions(geometry)


def test_mesh_orientation_rejects_near_zero_signed_volume(tmp_path):
    vertices = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                         (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)))
    faces = np.array(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    with pytest.raises(ValueError, match="closed mesh has near-zero signed volume"):
        mesh.MeshGeometry.from_obj(_write_indexed_obj(tmp_path / "flat.obj", vertices, faces))


def test_mesh_orientation_rejects_unreliable_boundary_caps():
    nonplanar = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                          (1.0, 1.0, 0.0), (0.0, 1.0, 0.1)))
    with pytest.raises(ValueError, match="severely non-planar"):
        mesh._temporary_cap_for_loop(nonplanar, [0, 1, 2, 3])

    self_intersecting = np.array(((0.0, 0.0, 0.0), (1.0, 1.0, 0.0),
                                  (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="self-intersecting"):
        mesh._temporary_cap_for_loop(self_intersecting, [0, 1, 2, 3])


def test_mesh_orientation_orients_disconnected_components_independently(tmp_path):
    import mesh_contact_v4 as mesh

    first_vertices, first_faces, _ = _swept_tube(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), radius=0.2, sides=12)
    second_vertices, second_faces, _ = _swept_tube(((0.0, 3.0, 0.0), (1.0, 3.0, 0.0)), radius=0.2, sides=12)
    vertices = np.concatenate((first_vertices, second_vertices))
    faces = np.concatenate((first_faces, second_faces + len(first_vertices)))
    geometry = mesh.MeshGeometry.from_obj(_write_indexed_obj(tmp_path / "disconnected.obj", vertices, faces))
    assert geometry.connected_component_count == 2
    assert geometry.boundary_loop_count == 4
    _assert_inward_to_centerline(geometry, np.concatenate((
        np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        np.array(((0.0, 3.0, 0.0), (1.0, 3.0, 0.0))),
    )))


def test_mesh_orientation_rejects_nonmanifold_protection_surface(tmp_path):
    import mesh_contact_v4 as mesh

    vertices = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                         (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)))
    faces = np.array(((0, 1, 2), (1, 0, 3), (0, 1, 4)), dtype=np.int64)
    path = _write_indexed_obj(tmp_path / "nonmanifold.obj", vertices, faces)
    with pytest.raises(ValueError, match="non-manifold protection surface"):
        mesh.MeshGeometry.from_obj(path)


def test_mesh_orientation_rejects_boundary_that_cannot_form_a_loop(tmp_path):
    import mesh_contact_v4 as mesh

    vertices = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                         (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)))
    faces = np.array(((0, 1, 2), (0, 3, 4)), dtype=np.int64)
    path = _write_indexed_obj(tmp_path / "bad_boundary.obj", vertices, faces)
    with pytest.raises(ValueError, match="cannot determine lumen orientation from mesh topology|boundary loop"):
        parsed_vertices, parsed_faces, groups = mesh._parse_obj(path)
        del parsed_vertices, groups
        mesh._extract_boundary_loops(
            parsed_faces, mesh._build_edge_owners(parsed_faces), [[0, 1]],
        )


def test_mesh_geometry_accepts_ascii_and_binary_stl(tmp_path):
    import mesh_contact_v4 as mesh

    vertices, faces, centers = _swept_tube(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), radius=0.2, sides=12)
    obj_geometry = mesh.MeshGeometry.from_obj(_write_indexed_obj(tmp_path / "tube.obj", vertices, faces))
    ascii_geometry = mesh.MeshGeometry.from_obj(_write_ascii_stl(tmp_path / "tube_ascii.stl", vertices, faces))
    binary_geometry = mesh.MeshGeometry.from_obj(_write_binary_stl(tmp_path / "tube_binary.stl", vertices, faces))
    for geometry in (ascii_geometry, binary_geometry):
        assert geometry.boundary_loop_count == 2
        _assert_inward_to_centerline(geometry, centers)
        np.testing.assert_allclose(geometry.triangles, obj_geometry.triangles, rtol=0.0, atol=2.0e-7)


def test_mesh_geometry_rejects_malformed_stl(tmp_path):
    import mesh_contact_v4 as mesh

    path = tmp_path / "malformed.stl"
    path.write_bytes(b"solid broken\n facet normal 0 0 1\n")
    with pytest.raises(ValueError, match="STL"):
        mesh.MeshGeometry.from_obj(path)


def test_stl_parser_accepts_short_ascii_header(tmp_path):
    path = tmp_path / "short_ascii.stl"
    path.write_text(
        "solid x\n"
        "vertex 0 0 0\n"
        "vertex 1 0 0\n"
        "vertex 0 1 0\n"
        "endsolid x\n",
        encoding="ascii",
    )
    vertices, faces, groups = mesh._parse_stl(path)
    assert vertices.shape == (3, 3)
    assert faces.tolist() == [[0, 1, 2]]
    assert groups["protection"].tolist() == [0]


def _python_contact_for_geometry(geometry, positions):
    details = geometry.nearest(torch.as_tensor(positions, dtype=torch.float64))
    force, _blocks, state = mesh.fixed_face_contact(
        torch.as_tensor(positions, dtype=torch.float64), details,
        torch.zeros(len(positions), dtype=torch.float64),
        torch.zeros(len(positions), dtype=torch.float64),
        dt=0.1, kp=3000.0, ki=100.0, safe_distance=0.01,
        max_pi_force=20.0, max_integral_state=0.05,
        leak_rate=5.0, smooth_length=2.0e-5, direction_epsilon=1.0e-6,
    )
    return {
        "force": force.detach().cpu().numpy(),
        "active": state.active.detach().cpu().numpy(),
        "clearance": state.distances.detach().cpu().numpy(),
        "triangle_ids": details.triangle_obj.detach().cpu().numpy(),
    }


def _cpp_contact_for_geometry(geometry, positions):
    import der_cpp_backend as backend

    nodes = len(positions)
    core = backend.DERCore(
        np.full(nodes - 1, 0.1, dtype=np.float64),
        np.full(nodes, 0.01, dtype=np.float64),
        10.0, 2.0, 0.1, 0.06, 0.01, 3000.0, 100.0, 20.0, 5.0, 0.05,
    )
    core.configure_mesh(
        np.ascontiguousarray(geometry.triangles.cpu().numpy(), dtype=np.float64),
        np.ascontiguousarray(geometry.triangle_vertex_normals.cpu().numpy(), dtype=np.float64),
        np.ascontiguousarray(geometry.triangle_obj_ids.cpu().numpy(), dtype=np.int64),
    )
    return core.contact_kernel(
        np.ascontiguousarray(positions, dtype=np.float64),
        np.zeros(nodes, dtype=np.float64), np.zeros(nodes, dtype=np.float64),
        0.1, 100.0,
    )


def test_python_cpp_rotated_mesh_contact_parity_and_rotation_equivalence(tmp_path):
    rotation = np.array(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)))
    vertices, faces, _ = _swept_tube(
        ((0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (0.8, 0.0, 0.0)),
        radius=0.15, sides=16,
    )
    base = mesh.MeshGeometry.from_obj(_write_indexed_obj(tmp_path / "base.obj", vertices, faces))
    rotated = mesh.MeshGeometry.from_obj(_write_indexed_obj(
        tmp_path / "rotated.obj", vertices @ rotation.T, faces,
    ))
    points = np.array(((0.22, 0.07, 0.03), (0.22, 0.14, 0.05), (0.22, 0.23, 0.08)))
    rotated_points = points @ rotation.T

    base_python = _python_contact_for_geometry(base, points)
    rotated_python = _python_contact_for_geometry(rotated, rotated_points)
    base_cpp = _cpp_contact_for_geometry(base, points)
    rotated_cpp = _cpp_contact_for_geometry(rotated, rotated_points)
    for python_result, cpp_result in ((base_python, base_cpp), (rotated_python, rotated_cpp)):
        np.testing.assert_allclose(python_result["force"], cpp_result["force"], rtol=1.0e-10, atol=1.0e-12)
        np.testing.assert_array_equal(python_result["active"], cpp_result["active"])
        np.testing.assert_allclose(python_result["clearance"], cpp_result["distances"], rtol=1.0e-10, atol=1.0e-12)
        np.testing.assert_array_equal(python_result["triangle_ids"], cpp_result["triangle_ids"])
    np.testing.assert_allclose(
        rotated_python["force"], base_python["force"] @ rotation.T, rtol=1.0e-10, atol=1.0e-12,
    )
    np.testing.assert_allclose(rotated_python["clearance"], base_python["clearance"], rtol=1.0e-10, atol=1.0e-12)
    np.testing.assert_array_equal(rotated_python["active"], base_python["active"])
    np.testing.assert_array_equal(rotated_python["triangle_ids"], base_python["triangle_ids"])
    assert base_python["active"].tolist() == [False, False, True]
    assert float(np.dot(base_python["force"][2], points[2] - np.array((0.22, 0.0, 0.0)))) < 0.0


def test_python_cpp_smooth_mesh_contact_parity_near_shared_edge(tmp_path):
    vertices, faces, _ = _swept_tube(
        ((0.0, 0.0, 0.0), (0.35, 0.0, 0.0), (0.65, 0.18, 0.0)),
        radius=0.12, sides=8,
    )
    geometry = mesh.MeshGeometry.from_obj(
        _write_indexed_obj(tmp_path / "bent_parity.obj", vertices, faces)
    )
    first_face, second_face = 1, 16
    shared = sorted(set(faces[first_face]) & set(faces[second_face]))
    first_other = next(index for index in faces[first_face] if index not in shared)
    second_other = next(index for index in faces[second_face] if index not in shared)
    edge_midpoint = vertices[shared].mean(axis=0)
    surface_points = np.stack((
        edge_midpoint + 1.0e-6 * (vertices[first_other] - edge_midpoint),
        edge_midpoint + 1.0e-6 * (vertices[second_other] - edge_midpoint),
    ))
    query = surface_points - 2.0e-4 * geometry.normals[
        [first_face, second_face]
    ].cpu().numpy()

    python_result = _python_contact_for_geometry(geometry, query)
    cpp_result = _cpp_contact_for_geometry(geometry, query)

    np.testing.assert_allclose(cpp_result["force"], python_result["force"], rtol=1.0e-9, atol=1.0e-11)
    np.testing.assert_allclose(cpp_result["distances"], python_result["clearance"], rtol=1.0e-9, atol=1.0e-11)
    np.testing.assert_array_equal(cpp_result["triangle_ids"], python_result["triangle_ids"])


def test_mesh_preprocessing_is_one_time_and_not_called_by_runtime_contact(monkeypatch, tmp_path):
    vertices, faces, _ = _swept_tube(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), radius=0.2, sides=12)
    geometry = mesh.MeshGeometry.from_obj(_write_indexed_obj(tmp_path / "runtime.obj", vertices, faces))
    stats = geometry.preprocess_stats
    assert all(np.isfinite(value) and value >= 0.0 for value in stats.as_dict().values())
    assert stats.mesh_preprocess_total_ms >= stats.mesh_parse_ms
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("mesh orientation must not run in runtime contact")

    monkeypatch.setattr(mesh, "_temporary_cap_for_loop", forbidden)
    monkeypatch.setattr(mesh, "_component_signed_volume", forbidden)
    points = torch.tensor(((0.4, 0.05, 0.02), (0.4, 0.25, 0.04)), dtype=torch.float64)
    for _ in range(3):
        details = geometry.nearest(points)
        mesh.fixed_face_contact(
            points, details, torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64),
            dt=0.1, kp=3000.0, ki=100.0, safe_distance=0.01,
            max_pi_force=20.0, max_integral_state=0.05, leak_rate=5.0,
            smooth_length=2.0e-5, direction_epsilon=1.0e-6,
        )
    assert calls == []
    assert geometry.preprocess_stats == stats


def test_mesh_smooth_normal_is_continuous_across_non_coplanar_shared_edge(tmp_path):
    vertices, faces, _ = _swept_tube(
        ((0.0, 0.0, 0.0), (0.35, 0.0, 0.0), (0.65, 0.18, 0.0)),
        radius=0.12, sides=8,
    )
    geometry = mesh.MeshGeometry.from_obj(
        _write_indexed_obj(tmp_path / "bent.obj", vertices, faces)
    )
    first_face, second_face = 1, 16
    shared = sorted(set(faces[first_face]) & set(faces[second_face]))
    assert len(shared) == 2
    first_other = next(index for index in faces[first_face] if index not in shared)
    second_other = next(index for index in faces[second_face] if index not in shared)
    edge_midpoint = vertices[shared].mean(axis=0)
    epsilon = 1.0e-6
    surface_points = np.stack((
        edge_midpoint + epsilon * (vertices[first_other] - edge_midpoint),
        edge_midpoint + epsilon * (vertices[second_other] - edge_midpoint),
    ))
    inward_hint = geometry.normals[[first_face, second_face]].cpu().numpy()
    query = torch.as_tensor(surface_points + 1.0e-5 * inward_hint, dtype=torch.float64)
    details = geometry.nearest(query)

    assert details.triangle_local.tolist() == [first_face, second_face]
    face_jump = torch.linalg.vector_norm(
        geometry.normals[first_face] - geometry.normals[second_face]
    )
    smooth_jump = torch.linalg.vector_norm(details.normal[0] - details.normal[1])
    assert float(face_jump) > 0.1
    assert float(smooth_jump) < 1.0e-4
    torch.testing.assert_close(details.reference, torch.as_tensor(surface_points), atol=2.0e-8, rtol=0.0)


def test_accepted_mesh_diagnostics_reuse_solver_clearance(monkeypatch):
    args = type("Args", (), {"contact_type": "mesh"})()
    positions = torch.zeros((3, 3), dtype=torch.float64)
    accepted_clearance = torch.tensor((0.02, -0.0003, -0.0012), dtype=torch.float64)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("accepted mesh diagnostics must not repeat nearest-face search")

    monkeypatch.setattr(v4, "_mesh_details", forbidden)
    clearance, protection = v4.contact_diagnostics(
        positions, args, accepted_clearance=accepted_clearance,
    )

    torch.testing.assert_close(clearance, accepted_clearance)
    torch.testing.assert_close(
        protection, torch.tensor((0.0, 0.0003, 0.0012), dtype=torch.float64)
    )


def test_initial_cpp_mesh_clearance_uses_cpp_bvh(monkeypatch):
    args = type(
        "Args", (),
        {"contact_type": "mesh", "backend": "cpp", "dt": 0.005, "ki_active": 100.0},
    )()
    positions = torch.zeros((3, 3), dtype=torch.float64)
    rest_lengths = torch.ones(2, dtype=torch.float64)
    masses = torch.ones(3, dtype=torch.float64)
    expected = np.array((0.05, 0.049, 0.048), dtype=np.float64)

    class FakeCore:
        def contact_kernel(self, p, integrals, previous_errors, dt, ki_active):
            np.testing.assert_array_equal(p, positions.numpy())
            np.testing.assert_array_equal(integrals, np.zeros(3))
            np.testing.assert_array_equal(previous_errors, np.zeros(3))
            assert dt == args.dt
            assert ki_active == args.ki_active
            return {"distances": expected}

    monkeypatch.setattr(v4, "_get_core", lambda *_args: FakeCore())
    monkeypatch.setattr(
        v4, "_mesh_details",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("initial C++ diagnostics must use the C++ BVH")
        ),
    )

    actual = v4.initial_accepted_clearance(positions, rest_lengths, masses, args)

    torch.testing.assert_close(actual, torch.as_tensor(expected))


def test_mesh_visualization_proxy_is_low_resolution_and_follows_imported_curve():
    import mesh_contact_v4 as mesh

    geometry = mesh.MeshGeometry.from_obj(CURVED_OBJ, device=torch.device("cpu"))
    args = type("Args", (), dict(
        contact_type="mesh", mesh_geometry=geometry, total_length=1.5,
        tube_radius=0.06,
    ))()
    surface = v4._visualization_surface(
        args, axial_samples=16, circumferential_samples=12,
    )
    cached_surface = v4._visualization_surface(
        args, axial_samples=16, circumferential_samples=12,
    )
    assert all(actual is cached for actual, cached in zip(surface, cached_surface))
    assert surface[0].shape == (13, 17)
    assert surface[1].shape == surface[0].shape
    assert surface[2].shape == surface[0].shape
    assert np.isfinite(np.stack(surface)).all()
    # The proxy must retain the xz-plane arc rather than silently becoming a
    # straight analytic cylinder; its ring centres vary by more than 10 mm in z.
    centerline_z = np.mean(surface[2], axis=0)
    assert float(centerline_z.max() - centerline_z.min()) > 0.01


def test_tube_visualization_surface_keeps_analytic_cylinder_fallback():
    args = type("Args", (), dict(
        contact_type="tube", total_length=1.5, tube_radius=0.06,
    ))()
    x, y, z = v4._visualization_surface(
        args, axial_samples=8, circumferential_samples=10,
    )
    np.testing.assert_allclose(np.mean(y[:-1], axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.mean(z[:-1], axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.sqrt(y[:, 0] ** 2 + z[:, 0] ** 2), 0.06)
    assert x.shape == (11, 9)


def test_mesh_module_import_is_independent_of_v3():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    code = (
        "import sys; import mesh_contact_v4; "
        "assert 'rod_soft_contact_PI_dynamic_quasi_static_3d_v3' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_obj_loader_and_nearest_face_return_inward_contact_geometry():
    import mesh_contact_v4 as mesh

    geometry = mesh.MeshGeometry.from_obj(
        OBJ, protection_keyword="protection", device=torch.device("cpu"),
    )
    assert geometry.triangles.shape[1:] == (3, 3)
    assert len(geometry.triangles) == len(geometry.normals)
    points = torch.tensor(
        [[0.2, 0.051, 0.0], [0.8, 0.0, 0.0]], dtype=torch.float64,
    )
    cache = mesh.MeshContactCache.for_nodes(len(points), torch.device("cpu"))
    details = geometry.nearest(points, cache)
    assert details.triangle_local[0].item() >= 0
    assert details.clearance[0].item() < 0.0
    assert details.clearance[1].item() > 0.0
    assert torch.isfinite(details.reference).all()
    opposite = torch.tensor([[0.2, -0.06, 0.0]], dtype=torch.float64)
    opposite_cache = mesh.MeshContactCache.for_nodes(1, torch.device("cpu"))
    opposite_details = geometry.nearest(opposite, opposite_cache)
    _, all_d2 = mesh._closest_points_on_triangles(
        opposite, geometry.triangles.unsqueeze(0),
        torch.ones((1, len(geometry.triangles)), dtype=torch.bool),
    )
    torch.testing.assert_close(opposite_details.nearest_distance, all_d2.min(dim=1).values.sqrt())


def test_v4_parser_exposes_mesh_contact_without_changing_tube_default(monkeypatch):
    monkeypatch.setattr(v4.sys, "argv", ["v4"])
    parsed = v4._parse_args_with_backend()
    assert parsed.contact_type == "tube"
    monkeypatch.setattr(
        v4.sys, "argv", ["v4", "--backend", "python", "--contact-type", "mesh", "--mesh-obj", str(OBJ)]
    )
    parsed = v4._parse_args_with_backend()
    assert parsed.contact_type == "mesh"
    assert parsed.mesh_obj == OBJ


def test_mesh_contact_force_and_jacobian_match_fixed_face_reference():
    import mesh_contact_v4 as mesh

    geometry = mesh.MeshGeometry.from_obj(
        OBJ, protection_keyword="protection", device=torch.device("cpu"),
    )
    points = torch.tensor([[0.2, 0.052, 0.0], [0.2, 0.0, 0.0]], dtype=torch.float64)
    cache = mesh.MeshContactCache.for_nodes(len(points), torch.device("cpu"))
    details = geometry.nearest(points, cache)
    integrals = torch.zeros(2, dtype=torch.float64)
    previous = torch.zeros(2, dtype=torch.float64)
    force, blocks, state = mesh.fixed_face_contact(
        points, details, integrals, previous, dt=0.1, kp=3000.0, ki=100.0,
        safe_distance=0.01, max_pi_force=20.0, max_integral_state=0.05,
        leak_rate=5.0, smooth_length=2.0e-5, direction_epsilon=1.0e-6,
    )
    assert force.shape == points.shape
    assert blocks.shape == (2, 3, 3)
    assert state.new_integrals.shape == (2,)
    assert torch.isfinite(force).all() and torch.isfinite(blocks).all()
    assert torch.allclose(force[1], torch.zeros(3, dtype=torch.float64))
    assert blocks[0, 1, 2].abs() > 0.0


def test_v4_python_mesh_backend_completes_one_implicit_step(monkeypatch):
    monkeypatch.setattr(
        v4.sys, "argv", ["v4", "--backend", "python", "--contact-type", "mesh",
                         "--mesh-obj", str(OBJ), "--num-nodes", "8", "--pulse-count-min", "4",
                         "--pulse-count-max", "4", "--device", "cpu"]
    )
    args = v4._parse_args_with_backend()
    positions_np, velocities_np, rest_np, masses_np, integrals_np = v4.initialize_rod(
        args.total_length, args.num_nodes, args.diameter, args.density,
    )
    positions = torch.tensor(positions_np, dtype=v4.DTYPE)
    velocities = torch.tensor(velocities_np, dtype=v4.DTYPE)
    rest = torch.tensor(rest_np, dtype=v4.DTYPE)
    masses = torch.tensor(masses_np, dtype=v4.DTYPE)
    integrals = torch.tensor(integrals_np, dtype=v4.DTYPE)
    pulse = torch.zeros_like(positions)
    result = v4._python_implicit_euler_step(
        positions, velocities, rest, masses, integrals, positions.clone(), pulse,
        False, 0.0, 0.01, args,
    )
    assert result[0].shape == positions.shape
    assert result[8].converged


def test_python_mesh_accepted_state_uses_newton_frozen_topology(monkeypatch):
    monkeypatch.setattr(
        v4.sys, "argv", ["v4", "--backend", "python", "--contact-type", "mesh",
                         "--mesh-obj", str(OBJ), "--num-nodes", "8", "--pulse-count-min", "4",
                         "--pulse-count-max", "4", "--device", "cpu"]
    )
    args = v4._parse_args_with_backend()
    positions_np, velocities_np, rest_np, masses_np, integrals_np = v4.initialize_rod(
        args.total_length, args.num_nodes, args.diameter, args.density,
    )
    positions = torch.tensor(positions_np, dtype=v4.DTYPE)
    velocities = torch.tensor(velocities_np, dtype=v4.DTYPE)
    rest = torch.tensor(rest_np, dtype=v4.DTYPE)
    masses = torch.tensor(masses_np, dtype=v4.DTYPE)
    integrals = torch.tensor(integrals_np, dtype=v4.DTYPE)
    pulse = torch.zeros_like(positions)
    original_contact_quantities = v4.contact_quantities
    accepted_topologies = []

    def recording_contact_quantities(current, old_integrals, previous_errors, current_args):
        if current is not positions and not torch.is_grad_enabled():
            accepted_topologies.append(getattr(current_args, "_frozen_mesh_topology", None))
        return original_contact_quantities(current, old_integrals, previous_errors, current_args)

    monkeypatch.setattr(v4, "contact_quantities", recording_contact_quantities)
    v4._python_implicit_euler_step(
        positions, velocities, rest, masses, integrals, positions.clone(), pulse,
        False, 0.0, 0.01, args,
    )

    assert accepted_topologies
    assert accepted_topologies[-1] is not None
    assert args._frozen_mesh_topology is None


def test_cpp_mesh_contact_kernel_accepts_external_geometry():
    import der_cpp_backend as backend

    nodes = 4
    core = backend.DERCore(
        np.full(nodes - 1, 0.1, dtype=np.float64), np.full(nodes, 0.01, dtype=np.float64),
        10.0, 2.0, 0.1, 0.06, 0.01, 3000.0, 100.0, 20.0, 5.0, 0.05,
    )
    triangles = np.array([[[0.0, 0.05, -0.05], [1.0, 0.05, -0.05], [0.0, 0.05, 0.05]]], dtype=np.float64)
    normals = np.array([[0.0, -1.0, 0.0]], dtype=np.float64)
    core.configure_mesh(triangles, normals, np.array([17], dtype=np.int64))
    positions = np.zeros((nodes, 3), dtype=np.float64)
    positions[:, 0] = np.linspace(0.0, 0.3, nodes)
    positions[:, 1] = 0.052
    actual = core.contact_kernel(positions, np.zeros(nodes), np.zeros(nodes), 0.1, 100.0)
    assert actual["jacobian"].shape == (nodes, 3, 3)
    assert actual["force"][1, 1] < 0.0
    assert actual["triangle_ids"].tolist() == [17] * nodes
    import mesh_contact_v4 as mesh
    details = mesh.MeshContactDetails(
        triangle_local=torch.zeros(nodes, dtype=torch.long),
        triangle_obj=torch.full((nodes,), 17, dtype=torch.long),
        normal=torch.tensor([[0.0, -1.0, 0.0]] * nodes, dtype=torch.float64),
        reference=torch.tensor([[0.0, 0.05, -0.05]] * nodes, dtype=torch.float64),
        clearance=torch.zeros(nodes, dtype=torch.float64),
        nearest_distance=torch.zeros(nodes, dtype=torch.float64),
        candidate_count=torch.ones(nodes, dtype=torch.long),
        search_ring=torch.zeros(nodes, dtype=torch.long),
    )
    expected_force, expected_jac, _ = mesh.fixed_face_contact(
        torch.tensor(positions), details, torch.zeros(nodes), torch.zeros(nodes),
        dt=0.1, kp=3000.0, ki=100.0, safe_distance=0.01,
        max_pi_force=20.0, max_integral_state=0.05, leak_rate=5.0,
        smooth_length=2.0e-5, direction_epsilon=1e-6,
    )
    np.testing.assert_allclose(actual["force"], expected_force.numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual["jacobian"], expected_jac.numpy(), rtol=1e-10, atol=1e-12)


def test_v4_cpp_mesh_backend_completes_one_implicit_step(monkeypatch):
    monkeypatch.setattr(
        v4.sys, "argv", ["v4", "--backend", "cpp", "--contact-type", "mesh",
                         "--mesh-obj", str(OBJ), "--num-nodes", "8", "--pulse-count-min", "4",
                         "--pulse-count-max", "4", "--device", "cpu"]
    )
    args = v4._parse_args_with_backend()
    positions_np, velocities_np, rest_np, masses_np, integrals_np = v4.initialize_rod(
        args.total_length, args.num_nodes, args.diameter, args.density,
    )
    positions = torch.tensor(positions_np, dtype=v4.DTYPE)
    velocities = torch.tensor(velocities_np, dtype=v4.DTYPE)
    rest = torch.tensor(rest_np, dtype=v4.DTYPE)
    masses = torch.tensor(masses_np, dtype=v4.DTYPE)
    integrals = torch.tensor(integrals_np, dtype=v4.DTYPE)
    pulse = torch.zeros_like(positions)
    result = v4._cpp_implicit_euler_step(
        positions, velocities, rest, masses, integrals, positions.clone(), pulse,
        False, 0.0, 0.01, args,
    )
    assert result[0].shape == positions.shape
    assert result[8].converged


def test_v4_python_cpp_mesh_single_step_parity(monkeypatch):
    def make_args(backend):
        monkeypatch.setattr(
            v4.sys, "argv", ["v4", "--backend", backend, "--contact-type", "mesh",
                             "--mesh-obj", str(OBJ), "--num-nodes", "8", "--pulse-count-min", "4",
                             "--pulse-count-max", "4", "--device", "cpu"]
        )
        return v4._parse_args_with_backend()

    py_args = make_args("python")
    cpp_args = make_args("cpp")
    positions_np, velocities_np, rest_np, masses_np, integrals_np = v4.initialize_rod(
        py_args.total_length, py_args.num_nodes, py_args.diameter, py_args.density,
    )
    positions_np[:, 1] = 0.052
    positions_np[:, 2] = np.linspace(0.0, 0.002, py_args.num_nodes)
    as_tensor = lambda value: torch.tensor(value, dtype=v4.DTYPE)
    positions, velocities = as_tensor(positions_np), as_tensor(velocities_np)
    rest, masses, integrals = as_tensor(rest_np), as_tensor(masses_np), as_tensor(integrals_np)
    pulse = torch.zeros_like(positions)
    py = v4._python_implicit_euler_step(
        positions, velocities, rest, masses, integrals, positions.clone(), pulse,
        False, 0.0, 0.01, py_args,
    )
    cpp = v4._cpp_implicit_euler_step(
        positions, velocities, rest, masses, integrals, positions.clone(), pulse,
        False, 0.0, 0.01, cpp_args,
    )
    torch.testing.assert_close(cpp[0], py[0], rtol=2e-8, atol=2e-10)
    torch.testing.assert_close(cpp[2], py[2], rtol=2e-8, atol=2e-10)
    torch.testing.assert_close(cpp[3], py[3], rtol=2e-8, atol=2e-10)
