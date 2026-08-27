from pathlib import Path

import numpy as np
import pytest
import torch

import der_cpp_backend as backend


EI = 1.7


def _core(rest_lengths):
    rest_lengths = np.ascontiguousarray(rest_lengths, dtype=np.float64)
    return backend.DERCore(
        rest_lengths,
        np.ones(rest_lengths.size + 1, dtype=np.float64),
        10.0,
        EI,
        0.1,
        0.05,
        0.001,
        1.0,
        1.0,
        1.0,
        0.1,
        1.0,
    )


def _torch_bending_energy(local_positions, left_rest, right_rest):
    """Independent statement of the preserved DER energy contract."""
    left = local_positions[1] - local_positions[0]
    right = local_positions[2] - local_positions[1]
    t0 = left / torch.sqrt(torch.clamp(torch.dot(left, left), min=1.0e-24))
    t1 = right / torch.sqrt(torch.clamp(torch.dot(right, right), min=1.0e-24))
    denominator = torch.clamp(1.0 + torch.dot(t0, t1), min=1.0e-12)
    curvature = 2.0 * torch.linalg.cross(t0, t1) / denominator
    voronoi_length = 0.5 * (left_rest + right_rest)
    return 0.5 * EI * torch.dot(curvature, curvature) / voronoi_length


_torch_gradient = torch.func.grad(_torch_bending_energy, argnums=0)
_torch_hessian = torch.func.jacrev(_torch_gradient, argnums=0)


def _oracle(local_positions, left_rest, right_rest):
    positions = torch.as_tensor(local_positions, dtype=torch.float64)
    left = torch.as_tensor(left_rest, dtype=torch.float64)
    right = torch.as_tensor(right_rest, dtype=torch.float64)
    gradient = torch.vmap(_torch_gradient)(positions, left, right).reshape(-1, 9)
    hessian = torch.vmap(_torch_hessian)(positions, left, right).reshape(-1, 9, 9)
    return gradient.detach().numpy(), hessian.detach().numpy()


def _random_nondegenerate_chain(local_count=1000):
    rng = np.random.default_rng(20260825)
    directions = []
    previous = None
    while len(directions) < local_count + 1:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        if previous is not None and float(direction @ previous) < -0.92:
            continue
        directions.append(direction)
        previous = direction
    edge_lengths = rng.uniform(0.025, 0.16, size=local_count + 1)
    edges = edge_lengths[:, None] * np.asarray(directions)
    positions = np.concatenate(
        (np.zeros((1, 3), dtype=np.float64), np.cumsum(edges, axis=0)), axis=0
    )
    rest = rng.uniform(0.03, 0.14, size=local_count + 1).astype(np.float64)
    return np.ascontiguousarray(positions), rest


def test_cpp_analytic_bending_sources_are_compiled_hashed_and_used():
    root = Path(backend.__file__).resolve().parent
    header = root / "cpp" / "der_bending.h"
    source = root / "cpp" / "der_bending.cpp"
    assert header.is_file()
    assert source.is_file()

    loader_source = Path(backend.__file__).read_text()
    core_source = (root / "cpp" / "der_core.cpp").read_text()
    assert '"der_bending.cpp"' in loader_source
    assert '"der_bending.h"' in loader_source
    assert "analytic_bending_derivatives" in core_source
    assert "analytic_bending_gradient" in core_source


def test_cpp_analytic_bending_matches_1000_random_autograd_cases():
    positions, rest = _random_nondegenerate_chain(1000)
    actual_gradient, actual_hessian = _core(rest).bending_kernel(positions, rest)
    local_positions = np.stack(
        (positions[:-2], positions[1:-1], positions[2:]), axis=1
    )
    expected_gradient, expected_hessian = _oracle(
        local_positions, rest[:-1], rest[1:]
    )
    np.testing.assert_allclose(
        actual_gradient, expected_gradient, rtol=2.0e-9, atol=2.0e-8
    )
    np.testing.assert_allclose(
        actual_hessian, expected_hessian, rtol=5.0e-9, atol=2.0e-6
    )


@pytest.mark.parametrize(
    "positions",
    [
        np.array(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.22, 0.0, 0.0))),
        np.array(
            (
                (0.0, 0.0, 0.0),
                (0.1, 0.0, 0.0),
                (0.1 - 0.12 * np.cos(5.0e-2), 0.12 * np.sin(5.0e-2), 0.0),
            )
        ),
    ],
    ids=("straight", "high_curvature"),
)
def test_cpp_analytic_bending_straight_and_high_curvature(positions):
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    rest = np.array((0.09, 0.13), dtype=np.float64)
    actual_gradient, actual_hessian = _core(rest).bending_kernel(positions, rest)
    expected_gradient, expected_hessian = _oracle(
        positions[None, ...], rest[:1], rest[1:]
    )
    np.testing.assert_allclose(
        actual_gradient, expected_gradient, rtol=5.0e-8, atol=2.0e-6
    )
    np.testing.assert_allclose(
        actual_hessian, expected_hessian, rtol=2.0e-6, atol=2.0e-3
    )


@pytest.mark.parametrize(
    "positions",
    [
        np.array(((0.0, 0.0, 0.0), (5.0e-13, 0.0, 0.0), (0.1, 0.02, 0.0))),
        np.array(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.0, 0.0))),
        np.array(
            (
                (0.0, 0.0, 0.0),
                (0.1, 0.0, 0.0),
                (0.1 - 0.12 * np.cos(1.0e-7), 0.12 * np.sin(1.0e-7), 0.0),
            )
        ),
    ],
    ids=("near_zero_edge", "exact_antiparallel", "clamped_antiparallel"),
)
def test_cpp_bending_guarded_fallback_preserves_clamp_branches(positions):
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    rest = np.array((0.09, 0.13), dtype=np.float64)
    actual_gradient, actual_hessian = _core(rest).bending_kernel(positions, rest)
    expected_gradient, expected_hessian = _oracle(
        positions[None, ...], rest[:1], rest[1:]
    )
    np.testing.assert_allclose(
        actual_gradient, expected_gradient, rtol=5.0e-9, atol=1.0e-7
    )
    # In the clamped antiparallel branch individual entries reach ~1e27 while
    # exact translation-null entries are formed by cancellation.  Compare the
    # global double-precision error scale instead of assigning relative error
    # to those mathematical zeros.
    hessian_scale = max(1.0, float(np.max(np.abs(expected_hessian))))
    assert float(np.max(np.abs(actual_hessian - expected_hessian))) <= (
        5.0e-13 * hessian_scale + 1.0e-4
    )
