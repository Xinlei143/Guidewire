"""Lazy loader and dtype/device boundary for the C++ DER CPU backend.

The extension is compiled once with PyTorch's bundled C++ extension toolchain.
The numerical core itself uses pybind11 buffers and Eigen fixed-size kernels;
the Python boundary accepts NumPy arrays so it can also be tested without
coupling the extension to ATen's tensor ABI.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


_ROOT = Path(__file__).resolve().parent
_MODULE = None


def _load_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    from torch.utils.cpp_extension import load

    source_files = [
        str(_ROOT / "cpp" / "der_core.cpp"),
        str(_ROOT / "cpp" / "der_bending.cpp"),
        str(_ROOT / "cpp" / "mesh_bvh.cpp"),
        str(_ROOT / "cpp" / "arc2d_core.cpp"),
        str(_ROOT / "cpp" / "bindings.cpp"),
    ]
    digest = hashlib.sha1()
    for path in source_files + [
        str(_ROOT / "cpp" / "der_core.h"),
        str(_ROOT / "cpp" / "der_bending.h"),
        str(_ROOT / "cpp" / "mesh_bvh.h"),
        str(_ROOT / "cpp" / "arc2d_core.h"),
    ]:
        digest.update(Path(path).read_bytes())
    name = f"der_core_cpp_{digest.hexdigest()[:12]}"
    build_directory = Path(tempfile.gettempdir()) / "softrobot_der_cpp"
    build_directory.mkdir(parents=True, exist_ok=True)
    _MODULE = load(
        name=name,
        sources=source_files,
        build_directory=str(build_directory),
        extra_include_paths=["/usr/include/eigen3"],
        extra_cflags=["-O3", "-DNDEBUG", "-DEIGEN_NO_DEBUG", "-g0", "-std=c++17"],
        verbose=False,
    )
    return _MODULE


def _array(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU array/tensor")
        value = value.detach().numpy()
    result = np.asarray(value)
    if result.dtype != np.float64:
        raise TypeError(f"{name} must have dtype float64")
    if not result.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return result


class DERCore:
    """Python façade whose methods preserve the C++ core's strict boundary."""

    def __init__(self, rest_lengths, masses, ea, ei, damping, tube_radius,
                 safe_distance, kp, ki, max_pi_force, leak_rate,
                 max_integral_state, contact_smooth_length=2.0e-5,
                 direction_epsilon=1.0e-6):
        self._rest_lengths = _array(rest_lengths, "rest_lengths")
        self._masses = _array(masses, "masses")
        self._ki = float(ki)
        self._impl = _load_module().DERCore(
            self._rest_lengths,
            self._masses,
            float(ea),
            float(ei),
            float(damping),
            float(tube_radius),
            float(safe_distance),
            float(kp),
            float(ki),
            float(max_pi_force),
            float(leak_rate),
            float(max_integral_state),
            float(contact_smooth_length),
            float(direction_epsilon),
        )

    def stretch_kernel(self, positions, rest_lengths=None):
        return self._impl.stretch_kernel(
            _array(positions, "positions"),
            _array(self._rest_lengths if rest_lengths is None else rest_lengths,
                   "rest_lengths"),
        )

    def bending_kernel(self, positions, rest_lengths=None):
        return self._impl.bending_kernel(
            _array(positions, "positions"),
            _array(self._rest_lengths if rest_lengths is None else rest_lengths,
                   "rest_lengths"),
        )

    def contact_kernel(self, positions, integrals, previous_errors, dt,
                       ki_active=None):
        if ki_active is None:
            ki_active = self._ki
        return self._impl.contact_kernel(
            _array(positions, "positions"),
            _array(integrals, "integrals"),
            _array(previous_errors, "previous_errors"),
            float(dt),
            float(ki_active),
        )

    def configure_mesh(self, triangles, normals, triangle_ids):
        tri = np.asarray(triangles)
        nor = np.asarray(normals)
        ids = np.asarray(triangle_ids)
        if tri.dtype != np.float64 or nor.dtype != np.float64:
            raise TypeError("triangles and normals must have dtype float64")
        if ids.dtype != np.int64:
            raise TypeError("triangle_ids must have dtype int64")
        if not tri.flags.c_contiguous or not nor.flags.c_contiguous or not ids.flags.c_contiguous:
            raise ValueError("mesh geometry arrays must be C-contiguous")
        self._impl.configure_mesh(tri, nor, ids)

    def implicit_euler_step(self, positions, velocities, *args):
        """Run one step using either the legacy or cached production signature.

        Legacy: ``rest_lengths, masses, integrals, ...``.  Production:
        ``integrals, initial_positions, ...``; the constructor-owned geometry
        buffers are reused in that form.
        """
        if len(args) == 10:
            return self.implicit_euler_step_cached(
                positions, velocities, *args
            )
        if len(args) != 12:
            raise TypeError("implicit_euler_step expects 10 (cached) or 12 (legacy) trailing arguments")
        (rest_lengths, masses, integrals, initial_positions, pulse_forces,
         tip_constrained, tip_displacement, dt, max_newton_iters,
         residual_tol, step_tol, ki_active) = args
        return self._impl.implicit_euler_step(
            _array(positions, "positions"),
            _array(velocities, "velocities"),
            _array(rest_lengths, "rest_lengths"),
            _array(masses, "masses"),
            _array(integrals, "integrals"),
            _array(initial_positions, "initial_positions"),
            _array(pulse_forces, "pulse_forces"),
            bool(tip_constrained),
            float(tip_displacement),
            float(dt),
            int(max_newton_iters),
            float(residual_tol),
            float(step_tol),
            float(ki_active),
        )

    def implicit_euler_step_cached(
        self, positions, velocities, integrals, initial_positions, pulse_forces,
        tip_constrained, tip_displacement, dt, max_newton_iters, residual_tol,
        step_tol, ki_active,
    ):
        return self._impl.implicit_euler_step_cached(
            _array(positions, "positions"),
            _array(velocities, "velocities"),
            _array(integrals, "integrals"),
            _array(initial_positions, "initial_positions"),
            _array(pulse_forces, "pulse_forces"),
            bool(tip_constrained),
            float(tip_displacement),
            float(dt),
            int(max_newton_iters),
            float(residual_tol),
            float(step_tol),
            float(ki_active),
        )

    def implicit_euler_step_prescribed(
        self, positions, velocities, rest_lengths, masses, integrals,
        prescribed_positions, pulse_forces, tip_constrained, dt,
        max_newton_iters, residual_tol, step_tol, ki_active,
    ):
        return self._impl.implicit_euler_step_prescribed(
            _array(positions, "positions"),
            _array(velocities, "velocities"),
            _array(rest_lengths, "rest_lengths"),
            _array(masses, "masses"),
            _array(integrals, "integrals"),
            _array(prescribed_positions, "prescribed_positions"),
            _array(pulse_forces, "pulse_forces"),
            bool(tip_constrained), float(dt), int(max_newton_iters),
            float(residual_tol), float(step_tol), float(ki_active),
        )

    def implicit_euler_step_prescribed_cached(
        self, positions, velocities, integrals, prescribed_positions,
        pulse_forces, tip_constrained, dt, max_newton_iters,
        residual_tol, step_tol, ki_active,
    ):
        return self._impl.implicit_euler_step_prescribed_cached(
            _array(positions, "positions"),
            _array(velocities, "velocities"),
            _array(integrals, "integrals"),
            _array(prescribed_positions, "prescribed_positions"),
            _array(pulse_forces, "pulse_forces"),
            bool(tip_constrained), float(dt), int(max_newton_iters),
            float(residual_tol), float(step_tol), float(ki_active),
        )


class Arc2DCore:
    """Strict NumPy facade for the independent CPU float64 Arc2D core."""

    def __init__(
        self, rest_lengths, masses, ea, ei, damping, arc_radius,
        tube_half_width, safe_distance, kp, ki, max_pi_force,
        max_integral_state, leak_rate, contact_smooth_length=2.0e-5,
        direction_epsilon=1.0e-6,
    ):
        self._rest_lengths = _array(rest_lengths, "rest_lengths")
        self._masses = _array(masses, "masses")
        self._kp = float(kp)
        self._ki = float(ki)
        self._impl = _load_module().Arc2DCore(
            self._rest_lengths,
            self._masses,
            float(ea),
            float(ei),
            float(damping),
            float(arc_radius),
            float(tube_half_width),
            float(safe_distance),
            self._kp,
            self._ki,
            float(max_pi_force),
            float(max_integral_state),
            float(leak_rate),
            float(contact_smooth_length),
            float(direction_epsilon),
        )

    def contact_kernel(
        self, positions, integrals, previous_errors, dt,
        kp_active=None, ki_active=None,
    ):
        return self._impl.contact_kernel(
            _array(positions, "positions"),
            _array(integrals, "integrals"),
            _array(previous_errors, "previous_errors"),
            float(dt),
            self._kp if kp_active is None else float(kp_active),
            self._ki if ki_active is None else float(ki_active),
        )

    def implicit_euler_step(
        self, positions, velocities, integrals, previous_errors, prescribed,
        pulse_forces, reference_curvature, tip_constrained, dt,
        max_newton_iters, residual_tol, step_tol, kp_active=None,
        ki_active=None,
    ):
        return self._impl.implicit_euler_step(
            _array(positions, "positions"),
            _array(velocities, "velocities"),
            _array(integrals, "integrals"),
            _array(previous_errors, "previous_errors"),
            _array(prescribed, "prescribed"),
            _array(pulse_forces, "pulse_forces"),
            _array(reference_curvature, "reference_curvature"),
            bool(tip_constrained),
            float(dt),
            int(max_newton_iters),
            float(residual_tol),
            float(step_tol),
            self._kp if kp_active is None else float(kp_active),
            self._ki if ki_active is None else float(ki_active),
        )

    def implicit_euler_step_cached(self, *args, **kwargs):
        """Alias retaining the constructor-owned rest/mass cache contract."""
        return self.implicit_euler_step(*args, **kwargs)


__all__ = ["DERCore", "Arc2DCore"]
