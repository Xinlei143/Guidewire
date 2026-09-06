"""Piecewise-linear 3-D centerlines for curved-vessel simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


_EPS = 1.0e-14


def _unit(values: np.ndarray, *, label: str) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= _EPS):
        raise ValueError(f"{label} contains a degenerate direction")
    return values / norms


@dataclass(frozen=True)
class Centerline3D:
    points: np.ndarray
    cumulative_s: np.ndarray

    @classmethod
    def from_csv(cls, path: str | Path) -> "Centerline3D":
        path = Path(path)
        try:
            values = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
        except (OSError, ValueError) as error:
            raise ValueError(f"could not read centerline CSV {path}: {error}") from error
        if values.dtype.names != ("x", "y", "z"):
            raise ValueError("centerline CSV header must be exactly x,y,z")
        points = np.column_stack(tuple(np.atleast_1d(values[name]) for name in values.dtype.names))
        if len(points) < 2:
            raise ValueError("centerline CSV must contain at least two points")
        if not np.isfinite(points).all():
            raise ValueError("centerline coordinates must be finite")
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.any(lengths <= _EPS):
            raise ValueError("centerline contains duplicate adjacent points")
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        return cls(np.ascontiguousarray(points), np.ascontiguousarray(cumulative))

    @property
    def length(self) -> float:
        return float(self.cumulative_s[-1])

    def _checked_s(self, s: float | np.ndarray) -> tuple[np.ndarray, bool]:
        raw = np.asarray(s, dtype=np.float64)
        scalar = raw.ndim == 0
        query = raw.reshape(-1)
        if not np.isfinite(query).all():
            raise ValueError("centerline arclength queries must be finite")
        tolerance = 2.0e-12 * max(1.0, self.length)
        if np.any(query < -tolerance) or np.any(query > self.length + tolerance):
            raise ValueError(f"centerline arclength must lie in [0, {self.length:.17g}]")
        return np.clip(query, 0.0, self.length), scalar

    def point_at_s(self, s: float | np.ndarray) -> np.ndarray:
        query, scalar = self._checked_s(s)
        indices = np.searchsorted(self.cumulative_s, query, side="right") - 1
        indices = np.clip(indices, 0, len(self.points) - 2)
        segment_length = self.cumulative_s[indices + 1] - self.cumulative_s[indices]
        fraction = (query - self.cumulative_s[indices]) / segment_length
        result = self.points[indices] + fraction[:, None] * (
            self.points[indices + 1] - self.points[indices]
        )
        return result[0] if scalar else result

    def sample_by_arclength(self, rod_length: float, num_nodes: int) -> np.ndarray:
        if not np.isfinite(rod_length) or rod_length <= 0.0:
            raise ValueError("rod_length must be finite and positive")
        if num_nodes < 2:
            raise ValueError("num_nodes must be at least 2")
        if rod_length > self.length + 2.0e-12 * max(1.0, self.length):
            raise ValueError(
                f"centerline length {self.length:.17g} is shorter than rod length {rod_length:.17g}"
            )
        return self.point_at_s(np.linspace(0.0, min(rod_length, self.length), num_nodes))

    def frames_at_s(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query, _ = self._checked_s(s)
        if len(query) < 2:
            raise ValueError("at least two arclength samples are required for frames")
        positions = self.point_at_s(query)
        tangents = np.empty_like(positions)
        tangents[0] = positions[1] - positions[0]
        tangents[-1] = positions[-1] - positions[-2]
        if len(positions) > 2:
            tangents[1:-1] = positions[2:] - positions[:-2]
        tangents = _unit(tangents, label="centerline tangent")

        reference = np.array((0.0, 0.0, 1.0))
        normal0 = reference - tangents[0] * float(reference @ tangents[0])
        if np.linalg.norm(normal0) <= 1.0e-10:
            reference = np.array((0.0, 1.0, 0.0))
            normal0 = reference - tangents[0] * float(reference @ tangents[0])
        normals = np.empty_like(tangents)
        binormals = np.empty_like(tangents)
        normals[0] = _unit(normal0[None, :], label="initial Bishop normal")[0]
        binormals[0] = _unit(np.cross(tangents[0], normals[0])[None, :], label="initial Bishop binormal")[0]
        for index in range(1, len(tangents)):
            previous_tangent = tangents[index - 1]
            tangent = tangents[index]
            axis = np.cross(previous_tangent, tangent)
            sine = float(np.linalg.norm(axis))
            cosine = float(np.clip(previous_tangent @ tangent, -1.0, 1.0))
            if sine <= 1.0e-12:
                if cosine < 0.0:
                    raise ValueError("centerline tangent reverses by 180 degrees")
                transported = normals[index - 1]
            else:
                axis /= sine
                previous = normals[index - 1]
                transported = (
                    previous * cosine
                    + np.cross(axis, previous) * sine
                    + axis * float(axis @ previous) * (1.0 - cosine)
                )
            transported -= tangent * float(tangent @ transported)
            normals[index] = _unit(transported[None, :], label="Bishop normal")[0]
            binormals[index] = _unit(
                np.cross(tangent, normals[index])[None, :], label="Bishop binormal"
            )[0]
        return tangents, normals, binormals

    def project(self, points: np.ndarray, *, chunk_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(points, dtype=np.float64)
        scalar = query.ndim == 1
        query = np.atleast_2d(query)
        if query.shape[1] != 3 or not np.isfinite(query).all():
            raise ValueError("projection points must be finite with shape (N,3)")
        starts = self.points[:-1]
        vectors = self.points[1:] - starts
        squared = np.sum(vectors * vectors, axis=1)
        projected_s = np.empty(len(query), dtype=np.float64)
        distances = np.empty(len(query), dtype=np.float64)
        for begin in range(0, len(query), chunk_size):
            batch = query[begin:begin + chunk_size]
            offsets = batch[:, None, :] - starts[None, :, :]
            fraction = np.clip(np.sum(offsets * vectors[None, :, :], axis=2) / squared, 0.0, 1.0)
            closest = starts[None, :, :] + fraction[:, :, None] * vectors[None, :, :]
            distance2 = np.sum((batch[:, None, :] - closest) ** 2, axis=2)
            best = np.argmin(distance2, axis=1)
            rows = np.arange(len(batch))
            projected_s[begin:begin + len(batch)] = (
                self.cumulative_s[best] + fraction[rows, best] * np.sqrt(squared[best])
            )
            distances[begin:begin + len(batch)] = np.sqrt(distance2[rows, best])
        if scalar:
            return projected_s[0], distances[0]
        return projected_s, distances


__all__ = ["Centerline3D"]
