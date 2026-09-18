"""Deterministic two-dimensional sinusoidal tube geometry."""

from dataclasses import dataclass

import numpy as np


_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(256)
_EPS = np.finfo(np.float64).eps
_TRACKING_SCREEN_INTERVALS = 64


@dataclass(frozen=True)
class ProjectionResult:
    """Closest-centerline information for one point or a batch of points."""

    q: np.ndarray | float
    point: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    signed_distance: np.ndarray | float
    used_global_fallback: np.ndarray | bool

    @property
    def x(self):
        return self.q

    @property
    def points(self):
        return self.point


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray, tol: float) -> bool:
    return (
        min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol
        and min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol
        and abs(_cross(b - a, p - a)) <= tol
    )


def _segments_intersect(a, b, c, d, tol: float = 1.0e-12) -> bool:
    """Return whether two closed two-dimensional line segments intersect."""
    ab_c, ab_d = _cross(b - a, c - a), _cross(b - a, d - a)
    cd_a, cd_b = _cross(d - c, a - c), _cross(d - c, b - c)
    if ((ab_c > tol and ab_d < -tol) or (ab_c < -tol and ab_d > tol)) and (
        (cd_a > tol and cd_b < -tol) or (cd_a < -tol and cd_b > tol)
    ):
        return True
    return (
        (abs(ab_c) <= tol and _on_segment(a, b, c, tol))
        or (abs(ab_d) <= tol and _on_segment(a, b, d, tol))
        or (abs(cd_a) <= tol and _on_segment(c, d, a, tol))
        or (abs(cd_b) <= tol and _on_segment(c, d, b, tol))
    )


def _assert_polyline_simple(polyline: np.ndarray, label: str) -> None:
    starts, ends = polyline[:-1], polyline[1:]
    n = len(starts)
    for i in range(n):
        lo = np.maximum(np.minimum(starts[i], ends[i]), np.minimum(starts, ends))
        hi = np.minimum(np.maximum(starts[i], ends[i]), np.maximum(starts, ends))
        possible = np.flatnonzero(np.all(lo <= hi + 1.0e-12, axis=1))
        for j in possible:
            if j <= i + 1:
                continue
            if _segments_intersect(starts[i], ends[i], starts[j], ends[j]):
                raise ValueError(f"{label} wall polyline self-intersects")


def validate_offset_polylines(centerline: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> None:
    """Validate sampled offsets are finite, simple, and do not cross each other."""
    arrays = tuple(np.asarray(p, dtype=np.float64) for p in (centerline, upper, lower))
    if any(p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 for p in arrays):
        raise ValueError("offset polylines must have at least two finite 2-D points")
    if any(not np.all(np.isfinite(p)) for p in arrays):
        raise ValueError("offset polylines must be finite")
    if not (len(arrays[0]) == len(arrays[1]) == len(arrays[2])):
        raise ValueError("offset polylines must have matching samples")
    _assert_polyline_simple(arrays[1], "upper")
    _assert_polyline_simple(arrays[2], "lower")

    upper_start, upper_end = arrays[1][:-1], arrays[1][1:]
    lower_start, lower_end = arrays[2][:-1], arrays[2][1:]
    for i, (a, b) in enumerate(zip(upper_start, upper_end)):
        lo = np.maximum(np.minimum(a, b), np.minimum(lower_start, lower_end))
        hi = np.minimum(np.maximum(a, b), np.maximum(lower_start, lower_end))
        for j in np.flatnonzero(np.all(lo <= hi + 1.0e-12, axis=1)):
            if _segments_intersect(a, b, lower_start[j], lower_end[j]):
                raise ValueError("upper and lower wall polylines intersect")


class SineTube2D:
    """Tube around ``y = amplitude * cos(3*pi*x/x_span)``."""

    def __init__(self, amplitude: float = 0.225, x_span: float = 1.5,
                 tube_half_width: float = 0.05):
        self.amplitude = float(amplitude)
        self.x_span = float(x_span)
        self.tube_half_width = float(tube_half_width)
        if not all(np.isfinite(v) and v > 0.0 for v in (
            self.amplitude, self.x_span, self.tube_half_width,
        )):
            raise ValueError("amplitude, x_span, and tube_half_width must be finite and positive")
        self._k = 3.0 * np.pi / self.x_span
        self._minimum_radius = 1.0 / (self.amplitude * self._k ** 2)
        if self._minimum_radius <= self.tube_half_width:
            raise ValueError("tube_half_width must be smaller than the minimum curvature radius")
        self.validate_geometry()

    @property
    def length_x(self) -> float:
        return self.x_span

    @property
    def half_width(self) -> float:
        return self.tube_half_width

    @property
    def min_radius(self) -> float:
        return self._minimum_radius

    @property
    def vessel_length(self) -> float:
        return float(self.arclength(self.x_span))

    def _q(self, q):
        values = np.asarray(q, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > self.x_span):
            raise ValueError(f"q must be finite and lie in [0, {self.x_span:.17g}]")
        return values

    def centerline(self, q):
        q = self._q(q)
        return np.stack((q, self.amplitude * np.cos(self._k * q)), axis=-1)

    def derivative(self, q):
        q = self._q(q)
        return np.stack((np.ones_like(q), -self.amplitude * self._k * np.sin(self._k * q)), axis=-1)

    def _second_derivative(self, q):
        q = self._q(q)
        return np.stack((np.zeros_like(q), -self.amplitude * self._k ** 2 * np.cos(self._k * q)), axis=-1)

    def tangent(self, q):
        derivative = self.derivative(q)
        return derivative / np.linalg.norm(derivative, axis=-1, keepdims=True)

    def normal(self, q):
        derivative = self.derivative(q)
        normal = np.stack((-derivative[..., 1], derivative[..., 0]), axis=-1)
        return normal / np.linalg.norm(normal, axis=-1, keepdims=True)

    def curvature(self, q):
        q = self._q(q)
        slope = -self.amplitude * self._k * np.sin(self._k * q)
        return -self.amplitude * self._k ** 2 * np.cos(self._k * q) / (1.0 + slope ** 2) ** 1.5

    def minimum_curvature_radius(self) -> float:
        return self._minimum_radius

    def _speed(self, q):
        return np.linalg.norm(self.derivative(q), axis=-1)

    def arclength(self, q):
        q = self._q(q)
        flat = q.reshape(-1)
        nodes = 0.5 * flat[:, None] * (_GAUSS_X + 1.0)
        integral = 0.5 * flat * np.sum(_GAUSS_W * self._speed(nodes), axis=1)
        return integral.reshape(q.shape)[()] if q.ndim == 0 else integral.reshape(q.shape)

    def parameter_at_arclength(self, s):
        s = np.asarray(s, dtype=np.float64)
        total = float(self.arclength(self.x_span))
        if not np.all(np.isfinite(s)) or np.any(s < 0.0) or np.any(s > total):
            raise ValueError(f"arclength must be finite and lie in [0, {total:.17g}]")
        q = s * (self.x_span / total)
        for _ in range(12):
            q = np.clip(q - (self.arclength(q) - s) / self._speed(q), 0.0, self.x_span)
        return q[()] if s.ndim == 0 else q

    inverse_arclength = parameter_at_arclength

    def _objective(self, q: float, point: np.ndarray) -> tuple[float, float, float]:
        delta = self.centerline(q) - point
        first = self.derivative(q)
        second = self._second_derivative(q)
        return (0.5 * float(delta @ delta), float(delta @ first),
                float(first @ first + delta @ second))

    def _warm_refine(self, q: float, point: np.ndarray) -> float:
        """Bounded Newton minimization with monotonic backtracking."""
        q = float(np.clip(q, 0.0, self.x_span))
        for _ in range(32):
            value, gradient, hessian = self._objective(q, point)
            if abs(gradient) <= 32.0 * _EPS:
                break
            step = -gradient / hessian if hessian > 1.0e-14 else -np.sign(gradient) * self.x_span / 16.0
            step = float(np.clip(step, -self.x_span / 4.0, self.x_span / 4.0))
            trial = float(np.clip(q + step, 0.0, self.x_span))
            for _ in range(32):
                if trial == q or self._objective(trial, point)[0] <= value:
                    break
                next_trial = 0.5 * (trial + q)
                if next_trial == trial or next_trial == q:
                    trial = q
                    break
                trial = next_trial
            if trial == q:
                break
            if abs(trial - q) <= 16.0 * _EPS * max(1.0, self.x_span):
                q = trial
                break
            q = trial
        return q

    def _bracketed_stationary(self, left: float, right: float, point: np.ndarray) -> float:
        """Safeguarded Newton solve of the distance derivative in a sign bracket."""
        g_left = self._objective(left, point)[1]
        q = 0.5 * (left + right)
        for _ in range(48):
            _, gradient, hessian = self._objective(q, point)
            if abs(gradient) <= 32.0 * _EPS:
                return q
            newton = q - gradient / hessian if abs(hessian) > 1.0e-14 else np.nan
            candidate = newton if left < newton < right else 0.5 * (left + right)
            g_candidate = self._objective(candidate, point)[1]
            if (g_left <= 0.0 <= g_candidate) or (g_left >= 0.0 >= g_candidate):
                right = candidate
            else:
                left, g_left = candidate, g_candidate
            q = candidate
            if right - left <= 32.0 * _EPS * max(1.0, self.x_span):
                break
        return q

    def _screened_global_candidate(self, point: np.ndarray) -> float:
        """Cheaply seed the best distant lobe before trusting a cached minimum."""
        grid = np.linspace(
            0.0, self.x_span, _TRACKING_SCREEN_INTERVALS + 1, dtype=np.float64
        )
        delta = self.centerline(grid) - point
        values = 0.5 * np.sum(delta * delta, axis=1)
        seed = float(grid[int(np.argmin(values))])
        return self._warm_refine(seed, point)

    def _project_one(self, point: np.ndarray, initial_q: float | None):
        seed = 0.5 * self.x_span if initial_q is None else float(initial_q)
        warm_q = self._warm_refine(seed, point)
        candidates: list[tuple[float, str]] = [(0.0, "endpoint"), (self.x_span, "endpoint"), (warm_q, "warm")]

        grid = np.linspace(0.0, self.x_span, 513, dtype=np.float64)
        values = np.array([self._objective(q, point)[0] for q in grid])
        gradients = np.array([self._objective(q, point)[1] for q in grid])
        local = np.flatnonzero((values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:])) + 1
        candidates.extend((self._warm_refine(float(grid[i]), point), "global") for i in local)
        sign_changes = np.flatnonzero(gradients[:-1] * gradients[1:] < 0.0)
        candidates.extend(
            (self._bracketed_stationary(float(grid[i]), float(grid[i + 1]), point), "global")
            for i in sign_changes
        )
        scored = [(self._objective(q, point)[0], q, source) for q, source in candidates]
        _, q, source = min(scored, key=lambda item: (item[0], item[1]))
        center = self.centerline(q)
        normal = self.normal(q)
        return q, center, self.tangent(q), normal, float((point - center) @ normal), source == "global"

    def project(self, point, initial_q=None) -> ProjectionResult:
        points = np.asarray(point, dtype=np.float64)
        scalar = points.ndim == 1
        if (scalar and points.shape != (2,)) or (not scalar and (points.ndim != 2 or points.shape[1] != 2)):
            raise ValueError("point must be a finite 2-D point or an N-by-2 array")
        if not np.all(np.isfinite(points)):
            raise ValueError("point must be finite")
        batch = points[None, :] if scalar else points
        starts = np.full(len(batch), 0.5 * self.x_span) if initial_q is None else np.broadcast_to(
            np.asarray(initial_q, dtype=np.float64), (len(batch),)
        )
        if not np.all(np.isfinite(starts)):
            raise ValueError("initial_q must be finite")
        result = [self._project_one(p, q) for p, q in zip(batch, starts)]
        q, centers, tangents, normals, distance, fallback = (np.asarray(v) for v in zip(*result))
        if scalar:
            return ProjectionResult(float(q[0]), centers[0], tangents[0], normals[0], float(distance[0]), bool(fallback[0]))
        return ProjectionResult(q, centers, tangents, normals, distance, fallback)

    def project_tracked(self, point, accepted_q, trust_radius: float = 0.02) -> ProjectionResult:
        """Track nearby closest points and fall back to :meth:`project` if untrusted.

        ``accepted_q`` is the closest-centerline parameter from the preceding
        accepted nonlinear iterate.  A bounded warm Newton result is used only
        when it stays inside ``trust_radius``, improves the distance objective,
        and satisfies the first-order condition for an interior or constrained
        endpoint minimum.  Every other case delegates to the existing global
        projection, so this fast path never weakens ``project`` semantics.
        """
        points = np.asarray(point, dtype=np.float64)
        scalar = points.ndim == 1
        if (scalar and points.shape != (2,)) or (
            not scalar and (points.ndim != 2 or points.shape[1] != 2)
        ):
            raise ValueError("point must be a finite 2-D point or an N-by-2 array")
        if not np.all(np.isfinite(points)):
            raise ValueError("point must be finite")
        if not np.isfinite(trust_radius) or trust_radius <= 0.0:
            raise ValueError("trust_radius must be finite and positive")
        batch = points[None, :] if scalar else points
        starts = np.broadcast_to(np.asarray(accepted_q, dtype=np.float64), (len(batch),))
        if not np.all(np.isfinite(starts)) or np.any(starts < 0.0) or np.any(starts > self.x_span):
            raise ValueError(f"accepted_q must be finite and lie in [0, {self.x_span:.17g}]")

        tracked = []
        stationarity_tol = 256.0 * _EPS * max(1.0, self.x_span)
        for current_point, start in zip(batch, starts):
            warm_q = self._warm_refine(float(start), current_point)
            value, gradient, hessian = self._objective(warm_q, current_point)
            start_value = self._objective(float(start), current_point)[0]
            endpoint_ok = (
                (warm_q <= stationarity_tol and gradient >= -stationarity_tol)
                or (warm_q >= self.x_span - stationarity_tol and gradient <= stationarity_tol)
            )
            interior_ok = (
                stationarity_tol < warm_q < self.x_span - stationarity_tol
                and abs(gradient) <= stationarity_tol
                and hessian > 0.0
            )
            objective_ok = value <= start_value + 128.0 * _EPS * max(1.0, start_value)
            trusted = (
                abs(warm_q - float(start)) <= trust_radius
                and objective_ok
                and (interior_ok or endpoint_ok)
            )
            if trusted:
                screened_q = self._screened_global_candidate(current_point)
                screened_value = self._objective(screened_q, current_point)[0]
                comparison_tol = 128.0 * _EPS * max(
                    1.0, abs(value), abs(screened_value)
                )
                competing_lobe = (
                    abs(screened_q - warm_q) > stationarity_tol
                    and screened_value <= value + comparison_tol
                )
                trusted = not competing_lobe
            if trusted:
                center = self.centerline(warm_q)
                tangent = self.tangent(warm_q)
                normal = self.normal(warm_q)
                distance = float((current_point - center) @ normal)
                tracked.append((warm_q, center, tangent, normal, distance, False))
            else:
                exact = self.project(current_point, initial_q=float(start))
                tracked.append((
                    exact.q, exact.point, exact.tangent, exact.normal,
                    exact.signed_distance, True,
                ))

        q, centers, tangents, normals, distance, fallback = (
            np.asarray(value) for value in zip(*tracked)
        )
        if scalar:
            return ProjectionResult(
                float(q[0]), centers[0], tangents[0], normals[0],
                float(distance[0]), bool(fallback[0]),
            )
        return ProjectionResult(q, centers, tangents, normals, distance, fallback)

    def signed_normal_distance(self, point, initial_q=None):
        return self.project(point, initial_q=initial_q).signed_distance

    def walls(self, count):
        if isinstance(count, (int, np.integer)):
            if count < 2:
                raise ValueError("wall sample count must be at least two")
            q = np.linspace(0.0, self.x_span, int(count), dtype=np.float64)
        else:
            q = self._q(count)
        center = self.centerline(q)
        normal = self.normal(q)
        return center + self.tube_half_width * normal, center - self.tube_half_width * normal

    def validate_geometry(self, count: int = 257) -> None:
        if not isinstance(count, (int, np.integer)) or count < 4:
            raise ValueError("validation sample count must be an integer of at least four")
        if self._minimum_radius <= self.tube_half_width:
            raise ValueError("tube_half_width must be smaller than the minimum curvature radius")
        q = np.linspace(0.0, self.x_span, int(count), dtype=np.float64)
        upper, lower = self.walls(q)
        validate_offset_polylines(self.centerline(q), upper, lower)
