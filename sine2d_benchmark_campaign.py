"""Formal convergence and friction-sweep orchestration for the sine benchmark.

The runner deliberately creates a new campaign directory and never resumes or
overwrites one.  This keeps expensive numerical evidence attributable to one
immutable configuration; use a new ``--campaign-name`` for every rerun.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping as MappingABC
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
from typing import Callable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rod_soft_contact_PI_dynamic_quasi_static_2d_sine_insert import (
    SineInsertConfig,
)


NOMINAL_MU = 0.10
SWEEP_MU = (0.00, 0.05, 0.10, 0.20)
REQUIRED_FILE_ARTIFACT_KEYS = frozenset(
    {
        "config",
        "settled",
        "history",
        "configurations",
        "insertion_force_plot",
        "normal_friction_plot",
        "elastic_energy_plot",
    }
)
REQUIRED_ARTIFACT_KEYS = REQUIRED_FILE_ARTIFACT_KEYS | {"run_dir"}


@dataclass(frozen=True)
class ConvergenceThresholds:
    force_relative_l2: float = 0.05
    force_peak_relative: float = 0.05
    force_final_relative: float = 0.05
    tip_s_difference_m: float = 0.005
    max_penetration_difference_m: float = 0.0001


@dataclass(frozen=True)
class CampaignSettings:
    output_dir: str = "2d_npz"
    campaign_name: str = "sine2d_formal_campaign"
    backend: str = "cpp"
    study: str = "all"
    grid_samples: int = 241

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("output_dir must be a non-empty string")
        if (
            not isinstance(self.campaign_name, str)
            or not self.campaign_name.strip()
            or self.campaign_name in (".", "..")
            or "/" in self.campaign_name
            or "\\" in self.campaign_name
        ):
            raise ValueError("campaign_name must be a safe directory name")
        if self.backend not in ("python", "cpp"):
            raise ValueError("backend must be 'python' or 'cpp'")
        if self.study not in ("convergence", "sweep", "all"):
            raise ValueError("study must be 'convergence', 'sweep', or 'all'")
        if (
            not isinstance(self.grid_samples, int)
            or isinstance(self.grid_samples, bool)
            or self.grid_samples < 2
        ):
            raise ValueError("grid_samples must be an integer of at least two")


@dataclass(frozen=True)
class SettledCurve:
    insertion: np.ndarray
    insertion_force: np.ndarray
    tip_s: np.ndarray
    max_penetration: np.ndarray

    def __post_init__(self) -> None:
        arrays = []
        for name in (
            "insertion",
            "insertion_force",
            "tip_s",
            "max_penetration",
        ):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.ndim != 1 or len(value) < 2:
                raise ValueError(f"{name} must be a one-dimensional array of length >= 2")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
            arrays.append(value)
        if len({len(value) for value in arrays}) != 1:
            raise ValueError("settled curve arrays must have equal lengths")
        if not np.all(np.diff(self.insertion) > 0.0):
            raise ValueError("insertion samples must be strictly increasing")
        if np.any(self.max_penetration < 0.0):
            raise ValueError("max penetration must be non-negative")


@dataclass(frozen=True)
class PairMetrics:
    force_relative_l2: float
    force_peak_relative: float
    force_final_relative: float
    tip_s_difference_m: float
    max_penetration_difference_m: float
    passed: bool
    insertion_grid: np.ndarray


@dataclass(frozen=True)
class CampaignResult:
    campaign_dir: Path
    manifest: Path
    convergence_status: str
    sweep_status: str
    overall_status: str
    sweep_summary: Path | None
    sweep_plot: Path | None


@dataclass
class _Case:
    name: str
    config: SineInsertConfig
    result: object | None
    artifacts: Mapping[str, Path]
    error: str | None

    @property
    def status(self) -> str:
        if self.error is not None or self.result is None:
            return "incomplete"
        if not bool(getattr(self.result, "success", False)):
            return "numerical_failure"
        if getattr(self.result, "completed_load_steps", None) != self.config.num_load_steps:
            return "incomplete"
        try:
            extract_settled_curve(self.result)
        except (TypeError, ValueError, AttributeError):
            return "incomplete"
        if _settled_load_target_issue(self.result, self.config) is not None:
            return "incomplete"
        return "complete"

    @property
    def success(self) -> bool:
        """Whether this case is eligible to support aggregate evidence."""
        return self.status == "complete"

    @property
    def incomplete(self) -> bool:
        return self.status == "incomplete"

    @property
    def eligibility_reason(self) -> str | None:
        if self.error is not None:
            return self.error
        if self.result is None:
            return "simulation result is unavailable"
        if not bool(getattr(self.result, "success", False)):
            return getattr(self.result, "abort_reason", None) or "simulation failed"
        completed = getattr(self.result, "completed_load_steps", None)
        if completed != self.config.num_load_steps:
            return (
                f"completed_load_steps={completed!r}, expected "
                f"{self.config.num_load_steps}"
            )
        try:
            extract_settled_curve(self.result)
        except (TypeError, ValueError, AttributeError) as exc:
            return f"settled curve unavailable: {exc}"
        return _settled_load_target_issue(self.result, self.config)


def extract_settled_curve(result: object) -> SettledCurve:
    """Extract one sorted, unique load curve from a simulation result."""
    records = tuple(getattr(result, "settled_records"))
    unique = {}
    for record in records:
        insertion = float(record.insertion)
        if not math.isfinite(insertion):
            raise ValueError("settled insertion must be finite")
        unique[insertion] = record
    ordered = [unique[key] for key in sorted(unique)]
    if len(ordered) < 2:
        raise ValueError("at least two unique settled insertion states are required")
    return SettledCurve(
        insertion=np.asarray([record.insertion for record in ordered]),
        insertion_force=np.asarray(
            [record.reaction.insertion_force for record in ordered]
        ),
        tip_s=np.asarray([record.tip_s for record in ordered]),
        max_penetration=np.asarray(
            [record.max_contact_error for record in ordered]
        ),
    )


def _settled_load_target_issue(
    result: object, config: SineInsertConfig
) -> str | None:
    """Return why the committed records do not cover the standard load grid."""
    tolerance = max(
        1.0e-10, 1.0e-9 * max(1.0, abs(config.total_insertion))
    )
    try:
        values = sorted(float(record.insertion) for record in result.settled_records)
    except (TypeError, ValueError, AttributeError) as exc:
        return f"settled load targets unavailable: {exc}"
    if not values or not np.all(np.isfinite(values)):
        return "settled load targets must be non-empty and finite"
    unique = []
    for value in values:
        if not unique or abs(value - unique[-1]) > tolerance:
            unique.append(value)
    if unique[0] < -tolerance or unique[-1] > config.total_insertion + tolerance:
        return "settled load targets must lie within the insertion interval"
    expected = (
        config.total_insertion
        * np.arange(1, config.num_load_steps + 1, dtype=np.float64)
        / config.num_load_steps
    )
    actual = np.asarray(unique, dtype=np.float64)
    for index, target in enumerate(expected):
        if float(np.min(np.abs(actual - target))) > tolerance:
            return (
                f"settled load targets are missing index {index + 1}: "
                f"expected {target:.17g} m"
            )
    return None


def unified_insertion_grid(
    curves: tuple[SettledCurve, ...], samples: int
) -> np.ndarray:
    if not curves:
        raise ValueError("at least one curve is required")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 2:
        raise ValueError("samples must be an integer of at least two")
    lower = max(float(curve.insertion[0]) for curve in curves)
    upper = min(float(curve.insertion[-1]) for curve in curves)
    if not upper > lower:
        raise ValueError("settled curves have no non-zero common insertion interval")
    regular = np.linspace(lower, upper, samples, dtype=np.float64)
    knots = [regular]
    for curve in curves:
        inside = curve.insertion[
            (curve.insertion >= lower) & (curve.insertion <= upper)
        ]
        knots.append(inside)
    return np.unique(np.concatenate(knots))


def compare_curves(
    candidate: SettledCurve,
    reference: SettledCurve,
    *,
    samples: int = 241,
    thresholds: ConvergenceThresholds = ConvergenceThresholds(),
) -> PairMetrics:
    """Compare a coarser candidate to a finer reference on one common grid."""
    grid = unified_insertion_grid((candidate, reference), samples)
    candidate_force = np.interp(grid, candidate.insertion, candidate.insertion_force)
    reference_force = np.interp(grid, reference.insertion, reference.insertion_force)
    force_difference = candidate_force - reference_force
    relative_l2 = float(
        np.linalg.norm(force_difference)
        / max(float(np.linalg.norm(reference_force)), 1.0e-12)
    )
    candidate_peak = float(np.max(np.abs(candidate_force)))
    reference_peak = float(np.max(np.abs(reference_force)))
    peak_relative = abs(candidate_peak - reference_peak) / max(
        abs(reference_peak), 1.0e-12
    )
    final_relative = abs(candidate_force[-1] - reference_force[-1]) / max(
        abs(reference_force[-1]), 1.0e-12
    )
    candidate_tip = float(np.interp(grid[-1], candidate.insertion, candidate.tip_s))
    reference_tip = float(np.interp(grid[-1], reference.insertion, reference.tip_s))
    tip_difference = abs(candidate_tip - reference_tip)
    penetration_difference = abs(
        float(np.max(candidate.max_penetration))
        - float(np.max(reference.max_penetration))
    )
    passed = bool(
        relative_l2 <= thresholds.force_relative_l2
        and peak_relative <= thresholds.force_peak_relative
        and final_relative <= thresholds.force_final_relative
        and tip_difference <= thresholds.tip_s_difference_m
        and penetration_difference <= thresholds.max_penetration_difference_m
    )
    grid.setflags(write=False)
    return PairMetrics(
        force_relative_l2=relative_l2,
        force_peak_relative=peak_relative,
        force_final_relative=final_relative,
        tip_s_difference_m=tip_difference,
        max_penetration_difference_m=penetration_difference,
        passed=passed,
        insertion_grid=grid,
    )


def _mu_token(mu: float) -> str:
    return f"{mu:.2f}".replace(".", "p")


def _case_name(nodes: int, steps: int, mu: float) -> str:
    return f"n{nodes:04d}_steps{steps:04d}_mu{_mu_token(mu)}"


def _metrics_payload(metrics: PairMetrics | None) -> dict[str, object]:
    if metrics is None:
        return {"available": False, "passed": False}
    return {
        "available": True,
        "force_relative_l2": metrics.force_relative_l2,
        "force_peak_relative": metrics.force_peak_relative,
        "force_final_relative": metrics.force_final_relative,
        "tip_s_difference_m": metrics.tip_s_difference_m,
        "max_penetration_difference_m": metrics.max_penetration_difference_m,
        "passed": metrics.passed,
        "common_insertion_min_m": float(metrics.insertion_grid[0]),
        "common_insertion_max_m": float(metrics.insertion_grid[-1]),
        "common_grid_samples": len(metrics.insertion_grid),
    }


def _try_compare(
    candidate: _Case,
    reference: _Case,
    settings: CampaignSettings,
    thresholds: ConvergenceThresholds,
) -> PairMetrics | None:
    if not candidate.success or not reference.success:
        return None
    try:
        return compare_curves(
            extract_settled_curve(candidate.result),
            extract_settled_curve(reference.result),
            samples=settings.grid_samples,
            thresholds=thresholds,
        )
    except (TypeError, ValueError, AttributeError):
        return None


def _write_sweep_summary(
    cases: tuple[_Case, ...], path: Path, samples: int
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {path}")
    curves = tuple(extract_settled_curve(case.result) for case in cases)
    grid = unified_insertion_grid(curves, samples)
    forces = tuple(
        np.interp(grid, curve.insertion, curve.insertion_force) for curve in curves
    )
    fields = ["insertion_m"] + [
        f"insertion_force_mu_{_mu_token(mu)}_N" for mu in SWEEP_MU
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(fields)
            for index, insertion in enumerate(grid):
                writer.writerow(
                    [f"{insertion:.17g}"]
                    + [f"{force[index]:.17g}" for force in forces]
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return grid, forces


def _write_sweep_plot(
    grid: np.ndarray, forces: tuple[np.ndarray, ...], path: Path
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    fig, axis = plt.subplots(figsize=(7.0, 4.4))
    try:
        for mu, force in zip(SWEEP_MU, forces):
            axis.plot(grid, force, linewidth=1.5, label=f"mu={mu:.2f}")
        axis.set_xlabel("insertion (m)")
        axis.set_ylabel("static actuator force on wire, x (N)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _lexical_absolute(path: Path) -> Path:
    """Normalize ``.``/``..`` without resolving any symbolic-link component."""
    return Path(os.path.abspath(os.fspath(path)))


def run_campaign(
    settings: CampaignSettings,
    *,
    simulation_runner: Callable[[SineInsertConfig], object] | None = None,
    simulation_exporter: Callable[[object], Mapping[str, Path]] | None = None,
) -> CampaignResult:
    """Run the requested formal studies in a fresh, non-overwriting directory."""
    if not isinstance(settings, CampaignSettings):
        raise TypeError("settings must be CampaignSettings")
    output_parent = Path(settings.output_dir)
    output_parent.mkdir(parents=True, exist_ok=True)
    campaign_dir = output_parent / settings.campaign_name
    try:
        campaign_dir.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"campaign directory already exists; choose a fresh campaign name: {campaign_dir}"
        ) from exc

    if simulation_runner is None:
        from rod_soft_contact_PI_dynamic_quasi_static_2d_sine_insert import (
            run_simulation,
        )

        simulation_runner = run_simulation
    if simulation_exporter is None:
        from sine2d_benchmark_io import export_simulation

        simulation_exporter = export_simulation

    base = SineInsertConfig(
        backend=settings.backend,
        output_dir=str(campaign_dir),
        run_name="campaign-placeholder",
    )
    thresholds = ConvergenceThresholds()
    cache: dict[tuple[int, int, float], _Case] = {}

    def run_case(nodes: int, steps: int, mu: float) -> _Case:
        key = (nodes, steps, round(mu, 12))
        if key in cache:
            return cache[key]
        name = _case_name(nodes, steps, mu)
        config = replace(
            base,
            num_nodes=nodes,
            num_load_steps=steps,
            friction_mu=mu,
            run_name=name,
        )
        result = None
        artifacts: Mapping[str, Path] = {}
        error = None
        try:
            result = simulation_runner(config)
        except Exception as exc:  # retain the other independent evidence cases
            error = f"simulation infrastructure failure: {type(exc).__name__}: {exc}"
        if error is None:
            try:
                declared = simulation_exporter(result)
                if not isinstance(declared, MappingABC):
                    raise TypeError("exporter must return a path mapping")
                artifacts = {}
                invalid = []
                declared_keys = set(declared)
                if declared_keys != REQUIRED_ARTIFACT_KEYS:
                    missing_keys = sorted(REQUIRED_ARTIFACT_KEYS - declared_keys)
                    extra_keys = sorted(declared_keys - REQUIRED_ARTIFACT_KEYS)
                    if missing_keys:
                        invalid.append("missing keys " + ", ".join(missing_keys))
                    if extra_keys:
                        invalid.append("extra keys " + ", ".join(extra_keys))
                run_dir = None
                run_dir_lexical = None
                if "run_dir" in declared:
                    try:
                        run_dir = Path(declared["run_dir"])
                    except TypeError:
                        invalid.append("run_dir is not path-like")
                    else:
                        expected_run_dir = Path(config.output_dir) / config.run_name
                        run_dir_lexical = _lexical_absolute(run_dir)
                        if (
                            not run_dir.is_dir()
                            or run_dir.is_symlink()
                            or run_dir_lexical != _lexical_absolute(expected_run_dir)
                            or run_dir.resolve() != expected_run_dir.resolve()
                        ):
                            invalid.append("run_dir is not the expected regular directory")
                        else:
                            artifacts["run_dir"] = run_dir_lexical
                for artifact_name in sorted(REQUIRED_FILE_ARTIFACT_KEYS):
                    if artifact_name not in declared:
                        continue
                    try:
                        candidate = Path(declared[artifact_name])
                    except TypeError:
                        invalid.append(f"{artifact_name} is not path-like")
                        continue
                    candidate_lexical = _lexical_absolute(candidate)
                    directly_in_run_dir = bool(
                        run_dir_lexical is not None
                        and candidate_lexical.parent == run_dir_lexical
                    )
                    resolves_in_run_dir = bool(
                        run_dir is not None
                        and candidate.resolve().parent == run_dir.resolve()
                    )
                    if (
                        candidate.is_file()
                        and not candidate.is_symlink()
                        and candidate.stat().st_size > 0
                        and directly_in_run_dir
                        and resolves_in_run_dir
                    ):
                        artifacts[artifact_name] = candidate_lexical
                    else:
                        invalid.append(
                            f"{artifact_name} is not a nonempty regular file in run_dir"
                        )
                if invalid:
                    raise OSError(
                        "export did not produce declared artifacts: " + "; ".join(invalid)
                    )
            except Exception as exc:
                error = f"artifact export failure: {type(exc).__name__}: {exc}"
        case = _Case(name, config, result, artifacts, error)
        cache[key] = case
        return case

    convergence_status = "not_requested"
    convergence_payload: dict[str, object] | None = None
    if settings.study in ("convergence", "all"):
        n400 = run_case(400, 60, NOMINAL_MU)
        n800s60 = run_case(800, 60, NOMINAL_MU)
        n1200 = run_case(1200, 60, NOMINAL_MU)
        n800s120 = run_case(800, 120, NOMINAL_MU)

        spatial_400_800 = _try_compare(
            n400, n800s60, settings, thresholds
        )
        spatial_800_1200 = _try_compare(
            n800s60, n1200, settings, thresholds
        )
        spatial_initial = bool(
            spatial_400_800 is not None
            and spatial_400_800.passed
            and spatial_800_1200 is not None
            and spatial_800_1200.passed
        )
        temporal_60_120 = _try_compare(
            n800s60, n800s120, settings, thresholds
        )
        temporal_initial = bool(
            temporal_60_120 is not None and temporal_60_120.passed
        )

        prerequisite_incomplete = any(
            case.incomplete for case in (n400, n800s60, n1200, n800s120)
        )
        spatial_refined = not spatial_initial and not prerequisite_incomplete
        temporal_refined = not temporal_initial and not prerequisite_incomplete
        spatial_final = spatial_initial
        temporal_final = temporal_initial
        spatial_1200_1600 = None
        temporal_120_240 = None
        if spatial_refined:
            n1600 = run_case(1600, 60, NOMINAL_MU)
            spatial_1200_1600 = _try_compare(
                n1200, n1600, settings, thresholds
            )
            spatial_final = bool(
                spatial_1200_1600 is not None and spatial_1200_1600.passed
            )
        if temporal_refined:
            n800s240 = run_case(800, 240, NOMINAL_MU)
            temporal_120_240 = _try_compare(
                n800s120, n800s240, settings, thresholds
            )
            temporal_final = bool(
                temporal_120_240 is not None and temporal_120_240.passed
            )
        refinement_incomplete = any(
            case.incomplete
            for case in cache.values()
            if case.config.num_nodes == 1600 or case.config.num_load_steps == 240
        )
        numerical_failure = any(
            case.status == "numerical_failure" for case in cache.values()
        )
        if prerequisite_incomplete or refinement_incomplete:
            convergence_status = "incomplete"
        elif numerical_failure:
            convergence_status = "not_converged"
        elif spatial_final and temporal_final:
            convergence_status = (
                "converged_after_refinement"
                if spatial_refined or temporal_refined
                else "converged"
            )
        else:
            convergence_status = "not_converged"
        convergence_payload = {
            "status": convergence_status,
            "thresholds": asdict(thresholds),
            "spatial": {
                "initial_passed": spatial_initial,
                "refined": spatial_refined,
                "final_passed": spatial_final,
                "n400_vs_n800": _metrics_payload(spatial_400_800),
                "n800_vs_n1200": _metrics_payload(spatial_800_1200),
                "n1200_vs_n1600": _metrics_payload(spatial_1200_1600),
            },
            "temporal": {
                "initial_passed": temporal_initial,
                "refined": temporal_refined,
                "final_passed": temporal_final,
                "steps60_vs_steps120": _metrics_payload(temporal_60_120),
                "steps120_vs_steps240": _metrics_payload(temporal_120_240),
            },
        }

    sweep_status = "not_requested"
    sweep_summary: Path | None = None
    sweep_plot: Path | None = None
    sweep_payload: dict[str, object] | None = None
    if settings.study in ("sweep", "all"):
        sweep_cases = tuple(run_case(800, 60, mu) for mu in SWEEP_MU)
        if all(case.success for case in sweep_cases):
            summary_candidate = campaign_dir / "mu_sweep_unified.tsv"
            plot_candidate = campaign_dir / "mu_sweep_insertion_force_overlay.png"
            try:
                grid, forces = _write_sweep_summary(
                    sweep_cases, summary_candidate, settings.grid_samples
                )
                sweep_summary = summary_candidate
                _write_sweep_plot(grid, forces, plot_candidate)
                sweep_plot = plot_candidate
                sweep_status = "complete"
            except (TypeError, ValueError, AttributeError, OSError) as exc:
                sweep_status = "incomplete"
                sweep_payload = {"error": f"{type(exc).__name__}: {exc}"}
                sweep_summary = (
                    sweep_summary
                    if sweep_summary is not None and sweep_summary.is_file()
                    else None
                )
                sweep_plot = (
                    sweep_plot
                    if sweep_plot is not None and sweep_plot.is_file()
                    else None
                )
        else:
            sweep_status = "incomplete"
        sweep_payload = {
            **(sweep_payload or {}),
            "status": sweep_status,
            "mu_values": list(SWEEP_MU),
            "grid_samples": settings.grid_samples,
            "summary_tsv": (
                _relative_path(sweep_summary, campaign_dir)
                if sweep_summary is not None
                else None
            ),
            "overlay_png": (
                _relative_path(sweep_plot, campaign_dir)
                if sweep_plot is not None
                else None
            ),
        }

    if convergence_status == "not_converged":
        overall_status = "not_converged"
    elif convergence_status == "incomplete":
        overall_status = "incomplete"
    elif sweep_status == "incomplete":
        overall_status = "incomplete"
    else:
        overall_status = "complete"
    if settings.study == "sweep":
        evidence_status = "not_assessed"
    elif convergence_status == "incomplete":
        evidence_status = "campaign_incomplete"
    elif settings.study == "all" and overall_status != "complete":
        evidence_status = (
            "not_converged"
            if convergence_status == "not_converged"
            else "campaign_incomplete"
        )
    else:
        evidence_status = convergence_status
    case_payload = []
    for case in cache.values():
        case_payload.append(
            {
                "name": case.name,
                "num_nodes": case.config.num_nodes,
                "num_load_steps": case.config.num_load_steps,
                "friction_mu": case.config.friction_mu,
                "backend": case.config.backend,
                "status": case.status,
                "success": case.success,
                "abort_reason": case.eligibility_reason,
                "completed_load_steps": (
                    int(getattr(case.result, "completed_load_steps", 0))
                    if case.result is not None
                    else 0
                ),
                "artifacts": {
                    name: _relative_path(Path(path), campaign_dir)
                    for name, path in case.artifacts.items()
                    if Path(path).exists()
                },
            }
        )
    manifest = campaign_dir / "campaign_manifest.json"
    payload = {
        "schema_version": 1,
        "study": settings.study,
        "backend": settings.backend,
        "overall_status": overall_status,
        "evidence_status": evidence_status,
        "claim_boundary": (
            "Numerical 2-D rigid-wall benchmark evidence only; no material "
            "calibration, experiment, device safety, or clinical validation claim."
        ),
        "cases": case_payload,
        "convergence": convergence_payload,
        "friction_sweep": sweep_payload,
    }
    _write_json_atomic(manifest, payload)
    return CampaignResult(
        campaign_dir=campaign_dir,
        manifest=manifest,
        convergence_status=convergence_status,
        sweep_status=sweep_status,
        overall_status=overall_status,
        sweep_summary=sweep_summary,
        sweep_plot=sweep_plot,
    )


def parse_args(argv: list[str] | None = None) -> CampaignSettings:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    defaults = CampaignSettings()
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--campaign-name", default=defaults.campaign_name)
    parser.add_argument("--backend", choices=("python", "cpp"), default=defaults.backend)
    parser.add_argument(
        "--study", choices=("convergence", "sweep", "all"), default=defaults.study
    )
    parser.add_argument("--grid-samples", type=int, default=defaults.grid_samples)
    values = vars(parser.parse_args(argv))
    try:
        return CampaignSettings(**values)
    except ValueError as exc:
        parser.error(str(exc))


def main(argv: list[str] | None = None) -> None:
    result = run_campaign(parse_args(argv))
    print(f"campaign_dir: {result.campaign_dir}")
    print(f"manifest: {result.manifest}")
    print(f"convergence_status: {result.convergence_status}")
    print(f"sweep_status: {result.sweep_status}")


if __name__ == "__main__":
    main()
