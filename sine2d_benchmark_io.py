"""Artifact export for the two-dimensional sine-tube insertion benchmark."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sine_tube_2d import SineTube2D

if TYPE_CHECKING:
    from rod_soft_contact_PI_dynamic_quasi_static_2d_sine_insert import (
        HistoryRecord,
        SimulationResult,
    )


_TSV_FIELDS = (
    "time",
    "phase",
    "load_step",
    "insertion",
    "insertion_force",
    "actuator_force_on_wire_x",
    "actuator_force_on_wire_y",
    "proximal_reaction_x",
    "proximal_reaction_y",
    "dynamic_actuator_force_on_wire_x",
    "dynamic_actuator_force_on_wire_y",
    "normal_resultant_x",
    "normal_resultant_y",
    "friction_resultant_x",
    "friction_resultant_y",
    "feeder_resultant_x",
    "feeder_resultant_y",
    "wall_resultant_x",
    "wall_resultant_y",
    "static_imbalance_x",
    "static_imbalance_y",
    "static_balance_abs",
    "static_balance_rel",
    "dynamic_proximal_remainder_x",
    "dynamic_proximal_remainder_y",
    "global_dynamic_resultant_x",
    "global_dynamic_resultant_y",
    "full_balance_x",
    "full_balance_y",
    "full_balance_abs",
    "full_balance_rel",
    "tip_x",
    "tip_y",
    "tip_s",
    "elastic_stretch",
    "elastic_bend",
    "elastic_total",
    "active_contact_count",
    "max_contact_error",
    "newton_converged",
    "newton_iterations",
    "newton_residual_norm",
    "newton_step_norm",
    "newton_line_search_failures",
    "newton_dense_fallback_count",
    "settled",
)

_QUASI_STATIC_REMAINDER_REL_TOL = 0.05


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _record_row(record: "HistoryRecord") -> dict[str, object]:
    reaction = record.reaction
    vector_fields = {
        "actuator_force_on_wire": reaction.actuator_force_on_wire,
        "proximal_reaction": reaction.proximal_reaction,
        "dynamic_actuator_force_on_wire": reaction.dynamic_actuator_force_on_wire,
        "normal_resultant": reaction.normal_resultant,
        "friction_resultant": reaction.friction_resultant,
        "feeder_resultant": reaction.feeder_resultant,
        "wall_resultant": reaction.wall_resultant,
        "static_imbalance": reaction.static_imbalance,
        "dynamic_proximal_remainder": reaction.dynamic_proximal_remainder,
        "global_dynamic_resultant": reaction.global_dynamic_resultant,
        "full_balance": reaction.full_balance,
    }
    row: dict[str, object] = {
        "time": record.time,
        "phase": record.phase,
        "load_step": record.load_step,
        "insertion": record.insertion,
        "insertion_force": reaction.insertion_force,
        "static_balance_abs": reaction.static_balance_abs,
        "static_balance_rel": reaction.static_balance_rel,
        "full_balance_abs": reaction.full_balance_abs,
        "full_balance_rel": reaction.full_balance_rel,
        "tip_x": float(record.tip_position[0].item()),
        "tip_y": float(record.tip_position[1].item()),
        "tip_s": record.tip_s,
        "elastic_stretch": record.elastic_stretch,
        "elastic_bend": record.elastic_bend,
        "elastic_total": record.elastic_total,
        "active_contact_count": int(record.active.sum().item()),
        "max_contact_error": record.max_contact_error,
        "newton_converged": record.newton.converged,
        "newton_iterations": record.newton.iterations,
        "newton_residual_norm": record.newton.residual_norm,
        "newton_step_norm": record.newton.step_norm,
        "newton_line_search_failures": record.newton.line_search_failures,
        "newton_dense_fallback_count": record.newton.dense_fallback_count,
        "settled": record.settled,
    }
    for name, value in vector_fields.items():
        row[f"{name}_x"] = float(value[0].item())
        row[f"{name}_y"] = float(value[1].item())
    return row


def _write_config(result: "SimulationResult", path: Path) -> None:
    tube = SineTube2D(
        result.config.amplitude,
        result.config.x_span,
        result.config.tube_half_width,
    )
    payload = asdict(result.config)
    payload["vessel_length"] = tube.vessel_length
    payload["Rmin"] = tube.min_radius
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_settled_tsv(result: "SimulationResult", path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_TSV_FIELDS, delimiter="\t")
        writer.writeheader()
        for record in result.settled_records:
            writer.writerow(_record_row(record))


def _stack_records(
    records: tuple["HistoryRecord", ...],
    getter,
    empty_shape: tuple[int, ...],
    dtype=np.float64,
) -> np.ndarray:
    if not records:
        return np.empty((0, *empty_shape), dtype=dtype)
    return np.stack(
        [np.asarray(getter(record), dtype=dtype) for record in records], axis=0
    )


def _write_history_npz(result: "SimulationResult", path: Path) -> None:
    records = result.history
    count = len(records)
    nodes = result.config.num_nodes
    scalar = lambda getter, dtype=np.float64: np.asarray(
        [getter(record) for record in records], dtype=dtype
    )
    vector = lambda getter: _stack_records(records, getter, (2,))
    nodal_vector = lambda getter: _stack_records(records, getter, (nodes, 2))
    nodal_scalar = lambda getter, dtype=np.float64: _stack_records(
        records, getter, (nodes,), dtype=dtype
    )
    np.savez(
        path,
        time=scalar(lambda record: record.time),
        phase=scalar(lambda record: record.phase, dtype=np.str_),
        load_step=scalar(lambda record: record.load_step, dtype=np.int64),
        insertion=scalar(lambda record: record.insertion),
        positions=nodal_vector(lambda record: record.positions.detach().numpy()),
        velocities=nodal_vector(lambda record: record.velocities.detach().numpy()),
        normal_force=nodal_vector(lambda record: record.normal_force.detach().numpy()),
        friction_force=nodal_vector(lambda record: record.friction_force.detach().numpy()),
        feeder_force=nodal_vector(lambda record: record.feeder_force.detach().numpy()),
        pi_integrals=nodal_scalar(lambda record: record.integrals.detach().numpy()),
        friction_xi=nodal_scalar(lambda record: record.xi.detach().numpy()),
        contact_active=nodal_scalar(
            lambda record: record.active.detach().numpy(), dtype=np.bool_
        ),
        contact_error=nodal_scalar(
            lambda record: record.contact_error.detach().numpy()
        ),
        tip_position=vector(lambda record: record.tip_position.detach().numpy()),
        tip_s=scalar(lambda record: record.tip_s),
        elastic_stretch=scalar(lambda record: record.elastic_stretch),
        elastic_bend=scalar(lambda record: record.elastic_bend),
        elastic_total=scalar(lambda record: record.elastic_total),
        newton_converged=scalar(
            lambda record: record.newton.converged, dtype=np.bool_
        ),
        newton_iterations=scalar(
            lambda record: record.newton.iterations, dtype=np.int64
        ),
        newton_residual_norm=scalar(lambda record: record.newton.residual_norm),
        newton_step_norm=scalar(lambda record: record.newton.step_norm),
        newton_line_search_failures=scalar(
            lambda record: record.newton.line_search_failures, dtype=np.int64
        ),
        newton_dense_fallback_count=scalar(
            lambda record: record.newton.dense_fallback_count, dtype=np.int64
        ),
        max_contact_error=scalar(lambda record: record.max_contact_error),
        settled=scalar(lambda record: record.settled, dtype=np.bool_),
        insertion_force=scalar(lambda record: record.reaction.insertion_force),
        actuator_force_on_wire=vector(
            lambda record: record.reaction.actuator_force_on_wire.detach().numpy()
        ),
        proximal_reaction=vector(
            lambda record: record.reaction.proximal_reaction.detach().numpy()
        ),
        dynamic_actuator_force_on_wire=vector(
            lambda record: record.reaction.dynamic_actuator_force_on_wire.detach().numpy()
        ),
        normal_resultant=vector(
            lambda record: record.reaction.normal_resultant.detach().numpy()
        ),
        friction_resultant=vector(
            lambda record: record.reaction.friction_resultant.detach().numpy()
        ),
        feeder_resultant=vector(
            lambda record: record.reaction.feeder_resultant.detach().numpy()
        ),
        wall_resultant=vector(
            lambda record: record.reaction.wall_resultant.detach().numpy()
        ),
        dynamic_proximal_remainder=vector(
            lambda record: record.reaction.dynamic_proximal_remainder.detach().numpy()
        ),
        global_dynamic_resultant=vector(
            lambda record: record.reaction.global_dynamic_resultant.detach().numpy()
        ),
        static_imbalance=vector(
            lambda record: record.reaction.static_imbalance.detach().numpy()
        ),
        static_balance_abs=scalar(lambda record: record.reaction.static_balance_abs),
        static_balance_rel=scalar(lambda record: record.reaction.static_balance_rel),
        full_balance=vector(
            lambda record: record.reaction.full_balance.detach().numpy()
        ),
        full_balance_abs=scalar(lambda record: record.reaction.full_balance_abs),
        full_balance_rel=scalar(lambda record: record.reaction.full_balance_rel),
    )
    if count == 0:
        # Document the intended first dimension while retaining ordinary NPZ arrays.
        assert np.load(path, allow_pickle=False)["time"].shape == (0,)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_configurations(result: "SimulationResult", path: Path) -> None:
    config = result.config
    tube = SineTube2D(config.amplitude, config.x_span, config.tube_half_width)
    q = np.linspace(0.0, config.x_span, 500)
    center = tube.centerline(q)
    normal = tube.normal(q)
    upper = center + config.tube_half_width * normal
    lower = center - config.tube_half_width * normal
    fig, axis = plt.subplots(figsize=(10.0, 4.5))
    axis.plot(center[:, 0], center[:, 1], "k--", linewidth=1.0, label="centerline")
    axis.plot(upper[:, 0], upper[:, 1], color="0.35", linewidth=1.4, label="walls")
    axis.plot(lower[:, 0], lower[:, 1], color="0.35", linewidth=1.4)
    minimum_x = min(float(result.final_state.positions[:, 0].min().item()), -0.05)
    axis.plot(
        [minimum_x, 0.0],
        [config.amplitude + config.feeder_half_width] * 2,
        color="0.55",
        linewidth=1.2,
        label="feeder",
    )
    axis.plot(
        [minimum_x, 0.0],
        [config.amplitude - config.feeder_half_width] * 2,
        color="0.55",
        linewidth=1.2,
    )
    records = result.history
    if records:
        indices = np.unique(np.linspace(0, len(records) - 1, min(6, len(records)), dtype=int))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(indices)))
        for color, index in zip(colors, indices):
            record = records[int(index)]
            positions = record.positions.detach().numpy()
            axis.plot(
                positions[:, 0], positions[:, 1], color=color, linewidth=1.2,
                label=f"rod {record.insertion:.4g} m",
            )
            active = record.active.detach().numpy()
            if np.any(active):
                axis.scatter(
                    positions[active, 0], positions[active, 1], s=14,
                    facecolors="none", edgecolors=[color], linewidths=0.8,
                )
    else:
        positions = result.final_state.positions.detach().numpy()
        axis.plot(positions[:, 0], positions[:, 1], linewidth=1.2, label="rod")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=7, ncol=2)
    _save_figure(fig, path)


def _settled_series(result: "SimulationResult") -> tuple["HistoryRecord", ...]:
    return result.settled_records


def _plot_insertion_force(result: "SimulationResult", path: Path) -> None:
    records = _settled_series(result)
    insertion = [record.insertion for record in records]
    force = [record.reaction.insertion_force for record in records]
    fig, axis = plt.subplots(figsize=(6.5, 4.0))
    axis.plot(insertion, force, "o-", linewidth=1.4, markersize=4)
    axis.set_xlabel("insertion (m)")
    axis.set_ylabel("static actuator force on wire, x (N)")
    axis.grid(alpha=0.25)
    _save_figure(fig, path)


def _plot_wall_resultants(result: "SimulationResult", path: Path) -> None:
    records = _settled_series(result)
    insertion = np.asarray([record.insertion for record in records])
    normal = np.asarray(
        [record.reaction.normal_resultant.detach().numpy() for record in records]
    ).reshape(-1, 2)
    friction = np.asarray(
        [record.reaction.friction_resultant.detach().numpy() for record in records]
    ).reshape(-1, 2)
    fig, axis = plt.subplots(figsize=(6.5, 4.0))
    for values, label, style in (
        (normal[:, 0], "normal x", "-"),
        (normal[:, 1], "normal y", "--"),
        (friction[:, 0], "friction x", "-"),
        (friction[:, 1], "friction y", "--"),
    ):
        axis.plot(insertion, values, marker="o", linestyle=style, label=label)
    axis.set_xlabel("insertion (m)")
    axis.set_ylabel("wall-on-wire resultant (N)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    _save_figure(fig, path)


def _plot_elastic_energy(result: "SimulationResult", path: Path) -> None:
    records = _settled_series(result)
    insertion = [record.insertion for record in records]
    fig, axis = plt.subplots(figsize=(6.5, 4.0))
    axis.plot(insertion, [record.elastic_stretch for record in records], "o-", label="stretch")
    axis.plot(insertion, [record.elastic_bend for record in records], "o-", label="bend")
    axis.plot(insertion, [record.elastic_total for record in records], "o-", label="total elastic")
    axis.set_xlabel("insertion (m)")
    axis.set_ylabel("elastic energy (J)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    _save_figure(fig, path)


def export_simulation(result: "SimulationResult") -> dict[str, Path]:
    """Write one run's deterministic benchmark artifacts and return their paths."""
    # Do not rely on class identity here: when the benchmark entrypoint is run
    # directly its dataclasses belong to ``__main__``, while importing this IO
    # module also loads the same source under its normal module name.
    required = ("config", "history", "settled_records", "final_state")
    if not all(hasattr(result, name) for name in required):
        raise TypeError("result must provide the SimulationResult interface")
    for record in result.settled_records:
        reaction = record.reaction
        feeder_x = float(reaction.feeder_resultant[0].item())
        if feeder_x != 0.0:
            raise RuntimeError(
                f"settled feeder x-force must be exactly zero, got {feeder_x:.17g} N"
            )
        actuator_norm = float(
            np.linalg.norm(reaction.actuator_force_on_wire.detach().numpy())
        )
        remainder_norm = float(
            np.linalg.norm(reaction.dynamic_proximal_remainder.detach().numpy())
        )
        remainder_rel = remainder_norm / max(actuator_norm, 1.0e-12)
        if not all(math.isfinite(value) for value in (
            feeder_x, actuator_norm, remainder_norm, remainder_rel,
        )):
            raise RuntimeError("settled force diagnostics must be finite")
        if remainder_rel > _QUASI_STATIC_REMAINDER_REL_TOL:
            raise RuntimeError(
                "settled quasi-static remainder is not significantly smaller "
                f"than the proximal actuator force: relative={remainder_rel:.9g}, "
                f"limit={_QUASI_STATIC_REMAINDER_REL_TOL:.9g}"
            )
    run_dir = Path(result.config.output_dir) / result.config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_dir": run_dir,
        "config": run_dir / "config.json",
        "settled": run_dir / "settled.tsv",
        "history": run_dir / "history.npz",
        "configurations": run_dir / "configurations.png",
        "insertion_force_plot": run_dir / "insertion_force_vs_insertion.png",
        "normal_friction_plot": run_dir / "normal_friction_resultants_vs_insertion.png",
        "elastic_energy_plot": run_dir / "elastic_energy_vs_insertion.png",
    }
    _write_config(result, manifest["config"])
    _write_settled_tsv(result, manifest["settled"])
    _write_history_npz(result, manifest["history"])
    _plot_configurations(result, manifest["configurations"])
    _plot_insertion_force(result, manifest["insertion_force_plot"])
    _plot_wall_resultants(result, manifest["normal_friction_plot"])
    _plot_elastic_energy(result, manifest["elastic_energy_plot"])
    return manifest
