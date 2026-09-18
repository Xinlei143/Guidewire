import csv
import importlib
import importlib.util
import sys
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch


def _api():
    import rod_soft_contact_PI_dynamic_quasi_static_2d_sine_insert as module

    return module


def test_benchmark_output_module_exists():
    assert importlib.util.find_spec("sine2d_benchmark_io") is not None


def test_reaction_sign_unique_insertion_force_and_static_balance():
    api = _api()
    elastic_gradient = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [-2.0, 0.0]], dtype=torch.float64
    )
    zeros = torch.zeros_like(elastic_gradient)
    normal = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [-2.0, 0.0]], dtype=torch.float64
    )
    prescribed = torch.tensor(
        [[True, True], [True, True], [False, False]], dtype=torch.bool
    )

    result = api.compute_reaction_diagnostics(
        elastic_gradient, zeros, normal, zeros, zeros, prescribed
    )

    torch.testing.assert_close(
        result.actuator_force_on_wire,
        torch.tensor([2.0, 0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.proximal_reaction, -result.actuator_force_on_wire
    )
    assert result.insertion_force == pytest.approx(2.0)
    assert result.insertion_force == pytest.approx(
        -result.proximal_reaction[0].item()
    )
    assert result.insertion_force != pytest.approx(
        (result.actuator_force_on_wire + result.wall_resultant)[0].item()
    )
    assert result.static_balance_abs == pytest.approx(0.0, abs=1.0e-15)
    assert result.static_balance_rel == pytest.approx(0.0, abs=1.0e-15)
    assert torch.linalg.vector_norm(result.internal_elastic_resultant).item() < 1.0e-15


def test_dynamic_remainder_is_separate_and_full_balance_accounts_for_it():
    api = _api()
    elastic_gradient = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [-2.0, 0.0]], dtype=torch.float64
    )
    dynamic = torch.tensor(
        [[0.2, 0.1], [0.3, -0.1], [0.1, 0.0]], dtype=torch.float64
    )
    normal = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [-1.9, 0.0]], dtype=torch.float64
    )
    zeros = torch.zeros_like(elastic_gradient)
    prescribed = torch.tensor(
        [[True, True], [True, True], [False, False]], dtype=torch.bool
    )

    result = api.compute_reaction_diagnostics(
        elastic_gradient, dynamic, normal, zeros, zeros, prescribed
    )

    torch.testing.assert_close(
        result.dynamic_proximal_remainder,
        torch.tensor([0.5, 0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.actuator_force_on_wire,
        torch.tensor([2.0, 0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.proximal_reaction, -result.actuator_force_on_wire
    )
    torch.testing.assert_close(
        result.dynamic_actuator_force_on_wire,
        result.actuator_force_on_wire + result.dynamic_proximal_remainder,
    )
    torch.testing.assert_close(
        result.global_dynamic_resultant,
        torch.tensor([0.6, 0.0], dtype=torch.float64),
    )
    assert result.insertion_force == pytest.approx(2.0)
    assert result.insertion_force == result.actuator_force_on_wire[0].item()
    assert result.insertion_force == -result.proximal_reaction[0].item()
    torch.testing.assert_close(
        result.static_imbalance,
        torch.tensor([0.1, 0.0], dtype=torch.float64),
    )
    assert result.static_balance_rel == pytest.approx(0.05)
    assert result.full_balance_abs == pytest.approx(0.0, abs=1.0e-15)
    assert result.full_balance_rel == pytest.approx(0.0, abs=1.0e-15)


def test_feeder_resultant_has_exact_zero_x_component():
    api = _api()
    zeros = torch.zeros((3, 2), dtype=torch.float64)
    feeder = torch.tensor(
        [[0.0, 2.0], [0.0, -0.5], [0.0, -0.25]], dtype=torch.float64
    )
    prescribed = torch.tensor(
        [[True, True], [True, True], [False, False]], dtype=torch.bool
    )

    result = api.compute_reaction_diagnostics(
        zeros, zeros, zeros, zeros, feeder, prescribed
    )

    assert result.feeder_resultant[0].item() == 0.0


def test_step_retains_dynamic_term_and_record_computes_elastic_reaction():
    api = _api()
    config = api.SineInsertConfig(
        num_nodes=12,
        wire_length=0.30,
        initial_inserted_length=0.05,
        total_insertion=2.0e-4,
        num_load_steps=1,
        young_modulus=2.0e6,
        contact_kp_line=2.0e3,
        contact_ki_line=0.0,
        feeder_kp_line=2.0e3,
        friction_kt_line=2.0e3,
        dt=0.02,
        max_newton_iters=10,
    )
    tube = api.SineTube2D(
        config.amplitude, config.x_span, config.tube_half_width
    )
    initial = api.initialize_rod(config, tube)
    positions = torch.as_tensor(initial.positions, dtype=torch.float64)
    velocities = torch.as_tensor(initial.velocities, dtype=torch.float64)
    rest = torch.as_tensor(initial.rest_lengths, dtype=torch.float64)
    reference = torch.as_tensor(initial.reference_curvature, dtype=torch.float64)
    masses = torch.as_tensor(initial.masses, dtype=torch.float64)
    nodal = torch.as_tensor(initial.nodal_lengths, dtype=torch.float64)
    integrals = torch.as_tensor(initial.integrals, dtype=torch.float64)
    errors = torch.as_tensor(initial.previous_errors, dtype=torch.float64)
    frame = api.freeze_contact_frames(positions, tube)
    contact = api.contact_quantities(
        positions, integrals, errors, nodal, frame, config, config.dt
    )
    friction = api.initialize_friction_history(positions, frame, contact.active)
    context = api.begin_friction_increment(friction, positions)
    prescription = api.proximal_prescribed_state(
        initial.positions, 1.0e-5, config.amplitude
    )
    mask = torch.as_tensor(prescription.mask, dtype=torch.bool)
    prescribed = torch.as_tensor(prescription.positions, dtype=torch.float64)

    step = api.implicit_euler_step(
        positions,
        velocities,
        rest,
        reference,
        masses,
        integrals,
        errors,
        nodal,
        mask,
        prescribed,
        tube,
        config,
        context,
        friction_enabled=True,
        accepted_frame=frame,
    )

    area_moment = config.young_modulus * np.pi * config.diameter**4 / 64.0
    damping = 2.0 * config.damping_ratio * np.sqrt(
        (3.0 * area_moment / config.wire_length**3) * masses.sum().item()
    )
    expected_dynamic = (
        masses[:, None]
        * (step.positions - positions - config.dt * velocities)
        / config.dt**2
        + damping * (step.positions - positions) / config.dt
    )
    torch.testing.assert_close(step.dynamic_force, expected_dynamic)

    state = api.SimulationState(
        step.positions,
        step.velocities,
        step.integrals,
        step.previous_errors,
        friction,
        step.frame,
        config.dt,
        1.0e-5,
    )
    record = api._record(
        state, step, rest, reference, config, tube, 1, "settled", True
    )

    assert record.reaction.insertion_force == pytest.approx(
        record.reaction.actuator_force_on_wire[0].item()
    )
    torch.testing.assert_close(
        record.reaction.proximal_reaction,
        -record.reaction.actuator_force_on_wire,
    )
    assert (
        torch.linalg.vector_norm(record.reaction.internal_elastic_resultant).item()
        < 1.0e-8
    )


@pytest.fixture(scope="module")
def exported_run(tmp_path_factory):
    api = _api()
    output_dir = tmp_path_factory.mktemp("sine2d-output")
    config = api.SineInsertConfig(
        backend="python",
        num_nodes=12,
        wire_length=0.30,
        initial_inserted_length=0.05,
        total_insertion=2.0e-4,
        num_load_steps=1,
        young_modulus=2.0e6,
        contact_kp_line=2.0e3,
        contact_ki_line=0.0,
        feeder_kp_line=2.0e3,
        friction_mu=0.0,
        friction_kt_line=2.0e3,
        dt=0.02,
        ramp_steps=2,
        max_newton_iters=10,
        residual_tol=1.0e-8,
        step_tol=1.0e-11,
        settle_speed_tol=1.0,
        settle_residual_tol=1.0e-6,
        contact_tolerance=2.0e-3,
        settle_confirm_steps=1,
        max_settle_steps=30,
        prebalance=True,
        prebalance_max_steps=30,
        history_save_interval=5,
        retry_subdivisions=(),
        output_dir=str(output_dir),
        run_name="tiny-success",
    )
    result = api.run_simulation(config)
    assert result.success
    output_api = importlib.import_module("sine2d_benchmark_io")
    manifest = output_api.export_simulation(result)
    return result, manifest


def test_export_writes_config_with_dataclass_and_derived_geometry(exported_run):
    result, manifest = exported_run
    config_path = Path(manifest["config"])

    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert payload["run_name"] == result.config.run_name
    assert payload["retry_subdivisions"] == []
    tube = result.config and _api().SineTube2D(
        result.config.amplitude,
        result.config.x_span,
        result.config.tube_half_width,
    )
    assert payload["vessel_length"] == pytest.approx(tube.vessel_length)
    assert payload["Rmin"] == pytest.approx(tube.min_radius)


def test_settled_tsv_has_one_data_row_per_settled_record(exported_run):
    result, manifest = exported_run
    with Path(manifest["settled"]).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))

    required = {
        "insertion",
        "insertion_force",
        "actuator_force_on_wire_x",
        "actuator_force_on_wire_y",
        "proximal_reaction_x",
        "proximal_reaction_y",
        "dynamic_actuator_force_on_wire_x",
        "dynamic_actuator_force_on_wire_y",
        "normal_resultant_x",
        "friction_resultant_x",
        "feeder_resultant_x",
        "static_balance_abs",
        "static_balance_rel",
        "dynamic_proximal_remainder_x",
        "tip_x",
        "tip_y",
        "elastic_stretch",
        "elastic_bend",
        "elastic_total",
        "max_contact_error",
        "newton_residual_norm",
        "settled",
    }
    assert required <= set(rows[0])
    assert len(rows) == len(result.settled_records)
    assert all(row["settled"] == "True" for row in rows)


def test_history_npz_contains_required_arrays_with_record_shapes(exported_run):
    result, manifest = exported_run
    with np.load(manifest["history"], allow_pickle=False) as data:
        required = {
            "time",
            "phase",
            "load_step",
            "insertion",
            "positions",
            "velocities",
            "normal_force",
            "friction_force",
            "feeder_force",
            "pi_integrals",
            "friction_xi",
            "contact_active",
            "tip_position",
            "tip_s",
            "elastic_stretch",
            "elastic_bend",
            "elastic_total",
            "newton_converged",
            "newton_iterations",
            "newton_residual_norm",
            "newton_step_norm",
            "newton_line_search_failures",
            "newton_dense_fallback_count",
            "contact_error",
            "max_contact_error",
            "settled",
            "insertion_force",
            "actuator_force_on_wire",
            "proximal_reaction",
            "dynamic_actuator_force_on_wire",
            "normal_resultant",
            "friction_resultant",
            "feeder_resultant",
            "wall_resultant",
            "dynamic_proximal_remainder",
            "global_dynamic_resultant",
            "static_imbalance",
            "static_balance_abs",
            "static_balance_rel",
            "full_balance",
            "full_balance_abs",
            "full_balance_rel",
            "newton_dense_fallback_count",
        }
        assert required <= set(data.files)
        record_count = len(result.history)
        node_count = result.config.num_nodes
        assert data["positions"].shape == (record_count, node_count, 2)
        assert data["pi_integrals"].shape == (record_count, node_count)
        assert data["contact_error"].shape == (record_count, node_count)
        assert data["actuator_force_on_wire"].shape == (record_count, 2)
        assert data["insertion_force"].shape == (record_count,)
        assert not any("contact_energy" in key for key in data.files)


def test_export_writes_four_nonempty_pngs_without_contact_energy_claim(exported_run):
    _, manifest = exported_run
    expected = {
        "configurations",
        "insertion_force_plot",
        "normal_friction_plot",
        "elastic_energy_plot",
    }
    assert expected <= set(manifest)
    for key in expected:
        path = Path(manifest[key])
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
    assert "contact_energy" not in Path(manifest["config"]).read_text(encoding="utf-8")
    assert "contact_energy" not in Path(manifest["settled"]).read_text(encoding="utf-8")


def test_export_replaces_only_known_files_in_its_run_directory(exported_run):
    result, manifest = exported_run
    run_dir = Path(manifest["run_dir"])
    unknown = run_dir / "keep-me.txt"
    unknown.write_text("preserve", encoding="utf-8")
    sibling = run_dir.parent / "sibling-run"
    sibling.mkdir()
    sibling_marker = sibling / "keep-me-too.txt"
    sibling_marker.write_text("preserve", encoding="utf-8")

    repeated = importlib.import_module("sine2d_benchmark_io").export_simulation(result)

    assert repeated == manifest
    assert unknown.read_text(encoding="utf-8") == "preserve"
    assert sibling_marker.read_text(encoding="utf-8") == "preserve"


def test_export_accepts_result_from_direct_script_module_identity(exported_run):
    """Direct ``python script.py`` must not fail on duplicate class identity."""
    result, _ = exported_run
    script = Path(__file__).parents[1] / (
        "rod_soft_contact_PI_dynamic_quasi_static_2d_sine_insert.py"
    )
    spec = importlib.util.spec_from_file_location("_sine2d_direct_script", script)
    assert spec is not None and spec.loader is not None
    direct = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = direct
    try:
        spec.loader.exec_module(direct)
    finally:
        sys.modules.pop(spec.name, None)
    duplicate = direct.SimulationResult(
        config=result.config,
        success=result.success,
        abort_reason=result.abort_reason,
        completed_load_steps=result.completed_load_steps,
        history=result.history,
        final_state=result.final_state,
        prebalanced_state=result.prebalanced_state,
        prebalance_history_length=result.prebalance_history_length,
    )

    manifest = importlib.import_module("sine2d_benchmark_io").export_simulation(
        duplicate
    )

    assert Path(manifest["config"]).is_file()


def test_export_rejects_nonzero_feeder_x_and_non_quasistatic_settled_state(
    exported_run,
):
    result, _ = exported_run
    record = result.settled_records[-1]

    bad_feeder_reaction = replace(
        record.reaction,
        feeder_resultant=torch.tensor([1.0e-12, 0.0], dtype=torch.float64),
    )
    bad_feeder = replace(
        result,
        history=(replace(record, reaction=bad_feeder_reaction),),
    )
    with pytest.raises(RuntimeError, match="feeder x-force"):
        importlib.import_module("sine2d_benchmark_io").export_simulation(bad_feeder)

    bad_dynamic_reaction = replace(
        record.reaction,
        dynamic_proximal_remainder=torch.tensor([1.0, 0.0], dtype=torch.float64),
    )
    bad_dynamic = replace(
        result,
        history=(replace(record, reaction=bad_dynamic_reaction),),
    )
    with pytest.raises(RuntimeError, match="quasi-static remainder"):
        importlib.import_module("sine2d_benchmark_io").export_simulation(bad_dynamic)

    nonfinite_reaction = replace(
        record.reaction,
        dynamic_proximal_remainder=torch.tensor([float("nan"), 0.0], dtype=torch.float64),
    )
    nonfinite = replace(
        result,
        history=(replace(record, reaction=nonfinite_reaction),),
    )
    with pytest.raises(RuntimeError, match="must be finite"):
        importlib.import_module("sine2d_benchmark_io").export_simulation(nonfinite)


@pytest.mark.parametrize("run_name", [".", "..", "../escape", "nested/run", "   "])
def test_config_rejects_unsafe_run_names(run_name):
    with pytest.raises(ValueError, match="run_name"):
        _api().SineInsertConfig(run_name=run_name)


def test_main_runs_exports_and_prints_concise_artifact_paths(monkeypatch, capsys):
    api = _api()
    sentinel = object()
    calls = []

    def fake_run(config):
        calls.append(config)
        return sentinel

    def fake_export(result):
        assert result is sentinel
        return {
            "run_dir": Path("out/case"),
            "config": Path("out/case/config.json"),
            "settled": Path("out/case/settled.tsv"),
            "history": Path("out/case/history.npz"),
            "configurations": Path("out/case/configurations.png"),
            "insertion_force_plot": Path("out/case/insertion_force_vs_insertion.png"),
            "normal_friction_plot": Path("out/case/normal_friction_resultants_vs_insertion.png"),
            "elastic_energy_plot": Path("out/case/elastic_energy_vs_insertion.png"),
        }

    monkeypatch.setattr(api, "run_simulation", fake_run)
    monkeypatch.setattr(
        importlib.import_module("sine2d_benchmark_io"),
        "export_simulation",
        fake_export,
    )

    api.main(["--output-dir", "out", "--run-name", "case"])

    output = capsys.readouterr().out
    assert len(calls) == 1
    assert calls[0].output_dir == "out"
    assert "run_dir: out/case" in output
    assert "config: out/case/config.json" in output
    assert "history: out/case/history.npz" in output
    assert "SineInsertConfig(" not in output
