import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _record(insertion, force, tip_s, penetration):
    return SimpleNamespace(
        settled=True,
        insertion=float(insertion),
        reaction=SimpleNamespace(insertion_force=float(force)),
        tip_s=float(tip_s),
        max_contact_error=float(penetration),
    )


def _result(config, force_scale=1.0):
    insertion = np.linspace(0.0, config.total_insertion, config.num_load_steps + 1)
    records = tuple(
        _record(x, force_scale * (1.0 + x), 0.12 + x, 2.0e-4 + 1.0e-5 * x)
        for x in insertion
    )
    return SimpleNamespace(
        config=config,
        success=True,
        abort_reason=None,
        completed_load_steps=config.num_load_steps,
        settled_records=records,
    )


def _fake_exporter(result):
    run_dir = Path(result.config.output_dir) / result.config.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    names = {
        "config": "config.json",
        "settled": "settled.tsv",
        "history": "history.npz",
        "configurations": "configurations.png",
        "insertion_force_plot": "insertion_force_vs_insertion.png",
        "normal_friction_plot": "normal_friction_resultants_vs_insertion.png",
        "elastic_energy_plot": "elastic_energy_vs_insertion.png",
    }
    artifacts = {"run_dir": run_dir}
    for key, filename in names.items():
        path = run_dir / filename
        path.write_bytes(b"synthetic artifact\n")
        artifacts[key] = path
    return artifacts


def test_comparison_uses_one_grid_and_all_four_convergence_gates():
    import sine2d_benchmark_campaign as campaign

    reference = campaign.SettledCurve(
        insertion=np.array([0.0, 0.5, 1.0]),
        insertion_force=np.array([1.0, 2.0, 3.0]),
        tip_s=np.array([0.12, 0.62, 1.12]),
        max_penetration=np.array([1.0e-4, 2.0e-4, 3.0e-4]),
    )
    candidate = campaign.SettledCurve(
        insertion=np.array([0.0, 0.25, 0.75, 1.0]),
        insertion_force=np.array([1.02, 1.53, 2.55, 3.06]),
        tip_s=np.array([0.12, 0.37, 0.87, 1.122]),
        max_penetration=np.array([1.0e-4, 1.5e-4, 2.5e-4, 3.5e-4]),
    )

    metrics = campaign.compare_curves(candidate, reference, samples=21)

    assert metrics.force_relative_l2 == pytest.approx(0.02)
    assert metrics.force_peak_relative == pytest.approx(0.02)
    assert metrics.force_final_relative == pytest.approx(0.02)
    assert metrics.tip_s_difference_m == pytest.approx(0.002)
    assert metrics.max_penetration_difference_m == pytest.approx(5.0e-5)
    assert metrics.passed
    assert len(metrics.insertion_grid) == 21

    failed = campaign.compare_curves(
        campaign.SettledCurve(
            insertion=candidate.insertion,
            insertion_force=candidate.insertion_force,
            tip_s=candidate.tip_s + 0.006,
            max_penetration=candidate.max_penetration,
        ),
        reference,
        samples=21,
    )
    assert not failed.passed


def test_comparison_grid_includes_source_knots_and_cannot_miss_interior_peak():
    import sine2d_benchmark_campaign as campaign

    candidate = campaign.SettledCurve(
        insertion=np.array([0.0, 0.5, 1.0]),
        insertion_force=np.array([0.0, 10.0, 0.0]),
        tip_s=np.array([0.0, 0.5, 1.0]),
        max_penetration=np.zeros(3),
    )
    reference = campaign.SettledCurve(
        insertion=np.array([0.0, 1.0]),
        insertion_force=np.array([0.0, 0.0]),
        tip_s=np.array([0.0, 1.0]),
        max_penetration=np.zeros(2),
    )

    metrics = campaign.compare_curves(candidate, reference, samples=2)

    np.testing.assert_array_equal(metrics.insertion_grid, [0.0, 0.5, 1.0])
    assert metrics.force_peak_relative > 1.0e12
    assert not metrics.passed


def test_campaign_rejects_an_existing_directory_before_running(tmp_path):
    import sine2d_benchmark_campaign as campaign

    root = tmp_path / "already-there"
    root.mkdir()
    settings = campaign.CampaignSettings(
        output_dir=str(tmp_path), campaign_name=root.name, study="sweep"
    )

    with pytest.raises(FileExistsError, match="fresh campaign name"):
        campaign.run_campaign(settings, simulation_runner=lambda _: None)


def test_all_campaign_reuses_nominal_case_and_writes_sweep_products(tmp_path):
    import sine2d_benchmark_campaign as campaign

    seen = []

    def runner(config):
        seen.append((config.num_nodes, config.num_load_steps, config.friction_mu))
        return _result(config)

    settings = campaign.CampaignSettings(
        output_dir=str(tmp_path), campaign_name="all-pass", study="all", grid_samples=17
    )
    result = campaign.run_campaign(
        settings, simulation_runner=runner, simulation_exporter=_fake_exporter
    )

    # 400/800/1200 at 60, 800 at 120, plus three non-nominal mu values.
    assert len(seen) == 7
    assert seen.count((800, 60, 0.10)) == 1
    assert result.convergence_status == "converged"
    assert result.sweep_status == "complete"
    assert result.overall_status == "complete"
    assert result.sweep_summary.is_file()
    assert result.sweep_plot.is_file()
    assert result.manifest.is_file()

    with result.sweep_summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) > 17
    assert list(rows[0]) == [
        "insertion_m",
        "insertion_force_mu_0p00_N",
        "insertion_force_mu_0p05_N",
        "insertion_force_mu_0p10_N",
        "insertion_force_mu_0p20_N",
    ]
    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert payload["evidence_status"] == "converged"
    assert len(payload["cases"]) == 7


def test_only_failed_convergence_dimension_is_refined(tmp_path):
    import sine2d_benchmark_campaign as campaign

    seen = []

    def runner(config):
        seen.append((config.num_nodes, config.num_load_steps, config.friction_mu))
        # Break only the 1200-node endpoint of the spatial comparison.
        scale = 1.20 if config.num_nodes == 1200 else 1.0
        return _result(config, force_scale=scale)

    settings = campaign.CampaignSettings(
        output_dir=str(tmp_path),
        campaign_name="spatial-refine",
        study="convergence",
        grid_samples=11,
    )
    result = campaign.run_campaign(
        settings, simulation_runner=runner, simulation_exporter=_fake_exporter
    )

    assert (1600, 60, 0.10) in seen
    assert (800, 240, 0.10) not in seen
    assert result.convergence_status == "not_converged"
    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert payload["convergence"]["spatial"]["refined"] is True
    assert payload["convergence"]["temporal"]["refined"] is False


def test_failed_sweep_is_incomplete_and_never_claims_convergence(tmp_path):
    import sine2d_benchmark_campaign as campaign

    def runner(config):
        result = _result(config)
        if config.friction_mu == 0.20:
            result.success = False
            result.abort_reason = "synthetic non-settling"
            result.completed_load_steps = 3
        return result

    settings = campaign.CampaignSettings(
        output_dir=str(tmp_path), campaign_name="failed-sweep", study="sweep"
    )
    result = campaign.run_campaign(
        settings, simulation_runner=runner, simulation_exporter=_fake_exporter
    )

    assert result.sweep_status == "incomplete"
    assert result.overall_status == "incomplete"
    assert result.convergence_status == "not_requested"
    assert result.sweep_summary is None
    assert result.sweep_plot is None
    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert payload["evidence_status"] == "not_assessed"
    assert payload["friction_sweep"]["status"] == "incomplete"
    failed = next(case for case in payload["cases"] if case["friction_mu"] == 0.20)
    assert failed["success"] is False
    assert failed["abort_reason"] == "synthetic non-settling"


def test_export_failure_marks_case_unsuccessful_in_manifest(tmp_path):
    import sine2d_benchmark_campaign as campaign

    def exporter(result):
        if result.config.friction_mu == 0.05:
            raise OSError("synthetic artifact failure")
        return _fake_exporter(result)

    settings = campaign.CampaignSettings(
        output_dir=str(tmp_path), campaign_name="failed-export", study="sweep"
    )
    result = campaign.run_campaign(
        settings,
        simulation_runner=_result,
        simulation_exporter=exporter,
    )

    assert result.sweep_status == "incomplete"
    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    failed = next(case for case in payload["cases"] if case["friction_mu"] == 0.05)
    assert failed["success"] is False
    assert "synthetic artifact failure" in failed["abort_reason"]


def test_manifest_omits_exporter_paths_that_do_not_exist(tmp_path):
    import sine2d_benchmark_campaign as campaign

    def exporter(result):
        if result.config.friction_mu == 0.05:
            return {"settled": Path(result.config.output_dir) / "missing.tsv"}
        return _fake_exporter(result)

    settings = campaign.CampaignSettings(
        output_dir=str(tmp_path), campaign_name="missing-artifact", study="sweep"
    )
    result = campaign.run_campaign(
        settings, simulation_runner=_result, simulation_exporter=exporter
    )

    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    failed = next(case for case in payload["cases"] if case["friction_mu"] == 0.05)
    assert failed["status"] == "incomplete"
    assert failed["artifacts"] == {}
    assert "export did not produce declared artifacts" in failed["abort_reason"]


def test_invalid_export_manifest_still_produces_campaign_failure_manifest(tmp_path):
    import sine2d_benchmark_campaign as campaign

    def exporter(result):
        if result.config.friction_mu == 0.05:
            return None
        return _fake_exporter(result)

    settings = campaign.CampaignSettings(
        output_dir=str(tmp_path), campaign_name="invalid-exporter", study="sweep"
    )
    result = campaign.run_campaign(
        settings, simulation_runner=_result, simulation_exporter=exporter
    )

    assert result.manifest.is_file()
    assert result.overall_status == "incomplete"
    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    failed = next(case for case in payload["cases"] if case["friction_mu"] == 0.05)
    assert failed["artifacts"] == {}
    assert failed["status"] == "incomplete"


def test_all_study_does_not_claim_complete_evidence_if_a_sweep_run_fails(tmp_path):
    import sine2d_benchmark_campaign as campaign

    def runner(config):
        result = _result(config)
        if config.friction_mu == 0.20:
            result.success = False
            result.abort_reason = "synthetic non-settling"
        return result

    settings = campaign.CampaignSettings(
        output_dir=str(tmp_path), campaign_name="partial-all", study="all"
    )
    result = campaign.run_campaign(
        settings, simulation_runner=runner, simulation_exporter=_fake_exporter
    )

    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert result.convergence_status == "converged"
    assert result.sweep_status == "incomplete"
    assert result.overall_status == "incomplete"
    assert payload["evidence_status"] == "campaign_incomplete"


def test_claimed_success_with_truncated_steps_and_curve_is_incomplete_not_refined(
    tmp_path,
):
    import sine2d_benchmark_campaign as campaign

    seen = []

    def runner(config):
        seen.append((config.num_nodes, config.num_load_steps))
        result = _result(config)
        if config.num_nodes == 1200:
            result.completed_load_steps = config.num_load_steps - 1
            result.settled_records = result.settled_records[:-1]
        return result

    settings = campaign.CampaignSettings(
        output_dir=str(tmp_path), campaign_name="truncated", study="convergence"
    )
    result = campaign.run_campaign(
        settings, simulation_runner=runner, simulation_exporter=_fake_exporter
    )

    assert seen == [(400, 60), (800, 60), (1200, 60), (800, 120)]
    assert result.convergence_status == "incomplete"
    assert result.overall_status == "incomplete"
    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert payload["evidence_status"] == "campaign_incomplete"
    truncated = next(case for case in payload["cases"] if case["num_nodes"] == 1200)
    assert truncated["status"] == "incomplete"
    assert truncated["success"] is False


def test_convergence_export_failure_is_incomplete_and_does_not_trigger_refinement(
    tmp_path,
):
    import sine2d_benchmark_campaign as campaign

    seen = []

    def runner(config):
        seen.append((config.num_nodes, config.num_load_steps))
        return _result(config)

    def exporter(result):
        if result.config.num_nodes == 1200:
            raise OSError("synthetic export failure")
        return _fake_exporter(result)

    settings = campaign.CampaignSettings(
        output_dir=str(tmp_path), campaign_name="export-failure", study="convergence"
    )
    result = campaign.run_campaign(
        settings, simulation_runner=runner, simulation_exporter=exporter
    )

    assert seen == [(400, 60), (800, 60), (1200, 60), (800, 120)]
    assert result.convergence_status == "incomplete"
    assert result.overall_status == "incomplete"
    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert payload["evidence_status"] == "campaign_incomplete"


def test_plot_failure_keeps_atomic_summary_and_writes_recoverable_manifest(
    tmp_path, monkeypatch
):
    import sine2d_benchmark_campaign as campaign

    def fail_plot(*_args, **_kwargs):
        raise OSError("synthetic plot failure")

    monkeypatch.setattr(campaign, "_write_sweep_plot", fail_plot)
    settings = campaign.CampaignSettings(
        output_dir=str(tmp_path), campaign_name="plot-failure", study="sweep"
    )
    result = campaign.run_campaign(
        settings,
        simulation_runner=_result,
        simulation_exporter=_fake_exporter,
    )

    assert result.sweep_status == "incomplete"
    assert result.overall_status == "incomplete"
    assert result.manifest.is_file()
    assert result.sweep_summary is not None and result.sweep_summary.is_file()
    assert result.sweep_plot is None
    assert not tuple(result.campaign_dir.glob("*.tmp"))
    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert payload["friction_sweep"]["summary_tsv"] == result.sweep_summary.name
    assert payload["friction_sweep"]["overlay_png"] is None
    assert "synthetic plot failure" in payload["friction_sweep"]["error"]


def test_numerical_failure_remains_not_converged_after_refinement(tmp_path):
    import sine2d_benchmark_campaign as campaign

    def runner(config):
        result = _result(config)
        if config.num_nodes == 400:
            result.success = False
            result.abort_reason = "synthetic Newton failure"
        return result

    settings = campaign.CampaignSettings(
        output_dir=str(tmp_path), campaign_name="numerical-failure", study="convergence"
    )
    result = campaign.run_campaign(
        settings, simulation_runner=runner, simulation_exporter=_fake_exporter
    )

    assert result.convergence_status == "not_converged"
    assert result.overall_status == "not_converged"
    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    failed = next(case for case in payload["cases"] if case["num_nodes"] == 400)
    assert failed["status"] == "numerical_failure"
    assert payload["evidence_status"] == "not_converged"


def test_directory_only_exporter_is_not_evidence_eligible(tmp_path):
    import sine2d_benchmark_campaign as campaign

    def exporter(result):
        run_dir = Path(result.config.output_dir) / result.config.run_name
        run_dir.mkdir(parents=True)
        artifacts = {"run_dir": run_dir}
        for key in campaign.REQUIRED_FILE_ARTIFACT_KEYS:
            directory = run_dir / key
            directory.mkdir()
            artifacts[key] = directory
        return artifacts

    settings = campaign.CampaignSettings(
        output_dir=str(tmp_path), campaign_name="directory-artifacts", study="sweep"
    )
    result = campaign.run_campaign(
        settings, simulation_runner=_result, simulation_exporter=exporter
    )

    assert result.overall_status == "incomplete"
    assert result.sweep_summary is None
    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert all(case["status"] == "incomplete" for case in payload["cases"])
    assert all(case["artifacts"] == {"run_dir": case["name"]} for case in payload["cases"])


def test_sparse_endpoint_only_settled_curve_is_incomplete_and_not_refined(tmp_path):
    import sine2d_benchmark_campaign as campaign

    seen = []

    def runner(config):
        seen.append((config.num_nodes, config.num_load_steps))
        result = _result(config)
        if config.num_nodes == 1200:
            result.settled_records = (
                result.settled_records[0],
                result.settled_records[-1],
            )
        return result

    settings = campaign.CampaignSettings(
        output_dir=str(tmp_path), campaign_name="sparse-curve", study="convergence"
    )
    result = campaign.run_campaign(
        settings, simulation_runner=runner, simulation_exporter=_fake_exporter
    )

    assert seen == [(400, 60), (800, 60), (1200, 60), (800, 120)]
    assert result.convergence_status == "incomplete"
    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    sparse = next(case for case in payload["cases"] if case["num_nodes"] == 1200)
    assert sparse["status"] == "incomplete"
    assert "settled load targets" in sparse["abort_reason"]


def test_settled_targets_allow_retry_subdivision_knots_but_require_every_target():
    import sine2d_benchmark_campaign as campaign
    from rod_soft_contact_PI_dynamic_quasi_static_2d_sine_insert import (
        SineInsertConfig,
    )

    config = SineInsertConfig(total_insertion=1.0, num_load_steps=4)
    with_subdivision = SimpleNamespace(
        settled_records=tuple(
            SimpleNamespace(insertion=value)
            for value in (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
        )
    )
    missing_target = SimpleNamespace(
        settled_records=tuple(
            SimpleNamespace(insertion=value)
            for value in (0.0, 0.125, 0.25, 0.5, 1.0)
        )
    )

    assert campaign._settled_load_target_issue(with_subdivision, config) is None
    issue = campaign._settled_load_target_issue(missing_target, config)
    assert issue is not None
    assert "0.75" in issue


def test_artifacts_reached_through_symlinked_parent_are_incomplete(tmp_path):
    import sine2d_benchmark_campaign as campaign

    filenames = {
        "config": "config.json",
        "settled": "settled.tsv",
        "history": "history.npz",
        "configurations": "configurations.png",
        "insertion_force_plot": "insertion_force.png",
        "normal_friction_plot": "normal_friction.png",
        "elastic_energy_plot": "elastic_energy.png",
    }

    def exporter(result):
        run_dir = Path(result.config.output_dir) / result.config.run_name
        run_dir.mkdir(parents=True)
        for filename in filenames.values():
            (run_dir / filename).write_bytes(b"artifact\n")
        alias = run_dir / "alias"
        alias.symlink_to(run_dir, target_is_directory=True)
        return {
            "run_dir": run_dir,
            **{key: alias / filename for key, filename in filenames.items()},
        }

    settings = campaign.CampaignSettings(
        output_dir=str(tmp_path), campaign_name="symlink-parent", study="sweep"
    )
    result = campaign.run_campaign(
        settings, simulation_runner=_result, simulation_exporter=exporter
    )

    assert result.overall_status == "incomplete"
    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert all(case["status"] == "incomplete" for case in payload["cases"])
    assert all(case["artifacts"] == {"run_dir": case["name"]} for case in payload["cases"])
