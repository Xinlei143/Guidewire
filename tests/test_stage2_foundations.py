from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
NEW_MODELS = {
    "foundation-2d-stretch-numpy": "softrobot.foundations.stretch_2d_numpy",
    "foundation-2d-stretch-bend-torch": "softrobot.foundations.stretch_bend_2d_torch",
    "foundation-3d-stretch-bend-torch": "softrobot.foundations.stretch_bend_3d_torch",
    "foundation-3d-implicit-newton": "softrobot.foundations.implicit_newton_3d",
    "foundation-cantilever-der": "softrobot.foundations.cantilever.run_cantilever_demo",
    "contact-2d-dynamic-pi": "softrobot.contact.dynamic_pi_2d",
    "contact-2d-quasistatic-pi-v1": "softrobot.contact.quasistatic_pi_v1",
    "contact-2d-quasistatic-pi-v2": "softrobot.contact.quasistatic_pi_v2",
}


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["MPLBACKEND"] = "Agg"
    return environment


def test_package_modules_use_relative_internal_imports() -> None:
    violations: list[str] = []
    for path in (ROOT / "src/softrobot").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.level == 0 and node.module == "softrobot._smoke":
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.module}")

    assert violations == []


def test_quasistatic_registry_sources_use_complete_sha256() -> None:
    from softrobot.registry import get_model

    assert get_model("contact-2d-quasistatic-pi-v1").source == (
        "SoftRobot.zip root untracked SHA256 "
        "1a0b234ef0693053d2296a99e5b023888c31fe6f913d099f6b551610a40a894b"
    )
    assert get_model("contact-2d-quasistatic-pi-v2").source == (
        "SoftRobot.zip root untracked SHA256 "
        "baa0c14e3b9b12bd93b14b0def26c03006b494b4eed61f5f69cb5398945d76cf"
    )


def test_cpu_smoke_runtime_restores_process_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib
    import numpy as np
    import torch
    from softrobot import _smoke

    assert hasattr(_smoke, "cpu_smoke_runtime")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "existing-selection")
    np.random.seed(314159)
    torch.manual_seed(271828)
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    device_before = torch.get_default_device()
    backend_before = matplotlib.get_backend()

    with _smoke.cpu_smoke_runtime():
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        assert torch.get_default_device() == torch.device("cpu")
        assert matplotlib.get_backend().lower() == "agg"
        np.random.random()
        torch.rand(1)

    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert torch.get_default_device() == device_before
    assert matplotlib.get_backend() == backend_before
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "existing-selection"


def test_atomic_json_replaces_complete_document(tmp_path: Path) -> None:
    from softrobot import _smoke

    assert hasattr(_smoke, "write_json_atomic")
    destination = tmp_path / "result.json"
    destination.write_text('{"old": true}\n')

    _smoke.write_json_atomic(destination, {"new": True})

    assert json.loads(destination.read_text()) == {"new": True}
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_atomic_json_preserves_old_document_and_cleans_temp_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from softrobot import _smoke

    assert hasattr(_smoke, "write_json_atomic")
    destination = tmp_path / "result.json"
    destination.write_text('{"old": true}\n')

    def fail_replace(_source, _destination):
        raise OSError("interrupted replace")

    monkeypatch.setattr(_smoke.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted replace"):
        _smoke.write_json_atomic(destination, {"new": True})

    assert json.loads(destination.read_text()) == {"old": True}
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_registry_exposes_stage_two_metadata() -> None:
    from softrobot.registry import get_model

    for model_id, module in NEW_MODELS.items():
        spec = get_model(model_id)
        assert spec.module == module
        assert spec.smoke_test is True
        assert spec.task in {"foundation", "contact"}
        assert spec.status == "historical"
        assert spec.dimension in {"2d", "3d"}
        assert spec.geometry in {"straight", "cantilever", "parallel-walls"}
        assert spec.contact in {"none", "pi-soft-wall"}
        assert spec.backend in {"numpy", "torch-cpu"}
        assert spec.source


def test_historical_default_constants_are_preserved() -> None:
    from softrobot.contact import dynamic_pi_2d, quasistatic_pi_v1, quasistatic_pi_v2
    from softrobot.foundations import (
        implicit_newton_3d,
        stretch_2d_numpy,
        stretch_bend_2d_torch,
        stretch_bend_3d_torch,
    )
    from softrobot.foundations.cantilever import run_cantilever_demo

    assert (stretch_2d_numpy.total_length, stretch_2d_numpy.num_nodes) == (2, 10)
    assert (stretch_2d_numpy.EA, stretch_2d_numpy.dt, stretch_2d_numpy.total_time) == (
        1000.0,
        0.0005,
        10,
    )
    assert (
        stretch_bend_2d_torch.total_length,
        stretch_bend_2d_torch.num_segments,
        stretch_bend_2d_torch.EA,
        stretch_bend_2d_torch.EI,
    ) == (2.0, 10, 1000.0, 100.0)
    assert (
        stretch_bend_3d_torch.total_length,
        stretch_bend_3d_torch.num_segments,
        stretch_bend_3d_torch.EA,
        stretch_bend_3d_torch.EI,
    ) == (50.0, 50, 1000.0, 200)
    assert (
        implicit_newton_3d.default_total_length,
        implicit_newton_3d.default_num_segments,
        implicit_newton_3d.default_dt,
        implicit_newton_3d.default_total_time,
    ) == (50.0, 50, 0.01, 5)
    defaults = run_cantilever_demo.parse_args([])
    assert (defaults.force_x, defaults.force_y, defaults.force_z) == (0.0, 0.0, 0.0)
    assert (defaults.dt, defaults.steps) == (2e-5, 600000)
    assert (dynamic_pi_2d.TOTAL_LENGTH, dynamic_pi_2d.NUM_NODES, dynamic_pi_2d.DT) == (
        1.0,
        30,
        1.0e-3,
    )
    assert (quasistatic_pi_v1.NUM_NODES, quasistatic_pi_v1.KI) == (100, 0)
    assert (quasistatic_pi_v2.NUM_NODES, quasistatic_pi_v2.KI) == (100, 400)


@pytest.mark.parametrize(("model_id", "module"), NEW_MODELS.items())
def test_each_stage_two_module_runs_a_real_smoke(
    model_id: str, module: str, tmp_path: Path
) -> None:
    output_dir = tmp_path / model_id
    result = subprocess.run(
        [sys.executable, "-m", module, "--smoke-test", "--output-dir", str(output_dir)],
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads((output_dir / "smoke_result.json").read_text())
    assert payload["model_id"] == model_id
    assert payload["status"] == "passed"
    assert payload["finite"] is True
    assert payload["steps"] >= 1
    assert len(payload["state_shape"]) == 2
    if model_id.startswith("contact-"):
        assert payload["converged"] is True
        assert payload["newton_iterations"] >= 1
        assert payload["residual"] >= 0.0
        assert payload["residual_norm"] == payload["residual"]


def test_contact_smoke_rejects_an_unconverged_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from softrobot.contact import dynamic_pi_2d
    from softrobot.contact._smoke import run_contact_smoke

    original_step = dynamic_pi_2d.implicit_euler_step

    def unconverged_step(*args, **kwargs):
        result = list(original_step(*args, **kwargs))
        result[-1] = SimpleNamespace(
            converged=False, iterations=2, residual_norm=1.0,
            step_norm=1.0, line_search_failures=1,
        )
        return tuple(result)

    monkeypatch.setattr(dynamic_pi_2d, "implicit_euler_step", unconverged_step)
    with pytest.raises(RuntimeError, match="did not converge"):
        run_contact_smoke(
            dynamic_pi_2d, "contact-2d-dynamic-pi", tmp_path, quasistatic=False
        )
    assert not (tmp_path / "smoke_result.json").exists()


def test_cli_runs_registered_stage_two_model(tmp_path: Path) -> None:
    from softrobot.cli import main

    output_dir = tmp_path / "single"
    assert main([
        "run",
        "foundation-2d-stretch-numpy",
        "--smoke-test",
        "--output-dir",
        str(output_dir),
    ]) == 0
    assert (output_dir / "smoke_result.json").is_file()


def test_smoke_all_reports_passed_and_skipped_models(tmp_path: Path) -> None:
    from softrobot.cli import main

    assert main(["smoke", "--all", "--output-dir", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "smoke_summary.json").read_text())
    assert set(summary["passed"]) == set(NEW_MODELS)
    assert set(summary["skipped"]) == {
        "curved-tube-2d-arc",
        "straight-tube-mesh-3d",
        "straight-tube-v3",
        "straight-tube-v4",
    }
    assert summary["failed"] == []


def test_smoke_all_uses_cli_run_subprocess_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from softrobot import cli
    from softrobot.registry import ModelSpec

    model = ModelSpec("isolated-model", "unused", smoke_test=True)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(cli, "model_ids", lambda: ("isolated-model",))
    monkeypatch.setattr(cli, "get_model", lambda _model_id: model)
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert cli._run_all_smokes(tmp_path) == 0
    assert calls == [[
        sys.executable, "-m", "softrobot.cli", "run", "isolated-model",
        "--smoke-test", "--output-dir", str(tmp_path / "isolated-model"),
    ]]


def test_smoke_all_removes_stale_passed_artifact_when_child_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from softrobot import cli
    from softrobot.registry import ModelSpec

    model = ModelSpec("broken-model", "unused", smoke_test=True)
    stale = tmp_path / "broken-model" / "smoke_result.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"status": "passed"}\n')

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 7, stdout="", stderr="forced child failure\n"
        )

    monkeypatch.setattr(cli, "model_ids", lambda: ("broken-model",))
    monkeypatch.setattr(cli, "get_model", lambda _model_id: model)
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert cli._run_all_smokes(tmp_path) == 1
    assert not stale.exists()
    summary = json.loads((tmp_path / "smoke_summary.json").read_text())
    assert summary["passed"] == []
    assert summary["skipped"] == []
    assert summary["failed"] == [
        {"model_id": "broken-model", "error": "exit code 7: forced child failure"}
    ]


@pytest.mark.parametrize("terminal_error", (KeyboardInterrupt, GeneratorExit))
def test_smoke_all_does_not_swallow_process_terminal_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_error: type[BaseException]
) -> None:
    from softrobot import cli
    from softrobot.registry import ModelSpec

    model = ModelSpec("terminal-model", "unused", smoke_test=True)

    def fake_run(command, **kwargs):
        raise terminal_error()

    monkeypatch.setattr(cli, "model_ids", lambda: ("terminal-model",))
    monkeypatch.setattr(cli, "get_model", lambda _model_id: model)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(terminal_error):
        cli._run_all_smokes(tmp_path)


def test_stage_two_source_tree_contains_no_source_junk() -> None:
    forbidden_suffixes = {".docx", ".mat", ".npz", ".pyc"}
    result = subprocess.run(
        ["git", "ls-files", "src/softrobot/foundations", "src/softrobot/contact"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    files = [Path(line) for line in result.stdout.splitlines()]

    assert files
    assert not [path for path in files if path.suffix.lower() in forbidden_suffixes]
    assert not [path for path in files if "__pycache__" in path.parts]
    assert not [path for path in files if path.name == ".DS_Store" or ".git" in path.parts]


def test_lineage_records_all_stage_two_models_and_sources() -> None:
    lineage = (ROOT / "docs/model_lineage.md").read_text()

    for model_id in NEW_MODELS:
        assert model_id in lineage
    assert "eb33bf3" in lineage
    assert "1a0b234ef0693053d2296a99e5b023888c31fe6f913d099f6b551610a40a894b" in lineage
    assert "baa0c14e3b9b12bd93b14b0def26c03006b494b4eed61f5f69cb5398945d76cf" in lineage
