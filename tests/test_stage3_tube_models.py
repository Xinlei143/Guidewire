from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


TUBE_MODEL_IDS = (
    "curved-tube-2d-arc",
    "curved-tube-2d-legacy-python",
    "curved-tube-3d-legacy-analytic",
    "curved-tube-3d-legacy-mesh",
    "curved-tube-v4-analytic",
    "curved-tube-v4-mesh",
    "straight-tube-2d-pi",
    "straight-tube-mesh-3d",
    "straight-tube-v2-3d",
    "straight-tube-v3",
    "straight-tube-v4",
)


def test_registry_covers_stage3_tube_lineage() -> None:
    from softrobot.registry import get_model, model_ids

    assert set(TUBE_MODEL_IDS) <= set(model_ids())
    for model_id in TUBE_MODEL_IDS:
        model = get_model(model_id)
        assert model.smoke_test
        assert model.smoke_timeout_s > 0
        assert model.task == "guidewire"


def test_v4_aliases_are_registry_presets() -> None:
    from softrobot.registry import get_model

    analytic = get_model("curved-tube-v4-analytic")
    mesh = get_model("curved-tube-v4-mesh")
    assert analytic.module == mesh.module == "softrobot.straight_tube.v4"
    assert analytic.default_model_args == ("--contact-type", "curved-tube")
    assert mesh.default_model_args[:2] == ("--contact-type", "mesh")
    assert "--centerline-csv" in mesh.default_model_args


def test_explicit_model_arguments_override_registry_preset() -> None:
    from softrobot.cli import _merge_default_model_args

    assert _merge_default_model_args(
        ("--backend", "cpp", "--contact-type", "mesh"),
        ("--backend", "python", "--num-nodes", "8"),
    ) == ["--contact-type", "mesh", "--backend", "python", "--num-nodes", "8"]
    assert _merge_default_model_args(
        ("--backend=cpp", "--contact-type=mesh"),
        ("--backend=python",),
    ) == ["--contact-type=mesh", "--backend=python"]


@pytest.mark.parametrize("model_id", TUBE_MODEL_IDS)
def test_tube_model_smoke_writes_solver_evidence(tmp_path: Path, model_id: str) -> None:
    output = tmp_path / model_id
    result = subprocess.run(
        [sys.executable, "-m", "softrobot.cli", "run", model_id,
         "--smoke-test", "--output-dir", str(output)],
        text=True,
        capture_output=True,
        timeout=320,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads((output / "smoke_result.json").read_text(encoding="utf-8"))
    assert payload["model_id"] == model_id
    assert payload["status"] == "passed"
    assert payload["finite"] is True
    assert payload["steps"] >= 1
    variants = payload["variants"]
    assert variants
    assert all(item["finite"] is True for item in variants)
    assert all(item["converged"] is True for item in variants)
    assert all(item["iterations"] >= 1 for item in variants)
    assert all(item["residual"] >= 0 for item in variants)


def test_required_multivariant_smokes_are_explicit(tmp_path: Path) -> None:
    expected = {
        "straight-tube-v4": {"python+tube", "cpp+tube", "cpp+straight-mesh"},
        "curved-tube-2d-arc": {"python", "cpp"},
        "curved-tube-v4-analytic": {"python", "cpp"},
        "curved-tube-v4-mesh": {"python", "cpp"},
    }
    for model_id, names in expected.items():
        output = tmp_path / model_id
        result = subprocess.run(
            [sys.executable, "-m", "softrobot.cli", "run", model_id,
             "--smoke-test", "--output-dir", str(output)],
            text=True, capture_output=True, timeout=320,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        payload = json.loads((output / "smoke_result.json").read_text(encoding="utf-8"))
        assert {item["name"] for item in payload["variants"]} == names


def test_no_forbidden_historical_payloads_are_packaged() -> None:
    package = Path(__file__).parents[1] / "src" / "softrobot"
    forbidden_parts = {"third_party", "outputs", "output", "__pycache__", "media"}
    forbidden_suffixes = {".npz", ".zip", ".gif", ".png", ".mp4"}
    for path in package.rglob("*"):
        if "__pycache__" in path.parts:
            continue
        assert not (forbidden_parts & set(path.parts)), path
        assert path.suffix.lower() not in forbidden_suffixes, path
        if path.is_file() and path.suffix == ".py":
            assert not path.read_bytes().startswith(b"\xef\xbb\xbf"), path


def test_lineage_records_packaged_legacy_arc_mesh_hash() -> None:
    import hashlib

    root = Path(__file__).parents[1]
    relative_asset = Path(
        "src/softrobot/resources/tube_obj/arc90_legacy_protectionR40mm_40x80.obj"
    )
    digest = hashlib.sha256((root / relative_asset).read_bytes()).hexdigest()
    lineage = (root / "docs" / "model_lineage.md").read_text(encoding="utf-8")

    assert digest == "9d68298882fe013ea3ef24a125d268ad49a3b0f40be910d4aeb3e8f75c79354a"
    assert (
        f"packaged LF-normalized `{relative_asset.as_posix()}` SHA256 `{digest}`"
        in lineage
    )
    assert (
        "source OBJ snapshot SHA256 "
        "`88ba95136f106cb23cc76c2f73b6fdee0aae47cde7e4f4c5283fbc2707ed05eb`"
        in lineage
    )


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    (
        ("softrobot.straight_tube.legacy_pi_2d", "implicit_euler_step"),
        ("softrobot.straight_tube.v2", "implicit_euler_step"),
        ("softrobot.straight_tube.v3", "implicit_euler_step"),
        ("softrobot.straight_tube.quasi_static_mesh_3d", "mesh_implicit_euler_step"),
        ("softrobot.curved_tube.legacy_arc_2d", "implicit_euler_step"),
        ("softrobot.curved_tube.legacy_arc_3d", "implicit_step"),
        ("softrobot.curved_tube.arc_2d", "implicit_euler_step"),
        ("softrobot.straight_tube.v4", "_python_implicit_euler_step"),
    ),
)
def test_smoke_calls_corresponding_production_implicit_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_name: str, function_name: str,
) -> None:
    module = importlib.import_module(module_name)

    def production_probe(*args, **kwargs):
        raise RuntimeError("production-step-probe")

    monkeypatch.setattr(module, function_name, production_probe)
    with pytest.raises(RuntimeError, match="production-step-probe"):
        module.main(["--smoke-test", "--output-dir", str(tmp_path)])


def test_tube_smoke_artifacts_require_measured_contact_and_motion(tmp_path: Path) -> None:
    from softrobot.cli import main

    for model_id in TUBE_MODEL_IDS:
        output = tmp_path / model_id
        assert main(["run", model_id, "--smoke-test", "--output-dir", str(output)]) == 0
        payload = json.loads((output / "smoke_result.json").read_text(encoding="utf-8"))
        for variant in payload["variants"]:
            assert variant["state_delta_norm"] > 0.0
            assert variant["contact_force_norm"] > 0.0 or variant["contact_count"] > 0
            if "mesh" in model_id or "mesh" in variant["name"]:
                assert variant["mesh_obj"]
                assert variant["mesh_face_count"] > 0


def test_shared_surrogate_is_not_present() -> None:
    from softrobot import _tube_smoke

    assert not hasattr(_tube_smoke, "_python_newton")

@pytest.mark.parametrize(
    ("runner_module", "probe_module", "function_name"),
    (
        ("softrobot.straight_tube.legacy_pi_2d", "softrobot.straight_tube._legacy_pi_2d_source", "compute_PI_wall_force"),
        ("softrobot.straight_tube.v2", "softrobot.straight_tube._v2_source", "contact_quantities"),
        ("softrobot.straight_tube.v3", "softrobot.straight_tube.v3", "contact_quantities"),
        ("softrobot.straight_tube.quasi_static_mesh_3d", "softrobot.straight_tube.quasi_static_mesh_3d", "mesh_contact_quantities"),
        ("softrobot.curved_tube.legacy_arc_2d", "softrobot.curved_tube._legacy_arc_2d_source", "contact_quantities"),
        ("softrobot.curved_tube.legacy_arc_3d", "softrobot.curved_tube._legacy_arc_3d_source", "contact_response"),
        ("softrobot.curved_tube.arc_2d", "softrobot.curved_tube.arc_2d", "contact_quantities"),
        ("softrobot.straight_tube.v4", "softrobot.straight_tube.v4", "contact_quantities"),
    ),
)
def test_smoke_calls_corresponding_production_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner_module: str,
    probe_module: str, function_name: str,
) -> None:
    runner = importlib.import_module(runner_module)
    probe = importlib.import_module(probe_module)

    def production_probe(*args, **kwargs):
        raise RuntimeError("production-contact-probe")

    monkeypatch.setattr(probe, function_name, production_probe)
    with pytest.raises(RuntimeError, match="production-contact-probe"):
        runner.main(["--smoke-test", "--output-dir", str(tmp_path)])
