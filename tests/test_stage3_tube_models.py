from __future__ import annotations

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
