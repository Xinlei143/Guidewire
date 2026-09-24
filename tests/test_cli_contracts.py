from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RELATIVE_MESH = Path(
    "assets/geometry/"
    "straight_tube_L1p5m_boundaryR50mm_shellR60mm_32sides_axial2.obj"
)


def test_model_help_is_forwarded_to_model_parser(capsys: pytest.CaptureFixture[str]) -> None:
    from softrobot.cli import main

    with pytest.raises(SystemExit) as caught:
        main(["run", "straight-tube-v4", "--help"])

    assert caught.value.code == 0
    output = capsys.readouterr().out
    assert "--backend" in output
    assert "--mesh-obj" in output


def test_run_help_remains_the_generic_run_help(capsys: pytest.CaptureFixture[str]) -> None:
    from softrobot.cli import main

    with pytest.raises(SystemExit) as caught:
        main(["run", "--help"])

    assert caught.value.code == 0
    output = capsys.readouterr().out
    assert "--output-dir" in output
    assert "--backend" not in output


def test_registry_declares_only_model_input_path_options() -> None:
    from softrobot.registry import get_model

    assert get_model("straight-tube-mesh-3d").input_path_options == ("--mesh-obj",)
    assert get_model("straight-tube-v4").input_path_options == (
        "--mesh-obj",
        "--centerline-csv",
    )
    assert get_model("curved-tube-2d-arc").input_path_options == ()


def test_input_path_rewrite_absolutizes_inputs_but_not_outputs() -> None:
    from softrobot.cli import _absolutize_input_paths
    from softrobot.registry import get_model

    arguments = [
        "--mesh-obj",
        str(RELATIVE_MESH),
        "--output-prefix",
        "relative/result",
    ]

    rewritten = _absolutize_input_paths(
        arguments,
        get_model("straight-tube-mesh-3d").input_path_options,
        ROOT,
    )

    assert rewritten[:2] == ["--mesh-obj", str((ROOT / RELATIVE_MESH).resolve())]
    assert rewritten[2:] == ["--output-prefix", "relative/result"]


def test_cli_resolves_relative_mesh_before_changing_output_directory(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "softrobot.cli",
            "run",
            "straight-tube-v4",
            "--output-dir",
            str(tmp_path),
            "--contact-type",
            "mesh",
            "--mesh-obj",
            str(RELATIVE_MESH),
            "--backend",
            "cpp",
            "--device",
            "cuda",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--backend cpp is CPU-only" in result.stderr
    assert "does not exist" not in result.stderr
