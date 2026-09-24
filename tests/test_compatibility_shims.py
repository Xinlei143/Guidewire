from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _environment_without_pythonpath() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    return environment


@pytest.mark.parametrize(
    "module_name",
    ("centerline_3d", "der_cpp_backend", "mesh_contact_v4"),
)
def test_import_shim_finds_src_package_without_pythonpath(module_name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        cwd=ROOT,
        env=_environment_without_pythonpath(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "script_name",
    (
        "quasi_static_mesh_3d.py",
        "rod_soft_contact_PI_dynamic_quasi_static_2d_arc.py",
        "rod_soft_contact_PI_dynamic_quasi_static_3d_v3.py",
        "rod_soft_contact_PI_dynamic_quasi_static_3d_v4.py",
    ),
)
def test_cli_shim_help_finds_src_package_without_pythonpath(
    script_name: str, tmp_path: Path
) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / script_name), "--help"],
        cwd=tmp_path,
        env=_environment_without_pythonpath(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
