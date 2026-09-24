from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = (
    "foundation-2d-stretch-bend-torch",
    "foundation-2d-stretch-numpy",
    "foundation-3d-implicit-newton",
    "foundation-3d-stretch-bend-torch",
    "foundation-cantilever-der",
    "contact-2d-dynamic-pi",
    "contact-2d-quasistatic-pi-v1",
    "contact-2d-quasistatic-pi-v2",
    "straight-tube-2d-pi",
    "straight-tube-mesh-3d",
    "straight-tube-v2-3d",
    "straight-tube-v3",
    "straight-tube-v4",
    "curved-tube-2d-arc",
    "curved-tube-2d-legacy-python",
    "curved-tube-3d-legacy-analytic",
    "curved-tube-3d-legacy-mesh",
    "curved-tube-v4-analytic",
    "curved-tube-v4-mesh",
)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_root = str(ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not existing else os.pathsep.join((source_root, existing))
    )
    environment["MPLBACKEND"] = "Agg"
    return environment


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_registry_module_runs_its_own_smoke_from_arbitrary_cwd(
    model_id: str, tmp_path: Path,
) -> None:
    from softrobot.registry import get_model

    spec = get_model(model_id)
    output = tmp_path / "outputs" / model_id
    working_directory = tmp_path / "unrelated-cwd"
    working_directory.mkdir(exist_ok=True)
    result = subprocess.run(
        [
            sys.executable, "-m", spec.module, "--smoke-test", "--output-dir", str(output),
        ],
        cwd=working_directory,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=spec.smoke_timeout_s,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads((output / "smoke_result.json").read_text(encoding="utf-8"))
    assert payload["model_id"] == model_id
    assert payload["status"] == "passed"


@pytest.mark.parametrize(
    ("module_name", "default_contact"),
    (
        ("softrobot.curved_tube.legacy_arc_3d_analytic", "analytic"),
        ("softrobot.curved_tube.legacy_arc_3d_mesh", "mesh"),
        ("softrobot.curved_tube.v4_analytic", "curved-tube"),
        ("softrobot.curved_tube.v4_mesh", "mesh"),
    ),
)
def test_curved_entrypoint_explicit_contact_type_overrides_preset(
    module_name: str, default_contact: str,
) -> None:
    module = importlib.import_module(module_name)

    defaults = module._merge_args([])
    explicit = module._merge_args(["--contact-type", "tube"])

    assert defaults[defaults.index("--contact-type") + 1] == default_contact
    assert explicit.count("--contact-type") == 1
    assert explicit[explicit.index("--contact-type") + 1] == "tube"


def test_curved_v4_mesh_direct_defaults_use_packaged_absolute_assets() -> None:
    from softrobot.curved_tube import v4_mesh

    arguments = v4_mesh._merge_args([])
    for option in ("--mesh-obj", "--centerline-csv"):
        path = Path(arguments[arguments.index(option) + 1])
        assert path.is_absolute()
        assert path.is_file()
    centerline = Path(arguments[arguments.index("--centerline-csv") + 1])
    assert centerline.name == "arc90_L1p5m_xz_centerline_800.csv"
