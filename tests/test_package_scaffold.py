from __future__ import annotations

import importlib
from pathlib import Path

import pytest


EXPECTED_MODEL_IDS = (
    "contact-2d-dynamic-pi",
    "contact-2d-quasistatic-pi-v1",
    "contact-2d-quasistatic-pi-v2",
    "curved-tube-2d-arc",
    "foundation-2d-stretch-bend-torch",
    "foundation-2d-stretch-numpy",
    "foundation-3d-implicit-newton",
    "foundation-3d-stretch-bend-torch",
    "foundation-cantilever-der",
    "straight-tube-mesh-3d",
    "straight-tube-v3",
    "straight-tube-v4",
)


def test_registry_lists_all_packaged_models() -> None:
    from softrobot.registry import model_ids

    assert model_ids() == EXPECTED_MODEL_IDS


def test_cli_models_prints_one_registered_id_per_line(capsys: pytest.CaptureFixture[str]) -> None:
    from softrobot.cli import main

    assert main(["models"]) == 0
    assert tuple(capsys.readouterr().out.splitlines()) == EXPECTED_MODEL_IDS


def test_cli_run_accepts_common_options_after_model_id(tmp_path: Path) -> None:
    from softrobot.cli import main

    with pytest.raises(SystemExit, match="2"):
        main([
            "run",
            "straight-tube-v4",
            "--smoke-test",
            "--output-dir",
            str(tmp_path),
        ])


@pytest.mark.parametrize(
    ("legacy_name", "package_name", "symbol"),
    (
        ("centerline_3d", "softrobot.common.centerline_3d", "Centerline3D"),
        ("mesh_contact_v4", "softrobot.backends.mesh_contact_v4", "MeshGeometry"),
        ("der_cpp_backend", "softrobot.backends.der_cpp_backend", "DERCore"),
        (
            "rod_soft_contact_PI_dynamic_quasi_static_3d_v3",
            "softrobot.straight_tube.v3",
            "TOTAL_LENGTH",
        ),
        ("quasi_static_mesh_3d", "softrobot.straight_tube.quasi_static_mesh_3d", "DEFAULT_MESH_OBJ"),
        (
            "rod_soft_contact_PI_dynamic_quasi_static_3d_v4",
            "softrobot.straight_tube.v4",
            "TOTAL_LENGTH",
        ),
        (
            "rod_soft_contact_PI_dynamic_quasi_static_2d_arc",
            "softrobot.curved_tube.arc_2d",
            "TOTAL_LENGTH",
        ),
    ),
)
def test_legacy_modules_reexport_package_symbols(
    legacy_name: str, package_name: str, symbol: str
) -> None:
    legacy = importlib.import_module(legacy_name)
    package = importlib.import_module(package_name)

    assert getattr(legacy, symbol) is getattr(package, symbol)


def test_packaged_resource_paths_exist() -> None:
    from softrobot.backends import der_cpp_backend
    from softrobot.straight_tube import quasi_static_mesh_3d, v4

    assert quasi_static_mesh_3d.DEFAULT_MESH_OBJ.is_file()
    assert v4.MESH_OBJ_DEFAULT.is_file()
    assert "softrobot/resources/tube_obj" in v4.MESH_OBJ_DEFAULT.as_posix()
    assert (Path(der_cpp_backend.__file__).parent / "cpp" / "bindings.cpp").is_file()


def test_migrated_default_constants_are_unchanged() -> None:
    from softrobot.curved_tube import arc_2d
    from softrobot.straight_tube import v3, v4

    assert (v3.TOTAL_LENGTH, v3.DIAMETER, v3.NUM_NODES) == (1.5, 0.001, 250)
    assert (v4.TOTAL_LENGTH, v4.DIAMETER, v4.NUM_NODES) == (1.5, 0.001, 250)
    assert (v4.BACKEND_DEFAULT, v4.CONTACT_TYPE_DEFAULT) == ("cpp", "tube")
    assert (arc_2d.TOTAL_LENGTH, arc_2d.DIAMETER, arc_2d.NUM_NODES) == (1.5, 0.001, 800)
