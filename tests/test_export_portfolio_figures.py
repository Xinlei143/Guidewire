import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "tools" / "export_portfolio_figures.py"


def _write_fixture(tmp_path: Path, name: str, *, completed: int = 20, nonfinite: bool = False) -> Path:
    theta = np.linspace(0.0, np.pi / 2.0, 40)
    radius = 0.95
    centerline = np.column_stack(
        (radius * np.sin(theta), np.zeros_like(theta), radius * (1.0 - np.cos(theta)))
    )
    node_s = np.linspace(0, len(centerline) - 1, 12).astype(int)
    final = centerline[node_s].copy()
    final[:, 1] = 0.012 * np.sin(np.linspace(0.0, 3.0 * np.pi, len(final)))
    positions = np.stack((centerline[node_s], final))
    if nonfinite:
        positions[-1, 4, 1] = np.nan
    path = tmp_path / f"{name}.npz"
    np.savez_compressed(
        path,
        positions=positions,
        velocities=np.zeros_like(positions),
        completed_load_steps=np.asarray(completed),
        newton_residual=np.asarray((1.0e-8, 2.0e-8)),
        max_protection_error=np.asarray((0.0, 4.0e-5)),
        centerline_points=centerline,
        initial_pulse_seed=np.asarray(7),
        initial_pulse_nodes=np.asarray((3, 5, 7, 9)),
        initial_pulse_magnitudes_n=np.asarray((0.02, 0.03, 0.04, 0.05)),
    )
    return path


def _write_centerline(tmp_path: Path) -> Path:
    data = np.load(_write_fixture(tmp_path, "centerline_source"))["centerline_points"]
    path = tmp_path / "centerline.csv"
    np.savetxt(path, data, delimiter=",", header="x,y,z", comments="")
    return path


def _write_mesh(tmp_path: Path) -> Path:
    path = tmp_path / "vessel.obj"
    path.write_text(
        "\n".join(
            (
                "g protection",
                "v 0 -0.05 0",
                "v 0 0.05 0",
                "v 0 0 0.05",
                "v 0.95 -0.05 0.95",
                "v 0.95 0.05 0.95",
                "v 0.95 0 1.00",
                "f 1 2 3",
                "f 4 6 5",
                "f 1 4 5",
                "f 1 5 2",
                "f 2 5 6",
                "f 2 6 3",
                "f 3 6 4",
                "f 3 4 1",
            )
        )
        + "\n",
        encoding="ascii",
    )
    return path


def _command(tmp_path: Path, analytic: Path, mesh: Path) -> list[str]:
    return [
        sys.executable,
        str(EXPORTER),
        "--analytic-npz",
        str(analytic),
        "--mesh-npz",
        str(mesh),
        "--mesh-obj",
        str(_write_mesh(tmp_path)),
        "--centerline-csv",
        str(_write_centerline(tmp_path)),
        "--output-dir",
        str(tmp_path / "figures"),
        "--expected-load-steps",
        "20",
    ]


def test_exporter_creates_traceable_fixed_size_webp_assets(tmp_path):
    analytic = _write_fixture(tmp_path, "analytic")
    mesh = _write_fixture(tmp_path, "mesh")

    subprocess.run(_command(tmp_path, analytic, mesh), cwd=ROOT, check=True)

    output_dir = tmp_path / "figures"
    outputs = [output_dir / "hero-curved-guidewire.webp", output_dir / "contact-models.webp"]
    for output in outputs:
        assert output.is_file()
        with Image.open(output) as image:
            assert image.size == (1600, 900)
            assert image.format == "WEBP"

    with Image.open(outputs[0]) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    foreground = np.max(np.abs(pixels - np.asarray((251, 251, 250))), axis=2) > 12
    rows, columns = np.where(foreground)
    assert (columns.max() - columns.min() + 1) / 1600 >= 0.45
    assert (rows.max() - rows.min() + 1) / 900 >= 0.55

    metadata = json.loads((output_dir / "figure-metadata.json").read_text(encoding="utf-8"))
    assert metadata["validation"]["analytic"]["completed_load_steps"] == 20
    assert metadata["validation"]["mesh"]["completed_load_steps"] == 20
    assert metadata["figures"]["hero-curved-guidewire.webp"]["width"] == 1600
    assert metadata["figures"]["contact-models.webp"]["height"] == 900
    assert len(metadata["inputs"]["analytic_npz"]["sha256"]) == 64
    assert len(metadata["figures"]["hero-curved-guidewire.webp"]["sha256"]) == 64


def test_exporter_rejects_incomplete_runs_before_writing_assets(tmp_path):
    analytic = _write_fixture(tmp_path, "analytic", completed=19)
    mesh = _write_fixture(tmp_path, "mesh")

    result = subprocess.run(
        _command(tmp_path, analytic, mesh), cwd=ROOT, text=True, capture_output=True
    )

    assert result.returncode != 0
    assert "completed 19/20 load steps" in result.stderr
    assert not (tmp_path / "figures" / "hero-curved-guidewire.webp").exists()


def test_exporter_rejects_nonfinite_solver_state(tmp_path):
    analytic = _write_fixture(tmp_path, "analytic", nonfinite=True)
    mesh = _write_fixture(tmp_path, "mesh")

    result = subprocess.run(
        _command(tmp_path, analytic, mesh), cwd=ROOT, text=True, capture_output=True
    )

    assert result.returncode != 0
    assert "contains NaN or Inf" in result.stderr
    assert not (tmp_path / "figures" / "hero-curved-guidewire.webp").exists()
