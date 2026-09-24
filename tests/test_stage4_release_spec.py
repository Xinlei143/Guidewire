from __future__ import annotations

import os
import subprocess
import sys
import venv
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_NAMES = {
    "arc90_L1p5m_centerline_800.csv",
    "arc90_L1p5m_xz_centerline_800.csv",
    "arc90_L1p5m_xz_wallR60mm_protectionR50mm_64sides_axial256.obj",
    "arc90_legacy_protectionR40mm_40x80.obj",
    "straight_tube_L1p5m_boundaryR50mm_shellR60mm_32sides_axial2.obj",
}


def test_geometry_has_one_canonical_root_location() -> None:
    geometry = ROOT / "assets" / "geometry"
    assert {path.name for path in geometry.iterdir() if path.is_file()} == ASSET_NAMES
    assert not (ROOT / "src/softrobot/resources/tube_obj").exists()
    assert not list((ROOT / "src").rglob("*.obj"))
    assert not list((ROOT / "src").rglob("*.csv"))


def test_asset_resolver_works_outside_repository_cwd(tmp_path: Path, monkeypatch) -> None:
    from softrobot.common.assets import geometry_asset

    monkeypatch.chdir(tmp_path)
    for name in ASSET_NAMES:
        resolved = geometry_asset(name)
        assert resolved.is_file()
        assert resolved.name == name


def test_model_ids_follow_physical_task_order() -> None:
    from softrobot.registry import model_ids

    ids = model_ids()
    starts = tuple(value.split("-", 1)[0] for value in ids)
    ranks = {"foundation": 0, "contact": 1, "straight": 2, "curved": 3}
    assert [ranks[value] for value in starts] == sorted(ranks[value] for value in starts)


def test_lock_separates_python_distributions_from_system_tools() -> None:
    lock = (ROOT / "requirements-cpu.lock").read_text(encoding="utf-8")
    environment = (ROOT / "docs/environment.md").read_text(encoding="utf-8")
    assert "ninja==" not in lock
    assert "g++==" not in lock
    assert "1.13.0.git.kitware.jobserver-pipe-1" in environment
    assert "g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0" in environment


def test_documented_install_command_is_exact_and_offline_capable() -> None:
    expected = "python -m pip install --no-deps --no-build-isolation ."
    assert expected in (ROOT / "README.md").read_text(encoding="utf-8")
    assert expected in (ROOT / "docs/environment.md").read_text(encoding="utf-8")


def test_release_fixture_installs_offline_with_documented_command(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    import build_teacher_release

    repo = tmp_path / "repo"
    package = repo / "src" / "offline_fixture"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        "[build-system]\nrequires = ['setuptools']\nbuild-backend = 'setuptools.build_meta'\n"
        "[project]\nname = 'offline-fixture'\nversion = '0.0.1'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    archive = tmp_path / "release.zip"
    build_teacher_release.build_release(repo, archive)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(extracted)
    release_root = next(extracted.iterdir())
    environment_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment_dir)
    python = environment_dir / "bin" / "python"
    environment = os.environ.copy()
    environment["PIP_NO_INDEX"] = "1"
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", "--no-build-isolation", "."],
        cwd=release_root, env=environment, check=True,
    )
    subprocess.run(
        [str(python), "-c", "import offline_fixture; assert offline_fixture.VALUE == 1"],
        cwd=tmp_path, env=environment, check=True,
    )
