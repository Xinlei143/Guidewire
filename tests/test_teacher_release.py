from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_teacher_release as release  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "release-test@example.invalid")
    _git(repo, "config", "user.name", "Release Test")
    (repo / "src").mkdir()
    (repo / "src" / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "README.md").write_text("teacher release\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    return repo


def _manifest_members(archive: Path) -> tuple[str, dict[str, str]]:
    with zipfile.ZipFile(archive) as zf:
        names = sorted(info.filename for info in zf.infolist() if not info.is_dir())
        root = names[0].split("/", 1)[0]
        manifest_name = f"{root}/MANIFEST.sha256"
        manifest = {}
        for line in zf.read(manifest_name).decode("utf-8").splitlines():
            digest, relpath = line.split("  ", 1)
            manifest[relpath] = digest
        for relpath, digest in manifest.items():
            assert hashlib.sha256(zf.read(f"{root}/{relpath}")).hexdigest() == digest
    return manifest_name, manifest


def test_release_is_deterministic_manifested_and_uses_only_git_tree(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    (repo / "untracked.txt").write_text("must not leak\n", encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    one = release.build_release(repo, first, ref="HEAD", release_date="20260924")
    two = release.build_release(repo, second, ref="HEAD", release_date="20260924")

    assert one.archive_sha256 == two.archive_sha256
    assert first.read_bytes() == second.read_bytes()
    manifest_name, manifest = _manifest_members(first)
    assert manifest_name == "SoftRobot_teacher_code_20260924/MANIFEST.sha256"
    assert sorted(manifest) == ["README.md", "src/model.py"]
    with zipfile.ZipFile(first) as zf:
        assert all(info.date_time == release.ZIP_TIMESTAMP for info in zf.infolist())
        assert not any("untracked" in info.filename for info in zf.infolist())


def test_release_rejects_dirty_tracked_tree(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    (repo / "README.md").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(release.ReleaseError, match="tracked worktree"):
        release.build_release(repo, tmp_path / "release.zip")


@pytest.mark.parametrize(
    "path",
    [
        ".git/config", "wire_report/report.md", "history/血管导丝/a.py",
        "src/free_insert.py", "src/free-insert.py", "src/sine_demo.py",
        "src/正弦模型.py", "__MACOSX/x", "src/__pycache__/x.pyc",
        "src/.cpp_extension_build/x.o", "third_party/eigen.h", "data/run.npz",
        "paper.pdf", "movie.mp4", "bundle.zip", "tests/benchmark_speed.py",
        "src/softrobot/backends/cpp/benchmark_der_core.py",
        "src/softrobot/resources/tube_obj/duplicate.obj",
        "history/duplicate.csv",
        "tests/test_export_portfolio_figures.py",
        "tools/export_portfolio_figures.py", "portfolio_evidence/manifest.json",
    ],
)
def test_release_forbidden_path_policy(path: str) -> None:
    assert release.exclusion_reason(path) is not None


def test_repository_release_contract_documented() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for phrase in (
        "softrobot models", "softrobot smoke --all", "--smoke-test",
        "python -m softrobot", "Python 3.10", "g++", "Ninja",
        "smoke test 不是物理验证", "老师版 ZIP 不含", "正弦插入", "自由插入",
        "当前合并后的研究工作区仍保留",
    ):
        assert phrase in readme
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    for lower_bound in ("numpy>=", "scipy>=", "torch>=", "matplotlib>=", "Pillow>=", "pytest>="):
        assert lower_bound in pyproject
    lock = (REPO / "requirements-cpu.lock").read_text(encoding="utf-8")
    for requirement in (
        "numpy==", "scipy==", "torch==", "matplotlib==", "Pillow==",
        "pytest==", "setuptools==",
    ):
        assert requirement in lock
