from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_teacher_release as release  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "security@example.invalid")
    _git(repo, "config", "user.name", "Security Test")
    (repo / "README.md").write_text("safe\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-qm", "safe")
    return repo


@pytest.mark.parametrize("date", ["", "2026092", "202609240", "2026-09-24", "abcdefgh", "２０２６０９２４", "../20260924"])
def test_release_date_requires_eight_ascii_digits(tmp_path: Path, date: str) -> None:
    with pytest.raises(release.ReleaseError, match="release date"):
        release.build_release(_repo(tmp_path), tmp_path / "bad.zip", release_date=date)


@pytest.mark.parametrize(
    "path",
    ["", "/absolute", "a//b", "a/./b", "a/../b", "../b", "a\\b", "a\x00b", "a\nb", "e\u0301.txt"],
)
def test_member_path_validation_rejects_unsafe_or_noncanonical_paths(path: str) -> None:
    with pytest.raises(release.ReleaseError, match="unsafe|canonical|empty"):
        release.validate_member_path(path)


def test_release_rejects_symlink_tree_entry(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    os.symlink("README.md", repo / "link")
    _git(repo, "add", "link")
    _git(repo, "commit", "-qm", "symlink")
    with pytest.raises(release.ReleaseError, match="mode 120000"):
        release.build_release(repo, tmp_path / "symlink.zip")


def test_release_rejects_gitlink_tree_entry(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{commit},vendor/submodule")
    _git(repo, "commit", "-qm", "gitlink")
    tree = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(release.ReleaseError, match="mode 160000"):
        release._tracked_entries(repo, tree)


def _info(name: str, mode: int = 0o100644) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, release.ZIP_TIMESTAMP)
    info.create_system = 3
    info.external_attr = mode << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def test_archive_verifier_rejects_duplicate_members(tmp_path: Path) -> None:
    archive = tmp_path / "duplicate.zip"
    root = "SoftRobot_teacher_code_20260924"
    with pytest.warns(UserWarning, match="Duplicate"):
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr(_info(f"{root}/README.md"), b"one")
            handle.writestr(_info(f"{root}/README.md"), b"two")
            handle.writestr(_info(f"{root}/MANIFEST.sha256"), b"")
    with pytest.raises(release.ReleaseError, match="duplicate"):
        release.verify_release_archive(archive, root, {"README.md"})


def test_archive_verifier_rejects_manifest_mismatch_and_bad_mode(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    root = "SoftRobot_teacher_code_20260924"
    payload = b"content"
    manifest = f"{hashlib.sha256(payload).hexdigest()}  README.md\n".encode()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(_info(f"{root}/README.md", 0o100755), payload)
        handle.writestr(_info(f"{root}/EXTRA.md"), b"extra")
        handle.writestr(_info(f"{root}/MANIFEST.sha256"), manifest)
    with pytest.raises(release.ReleaseError, match="mode|payload|extra"):
        release.verify_release_archive(archive, root, {"README.md"})


def test_archive_verifier_rejects_manifest_self_reference(tmp_path: Path) -> None:
    archive = tmp_path / "self.zip"
    root = "SoftRobot_teacher_code_20260924"
    manifest = f"{'0' * 64}  MANIFEST.sha256\n".encode()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(_info(f"{root}/MANIFEST.sha256"), manifest)
    with pytest.raises(release.ReleaseError, match="self-reference"):
        release.verify_release_archive(archive, root, set())


def test_path_conflicts_and_casefold_duplicates_are_rejected() -> None:
    with pytest.raises(release.ReleaseError, match="conflicting"):
        release._validate_unique_paths(["a", "a/b"], "test")
    with pytest.raises(release.ReleaseError, match="duplicate"):
        release._validate_unique_paths(["README.md", "readme.md"], "test")
    with pytest.raises(release.ReleaseError, match="conflicting"):
        release._validate_unique_paths(["A", "a/b"], "test")
    with pytest.raises(release.ReleaseError, match="conflicting"):
        release._validate_unique_paths(["dir", "DIR/file"], "test")


def test_archive_verifier_rejects_hash_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "hash.zip"
    root = "SoftRobot_teacher_code_20260924"
    manifest = f"{'0' * 64}  README.md\n".encode()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(_info(f"{root}/README.md"), b"content")
        handle.writestr(_info(f"{root}/MANIFEST.sha256"), manifest)
    with pytest.raises(release.ReleaseError, match="hash mismatch"):
        release.verify_release_archive(archive, root, {"README.md"})


def test_built_archive_has_exact_payload_safe_names_and_regular_0644_modes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    archive = tmp_path / "safe.zip"
    release.build_release(repo, archive)
    root = "SoftRobot_teacher_code_20260924"
    release.verify_release_archive(archive, root, {"README.md"})
    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
        assert len(infos) == 2
        assert len({item.filename for item in infos}) == 2
        for item in infos:
            assert release.validate_member_path(item.filename) == item.filename
            assert item.create_system == 3
            assert item.external_attr >> 16 == 0o100644
