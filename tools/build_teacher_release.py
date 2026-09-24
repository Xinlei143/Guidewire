#!/usr/bin/env python3
"""Build a deterministic, source-only teacher release from a committed Git tree."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ZIP_TIMESTAMP = (2026, 9, 24, 0, 0, 0)
FORBIDDEN_PARTS = {".git", ".cpp_extension_build", "__macosx", "__pycache__", "third_party", "wire_report", "血管导丝"}
FORBIDDEN_FRAGMENTS = ("free_insert", "free-insert", "自由插入", "sine", "正弦")
FORBIDDEN_SUFFIXES = {".gif", ".mp4", ".npz", ".o", ".pdf", ".pyc", ".so", ".zip"}


class ReleaseError(RuntimeError):
    """Raised when repository state cannot produce an auditable release."""


@dataclass(frozen=True)
class ReleaseResult:
    archive: Path
    checksum_file: Path
    archive_sha256: str
    file_count: int
    commit: str


def _git(repo: Path, *args: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=not binary,
    )
    return completed.stdout


def exclusion_reason(path: str) -> str | None:
    """Return why a repository path is excluded, or ``None`` when releasable."""
    normalized = PurePosixPath(path).as_posix()
    folded = normalized.casefold()
    parts = set(PurePosixPath(folded).parts)
    if parts & FORBIDDEN_PARTS:
        return "forbidden path component"
    if any(fragment in folded for fragment in FORBIDDEN_FRAGMENTS):
        return "excluded research line"
    suffix = PurePosixPath(folded).suffix
    if suffix in FORBIDDEN_SUFFIXES:
        return "forbidden generated or binary extension"
    if suffix in {".obj", ".csv"} and not folded.startswith("assets/geometry/"):
        return "noncanonical geometry location"
    if PurePosixPath(folded).name.startswith("benchmark_"):
        return "benchmark helper or test"
    if folded == "tests/test_export_portfolio_figures.py":
        return "report generator test"
    if folded == "tools/export_portfolio_figures.py" or folded.startswith("portfolio_evidence/"):
        return "report or portfolio generator"
    if any(part in {"output", "outputs"} for part in PurePosixPath(folded).parts):
        return "historical output"
    return None


def _tracked_paths(repo: Path, commit: str) -> list[str]:
    raw = _git(repo, "ls-tree", "-r", "-z", "--name-only", commit, binary=True)
    assert isinstance(raw, bytes)
    paths = [item.decode("utf-8") for item in raw.split(b"\0") if item]
    return sorted(path for path in paths if exclusion_reason(path) is None)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def build_release(repo: Path | str, archive: Path | str, *, ref: str = "HEAD", release_date: str = "20260924") -> ReleaseResult:
    """Build a deterministic ZIP from *ref*, never from working-tree file contents."""
    repo = Path(repo).resolve()
    archive = Path(archive).resolve()
    dirty = str(_git(repo, "status", "--porcelain", "--untracked-files=no")).strip()
    if dirty:
        raise ReleaseError("refusing release from a dirty tracked worktree")
    commit = str(_git(repo, "rev-parse", f"{ref}^{{commit}}")).strip()
    root = f"SoftRobot_teacher_code_{release_date}"
    payload: dict[str, bytes] = {}
    for path in _tracked_paths(repo, commit):
        data = _git(repo, "show", f"{commit}:{path}", binary=True)
        assert isinstance(data, bytes)
        payload[path] = data
    violations = [path for path in payload if exclusion_reason(path) is not None]
    if violations:
        raise ReleaseError(f"release policy violation: {violations[0]}")
    manifest = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {path}\n"
        for path, data in sorted(payload.items())
    ).encode("utf-8")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path, data in sorted(payload.items()):
            zf.writestr(_zip_info(f"{root}/{path}"), data)
        zf.writestr(_zip_info(f"{root}/MANIFEST.sha256"), manifest)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum_file = archive.with_suffix(archive.suffix + ".sha256")
    checksum_file.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    return ReleaseResult(archive, checksum_file, digest, len(payload) + 1, commit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--release-date", default="20260924")
    args = parser.parse_args()
    result = build_release(args.repo, args.output, ref=args.ref, release_date=args.release_date)
    print(f"commit={result.commit}")
    print(f"files={result.file_count}")
    print(f"sha256={result.archive_sha256}")
    print(f"archive={result.archive}")
    print(f"checksum={result.checksum_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
