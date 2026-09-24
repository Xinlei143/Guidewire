#!/usr/bin/env python3
"""Build and self-verify a deterministic source-only teacher release."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


ZIP_TIMESTAMP = (2026, 9, 24, 0, 0, 0)
REGULAR_BLOB_MODES = {"100644"}
FORBIDDEN_PARTS = {".git", ".cpp_extension_build", "__macosx", "__pycache__", "third_party", "wire_report", "血管导丝"}
FORBIDDEN_FRAGMENTS = ("free_insert", "free-insert", "自由插入", "sine", "正弦")
FORBIDDEN_SUFFIXES = {".gif", ".mp4", ".npz", ".o", ".pdf", ".pyc", ".so", ".zip"}


class ReleaseError(RuntimeError):
    """Raised when repository or archive state violates the release contract."""


@dataclass(frozen=True)
class TreeEntry:
    mode: str
    object_type: str
    object_id: str
    path: str


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


def validate_member_path(path: str) -> str:
    """Validate an exact, normalized, relative POSIX archive path."""
    if not path:
        raise ReleaseError("empty archive path")
    if "\\" in path or "\0" in path or "\n" in path or "\r" in path:
        raise ReleaseError(f"unsafe archive path: {path!r}")
    if unicodedata.normalize("NFC", path) != path:
        raise ReleaseError(f"archive path is not canonical NFC: {path!r}")
    if path.startswith("/"):
        raise ReleaseError(f"unsafe absolute archive path: {path!r}")
    raw_parts = path.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ReleaseError(f"unsafe archive path component: {path!r}")
    normalized = PurePosixPath(path).as_posix()
    if normalized != path or PurePosixPath(path).is_absolute():
        raise ReleaseError(f"archive path is not canonical: {path!r}")
    return normalized


def _validate_unique_paths(paths: Iterable[str], context: str) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: dict[str, str] = {}
    for raw in paths:
        path = validate_member_path(raw)
        key = path.casefold()
        if key in seen:
            raise ReleaseError(f"duplicate {context} path: {path!r} conflicts with {seen[key]!r}")
        seen[key] = path
        normalized.append(path)
    exact = set(normalized)
    for path in normalized:
        parts = path.split("/")
        for index in range(1, len(parts)):
            prefix = "/".join(parts[:index])
            if prefix in exact:
                raise ReleaseError(f"conflicting {context} paths: {prefix!r} and {path!r}")
    return tuple(normalized)


def exclusion_reason(path: str) -> str | None:
    """Return why a safe repository path is excluded, or ``None``."""
    normalized = validate_member_path(path)
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


def _tracked_entries(repo: Path, commit: str) -> list[TreeEntry]:
    raw = _git(repo, "ls-tree", "-r", "-z", "--full-tree", commit, binary=True)
    assert isinstance(raw, bytes)
    entries: list[TreeEntry] = []
    all_paths: list[str] = []
    for record in (item for item in raw.split(b"\0") if item):
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode_bytes, type_bytes, object_id_bytes = metadata.split(b" ", 2)
            mode = mode_bytes.decode("ascii")
            object_type = type_bytes.decode("ascii")
            object_id = object_id_bytes.decode("ascii")
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as error:
            raise ReleaseError("malformed or non-UTF-8 git ls-tree record") from error
        validate_member_path(path)
        all_paths.append(path)
        if object_type != "blob" or mode not in REGULAR_BLOB_MODES:
            raise ReleaseError(
                f"unsupported git tree entry {path!r}: mode {mode}, type {object_type}"
            )
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", object_id):
            raise ReleaseError(f"invalid git object id for {path!r}")
        if exclusion_reason(path) is None:
            entries.append(TreeEntry(mode, object_type, object_id, path))
    _validate_unique_paths(all_paths, "git tree")
    return sorted(entries, key=lambda entry: entry.path)


def _zip_info(name: str) -> zipfile.ZipInfo:
    validate_member_path(name)
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _parse_manifest(data: bytes) -> dict[str, str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReleaseError("manifest is not UTF-8") from error
    manifest: dict[str, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ReleaseError("malformed manifest line")
        digest, path = match.groups()
        validate_member_path(path)
        if path == "MANIFEST.sha256":
            raise ReleaseError("manifest self-reference is forbidden")
        key = path.casefold()
        if any(existing.casefold() == key for existing in manifest):
            raise ReleaseError(f"duplicate manifest path: {path!r}")
        manifest[path] = digest
    _validate_unique_paths(manifest, "manifest")
    return manifest


def verify_release_archive(archive: Path | str, root: str, expected_payload: set[str]) -> None:
    """Reopen and strictly verify names, modes, manifest, hashes, and membership."""
    archive = Path(archive)
    validate_member_path(root)
    if "/" in root:
        raise ReleaseError("archive root must be one safe path component")
    with zipfile.ZipFile(archive, "r") as handle:
        infos = handle.infolist()
        names = [item.filename for item in infos]
        _validate_unique_paths(names, "ZIP member")
        prefix = f"{root}/"
        relative: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            if info.is_dir() or not info.filename.startswith(prefix):
                raise ReleaseError(f"unsafe or unexpected ZIP member: {info.filename!r}")
            relpath = validate_member_path(info.filename[len(prefix):])
            if info.create_system != 3 or info.external_attr >> 16 != 0o100644:
                raise ReleaseError(f"ZIP member has non-regular or non-0644 mode: {info.filename!r}")
            if info.date_time != ZIP_TIMESTAMP:
                raise ReleaseError(f"ZIP member has non-deterministic timestamp: {info.filename!r}")
            relative[relpath] = info
        expected_members = {*expected_payload, "MANIFEST.sha256"}
        if set(relative) != expected_members:
            extra = sorted(set(relative) - expected_members)
            missing = sorted(expected_members - set(relative))
            raise ReleaseError(f"ZIP payload mismatch; extra={extra}, missing={missing}")
        manifest = _parse_manifest(handle.read(relative["MANIFEST.sha256"]))
        payload_paths = set(relative) - {"MANIFEST.sha256"}
        if set(manifest) != payload_paths or set(manifest) != expected_payload:
            raise ReleaseError("manifest keys do not exactly match ZIP payload")
        for path, expected_digest in manifest.items():
            actual_digest = hashlib.sha256(handle.read(relative[path])).hexdigest()
            if actual_digest != expected_digest:
                raise ReleaseError(f"manifest hash mismatch for {path!r}")


def build_release(repo: Path | str, archive: Path | str, *, ref: str = "HEAD", release_date: str = "20260924") -> ReleaseResult:
    """Build a deterministic ZIP from *ref*, never from working-tree contents."""
    if re.fullmatch(r"[0-9]{8}", release_date) is None:
        raise ReleaseError("release date must be exactly eight ASCII digits")
    repo = Path(repo).resolve()
    archive = Path(archive).resolve()
    dirty = str(_git(repo, "status", "--porcelain", "--untracked-files=no")).strip()
    if dirty:
        raise ReleaseError("refusing release from a dirty tracked worktree")
    commit = str(_git(repo, "rev-parse", f"{ref}^{{commit}}")).strip()
    root = validate_member_path(f"SoftRobot_teacher_code_{release_date}")
    payload: dict[str, bytes] = {}
    for entry in _tracked_entries(repo, commit):
        data = _git(repo, "cat-file", "blob", entry.object_id, binary=True)
        assert isinstance(data, bytes)
        payload[entry.path] = data
    _validate_unique_paths(payload, "release payload")
    manifest = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {path}\n"
        for path, data in sorted(payload.items())
    ).encode("utf-8")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path, data in sorted(payload.items()):
            handle.writestr(_zip_info(f"{root}/{path}"), data)
        handle.writestr(_zip_info(f"{root}/MANIFEST.sha256"), manifest)
    verify_release_archive(archive, root, set(payload))
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
