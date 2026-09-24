"""Resolve the single canonical ``assets/geometry`` data-file installation."""

from __future__ import annotations

import sys
from pathlib import Path


GEOMETRY_ASSETS = frozenset({
    "arc90_L1p5m_centerline_800.csv",
    "arc90_L1p5m_xz_centerline_800.csv",
    "arc90_L1p5m_xz_wallR60mm_protectionR50mm_64sides_axial256.obj",
    "arc90_legacy_protectionR40mm_40x80.obj",
    "straight_tube_L1p5m_boundaryR50mm_shellR60mm_32sides_axial2.obj",
})


def geometry_directories() -> tuple[Path, ...]:
    """Return source-tree, ``pip --target``, and normal-install candidates."""
    module = Path(__file__).resolve()
    candidates = (
        module.parents[3] / "assets" / "geometry",
        module.parents[2] / "assets" / "geometry",
        Path(sys.prefix).resolve() / "assets" / "geometry",
    )
    return tuple(dict.fromkeys(candidates))


def geometry_asset(name: str) -> Path:
    """Locate one named canonical geometry file without using the process cwd."""
    if name not in GEOMETRY_ASSETS:
        raise KeyError(f"unknown canonical geometry asset: {name!r}")
    for directory in geometry_directories():
        candidate = directory / name
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in geometry_directories())
    raise FileNotFoundError(f"geometry asset {name!r} was not installed; searched: {searched}")
