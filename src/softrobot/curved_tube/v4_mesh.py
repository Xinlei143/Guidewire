"""Dedicated packaged-mesh entrypoint for the current curved v4 task."""

from __future__ import annotations

import sys

from .._tube_smoke import run_smoke
from ..cli import _merge_default_model_args
from ..common.assets import geometry_asset
from ..straight_tube import v4 as _solver


MODEL_ID = "curved-tube-v4-mesh"


def _default_model_args() -> tuple[str, ...]:
    return (
        "--contact-type", "mesh",
        "--mesh-obj", str(geometry_asset(
            "arc90_L1p5m_xz_wallR60mm_protectionR50mm_64sides_axial256.obj"
        )),
        "--centerline-csv", str(geometry_asset("arc90_L1p5m_xz_centerline_800.csv")),
    )


def _merge_args(argv=None) -> list[str]:
    explicit = list(sys.argv[1:] if argv is None else argv)
    return _merge_default_model_args(_default_model_args(), explicit)


def main(argv=None) -> int:
    arguments = _merge_args(argv)
    smoke_result = run_smoke(arguments, MODEL_ID)
    if smoke_result >= 0:
        return smoke_result
    return _solver.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
