"""Dedicated analytic-contact entrypoint for the historical 3-D arc model."""

from __future__ import annotations

import sys

from .._tube_smoke import run_smoke
from ..cli import _merge_default_model_args
from . import legacy_arc_3d as _solver


MODEL_ID = "curved-tube-3d-legacy-analytic"
DEFAULT_MODEL_ARGS = ("--contact-type", "analytic")


def _merge_args(argv=None) -> list[str]:
    explicit = list(sys.argv[1:] if argv is None else argv)
    return _merge_default_model_args(DEFAULT_MODEL_ARGS, explicit)


def main(argv=None) -> int:
    arguments = _merge_args(argv)
    smoke_result = run_smoke(arguments, MODEL_ID)
    if smoke_result >= 0:
        return smoke_result
    return _solver.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
