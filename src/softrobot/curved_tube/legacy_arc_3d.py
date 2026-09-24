"""Packaged entry point for the historical 3-D arc analytic/mesh model."""
from ._legacy_arc_3d_source import *  # noqa: F401,F403
from . import _legacy_arc_3d_source as _source
from .._tube_smoke import run_smoke
from ..common.cli import cli_arguments

def main(argv=None):
    result=run_smoke(argv, "curved-tube-3d-legacy-analytic")
    if result >= 0: return result
    with cli_arguments(argv):
        args=_source.parse_args()
        if args.self_test: _source.self_test(args)
        else: _source.visualize(_source.simulate(args), args)
    return 0

if __name__ == "__main__": raise SystemExit(main())
