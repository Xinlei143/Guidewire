"""Packaged entry point for the historical pure-Python 2-D arc model."""
from ._legacy_arc_2d_source import *  # noqa: F401,F403
from . import _legacy_arc_2d_source as _source
from .._tube_smoke import run_smoke
from ..common.cli import cli_arguments

def main(argv=None):
    result=run_smoke(argv, "curved-tube-2d-legacy-python")
    if result >= 0: return result
    with cli_arguments(argv):
        args=_source.parse_args()
        if args.self_test: _source.run_self_test(args)
        else: _source.visualize_results(_source.run_simulation(args), args)
    return 0

if __name__ == "__main__": raise SystemExit(main())
