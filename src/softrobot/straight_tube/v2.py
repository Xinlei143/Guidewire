"""Packaged entry point for the historical 3-D straight-tube v2 model."""
from ._v2_source import *  # noqa: F401,F403
from . import _v2_source as _source
from .._tube_smoke import run_smoke
from ..common.cli import cli_arguments

def main(argv=None):
    result=run_smoke(argv, "straight-tube-v2-3d")
    if result >= 0: return result
    with cli_arguments(argv):
        args=_source.parse_args()
        _source.self_test(args) if args.self_test else _source.run_simulation(args)
    return 0

if __name__ == "__main__": raise SystemExit(main())
