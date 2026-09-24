"""Packaged entry point for the historical 2-D straight-tube PI model."""
from ._legacy_pi_2d_source import *  # noqa: F401,F403
from . import _legacy_pi_2d_source as _source
from .._tube_smoke import run_smoke

def main(argv=None):
    result=run_smoke(argv, "straight-tube-2d-pi")
    if result >= 0: return result
    history, initial, rest = _source.run_simulation()
    if _source.show_animation: _source.visualize_animation(history, initial)
    if _source.show_final_result: _source.visualize_result(history, initial, rest)
    return 0

if __name__ == "__main__": raise SystemExit(main())
