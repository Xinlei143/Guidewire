"""Compatibility entry point for :mod:`softrobot.curved_tube.arc_2d`."""

from pathlib import Path as _Path
import sys as _sys

_src = str(_Path(__file__).resolve().parent / "src")
if _src not in _sys.path:
    _sys.path.insert(0, _src)
del _Path, _sys, _src

from softrobot.curved_tube.arc_2d import *  # noqa: F401,F403
from softrobot.curved_tube.arc_2d import main as _main


if __name__ == "__main__":
    _main()
