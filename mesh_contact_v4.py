"""Compatibility import for :mod:`softrobot.backends.mesh_contact_v4`."""

from pathlib import Path as _Path
import sys as _sys

_src = str(_Path(__file__).resolve().parent / "src")
if _src not in _sys.path:
    _sys.path.insert(0, _src)
del _Path, _sys, _src

from softrobot.backends.mesh_contact_v4 import *  # noqa: F401,F403
