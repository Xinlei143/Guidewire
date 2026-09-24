"""Small helpers shared by package and compatibility command-line entry points."""

from __future__ import annotations

from contextlib import contextmanager
import sys
from collections.abc import Iterator, Sequence


@contextmanager
def cli_arguments(arguments: Sequence[str] | None) -> Iterator[None]:
    """Temporarily expose explicit arguments to legacy argparse parsers."""
    if arguments is None:
        yield
        return
    original = sys.argv
    sys.argv = [original[0], *arguments]
    try:
        yield
    finally:
        sys.argv = original
