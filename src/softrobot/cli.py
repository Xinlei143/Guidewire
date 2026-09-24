"""Top-level command-line interface for registered guidewire models."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
from collections.abc import Iterator, Sequence

from .registry import get_model, model_ids


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="softrobot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("models", help="list registered model IDs")
    run = subparsers.add_parser("run", help="run a registered model")
    run.add_argument("model_id", choices=model_ids())
    run.add_argument("--smoke-test", action="store_true")
    run.add_argument("--output-dir", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args, model_args = parser.parse_known_args(argv)
    if args.command == "models":
        if model_args:
            parser.error(f"unrecognized arguments: {' '.join(model_args)}")
        print(*model_ids(), sep="\n")
        return 0

    model = get_model(args.model_id)
    if args.smoke_test and not model.smoke_test:
        parser.error(
            f"model {model.model_id!r} does not implement the common --smoke-test contract yet"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with _working_directory(args.output_dir):
        result = model.load_main()(model_args)
    return 0 if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
