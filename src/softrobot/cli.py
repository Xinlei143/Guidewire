"""Top-level command-line interface for registered guidewire models."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import sys
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


def _absolutize_input_paths(
    arguments: Sequence[str],
    option_names: Sequence[str],
    base_directory: Path,
) -> list[str]:
    """Resolve registered input-path values against the invocation directory."""
    rewritten = list(arguments)
    options = frozenset(option_names)
    index = 0
    while index < len(rewritten):
        argument = rewritten[index]
        option, separator, inline_value = argument.partition("=")
        if separator and option in options:
            rewritten[index] = f"{option}={_absolute_path(inline_value, base_directory)}"
        elif argument in options and index + 1 < len(rewritten):
            index += 1
            rewritten[index] = str(_absolute_path(rewritten[index], base_directory))
        index += 1
    return rewritten


def _absolute_path(value: str, base_directory: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _parsers() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(prog="softrobot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("models", help="list registered model IDs")
    run = subparsers.add_parser("run", help="run a registered model", add_help=False)
    run.add_argument("-h", "--help", action="store_true", dest="run_help")
    run.add_argument("model_id", nargs="?", choices=model_ids())
    run.add_argument("--smoke-test", action="store_true")
    run.add_argument("--output-dir", type=Path, default=Path.cwd())
    return parser, run


def main(argv: Sequence[str] | None = None) -> int:
    invocation_directory = Path.cwd()
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser, run_parser = _parsers()
    args, model_args = parser.parse_known_args(arguments)
    if args.command == "models":
        if model_args:
            parser.error(f"unrecognized arguments: {' '.join(model_args)}")
        print(*model_ids(), sep="\n")
        return 0

    if args.run_help and args.model_id is None:
        run_parser.print_help()
        run_parser.exit(0)
    if args.model_id is None:
        run_parser.error("the following arguments are required: model_id")

    model = get_model(args.model_id)
    if args.run_help:
        model_args.append("--help")
    if args.smoke_test and not model.smoke_test:
        parser.error(
            f"model {model.model_id!r} does not implement the common --smoke-test contract yet"
        )

    model_args = _absolutize_input_paths(
        model_args, model.input_path_options, invocation_directory
    )
    output_directory = _absolute_path(str(args.output_dir), invocation_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with _working_directory(output_directory):
        result = model.load_main()(model_args)
    return 0 if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
