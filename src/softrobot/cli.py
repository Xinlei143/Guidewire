"""Top-level command-line interface for registered guidewire models."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence

from ._smoke import write_json_atomic
from .registry import get_model, model_ids


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def _absolutize_input_paths(arguments: Sequence[str], option_names: Sequence[str], base_directory: Path) -> list[str]:
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


def _option_name(argument: str) -> str | None:
    if not argument.startswith("--"):
        return None
    return argument.split("=", 1)[0]


def _merge_default_model_args(defaults: Sequence[str], explicit: Sequence[str]) -> list[str]:
    """Prepend preset arguments except options explicitly supplied by the user."""
    explicit_names = {_option_name(value) for value in explicit}
    merged: list[str] = []
    index = 0
    while index < len(defaults):
        argument = defaults[index]
        name = _option_name(argument)
        inline = "=" in argument
        has_value = not inline and index + 1 < len(defaults) and not defaults[index + 1].startswith("--")
        if name not in explicit_names:
            merged.append(argument)
            if has_value:
                merged.append(defaults[index + 1])
        index += 2 if has_value else 1
    return [*merged, *explicit]


def _resolve_preset_resources(arguments: Sequence[str]) -> list[str]:
    resource_root = Path(__file__).resolve().parent / "resources" / "tube_obj"
    replacements = {
        "__ARC_CENTERLINE__": str(resource_root / "arc90_L1p5m_centerline_800.csv"),
        "__ARC_MESH__": str(resource_root / "arc90_L1p5m_xz_wallR60mm_protectionR50mm_64sides_axial256.obj"),
    }
    return [replacements.get(value, value) for value in arguments]


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
    smoke = subparsers.add_parser("smoke", help="run registered smoke tests")
    smoke.add_argument("--all", action="store_true", required=True)
    smoke.add_argument("--output-dir", type=Path, default=Path.cwd())
    return parser, run


def _validate_smoke_result(result_path: Path, model_id: str) -> None:
    if not result_path.is_file():
        raise ValueError("missing smoke_result.json")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid smoke_result.json: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("invalid smoke_result.json: root must be an object")
    if payload.get("model_id") != model_id:
        raise ValueError(
            f"smoke_result.json model_id mismatch: expected {model_id!r}, "
            f"got {payload.get('model_id')!r}"
        )
    if payload.get("status") != "passed":
        raise ValueError("smoke_result.json status is not 'passed'")
    if payload.get("finite") is not True:
        raise ValueError("smoke_result.json finite is not true")
    variants = payload.get("variants")
    if variants is not None:
        if not isinstance(variants, list) or not variants:
            raise ValueError("smoke_result.json variants must be a non-empty list")
        for index, variant in enumerate(variants):
            if not isinstance(variant, dict):
                raise ValueError(f"smoke_result.json variant {index} must be an object")
            if variant.get("finite") is not True or variant.get("converged") is not True:
                raise ValueError(f"smoke_result.json variant {index} is not finite and converged")
            if not isinstance(variant.get("iterations"), int) or variant["iterations"] < 1:
                raise ValueError(f"smoke_result.json variant {index} has invalid iterations")
            for key in ("residual", "state_delta_norm", "contact_force_norm"):
                value = variant.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"smoke_result.json variant {index} has invalid {key}")
            if variant["state_delta_norm"] <= 0.0:
                raise ValueError(f"smoke_result.json variant {index} did not move")
            contact_count = variant.get("contact_count")
            if not isinstance(contact_count, int) or contact_count < 0:
                raise ValueError(f"smoke_result.json variant {index} has invalid contact_count")
            if contact_count == 0 and variant["contact_force_norm"] <= 0.0:
                raise ValueError(f"smoke_result.json variant {index} did not exercise contact")


def _run_all_smokes(output_directory: Path) -> int:
    output_directory.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list] = {"passed": [], "skipped": [], "failed": []}
    source_root = str(Path(__file__).resolve().parents[1])
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not existing_pythonpath
        else source_root + os.pathsep + existing_pythonpath
    )
    for model_id in model_ids():
        model = get_model(model_id)
        if not model.smoke_test:
            summary["skipped"].append(model_id)
            print(f"SKIPPED {model_id}: common smoke contract not implemented yet")
            continue
        model_output = output_directory / model_id
        model_output.mkdir(parents=True, exist_ok=True)
        result_path = model_output / "smoke_result.json"
        result_path.unlink(missing_ok=True)
        command = [
            sys.executable, "-m", "softrobot.cli", "run", model_id,
            "--smoke-test", "--output-dir", str(model_output),
        ]
        started = time.perf_counter()
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, env=environment,
                timeout=model.smoke_timeout_s,
            )
        except subprocess.TimeoutExpired:
            result_path.unlink(missing_ok=True)
            error = f"timed out after {model.smoke_timeout_s:g} seconds"
            summary["failed"].append({"model_id": model_id, "error": error})
            print(f"FAILED {model_id}: {error}")
            continue
        except Exception as error:
            result_path.unlink(missing_ok=True)
            summary["failed"].append({"model_id": model_id, "error": str(error)})
            print(f"FAILED {model_id}: {error}")
            continue
        if result.returncode != 0:
            result_path.unlink(missing_ok=True)
            detail = result.stderr.strip() or result.stdout.strip() or "no child output"
            error = f"exit code {result.returncode}: {detail}"
            summary["failed"].append({"model_id": model_id, "error": error})
            print(f"FAILED {model_id}: {error}")
            continue
        try:
            _validate_smoke_result(result_path, model_id)
        except ValueError as validation_error:
            result_path.unlink(missing_ok=True)
            error = str(validation_error)
            summary["failed"].append({"model_id": model_id, "error": error})
            print(f"FAILED {model_id}: {error}")
            continue
        elapsed = time.perf_counter() - started
        summary["passed"].append(model_id)
        print(f"PASSED {model_id} ({elapsed:.3f}s)")
    write_json_atomic(output_directory / "smoke_summary.json", summary)
    return 1 if summary["failed"] else 0


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
    if args.command == "smoke":
        if model_args:
            parser.error(f"unrecognized arguments: {' '.join(model_args)}")
        return _run_all_smokes(_absolute_path(str(args.output_dir), invocation_directory))

    if args.run_help and args.model_id is None:
        run_parser.print_help()
        run_parser.exit(0)
    if args.model_id is None:
        run_parser.error("the following arguments are required: model_id")
    model = get_model(args.model_id)
    if args.run_help:
        model_args.append("--help")
    if args.smoke_test and not model.smoke_test:
        parser.error(f"model {model.model_id!r} does not implement the common --smoke-test contract yet")
    model_args = _merge_default_model_args(model.default_model_args, model_args)
    model_args = _resolve_preset_resources(model_args)
    model_args = _absolutize_input_paths(model_args, model.input_path_options, invocation_directory)
    output_directory = _absolute_path(str(args.output_dir), invocation_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    result_path = output_directory / "smoke_result.json"
    if args.smoke_test:
        result_path.unlink(missing_ok=True)
        model_args.extend(("--smoke-test", "--output-dir", "."))
    previous_model_id = os.environ.get("SOFTROBOT_SMOKE_MODEL_ID")
    if args.smoke_test:
        os.environ["SOFTROBOT_SMOKE_MODEL_ID"] = model.model_id
    try:
        try:
            with _working_directory(output_directory):
                result = model.load_main()(model_args)
            if args.smoke_test:
                _validate_smoke_result(result_path, model.model_id)
        except Exception:
            if args.smoke_test:
                result_path.unlink(missing_ok=True)
            raise
    finally:
        if args.smoke_test:
            if previous_model_id is None:
                os.environ.pop("SOFTROBOT_SMOKE_MODEL_ID", None)
            else:
                os.environ["SOFTROBOT_SMOKE_MODEL_ID"] = previous_model_id
    return 0 if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
