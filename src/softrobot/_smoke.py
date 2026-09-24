"""Shared output and isolated runtime contracts for historical-model smokes."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

import numpy as np


@contextmanager
def cpu_smoke_runtime() -> Iterator[None]:
    """Temporarily select deterministic CPU/Agg execution and restore all state."""
    import matplotlib

    missing = object()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", missing)
    numpy_state = np.random.get_state()
    matplotlib_backend = matplotlib.get_backend()
    try:
        import torch
    except ImportError:
        torch = None
        torch_state = None
        torch_default_device = None
    else:
        torch_generator = torch.random.default_generator
        torch_state = torch_generator.get_state().clone()
        torch_default_device = torch.get_default_device()

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        np.random.seed(0)
        matplotlib.use("Agg", force=True)
        if torch is not None:
            torch_generator.manual_seed(0)
            torch.set_default_device("cpu")
        yield
    finally:
        try:
            if torch is not None:
                torch.set_default_device(torch_default_device)
                torch_generator.set_state(torch_state)
        finally:
            try:
                np.random.set_state(numpy_state)
            finally:
                try:
                    matplotlib.use(matplotlib_backend, force=True)
                finally:
                    if cuda_visible_devices is missing:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    """Atomically replace a JSON document using a temporary sibling file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_smoke_result(
    output_dir: str | Path,
    model_id: str,
    state: Any,
    *,
    steps: int,
    **details: Any,
) -> Path:
    values = np.asarray(state)
    finite = bool(values.size and np.all(np.isfinite(values)))
    payload = {
        "model_id": model_id,
        "status": "passed" if finite else "failed",
        "finite": finite,
        "steps": int(steps),
        "state_shape": list(values.shape),
        **details,
    }
    result_path = write_json_atomic(Path(output_dir) / "smoke_result.json", payload)
    if not finite:
        raise RuntimeError(f"{model_id} smoke produced non-finite state")
    return result_path
