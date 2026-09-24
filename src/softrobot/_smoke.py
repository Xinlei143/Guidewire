"""Shared output contract for small, deterministic historical-model smokes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def configure_cpu_runtime() -> None:
    """Make a smoke deterministic, headless, and CPU-only."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    np.random.seed(0)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(0)
    torch.set_default_device("cpu")


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
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "smoke_result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not finite:
        raise RuntimeError(f"{model_id} smoke produced non-finite state")
    return result_path
