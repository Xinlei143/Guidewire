from __future__ import annotations

import subprocess
from pathlib import Path


def test_teacher_release_documentation_is_not_gitignored() -> None:
    repo = Path(__file__).resolve().parents[1]
    for relative in ("README.md", "docs/environment.md", "docs/model_lineage.md"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", relative], cwd=repo, check=False,
        )
        assert result.returncode == 1, f"release documentation is ignored: {relative}"
