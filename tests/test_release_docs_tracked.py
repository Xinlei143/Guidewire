from __future__ import annotations

import subprocess
from pathlib import Path


def test_teacher_release_documentation_is_not_gitignored() -> None:
    repo = Path(__file__).resolve().parents[1]
    for relative in ("README.md", "docs/environment.md", "docs/model_lineage.md"):
        assert (repo / relative).is_file()
        result = subprocess.run(
            ["git", "check-ignore", "-q", relative], cwd=repo, check=False,
            stderr=subprocess.DEVNULL,
        )
        if (repo / ".git").exists():
            assert result.returncode == 1, f"release documentation is ignored: {relative}"
        else:
            assert result.returncode == 128
