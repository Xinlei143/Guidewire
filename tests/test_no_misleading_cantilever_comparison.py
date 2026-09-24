"""The teacher release must not carry the mismatched cantilever comparison."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mismatched_comparison_artifacts_are_absent():
    for relative in (
        "tools/validate_cantilever_displacement.py",
        "validation/cosserat/CantileverRodStretchBendCompare.m",
        "docs/cantilever_validation.md",
        "tests/test_cantilever_displacement_validation.py",
        "tests/test_cantilever_release_provenance.py",
    ):
        assert not (ROOT / relative).exists(), relative


def test_mismatched_comparison_results_are_not_claimed():
    for relative in ("README.md", "docs/model_lineage.md"):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "docs/cantilever_validation.md" not in content
        assert "CantileverRodStretchBendCompare.m" not in content
