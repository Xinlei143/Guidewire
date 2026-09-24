"""Release-level provenance and documentation contract for cantilever validation."""

import hashlib
from pathlib import Path

from softrobot.registry import model_ids


ROOT = Path(__file__).resolve().parents[1]
MATLAB = ROOT / "validation/cosserat/CantileverRodStretchBendCompare.m"


def test_matched_matlab_source_is_exact_and_no_third_party_example():
    assert hashlib.sha256(MATLAB.read_bytes()).hexdigest() == (
        "19f2c1d7799df500ab79cdec131a88f6dac14d76ba47d4ba45303eaa1e5a36ff"
    )
    assert not list(ROOT.rglob("untitled.m"))
    assert not list(ROOT.rglob("soft_robot_chain_bending_stretching_torsion.py"))


def test_validation_documented_without_new_model_id():
    assert len(model_ids()) == 19
    assert "foundation-cantilever-der" in model_ids()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    validation = (ROOT / "docs/cantilever_validation.md").read_text(encoding="utf-8")
    lineage = (ROOT / "docs/model_lineage.md").read_text(encoding="utf-8")
    assert "docs/cantilever_validation.md" in readme
    assert "L-BFGS-B" in validation
    assert "0.437051073" in validation
    assert "0.425090719" in validation
    assert "19f2c1d7799df500ab79cdec131a88f6dac14d76ba47d4ba45303eaa1e5a36ff" in lineage
    assert "a71dfce3662e99fd462cfc857b5b4a0b394274ae694b5166267a8193eee00b6d" in lineage
