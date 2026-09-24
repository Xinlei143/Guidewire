from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _delivery_evidence() -> str:
    return "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in ("README.md", "docs/environment.md", "requirements-cpu.lock")
    )


def test_cuda_evidence_describes_gate_without_claiming_no_availability_query() -> None:
    evidence = _delivery_evidence()
    for stale in (
        "CUDA availability and execution were not queried or validated",
        "没有调用 CUDA 检查",
        "未调用或验收 CUDA",
    ):
        assert stale not in evidence
    for required in (
        "torch.cuda.is_available()",
        "availability gate",
        "CUDA-only",
        "没有执行或验收 CUDA 算例",
        "所有实际模型与 smoke 显式使用 CPU",
        "skip 不代表 CPU 功能缺陷",
        "CPU-only wheel 未单独验收",
    ):
        assert required in evidence


def test_readme_and_lineage_record_model_contact_and_xz_evidence() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    lineage = (ROOT / "docs/model_lineage.md").read_text(encoding="utf-8")
    combined = readme + "\n" + lineage
    assert "19 个 Model ID 各自对应唯一模块入口" in readme
    for module in (
        "softrobot.curved_tube.legacy_arc_3d_analytic",
        "softrobot.curved_tube.legacy_arc_3d_mesh",
        "softrobot.curved_tube.v4_analytic",
        "softrobot.curved_tube.v4_mesh",
    ):
        assert module in readme
        assert module in lineage
    for evidence in (
        "measured active PI contact",
        "updated integral state",
        "positive integral state/force contribution",
        "arc90_L1p5m_xz_centerline_800.csv",
        "y=0",
        "xz",
    ):
        assert evidence in combined
