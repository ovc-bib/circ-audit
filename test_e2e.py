"""
test_e2e.py — End-to-End Test for CircAudit Tool
==================================================
Validates the complete pipeline from initialization to report generation.
"""

import numpy as np
import pandas as pd
import os
import sys
import tempfile
import shutil

# Ensure package root is on path for editable installs or direct testing
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def test_framework_definitions():
    """Test circularity definitions (Def 1-3)."""
    from circ_audit_tool.framework.circularity import (
        CircularityType, FeatureInfo, KnowledgeExposure,
        CircularityScorer, CircularityInflation,
    )

    # Test CircularityType
    assert CircularityType.C2_FEATURE_CONTAMINATION.description != ""
    assert CircularityType.C1_LABEL_LEAKAGE.short_name == "LABEL_LEAKAGE"
    print("  [PASS] CircularityType enum")

    # Test CircularityScorer with fixed seed for reproducibility
    scorer = CircularityScorer(method="spearman")
    rng = np.random.RandomState(42)
    scores = rng.rand(100)
    exposure = scores * 0.8 + rng.rand(100) * 0.2  # strongly correlated
    result = scorer.compute(scores, exposure)
    assert result["rho"] > 0.3, f"Expected rho > 0.3, got {result['rho']}"
    print("  [PASS] CircularityScorer (correlated)")

    # Test with uncorrelated data (use new RNG state)
    rng2 = np.random.RandomState(123)
    scores2 = rng2.rand(100)
    exposure2 = rng2.rand(100)
    result2 = scorer.compute(scores2, exposure2)
    assert abs(result2["rho"]) < 0.4, f"Expected |rho| < 0.4, got {result2['rho']}"
    print("  [PASS] CircularityScorer (uncorrelated)")

    # Test CircularityInflation
    rng3 = np.random.RandomState(456)
    y_true = np.array([0]*50 + [1]*50)
    y_all = np.concatenate([rng3.rand(50)*0.3, rng3.rand(50)*0.7+0.3])
    y_nc = np.concatenate([rng3.rand(50)*0.4, rng3.rand(50)*0.6+0.3])
    inflator = CircularityInflation(metric="auroc")
    inflation = inflator.compute(y_true, y_all, y_nc)
    assert "delta" in inflation
    assert "inflation_ratio" in inflation
    print("  [PASS] CircularityInflation")


def test_info_theory():
    """Test information-theoretic analysis."""
    from circ_audit_tool.framework.info_theory import auroc_to_mi, mi_to_auroc, PPICeilingAnalyzer, MarginalInfoGainAnalyzer

    # Test AUROC <-> MI conversion
    mi = auroc_to_mi(0.9)
    auroc_back = mi_to_auroc(mi)
    assert abs(auroc_back - 0.9) < 0.01, f"Roundtrip failed: {auroc_back}"
    print("  [PASS] AUROC <-> MI conversion")

    # Test PPI Ceiling
    analyzer = PPICeilingAnalyzer(positive_rate=0.05)
    result = analyzer.compute_ceiling(0.891, 0.917, "test")
    assert hasattr(result, "ceiling_gap")
    assert hasattr(result, "is_above_ceiling")
    print("  [PASS] PPICeilingAnalyzer")

    # Test MarginalInfoGain
    gain_analyzer = MarginalInfoGainAnalyzer()
    gain = gain_analyzer.compute_gain(
        auroc_base=0.891, auroc_combined=0.917,
        base_source="PPI", added_source="CRISPR",
        circ_score=0.068,
    )
    assert gain.delta_auroc > 0
    assert gain.is_genuine  # low circ score → genuine
    print("  [PASS] MarginalInfoGainAnalyzer")


def test_audit_pipeline():
    """Test the complete audit pipeline."""
    from circ_audit_tool import CircAudit

    # Define features
    feature_metadata = {
        "crispr_score": {"source": "DepMap", "circ_type": None, "exposure_score": 0.03},
        "ppi_score": {"source": "STRING", "circ_type": "C2", "exposure_score": 0.52},
        "lit_score": {"source": "PubMed", "circ_type": "C2", "exposure_score": 0.80},
        "go_sim": {"source": "GO", "circ_type": "C4", "exposure_score": 0.25},
    }

    audit = CircAudit(
        task_name="Test_Task",
        feature_metadata=feature_metadata,
        exposure_threshold=0.3,
    )

    # Classify features
    classification = audit.classify_features()
    assert "crispr_score" in classification["non_circular"]
    assert "ppi_score" in classification["circular"]
    assert "lit_score" in classification["circular"]
    print("  [PASS] Feature classification")

    # Audit from summary
    report = audit.audit_from_summary(
        method_name="TestMethod",
        auroc=0.95,
        circ_rho=0.6,
        c1=False, c2=True, c3=False, c4=True,
        evidence={"C2": "Uses PPI features"},
    )
    assert report.severity in ("severe", "moderate")  # rho=0.6 triggers severe
    assert not report.is_clean
    assert report.n_circularity_types >= 1
    print("  [PASS] Audit from summary")

    # Audit a clean method
    report2 = audit.audit_from_summary(
        method_name="CleanMethod",
        auroc=0.85,
        circ_rho=0.05,
        c1=False, c2=False, c3=False, c4=False,
    )
    assert report2.severity == "clean"
    assert report2.is_clean
    print("  [PASS] Clean method audit")

    # Generate summary
    summary = audit.generate_summary()
    assert len(summary) == 2
    assert "TestMethod" in summary["method"].values
    assert "CleanMethod" in summary["method"].values
    print("  [PASS] Summary generation")

    # Save results
    tmpdir = tempfile.mkdtemp()
    try:
        audit.save_results(tmpdir)
        files = os.listdir(tmpdir)
        assert any(f.endswith(".csv") for f in files)
        assert any(f.endswith(".json") for f in files)
        print("  [PASS] Save results")
    finally:
        shutil.rmtree(tmpdir)


def test_report_generator():
    """Test report generation."""
    from circ_audit_tool import CircAudit
    from circ_audit_tool import ReportGenerator

    feature_metadata = {
        "ppi_score": {"source": "STRING", "circ_type": "C2", "exposure_score": 0.5},
        "crispr": {"source": "DepMap", "circ_type": None, "exposure_score": 0.03},
    }
    audit = CircAudit(task_name="Test", feature_metadata=feature_metadata)
    audit.audit_from_summary("MethodA", 0.95, 0.6, c2=True)
    audit.audit_from_summary("MethodB", 0.85, 0.05)

    # Text report
    tmpdir = tempfile.mkdtemp()
    try:
        reporter = ReportGenerator(audit)
        txt_path = os.path.join(tmpdir, "report.txt")
        reporter.generate_text_report(output_path=txt_path, task_name="Test")
        assert os.path.exists(txt_path)
        with open(txt_path, encoding="utf-8") as f:
            content = f.read()
        assert "MethodA" in content
        assert "MethodB" in content
        print("  [PASS] Text report")

        # HTML report
        html_path = os.path.join(tmpdir, "report.html")
        reporter.generate_html_report(output_path=html_path, task_name="Test")
        assert os.path.exists(html_path)
        with open(html_path, encoding="utf-8") as f:
            content = f.read()
        assert "CircAudit" in content
        assert "severe" in content
        assert "clean" in content
        print("  [PASS] HTML report")
    finally:
        shutil.rmtree(tmpdir)


def test_visualizer():
    """Test figure generation."""
    from circ_audit_tool import CircAuditVisualizer

    tmpdir = tempfile.mkdtemp()
    try:
        viz = CircAuditVisualizer(output_dir=tmpdir)

        # Test circ vs auroc plot
        df = pd.DataFrame({
            "method": ["A", "B", "C"],
            "auroc": [0.95, 0.85, 0.80],
            "circ_rho": [0.6, 0.2, 0.05],
            "severity": ["severe", "mild", "clean"],
        })
        viz.plot_circ_vs_auroc(df, output_name="test_fig", format="png")
        assert os.path.exists(os.path.join(tmpdir, "test_fig.png"))
        print("  [PASS] Circ vs AUROC plot")

        # Test heatmap
        hm = pd.DataFrame({
            "method": ["A", "B"],
            "PPI": [0.5, 0.1],
            "CRISPR": [0.0, 0.1],
        })
        viz.plot_feature_heatmap(hm, output_name="test_heatmap", format="png")
        assert os.path.exists(os.path.join(tmpdir, "test_heatmap.png"))
        print("  [PASS] Feature heatmap")

    finally:
        shutil.rmtree(tmpdir)


def main():
    print("=" * 60)
    print("CircAudit End-to-End Test Suite")
    print("=" * 60)

    tests = [
        ("Framework Definitions", test_framework_definitions),
        ("Information Theory", test_info_theory),
        ("Audit Pipeline", test_audit_pipeline),
        ("Report Generator", test_report_generator),
        ("Visualizer", test_visualizer),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
            print(f"  >> {name}: ALL PASSED")
        except Exception as e:
            failed += 1
            print(f"  >> {name}: FAILED - {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
