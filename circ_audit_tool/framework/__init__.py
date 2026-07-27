"""
Circularity-Aware Biomedical Prediction Framework
==================================================
Core framework for detecting and quantifying knowledge leakage
in computational biomedical prediction tasks.

Modules:
    circularity  — Formal definitions (Def 1-3) and circularity taxonomy
    info_theory  — Information-theoretic analysis (PPI Ceiling Theorem)
    audit_algorithm — Complete audit pipeline (5-step procedure)
"""

from .circularity import (
    CircularityType,
    FeatureInfo,
    KnowledgeExposure,
    CircularityScorer,
    CircularityInflation,
    CircularityReport,
    assess_circularity_batch,
)

from .info_theory import (
    auroc_to_mi,
    mi_to_auroc,
    PPICeilingAnalyzer,
    CeilingAnalysis,
    MarginalInfoGainAnalyzer,
    MarginalInfoGain,
    FeatureInfoDecomposer,
)

from .audit_algorithm import (
    FeatureExposureAnalyzer,
    LabelIndependenceChecker,
    PerformanceDecomposer,
    EvaluationRigorChecker,
    CircularityAuditPipeline,
)

__version__ = "0.1.0"
__all__ = [
    "CircularityType", "FeatureInfo", "KnowledgeExposure",
    "CircularityScorer", "CircularityInflation", "CircularityReport",
    "assess_circularity_batch",
    "auroc_to_mi", "mi_to_auroc",
    "PPICeilingAnalyzer", "CeilingAnalysis",
    "MarginalInfoGainAnalyzer", "MarginalInfoGain",
    "FeatureInfoDecomposer",
    "FeatureExposureAnalyzer", "LabelIndependenceChecker",
    "PerformanceDecomposer", "EvaluationRigorChecker",
    "CircularityAuditPipeline",
]
