"""
CircAudit — Circularity Audit Tool for Biomedical Prediction
=============================================================
Open-source tool for systematic circularity auditing.

Usage:
    from circ_audit_tool import CircAudit

    audit = CircAudit(
        task_name="SL Prediction",
        feature_metadata={...},
    )
    audit.classify_features(X, feature_names, knowledge_indicator)
    circ_scores = audit.compute_circularity(predictions, exposure)
    report = audit.audit_method("SINaTRA", y_true, y_pred, ...)
    audit.report(output="report.html")
"""

__version__ = "0.1.0"

from .auditor import CircAudit
from .visualize import CircAuditVisualizer
from .report import ReportGenerator

__all__ = ["CircAudit", "CircAuditVisualizer", "ReportGenerator"]
