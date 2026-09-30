"""
auditor.py — Core CircAudit Auditor
====================================
Main interface for circularity auditing of biomedical prediction methods.
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

from .framework.circularity import (
    CircularityType, FeatureInfo, KnowledgeExposure,
    CircularityScorer, CircularityInflation, CircularityReport,
)
from .framework.info_theory import (
    PPICeilingAnalyzer, MarginalInfoGainAnalyzer,
    FeatureInfoDecomposer, auroc_to_mi, mi_to_auroc,
)


class CircAudit:
    """
    CircAudit: Systematic circularity auditor for biomedical prediction.

    Workflow:
        1. CircAudit(task_name, feature_metadata)
        2. .classify_features(X, feature_names, knowledge_indicator)
        3. .compute_circularity(predictions, exposure_scores)
        4. .audit_method(method_name, y_true, y_pred, ...)
        5. .report(output="report.html")
    """

    def __init__(
        self,
        task_name: str,
        feature_metadata: Optional[Dict[str, Dict]] = None,
        positive_rate: float = 0.05,
        exposure_threshold: float = 0.3,
    ):
        """
        Parameters
        ----------
        task_name : name of the prediction task (e.g., "SL Prediction")
        feature_metadata : dict mapping feature_name -> {
            'source': str,
            'circ_type': str or None ('C1', 'C2', 'C3', 'C4'),
            'exposure_score': float (0-1, optional),
        }
        positive_rate : prevalence of positive class
        exposure_threshold : threshold for classifying features as circular
        """
        self.task_name = task_name
        self.positive_rate = positive_rate
        self.exposure_threshold = exposure_threshold

        # Convert metadata to FeatureInfo objects
        self.feature_metadata = {}
        if feature_metadata:
            for fname, meta in feature_metadata.items():
                circ_type = None
                if meta.get("circ_type"):
                    type_map = {
                        "C1": CircularityType.C1_LABEL_LEAKAGE,
                        "C2": CircularityType.C2_FEATURE_CONTAMINATION,
                        "C3": CircularityType.C3_EVALUATION_BIAS,
                        "C4": CircularityType.C4_SELECTION_BIAS,
                    }
                    circ_type = type_map.get(meta["circ_type"])

                self.feature_metadata[fname] = FeatureInfo(
                    name=fname,
                    source=meta.get("source", "unknown"),
                    circ_type=circ_type,
                    exposure_score=meta.get("exposure_score", 0.0),
                )

        # Initialize analyzers
        self.scorer = CircularityScorer(method="spearman")
        self.ceiling_analyzer = PPICeilingAnalyzer(positive_rate)
        self.gain_analyzer = MarginalInfoGainAnalyzer(positive_rate)
        self.decomposer = FeatureInfoDecomposer(positive_rate)

        # Storage
        self.feature_exposure_df: Optional[pd.DataFrame] = None
        self.feature_classification: Optional[Dict[str, List[str]]] = None
        self.audit_results: Dict[str, CircularityReport] = {}
        self.methods_summary: Optional[pd.DataFrame] = None

    def classify_features(
        self,
        X: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        knowledge_indicator: Optional[np.ndarray] = None,
    ) -> Dict[str, List[str]]:
        """
        Step 1: Classify features as circular vs non-circular.

        If X and knowledge_indicator provided, computes empirical exposure.
        Otherwise, uses predefined feature_metadata.
        """
        if X is not None and feature_names is not None and knowledge_indicator is not None:
            # Empirical computation
            results = []
            for j, fname in enumerate(feature_names):
                feature_vals = X[:, j]
                mask = ~(np.isnan(feature_vals) | np.isnan(knowledge_indicator.astype(float)))

                if mask.sum() < 10:
                    empirical_exposure = 0.0
                else:
                    from scipy.stats import spearmanr
                    rho, _ = spearmanr(
                        feature_vals[mask],
                        knowledge_indicator[mask].astype(float),
                    )
                    empirical_exposure = abs(rho) if not np.isnan(rho) else 0.0

                # Merge with predefined metadata
                if fname in self.feature_metadata:
                    info = self.feature_metadata[fname]
                    is_circ = info.is_circular or empirical_exposure > self.exposure_threshold
                    circ_type = info.circ_type
                    source = info.source
                else:
                    is_circ = empirical_exposure > self.exposure_threshold
                    circ_type = (
                        CircularityType.C2_FEATURE_CONTAMINATION if is_circ else None
                    )
                    source = "unknown"

                results.append({
                    "feature": fname,
                    "source": source,
                    "predefined_circular": (
                        self.feature_metadata[fname].is_circular
                        if fname in self.feature_metadata else False
                    ),
                    "empirical_exposure": empirical_exposure,
                    "is_circular": is_circ,
                    "circ_type": circ_type.value if circ_type else None,
                })

            self.feature_exposure_df = pd.DataFrame(results)
        else:
            # Use predefined metadata only
            results = []
            for fname, info in self.feature_metadata.items():
                results.append({
                    "feature": fname,
                    "source": info.source,
                    "predefined_circular": info.is_circular,
                    "empirical_exposure": info.exposure_score,
                    "is_circular": info.is_circular,
                    "circ_type": info.circ_type.value if info.circ_type else None,
                })
            self.feature_exposure_df = pd.DataFrame(results)

        # Classify
        self.feature_classification = {
            "non_circular": self.feature_exposure_df[
                ~self.feature_exposure_df["is_circular"]
            ]["feature"].tolist(),
            "circular": self.feature_exposure_df[
                self.feature_exposure_df["is_circular"]
            ]["feature"].tolist(),
            "C2_contaminated": self.feature_exposure_df[
                self.feature_exposure_df["circ_type"] == "C2_feature_contamination"
            ]["feature"].tolist(),
            "C4_biased": self.feature_exposure_df[
                self.feature_exposure_df["circ_type"] == "C4_selection_bias"
            ]["feature"].tolist(),
        }

        return self.feature_classification

    def compute_circularity(
        self,
        prediction_scores: np.ndarray,
        exposure_scores: np.ndarray,
    ) -> Dict[str, float]:
        """
        Step 2: Compute Circ(M, T, K) = ρ(S_M, E_K).
        """
        return self.scorer.compute(prediction_scores, exposure_scores)

    def audit_method(
        self,
        method_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        exposure_scores: Optional[np.ndarray] = None,
        y_pred_noncirc: Optional[np.ndarray] = None,
        auroc: Optional[float] = None,
        auprc: Optional[float] = None,
        # Circularity flags
        c1_label_leakage: bool = False,
        c2_feature_contamination: bool = False,
        c3_evaluation_bias: bool = False,
        c4_selection_bias: bool = False,
        # Evaluation rigor
        uses_nested_cv: bool = False,
        uses_temporal_split: bool = False,
        positive_source: str = "database",
        literature_coverage: Optional[float] = None,
        # Additional evidence
        evidence: Optional[Dict[str, str]] = None,
    ) -> CircularityReport:
        """
        Full audit of a single prediction method.
        """
        # Compute circularity score
        circ_score = {"rho": np.nan, "p_value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "is_significant": False}
        if exposure_scores is not None:
            circ_score = self.scorer.compute(y_pred, exposure_scores)
            # Auto-flag C2 if circ score is high
            if circ_score["rho"] > self.exposure_threshold:
                c2_feature_contamination = True

        # Compute AUROC
        if auroc is None:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(y_true, y_pred))

        # Compute inflation
        inflation = None
        if y_pred_noncirc is not None:
            inflator = CircularityInflation(metric="auroc")
            inflation = inflator.compute(y_true, y_pred, y_pred_noncirc)

        # Build report
        report = CircularityReport(
            method_name=method_name,
            circ_score=circ_score,
            inflation=inflation,
            c1_label_leakage=c1_label_leakage,
            c2_feature_contamination=c2_feature_contamination,
            c3_evaluation_bias=c3_evaluation_bias,
            c4_selection_bias=c4_selection_bias,
            auroc=auroc,
            auprc=auprc if auprc else 0.0,
            evidence=evidence or {},
        )

        self.audit_results[method_name] = report
        return report

    def audit_from_summary(
        self,
        method_name: str,
        auroc: float,
        circ_rho: float,
        c1: bool = False,
        c2: bool = False,
        c3: bool = False,
        c4: bool = False,
        auprc: float = None,
        evidence: Optional[Dict[str, str]] = None,
    ) -> CircularityReport:
        """
        Audit from summary statistics (no raw predictions needed).

        Useful for auditing published methods where only summary
        metrics are available.
        """
        report = CircularityReport(
            method_name=method_name,
            circ_score={
                "rho": circ_rho,
                "p_value": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "is_significant": False if (circ_rho is None or np.isnan(circ_rho)) else circ_rho > self.exposure_threshold,
            },
            c1_label_leakage=c1,
            c2_feature_contamination=c2,
            c3_evaluation_bias=c3,
            c4_selection_bias=c4,
            auroc=auroc,
            auprc=auprc if auprc else 0.0,
            evidence=evidence or {},
        )

        self.audit_results[method_name] = report
        return report

    def ceiling_analysis(
        self,
        auroc_ppi_only: float,
        method_name: str = None,
        auroc_method: float = None,
    ) -> Dict[str, Any]:
        """
        PPI Ceiling analysis for a method.
        """
        if method_name and method_name in self.audit_results:
            auroc_method = self.audit_results[method_name].auroc
        if auroc_method is None:
            raise ValueError("Must provide auroc_method or method_name")

        analysis = self.ceiling_analyzer.compute_ceiling(
            auroc_ppi_only, auroc_method, method_name or "unknown"
        )
        return {
            "auroc_ppi_only": auroc_ppi_only,
            "auroc_method": auroc_method,
            "mi_ppi_nats": analysis.mi_estimated,
            "auroc_ceiling": analysis.auroc_ceiling,
            "ceiling_gap": analysis.ceiling_gap,
            "is_above_ceiling": analysis.is_above_ceiling,
            "interpretation": analysis.interpretation,
        }

    def generate_summary(self) -> pd.DataFrame:
        """Generate summary DataFrame of all audited methods."""
        rows = []
        for name, report in self.audit_results.items():
            rows.append(report.to_dict())
        self.methods_summary = pd.DataFrame(rows)
        return self.methods_summary

    def save_results(self, output_dir: str) -> str:
        """Save audit results to files. Returns output directory path."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Summary
        summary_df = self.generate_summary()
        csv_path = os.path.join(output_dir, f"circ_audit_{self.task_name}_{timestamp}.csv")
        summary_df.to_csv(csv_path, index=False)

        # Feature analysis
        if self.feature_exposure_df is not None:
            feat_path = os.path.join(output_dir, f"feature_exposure_{timestamp}.csv")
            self.feature_exposure_df.to_csv(feat_path, index=False)

        # Detailed JSON
        detail = {}
        for name, report in self.audit_results.items():
            detail[name] = {
                "circ_rho": report.circ_score.get("rho", np.nan),
                "auroc": report.auroc,
                "severity": report.severity,
                "C1": report.c1_label_leakage,
                "C2": report.c2_feature_contamination,
                "C3": report.c3_evaluation_bias,
                "C4": report.c4_selection_bias,
                "evidence": report.evidence,
            }
        json_path = os.path.join(output_dir, f"circ_audit_{self.task_name}_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(detail, f, indent=2, default=str)

        print(f"Results saved to {output_dir}/")
        print(f"  Summary: {os.path.basename(csv_path)}")
        print(f"  Detail:  {os.path.basename(json_path)}")
        return output_dir
