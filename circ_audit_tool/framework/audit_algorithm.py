"""
audit_algorithm.py — Circularity Audit Algorithm
=================================================
Core algorithm for systematic circularity auditing of biomedical
prediction methods. Implements the 5-step audit procedure described
in the BIB Design Document, Section 4.2.
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .circularity import (
    CircularityType, FeatureInfo, KnowledgeExposure,
    CircularityScorer, CircularityInflation, CircularityReport,
    assess_circularity_batch,
)
from .info_theory import (
    PPICeilingAnalyzer, MarginalInfoGainAnalyzer,
    FeatureInfoDecomposer, auroc_to_mi,
)


# =============================================================================
# Step 1: Feature Exposure Analysis
# =============================================================================

class FeatureExposureAnalyzer:
    """
    Step 1: Analyze each feature's exposure to knowledge base K.

    For each feature x_j:
      a. Compute E_K(x_j) = correlation(x_j, exposure_to_K)
      b. Classify: if E_K > threshold → circular feature
    """

    def __init__(
        self,
        feature_metadata: Dict[str, FeatureInfo],
        exposure_threshold: float = 0.3,
    ):
        self.feature_metadata = feature_metadata
        self.exposure_threshold = exposure_threshold

    def compute_feature_exposure(
        self,
        X: np.ndarray,
        feature_names: List[str],
        knowledge_indicator: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compute per-feature exposure to knowledge base.

        Parameters
        ----------
        X : feature matrix (n_samples, n_features)
        feature_names : list of feature names
        knowledge_indicator : binary array, 1 = sample is in knowledge base K

        Returns
        -------
        DataFrame with columns: feature, source, exposure, is_circular, circ_type
        """
        results = []
        for j, fname in enumerate(feature_names):
            # Compute correlation between feature and knowledge indicator
            feature_vals = X[:, j]

            # Remove NaN
            mask = ~(np.isnan(feature_vals) | np.isnan(knowledge_indicator.astype(float)))
            if mask.sum() < 10:
                exposure = 0.0
            else:
                from scipy.stats import spearmanr
                rho, p = spearmanr(feature_vals[mask], knowledge_indicator[mask].astype(float))
                exposure = abs(rho) if not np.isnan(rho) else 0.0

            # Look up metadata
            info = self.feature_metadata.get(fname, FeatureInfo(name=fname, source="unknown"))
            is_circ = info.is_circular or exposure > self.exposure_threshold
            circ_type = info.circ_type.value if info.circ_type else (
                "C2_feature_contamination" if is_circ else None
            )

            results.append({
                "feature": fname,
                "source": info.source,
                "exposure_score": float(exposure),
                "is_circular": is_circ,
                "circ_type": circ_type,
                "predefined_circular": info.is_circular,
            })

        return pd.DataFrame(results)

    def classify_features(
        self, exposure_df: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """Classify features into circular categories."""
        return {
            "non_circular": exposure_df[~exposure_df["is_circular"]]["feature"].tolist(),
            "circular": exposure_df[exposure_df["is_circular"]]["feature"].tolist(),
            "C2_contaminated": exposure_df[
                exposure_df["circ_type"] == "C2_feature_contamination"
            ]["feature"].tolist(),
            "C4_biased": exposure_df[
                exposure_df["circ_type"] == "C4_selection_bias"
            ]["feature"].tolist(),
        }


# =============================================================================
# Step 2: Label Independence Check
# =============================================================================

class LabelIndependenceChecker:
    """
    Step 2: Check if training labels are independent of test set.

    C1 (Label Leakage): P(Y_test ∈ K_train) > 0
    """

    def __init__(self):
        self.results = {}

    def check_label_leakage(
        self,
        train_labels: np.ndarray,
        test_labels: np.ndarray,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        knowledge_indices: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Check for label leakage between train and test sets.

        A method has C1 label leakage if test positive labels are
        directly derivable from training data or knowledge base.
        """
        # Overlap between train and test indices
        train_set = set(train_indices)
        test_set = set(test_indices)
        index_overlap = train_set & test_set
        index_overlap_rate = len(index_overlap) / len(test_set) if test_set else 0

        # Positive label overlap: fraction of test positives seen in train
        test_pos = set(np.where(test_labels == 1)[0])
        train_pos = set(np.where(train_labels == 1)[0])
        if len(test_pos) > 0:
            label_overlap = len(test_pos & train_pos) / len(test_pos)
        else:
            label_overlap = 0.0

        # Knowledge base overlap
        kb_overlap = 0.0
        if knowledge_indices is not None:
            kb_set = set(knowledge_indices)
            if len(test_pos) > 0:
                kb_overlap = len(test_pos & kb_set) / len(test_pos)

        has_leakage = index_overlap_rate > 0 or label_overlap > 0.9

        return {
            "index_overlap_rate": float(index_overlap_rate),
            "label_overlap_rate": float(label_overlap),
            "kb_overlap_rate": float(kb_overlap),
            "has_C1_leakage": has_leakage,
            "severity": "severe" if index_overlap_rate > 0 else (
                "moderate" if label_overlap > 0.9 else "clean"
            ),
        }


# =============================================================================
# Step 3: Performance Decomposition
# =============================================================================

class PerformanceDecomposer:
    """
    Step 3: Decompose method performance into genuine vs circular.

    a. Train M_full (all features)
    b. Train M_nc (non-circular features only)
    c. Δ_perf = Perf(M_full) - Perf(M_nc)
    d. Circ(M) = ρ(S_M_full, E_K)
    """

    def __init__(self, metric: str = "auroc"):
        self.metric = metric

    def decompose(
        self,
        y_true: np.ndarray,
        y_pred_full: np.ndarray,
        y_pred_noncirc: np.ndarray,
        exposure_scores: np.ndarray,
        y_pred_full_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Decompose performance and compute circularity.
        """
        from sklearn.metrics import roc_auc_score

        perf_full = roc_auc_score(y_true, y_pred_full)
        perf_nc = roc_auc_score(y_true, y_pred_noncirc)

        # Circularity score
        scorer = CircularityScorer(method="spearman")
        scores_for_circ = y_pred_full_scores if y_pred_full_scores is not None else y_pred_full
        circ_result = scorer.compute(scores_for_circ, exposure_scores)

        # Inflation
        inflator = CircularityInflation(metric=self.metric)
        inflation = inflator.compute(y_true, y_pred_full, y_pred_noncirc)

        return {
            "perf_full": perf_full,
            "perf_noncirc": perf_nc,
            "delta_perf": perf_full - perf_nc,
            "circ_score": circ_result,
            "inflation": inflation,
            "genuine_fraction": perf_nc / perf_full if perf_full > 0 else 1.0,
        }


# =============================================================================
# Step 4: Evaluation Rigor Check
# =============================================================================

class EvaluationRigorChecker:
    """
    Step 4: Check evaluation protocol rigor.

    C3 (Evaluation Bias):
      a. Is nested CV used?
      b. Is test set temporally independent?
    C4 (Selection Bias):
      c. Is positive set biased by literature?
    """

    def check(
        self,
        uses_nested_cv: bool = False,
        uses_temporal_split: bool = False,
        positive_source: str = "database",
        literature_coverage: Optional[float] = None,
        gene_degree_skew: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Check evaluation rigor and flag potential C3/C4 issues.

        Parameters
        ----------
        uses_nested_cv : whether nested cross-validation was used
        uses_temporal_split : whether temporal train/test split was used
        positive_source : source of positive labels ("database", "literature", "experiment")
        literature_coverage : fraction of positive set covered by literature
        gene_degree_skew : skewness of gene degree distribution in positive set
        """
        c3_flags = []
        c4_flags = []

        if not uses_nested_cv:
            c3_flags.append("No nested CV — potential data leakage in hyperparameter tuning")

        if not uses_temporal_split:
            c3_flags.append("No temporal split — test set may not be truly independent")

        if positive_source in ("database", "literature"):
            c4_flags.append(f"Positive labels from {positive_source} — literature bias possible")

        if literature_coverage is not None and literature_coverage > 0.8:
            c4_flags.append(
                f"High literature coverage ({literature_coverage:.1%}) — "
                "positive set likely literature-biased"
            )

        if gene_degree_skew is not None and gene_degree_skew > 2.0:
            c4_flags.append(
                f"High degree skew ({gene_degree_skew:.1f}) — "
                "well-studied genes over-represented"
            )

        has_c3 = len(c3_flags) > 0
        has_c4 = len(c4_flags) > 0

        return {
            "C3_evaluation_bias": has_c3,
            "C3_flags": c3_flags,
            "C4_selection_bias": has_c4,
            "C4_flags": c4_flags,
            "uses_nested_cv": uses_nested_cv,
            "uses_temporal_split": uses_temporal_split,
            "positive_source": positive_source,
        }


# =============================================================================
# Full Audit Pipeline
# =============================================================================

class CircularityAuditPipeline:
    """
    Complete 5-step circularity audit pipeline.

    Combines Steps 1-4 and produces a comprehensive CircularityReport.
    """

    def __init__(
        self,
        task_name: str,
        feature_metadata: Dict[str, FeatureInfo],
        exposure_threshold: float = 0.3,
        metric: str = "auroc",
    ):
        self.task_name = task_name
        self.feature_metadata = feature_metadata
        self.exposure_threshold = exposure_threshold
        self.metric = metric

        # Initialize components
        self.exposure_analyzer = FeatureExposureAnalyzer(
            feature_metadata, exposure_threshold
        )
        self.label_checker = LabelIndependenceChecker()
        self.perf_decomposer = PerformanceDecomposer(metric)
        self.rigor_checker = EvaluationRigorChecker()

        # Storage
        self.audit_results: Dict[str, CircularityReport] = {}

    def audit_method(
        self,
        method_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        y_pred_noncirc: Optional[np.ndarray] = None,
        exposure_scores: Optional[np.ndarray] = None,
        knowledge_indicator: Optional[np.ndarray] = None,
        # Evaluation rigor parameters
        uses_nested_cv: bool = False,
        uses_temporal_split: bool = False,
        positive_source: str = "database",
        literature_coverage: Optional[float] = None,
        # Label independence parameters
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        knowledge_indices: Optional[np.ndarray] = None,
        train_labels: Optional[np.ndarray] = None,
        test_labels: Optional[np.ndarray] = None,
        # Performance
        auroc: Optional[float] = None,
        auprc: Optional[float] = None,
    ) -> CircularityReport:
        """
        Run full 5-step audit on a single method.
        """
        # Step 1: Feature exposure
        feature_exposure_df = None
        feature_classification = None
        if X is not None and feature_names is not None and knowledge_indicator is not None:
            feature_exposure_df = self.exposure_analyzer.compute_feature_exposure(
                X, feature_names, knowledge_indicator
            )
            feature_classification = self.exposure_analyzer.classify_features(
                feature_exposure_df
            )

        # Step 2: Label independence
        label_check = {}
        c1_flag = False
        if train_indices is not None and test_indices is not None:
            label_check = self.label_checker.check_label_leakage(
                train_labels if train_labels is not None else y_true,
                test_labels if test_labels is not None else y_true,
                train_indices, test_indices, knowledge_indices,
            )
            c1_flag = label_check.get("has_C1_leakage", False)

        # Step 3: Performance decomposition
        perf_decomp = {}
        inflation = None
        circ_score = {"rho": np.nan, "p_value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "is_significant": False}
        c2_flag = False

        if exposure_scores is not None:
            scorer = CircularityScorer(method="spearman")
            circ_score = scorer.compute(y_pred, exposure_scores)
            c2_flag = circ_score.get("rho", 0) > self.exposure_threshold

        if y_pred_noncirc is not None:
            perf_decomp = self.perf_decomposer.decompose(
                y_true, y_pred, y_pred_noncirc, exposure_scores if exposure_scores is not None else np.zeros_like(y_pred)
            )
            inflation = perf_decomp.get("inflation")

        # Step 4: Evaluation rigor
        rigor = self.rigor_checker.check(
            uses_nested_cv=uses_nested_cv,
            uses_temporal_split=uses_temporal_split,
            positive_source=positive_source,
            literature_coverage=literature_coverage,
        )
        c3_flag = rigor.get("C3_evaluation_bias", False)
        c4_flag = rigor.get("C4_selection_bias", False)

        # Compute AUROC if not provided
        if auroc is None:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(y_true, y_pred))

        # Build report
        report = CircularityReport(
            method_name=method_name,
            circ_score=circ_score,
            inflation=inflation,
            c1_label_leakage=c1_flag,
            c2_feature_contamination=c2_flag,
            c3_evaluation_bias=c3_flag,
            c4_selection_bias=c4_flag,
            auroc=auroc,
            auprc=auprc if auprc else 0.0,
            evidence={
                "label_check": str(label_check),
                "rigor_check": str(rigor),
                "feature_classification": str(feature_classification) if feature_classification else "",
            },
        )

        self.audit_results[method_name] = report
        return report

    def generate_summary(self) -> pd.DataFrame:
        """Generate summary DataFrame of all audited methods."""
        rows = []
        for name, report in self.audit_results.items():
            rows.append(report.to_dict())
        return pd.DataFrame(rows)

    def save_results(self, output_dir: str) -> None:
        """Save audit results to JSON and CSV."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Summary DataFrame
        summary_df = self.generate_summary()
        summary_df.to_csv(
            os.path.join(output_dir, f"circ_audit_summary_{timestamp}.csv"),
            index=False,
        )

        # Detailed JSON
        detailed = {}
        for name, report in self.audit_results.items():
            detailed[name] = {
                "method": name,
                "circularity_score": report.circ_score,
                "inflation": report.inflation,
                "C1_label_leakage": report.c1_label_leakage,
                "C2_feature_contamination": report.c2_feature_contamination,
                "C3_evaluation_bias": report.c3_evaluation_bias,
                "C4_selection_bias": report.c4_selection_bias,
                "severity": report.severity,
                "is_clean": report.is_clean,
                "evidence": report.evidence,
            }

        with open(
            os.path.join(output_dir, f"circ_audit_detail_{timestamp}.json"), "w"
        ) as f:
            json.dump(detailed, f, indent=2, default=str)

        print(f"Audit results saved to {output_dir}/")
        print(f"  Summary: circ_audit_summary_{timestamp}.csv")
        print(f"  Detail:  circ_audit_detail_{timestamp}.json")
