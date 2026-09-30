"""
circularity.py — Circularity-Aware Biomedical Prediction Framework
================================================================
Formal definitions (Definition 1-3) and four-type circularity taxonomy.

References:
    BIB Design Document, Section 2.1-2.2

Definitions:
    Def 1: Knowledge Exposure  E_K(x) — how much knowledge base K covers x
    Def 2: Circularity         Circ(M, T, K) — correlation of method scores with exposure
    Def 3: Circularity Inflation Δ_perf — performance gain from circular features
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Enum: Circularity Type
# =============================================================================
class CircularityType(Enum):
    """Four types of knowledge leakage in biomedical prediction."""
    C1_LABEL_LEAKAGE = "C1_label_leakage"
    C2_FEATURE_CONTAMINATION = "C2_feature_contamination"
    C3_EVALUATION_BIAS = "C3_evaluation_bias"
    C4_SELECTION_BIAS = "C4_selection_bias"

    @property
    def description(self) -> str:
        descriptions = {
            CircularityType.C1_LABEL_LEAKAGE:
                "Training labels derived from K; test set not independent",
            CircularityType.C2_FEATURE_CONTAMINATION:
                "Features directly encode known associations from K",
            CircularityType.C3_EVALUATION_BIAS:
                "Test set not independent of training/feature construction",
            CircularityType.C4_SELECTION_BIAS:
                "Positive samples from K; literature bias = label bias",
        }
        return descriptions[self]

    @property
    def short_name(self) -> str:
        return self.name.split("_", 1)[1]


# =============================================================================
# Data Class: Feature Metadata
# =============================================================================
@dataclass
class FeatureInfo:
    """Metadata for a single feature regarding its circularity properties."""
    name: str
    source: str  # e.g., "PPI_database", "PubMed", "DepMap"
    circ_type: Optional[CircularityType] = None
    is_circular: bool = False
    exposure_score: float = 0.0  # E_K for this feature

    def __post_init__(self):
        if self.circ_type is not None:
            self.is_circular = True


# =============================================================================
# Definition 1: Knowledge Exposure
# =============================================================================
class KnowledgeExposure:
    """
    Definition 1: Knowledge Exposure E_K(x)

    Quantifies how much of the information in sample x is already
    accessible through knowledge base K.

    Three sub-measures:
        E_K^feat   = |{j: x_j derived from K}| / p   (feature exposure)
        E_K^label  = P(y=1 | known in K)              (label exposure)
        E_K^eval   = |D_test ∩ K| / |D_test|          (evaluation exposure)
    """

    def __init__(
        self,
        feature_metadata: Dict[str, FeatureInfo],
        knowledge_labels: Optional[np.ndarray] = None,
    ):
        self.feature_metadata = feature_metadata
        self.knowledge_labels = knowledge_labels

    def feature_exposure(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Compute per-sample feature exposure E_K^feat(x_i).

        For each sample, fraction of features that are derived from K.
        Weighted by each feature's exposure_score if available.
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        exposure = np.zeros(n_samples)

        for j, fname in enumerate(feature_names):
            if fname in self.feature_metadata:
                info = self.feature_metadata[fname]
                if info.is_circular:
                    weight = info.exposure_score if info.exposure_score > 0 else 1.0
                    # Contribution: feature value * weight (nonzero = exposed)
                    contribution = (X[:, j] != 0).astype(float) * weight
                    exposure += contribution

        # Normalize by total possible circular feature weight
        total_circ_weight = sum(
            (info.exposure_score if info.exposure_score > 0 else 1.0)
            for info in self.feature_metadata.values()
            if info.is_circular
        )
        if total_circ_weight > 0:
            exposure /= total_circ_weight

        return exposure

    def label_exposure(self, y: np.ndarray) -> float:
        """
        Compute label exposure E_K^label = P(y=1 | known in K).
        Fraction of positive labels that are already in knowledge base.
        """
        if self.knowledge_labels is None:
            return 0.0
        # Known positives in K that are also positive in y
        known_pos = (self.knowledge_labels == 1)
        label_pos = (y == 1)
        if label_pos.sum() == 0:
            return 0.0
        return np.sum(known_pos & label_pos) / label_pos.sum()

    def evaluation_exposure(
        self, test_indices: np.ndarray, knowledge_indices: np.ndarray
    ) -> float:
        """
        Compute evaluation exposure E_K^eval = |D_test ∩ K| / |D_test|.
        Fraction of test samples that appear in knowledge base.
        """
        if len(test_indices) == 0:
            return 0.0
        overlap = np.intersect1d(test_indices, knowledge_indices)
        return len(overlap) / len(test_indices)


# =============================================================================
# Definition 2: Circularity Score
# =============================================================================
class CircularityScorer:
    """
    Definition 2: Circularity Circ(M, T, K)

    Measures the correlation between a method's prediction scores and
    the knowledge exposure. High correlation = method is "memorizing"
    known associations rather than genuinely predicting.
    """

    def __init__(
        self,
        method: str = "spearman",
        significance_level: float = 0.05,
    ):
        self.method = method
        self.significance_level = significance_level

    def compute(
        self,
        prediction_scores: np.ndarray,
        exposure_scores: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute Circ(M, T, K) = ρ(S_M(x), E_K(x)).

        Parameters
        ----------
        prediction_scores : array of shape (n_samples,)
            Method M's prediction scores (higher = more likely positive)
        exposure_scores : array of shape (n_samples,)
            Knowledge exposure scores E_K(x)

        Returns
        -------
        dict with keys: 'rho', 'p_value', 'ci_lower', 'ci_upper', 'is_significant'
        """
        # Remove NaN entries
        mask = ~(np.isnan(prediction_scores) | np.isnan(exposure_scores))
        scores = prediction_scores[mask]
        exposure = exposure_scores[mask]

        if len(scores) < 3:
            return {
                "rho": np.nan, "p_value": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan,
                "is_significant": False,
            }

        if self.method == "spearman":
            rho, p_val = stats.spearmanr(scores, exposure)
        elif self.method == "pearson":
            rho, p_val = stats.pearsonr(scores, exposure)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Bootstrap 95% CI
        ci_lower, ci_upper = self._bootstrap_ci(
            scores, exposure, n_bootstrap=1000
        )

        return {
            "rho": float(rho),
            "p_value": float(p_val),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "is_significant": p_val < self.significance_level,
        }

    def _bootstrap_ci(
        self, x: np.ndarray, y: np.ndarray, n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Bootstrap 95% confidence interval for correlation."""
        n = len(x)
        rng = np.random.RandomState(42)
        boot_rhos = []

        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            if self.method == "spearman":
                r, _ = stats.spearmanr(x[idx], y[idx])
            else:
                r, _ = stats.pearsonr(x[idx], y[idx])
            if not np.isnan(r):
                boot_rhos.append(r)

        if len(boot_rhos) == 0:
            return np.nan, np.nan

        boot_rhos = np.array(boot_rhos)
        return float(np.percentile(boot_rhos, 2.5)), float(np.percentile(boot_rhos, 97.5))

    @staticmethod
    def compute_literature_circularity(
        prediction_scores: np.ndarray,
        literature_scores: np.ndarray,
    ) -> Dict[str, float]:
        """
        Convenience: compute circularity using literature prominence as exposure.

        This is the most common form: ρ(method_score, literature_prominence).
        High ρ means the method ranks literature-known genes higher,
        indicating circularity leakage.
        """
        scorer = CircularityScorer(method="spearman")
        return scorer.compute(prediction_scores, literature_scores)


# =============================================================================
# Definition 3: Circularity Inflation
# =============================================================================
class CircularityInflation:
    """
    Definition 3: Circularity Inflation Δ_perf

    Quantifies how much of a method's performance comes from circular
    features rather than genuine predictive signal.

    Δ_perf(M) = Perf(M_all_features) - Perf(M_non_circ_features)
    """

    def __init__(self, metric: str = "auroc"):
        self.metric = metric

    def compute(
        self,
        y_true: np.ndarray,
        y_pred_all: np.ndarray,
        y_pred_noncirc: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute circularity inflation.

        Parameters
        ----------
        y_true : ground truth labels
        y_pred_all : predictions from model with all features
        y_pred_noncirc : predictions from model with non-circular features only

        Returns
        -------
        dict with: 'perf_all', 'perf_noncirc', 'delta', 'inflation_ratio'
        """
        from sklearn.metrics import roc_auc_score, average_precision_score

        if self.metric == "auroc":
            score_fn = roc_auc_score
        elif self.metric == "auprc":
            score_fn = average_precision_score
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        perf_all = score_fn(y_true, y_pred_all)
        perf_noncirc = score_fn(y_true, y_pred_noncirc)
        delta = perf_all - perf_noncirc

        inflation_ratio = delta / perf_all if perf_all > 0 else 0.0

        return {
            "perf_all": float(perf_all),
            "perf_noncirc": float(perf_noncirc),
            "delta": float(delta),
            "inflation_ratio": float(inflation_ratio),
            "metric": self.metric,
        }


# =============================================================================
# Circularity Taxonomy: C1-C4 Classification
# =============================================================================
@dataclass
class CircularityReport:
    """Complete circularity report for a single method."""
    method_name: str
    circ_score: Dict[str, float]          # From Definition 2
    inflation: Optional[Dict[str, float]] = None  # From Definition 3
    c1_label_leakage: bool = False
    c2_feature_contamination: bool = False
    c3_evaluation_bias: bool = False
    c4_selection_bias: bool = False
    evidence: Dict[str, str] = field(default_factory=dict)
    auroc: float = 0.0
    auprc: float = 0.0

    @property
    def n_circularity_types(self) -> int:
        """Number of circularity types flagged."""
        return sum([
            self.c1_label_leakage, self.c2_feature_contamination,
            self.c3_evaluation_bias, self.c4_selection_bias,
        ])

    @property
    def is_clean(self) -> bool:
        """True if no circularity types flagged."""
        return self.n_circularity_types == 0

    @property
    def severity(self) -> str:
        """Classify severity based on circ_score and number of types."""
        rho = self.circ_score.get("rho", 0)
        if rho < 0.1 and self.n_circularity_types == 0:
            return "clean"
        elif rho < 0.2 and self.n_circularity_types <= 1:
            return "mild"
        elif rho < 0.4 and self.n_circularity_types <= 2:
            return "moderate"
        else:
            return "severe"

    def to_dict(self) -> Dict:
        """Convert to flat dictionary for DataFrame construction."""
        result = {
            "method": self.method_name,
            "circ_rho": self.circ_score.get("rho", np.nan),
            "circ_p": self.circ_score.get("p_value", np.nan),
            "circ_ci_lower": self.circ_score.get("ci_lower", np.nan),
            "circ_ci_upper": self.circ_score.get("ci_upper", np.nan),
            "circ_significant": self.circ_score.get("is_significant", False),
            "auroc": self.auroc,
            "auprc": self.auprc,
            "C1_label_leakage": self.c1_label_leakage,
            "C2_feature_contamination": self.c2_feature_contamination,
            "C3_evaluation_bias": self.c3_evaluation_bias,
            "C4_selection_bias": self.c4_selection_bias,
            "n_circ_types": self.n_circularity_types,
            "severity": self.severity,
            "is_clean": self.is_clean,
        }
        if self.inflation:
            result.update({
                "inflation_delta": self.inflation.get("delta", np.nan),
                "inflation_ratio": self.inflation.get("inflation_ratio", np.nan),
            })
        if self.evidence:
            for k, v in self.evidence.items():
                result[f"evidence_{k}"] = v
        return result


# =============================================================================
# Utility: Batch circularity assessment
# =============================================================================
def assess_circularity_batch(
    methods_results: Dict[str, Dict],
    exposure_scores: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    y_pred_noncirc: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Assess circularity for multiple methods at once.

    Parameters
    ----------
    methods_results : dict mapping method_name -> {
        'prediction_scores': array,
        'auroc': float,
        'c1': bool, 'c2': bool, 'c3': bool, 'c4': bool,
        'evidence': dict,
    }
    exposure_scores : array of knowledge exposure per sample
    y_true : ground truth (for inflation computation)
    y_pred_noncirc : non-circular model predictions (for inflation)

    Returns
    -------
    pd.DataFrame with one row per method
    """
    scorer = CircularityScorer(method="spearman")
    reports = []

    for method_name, data in methods_results.items():
        pred_scores = data["prediction_scores"]
        circ_score = scorer.compute(pred_scores, exposure_scores)

        report = CircularityReport(
            method_name=method_name,
            circ_score=circ_score,
            auroc=data.get("auroc", np.nan),
            auprc=data.get("auprc", np.nan),
            c1_label_leakage=data.get("c1", False),
            c2_feature_contamination=data.get("c2", False),
            c3_evaluation_bias=data.get("c3", False),
            c4_selection_bias=data.get("c4", False),
            evidence=data.get("evidence", {}),
        )

        # Compute inflation if possible
        if y_true is not None and y_pred_noncirc is not None:
            inflator = CircularityInflation(metric="auroc")
            inflation = inflator.compute(y_true, pred_scores, y_pred_noncirc)
            report.inflation = inflation

        reports.append(report.to_dict())

    return pd.DataFrame(reports)
