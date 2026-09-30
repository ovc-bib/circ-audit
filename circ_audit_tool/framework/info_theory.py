"""
info_theory.py — Information-Theoretic Analysis for Circularity-Aware Prediction
================================================================================
Implements the PPI Ceiling Theorem and marginal information gain analysis.

Key results:
    1. AUROC upper bound from mutual information: AUROC ≤ Φ(√(2·I(X;Y)))
    2. Marginal information gain: ΔI(source | existing)
    3. Feature decomposition: I(X;Y) = I(X_nc;Y) + I(X_c;Y|X_nc) + I(X_nc,X_c;Y)_joint
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erf, erfc
from sklearn.metrics import roc_auc_score, mutual_info_score
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Information-Theoretic Utilities
# =============================================================================

def _entropy_binary(p: float) -> float:
    """Binary entropy H(Y) = -p·ln(p) - (1-p)·ln(1-p) in nats."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def auroc_to_mi(auroc: float, positive_rate: float = 0.5) -> float:
    """
    Convert AUROC to estimated mutual information (nats) using
    numerical integration of the Gaussian equal-variance model
    with arbitrary priors.

    Under the Gaussian equal-variance model with priors p₁ = positive_rate,
    p₀ = 1-positive_rate, and score distributions:
        S|Y=1 ~ N(d'/2, 1),  S|Y=0 ~ N(-d'/2, 1)

    where d' = √2 · Φ⁻¹(AUROC) (prior-independent).

    MI(S;Y) = H(Y) - ∫ H_b(p₁·f₁(s)/(p₁·f₁(s)+p₀·f₀(s))) · (p₁·f₁(s)+p₀·f₀(s)) ds

    For balanced data (positive_rate=0.5), this reduces to:
        I ≈ [Φ⁻¹(AUROC)]² · ln(2)

    For imbalanced data, the simple formula overestimates MI, potentially
    exceeding H(Y), which is impossible. The numerical integration handles
    this correctly.

    Parameters
    ----------
    auroc : float, AUROC value in (0.5, 1)
    positive_rate : float, proportion of positive samples (default 0.5 = balanced)

    Returns
    -------
    float, estimated mutual information in nats (guaranteed ≤ H(Y))
    """
    if auroc <= 0.5 or auroc >= 1.0:
        return 0.0
    p1 = positive_rate
    p0 = 1.0 - positive_rate
    h_y = _entropy_binary(positive_rate)
    if h_y <= 0 or p1 <= 0 or p0 <= 0:
        return 0.0

    # d' from AUROC (prior-independent): AUC = Φ(d'/√2)
    d_prime = np.sqrt(2) * stats.norm.ppf(auroc)

    # Numerical integration over score space
    # Score distributions: f0 = N(-d'/2, 1), f1 = N(d'/2, 1)
    mu0, mu1 = -d_prime / 2, d_prime / 2
    sigma = 1.0

    # Sample from the mixture distribution for Monte Carlo integration
    n_samples = 50000
    rng = np.random.default_rng(42)
    # Sample class labels according to priors
    labels = rng.binomial(1, p1, n_samples)
    # Sample scores from corresponding class-conditional distributions
    scores = mu0 + rng.standard_normal(n_samples)
    scores[labels == 1] = mu1 + rng.standard_normal(np.sum(labels == 1))

    # Compute posterior probability P(Y=1|S=s) for each sample
    f0 = stats.norm.pdf(scores, mu0, sigma)  # likelihood under class 0
    f1 = stats.norm.pdf(scores, mu1, sigma)  # likelihood under class 1
    posterior_1 = p1 * f1 / (p1 * f1 + p0 * f0)

    # Compute conditional entropy H(Y|S) by averaging H_b(posterior) over samples
    # H_b(q) = -q*ln(q) - (1-q)*ln(1-q), but need to handle q=0,1
    eps = 1e-15
    q = np.clip(posterior_1, eps, 1 - eps)
    h_b = -q * np.log(q) - (1 - q) * np.log(1 - q)

    # Average conditional entropy (Monte Carlo estimate)
    h_y_given_s = np.mean(h_b)

    # MI = H(Y) - H(Y|S)
    mi = h_y - h_y_given_s

    # Safety: MI must be non-negative and ≤ H(Y)
    mi = max(0.0, min(mi, h_y))

    return float(mi)


def mi_to_auroc(mi: float, positive_rate: float = 0.5, h_y: Optional[float] = None) -> float:
    """
    Convert mutual information to AUROC via numerical inversion of auroc_to_mi.

    Finds AUROC such that auroc_to_mi(AUROC, positive_rate) = mi.

    Parameters
    ----------
    mi : float, mutual information in nats
    positive_rate : float, proportion of positive samples (default 0.5 = balanced)
    h_y : float or None, pre-computed H(Y) in nats. Ignored — computed from positive_rate
        for consistency with auroc_to_mi.
    """
    if mi <= 0:
        return 0.5
    h_y_calc = _entropy_binary(positive_rate)
    if h_y_calc <= 0 or mi > h_y_calc:
        # MI cannot exceed H(Y); return maximum achievable AUROC
        # which approaches 1.0 but is never exactly 1.0
        return 0.9999

    # Numerical inversion: find AUROC where auroc_to_mi(auroc, positive_rate) = mi
    # Use bisection search since auroc_to_mi is monotonically increasing in AUROC
    auroc_lo, auroc_hi = 0.501, 0.999
    for _ in range(50):  # ~50 iterations for double precision
        auroc_mid = (auroc_lo + auroc_hi) / 2
        mi_mid = auroc_to_mi(auroc_mid, positive_rate)
        if mi_mid < mi:
            auroc_lo = auroc_mid
        else:
            auroc_hi = auroc_mid
    return float((auroc_lo + auroc_hi) / 2)


def estimate_mi_continuous(
    X: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> float:
    """
    Estimate mutual information I(X;Y) for continuous X and binary Y
    using histogram-based discretization.
    """
    # Discretize X
    bins = np.percentile(X, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    X_binned = np.digitize(X, bins[1:-1])

    return mutual_info_score(X_binned, y) * np.log(2)  # bits to nats


# =============================================================================
# PPI Ceiling Theorem
# =============================================================================

@dataclass
class CeilingAnalysis:
    """Result of PPI Ceiling analysis."""
    auroc_observed: float
    mi_estimated: float           # I(PPI; SL) in nats
    auroc_ceiling: float          # Theoretical ceiling from MI
    auroc_noncirc: float          # Observed non-circular AUROC
    ceiling_gap: float            # auroc_ceiling - auroc_noncirc
    nats_above_ceiling: float     # MI gap suggesting circularity
    is_above_ceiling: bool        # Whether observed AUROC exceeds ceiling
    interpretation: str = ""

    def __post_init__(self):
        if self.ceiling_gap < 0:
            self.is_above_ceiling = False
            self.interpretation = (
                f"Non-circular AUROC ({self.auroc_noncirc:.3f}) within "
                f"PPI ceiling ({self.auroc_ceiling:.3f}). No circularity leakage."
            )
        elif self.ceiling_gap < 0.03:
            self.is_above_ceiling = True
            self.interpretation = (
                f"Non-circular AUROC ({self.auroc_noncirc:.3f}) marginally above "
                f"PPI ceiling ({self.auroc_ceiling:.3f}). Minor leakage possible."
            )
        else:
            self.is_above_ceiling = True
            self.interpretation = (
                f"Non-circular AUROC ({self.auroc_noncirc:.3f}) substantially above "
                f"PPI ceiling ({self.auroc_ceiling:.3f}). Significant leakage likely."
            )


class PPICeilingAnalyzer:
    """
    Analyzes the PPI Ceiling phenomenon.

    The "PPI ceiling" is the information-theoretic upper bound on SL
    prediction performance when only PPI-derived features are used.
    Methods exceeding this bound must introduce either:
      (a) new genuine information (e.g., CRISPR), or
      (b) circularity leakage.
    """

    def __init__(self, positive_rate: float = 0.05):
        self.positive_rate = positive_rate
        self.h_y = _entropy_binary(positive_rate)

    def compute_ceiling(
        self,
        auroc_ppi_only: float,
        auroc_method: float,
        method_name: str = "",
    ) -> CeilingAnalysis:
        """
        Compute the PPI ceiling and check if a method exceeds it.

        Parameters
        ----------
        auroc_ppi_only : AUROC using only PPI features (non-circular)
        auroc_method : AUROC of the method being audited
        method_name : name for reporting

        Returns
        -------
        CeilingAnalysis
        """
        # Estimate I(PPI; SL) from PPI-only AUROC
        mi_ppi = auroc_to_mi(auroc_ppi_only, self.positive_rate)

        # Compute theoretical ceiling AUROC
        auroc_ceiling = mi_to_auroc(mi_ppi, h_y=self.h_y)

        # Compare method to ceiling
        ceiling_gap = auroc_method - auroc_ceiling

        # MI gap
        mi_method = auroc_to_mi(auroc_method, self.positive_rate)
        nats_gap = mi_method - mi_ppi

        return CeilingAnalysis(
            auroc_observed=auroc_method,
            mi_estimated=mi_ppi,
            auroc_ceiling=auroc_ceiling,
            auroc_noncirc=auroc_ppi_only,
            ceiling_gap=ceiling_gap,
            nats_above_ceiling=nats_gap,
            is_above_ceiling=ceiling_gap > 0.03,
        )

    def degree_stratified_ceiling(
        self,
        degree_bins: Dict[str, Tuple[float, float]],  # bin_name -> (auroc_ppi, auroc_method)
    ) -> pd.DataFrame:
        """
        Compute ceiling analysis per degree stratum.

        Demonstrates that the ceiling is lower for low-degree genes
        (less PPI information available), creating more room for
        non-PPI information sources.
        """
        results = []
        for bin_name, (auroc_ppi, auroc_method) in degree_bins.items():
            analysis = self.compute_ceiling(auroc_ppi, auroc_method, bin_name)
            results.append({
                "degree_bin": bin_name,
                "auroc_ppi_only": auroc_ppi,
                "auroc_method": auroc_method,
                "mi_ppi": analysis.mi_estimated,
                "ceiling_auroc": analysis.auroc_ceiling,
                "ceiling_gap": analysis.ceiling_gap,
                "nats_above": analysis.nats_above_ceiling,
            })
        return pd.DataFrame(results)


# =============================================================================
# Marginal Information Gain Analysis
# =============================================================================

@dataclass
class MarginalInfoGain:
    """Result of marginal information gain analysis."""
    base_source: str
    added_source: str
    auroc_base: float
    auroc_combined: float
    mi_base: float           # nats
    mi_combined: float       # nats
    delta_mi: float          # nats gained
    delta_auroc: float       # AUROC gained
    is_genuine: bool         # True if gain is likely genuine (not circular)
    evidence: str = ""


class MarginalInfoGainAnalyzer:
    """
    Analyze marginal information gain from adding feature sources.

    ΔI(source | existing) = I(X_existing + X_source; Y) - I(X_existing; Y)

    This quantifies whether a new feature source contributes genuine
    predictive information or just circular leakage.
    """

    def __init__(self, positive_rate: float = 0.05):
        self.positive_rate = positive_rate

    def compute_gain(
        self,
        auroc_base: float,
        auroc_combined: float,
        base_source: str = "PPI",
        added_source: str = "CRISPR",
        circularity_threshold: float = 0.2,
        circ_score: Optional[float] = None,
    ) -> MarginalInfoGain:
        """
        Compute marginal information gain from adding a feature source.

        Parameters
        ----------
        auroc_base : AUROC with base features only
        auroc_combined : AUROC with base + added features
        base_source : name of base feature source
        added_source : name of added feature source
        circularity_threshold : circ score above which gain is likely circular
        circ_score : optional circularity score for the added source
        """
        mi_base = auroc_to_mi(auroc_base, self.positive_rate)
        mi_combined = auroc_to_mi(auroc_combined, self.positive_rate)
        delta_mi = mi_combined - mi_base
        delta_auroc = auroc_combined - auroc_base

        # Determine if gain is genuine
        is_genuine = True
        evidence_parts = []

        if circ_score is not None and circ_score > circularity_threshold:
            is_genuine = False
            evidence_parts.append(
                f"Circ score ({circ_score:.3f}) > threshold ({circularity_threshold})"
            )

        if delta_auroc < 0:
            evidence_parts.append("Negative gain (feature may be noise)")
        elif delta_auroc < 0.01:
            evidence_parts.append("Negligible gain (<0.01)")
        elif delta_auroc > 0.1 and circ_score is None:
            evidence_parts.append("Large gain without circ audit - uncertain")

        if not evidence_parts:
            evidence_parts.append("Gain appears genuine (low circ, positive gain)")

        return MarginalInfoGain(
            base_source=base_source,
            added_source=added_source,
            auroc_base=auroc_base,
            auroc_combined=auroc_combined,
            mi_base=mi_base,
            mi_combined=mi_combined,
            delta_mi=delta_mi,
            delta_auroc=delta_auroc,
            is_genuine=is_genuine,
            evidence="; ".join(evidence_parts),
        )

    def batch_gain_analysis(
        self,
        gains_config: List[Dict],
    ) -> pd.DataFrame:
        """
        Compute marginal gains for multiple source combinations.

        Parameters
        ----------
        gains_config : list of dicts, each with keys:
            'base_source', 'added_source', 'auroc_base', 'auroc_combined',
            optional: 'circ_score'
        """
        results = []
        for config in gains_config:
            gain = self.compute_gain(**config)
            results.append({
                "base_source": gain.base_source,
                "added_source": gain.added_source,
                "auroc_base": gain.auroc_base,
                "auroc_combined": gain.auroc_combined,
                "delta_auroc": gain.delta_auroc,
                "mi_base_nats": gain.mi_base,
                "mi_combined_nats": gain.mi_combined,
                "delta_mi_nats": gain.delta_mi,
                "is_genuine": gain.is_genuine,
                "evidence": gain.evidence,
            })
        return pd.DataFrame(results)


# =============================================================================
# Feature Information Decomposition
# =============================================================================

class FeatureInfoDecomposer:
    """
    Decompose I(X;Y) into genuine and circular components.

    I(X;Y) ≈ I(X_nc;Y) + I(X_c;Y|X_nc)

    Where:
      I(X_nc;Y) = genuine predictive information (non-circular features)
      I(X_c;Y|X_nc) = information from circular features beyond non-circular
    """

    def __init__(self, positive_rate: float = 0.05):
        self.positive_rate = positive_rate

    def decompose(
        self,
        auroc_noncirc: float,
        auroc_all: float,
        auroc_circ_only: float = 0.5,
    ) -> Dict[str, float]:
        """
        Decompose prediction information into genuine vs circular.

        Parameters
        ----------
        auroc_noncirc : AUROC with non-circular features only
        auroc_all : AUROC with all features
        auroc_circ_only : AUROC with circular features only (default 0.5 = random)

        Notes
        -----
        When auroc_all < auroc_noncirc (method underperforms the non-circ baseline),
        the circular conditional contribution is negative, meaning circular features
        are actually counterproductive. In this case:
          - genuine_fraction = 1.0 (all information is genuine, even if method is weak)
          - circular_fraction = 0.0 (circular features hurt rather than help)
          - underperforming flag is set
        """
        mi_noncirc = auroc_to_mi(auroc_noncirc, self.positive_rate)
        mi_all = auroc_to_mi(auroc_all, self.positive_rate)
        mi_circ_only = auroc_to_mi(auroc_circ_only, self.positive_rate)

        # Conditional: I(X_c;Y|X_nc) ≈ I(X_all;Y) - I(X_nc;Y)
        # Can be negative if method underperforms the non-circ baseline
        mi_circ_cond_raw = mi_all - mi_noncirc

        # Handle underperforming case (AUROC_all < AUROC_noncirc)
        underperforming = auroc_all < auroc_noncirc

        if underperforming:
            # Method performs below non-circ baseline: circular features hurt
            mi_circ_cond = 0.0
            genuine_frac = 1.0
            circular_frac = 0.0
        elif mi_all <= 0:
            # Method is essentially random
            genuine_frac = 1.0
            circular_frac = 0.0
            mi_circ_cond = 0.0
        else:
            mi_circ_cond = max(0, mi_circ_cond_raw)
            genuine_frac = min(1.0, mi_noncirc / mi_all)
            circular_frac = max(0, 1.0 - genuine_frac)

        return {
            "mi_total_nats": mi_all,
            "mi_genuine_nats": mi_noncirc,
            "mi_circ_conditional_nats": mi_circ_cond,
            "mi_circ_only_nats": mi_circ_only,
            "genuine_fraction": genuine_frac,
            "circular_fraction": circular_frac,
            "auroc_noncirc": auroc_noncirc,
            "auroc_all": auroc_all,
            "underperforming": underperforming,
        }

    def cross_method_decomposition(
        self,
        methods_data: Dict[str, Dict],
    ) -> pd.DataFrame:
        """
        Decompose information for multiple methods.

        methods_data : dict mapping method_name -> {
            'auroc_noncirc': float, 'auroc_all': float,
            optional: 'auroc_circ_only': float
        }
        """
        results = []
        for name, data in methods_data.items():
            decomp = self.decompose(
                auroc_noncirc=data["auroc_noncirc"],
                auroc_all=data["auroc_all"],
                auroc_circ_only=data.get("auroc_circ_only", 0.5),
            )
            decomp["method"] = name
            results.append(decomp)
        return pd.DataFrame(results)
