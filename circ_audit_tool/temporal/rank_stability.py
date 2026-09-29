"""
rank_stability.py — Vintage rank stability and feature-side absorption
======================================================================
The diagnostic identified by the temporal evaluation: a knowledge channel's
inflation tracks how much it reorders gene prominence between vintages
(the Spearman stability of Eq (8)) rather than how much it grows, so rank
stability between a candidate vintage and the present is itself a checkable
quantity.

Feature-side absorption (Eq (9)) is the AUROC difference of the identical
model, protocol, labels and test rows when only the knowledge channel's
snapshot moves from vintage V_a to vintage V_b.
"""

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def auroc_pairwise(y_true: Sequence[float], scores: Sequence[float]) -> float:
    """Eq (7): AUROC in the pairwise rank form."""
    y = np.asarray(y_true)
    s = np.asarray(scores)
    pos, neg = s[y == 1], s[y != 1]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg)))


def rank_stability(feature_a: pd.Series, feature_b: pd.Series) -> float:
    """Eq (8): Spearman correlation over candidates non-zero in either vintage."""
    both = pd.concat([feature_a.rename("a"), feature_b.rename("b")], axis=1).dropna()
    union = both[(both["a"] > 0) | (both["b"] > 0)]
    if len(union) < 3:
        return float("nan")
    return float(stats.spearmanr(union["a"], union["b"]).correlation)


def absorption(auroc_vintage_a: float, auroc_vintage_b: float) -> float:
    """Eq (9): A(V_b) - A(V_a) with protocol, labels, learner and test rows fixed."""
    return float(auroc_vintage_b - auroc_vintage_a)
