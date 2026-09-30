"""
tm_utils.py — shared machinery for the v11 time-machine experiments
====================================================================
Single source of truth for:
  - era masks at an arbitrary cutoff (first experimental evidence year)
  - the ONE negative split shared by every cutoff (seed 42)
  - gene-level bootstrap CIs for AUROC (and paired differences)
  - the three feature matrices (PPI / CRISPR / Fusion) replicated from v7/e4b

Contamination rules (DESIGN.md §5):
  negative pool = candidates minus ALL strict gold (cutoff-independent);
  train/test negatives split once, shared across cutoffs.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import os

# Project root of the original analysis layout. The scripts in analysis/
# reference it for their raw inputs (archived database releases, DepMap
# matrices); override with the CIRCAUDIT_HOME environment variable to
# relocate. The derived result tables ship in this repository under data/.
ROOT = os.environ.get("CIRCAUDIT_HOME", "E:/崔雷/博士/设计/ovc_project")
V10 = f"{ROOT}/circ_aware_bib/v10_bioinformatics"
V11 = f"{ROOT}/circ_aware_bib/v11_timemachine"
SEED = 42
CUTOFFS = [2012, 2014, 2016]


# ------------------------------------------------------------------
# data access
# ------------------------------------------------------------------
def load_scores():
    """Candidate genes, strict-gold labels, degree, and the nine v10 method scores."""
    return pd.read_csv(f"{V10}/results/e3_per_gene_scores.csv")


def first_evidence_year():
    """gene -> first experimental evidence year (float, NaN if undated)."""
    split = pd.read_csv(f"{V10}/data/gold_temporal_split.csv")
    return dict(zip(split["gene"], split["first_year_experimental"]))


def era_masks(scores_df, cutoff):
    """Boolean masks over candidate rows at a given cutoff.

    old_pos: strict gold first evidenced <= cutoff (training positives)
    new_pos: strict gold first evidenced >  cutoff (test positives)
    negatives: everything that is not strict gold (cutoff-independent pool)
    """
    yr = first_evidence_year()
    y = scores_df["y_gold"].to_numpy()
    genes = scores_df["gene"].tolist()
    first_y = np.array([yr.get(g, np.nan) for g in genes], dtype=float)
    old_pos = (y == 1) & np.isfinite(first_y) & (first_y <= cutoff)
    new_pos = (y == 1) & np.isfinite(first_y) & (first_y > cutoff)
    negatives = y == 0
    return old_pos, new_pos, negatives


def shared_negative_split(negatives, test_frac=0.2, seed=SEED):
    """The one negative train/test split, shared by every cutoff and method."""
    rng = np.random.default_rng(seed)
    idx = np.where(negatives)[0]
    perm = rng.permutation(idx)
    n_test = int(test_frac * len(perm))
    return perm[n_test:], perm[:n_test]          # train_neg, test_neg


def test_indices(scores_df, cutoff, test_neg):
    """Test rows = new-era positives + shared held-out negatives."""
    _, new_pos, _ = era_masks(scores_df, cutoff)
    return np.concatenate([np.where(new_pos)[0], test_neg])


# ------------------------------------------------------------------
# feature matrices (v7/e4b replication, CRISPR identical to published)
# ------------------------------------------------------------------
def build_matrices(scores_df):
    import os
    sys.path.insert(0, f"{ROOT}/crisprsl")
    sys.path.insert(0, ROOT)
    import config as cfg
    from v6_supplementary import build_features
    from sklearn.preprocessing import StandardScaler

    ppi_cols = ["ppi_proximity_score", "ppr_score", "ppr_score_norm",
                "degree_centrality"]
    cand = pd.read_csv(f"{ROOT}/saarf/results/universal_candidates.csv")
    cand_idx = {g: i for i, g in enumerate(cand["gene"])}
    X_ppi = np.zeros((len(scores_df), len(ppi_cols)), dtype=np.float32)
    for i, g in enumerate(scores_df["gene"]):
        j = cand_idx.get(g)
        if j is not None:
            X_ppi[i] = cand.iloc[j][ppi_cols].to_numpy(dtype=np.float32)
    X_ppi = np.nan_to_num(X_ppi)

    merged = pd.read_csv(f"{ROOT}/crisprsl/results/v2_merged_features.csv")
    X_crispr, _ = build_features(merged, list(scores_df["gene"]),
                                 comp_hr=cfg.HR_SEED_GENES)
    X_crispr = StandardScaler().fit_transform(X_crispr)
    return {"PPI_GBM": X_ppi, "CRISPR_Only": X_crispr,
            "Fusion": np.hstack([X_crispr, X_ppi])}


GBM_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.1,
                  subsample=0.8, random_state=SEED,
                  eval_metric="logloss", verbosity=0)


# ------------------------------------------------------------------
# statistics
# ------------------------------------------------------------------
def auroc_ci(y_true, scores, n_boot=2000, seed=SEED):
    """AUROC with a gene-level bootstrap 95% CI (percentile)."""
    y_true = np.asarray(y_true); scores = np.asarray(scores)
    auc = roc_auc_score(y_true, scores)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(y_true)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.sum() in (0, len(yt)):
            boots[b] = np.nan
            continue
        boots[b] = roc_auc_score(yt, scores[idx])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(auc), float(lo), float(hi)


def paired_auroc_diff_ci(y_true, s1, s2, n_boot=2000, seed=SEED):
    """CI for AUROC(s1) - AUROC(s2) on the same rows (paired bootstrap)."""
    y_true = np.asarray(y_true)
    d_obs = roc_auc_score(y_true, s1) - roc_auc_score(y_true, s2)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(y_true)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.sum() in (0, len(yt)):
            boots[b] = np.nan
            continue
        boots[b] = roc_auc_score(yt, s1[idx]) - roc_auc_score(yt, s2[idx])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(d_obs), float(lo), float(hi)


def eval_block(y_true, scores, n_boot=2000, seed=SEED):
    """AUROC[CI] + AUPRC + P@20 on the given test rows."""
    order = np.argsort(scores)[::-1]
    p20 = float(y_true[order[:20]].mean())
    auc, lo, hi = auroc_ci(y_true, scores, n_boot, seed)
    return {"auroc": round(auc, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "auprc": round(float(average_precision_score(y_true, scores)), 4),
            "p_at_20": round(p20, 3)}


def retro_reference():
    """Retrospective CV AUROCs from the published v10 benchmark run."""
    import json
    with open(f"{V10}/results/e3_v7_rerun_summary.json", encoding="utf-8") as f:
        r = json.load(f)
    return {k: round(v["auroc"], 4) for k, v in r.items()
            if isinstance(v, dict) and "auroc" in v}
