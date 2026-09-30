"""
c9_crispr_dose.py — CRISPR feature vintage dose-response (C9)
=============================================================
The paper states that CRISPR perturbation features cannot be dated, because
DepMap knockout screens are measurements of present-day cell lines.  A dose
chain across DepMap vintages tests the invariance claim experimentally.  The
current end is the benchmark's 24Q4 matrix (local).  The early end is the
archived 19Q1 release (558 Avana cell lines, 2019).

STATUS: the 19Q1 file is NOT retrievable from this network (figshare CDN
403, DepMap portal bot-wall, Sanger and Wayback dead).  Place
gene_effect_corrected.csv from figshare article 7655150 into
data/depmap_archives/ manually and rerun this script.

Core feature recipe, applied identically to both matrices: per candidate
gene, median and std of the knockout-effect profile, plus the Pearson
correlation with each HR seed gene's profile.

Caveat stated up front: 19Q1 (2019) postdates part of the 2017-2019 test
window, so this is a dose-response under partial overlap, not a clean
pre-window vintage.

Output: results/c9_crispr_dose.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, "E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/src")
import tm_utils as U
from a_class import paired_test

V11 = U.V11
EARLY = f"{V11}/data/depmap_archives/gene_effect_corrected.csv"
CURRENT = "E:/崔雷/博士/设计/ovc_project/data/depmap_crispr.csv"


def load_matrix(path):
    df = pd.read_csv(path, index_col=0)
    df.columns = [c.split(" (")[0].strip() for c in df.columns]
    df.index = [str(i).strip() for i in df.index]
    return df


def core_block(mat, genes, seeds):
    M = mat[ [g for g in genes if g in mat.columns] ].to_numpy(dtype=np.float64)
    have = [g for g in genes if g in mat.columns]
    med = np.zeros(len(genes)); std = np.zeros(len(genes))
    pos = {g: i for i, g in enumerate(have)}
    for i, g in enumerate(genes):
        if g in pos:
            v = M[:, pos[g]]
            v = v[~np.isnan(v)]
            med[i] = np.median(v) if len(v) else 0.0
            std[i] = np.std(v) if len(v) > 1 else 0.0
    comp = np.zeros((len(genes), len(seeds)))
    seed_cols = [pos.get(s) for s in seeds]
    with np.errstate(invalid="ignore"):
        for j, sc in enumerate(seed_cols):
            if sc is None:
                continue
            a = M[:, sc]
            for i, g in enumerate(genes):
                if g not in pos:
                    continue
                b = M[:, pos[g]]
                ok = ~np.isnan(a) & ~np.isnan(b)
                if ok.sum() > 20:
                    sa, sb = a[ok], b[ok]
                    if sa.std() > 0 and sb.std() > 0:
                        comp[i, j] = np.corrcoef(sa, sb)[0, 1]
    return np.column_stack([med, std, comp])


def main():
    if not os.path.exists(EARLY):
        sys.exit(f"missing {EARLY}\ndownload gene_effect_corrected.csv from "
                 "figshare article 7655150 (DepMap Achilles 19Q1 Public) "
                 "in a browser and rerun")
    scores = U.load_scores()
    genes = scores["gene"].tolist()
    y = scores["y_gold"].to_numpy()
    old_pos, new_pos, negatives = U.era_masks(scores, 2016)
    train_neg, test_neg = U.shared_negative_split(negatives)
    tr = np.concatenate([np.where(old_pos)[0], train_neg])
    te = np.concatenate([np.where(new_pos)[0], test_neg])

    sys.path.insert(0, f"{U.ROOT}/crisprsl")
    sys.path.insert(0, U.ROOT)
    import config as cfg
    seeds = list(cfg.HR_SEED_GENES)

    rows, s_te = [], {}
    for name, path in (("DepMap_19Q1", EARLY), ("DepMap_24Q4", CURRENT)):
        mat = load_matrix(path)
        print(f"  {name}: {mat.shape[0]} cell lines x {mat.shape[1]} genes")
        X = core_block(mat, genes, seeds)
        m = XGBClassifier(**U.GBM_PARAMS).fit(X[tr], y[tr])
        s_te[name] = m.predict_proba(X[te])[:, 1]
        ev = U.eval_block(y[te], s_te[name], seed=U.SEED + len(name))
        rows.append({"vintage": name, "n_cell_lines": mat.shape[0], **ev})
        print(f"  {name} core-block AUROC={ev['auroc']:.4f} "
              f"[{ev['ci_lo']:.3f},{ev['ci_hi']:.3f}]")
    d, lo, hi, p = paired_test(y[te], s_te["DepMap_19Q1"], s_te["DepMap_24Q4"])
    rows.append({"vintage": "19Q1 -> 24Q4 absorption", "auroc": round(d, 4),
                 "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                 "p_paired_bootstrap": round(p, 4)})
    print(f"  dose delta: {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}")
    pd.DataFrame(rows).to_csv(f"{V11}/results/c9_crispr_dose.csv", index=False)


if __name__ == "__main__":
    main()
