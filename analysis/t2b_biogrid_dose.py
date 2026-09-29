"""
t2b_biogrid_dose.py — T2b: within-resource absorption pair (BioGRID only)
=========================================================================
Same resource, same parser, same feature code, same train/test protocol:
  BioGRID 3.2.121 (2013)  vs  BioGRID 3.4.136 (2015)  vs  BioGRID 4.4.249 (current)
Trained on <=2016 gold, tested on 2017+ gold. Because all three graphs are
BioGRID physical human interactions, any performance rise toward the present
release is retroactive curation, not resource identity.

Output: results/t2b_biogrid_dose.csv, results/t2b_summary.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tm_utils as U
from t2_vintage_chain import graph_biogrid, features_on_graph


def main():
    scores = U.load_scores()
    genes = scores["gene"].tolist()
    y = scores["y_gold"].to_numpy()
    old_pos, new_pos, negatives = U.era_masks(scores, 2016)
    train_neg, test_neg = U.shared_negative_split(negatives)
    tr = np.concatenate([np.where(old_pos)[0], train_neg])
    te = np.concatenate([np.where(new_pos)[0], test_neg])
    y_te = y[te]

    sys.path.insert(0, f"{U.ROOT}/crisprsl")
    sys.path.insert(0, U.ROOT)
    import config as cfg
    seeds = list(cfg.HR_SEED_GENES)

    versions = [("3.2.121", 2013), ("3.4.136", 2015), ("4.4.249", 2024)]
    rows, degree_table = [], {}
    for ver, year in versions:
        g = graph_biogrid(ver)
        X, deg = features_on_graph(g, genes, seeds)
        degree_table[ver] = {gn: deg.get(gn, 0) for gn in genes}
        m = XGBClassifier(**U.GBM_PARAMS).fit(X[tr], y[tr])
        ev = U.eval_block(y_te, m.predict_proba(X[te])[:, 1],
                          seed=U.SEED + year)
        rows.append({"network": f"BioGRID_{ver}", "vintage_year": year, **ev})
        print(f"  BioGRID {ver} ({year}): AUROC={ev['auroc']:.4f} "
              f"[{ev['ci_lo']:.3f},{ev['ci_hi']:.3f}]")

    # new-gold degree growth within BioGRID
    new_genes = [genes[i] for i in np.where(new_pos)[0]]
    growth = {ver: float(np.mean([degree_table[ver].get(g, 0)
                                  for g in new_genes])) for ver, _ in versions}
    cand_growth = {ver: float(np.mean([degree_table[ver].get(g, 0)
                                       for g in genes])) for ver, _ in versions}
    # rank stability
    dd = pd.DataFrame({ver: [degree_table[ver].get(g, 0) for g in genes]
                       for ver, _ in versions}, index=genes)
    stab = {f"{a}->{b}": round(float(dd[a].corr(dd[b], method="spearman")), 4)
            for (a, _), (b, _) in zip(versions, versions[1:])}

    df = pd.DataFrame(rows)
    df.to_csv(f"{U.V11}/results/t2b_biogrid_dose.csv", index=False)
    with open(f"{U.V11}/results/t2b_summary.json", "w", encoding="utf-8") as f:
        json.dump({"design": ("BioGRID-only vintage chain, identical parser/"
                              "features/protocol; train <=2016, test 2017+"),
                   "results": rows,
                   "new_gold_degree_mean": growth,
                   "candidate_degree_mean": cand_growth,
                   "degree_rank_stability": stab,
                   "n_new_gold": len(new_genes)}, f, indent=2)
    print("\nnew-gold mean degree:", {k: round(v, 1) for k, v in growth.items()})
    print("candidate mean degree:", {k: round(v, 1) for k, v in cand_growth.items()})
    print("rank stability:", stab)


if __name__ == "__main__":
    main()
