"""
b5_sinatra_vintage.py — SINaTRA-style reimplementation under 2016-vintage
network inputs
============================================================================
The Table 1 ablation shows the SINaTRA-style reimplementation keeps
0.910-0.929 prospectively on topology-plus-omics inputs, but those inputs are
computed on today's networks.  Here the four network columns of the input
table are overwritten with Hetionet-2016 versions (degree rank, PPR, BFS
proximity from the same features_on_graph code), knowledge columns are
dropped as in the topo variant, and the reimplementation is retrained per
cutoff.  The drop from the current-feature topo row quantifies how much of
the SVM's prospective score arrives through network vintage.

Output: results/b5_sinatra_vintage.csv
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/src")
import tm_utils as U
import tm_methods as M
from t2_vintage_chain import graph_hetionet, features_on_graph

V11 = U.V11

# saarf network column -> Hetionet-2016 feature slot (0 degree, 1 ppr, 2 prox)
SWITCH_MAP = {"degree_raw": 0, "degree_centrality": 0,
              "ppr_score": 1, "ppr_score_norm": 1, "ppr": 1,
              "ppi_proximity_score": 2, "proximity": 2}


def main():
    scores = U.load_scores()
    genes = scores["gene"].tolist()
    y = scores["y_gold"].to_numpy()
    all_gold = set(np.array(genes)[y == 1])

    sys.path.insert(0, f"{U.ROOT}/crisprsl")
    sys.path.insert(0, U.ROOT)
    from data_loader import CrisprSLDataLoader
    import config as cfg
    data = CrisprSLDataLoader().load_all()
    saarf_df = data["saarf_df"].copy()
    context_genes = [g for g in cfg.HR_SEED_GENES
                     if g not in data.get("gold_genes_d4_strict", set())]

    print("saarf columns:", saarf_df.columns.tolist())
    X, _ = features_on_graph(graph_hetionet(), genes, list(cfg.HR_SEED_GENES))
    Xrow = {g: X[i] for i, g in enumerate(genes)}

    saarf_v = saarf_df.copy()
    switched = []
    for col, slot in SWITCH_MAP.items():
        if col in saarf_v.columns:
            saarf_v[col] = saarf_v["gene"].map(
                lambda g, sl=slot: Xrow.get(g, np.zeros(3))[sl] if g in Xrow
                else 0.0)
            switched.append(col)
    print("switched to Hetionet-2016:", switched)

    rows = []
    for c in U.CUTOFFS:
        op, _, _ = U.era_masks(scores, c)
        train_gold = set(np.array(genes)[op])
        t = time.time()
        s = M.sinatra_retrain(genes, saarf_v, context_genes,
                              train_gold_set=train_gold,
                              neg_exclude=all_gold,
                              exclude_cols=M.KNOWLEDGE_COLUMNS)
        _, np_, _ = U.era_masks(scores, c)
        train_neg, test_neg = U.shared_negative_split(y == 0)
        te = np.concatenate([np.where(np_)[0], test_neg])
        ev = U.eval_block(y[te], s[te], seed=U.SEED + c)
        rows.append({"cutoff": c, "n_test_pos": int(np_.sum()), **ev})
        print(f"  SINaTRA-topo vintage@{c} AUROC={ev['auroc']:.4f} "
              f"[{ev['ci_lo']:.3f},{ev['ci_hi']:.3f}] ({time.time()-t:.0f}s)")
    pd.DataFrame(rows).to_csv(f"{V11}/results/b5_sinatra_vintage.csv", index=False)


if __name__ == "__main__":
    main()
