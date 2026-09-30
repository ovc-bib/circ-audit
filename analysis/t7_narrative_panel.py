"""
t7_narrative_panel.py — T7: the end-of-2016 shortlist (paper human anchor)
=========================================================================
Train the KB-restricted (<=2016 gold only) models, rank ALL candidates, and
ask where the 58 genes actually discovered in 2017-2019 ended up:
median rank, share inside top-100/500, and the named top-10 true future
discoveries. Two feature sets:
  current   : saarf PPI + CRISPR (as a 2016 researcher could NOT have built)
  vintage   : Hetionet-2016 PPI + CRISPR (honest end-of-2016 toolkit)

Output: results/t7_shortlist.csv, results/t7_summary.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tm_utils as U
from t2_vintage_chain import features_on_graph, graph_hetionet


def rank_stats(scores_all, y, new_pos, label):
    """Rank candidates by score (desc); where do new-gold genes sit?"""
    order = np.argsort(-scores_all)
    ranks = np.empty(len(scores_all), dtype=int)
    ranks[order] = np.arange(1, len(scores_all) + 1)
    new_ranks = ranks[new_pos]
    return {
        "variant": label,
        "median_rank_new_gold": int(np.median(new_ranks)),
        "mean_rank_new_gold": round(float(new_ranks.mean()), 1),
        "n_new_gold": int(new_pos.sum()),
        "in_top100": int((new_ranks <= 100).sum()),
        "in_top500": int((new_ranks <= 500).sum()),
        "in_top1000": int((new_ranks <= 1000).sum()),
    }, ranks


def main():
    scores = U.load_scores()
    genes = scores["gene"].tolist()
    y = scores["y_gold"].to_numpy()
    old_pos, new_pos, _ = U.era_masks(scores, 2016)
    train_neg, test_neg = U.shared_negative_split(y == 0)
    tr = np.concatenate([np.where(old_pos)[0], train_neg])

    sys.path.insert(0, f"{U.ROOT}/crisprsl")
    sys.path.insert(0, U.ROOT)
    import config as cfg

    mats = U.build_matrices(scores)

    # vintage PPI block from Hetionet 2016
    g = graph_hetionet()
    X_ppi_2016, _ = features_on_graph(g, genes, list(cfg.HR_SEED_GENES))
    X_fusion_vintage = np.hstack([mats["CRISPR_Only"], X_ppi_2016])

    yr = U.first_evidence_year()
    rows, top10_lists = [], {}
    score_dump = pd.DataFrame({"gene": genes})
    for label, X in (("Fusion_current", mats["Fusion"]),
                     ("Fusion_vintage2016", X_fusion_vintage),
                     ("PPI_GBM_current", mats["PPI_GBM"]),
                     ("PPI_vintage2016", X_ppi_2016),
                     ("CRISPR_Only", mats["CRISPR_Only"])):
        m = XGBClassifier(**U.GBM_PARAMS).fit(X[tr], y[tr])
        s = m.predict_proba(X)[:, 1]
        score_dump[label] = s
        stat, ranks = rank_stats(s, y, new_pos, label)
        rows.append(stat)
        # named top-10 true future discoveries for the headline variants
        if label in ("Fusion_vintage2016", "CRISPR_Only"):
            new_idx = set(np.where(new_pos)[0])
            order = np.argsort(-s)
            hits = [(genes[i], int(ranks[i]), int(yr.get(genes[i], 0)))
                    for i in order if i in new_idx][:10]
            top10_lists[label] = hits
        print(stat)
    score_dump.to_csv(f"{U.V11}/results/t7_per_gene_scores.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(f"{U.V11}/results/t7_shortlist.csv", index=False)
    with open(f"{U.V11}/results/t7_summary.json", "w", encoding="utf-8") as f:
        json.dump({"rank_stats": rows, "top10_future_discoveries": top10_lists},
                  f, indent=2)

    for k, hits in top10_lists.items():
        print(f"\n{k} — top-10 future discoveries (gene, rank, first-evidence year):")
        for gene, r, yv in hits:
            print(f"   {gene:<12} rank {r:>6}   ({int(yv)})")


if __name__ == "__main__":
    main()
