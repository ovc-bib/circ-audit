"""
t_diag_single_column.py — single-feature prospective diagnostics (paper Fig 3)
==============================================================================
Every input column of the benchmark's saarf table, evaluated alone against
2017+ gold (features are label-free by construction: the column predates any
model), plus the vintage-clean degree baselines from the snapshot chain.

Output: results/t_diag_single_column.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tm_utils as U
from t2_vintage_chain import graph_biogrid, graph_hetionet


def main():
    scores = U.load_scores()
    genes = scores["gene"].tolist()
    y = scores["y_gold"].to_numpy()
    _, test_neg = U.shared_negative_split(y == 0)
    _, new_pos, _ = U.era_masks(scores, 2016)
    te = np.concatenate([np.where(new_pos)[0], test_neg])
    y_te = y[te]

    sys.path.insert(0, f"{U.ROOT}/crisprsl")
    sys.path.insert(0, U.ROOT)
    from data_loader import CrisprSLDataLoader
    sa = CrisprSLDataLoader().load_all()["saarf_df"]
    sa_idx = {g: i for i, g in enumerate(sa["gene"])}

    mech = {
        "sl_pair_count": "label restatement",
        "is_known_sl_partner": "label restatement",
        "lit_score": "study bias + literature absorption",
        "ppi_proximity_score": "network position + absorption",
        "crispr_lfc_std": "perturbation variability (clean)",
        "crispr_lfc_median": "perturbation effect (clean)",
        "kg_score": "knowledge-graph channel",
        "kg_source_score": "knowledge-graph channel",
        "kg_non_hetionet_score": "knowledge-graph channel",
        "degree_raw": "study-bias carrier",
        "degree_centrality": "study-bias carrier",
        "ppr_score": "network position + absorption",
        "ppr_score_norm": "network position + absorption",
        "coess_score": "omics (weak)",
        "cond_ess_score": "omics (weak)",
        "pathway_score": "knowledge channel (weak)",
        "pathway_count": "knowledge channel (weak)",
        "pathway_enriched_score": "knowledge channel (weak)",
        "ctd_score": "knowledge channel",
    }
    rows = []
    for c in sa.columns:
        if c == "gene":
            continue
        v = np.array([sa.iloc[sa_idx[g]][c] if g in sa_idx else 0.0
                      for g in genes], dtype=float)
        try:
            auc = roc_auc_score(y_te, v[te])
        except Exception:
            continue
        rows.append({"feature": c, "auroc_2017plus": round(float(auc), 4),
                     "mechanism": mech.get(c, "other")})

    # vintage-clean degree baselines (no post-discovery curation possible)
    import config as cfg
    seeds = list(cfg.HR_SEED_GENES)
    yr = U.first_evidence_year()
    first_y = np.array([yr.get(g, np.nan) for g in genes], dtype=float)
    for name, g, v_year in (("BioGRID_2013_degree", graph_biogrid("3.2.121"), 2013),
                            ("BioGRID_2015_degree", graph_biogrid("3.4.136"), 2015),
                            ("Hetionet_2016_degree", graph_hetionet(), 2016)):
        fut = (y == 1) & np.isfinite(first_y) & (first_y > v_year)
        te_v = np.concatenate([np.where(fut)[0], test_neg])
        deg = np.array([len(g.get(gn, ())) for gn in genes], dtype=float)
        auc = roc_auc_score(y[te_v], deg[te_v])
        rows.append({"feature": name,
                     "auroc_2017plus": round(float(auc), 4),
                     "mechanism": "vintage-clean study bias"})

    df = pd.DataFrame(rows).sort_values("auroc_2017plus", ascending=False)
    df.to_csv(f"{U.V11}/results/t_diag_single_column.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
