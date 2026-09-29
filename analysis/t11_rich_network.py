"""
t11_rich_network.py — T11: rich network feature panel (revision R2.3)
=====================================================================
The main text uses three interpretable network primitives (degree, PPR,
BFS proximity).  Reviewer 2 noted that network-based models typically use
more features (SINaTRA > 10).  This module rebuilds the vintage-absorption
analysis with an eight-feature panel that adds second-order topology
(local clustering, eigenvector centrality, k-core number, average
neighbour degree, triangle count), on the same snapshots:

  BioGRID 3.2.121 (2013), 4.4.249 (2024), Hetionet (2016), STRING (2021)

Protocol identical to T2 (train <=2016 gold, test 2017+ gold, shared
negatives, XGBoost).  Also reports per-feature rank stability between
vintages, showing which features drive the reordering signal.

Output: results/t11_rich_network.csv, t11_summary.json
"""
import json
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, "E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/src")
import tm_utils as U
from t2_vintage_chain import (graph_biogrid, graph_hetionet, graph_string2021,
                              features_on_graph)

V11 = U.V11
SEED = U.SEED
FEATS = ["degree_rank", "ppr", "bfs_prox", "clustering", "eigenvec",
         "kcore", "avg_nbr_deg", "triangles"]


def rich_features(graph, candidates, seeds):
    """Eight network features per candidate gene (columns follow FEATS)."""
    G = nx.Graph()
    G.add_nodes_from(candidates)
    # undirected edge list, each pair once
    seen = set()
    elist = []
    for a, nbrs in graph.items():
        for b in nbrs:
            k = (a, b) if a < b else (b, a)
            if k not in seen:
                seen.add(k)
                elist.append(k)
    G.add_edges_from(elist)

    X3, _ = features_on_graph(graph, candidates, seeds)   # deg/ppr/prox
    # rank-normalised degree as in the main text
    deg_raw = {g: len(graph.get(g, ())) for g in candidates}
    order = np.argsort(np.argsort([-deg_raw[g] for g in candidates]))
    deg_rank = (order + 1) / len(candidates)

    clustering = nx.clustering(G)
    eigen = nx.eigenvector_centrality(G, max_iter=500, tol=1e-6)
    kcore = nx.core_number(G)
    avgnd = nx.average_neighbor_degree(G)
    tri = nx.triangles(G)

    X = np.column_stack([
        deg_rank,
        X3[:, 1], X3[:, 2],
        [clustering.get(g, 0.0) for g in candidates],
        [eigen.get(g, 0.0) for g in candidates],
        [float(kcore.get(g, 0)) for g in candidates],
        [avgnd.get(g, 0.0) for g in candidates],
        [float(tri.get(g, 0)) for g in candidates],
    ]).astype(np.float32)
    return X


def main():
    t0 = time.time()
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

    graphs = {"BioGRID_2013": graph_biogrid("3.2.121"),
              "BioGRID_2024": graph_biogrid("4.4.249"),
              "Hetionet_2016": graph_hetionet(),
              "STRING_2021": graph_string2021()}

    rows, feat_tables = [], {}
    for name, g in graphs.items():
        t = time.time()
        X = rich_features(g, genes, seeds)
        feat_tables[name] = pd.DataFrame(X, columns=FEATS, index=genes)
        m = XGBClassifier(**U.GBM_PARAMS).fit(X[tr], y[tr])
        s = m.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], s)
        rng = np.random.default_rng(SEED)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(te), len(te))
            yt = y[te][idx]
            if 0 < yt.sum() < len(yt):
                boots.append(roc_auc_score(yt, s[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"network": name, "panel": "rich-8", "auroc":
                     round(float(auc), 4), "ci_lo": round(float(lo), 4),
                     "ci_hi": round(float(hi), 4)})
        print(f"  rich-8 on {name}: AUROC={auc:.4f} [{lo:.3f},{hi:.3f}] "
              f"({time.time() - t:.0f}s)")

    # paired absorption tests, rich panel
    def scores_on(X):
        m = XGBClassifier(**U.GBM_PARAMS).fit(X[tr], y[tr])
        return m.predict_proba(X[te])[:, 1]

    def paired(y_te, s1, s2, seed=SEED):
        d = roc_auc_score(y_te, s2) - roc_auc_score(y_te, s1)
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(y_te), len(y_te))
            yt = y_te[idx]
            if 0 < yt.sum() < len(yt):
                boots.append(roc_auc_score(yt, s2[idx]) -
                             roc_auc_score(yt, s1[idx]))
        p = 2 * min((np.array(boots) <= 0).mean(), (np.array(boots) >= 0).mean())
        return float(d), min(float(p), 1.0)

    y_te = y[te]
    tests = [("BioGRID physical", "2013 → 2024", "BioGRID_2013", "BioGRID_2024"),
             ("STRING-family", "Hetionet 2016 → STRING 2021",
              "Hetionet_2016", "STRING_2021")]
    for chan, pair, a, b in tests:
        Xa = feat_tables[a][FEATS].to_numpy()
        Xb = feat_tables[b][FEATS].to_numpy()
        d, p = paired(y_te, scores_on(Xa), scores_on(Xb), seed=SEED + len(pair))
        rows.append({"network": f"{a} → {b}", "panel": "rich-8 absorption",
                     "auroc": round(d, 4), "p_paired_bootstrap": round(p, 4)})
        print(f"  absorption {chan} {pair}: {d:+.4f} (p={p:.4f})")

    # per-feature rank stability between the reordering contrast
    stab = {}
    for a, b in (("Hetionet_2016", "STRING_2021"),
                 ("BioGRID_2013", "BioGRID_2024")):
        A, B = feat_tables[a], feat_tables[b]
        mask = (A["degree_rank"] > 0) | (B["degree_rank"] > 0)
        stab[f"{a}->{b}"] = {f: round(float(A[f][mask].corr(
            B[f][mask], method="spearman")), 4) for f in FEATS}
    print("  per-feature rank stability:", json.dumps(stab, indent=1))

    pd.DataFrame(rows).to_csv(f"{V11}/results/t11_rich_network.csv", index=False)
    with open(f"{V11}/results/t11_summary.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "feature_rank_stability": stab,
                   "features": FEATS,
                   "elapsed_min": round((time.time() - t0) / 60, 1)},
                  f, indent=2)
    print(f"T11 done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
