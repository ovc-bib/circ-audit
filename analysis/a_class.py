"""
a_class.py — A-class sensitivity trio
=====================================
A1  Paired bootstrap tests for every Table 3 absorption delta (the "n.s."
    labels previously rested on CI overlap only).
A2  Row-deduplication sensitivity (99 duplicate gene symbols collapse to
    unique genes; headline leaderboard retrained on the deduplicated set).
A3  Universe-vintage shortlist (Fig 6 / Table 5 recomputed on the Hetionet
    2016 gene universe instead of today's 14,183-candidate universe).

Outputs: results/a1_paired_absorption.csv, a2_dedup_leaderboard.csv,
         a3_universe_shortlist.csv
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, "E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/src")
import tm_utils as U
from t2_vintage_chain import (graph_biogrid, graph_hetionet, graph_string2021,
                              features_on_graph)
from t3_go_vintage import goa_terms, go_features

V11 = U.V11
SEED = U.SEED


def paired_test(y_te, s1, s2, n_boot=2000, seed=SEED):
    """Delta = AUROC(s2) - AUROC(s1), bootstrap CI + two-sided p."""
    d_obs = roc_auc_score(y_te, s2) - roc_auc_score(y_te, s1)
    rng = np.random.default_rng(seed)
    n = len(y_te)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_te[idx]
        if yt.sum() in (0, len(yt)):
            boots[b] = np.nan
            continue
        boots[b] = roc_auc_score(yt, s2[idx]) - roc_auc_score(yt, s1[idx])
    boots = boots[~np.isnan(boots)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return d_obs, lo, hi, min(p, 1.0)


def a1():
    print("=== A1 paired bootstrap for Table 3 ===")
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

    def net_scores(graph):
        X, _ = features_on_graph(graph, genes, seeds)
        m = XGBClassifier(**U.GBM_PARAMS).fit(X[tr], y[tr])
        return m.predict_proba(X[te])[:, 1]

    s = {}
    for key, ver in (("BioGRID_2013", "3.2.121"), ("BioGRID_2015", "3.4.136"),
                     ("BioGRID_2024", "4.4.249")):
        s[key] = net_scores(graph_biogrid(ver))
    s["Hetionet_2016"] = net_scores(graph_hetionet())
    s["STRING_2021"] = net_scores(graph_string2021())

    for name, terms in (("GOA_2016", goa_terms(date_limit="20161231")),
                        ("GOA_current", goa_terms())):
        X = go_features(terms, genes, seeds)
        m = XGBClassifier(**U.GBM_PARAMS).fit(X[tr], y[tr])
        s[name] = m.predict_proba(X[te])[:, 1]

    pairs = [("BioGRID physical", "2015 → 2024", "BioGRID_2015", "BioGRID_2024"),
             ("BioGRID physical", "2013 → 2024", "BioGRID_2013", "BioGRID_2024"),
             ("STRING-family", "Hetionet 2016 → STRING 2021",
              "Hetionet_2016", "STRING_2021"),
             ("GO annotations", "≤2016 → current", "GOA_2016", "GOA_current")]
    rows = []
    for chan, pair, a, b in pairs:
        d, lo, hi, p = paired_test(y_te, s[a], s[b], seed=SEED + len(pair))
        rows.append({"channel": chan, "vintage_pair": pair,
                     "absorption": round(d, 4), "ci_lo": round(lo, 4),
                     "ci_hi": round(hi, 4), "p_paired_bootstrap": round(p, 4),
                     "significant_0.05": bool(p < 0.05)})
        print(f"  {chan:16s} {pair:28s} Δ={d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}")
    pd.DataFrame(rows).to_csv(f"{V11}/results/a1_paired_absorption.csv", index=False)


def a2():
    print("\n=== A2 deduplication sensitivity ===")
    scores = U.load_scores()
    genes = scores["gene"].tolist()
    y = scores["y_gold"].to_numpy()
    # label consistency across duplicate symbols
    chk = scores.groupby("gene")["y_gold"].nunique()
    print(f"  inconsistent-label symbols: {int((chk > 1).sum())}")
    dedup = scores.drop_duplicates(subset="gene", keep="first").reset_index(drop=True)
    print(f"  rows {len(scores)} -> {len(dedup)} unique genes "
          f"(gold {int(y.sum())} -> {int(dedup['y_gold'].sum())})")
    genes_d = dedup["gene"].tolist()
    y_d = dedup["y_gold"].to_numpy()
    _, _, neg_d = U.era_masks(dedup, 2016)
    train_neg, test_neg = U.shared_negative_split(neg_d)

    orig = pd.read_csv(f"{V11}/results/t1_prospective_leaderboard.csv")
    orig_map = {(r.method, r.cutoff): r.auroc for r in orig.itertuples()}

    mats = U.build_matrices(dedup)
    rows = []
    for c in U.CUTOFFS:
        op, np_, _ = U.era_masks(dedup, c)
        tr = np.concatenate([np.where(op)[0], train_neg])
        te = np.concatenate([np.where(np_)[0], test_neg])
        for name, X in mats.items():
            m = XGBClassifier(**U.GBM_PARAMS).fit(X[tr], y_d[tr])
            ev = U.eval_block(y_d[te], m.predict_proba(X[te])[:, 1], seed=SEED + c)
            o = orig_map.get((name, c))
            rows.append({"method": name, "cutoff": c,
                         "n_test_pos": int(np_.sum()), **ev,
                         "auroc_rowlevel": o,
                         "delta": round(ev["auroc"] - o, 4) if o else None})
            print(f"  {name:<12} @{c} dedup={ev['auroc']:.4f} "
                  f"orig={o} Δ={rows[-1]['delta']}")
        for name in ("PPI_Proximity", "SLant"):
            ev = U.eval_block(y_d[te], dedup[name].to_numpy()[te], seed=SEED + c)
            o = orig_map.get((name, c))
            rows.append({"method": name, "cutoff": c,
                         "n_test_pos": int(np_.sum()), **ev,
                         "auroc_rowlevel": o,
                         "delta": round(ev["auroc"] - o, 4) if o else None})
            print(f"  {name:<12} @{c} dedup={ev['auroc']:.4f} orig={o} "
                  f"Δ={rows[-1]['delta']}")

    # SINaTRA headline on deduplicated genes
    import tm_methods as M
    sys.path.insert(0, f"{U.ROOT}/crisprsl")
    sys.path.insert(0, U.ROOT)
    from data_loader import CrisprSLDataLoader
    data = CrisprSLDataLoader().load_all()
    saarf_df = data["saarf_df"]
    import config as cfg
    context_genes = [g for g in cfg.HR_SEED_GENES
                     if g not in data.get("gold_genes_d4_strict", set())]
    all_gold_d = set(genes_d[i] for i in np.where(y_d == 1)[0])
    for c in U.CUTOFFS:
        op, np_, _ = U.era_masks(dedup, c)
        train_gold = set(genes_d[i] for i in np.where(op)[0])
        t = time.time()
        s = M.sinatra_retrain(genes_d, saarf_df, context_genes,
                              train_gold_set=train_gold,
                              neg_exclude=all_gold_d)
        te = np.concatenate([np.where(np_)[0], test_neg])
        ev = U.eval_block(y_d[te], s[te], seed=SEED + c)
        o = orig_map.get(("SINaTRA", c))
        rows.append({"method": "SINaTRA", "cutoff": c,
                     "n_test_pos": int(np_.sum()), **ev,
                     "auroc_rowlevel": o,
                     "delta": round(ev["auroc"] - o, 4) if o else None})
        print(f"  SINaTRA       @{c} dedup={ev['auroc']:.4f} orig={o} "
              f"Δ={rows[-1]['delta']} ({time.time()-t:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(f"{V11}/results/a2_dedup_leaderboard.csv", index=False)
    d = df["delta"].dropna().abs()
    print(f"  max |Δ| = {d.max():.4f}  mean |Δ| = {d.mean():.4f}")


def a3():
    print("\n=== A3 universe-vintage shortlist ===")
    scores = U.load_scores()
    t7 = pd.read_csv(f"{V11}/results/t7_per_gene_scores.csv")
    assert (scores["gene"].to_numpy() == t7["gene"].to_numpy()).all()
    old_pos, new_pos, _ = U.era_masks(scores, 2016)

    nodes = pd.read_csv(f"{U.ROOT}/data/hetionet_nodes.tsv", sep="\t")
    het_genes = set(nodes.loc[nodes["kind"] == "Gene", "name"])
    pool = np.array([g in het_genes for g in scores["gene"]])
    print(f"  candidates in Hetionet-2016 universe: {int(pool.sum())} "
          f"of {len(scores)}")
    fut_in = int((new_pos & pool).sum())
    print(f"  future gold inside universe: {fut_in} of {int(new_pos.sum())}")

    disc = pool & ~old_pos
    N, n = int(disc.sum()), int((new_pos & pool).sum())
    order = ["Fusion_current", "PPI_GBM_current", "CRISPR_Only",
             "Fusion_vintage2016", "PPI_vintage2016"]
    rows = []
    for v in order:
        s = t7[v].to_numpy()
        idx = np.where(disc)[0]
        ranked = idx[np.argsort(-s[disc])]
        rank = np.full(len(scores), -1, dtype=int)
        rank[ranked] = np.arange(1, len(ranked) + 1)
        nr = rank[new_pos & pool]
        r = {"variant": v, "N_universe": N, "n_future": n,
             "median": int(np.median(nr)), "null_median": (N + 1) // 2}
        for K in (100, 500, 1000):
            obs = int((nr <= K).sum())
            r[f"top{K}"] = obs
            r[f"E{K}"] = round(K * n / N, 2)
            r[f"enrich{K}"] = round(obs / (K * n / N), 1)
        rows.append(r)
        print(f"  {v:<20} median={r['median']:>5} top100={r['top100']} "
              f"(E={r['E100']}, {r['enrich100']}x) top500={r['top500']} "
              f"top1000={r['top1000']} ({r['enrich1000']}x)")
    pd.DataFrame(rows).to_csv(f"{V11}/results/a3_universe_shortlist.csv",
                              index=False)


if __name__ == "__main__":
    a1()
    a3()
    a2()   # heaviest last (SINaTRA retraining)
