"""
t2_vintage_chain.py — T2: the feature-vintage chain (paper Fig 2)
=================================================================
Four network snapshots, identical feature code, identical train/test protocol:

  BioGRID 3.2.121 (2013)  -> post-2012-cutoff release
  BioGRID 3.4.136 (2015)  -> post-2014-cutoff release
  Hetionet v1    (2016)   -> STRING v10 + BioGRID frozen 2015-16 curation
  STRING v11.5   (2021)   -> the benchmark's own current network

Protocol (matches e4c so the 2016/2021 pair reproduces it):
  features   = degree (rank-normalised), PPR from 42 HR seeds (r=0.15),
               multi-source BFS proximity
  training   = gold first evidenced <=2016 + shared train negatives
  testing    = gold first evidenced 2017+ + shared held-out negatives
  learner    = XGBoost, v7 hyperparameters, seed 42

Also produced:
  - vintage degree as a study-bias baseline: degree@V -> future gold
  - mechanism panel: new-gold degree growth across the chain + rank stability
  - Fusion-vintage: CRISPR (modality-current) + PPI@vintage

Output: results/t2_vintage_chain.csv, results/t2_summary.json
"""
import json
import os
import sys
import zipfile
from collections import deque

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tm_utils as U

ROOT = U.ROOT
V11 = U.V11
SEED = U.SEED
V11_DATA = f"{V11}/data"


# ------------------------------------------------------------------
# graph builders
# ------------------------------------------------------------------
def graph_biogrid(version):
    """Human physical interactions from an archived BioGRID tab2 release.

    Handles both the old 3.2.x header (Official Symbol Interactor A,
    Organism Interactor A, Experimental System Type) and the 3.4+ header
    (Official Symbol A, Taxonomy ID Interactor A, Interaction Type).
    """
    z = zipfile.ZipFile(f"{V11_DATA}/BIOGRID-ORGANISM-{version}.tab2.zip")
    name = [n for n in z.namelist() if n.endswith(".tab2.txt")
            and "sapiens" in n.lower()][0]

    def pick(cols, *candidates):
        for c in candidates:
            if c in cols:
                return cols.index(c)
        raise KeyError(f"none of {candidates} in {cols[:10]}...")

    edges = set()
    with z.open(name) as f:
        cols = f.readline().decode("utf-8").lstrip("#").rstrip("\n").split("\t")
        ia = pick(cols, "Official Symbol A", "Official Symbol Interactor A")
        ib = pick(cols, "Official Symbol B", "Official Symbol Interactor B")
        ta = pick(cols, "Taxonomy ID Interactor A", "Organism Interactor A")
        tb = pick(cols, "Taxonomy ID Interactor B", "Organism Interactor B")
        it = pick(cols, "Interaction Type", "Experimental System Type")
        # both eras store a taxid ("9606") or the binomial; type casing varies
        human_vals = {"9606", "Homo sapiens"}
        for line in f:
            p = line.decode("utf-8", "ignore").rstrip("\n").split("\t")
            if len(p) <= it:
                continue
            if p[ta] not in human_vals or p[tb] not in human_vals:
                continue
            if p[it].lower() != "physical":
                continue
            a, b = p[ia], p[ib]
            if not a or not b or a == b or a == "-" or b == "-":
                continue
            edges.add((a, b) if a < b else (b, a))
    g = {}
    for a, b in edges:
        g.setdefault(a, set()).add(b)
        g.setdefault(b, set()).add(a)
    print(f"  BioGRID {version}: {len(g)} genes, {len(edges)} physical edges")
    return g


def graph_hetionet():
    nodes = pd.read_csv(f"{ROOT}/data/hetionet_nodes.tsv", sep="\t")
    gene_nodes = nodes[nodes["kind"] == "Gene"]
    id2sym = dict(zip(gene_nodes["id"], gene_nodes["name"]))
    edges = pd.read_csv(f"{ROOT}/data/hetionet_edges.tsv", sep="\t",
                        usecols=["source", "metaedge", "target"])
    gig = edges[edges["metaedge"] == "GiG"]
    g = {}
    n = 0
    for s, t in zip(gig["source"], gig["target"]):
        a = id2sym.get(s)
        b = id2sym.get(t)
        if a is None or b is None or a == b:
            continue
        sa = g.setdefault(a, set())
        sb = g.setdefault(b, set())
        if b not in sa:
            sa.add(b)
            sb.add(a)
            n += 1
    print(f"  Hetionet: {len(g)} genes, {n} edges")
    return g


def graph_string2021():
    st = pd.read_csv(f"{ROOT}/data/string_scores.csv")
    g = {}
    for a, b in zip(st["gene_a"], st["gene_b"]):
        if a == b:
            continue
        g.setdefault(a, set()).add(b)
        g.setdefault(b, set()).add(a)
    return g


# ------------------------------------------------------------------
# features on a graph (identical to e4c)
# ------------------------------------------------------------------
def features_on_graph(graph, candidates, seeds):
    deg = {g: len(graph.get(g, ())) for g in candidates}
    alpha = 0.15
    seeds_in = [s for s in seeds if s in graph]
    rank = {g: (1.0 / len(seeds_in)) if g in seeds_in else 0.0 for g in graph}
    for _ in range(40):
        new = {g: 0.0 for g in graph}
        for g, r in rank.items():
            nbrs = graph[g]
            if nbrs:
                share = (1 - alpha) * r / len(nbrs)
                for nb in nbrs:
                    new[nb] += share
            else:
                new[g] += (1 - alpha) * r
        for s in seeds_in:
            new[s] += alpha / len(seeds_in)
        rank = new
    ppr = {g: rank.get(g, 0.0) for g in candidates}

    dist = {g: None for g in candidates}
    q = deque()
    for s in seeds_in:
        dist[s] = 0
        q.append(s)
    while q:
        u = q.popleft()
        for v in graph.get(u, ()):
            if v not in dist or dist[v] is None:
                dist[v] = dist[u] + 1
                q.append(v)
    prox = {g: (0.0 if dist.get(g) is None else 1.0 / (1.0 + dist[g]))
            for g in candidates}

    maxd = max(deg.values()) + 1
    X = np.array([[deg[g] / maxd, ppr[g], prox[g]] for g in candidates],
                 dtype=np.float32)
    return X, deg


def main():
    scores = U.load_scores()
    genes = scores["gene"].tolist()
    y = scores["y_gold"].to_numpy()
    old_pos, new_pos, negatives = U.era_masks(scores, 2016)
    train_neg, test_neg = U.shared_negative_split(negatives)
    tr = np.concatenate([np.where(old_pos)[0], train_neg])
    te = np.concatenate([np.where(new_pos)[0], test_neg])
    y_te = y[te]

    sys.path.insert(0, f"{ROOT}/crisprsl")
    sys.path.insert(0, ROOT)
    import config as cfg
    seeds = list(cfg.HR_SEED_GENES)

    graphs = {
        "BioGRID_2013": graph_biogrid("3.2.121"),
        "BioGRID_2015": graph_biogrid("3.4.136"),
        "Hetionet_2016": graph_hetionet(),
        "STRING_2021": graph_string2021(),
    }

    # CRISPR block for the Fusion-vintage variant
    mats = U.build_matrices(scores)
    X_crispr = mats["CRISPR_Only"]

    rows, degree_table = [], {}
    vintage_years = {"BioGRID_2013": 2013, "BioGRID_2015": 2015,
                     "Hetionet_2016": 2016, "STRING_2021": 2021}
    for name, g in graphs.items():
        print(f"  building features on {name} ({len(g)} genes)...")
        X, deg = features_on_graph(g, genes, seeds)
        degree_table[name] = {gn: deg.get(gn, 0) for gn in genes}

        # PPI-only model on this vintage
        m = XGBClassifier(**U.GBM_PARAMS).fit(X[tr], y[tr])
        auc = roc_auc_score(y_te, m.predict_proba(X[te])[:, 1])
        ev = U.eval_block(y_te, m.predict_proba(X[te])[:, 1],
                          seed=SEED + int(vintage_years[name]))
        rows.append({"model": "PPI_vintage", "network": name,
                     "vintage_year": vintage_years[name], **ev})
        print(f"    PPI@{name}: AUROC={ev['auroc']:.4f} "
              f"[{ev['ci_lo']:.3f},{ev['ci_hi']:.3f}]")

        # Fusion-vintage: CRISPR (modality-current) + PPI@vintage
        Xf = np.hstack([X_crispr, X])
        mf = XGBClassifier(**U.GBM_PARAMS).fit(Xf[tr], y[tr])
        evf = U.eval_block(y_te, mf.predict_proba(Xf[te])[:, 1],
                           seed=SEED + int(vintage_years[name]) + 1)
        rows.append({"model": "Fusion_vintage", "network": name,
                     "vintage_year": vintage_years[name], **evf})
        print(f"    Fusion@{name}: AUROC={evf['auroc']:.4f} "
              f"[{evf['ci_lo']:.3f},{evf['ci_hi']:.3f}]")

    # ---- mechanism panel: new-gold degree growth + rank stability ----
    new_genes = [genes[i] for i in np.where(new_pos)[0]]
    cand_deg = pd.DataFrame({n: [degree_table[n].get(g, 0) for g in genes]
                             for n in graphs}, index=genes)
    new_deg = cand_deg.loc[new_genes]
    growth = {n: float(new_deg[n].mean()) for n in graphs}
    # rank stability between adjacent vintages (over all candidates with deg)
    stab = {}
    names = list(graphs)
    for a, b in zip(names, names[1:]):
        common = cand_deg[(cand_deg[a] > 0) | (cand_deg[b] > 0)]
        stab[f"{a}->{b}"] = round(float(common[a].corr(common[b],
                                                       method="spearman")), 4)

    # ---- study-bias baseline: degree@V as a single feature on FUTURE gold ----
    # (vintage-clean degree cannot contain post-V curation, so whatever
    #  predictive power it has on >V gold is study bias, not absorption)
    bias_rows = []
    yr = U.first_evidence_year()
    first_y = np.array([yr.get(g, np.nan) for g in genes], dtype=float)
    for name in ["BioGRID_2013", "BioGRID_2015", "Hetionet_2016"]:
        v = vintage_years[name]
        fut = (y == 1) & np.isfinite(first_y) & (first_y > v)
        te_v = np.concatenate([np.where(fut)[0], test_neg])
        degfeat = np.array([degree_table[name].get(g, 0) for g in genes],
                           dtype=float)
        ev = U.eval_block(y[te_v], degfeat[te_v], seed=SEED + v)
        bias_rows.append({"network": name, "vintage_year": v,
                          "n_future_gold": int(fut.sum()), **ev})

    df = pd.DataFrame(rows)
    df.to_csv(f"{V11}/results/t2_vintage_chain.csv", index=False)
    new_deg.to_csv(f"{V11}/results/t2_newgold_degree_by_vintage.csv")
    summary = {
        "design": ("4 network vintages, identical 3-feature code, "
                   "train <=2016 gold, test 2017+ gold, shared negatives"),
        "vintage_curve": rows,
        "new_gold_degree_mean": growth,
        "degree_rank_stability": stab,
        "study_bias_baseline": bias_rows,
        "n_new_gold": len(new_genes),
    }
    with open(f"{V11}/results/t2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== T2 vintage curve (train <=2016, test 2017+) ===")
    print(df[["model", "network", "auroc", "ci_lo", "ci_hi"]]
          .to_string(index=False))
    print("\nnew-gold mean degree by vintage:", {k: round(v, 1)
                                                 for k, v in growth.items()})
    print("rank stability:", stab)
    print("\nstudy-bias baseline (degree@V -> >V gold):")
    print(pd.DataFrame(bias_rows)[["network", "n_future_gold", "auroc"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
