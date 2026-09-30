"""
t3_go_vintage.py — T3: GO-layer vintage swap
=============================================
Three GO annotation conditions, identical gene-level feature construction
(per aspect: annotation count + mean Jaccard to the 42 HR seed genes),
identical protocol (train <=2016 gold, test 2017+ gold, shared negatives):

  GOA_2016    : goa_human.gaf rows with assignment date <= 2016-12-31
  GOA_current : full goa_human.gaf (2026 release)
  Hetionet_2016: the benchmark's own GO layer (GpBP/GpMF/GpCC metaedges),
                frozen to 2015-16 curation — the same-resource pair with
                GOA_2016 coming from the date-filtered GOA side

The GOA_2016 -> GOA_current gap on the same file is the clean within-resource
absorption estimate for the GO knowledge channel.

Output: results/t3_go_vintage.csv, results/t3_summary.json
"""
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tm_utils as U

ASPECT = {"P": "BP", "F": "MF", "C": "CC"}


def goa_terms(date_limit=None):
    """gene -> {aspect: set(GO terms)} from goa_human.gaf, date-filtered."""
    terms = defaultdict(lambda: defaultdict(set))
    with gzip.open(f"{U.V11}/data/goa_human.gaf.gz", "rt",
                   encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("!"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 14:
                continue
            if p[8] not in ASPECT:
                continue
            if date_limit is not None and p[13] > date_limit:
                continue
            terms[p[2]][ASPECT[p[8]]].add(p[4])
    return {g: dict(a) for g, a in terms.items()}


def hetionet_go_terms():
    """gene -> {aspect: set(term ids)} from the benchmark's Hetionet GO layer."""
    nodes = pd.read_csv(f"{U.ROOT}/data/hetionet_nodes.tsv", sep="\t")
    gene_nodes = nodes[nodes["id"].str.startswith("Gene::")]
    id_to_sym = dict(zip(gene_nodes["id"].str.replace("Gene::", "", regex=False),
                         gene_nodes["name"]))
    edges = pd.read_csv(f"{U.ROOT}/data/hetionet_edges.tsv", sep="\t",
                        usecols=["source", "metaedge", "target"])
    meta_aspect = {"GpBP": "BP", "GpMF": "MF", "GpCC": "CC"}
    sel = edges[edges["metaedge"].isin(meta_aspect)
                & edges["source"].astype(str).str.startswith("Gene::")]
    terms = defaultdict(lambda: defaultdict(set))
    for s, meta, t in zip(sel["source"], sel["metaedge"], sel["target"]):
        sym = id_to_sym.get(str(s).replace("Gene::", ""))
        if sym:
            terms[sym][meta_aspect[meta]].add(str(t))
    return {g: dict(a) for g, a in terms.items()}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def go_features(terms, genes, seeds):
    """6 features: per aspect (BP/MF/CC): n_terms, mean Jaccard to HR seeds."""
    seed_sets = {a: [terms.get(s, {}).get(a, frozenset()) for s in seeds]
                 for a in ("BP", "MF", "CC")}
    X = np.zeros((len(genes), 6), dtype=np.float32)
    for i, g in enumerate(genes):
        gt = terms.get(g, {})
        for j, a in enumerate(("BP", "MF", "CC")):
            t = gt.get(a, frozenset())
            X[i, 2 * j] = len(t)
            if t:
                X[i, 2 * j + 1] = float(np.mean([jaccard(t, s)
                                                 for s in seed_sets[a]]))
    return X


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

    print("parsing GO conditions...")
    conditions = {
        "GOA_2016": goa_terms(date_limit="20161231"),
        "GOA_current": goa_terms(),
        "Hetionet_2016": hetionet_go_terms(),
    }
    rows = []
    for name, terms in conditions.items():
        n_annot = sum(len(g) for g in terms.values())
        print(f"  {name}: {len(terms)} genes, {n_annot} gene-aspect entries")
        X = go_features(terms, genes, seeds)
        m = XGBClassifier(**U.GBM_PARAMS).fit(X[tr], y[tr])
        ev = U.eval_block(y_te, m.predict_proba(X[te])[:, 1],
                          seed=U.SEED + len(name))
        rows.append({"condition": name, "n_genes_annotated": len(terms),
                     "n_entries": n_annot, **ev})
        print(f"    AUROC={ev['auroc']:.4f} [{ev['ci_lo']:.3f},{ev['ci_hi']:.3f}]")

    # mechanism: annotation-count growth for new-gold vs candidates (GOA side)
    g16, gcur = conditions["GOA_2016"], conditions["GOA_current"]

    def total_terms(terms, gene):
        return sum(len(v) for v in terms.get(gene, {}).values())

    new_idx = np.where(new_pos)[0]
    growth = {
        "new_gold_mean_terms_2016": float(np.mean([total_terms(g16, genes[i])
                                                   for i in new_idx])),
        "new_gold_mean_terms_current": float(np.mean([total_terms(gcur, genes[i])
                                                      for i in new_idx])),
        "candidate_mean_terms_2016": float(np.mean([total_terms(g16, g)
                                                    for g in genes])),
        "candidate_mean_terms_current": float(np.mean([total_terms(gcur, g)
                                                       for g in genes])),
    }
    a = pd.Series([total_terms(g16, g) for g in genes])
    b = pd.Series([total_terms(gcur, g) for g in genes])
    growth["term_count_rank_stability"] = round(float(a.corr(b,
                                                             method="spearman")), 4)

    df = pd.DataFrame(rows)
    df.to_csv(f"{U.V11}/results/t3_go_vintage.csv", index=False)
    with open(f"{U.V11}/results/t3_summary.json", "w", encoding="utf-8") as f:
        json.dump({"design": ("GO vintage swap: identical 6-feature gene-level "
                              "construction; train <=2016, test 2017+"),
                   "results": rows, "growth": growth}, f, indent=2)
    print(json.dumps(growth, indent=2))


if __name__ == "__main__":
    main()
