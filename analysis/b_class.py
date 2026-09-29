"""
b_class.py — B-class sensitivity quartet
========================================
B4  Within-resource STRING vintage pair.  Bulk archived STRING releases
    (v10 of 2016, v11.5 of 2021, v12 of 2023) are rebuilt with one recipe
    (combined_score >= 400, ENSP->symbol via that release's protein.info,
    edges restricted to candidate genes) so that the Hetionet-proxy confound
    of Table 3 disappears, because resource identity is held fixed.
B6  Negative-pool sensitivity.  (a) exclude the 3,017 broad-gold genes from
    negatives, (b) five alternative negative-split seeds.
B7  HR seed-list vintage.  Seeds rebuilt as genes GO-annotated to homologous
    recombination repair (GO:0000724) with assignment dates <=2016, then the
    3-feature network model rerun on the 2015/2016 snapshots.

Outputs: results/b4_string_within.csv, b6_negative_pool.csv,
         b7_seed_vintage.csv
"""
import gzip
import sys
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, "E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/src")
import tm_utils as U
from t2_vintage_chain import graph_biogrid, graph_hetionet, features_on_graph
from a_class import paired_test

V11 = U.V11
ARCH = f"{V11}/data/string_archives"
SCORE_MIN = 400


def load_protein_map(path):
    m = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        header = f.readline()                      # '#string_protein_id ...'
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                m[p[0]] = p[1]                     # preferred_name
    return m


def build_v10_pmap():
    """ENSP -> symbol for STRING v10.  Tier 1 is the v11.5 protein.info map
    (most ENSPs stable across releases); tier 2 is mygene.info for the rest."""
    import gzip
    import json
    import os
    import subprocess
    cache = f"{ARCH}/v10_pmap.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    base = load_protein_map(f"{ARCH}/9606.protein.info.v11.5.txt.gz")
    need = set()
    with gzip.open(f"{ARCH}/9606.protein.links.v10.txt.gz", "rt") as f:
        f.readline()
        for line in f:
            a, b, s = line.split()
            if int(s) >= SCORE_MIN:
                if a not in base:
                    need.add(a)
                if b not in base:
                    need.add(b)
    print(f"  v10 score>=400 proteins unresolved by v11.5 map: {len(need)}")
    need = sorted(need)
    for i in range(0, len(need), 1000):
        batch = need[i:i + 1000]
        r = subprocess.run(["curl", "-s", "--ssl-no-revoke", "--max-time",
                            "60", "-X", "POST",
                            "https://mygene.info/v3/query",
                            "-H", "Content-Type: application/json",
                            "-d", json.dumps({"q": batch, "fields": "symbol",
                                              "scopes": "ensembl.protein",
                                              "species": "human"})],
                           capture_output=True, text=True)
        try:
            for hit in json.loads(r.stdout):
                if "symbol" in hit and "query" in hit:
                    base[hit["query"]] = hit["symbol"]
        except Exception:
            pass
    json.dump(base, open(cache, "w"))
    return base


def graph_string_bulk(links_gz, pmap, cand_set):
    edges = set()
    n_raw = n_sc = n_drop = 0
    for chunk in pd.read_csv(links_gz, sep=" ", chunksize=2_000_000):
        n_raw += len(chunk)
        ch = chunk[chunk["combined_score"] >= SCORE_MIN]
        n_sc += len(ch)
        a = ch["protein1"].map(pmap)
        b = ch["protein2"].map(pmap)
        ok = a.notna() & b.notna() & (a != b) & a.isin(cand_set) & b.isin(cand_set)
        n_drop += int((~(a.notna() & b.notna())).sum())
        for x, y in zip(a[ok], b[ok]):
            edges.add((x, y) if x < y else (y, x))
    g = {}
    for a, b in edges:
        g.setdefault(a, set()).add(b)
        g.setdefault(b, set()).add(a)
    print(f"  raw edges={n_raw:,} score>=400: {n_sc:,} "
          f"unmappable protein endpoints: {n_drop:,} "
          f"in-candidate edges={len(edges):,}")
    return g


def b4():
    print("=== B4 within-resource STRING pair ===")
    scores = U.load_scores()
    genes = scores["gene"].tolist()
    y = scores["y_gold"].to_numpy()
    cand_set = set(genes)
    old_pos, new_pos, negatives = U.era_masks(scores, 2016)
    train_neg, test_neg = U.shared_negative_split(negatives)
    tr = np.concatenate([np.where(old_pos)[0], train_neg])
    te = np.concatenate([np.where(new_pos)[0], test_neg])
    y_te = y[te]

    sys.path.insert(0, f"{U.ROOT}/crisprsl")
    sys.path.insert(0, U.ROOT)
    import config as cfg
    seeds = list(cfg.HR_SEED_GENES)

    files = {"STRING_v10_2016": ("9606.protein.links.v10.txt.gz", "V10MAP"),
             "STRING_v11.5_2021": ("9606.protein.links.v11.5.txt.gz",
                                   "9606.protein.info.v11.5.txt.gz"),
             "STRING_v12_2023": ("9606.protein.links.v12.0.txt.gz",
                                 "9606.protein.info.v12.0.txt.gz")}
    import os
    s_te, deg_tab = {}, {}
    rows = []
    for name, (lf, inf) in files.items():
        if not os.path.exists(f"{ARCH}/{lf}"):
            print(f"  {name}: file missing, skipped")
            continue
        t = time.time()
        pmap = build_v10_pmap() if inf == "V10MAP" else \
            load_protein_map(f"{ARCH}/{inf}")
        g = graph_string_bulk(f"{ARCH}/{lf}", pmap, cand_set)
        X, deg = features_on_graph(g, genes, seeds)
        deg_tab[name] = {gn: deg.get(gn, 0) for gn in genes}
        m = XGBClassifier(**U.GBM_PARAMS).fit(X[tr], y[tr])
        s_te[name] = m.predict_proba(X[te])[:, 1]
        ev = U.eval_block(y_te, s_te[name], seed=U.SEED + len(name))
        fut_deg = float(np.mean([deg_tab[name][genes[i]]
                                 for i in np.where(new_pos)[0]]))
        rows.append({"network": name, "n_genes": len(g), **ev,
                     "future_gold_mean_degree": round(fut_deg, 1)})
        print(f"  {name}: AUROC={ev['auroc']:.4f} "
              f"[{ev['ci_lo']:.3f},{ev['ci_hi']:.3f}] ({time.time()-t:.0f}s)")

    have = [n for n in files if n in s_te]
    for a, b in zip(have, have[1:]):
        d, lo, hi, p = paired_test(y_te, s_te[a], s_te[b])
        common = [(deg_tab[a][g], deg_tab[b][g]) for g in genes
                  if deg_tab[a][g] > 0 or deg_tab[b][g] > 0]
        rho = pd.Series([x for x, _ in common]).corr(
            pd.Series([z for _, z in common]), method="spearman")
        rows.append({"network": f"{a} -> {b}", "absorption": round(d, 4),
                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                     "p_paired_bootstrap": round(p, 4),
                     "degree_rank_stability": round(float(rho), 3)})
        print(f"  {a} -> {b}: Δ={d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f} "
              f"rank stability ρ={rho:.3f}")
    pd.DataFrame(rows).to_csv(f"{V11}/results/b4_string_within.csv", index=False)


def b6():
    print("\n=== B6 negative-pool sensitivity ===")
    scores = U.load_scores()
    genes = scores["gene"].tolist()
    y = scores["y_gold"].to_numpy()
    mats = U.build_matrices(scores)

    import json as _json
    with open(f"{U.ROOT}/saarf/results/universal_candidates_gold.json") as f:
        broad_set = set(_json.load(f))
    strict = set(np.array(genes)[y == 1])
    broad_set = broad_set - strict
    print(f"  broad (non-strict curated) genes: {len(broad_set)}")
    neg_base = (y == 0)
    neg_excl = neg_base & ~np.array([g in broad_set for g in genes])
    print(f"  negatives {int(neg_base.sum())} -> excl-broad {int(neg_excl.sum())}")

    rows = []
    for label, negmask, seed in (("baseline", neg_base, 42),
                                 ("excl_broad", neg_excl, 42),
                                 ("seed1", neg_base, 1), ("seed2", neg_base, 2),
                                 ("seed3", neg_base, 3), ("seed4", neg_base, 4)):
        _, new_pos, _ = U.era_masks(scores, 2016)
        train_neg, test_neg = U.shared_negative_split(negmask, seed=seed)
        old_pos, _, _ = U.era_masks(scores, 2016)
        tr = np.concatenate([np.where(old_pos)[0], train_neg])
        te = np.concatenate([np.where(new_pos)[0], test_neg])
        for name, X in mats.items():
            m = XGBClassifier(**U.GBM_PARAMS).fit(X[tr], y[tr])
            ev = U.eval_block(y[te], m.predict_proba(X[te])[:, 1],
                              seed=U.SEED + seed)
            rows.append({"variant": label, "method": name,
                         "n_train_neg": int(len(train_neg)),
                         "n_test_neg": int(len(test_neg)), **ev})
            print(f"  {label:<10} {name:<12} AUROC={ev['auroc']:.4f} "
                  f"[{ev['ci_lo']:.3f},{ev['ci_hi']:.3f}]")
        # label-free under excl-broad negatives (test-row definition changes)
        if label == "excl_broad":
            for name in ("PPI_Proximity", "SLant"):
                ev = U.eval_block(y[te], scores[name].to_numpy()[te],
                                  seed=U.SEED + 7)
                rows.append({"variant": label, "method": name, **ev,
                             "n_train_neg": None, "n_test_neg": int(len(test_neg))})
                print(f"  {label:<10} {name:<12} AUROC={ev['auroc']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{V11}/results/b6_negative_pool.csv", index=False)
    for meth in ("PPI_GBM", "CRISPR_Only", "Fusion"):
        v = df[(df.method == meth) & df.variant.str.startswith("seed")]["auroc"]
        print(f"  {meth} across 5 negative seeds: "
              f"{v.min():.4f}-{v.max():.4f} (spread {v.max()-v.min():.4f})")


def b7():
    print("\n=== B7 HR seed-list vintage ===")
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
    modern = list(cfg.HR_SEED_GENES)

    HR_TERM = "GO:0000724"      # double-strand break repair via homologous
                                # recombination
    vintage = set()
    with gzip.open(f"{V11}/data/goa_human.gaf.gz", "rt", encoding="utf-8",
                   errors="ignore") as f:
        for line in f:
            if line.startswith("!"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 15:
                continue
            if p[4] == HR_TERM and p[13] <= "20161231":
                vintage.add(p[2])
    vintage = sorted(vintage)
    inter = len(set(modern) & set(vintage))
    print(f"  modern seeds n={len(modern)}, GO<=2016 HR-repair genes n="
          f"{len(vintage)}, overlap {inter}")

    rows = []
    for gname, g in (("BioGRID_2015", graph_biogrid("3.4.136")),
                     ("Hetionet_2016", graph_hetionet())):
        for sname, seeds in (("modern_42", modern), ("vintage_go", vintage)):
            X, _ = features_on_graph(g, genes, seeds)
            m = XGBClassifier(**U.GBM_PARAMS).fit(X[tr], y[tr])
            ev = U.eval_block(y_te, m.predict_proba(X[te])[:, 1],
                              seed=U.SEED + len(sname))
            rows.append({"network": gname, "seed_set": sname,
                         "n_seeds": len(seeds), "n_seeds_in_graph":
                         len([s for s in seeds if s in g]), **ev})
            print(f"  {gname} seeds={sname:<10} n={len(seeds):<3} "
                  f"AUROC={ev['auroc']:.4f} [{ev['ci_lo']:.3f},{ev['ci_hi']:.3f}]")
    pd.DataFrame(rows).to_csv(f"{V11}/results/b7_seed_vintage.csv", index=False)


if __name__ == "__main__":
    b4()
    b6()
    b7()
