"""
t10_pair_level.py — T10: pair-level temporal evaluation (revision R2.1)
=======================================================================
Conventional SL prediction is pair-level.  This module rebuilds the whole
temporal protocol on unordered gene PAIRS so the gene-level conclusions can
be checked under the conventional formulation:

  positives   experimental SL pairs from the evidence export, dated by the
              publication year of their first experimental evidence row
              (sensitivity: BioGRID SL/SGD pairs dated through PubMed as an
              independent positive channel)
  negatives   sampled candidate pairs absent from all-time SL-direction
              evidence (main), studied-pair negatives (both genes with
              BioGRID-2013 degree > 0), and experimentally tested non-SL
              pairs (BioGRID alleviating "Positive Genetic" pairs)
  features    gene-level features of the main text (3 network features on
              STRING-2021 / Hetionet-2016, CRISPR block, benchmark input
              table) mapped to pairs with the Eq (6) transforms
  protocol    train on pairs first evidenced <= cutoff, test on later pairs
              against the shared held-out negative portion; twin models
              admit all gold pairs into training for the label-side
              increment; retro reference = 5-fold random CV over all pairs

Outputs: results/t10_pair_leaderboard.csv, t10_pair_sensitivity.csv,
         t10_summary.json
"""
import json
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

sys.path.insert(0, "E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/src")
import tm_utils as U
import tm_methods as M
from t2_vintage_chain import (graph_biogrid, graph_hetionet, graph_string2021,
                              features_on_graph)

V11 = U.V11
SEED = U.SEED
CUTOFFS = U.CUTOFFS
V10_DATA = f"{U.V10}/data"
PMID_CACHE = f"{V10_DATA}/pubmed_years.csv"

SWITCH_MAP = {"degree_raw": 0, "degree_centrality": 0,
              "ppr_score": 1, "ppr_score_norm": 1,
              "ppi_proximity_score": 2}


# ------------------------------------------------------------------
# PubMed year resolution (cached; eutils, polite batches)
# ------------------------------------------------------------------
def pmid_years(pmids):
    cache = pd.read_csv(PMID_CACHE, dtype={"pmid": str})
    have = dict(zip(cache["pmid"].astype(str), cache["year"]))
    missing = sorted({p for p in pmids if p and p not in have})
    for i in range(0, len(missing), 150):
        batch = missing[i:i + 150]
        url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
               "?db=pubmed&retmode=json&id=" + ",".join(batch))
        with urllib.request.urlopen(url, timeout=120) as r:
            res = json.load(r)
        for uid, rec in res.get("result", {}).items():
            if uid == "uids" or not isinstance(rec, dict):
                continue
            pub = rec.get("pubdate", "")
            if pub[:4].isdigit():
                have[uid] = int(pub[:4])
        time.sleep(0.4)
    if missing:
        extra = pd.DataFrame(
            [{"pmid": m, "year": have[m]} for m in missing if m in have])
        cache = pd.concat([cache, extra.astype({"pmid": str})],
                          ignore_index=True)
        cache.to_csv(PMID_CACHE, index=False)
        print(f"  fetched {len(extra)} new PubMed years")
    return have


# ------------------------------------------------------------------
# positive pairs from the evidence export (channel E)
# ------------------------------------------------------------------
def export_pairs(candidate_set):
    ev = pd.read_csv(f"{V10_DATA}/gold_evidence_with_years.csv")
    exp = ev[ev["experimental"] == True]
    exp = exp[exp["gene_a"].isin(candidate_set)
              & exp["gene_b"].isin(candidate_set)]
    rows = {}
    for a, b, pmid, year in zip(exp["gene_a"], exp["gene_b"], exp["pmid"],
                                exp["year"]):
        k = (a, b) if a < b else (b, a)
        d = rows.setdefault(k, {"year": 1e9, "pmids": set()})
        d["year"] = min(d["year"], year)
        d["pmids"].add(str(pmid))
    recs = [{"a": a, "b": b, "year": int(d["year"]), "n_pmids": len(d["pmids"])}
            for (a, b), d in rows.items()]
    return pd.DataFrame(recs).sort_values("year").reset_index(drop=True)


# ------------------------------------------------------------------
# BioGRID genetic-interaction pairs (channel B + validated negatives)
# ------------------------------------------------------------------
def biogrid_genetic(candidate_set):
    import zipfile
    z = zipfile.ZipFile(f"{V11}/data/BIOGRID-ORGANISM-4.4.249.tab2.zip")
    name = [n for n in z.namelist() if n.endswith(".tab2.txt")
            and "sapiens" in n.lower()][0]
    sl_dir, allev = {}, set()
    with z.open(name) as f:
        h = f.readline().decode("utf-8").lstrip("#").rstrip().split("\t")
        es, est, pm = (h.index("Experimental System"),
                       h.index("Experimental System Type"),
                       h.index("Pubmed ID"))
        ia = h.index("Official Symbol Interactor A")
        ib = h.index("Official Symbol Interactor B")
        for line in f:
            p = line.decode("utf-8", "ignore").rstrip("\n").split("\t")
            if len(p) <= pm or p[est] != "genetic":
                continue
            a, b = p[ia], p[ib]
            if a not in candidate_set or b not in candidate_set or a == b:
                continue
            k = (a, b) if a < b else (b, a)
            if p[es] in ("Synthetic Lethality", "Synthetic Growth Defect"):
                sl_dir.setdefault(k, set()).add(p[pm])
            elif p[es] == "Positive Genetic":
                allev.add(k)
    years = pmid_years([q for d in sl_dir.values() for q in d])
    recs = []
    for k, d in sl_dir.items():
        ys = [years[q] for q in d if q in years and years[q]]
        if ys:
            recs.append({"a": k[0], "b": k[1], "year": min(ys),
                         "n_pmids": len(d)})
    return (pd.DataFrame(recs).sort_values("year").reset_index(drop=True),
            allev)


# ------------------------------------------------------------------
# pair feature transforms (Eq 6 of the main text)
# ------------------------------------------------------------------
def pair_phi(fa, fb):
    cos = float(np.dot(fa, fb) /
                max(np.linalg.norm(fa) * np.linalg.norm(fb), 1e-8))
    return np.concatenate([fa + fb, np.abs(fa - fb), fa * fb, [cos]])


def build_pair_X(pairs, gmat, dim):
    X = np.zeros((len(pairs), 3 * dim + 1), dtype=np.float32)
    for i, (a, b) in enumerate(pairs):
        X[i] = pair_phi(gmat[a], gmat[b])
    return X


def eval_pairs(y_te, s, n_boot=2000, seed=SEED):
    order = np.argsort(-s)
    p20 = float(y_te[order[:20]].mean())
    auc = roc_auc_score(y_te, s)
    rng = np.random.default_rng(seed)
    n = len(y_te)
    boots = np.empty(n_boot)
    for b_i in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_te[idx]
        boots[b_i] = (roc_auc_score(yt, s[idx])
                      if 0 < yt.sum() < len(yt) else np.nan)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"auroc": round(float(auc), 4), "ci_lo": round(float(lo), 4),
            "ci_hi": round(float(hi), 4), "p_at_20": round(p20, 3)}


def paired_pair_test(y_te, s1, s2, n_boot=2000, seed=SEED):
    """Delta = AUROC(s2) - AUROC(s1) on the same test pairs."""
    d_obs = roc_auc_score(y_te, s2) - roc_auc_score(y_te, s1)
    rng = np.random.default_rng(seed)
    n = len(y_te)
    boots = np.empty(n_boot)
    for b_i in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_te[idx]
        boots[b_i] = (roc_auc_score(yt, s2[idx]) - roc_auc_score(yt, s1[idx])
                      if 0 < yt.sum() < len(yt) else np.nan)
    boots = boots[~np.isnan(boots)]
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(d_obs), float(np.percentile(boots, 2.5)), \
        float(np.percentile(boots, 97.5)), min(float(p), 1.0)


def main():
    t0 = time.time()
    scores = U.load_scores()
    cand = scores["gene"].tolist()
    cand_set = set(cand)
    print(f"candidates: {len(cand)}")

    # ---------------- positives, negatives ----------------
    posE = export_pairs(cand_set)
    print(f"channel-E pairs (both in candidates): {len(posE)}  "
          f"post-2016: {(posE['year'] > 2016).sum()}")
    posB, allev_pairs = biogrid_genetic(cand_set)
    print(f"channel-B BioGRID SL/SGD pairs: {len(posB)}  "
          f"alleviating (Positive Genetic) pairs: {len(allev_pairs)}")

    any_gold_pair = set(zip(posE["a"], posE["b"])) | set(zip(posB["a"], posB["b"]))
    rng = np.random.default_rng(SEED)
    n_pool = len(posE)

    bio13 = graph_biogrid("3.2.121")
    deg13 = {g: len(bio13.get(g, ())) for g in cand}

    def sample_pool(both_studied=False):
        pool, guard = [], 0
        while len(pool) < n_pool and guard < n_pool * 80:
            guard += 1
            a, b = rng.choice(cand, 2, replace=False)
            k = (a, b) if a < b else (b, a)
            if k in any_gold_pair:
                continue
            if both_studied and not (deg13.get(a, 0) > 0 and deg13.get(b, 0) > 0):
                continue
            pool.append(k)
        return pool

    pool_rand = sample_pool(False)
    pool_stud = sample_pool(True)
    pool_valid = [k for k in allev_pairs if k not in any_gold_pair]
    print(f"negative pools: random {len(pool_rand)}, studied {len(pool_stud)}, "
          f"validated {len(pool_valid)}")

    def split(pool):
        arr = np.array(sorted(pool), dtype=object)
        perm = rng.permutation(len(arr))
        n_te = int(0.2 * len(arr))
        te = [tuple(x) for x in arr[perm[:n_te]]]
        tr = [tuple(x) for x in arr[perm[n_te:]]]
        return tr, te

    neg_tr, neg_te = split(pool_rand)
    stud_tr, stud_te = split(pool_stud)
    valid_tr, valid_te = split(pool_valid)

    pool_rows = ([{"a": a, "b": b, "pool": "random", "split": "train"}
                  for a, b in neg_tr]
                 + [{"a": a, "b": b, "pool": "random", "split": "test"}
                    for a, b in neg_te]
                 + [{"a": a, "b": b, "pool": "studied", "split": "train"}
                    for a, b in stud_tr]
                 + [{"a": a, "b": b, "pool": "studied", "split": "test"}
                    for a, b in stud_te]
                 + [{"a": a, "b": b, "pool": "validated", "split": "train"}
                    for a, b in valid_tr]
                 + [{"a": a, "b": b, "pool": "validated", "split": "test"}
                    for a, b in valid_te])
    pd.DataFrame(pool_rows).to_csv(f"{V11}/results/t10_pools.csv", index=False)

    # ---------------- gene feature matrices ----------------
    sys.path.insert(0, f"{U.ROOT}/crisprsl")
    sys.path.insert(0, U.ROOT)
    from data_loader import CrisprSLDataLoader
    import config as cfg
    seeds = list(cfg.HR_SEED_GENES)
    data = CrisprSLDataLoader().load_all()
    saarf_df = data["saarf_df"].copy()

    mats = U.build_matrices(scores)
    X_crispr_g = mats["CRISPR_Only"]
    crispr_dim = X_crispr_g.shape[1]
    gmat_crispr = {g: X_crispr_g[i] for i, g in enumerate(cand)}

    Xs, _ = features_on_graph(graph_string2021(), cand, seeds)
    Xh, _ = features_on_graph(graph_hetionet(), cand, seeds)
    net_cur = {g: Xs[i] for i, g in enumerate(cand)}
    net_16 = {g: Xh[i] for i, g in enumerate(cand)}

    def saarf_gmat(df):
        cols = [c for c in df.columns if c != "gene" and
                df[c].dtype in ("float32", "float64", "int32", "int64")]
        vals = df[cols].to_numpy(dtype=float)
        m = {g: np.nan_to_num(vals[i], nan=0.0)
             for i, g in enumerate(df["gene"])}
        return m, cols

    full_g, full_cols = saarf_gmat(saarf_df)
    nosldb_df = saarf_df.drop(
        columns=[c for c in M.SLDB_COLUMNS if c in saarf_df.columns])
    nosldb_g, nosldb_cols = saarf_gmat(nosldb_df)
    topo_idx = [i for i, c in enumerate(full_cols)
                if c not in M.KNOWLEDGE_COLUMNS]
    topo_g = {g: full_g[g][topo_idx] for g in cand}

    Xrow16 = {g: Xh[i] for i, g in enumerate(cand)}
    saarf_v = saarf_df.copy()
    for col, slot in SWITCH_MAP.items():
        if col in saarf_v.columns:
            saarf_v[col] = saarf_v["gene"].map(lambda g, sl=slot: Xrow16[g][sl])
    vint_df = saarf_v.drop(columns=[c for c in M.KNOWLEDGE_COLUMNS
                                    if c in saarf_v.columns])
    vint_g, vint_cols = saarf_gmat(vint_df)

    gmat_full = {g: full_g.get(g, np.zeros(len(full_cols))) for g in cand}
    gmat_nosldb = {g: nosldb_g.get(g, np.zeros(len(nosldb_cols))) for g in cand}
    gmat_topo = {g: topo_g.get(g, np.zeros(len(topo_idx))) for g in cand}
    gmat_vint = {g: vint_g.get(g, np.zeros(len(vint_cols))) for g in cand}
    fusion_cur = {g: np.concatenate([gmat_crispr[g], net_cur[g]]) for g in cand}
    fusion_16 = {g: np.concatenate([gmat_crispr[g], net_16[g]]) for g in cand}

    posE_list = list(zip(posE["a"], posE["b"]))
    posE_year = dict(zip(posE_list, posE["year"]))
    posE_npmid = dict(zip(posE_list, posE["n_pmids"]))
    posB_list = list(zip(posB["a"], posB["b"]))
    posB_year = dict(zip(posB_list, posB["year"]))

    rows, sens_rows = [], []

    def add_row(method, cutoff, ev, extra=None):
        r = {"method": method, "cutoff": cutoff, **ev}
        if extra:
            r.update(extra)
        rows.append(r)

    def train_test(cutoff, pool_tr=None, pool_te=None, positives=None,
                   pos_year=None, twin=False):
        positives = posE_list if positives is None else positives
        pos_year = posE_year if pos_year is None else pos_year
        ptr = neg_tr if pool_tr is None else pool_tr
        pte = neg_te if pool_te is None else pool_te
        tr_pos = posE_list if twin else [k for k in positives
                                         if pos_year[k] <= cutoff]
        te_pos = [k for k in positives if pos_year[k] > cutoff]
        tr = tr_pos + list(ptr)
        te = te_pos + list(pte)
        ytr = np.array([1] * len(tr_pos) + [0] * len(ptr))
        yte = np.array([1] * len(te_pos) + [0] * len(pte))
        return tr, te, ytr, yte, len(te_pos)

    def fit_eval_xgb(name, gmat, dim, cutoff, twin=False):
        tr, te, ytr, yte, n_tepos = train_test(cutoff, twin=twin)
        m = XGBClassifier(**U.GBM_PARAMS).fit(build_pair_X(tr, gmat, dim), ytr)
        s = m.predict_proba(build_pair_X(te, gmat, dim))[:, 1]
        return eval_pairs(yte, s, seed=SEED + cutoff), yte, s, n_tepos

    def fit_eval_svm(name, gmat, dim, cutoff):
        tr, te, ytr, yte, n_tepos = train_test(cutoff)
        sc = StandardScaler().fit(build_pair_X(tr, gmat, dim))
        svm = LinearSVC(dual="auto", random_state=SEED, max_iter=8000)
        svm.fit(sc.transform(build_pair_X(tr, gmat, dim)), ytr)
        s = svm.decision_function(sc.transform(build_pair_X(te, gmat, dim)))
        return eval_pairs(yte, s, seed=SEED + cutoff), n_tepos

    def retro_cv(gmat, dim):
        allp = posE_list + list(neg_te) + list(neg_tr)
        X = build_pair_X(allp, gmat, dim)
        y = np.array([1] * len(posE_list) + [0] * len(neg_te + neg_tr))
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        aucs = []
        for tri, tei in skf.split(X, y):
            m = XGBClassifier(**U.GBM_PARAMS).fit(X[tri], y[tri])
            aucs.append(roc_auc_score(y[tei], m.predict_proba(X[tei])[:, 1]))
        return round(float(np.mean(aucs)), 4)

    print("  retro CV (pair, current features)...")
    retro = {"PPI pair": retro_cv(net_cur, 3),
             "CRISPR pair": retro_cv(gmat_crispr, crispr_dim),
             "Fusion pair": retro_cv(fusion_cur, crispr_dim + 3)}
    print("  retro:", retro)

    # ---- label-free PPR pair scorer, current vs 2016
    for cname, gmat in (("current", net_cur), ("v2016", net_16)):
        for c in CUTOFFS:
            _, te, _, yte, n_tepos = train_test(c)
            s = np.array([np.sqrt(max(gmat[a][1], 0) * max(gmat[b][1], 0))
                          for a, b in te])
            add_row(f"Label-free PPR pair ({cname})", c,
                    eval_pairs(yte, s, seed=SEED + c), {"n_test_pos": n_tepos})

    # ---- XGBoost pair models
    models = [
        ("PPI pair (STRING 2021)", net_cur, 3, "PPI pair"),
        ("PPI pair (Hetionet 2016)", net_16, 3, None),
        ("CRISPR pair", gmat_crispr, crispr_dim, "CRISPR pair"),
        ("Fusion pair (current)", fusion_cur, crispr_dim + 3, "Fusion pair"),
        ("Fusion pair (2016 networks)", fusion_16, crispr_dim + 3, None),
    ]
    for name, gmat, dim, rkey in models:
        for c in CUTOFFS:
            ev, _, _, n_tepos = fit_eval_xgb(name, gmat, dim, c)
            add_row(name, c, ev, {"retro": retro.get(rkey) if rkey else None})
            print(f"  {name} @{c}: {ev['auroc']} [{ev['ci_lo']},{ev['ci_hi']}] "
                  f"n+={n_tepos}")

    # ---- SINaTRA-style pair SVM variants
    svm_models = [
        ("SINaTRA-style pair (full inputs)", gmat_full, len(full_cols)),
        ("SINaTRA-style pair (−SL counts)", gmat_nosldb, len(nosldb_cols)),
        ("SINaTRA-style pair (topology+omics)", gmat_topo, len(topo_idx)),
        ("SINaTRA-style pair (topo, 2016 networks)", gmat_vint, len(vint_cols)),
    ]
    for name, gmat, dim in svm_models:
        for c in CUTOFFS:
            ev, n_tepos = fit_eval_svm(name, gmat, dim, c)
            add_row(name, c, ev, {"n_test_pos": n_tepos})
            print(f"  {name} @{c}: {ev['auroc']} [{ev['ci_lo']},{ev['ci_hi']}]")

    # ---- label-side twin (all-gold pairs into training)
    for name, gmat, dim in (("PPI pair (STRING 2021)", net_cur, 3),
                            ("Fusion pair (current)", fusion_cur,
                             crispr_dim + 3),
                            ("CRISPR pair", gmat_crispr, crispr_dim)):
        for c in CUTOFFS:
            ev_h, yte, s_h, _ = fit_eval_xgb(name, gmat, dim, c)
            ev_t, _, s_t, _ = fit_eval_xgb(name, gmat, dim, c, twin=True)
            d, lo, hi, p = paired_pair_test(yte, s_h, s_t, seed=SEED + 97 + c)
            add_row(f"{name} — all-gold twin", c, ev_t,
                    {"label_increment": round(d, 4), "p": round(p, 4)})
            print(f"  TWIN {name} @{c}: {ev_t['auroc']} "
                  f"(label increment {d:+.3f}, p={p:.3f})")

    # ---- absorption tests on identical test rows (2016 cutoff)
    for cond, ga, gb, dim in (
            ("PPI pair: STRING 2021 minus Hetionet 2016", net_cur, net_16, 3),
            ("Fusion pair: current minus 2016 networks", fusion_cur,
             fusion_16, crispr_dim + 3)):
        tr, te, ytr, yte, n_tepos = train_test(2016)
        m1 = XGBClassifier(**U.GBM_PARAMS).fit(build_pair_X(tr, ga, dim), ytr)
        m2 = XGBClassifier(**U.GBM_PARAMS).fit(build_pair_X(tr, gb, dim), ytr)
        s1 = m1.predict_proba(build_pair_X(te, ga, dim))[:, 1]
        s2 = m2.predict_proba(build_pair_X(te, gb, dim))[:, 1]
        d, lo, hi, p = paired_pair_test(yte, s2, s1, seed=SEED + 31)
        sens_rows.append({"analysis": "pair absorption", "condition": cond,
                          "delta": round(d, 4), "ci_lo": round(lo, 4),
                          "ci_hi": round(hi, 4), "p": round(p, 4),
                          "n_test_pos": n_tepos})
        print(f"  ABSORPTION {cond}: {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}")

    # ---- sensitivity: >=2-pmid positives
    strict = [k for k in posE_list if posE_npmid[k] >= 2]
    strict_year = {k: posE_year[k] for k in strict}
    tr, te, ytr, yte, n_tepos = train_test(2016, positives=strict,
                                           pos_year=strict_year)
    m = XGBClassifier(**U.GBM_PARAMS).fit(build_pair_X(tr, net_cur, 3), ytr)
    ev = eval_pairs(yte, m.predict_proba(build_pair_X(te, net_cur, 3))[:, 1],
                    seed=SEED + 7)
    sens_rows.append({"analysis": "positives >=2 pmids",
                      "condition": "PPI pair (STRING 2021) @2016",
                      "auroc": ev["auroc"], "ci_lo": ev["ci_lo"],
                      "ci_hi": ev["ci_hi"], "n_test_pos": n_tepos})
    print(f"  SENS >=2-pmid positives @2016: {ev['auroc']} n+={n_tepos}")

    # ---- sensitivity: negative composition at 2016
    for pool_name, ptr, pte in (("studied negatives", stud_tr, stud_te),
                                ("validated (alleviating) negatives",
                                 valid_tr, valid_te)):
        if not pte:
            continue
        tr, te, ytr, yte, n_tepos = train_test(2016, pool_tr=ptr, pool_te=pte)
        m = XGBClassifier(**U.GBM_PARAMS).fit(build_pair_X(tr, net_cur, 3), ytr)
        s = m.predict_proba(build_pair_X(te, net_cur, 3))[:, 1]
        ev = eval_pairs(yte, s, seed=SEED + 13)
        sens_rows.append({"analysis": "negative composition",
                          "condition": f"PPI pair (STRING 2021) @2016, "
                                       f"{pool_name}",
                          "auroc": ev["auroc"], "ci_lo": ev["ci_lo"],
                          "ci_hi": ev["ci_hi"], "n_test_pos": n_tepos,
                          "n_test_neg": len(pte)})
        print(f"  SENS {pool_name}: {ev['auroc']} "
              f"(n+={n_tepos}, n-={len(pte)})")

    # ---- sensitivity: channel B positives (BioGRID SL/SGD, PubMed-dated)
    for c in CUTOFFS:
        tr, te, ytr, yte, n_tepos = train_test(c, positives=posB_list,
                                               pos_year=posB_year)
        if n_tepos < 10:
            print(f"  channel B @{c}: only {n_tepos} test pairs, skipped")
            continue
        outs = {}
        for vname, gmat in (("current", net_cur), ("2016", net_16)):
            m = XGBClassifier(**U.GBM_PARAMS).fit(build_pair_X(tr, gmat, 3),
                                                  ytr)
            outs[vname] = m.predict_proba(build_pair_X(te, gmat, 3))[:, 1]
        ev = eval_pairs(yte, outs["current"], seed=SEED + 17 + c)
        d, lo, hi, p = paired_pair_test(yte, outs["2016"], outs["current"],
                                        seed=SEED + 31 + c)
        sens_rows.append({"analysis": "channel B positives (BioGRID)",
                          "condition": f"PPI pair @{c}",
                          "auroc": ev["auroc"], "ci_lo": ev["ci_lo"],
                          "ci_hi": ev["ci_hi"], "n_test_pos": n_tepos,
                          "absorption_delta": round(d, 4), "p": round(p, 4)})
        print(f"  channel B @{c}: AUROC {ev['auroc']} absorption {d:+.4f} "
              f"(p={p:.4f}) n+={n_tepos}")

    pd.DataFrame(rows).to_csv(f"{V11}/results/t10_pair_leaderboard.csv",
                              index=False)
    pd.DataFrame(sens_rows).to_csv(f"{V11}/results/t10_pair_sensitivity.csv",
                                   index=False)
    summary = {
        "design": ("pair-level temporal evaluation; positives dated by first "
                   "experimental evidence; negatives never contain all-time "
                   "SL-direction pairs; shared 80/20 negative split (seed 42)"),
        "n_pairs_channelE": int(len(posE)),
        "n_pairs_channelB": int(len(posB)),
        "n_neg_random": len(pool_rand), "n_neg_studied": len(pool_stud),
        "n_neg_validated": len(pool_valid),
        "retro_cv": retro,
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    with open(f"{V11}/results/t10_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nT10 done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
