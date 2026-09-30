"""
t9_modality_check.py — CRISPR discovery-modality confound check (T9)
====================================================================
The paper credits the CRISPR feature family with the only prospective signal
that beats its own retrospective value.  The alternative reading is that the
2017-2019 test window was itself discovered largely by CRISPR screens, so
CRISPR-effect features partly predict the discovery instrument rather than
the biology.  Two tests:

  T9a  Prevalence.  Classify each gold gene's first experimental evidence by
       modality (source naming CRISPR/CRISPRi or not) and cross-tabulate
       against future (>2016) versus past (<=2016) status, Fisher exact.
  T9b  Ranking.  Among the 58 future discoveries, compare discovery-list
       ranks of the CRISPR-only model between CRISPR-anchored and other
       genes (Mann-Whitney), with the honest 2016-vintage fusion as control.

Outputs: results/t9_modality_check.csv, t9_stats.json
"""
import json
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

sys.path.insert(0, "E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/src")
import tm_utils as U

HS = "E:/崔雷/博士/设计/ovc_project/data/Human_SL.csv"
V10, V11 = U.V10, U.V11
NONEXP = ("Computational Prediction", "Text Mining")


def fetch_years(pmids, cache):
    missing = [p for p in pmids if p not in cache]
    for i in range(0, len(missing), 200):
        batch = missing[i:i + 200]
        url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
               f"?db=pubmed&id={','.join(batch)}&retmode=json")
        for attempt in range(3):
            r = subprocess.run(["curl", "-s", "--ssl-no-revoke", "--max-time",
                                "60", url], capture_output=True, text=True)
            try:
                d = json.loads(r.stdout)["result"]
                break
            except Exception:
                time.sleep(3 + attempt * 3)
        else:
            continue
        for uid in d.get("uids", []):
            year = int("".join(c for c in d[uid].get("pubdate", "")[:4]
                               if c.isdigit()) or 0)
            if year:
                cache[uid] = year
        time.sleep(0.4)
    return cache


def main():
    scores = U.load_scores()
    y = scores["y_gold"].to_numpy()
    old_pos, new_pos, _ = U.era_masks(scores, 2016)

    hs = pd.read_csv(HS, dtype={"r.pubmed_id": str})
    hs["pmid_list"] = hs["r.pubmed_id"].map(lambda s: re.findall(r"\d{6,}", str(s)))
    hs_exp = hs[~hs["r.source"].str.contains("|".join(NONEXP), regex=True)
                & hs["pmid_list"].map(len) > 0].copy()

    cache_df = pd.read_csv(f"{V10}/data/pubmed_years.csv", dtype={"pmid": str})
    cache = {str(r.pmid): int(r.year) for r in cache_df.itertuples()
             if pd.notna(r.year)}
    involved = {p for lst in hs_exp["pmid_list"] for p in lst}
    cache = fetch_years(involved, cache)
    pd.DataFrame({"pmid": list(cache), "year": [cache[p] for p in cache]}
                 ).to_csv(f"{V10}/data/pubmed_years.csv", index=False)

    # per gene: earliest experimental year, modalities at that year, any-CRISPR
    raw = {}
    for _, r in hs_exp.iterrows():
        years = [cache[p] for p in r["pmid_list"] if p in cache and cache[p] > 1900]
        if not years:
            continue
        ymin = min(years)
        src = str(r["r.source"])
        for g in (r["n1.name"], r["n2.name"]):
            raw.setdefault(g, []).append((ymin, src))
    gene_rows = {}
    for g, lst in raw.items():
        ymin = min(y for y, _ in lst)
        gene_rows[g] = {"ymin": ymin,
                        "srcs_at_ymin": {s for y, s in lst if y == ymin},
                        "any_crispr": any("CRISPR" in s.upper() for _, s in lst)}

    def crispr_anchored(g):
        d = gene_rows.get(g)
        if not d:
            return np.nan
        return any("CRISPR" in s.upper() for s in d["srcs_at_ymin"])

    # ---------- T9a prevalence (unique genes, shipped years) ----------
    split = pd.read_csv(f"{V10}/data/gold_temporal_split.csv")
    yr = dict(zip(split["gene"], split["first_year_experimental"]))
    genes_all = sorted(set(scores["gene"].to_numpy()[y == 1]))
    tab = pd.DataFrame({
        "gene": genes_all,
        "year": [yr.get(g, np.nan) for g in genes_all],
        "crispr_first": [crispr_anchored(g) for g in genes_all],
        "crispr_any": [gene_rows.get(g, {}).get("any_crispr", False)
                       for g in genes_all],
    }).dropna(subset=["year", "crispr_first"])
    tab["future"] = tab["year"] > 2016

    out = {}
    for defn in ("crispr_first", "crispr_any"):
        ct = pd.crosstab(tab["future"], tab[defn])
        odds, p = fisher_exact(ct.to_numpy())
        frac_future = tab.loc[tab.future, defn].mean()
        frac_past = tab.loc[~tab.future, defn].mean()
        out[f"{defn}_future_frac"] = round(float(frac_future), 3)
        out[f"{defn}_past_frac"] = round(float(frac_past), 3)
        out[f"{defn}_oddsratio"] = round(float(odds), 2)
        out[f"{defn}_fisher_p"] = float(p)
        print(f"[{defn}] CRISPR-anchored share  future={frac_future:.2f} "
              f"past={frac_past:.2f}  OR={odds:.2f}  Fisher p={p:.2e}")
        print(ct)

    # modality mix by year band (context for the reviewer)
    def modality(g):
        d = gene_rows.get(g)
        if not d:
            return "undated"
        s = "|".join(d["srcs_at_ymin"]).upper()
        if "CRISPR" in s:
            return "CRISPR"
        if "GENOMERNAI" in s or "RNAI" in s:
            return "RNAi"
        return "other"
    tab["modality"] = tab["gene"].map(modality)
    mix = pd.crosstab(pd.cut(tab["year"], [1984, 2012, 2016, 2020],
                             labels=["<=2012", "2013-16", "2017-19"]),
                      tab["modality"])
    print("\nmodality mix by discovery era:")
    print(mix)
    out["modality_mix"] = mix.to_dict()

    # ---------- T9b ranking ----------
    t7 = pd.read_csv(f"{V11}/results/t7_per_gene_scores.csv")
    pool = ~old_pos
    res = {}
    for model in ("CRISPR_Only", "Fusion_vintage2016", "PPI_vintage2016"):
        s = t7[model].to_numpy()
        pool_idx = np.where(pool)[0]
        ranked = pool_idx[np.argsort(-s[pool])]
        rank = np.full(len(scores), -1, dtype=int)
        rank[ranked] = np.arange(1, len(ranked) + 1)
        genes = scores["gene"].to_numpy()
        rows = [(genes[i], rank[i]) for i in np.where(new_pos)[0]]
        df = pd.DataFrame(rows, columns=["gene", "rank"])
        df["crispr_first"] = df["gene"].map(
            lambda g: crispr_anchored(g) if pd.notna(crispr_anchored(g)) else False)
        a = df.loc[df.crispr_first, "rank"].dropna()
        b = df.loc[~df.crispr_first, "rank"].dropna()
        if len(a) >= 3 and len(b) >= 3:
            stat, p = mannwhitneyu(a, b, alternative="less")
            res[model] = {"n_crispr": int(len(a)), "n_other": int(len(b)),
                          "median_rank_crispr": int(np.median(a)),
                          "median_rank_other": int(np.median(b)),
                          "mw_U": float(stat), "p_one_sided": float(p)}
            print(f"\n[{model}] CRISPR-anchored n={len(a)} median={np.median(a):.0f}"
                  f"  other n={len(b)} median={np.median(b):.0f}"
                  f"  MW one-sided p={p:.3f}")
        else:
            print(f"\n[{model}] insufficient split n=({len(a)},{len(b)})")
        df.to_csv(f"{V11}/results/t9_ranks_{model}.csv", index=False)
    out["ranking"] = res

    tab.to_csv(f"{V11}/results/t9_modality_check.csv", index=False)
    with open(f"{V11}/results/t9_stats.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
