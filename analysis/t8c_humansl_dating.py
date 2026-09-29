"""
t8c_humansl_dating.py — third-source dating check, stage 3
===========================================================
Human_SL.csv (the HumanSL arm of the gold standard) turns out to carry
r.pubmed_id per evidence row, and its triples are 99.1% identical to the
shipped dating evidence file, so the "undated third source" of the T8
limitation is in fact the same resource the dating was built from.  This
stage makes that statement exact.  It (a) re-derives the experimental flag
from r.source alone (non-experimental iff the source string mentions
Computational Prediction or Text Mining, learned from the 35,771 overlapping
triples where both files agree), (b) rebuilds first-evidence years from
Human_SL.csv directly, (c) merges all channels and recounts the
test-to-train movement per cutoff.

Outputs: results/t8c_humansl_dating.csv, t8c_stats.json
"""
import json
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/src")
import tm_utils as U

HS = "E:/崔雷/博士/设计/ovc_project/data/Human_SL.csv"
V10, V11 = U.V10, U.V11
NONEXP_MARKERS = ("Computational Prediction", "Text Mining")


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
    gold_genes = set(scores["gene"].to_numpy()[y == 1])

    hs = pd.read_csv(HS, dtype={"r.pubmed_id": str})
    hs["experimental_hs"] = ~hs["r.source"].str.contains(
        "|".join(NONEXP_MARKERS), regex=True)
    # r.pubmed_id holds single ids and multi-id strings ("a/b", "a;b")
    hs["pmid_list"] = hs["r.pubmed_id"].map(
        lambda s: re.findall(r"\d{6,}", str(s)))
    hs_exp = hs[hs["experimental_hs"] & hs["pmid_list"].map(len) > 0]

    cache_df = pd.read_csv(f"{V10}/data/pubmed_years.csv", dtype={"pmid": str})
    cache = {str(r.pmid): int(r.year) for r in cache_df.itertuples()
             if pd.notna(r.year)}
    involved = {p for lst in hs_exp["pmid_list"] for p in lst}
    print(f"HS experimental pmids={len(involved)} cached={len(cache)}")
    cache = fetch_years(involved, cache)
    pd.DataFrame({"pmid": list(cache), "year": [cache[p] for p in cache]}
                 ).to_csv(f"{V10}/data/pubmed_years.csv", index=False)
    yr = cache

    hs_year = {}
    for g in gold_genes:
        rows_g = hs_exp[(hs_exp["n1.name"] == g) | (hs_exp["n2.name"] == g)]
        ys = [yr[p] for lst in rows_g["pmid_list"] for p in lst
              if p in yr and yr[p] > 1900]
        if ys:
            hs_year[g] = min(ys)

    t8 = pd.read_csv(f"{V11}/results/t8_biogrid_dating.csv")  # per gold gene
    base = dict(zip(t8["gene"], t8["synlethdb_year"]))
    bg = dict(zip(t8["gene"], t8["biogrid_year"]))

    rows = []
    for g in sorted(gold_genes):
        sy, by, hy = base.get(g, np.nan), bg.get(g, np.nan), hs_year.get(g, np.nan)
        cand = [x for x in (sy, by, hy) if pd.notna(x)]
        mv = min(cand) if cand else np.nan
        rows.append({"gene": g, "shipped_year": sy, "biogrid_year": by,
                     "humansl_direct_year": hy, "merged3_year": mv})
    df = pd.DataFrame(rows)
    df.to_csv(f"{V11}/results/t8c_humansl_dating.csv", index=False)

    diff = df[(df["humansl_direct_year"].notna()) &
              (df["shipped_year"].notna()) &
              (df["humansl_direct_year"] != df["shipped_year"])]
    print(f"gold genes dated by HS-direct: {df['humansl_direct_year'].notna().sum()}")
    print(f"HS-direct vs shipped disagree on {len(diff)} genes")
    if len(diff):
        print(diff.to_string(index=False))

    stats = {"n_gold": len(df),
             "n_hs_direct": int(df["humansl_direct_year"].notna().sum()),
             "n_hs_direct_earlier_than_shipped": int(
                 (df["humansl_direct_year"] < df["shipped_year"]).sum()),
             "n_disagree": int(len(diff))}
    per_gene = dict(zip(df["gene"], df["merged3_year"]))
    for c in (2012, 2014, 2016):
        moved = [r.gene for r in df.itertuples()
                 if pd.notna(r.shipped_year) and r.shipped_year > c
                 and pd.notna(r.merged3_year) and r.merged3_year <= c]
        stats[f"moved_test_to_train @{c}"] = len(moved)
        stats[f"moved_genes @{c}"] = moved
    with open(f"{V11}/results/t8c_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items()
                      if not k.startswith("moved_genes")}, indent=2))


if __name__ == "__main__":
    main()
