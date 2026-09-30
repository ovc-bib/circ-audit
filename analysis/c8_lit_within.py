"""
c8_lit_within.py — literature channel, within-instrument vintage (C8)
======================================================================
Table 3's literature-channel row (+0.21) compares two different instruments
(vintage-clean BioGRID degree vs today's lit_score).  Here the SAME
instrument, PubMed publication counts per gene symbol, is evaluated at two
date windows (pre-2017 vs all-time) on the standard test rows, so the delta
is a within-instrument absorption estimate for the literature channel.

Output: results/c8_lit_within.csv
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/src")
import tm_utils as U
from a_class import paired_test

V11 = U.V11
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


def pubmed_count(gene, dp):
    term = f"{gene}[Gene Name] AND {dp}[dp]"
    url = (f"{EUTILS}?db=pubmed&retmax=0&retmode=json&term="
           + requests_quote(term))
    for attempt in range(3):
        r = subprocess.run(["curl", "-s", "--ssl-no-revoke", "--max-time",
                            "30", url], capture_output=True, text=True)
        try:
            return int(json.loads(r.stdout)["esearchresult"]["count"])
        except Exception:
            time.sleep(1 + attempt)
    return -1


def requests_quote(s):
    import urllib.parse
    return urllib.parse.quote(s)


def main():
    scores = U.load_scores()
    genes = scores["gene"].tolist()
    y = scores["y_gold"].to_numpy()
    _, new_pos, negatives = U.era_masks(scores, 2016)
    _, test_neg = U.shared_negative_split(negatives)
    te = np.concatenate([np.where(new_pos)[0], test_neg])
    te_genes = sorted({genes[i] for i in te})
    print(f"test-row gene set: {len(te_genes)}")

    cache_path = f"{V11}/results/c8_pubmed_counts.csv"
    done = {}
    if os.path.exists(cache_path):
        prev = pd.read_csv(cache_path, dtype={"gene": str})
        done = {r.gene: (r.pubs_pre2017, r.pubs_all) for r in prev.itertuples()}
        print(f"resuming: {len(done)} genes already counted")
    rows = []
    todo = [g for g in te_genes if g not in done]
    for k, g in enumerate(todo):
        pre, alll = done.get(g, (pubmed_count(g, "1900:2016"),
                                 pubmed_count(g, "1900:2026")))
        rows.append({"gene": g, "pubs_pre2017": pre, "pubs_all": alll})
        if k % 50 == 0:
            print(f"  {k}/{len(todo)} last={g} pre={pre} all={alll} "
                  f"({time.strftime('%H:%M:%S')})", flush=True)
            merged = pd.DataFrame(rows + [{"gene": gg, **dict(zip(
                ("pubs_pre2017", "pubs_all"), done[gg]))} for gg in done])
            merged.to_csv(cache_path, index=False)
        time.sleep(0.45)
    merged = pd.DataFrame(rows + [{"gene": gg, **dict(zip(
        ("pubs_pre2017", "pubs_all"), done[gg]))} for gg in done])
    merged.drop_duplicates(subset="gene", keep="last").to_csv(cache_path,
                                                              index=False)

    cnt = pd.read_csv(cache_path, dtype={"gene": str}).set_index("gene")
    cnt = cnt.reindex(genes).fillna(0)
    pre = np.log1p(np.maximum(cnt["pubs_pre2017"].to_numpy(dtype=float), 0))
    cur = np.log1p(np.maximum(cnt["pubs_all"].to_numpy(dtype=float), 0))

    out = []
    for name, f in (("PubMed count pre-2017", pre), ("PubMed count all-time", cur)):
        auc, lo, hi = U.auroc_ci(y[te], f[te], seed=U.SEED)
        out.append({"instrument": name, "auroc": round(auc, 4),
                    "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
        print(f"  {name}: AUROC={auc:.4f} [{lo:.3f},{hi:.3f}]")
    d, lo, hi, p = paired_test(y[te], pre[te], cur[te])
    out.append({"instrument": "absorption pre-2017 -> all-time",
                "auroc": round(d, 4), "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4), "p_paired_bootstrap": round(p, 4)})
    print(f"  within-instrument absorption: Δ={d:+.4f} [{lo:+.4f},{hi:+.4f}] "
          f"p={p:.4f}")
    pd.DataFrame(out).to_csv(f"{V11}/results/c8_lit_within.csv", index=False)


if __name__ == "__main__":
    main()
