"""
c8_shard.py — shard worker for C8 (run 4 instances with shard ids 0-3).
Each worker counts PubMed publications (two date windows) for its share of
the test-row genes, keeping the combined request rate under NCBI's 3/s.
Usage: python c8_shard.py <shard_id>
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
from c8_lit_within import pubmed_count

V11 = U.V11
SHARD = int(sys.argv[1])
CACHE = f"{V11}/results/c8_pubmed_counts.csv"
SHARD_F = f"{V11}/results/c8_shard_{SHARD}.csv"


def main():
    scores = U.load_scores()
    genes = scores["gene"].tolist()
    _, new_pos, negatives = U.era_masks(scores, 2016)
    _, test_neg = U.shared_negative_split(negatives)
    te = np.concatenate([np.where(new_pos)[0], test_neg])
    te_genes = sorted({genes[i] for i in te})
    done = set()
    if os.path.exists(CACHE):
        done = set(pd.read_csv(CACHE, dtype={"gene": str})["gene"])
    todo = [g for i, g in enumerate(te_genes)
            if i % 4 == SHARD and g not in done]
    print(f"shard {SHARD}: {len(todo)} genes to count")
    rows = []
    for k, g in enumerate(todo):
        rows.append({"gene": g,
                     "pubs_pre2017": pubmed_count(g, "1900:2016"),
                     "pubs_all": pubmed_count(g, "1900:2026")})
        if k % 25 == 0:
            print(f"  shard{SHARD} {k}/{len(todo)} ({time.strftime('%H:%M:%S')})",
                  flush=True)
            pd.DataFrame(rows).to_csv(SHARD_F, index=False)
        time.sleep(0.45)
    pd.DataFrame(rows).to_csv(SHARD_F, index=False)
    print(f"shard {SHARD} done ({len(rows)})")


if __name__ == "__main__":
    main()
