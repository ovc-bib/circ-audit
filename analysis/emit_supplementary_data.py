"""
emit_supplementary_data.py — Supplementary Data S1 for the revision (R2 m3)
============================================================================
Emits the dated gene and pair gold standards, the pair negative pools and
the per-gene / per-pair score tables promised in the revised Data and code
availability section and Supplementary Data S1 list.

Output: manuscript/supplementary_data/*.csv
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/src")
import tm_utils as U
from t10_pair_level import export_pairs, biogrid_genetic

V11 = U.V11
OUT = f"{V11}/manuscript/supplementary_data"
os.makedirs(OUT, exist_ok=True)


def main():
    # 1. dated gene gold standard
    scores = U.load_scores()
    split = pd.read_csv(f"{U.V10}/data/gold_temporal_split.csv")
    gs = scores[["gene", "y_gold"]].merge(
        split[["gene", "first_year_experimental", "first_year_any",
               "n_evidence_rows"]], on="gene", how="left")
    gs.to_csv(f"{OUT}/dated_gene_gold_standard.csv", index=False)
    print(f"dated_gene_gold_standard.csv: {len(gs)} rows, "
          f"{int(gs['y_gold'].sum())} gold")

    # 2. dated pair gold lists (channel E + channel B)
    cand = set(scores["gene"])
    posE = export_pairs(cand)
    posE.to_csv(f"{OUT}/dated_pair_gold_channel_E.csv", index=False)
    print(f"dated_pair_gold_channel_E.csv: {len(posE)} pairs")
    posB, _ = biogrid_genetic(cand)
    posB.to_csv(f"{OUT}/dated_pair_gold_channel_B.csv", index=False)
    print(f"dated_pair_gold_channel_B.csv: {len(posB)} pairs")

    # 3. pools and result tables
    for f in ("t10_pools.csv", "t10_pair_leaderboard.csv",
              "t10_pair_sensitivity.csv", "t12_gat_pair.csv",
              "t11_rich_network.csv", "t11_summary.json",
              "t7_per_gene_scores.csv",
              "t1_prospective_leaderboard.csv"):
        src = f"{V11}/results/{f}"
        if os.path.exists(src):
            shutil.copy(src, f"{OUT}/{f if f != 't7_per_gene_scores.csv' else 'per_gene_scores.csv'}")
            print(f"copied {f}")


if __name__ == "__main__":
    main()
