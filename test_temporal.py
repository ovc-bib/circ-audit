"""
test_temporal.py — end-to-end tests of the temporal-evaluation instruments
==========================================================================
Synthetic data, no external files. Verifies the dating pipeline, the three
snapshot rebuilders, the pair transformations and the rank-stability /
absorption diagnostics against hand-computed values.
"""

import numpy as np
import pandas as pd

from circ_audit_tool.temporal import (ALPHA, absorption, auroc_pairwise,
                                      bfs_proximity, build_snapshot_features,
                                      cutoff_split, first_evidence_year,
                                      pair_transform,
                                      personalised_pagerank,
                                      rank_normalised_degree, rank_stability)

PASS = 0
FAIL = 0


def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"PASS  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}")


# ---------------------------------------------------------------- dating ----
evidence = pd.DataFrame({
    "gene": ["BRCA1", "BRCA1", "RAD51", "PALB2", "RAD51", "BRCA2"],
    "year": [1995, 2014, 2010, 2018, 2003, 2016],
    "source": ["experimental", "experimental", "prediction",
               "text-mining", "experimental", "experimental"],
})
years = first_evidence_year(evidence)
check("dating: min experimental year", years["BRCA1"] == 1995)
check("dating: prediction row excluded", years["RAD51"] == 2003)
check("dating: text-mining row excluded", "PALB2" not in years.index)
split = cutoff_split(years, 2010)
check("dating: cutoff split", split["BRCA1"] == "train" and split["BRCA2"] == "test")

# -------------------------------------------------------------- snapshots ---
nodes = ["S1", "S2", "A", "B", "C", "D"]
edges = pd.DataFrame([
    ("S1", "S2"), ("S1", "A"), ("S2", "A"), ("A", "B"),
    ("B", "C"), ("C", "D"),
], columns=["u", "v"])
seeds = ["S1", "S2"]

d_hat = rank_normalised_degree(edges, nodes)
# degrees: S1=2, S2=2, A=3, B=2, C=2, D=1 -> ascending ranks of raw degree
deg = dict(zip(nodes, [2, 2, 3, 2, 2, 1]))
check("snapshot: degree feature in [0,1]",
      bool((d_hat >= 0).all() and (d_hat <= 1).all()))
check("snapshot: highest degree ranks highest",
      d_hat.idxmax() == "A" and d_hat.idxmin() == "D")

ppr = personalised_pagerank(edges, nodes, seeds)
check("snapshot: PageRank mass on seeds",
      ppr[seeds].sum() > ppr[["A", "B", "C", "D"]].sum())
check("snapshot: PageRank decreases with distance",
      ppr["A"] > ppr["B"] > ppr["D"])

bfs = bfs_proximity(edges, nodes, seeds)
check("snapshot: BFS distances", bfs["S1"] == 0 and bfs["A"] == 1
      and bfs["B"] == 2 and bfs["D"] == 4)
disconnected = pd.DataFrame([("S1", "S2"), ("X", "Y")],
                            columns=["u", "v"])
bfs_disc = bfs_proximity(disconnected, ["S1", "S2", "X", "Y"], seeds)
check("snapshot: disconnected genes get diameter",
      bfs_disc["X"] == bfs_disc.max())

table = build_snapshot_features(edges, nodes, seeds)
check("snapshot: three-feature table",
      list(table.columns) == ["degree_rank", "ppr", "bfs_proximity"]
      and table.shape == (6, 3))

x_u = np.array([1.0, 0.0, 2.0])
x_v = np.array([0.0, 1.0, 2.0])
phi = pair_transform(x_u, x_v)
cos = 4.0 / (np.sqrt(5.0) * np.sqrt(5.0))
check("pair transform: four blocks",
      len(phi) == 10 and np.isclose(phi[9], cos)
      and np.allclose(phi[0:3], x_u + x_v)
      and np.allclose(phi[3:6], np.abs(x_u - x_v))
      and np.allclose(phi[6:9], x_u * x_v))

# ------------------------------------------------------- rank stability -----
y_true = [1, 1, 0, 0]
check("auroc pairwise: perfect separation", auroc_pairwise(y_true, [0.9, 0.8, 0.3, 0.2]) == 1.0)
check("auroc pairwise: ties counted as half",
      auroc_pairwise(y_true, [0.5, 0.5, 0.5, 0.5]) == 0.5)

f_a = pd.Series({"g1": 1.0, "g2": 2.0, "g3": 3.0, "g4": 0.0})
f_same = pd.Series({"g1": 10.0, "g2": 20.0, "g3": 30.0, "g4": 0.0})
f_flip = pd.Series({"g1": 3.0, "g2": 2.0, "g3": 1.0, "g4": 0.0})
check("rank stability: monotone transform is 1.0", rank_stability(f_a, f_same) == 1.0)
check("rank stability: reversal is -1.0", rank_stability(f_a, f_flip) == -1.0)
check("absorption: difference of AUROCs",
      np.isclose(absorption(0.693, 0.789), 0.096))

print(f"\ntemporal tests: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
