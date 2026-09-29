"""
t2c_degree_matrix.py — degree across five network snapshots + pairwise rank-stability matrix
============================================================================================
Per-candidate degree on BioGRID 3.2.121/3.4.136/4.4.249, Hetionet 2016 and
STRING 2021, plus the 5x5 Spearman rank-stability matrix (paper Fig 4d).

Output: results/t2c_degree_by_network.csv, results/t2c_rank_stability.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tm_utils as U
from t2_vintage_chain import graph_biogrid, graph_hetionet, graph_string2021


def main():
    scores = U.load_scores()
    genes = scores["gene"].tolist()

    graphs = {
        "BioGRID13": graph_biogrid("3.2.121"),
        "BioGRID15": graph_biogrid("3.4.136"),
        "Hetionet16": graph_hetionet(),
        "STRING21": graph_string2021(),
        "BioGRID24": graph_biogrid("4.4.249"),
    }
    deg = pd.DataFrame({n: [len(g.get(gn, ())) for gn in genes]
                        for n, g in graphs.items()}, index=genes)
    deg.to_csv(f"{U.V11}/results/t2c_degree_by_network.csv")

    names = list(graphs)
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        for b in names[:i + 1]:
            if a == b:
                M.loc[a, b] = 1.0
                continue
            sub = deg[(deg[a] > 0) | (deg[b] > 0)]
            r = float(sub[a].corr(sub[b], method="spearman"))
            M.loc[a, b] = round(r, 4)
    json.dump({"networks": names,
               "matrix": {a: {b: (None if pd.isna(M.loc[a, b])
                                  else float(M.loc[a, b]))
                              for b in names if not pd.isna(M.loc[a, b])}
                          for a in names}},
              open(f"{U.V11}/results/t2c_rank_stability.json", "w"),
              indent=2)
    print(M.to_string())


if __name__ == "__main__":
    main()
