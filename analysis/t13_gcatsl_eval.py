"""
t13_gcatsl_eval.py — T13: evaluate saved GCATSL score matrices (R1 revision)
=============================================================================
Reads the score matrices saved by the authors' (forked) GCATSL pipeline for
each (cutoff, vintage) run and evaluates them on the temporal pair protocol
of T10: post-cutoff positive pairs versus the shared random-pool test
negatives that fall inside the frozen universe. Absorption = AUROC(current
vintage) - AUROC(2016 vintage) on identical rows, paired bootstrap.

Output: results/t13_gcatsl_pair.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tm_utils as U
from t10_pair_level import export_pairs, eval_pairs, paired_pair_test

V11 = U.V11
RUNS = f"{V11}/external/runs"
CUTOFFS = U.CUTOFFS


def main():
    t0 = time.time()
    uni = pd.read_csv(f"{RUNS}/universe_genes.csv")
    idx = dict(zip(uni["gene"], uni["index"]))            # 1-based
    universe_set = set(uni["gene"])

    scores = U.load_scores()
    posE = export_pairs(universe_set)
    pos_list = list(zip(posE["a"], posE["b"]))
    pos_year = dict(zip(pos_list, posE["year"]))
    print(f"positives inside universe: {len(pos_list)}")

    pools = pd.read_csv(f"{RUNS}/test_pools_universe.csv")
    rnd = pools[pools["pool"] == "random"]
    neg_te = list(zip(rnd[rnd["split"] == "test"]["a"],
                      rnd[rnd["split"] == "test"]["b"]))
    neg_tr = list(zip(rnd[rnd["split"] == "train"]["a"],
                      rnd[rnd["split"] == "train"]["b"]))
    print(f"test negs {len(neg_te)}, train negs {len(neg_tr)}")

    rows = []
    for c in CUTOFFS:
        te_pos = [k for k in pos_list if pos_year[k] > c]
        yte = np.array([1] * len(te_pos) + [0] * len(neg_te))
        outs = {}
        for vname in ("hetionet2016", "string2021"):
            d = f"{RUNS}/{c}_{vname}/out"
            S = np.load(f"{d}/score_matrix_0.npy")
            def sc(pairs):
                return np.array([S[idx[min(a, b)] - 1, idx[max(a, b)] - 1]
                                 for a, b in pairs])
            s = np.concatenate([sc(te_pos), sc(neg_te)])
            ev = eval_pairs(yte, s, seed=U.SEED + c)
            outs[vname] = s
            rows.append({"model": f"GCATSL original ({vname})", "cutoff": c,
                         **ev, "n_test_pos": len(te_pos)})
            print(f"  GCATSL {vname} @{c}: {ev['auroc']} "
                  f"[{ev['ci_lo']},{ev['ci_hi']}] n+={len(te_pos)}")
        d, lo, hi, p = paired_pair_test(yte, outs["hetionet2016"],
                                        outs["string2021"],
                                        seed=U.SEED + 713 + c)
        rows.append({"model": "GCATSL absorption (2021 minus 2016 vintage)",
                     "cutoff": c, "auroc": round(d, 4), "ci_lo": round(lo, 4),
                     "ci_hi": round(hi, 4), "p": round(p, 4),
                     "n_test_pos": len(te_pos)})
        print(f"  GCATSL absorption @{c}: {d:+.4f} (p={p:.4f})")

    out = pd.DataFrame(rows)
    out.to_csv(f"{V11}/results/t13_gcatsl_pair.csv", index=False)
    print(f"T13 eval done in {(time.time() - t0) / 60:.1f} min")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
