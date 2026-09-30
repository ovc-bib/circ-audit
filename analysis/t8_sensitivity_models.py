"""
t8_sensitivity_models.py — multi-source dating sensitivity, stage 2
===================================================================
Reruns the headline rows of the prospective leaderboard (Table 1) under the
merged two-source dating (min of SynLethDB and BioGRID-SGI first-evidence
years, from t8_biogrid_dating.py).  GBM family retrains per cutoff,
label-free scores are re-evaluated on the shifted test rows, and the
SINaTRA-style reimplementation (full inputs) retrains as the headline.
Writes results/t8_dating_sensitivity.csv with original vs merged AUROCs.
"""
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/src")
import tm_utils as U
import tm_methods as M

V11 = U.V11
SEED = U.SEED


def main():
    # ---- merged dating patch ----
    dating = pd.read_csv(f"{V11}/results/t8_biogrid_dating.csv")
    merged = {r.gene: (np.nan if pd.isna(r.merged_year) else float(r.merged_year))
              for r in dating.itertuples()}
    U.first_evidence_year = lambda: merged          # era_masks picks this up

    scores = U.load_scores()
    genes = scores["gene"].tolist()
    y = scores["y_gold"].to_numpy()
    all_gold_set = set(np.array(genes)[y == 1])
    train_neg, test_neg = U.shared_negative_split(y == 0)

    orig = pd.read_csv(f"{V11}/results/t1_prospective_leaderboard.csv")
    orig_map = {(r.method, r.cutoff): r.auroc for r in orig.itertuples()}
    rows = []

    def add(method, cutoff, old_pos, new_pos, ev):
        key = (method, cutoff)
        o = orig_map.get(key)
        rows.append({"method": method, "cutoff": cutoff,
                     "n_train_pos": int(old_pos.sum()) if old_pos is not None else None,
                     "n_test_pos": int(new_pos.sum()),
                     "auroc_merged": ev["auroc"], "ci_lo": ev["ci_lo"],
                     "ci_hi": ev["ci_hi"],
                     "auroc_original": o,
                     "delta": None if o is None or pd.isna(o)
                              else round(ev["auroc"] - o, 4)})
        print(f"  {method:<14} @{cutoff} merged={ev['auroc']:.4f} "
              f"[{ev['ci_lo']:.3f},{ev['ci_hi']:.3f}]  orig={o}  "
              f"delta={rows[-1]['delta']}")

    # ---- new split sizes ----
    print("split sizes under merged dating (orig 415/273, 591/97, 630/58):")
    for c in U.CUTOFFS:
        op, np_, _ = U.era_masks(scores, c)
        print(f"  @{c}: train_pos={int(op.sum())} test_pos={int(np_.sum())}")

    # ---- GBM family ----
    print("\n[1/3] GBM family under merged dating...")
    mats = U.build_matrices(scores)
    from xgboost import XGBClassifier
    for c in U.CUTOFFS:
        op, np_, _ = U.era_masks(scores, c)
        tr = np.concatenate([np.where(op)[0], train_neg])
        te = np.concatenate([np.where(np_)[0], test_neg])
        for name, X in mats.items():
            m = XGBClassifier(**U.GBM_PARAMS).fit(X[tr], y[tr])
            ev = U.eval_block(y[te], m.predict_proba(X[te])[:, 1], seed=SEED + c)
            add(name, c, op, np_, ev)

    # ---- label-free ----
    print("\n[2/3] label-free rows on shifted test sets...")
    for c in U.CUTOFFS:
        _, np_, _ = U.era_masks(scores, c)
        te = np.concatenate([np.where(np_)[0], test_neg])
        for name in ("PPI_Proximity", "SLant"):
            if name in scores.columns:
                ev = U.eval_block(y[te], scores[name].to_numpy()[te], seed=SEED + c)
                add(name, c, None, np_, ev)

    # ---- SINaTRA headline ----
    print("\n[3/3] SINaTRA-style (full inputs) under merged dating...")
    sys.path.insert(0, f"{U.ROOT}/crisprsl")
    sys.path.insert(0, U.ROOT)
    from data_loader import CrisprSLDataLoader
    data = CrisprSLDataLoader().load_all()
    saarf_df = data["saarf_df"]
    import config as cfg
    context_genes = [g for g in cfg.HR_SEED_GENES
                     if g not in data.get("gold_genes_d4_strict", set())]
    for c in U.CUTOFFS:
        op, np_, _ = U.era_masks(scores, c)
        train_gold = set(np.array(genes)[op])
        t = time.time()
        s = M.sinatra_retrain(genes, saarf_df, context_genes,
                              train_gold_set=train_gold,
                              neg_exclude=all_gold_set)
        te = np.concatenate([np.where(np_)[0], test_neg])
        ev = U.eval_block(y[te], s[te], seed=SEED + c)
        add("SINaTRA", c, op, np_, ev)
        print(f"    ({time.time()-t:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(f"{V11}/results/t8_dating_sensitivity.csv", index=False)
    d = df["delta"].dropna().abs()
    print(f"\nmax |delta| = {d.max():.4f}  mean |delta| = {d.mean():.4f}  "
          f"n rows = {len(d)}")


if __name__ == "__main__":
    main()
