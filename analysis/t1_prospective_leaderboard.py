"""
t1_prospective_leaderboard.py — T1: the prospective leaderboard (paper Table 1)
================================================================================
For every cutoff in {2012, 2014, 2016} and every method:
  train with <=cutoff gold only (shared negative split, seed 42),
  test on >cutoff gold + held-out negatives, AUROC with gene-level bootstrap CI.

Label-free methods (PPI_Proximity, SLant) are evaluated directly — they never
see gold labels, so their fixed v10 scores are prospective by construction.
DebiasedSL is excluded (precomputed CV-trained scores cannot be made
prospective without its full retraining pipeline; DESIGN.md §2).

Validation: at cutoff 2016 the GBM numbers must reproduce e4b
(PPI 0.9015 / CRISPR 0.7285 / Fusion 0.9117).

Output: results/t1_prospective_leaderboard.csv, results/t1_summary.json
"""
import json
import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tm_utils as U
import tm_methods as M

ROOT = U.ROOT
V10 = U.V10
V11 = U.V11
SEED = U.SEED


def main():
    os.makedirs(f"{V11}/results", exist_ok=True)
    t0 = time.time()

    scores = U.load_scores()
    genes = scores["gene"].tolist()
    y = scores["y_gold"].to_numpy()
    all_gold_set = set(np.array(genes)[y == 1])
    train_neg, test_neg = U.shared_negative_split(scores["y_gold"].to_numpy() == 0)
    print(f"candidates={len(genes)} gold={int(y.sum())} "
          f"train_neg={len(train_neg)} test_neg={len(test_neg)}")

    retro = U.retro_reference()
    rows = []

    # ---------------- GBM family: retrain per cutoff ----------------
    print("\n[1/5] building feature matrices (v7 replication)...")
    mats = U.build_matrices(scores)

    from xgboost import XGBClassifier
    for cutoff in U.CUTOFFS:
        old_pos, new_pos, _ = U.era_masks(scores, cutoff)
        tr = np.concatenate([np.where(old_pos)[0], train_neg])
        te = np.concatenate([np.where(new_pos)[0], test_neg])
        y_te = y[te]
        for name, X in mats.items():
            m = XGBClassifier(**U.GBM_PARAMS).fit(X[tr], y[tr])
            s = m.predict_proba(X[te])[:, 1]
            ev = U.eval_block(y_te, s, seed=SEED + cutoff)
            rows.append({"method": name, "cutoff": cutoff,
                         "n_train_pos": int(old_pos.sum()),
                         "n_test_pos": int(new_pos.sum()), **ev,
                         "retro_cv": retro.get(name)})
            print(f"  {name:<12} @{cutoff} AUROC={ev['auroc']:.4f} "
                  f"[{ev['ci_lo']:.3f},{ev['ci_hi']:.3f}]")

    # ---------------- label-free: evaluate fixed scores ----------------
    print("\n[2/5] label-free methods (PPI_Proximity, SLant)...")
    for cutoff in U.CUTOFFS:
        _, new_pos, _ = U.era_masks(scores, cutoff)
        te = np.concatenate([np.where(new_pos)[0], test_neg])
        y_te = y[te]
        for name in ("PPI_Proximity", "SLant"):
            if name not in scores.columns:
                continue
            ev = U.eval_block(y_te, scores[name].to_numpy()[te], seed=SEED + cutoff)
            rows.append({"method": name, "cutoff": cutoff,
                         "n_train_pos": None,
                         "n_test_pos": int(new_pos.sum()), **ev,
                         "retro_cv": retro.get(name)})
            print(f"  {name:<12} @{cutoff} AUROC={ev['auroc']:.4f} "
                  f"[{ev['ci_lo']:.3f},{ev['ci_hi']:.3f}]")

    # ---------------- loader data for third-party methods ----------------
    print("\n[3/5] loading crisprsl data loader (saarf/merged)...")
    sys.path.insert(0, f"{ROOT}/crisprsl")
    sys.path.insert(0, ROOT)
    from data_loader import CrisprSLDataLoader
    loader = CrisprSLDataLoader()
    data = loader.load_all()
    saarf_df = data["saarf_df"]
    merged = pd.read_csv(f"{ROOT}/crisprsl/results/v2_merged_features.csv")
    import config as cfg
    context_genes = [g for g in cfg.HR_SEED_GENES
                     if g not in data.get("gold_genes_d4_strict", set())]

    # ---------------- SINaTRA: full retrain per cutoff ----------------
    # Two variants, both reported:
    #   SINaTRA( InputsAsShipped): full saarf input table, which includes
    #     sl_pair_count / is_known_sl_partner — SL-database counts that
    #     restate the label and carry post-discovery information.
    #   SINaTRA(-SLDB): same reimplementation with those two columns dropped
    #     (topology/omics inputs only) — the fair-input variant.
    print("\n[4/5] SINaTRA retraining per cutoff...")
    for variant, kwargs in (("SINaTRA", {}),
                            ("SINaTRA_noSLDB", {"exclude_cols": M.SLDB_COLUMNS}),
                            ("SINaTRA_topo", {"exclude_cols": M.KNOWLEDGE_COLUMNS})):
        for cutoff in U.CUTOFFS:
            old_pos, _, _ = U.era_masks(scores, cutoff)
            train_gold = set(np.array(genes)[old_pos])
            t = time.time()
            s = M.sinatra_retrain(genes, saarf_df, context_genes,
                                  train_gold_set=train_gold,
                                  neg_exclude=all_gold_set, **kwargs)
            _, new_pos, _ = U.era_masks(scores, cutoff)
            te = np.concatenate([np.where(new_pos)[0], test_neg])
            ev = U.eval_block(y[te], s[te], seed=SEED + cutoff)
            rows.append({"method": variant, "cutoff": cutoff,
                         "n_train_pos": int(old_pos.sum()),
                         "n_test_pos": int(new_pos.sum()), **ev,
                         "retro_cv": retro.get("SINaTRA")})
            print(f"  {variant:<14} @{cutoff} AUROC={ev['auroc']:.4f} "
                  f"[{ev['ci_lo']:.3f},{ev['ci_hi']:.3f}] ({time.time()-t:.0f}s)")

    # ---------------- SL2MF + KG4SL: label-free fit, per-cutoff centroid ----
    print("\n[5/5] SL2MF (NMF once) + KG4SL (embedding once)...")
    W = M.sl2mf_fit(genes, merged, saarf_df)
    het_edges = pd.read_csv(f"{ROOT}/data/hetionet_edges.tsv", sep="\t")
    het_nodes = pd.read_csv(f"{ROOT}/data/hetionet_nodes.tsv", sep="\t")
    emb, _ = M.kg4sl_embed(genes, het_edges, het_nodes,
                           cache_path=f"{V11}/results/kg4sl_embedding.npz")

    gene_row = {g: i for i, g in enumerate(genes)}
    for cutoff in U.CUTOFFS:
        old_pos, new_pos, _ = U.era_masks(scores, cutoff)
        gold_rows = np.where(old_pos)[0]
        te = np.concatenate([np.where(new_pos)[0], test_neg])

        s_sl2mf = M.sl2mf_score(W, gold_rows)
        ev = U.eval_block(y[te], s_sl2mf[te], seed=SEED + cutoff)
        rows.append({"method": "SL2MF", "cutoff": cutoff,
                     "n_train_pos": int(old_pos.sum()),
                     "n_test_pos": int(new_pos.sum()), **ev,
                     "retro_cv": retro.get("SL2MF")})
        print(f"  SL2MF        @{cutoff} AUROC={ev['auroc']:.4f} "
              f"[{ev['ci_lo']:.3f},{ev['ci_hi']:.3f}]")

        train_gold = set(np.array(genes)[old_pos])
        s_kg = M.kg4sl_score(emb, genes, train_gold)
        ev = U.eval_block(y[te], s_kg[te], seed=SEED + cutoff)
        rows.append({"method": "KG4SL", "cutoff": cutoff,
                     "n_train_pos": int(old_pos.sum()),
                     "n_test_pos": int(new_pos.sum()), **ev,
                     "retro_cv": retro.get("KG4SL")})
        print(f"  KG4SL        @{cutoff} AUROC={ev['auroc']:.4f} "
              f"[{ev['ci_lo']:.3f},{ev['ci_hi']:.3f}]")

    # ---------------- save ----------------
    df = pd.DataFrame(rows).sort_values(["cutoff", "method"])
    df.to_csv(f"{V11}/results/t1_prospective_leaderboard.csv", index=False)
    pivot = df.pivot_table(index="method", columns="cutoff", values="auroc")

    summary = {
        "design": ("train on <=cutoff gold + shared train negatives; "
                   "test on >cutoff gold + shared held-out negatives; "
                   "gene-level bootstrap CI (2000x)"),
        "cutoffs": U.CUTOFFS,
        "validation_check": {
            "e4b_reference": {"PPI_GBM": 0.9015, "CRISPR_Only": 0.7285,
                              "Fusion": 0.9117},
            "note": "cutoff-2016 GBM rows should match e4b to ~1e-3",
        },
        "leaderboard_auroc": {m: {int(c): float(v) for c, v in r.items()
                                  if pd.notna(v)}
                              for m, r in pivot.iterrows()},
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(f"{V11}/results/t1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== T1 leaderboard (AUROC, prospective) ===")
    print(pivot.round(4).to_string())
    print(f"\nruntime {time.time()-t0:.0f}s -> {V11}/results/")


if __name__ == "__main__":
    main()
