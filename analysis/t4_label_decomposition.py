"""
t4_label_decomposition.py — T4: label-side leakage decomposition, 3 cutoffs
==========================================================================
For each cutoff and each GBM-family model:
  KB-restricted twin : train on <=cutoff gold + shared train negatives
  all-gold twin      : identical except >cutoff gold ALSO enters training
Both evaluated on the same test rows (>cutoff gold + held-out negatives).
The difference = what cross validation-style label contamination alone adds
to apparent accuracy on the new era.

Extends e4b (cutoff-2016 only) to all three cutoffs with the shared v11
negative split, so the T1 numbers and the KB-restricted twins coincide.

Output: results/t4_label_decomposition.csv, results/t4_summary.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tm_utils as U


def main():
    scores = U.load_scores()
    y = scores["y_gold"].to_numpy()
    train_neg, test_neg = U.shared_negative_split(y == 0)

    mats = U.build_matrices(scores)
    rows = []
    for cutoff in U.CUTOFFS:
        old_pos, new_pos, _ = U.era_masks(scores, cutoff)
        tr_kb = np.concatenate([np.where(old_pos)[0], train_neg])
        tr_all = np.concatenate([np.where(y == 1)[0], train_neg])
        te = np.concatenate([np.where(new_pos)[0], test_neg])
        y_te = y[te]
        for name, X in mats.items():
            m1 = XGBClassifier(**U.GBM_PARAMS).fit(X[tr_kb], y[tr_kb])
            auc_kb = roc_auc_score(y_te, m1.predict_proba(X[te])[:, 1])
            m2 = XGBClassifier(**U.GBM_PARAMS).fit(X[tr_all], y[tr_all])
            auc_all = roc_auc_score(y_te, m2.predict_proba(X[te])[:, 1])
            rows.append({
                "model": name, "cutoff": cutoff,
                "n_train_pos_kb": int(old_pos.sum()),
                "n_test_pos": int(new_pos.sum()),
                "auroc_kb_restricted": round(float(auc_kb), 4),
                "auroc_allgold_trained": round(float(auc_all), 4),
                "label_leakage_gain": round(float(auc_all - auc_kb), 4),
            })
            print(f"  {name:<12} @{cutoff} KB={auc_kb:.4f} all-gold={auc_all:.4f} "
                  f"gain={auc_all-auc_kb:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{U.V11}/results/t4_label_decomposition.csv", index=False)
    with open(f"{U.V11}/results/t4_summary.json", "w", encoding="utf-8") as f:
        json.dump({"design": ("all-gold twin minus KB-restricted twin on the "
                              "same >cutoff test rows; shared negatives"),
                   "results": rows}, f, indent=2)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
