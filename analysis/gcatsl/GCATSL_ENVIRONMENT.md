# GCATSL original-pipeline execution (R1 revision)

This directory executes the authors' released GCATSL implementation under the
temporal protocol of the manuscript. It contains no code copied into the
analysis path silently; everything is pinned and diffed.

## Provenance

- Source: https://github.com/lichenbiostat/GCATSL (master, MIT licence),
  the implementation accompanying Long Y, Wu M, Liu Y, Cai R, Huang D,
  Kwoh CK. Graph contextualized attention network for predicting synthetic
  lethality in human cancers. Bioinformatics. 2021;37:2432-40.
  doi:10.1093/bioinformatics/btab110. Downloaded 2026-11-13 via the GitHub
  API; the file tree is archived in `gcatsl_tree.json`.
- `GCATSL_orig/` — the downloaded source files (flat names) plus the
  released example data (`adj.txt`, `test_arr_0..4.txt`, the compressed
  `global_interaction_matrix.rar` and the authors' supplementary PDF).
  The three 20-MB released feature matrices are not vendored here; they are
  listed in `gcatsl_tree.json` and obtainable from the authors' repository.
- `gcatsl_src/` — the same source arranged as the released package layout,
  used verbatim for the released-example validation run (five-fold log in
  `toy_run/main_run.log`: auc_ave 0.931168, auc_std 0.002500).
- `gcatsl_fork/` — identical to `gcatsl_src/` except for a two-line
  score-matrix export; see `gcatsl_score_export.patch`.

## Environment (pinned)

```
conda create -n gcatsl python=3.7
# pip install into the env:
tensorflow==1.13.1  numpy==1.16.6  scipy==1.4.1  scikit-learn==0.21.3
protobuf==3.20.3  matplotlib
```

Hyperparameters are those of the repository's README example command:
`--n_epoch 600 --n_head 2 --learning_rate 0.005 --weight_decay 0.0001
--dropout 0.7`, with `--n_fold 1` and a supplied `test_arr_0.txt` so that
the fold loop degenerates to the temporal split (the code trains on every
row outside the supplied test file).

## Scripts (`scripts/`)

- `run_gcatsl.py` — entry point that imports the authors' `main` and calls
  it with the pinned arguments (`--fork` selects the score-export fork).
- `pregen_fast.py` — converged-equivalent regeneration of the global
  interaction matrices: the authors' random-walk-with-restart iteration
  converges to float64 machine precision (delta < 1e-15) within ~15 steps
  because the column-normalised transition has spectral radius <= 0.1;
  `--verify` once ran the full 1000 iterations and confirmed identical
  global-neighbour masks (see `runs/pregen_fast.log`).
- `pregen_global.py` — calls the authors' own generator verbatim (kept for
  provenance; not used in the runs because its 1000-iteration loop costs
  ~14 CPU-hours per matrix).
- `wave*.sh` — orchestration of the runs.

## Runs (`runs/`)

One directory per (cutoff, feature vintage), plus the two cross-validation
runs. What this repository carries for every run: `run.log` (the full
600-epoch training log) and `out/test_result_0.txt` (the per-test-pair
scores the exported score matrix produces). The heavy regenerated inputs
and outputs — `adj.txt` (all 17,502 dated gold pairs, 1-based indices over
the frozen 6,400-gene universe), `test_arr_0.txt`, the three 128-dimensional
feature files (seed-42 uniform-random-walk skip-gram embeddings of the
vintage's PPI and GO BP/CC co-annotation graphs), the global matrices and
`out/score_matrix_0.npy` (6,400 x 6,400 float64) — are rebuilt by
`scripts/pregen_fast.py` and `../t13_gcatsl_inputs.py` and are not stored.
`universe_genes.csv` and `test_pools_universe.csv` define the frozen
universe and the negative-pool pairs inside it; `t13_universe.json` in
`../../data/` is the manifest.

## Evaluation

`../t13_gcatsl_inputs.py` builds the inputs;
`../t13_gcatsl_eval.py` scores the exported matrices on post-cutoff
positives versus the shared pool test negatives and computes the paired
absorption contrasts; output `data/t13_gcatsl_pair.csv`.
