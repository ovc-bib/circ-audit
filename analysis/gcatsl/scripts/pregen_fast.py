"""
pregen_fast.py — converged-equivalent interaction_global generator (R1 revision)
================================================================================
The authors' random_walk_with_restart (dataprocessor.py) iterates
    t <- 0.1 * (A / colsum) @ t + 0.9 * I
one thousand times. Because A / colsum is column-stochastic, the map has
spectral radius <= 0.1 and the iteration reaches float64 machine precision
within ~160 steps; the remaining ~840 steps cannot change a single float64
entry. This script applies the authors' exact formulas (same column
normalisation including the sum_col[sum_col == 0] = 2 quirk, same p = 0.9,
same extraction through argsort and per-column degree loops) but stops when
the infinity-norm change falls below 1e-15, then writes interaction_global
files byte-compatible with the authors' np.savetxt output.

A verification mode (--verify) runs the full 1000 iterations once and
asserts the extracted global-neighbour masks are identical.

Usage: python -I pregen_fast.py <input_dir> <n_node> [cv=0] [--verify]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def build_M(input_dir, cv):
    labels = np.loadtxt(input_dir + "adj.txt")
    test_arr = np.loadtxt(
        input_dir + "test_arr_" + str(cv) + ".txt").astype(int).tolist()
    train_arr = sorted(set(range(labels.shape[0])) - set(test_arr))
    n_node = None  # set by caller
    return labels, train_arr, test_arr


def rwr_converged(M, tol=1e-15, max_iter=1000, full=False):
    """Authors' iteration with convergence stopping (or full 1000)."""
    p = 0.9
    n = M.shape[0]
    sum_col = M.sum(axis=0)
    sum_col[sum_col == 0.] = 2
    A = np.divide(M, sum_col)
    t = np.identity(n)
    it = 0
    while it < (max_iter if full else max_iter):
        t_new = (1 - p) * (A @ t) + p * np.identity(n)
        delta = np.abs(t_new - t).max()
        t = t_new
        it += 1
        if not full and delta < tol:
            break
    return t, it, delta


def extract_global_neighbors(interaction, walk_matrix):
    """Verbatim logic of the authors' dataprocessor.extract_global_neighbors."""
    interaction = interaction.astype(int)
    interaction_mask = np.zeros_like(interaction)
    neigh_index = np.argsort(-walk_matrix, axis=0)
    for j in range(interaction.shape[1]):
        for i in range(np.sum(interaction[j, :])):
            interaction_mask[neigh_index[i, j], j] = 1
    return interaction_mask.T


def main():
    input_dir = sys.argv[1].replace("\\", "/")
    if not input_dir.endswith("/"):
        input_dir += "/"
    n_node = int(sys.argv[2])
    cv = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "--verify" \
        else 0
    verify = "--verify" in sys.argv

    labels = np.loadtxt(input_dir + "adj.txt")
    test_arr = np.loadtxt(
        input_dir + "test_arr_" + str(cv) + ".txt").astype(int).tolist()
    train_arr = sorted(set(range(labels.shape[0])) - set(test_arr))

    M = np.zeros((n_node, n_node))
    M[labels[train_arr, 0].astype(int) - 1,
      labels[train_arr, 1].astype(int) - 1] = labels[train_arr, 2]
    M = M + M.T
    print(f"pregen_fast: {input_dir} cv={cv} n={n_node} "
          f"train={len(train_arr)} test={len(test_arr)} edges={int(M.sum())}",
          flush=True)

    t0 = time.time()
    walk, it, delta = rwr_converged(M, full=False)
    print(f"  converged at iteration {it} (delta={delta:.2e}, "
          f"{time.time() - t0:.0f}s)", flush=True)
    mask = extract_global_neighbors(M, walk)
    out = input_dir + f"interaction_global_{cv}.txt"
    np.savetxt(out, mask, fmt="%s")

    if verify:
        t1 = time.time()
        walk_full, it_full, _ = rwr_converged(M, full=True)
        mask_full = extract_global_neighbors(M, walk_full)
        same = bool((mask == mask_full).all())
        maxdiff = float(np.abs(walk - walk_full).max())
        print(f"  VERIFY full 1000 iters: masks identical={same}, "
              f"max|walk diff|={maxdiff:.3e} ({time.time() - t1:.0f}s)",
              flush=True)
        if not same:
            sys.exit("verification FAILED: masks differ")
    print("pregen_fast done:", out, flush=True)


if __name__ == "__main__":
    main()
