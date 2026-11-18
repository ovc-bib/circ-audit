"""
pregen_global.py — pre-generate interaction_global_<cv>.txt per run directory
============================================================================
Calls the authors' own generate_global_interaction_matrix (dataprocessor.py,
verbatim) for a supplied split: train_arr = all adj rows minus the rows of
test_arr_<cv>.txt. The generated matrix depends only on the gene-index set of
the training SL graph, so it is identical for both feature vintages of a
cutoff and can be copied into both run directories.

Usage (gcatsl env): python -I pregen_global.py <input_dir> <n_node> [cv=0]
The input dir must already contain adj.txt and test_arr_<cv>.txt.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "gcatsl_src")))
from dataprocessor import generate_global_interaction_matrix  # noqa: E402


class A:
    pass


def main():
    input_dir = sys.argv[1].replace("\\", "/")
    if not input_dir.endswith("/"):
        input_dir += "/"
    n_node = int(sys.argv[2])
    cv = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    args = A()
    args.input_dir = input_dir
    args.n_nodes = n_node          # authors' generator reads .n_nodes
    labels = np.loadtxt(input_dir + "adj.txt")
    test_arr = np.loadtxt(
        input_dir + "test_arr_" + str(cv) + ".txt").astype(int).tolist()
    train_arr = sorted(set(range(labels.shape[0])) - set(test_arr))
    print(f"pregen: {input_dir} n={n_node} cv={cv} "
          f"train={len(train_arr)} test={len(test_arr)}", flush=True)
    generate_global_interaction_matrix(args, train_arr, cv)
    print("pregen done:", input_dir + f"interaction_global_{cv}.txt",
          flush=True)


if __name__ == "__main__":
    main()
