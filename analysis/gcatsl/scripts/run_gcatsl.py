"""
run_gcatsl.py — controlled entry point for the authors' GCATSL code
====================================================================
Imports the unmodified GCATSL source (external/GCATSL_src = github.com/
lichenbiostat/GCATSL @ master, MIT) — or the fork whose only diff is saving
the score matrix — and invokes its main() with the README hyperparameters.
Kept outside the external tree per sandbox discipline; run with `python -I`.

Usage:
  python -I run_gcatsl.py <input_dir> <output_dir> <n_node> [--fork]
                         [--n_epoch 600] [--n_fold 1]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    input_dir = sys.argv[1].replace("\\", "/")
    output_dir = sys.argv[2].replace("\\", "/")
    n_node = int(sys.argv[3])
    use_fork = "--fork" in sys.argv
    n_epoch = int(sys.argv[sys.argv.index("--n_epoch") + 1]) \
        if "--n_epoch" in sys.argv else 600
    n_fold = int(sys.argv[sys.argv.index("--n_fold") + 1]) \
        if "--n_fold" in sys.argv else 1

    for d in (input_dir, output_dir):
        if not d.endswith("/"):
            pass
    if not input_dir.endswith("/"):
        input_dir += "/"
    if not output_dir.endswith("/"):
        output_dir += "/"
    os.makedirs(output_dir, exist_ok=True)

    src = "gcatsl_fork" if use_fork else "gcatsl_src"
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", src)))
    import main as gcatsl_main                      # authors' module

    class A:
        pass
    args = A()
    args.n_epoch = n_epoch
    args.n_head = 2
    args.n_fold = n_fold
    args.n_node = n_node
    args.n_feature = 3
    args.learning_rate = 0.005
    args.weight_decay = 0.0001
    args.dropout = 0.7
    args.input_dir = input_dir
    args.output_dir = output_dir
    args.log_dir = output_dir
    print(f"[run_gcatsl] src={src} input={input_dir} n={n_node} "
          f"epochs={n_epoch} folds={n_fold}", flush=True)
    gcatsl_main.main(args)


if __name__ == "__main__":
    main()
