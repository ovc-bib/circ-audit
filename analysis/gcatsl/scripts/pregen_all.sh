#!/bin/bash
# pregen_all.sh — converged-equivalent global matrices for all runs, then toy folds
# Sequential, 12 MKL threads each. The 2016 temporal matrix is generated with
# --verify (full 1000-iteration equivalence check).
export PATH="/c/Users/Administrator/miniconda3/envs/gcatsl:/c/Users/Administrator/miniconda3/envs/gcatsl/Library/bin:$PATH"
export MKL_NUM_THREADS=12
export OMP_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
EXT="E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/external"

echo "=== temporal pregen ==="
for c in 2012 2014; do
  python -I -X utf8 -u "$EXT/scripts/pregen_fast.py" "$EXT/runs/${c}_hetionet2016" 6400 0 \
    >> "$EXT/runs/pregen_fast.log" 2>&1 && \
  cp "$EXT/runs/${c}_hetionet2016/interaction_global_0.txt" "$EXT/runs/${c}_string2021/"
done
python -I -X utf8 -u "$EXT/scripts/pregen_fast.py" "$EXT/runs/2016_hetionet2016" 6400 0 --verify \
  >> "$EXT/runs/pregen_fast.log" 2>&1 && \
cp "$EXT/runs/2016_hetionet2016/interaction_global_0.txt" "$EXT/runs/2016_string2021/"
echo "=== temporal pregen done ==="

echo "=== toy pregen (5 folds) ==="
for cv in 0 1 2 3 4; do
  python -I -X utf8 -u "$EXT/scripts/pregen_fast.py" "$EXT/toy_run" 6375 $cv \
    >> "$EXT/toy_run/pregen_fast.log" 2>&1
done
echo "=== toy pregen done ==="

echo "=== temporal GCATSL runs ==="
for c in 2012 2014 2016; do
  for v in hetionet2016 string2021; do
    d="$EXT/runs/${c}_${v}"
    mkdir -p "$d/out"
    echo "--- run $c $v ---"
    OMP_NUM_THREADS=6 python -I -X utf8 -u "$EXT/scripts/run_gcatsl.py" "$d" "$d/out" 6400 --fork \
      > "$d/run.log" 2>&1
    tail -1 "$d/run.log"
  done
done
echo "=== temporal runs done ==="

echo "=== toy 5-fold original-code validation run ==="
OMP_NUM_THREADS=6 python -I -X utf8 -u "$EXT/scripts/run_gcatsl.py" "$EXT/toy_run" "$EXT/toy_run/out" 6375 --n_fold 5 \
  > "$EXT/toy_run/main_run.log" 2>&1
tail -3 "$EXT/toy_run/main_run.log"
echo "=== toy run done ==="
echo "ALL DONE"
