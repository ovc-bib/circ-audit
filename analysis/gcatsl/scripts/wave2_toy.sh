#!/bin/bash
# wave2_toy.sh — toy-example validation: 5 fold global matrices + 5-fold CV run
# of the UNMODIFIED original source (gcatsl_src) on the authors' released data.
export PATH="/c/Users/Administrator/miniconda3/envs/gcatsl:/c/Users/Administrator/miniconda3/envs/gcatsl/Library/bin:$PATH"
export OMP_NUM_THREADS=4
EXT="E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/external"
mkdir -p "$EXT/toy_run/out"

# pre-generate the five fold matrices in parallel (2 waves to respect cores)
for cv in 0 1 2; do
  python -I -X utf8 -u "$EXT/scripts/pregen_global.py" "$EXT/toy_run" 6375 $cv \
    > "$EXT/toy_run/pregen_$cv.log" 2>&1 &
done
wait
for cv in 3 4; do
  python -I -X utf8 -u "$EXT/scripts/pregen_global.py" "$EXT/toy_run" 6375 $cv \
    > "$EXT/toy_run/pregen_$cv.log" 2>&1 &
done
wait
echo "toy pregen done"

# run the authors' unmodified main over the five released folds
python -I -X utf8 -u "$EXT/scripts/run_gcatsl.py" "$EXT/toy_run" "$EXT/toy_run/out" 6375 --n_fold 5 \
  > "$EXT/toy_run/main_run.log" 2>&1
echo "toy 5-fold done"
tail -3 "$EXT/toy_run/main_run.log"
