#!/bin/bash
# wave1_pregen.sh — pre-generate the three temporal interaction_global matrices
# Each calls the authors' generate_global_interaction_matrix verbatim.
export PATH="/c/Users/Administrator/miniconda3/envs/gcatsl:/c/Users/Administrator/miniconda3/envs/gcatsl/Library/bin:$PATH"
export OMP_NUM_THREADS=4
EXT="E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/external"
for c in 2012 2014 2016; do
  python -I -X utf8 "$EXT/scripts/pregen_global.py" "$EXT/runs/${c}_hetionet2016" 6400 \
    > "$EXT/runs/pregen_${c}.log" 2>&1 &
done
wait
echo "wave1 pregen all done"
for c in 2012 2014 2016; do
  cp "$EXT/runs/${c}_hetionet2016/interaction_global_0.txt" "$EXT/runs/${c}_string2021/"
done
echo "copied to string2021 dirs"
