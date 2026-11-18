#!/bin/bash
# wave3_temporal.sh — six temporal GCATSL runs (3 cutoffs x 2 vintages)
# Requires: interaction_global_0.txt already present in each run dir (wave1).
export PATH="/c/Users/Administrator/miniconda3/envs/gcatsl:/c/Users/Administrator/miniconda3/envs/gcatsl/Library/bin:$PATH"
export OMP_NUM_THREADS=6
EXT="E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/external"

for c in 2012 2014 2016; do
  for v in hetionet2016 string2021; do
    d="$EXT/runs/${c}_${v}"
    if [ ! -f "$d/interaction_global_0.txt" ]; then
      echo "MISSING global matrix: $d"; exit 1
    fi
  done
done
echo "all global matrices present"

for c in 2012 2014 2016; do
  for v in hetionet2016 string2021; do
    d="$EXT/runs/${c}_${v}"
    mkdir -p "$d/out"
    echo "=== run $c $v ==="
    python -I -X utf8 -u "$EXT/scripts/run_gcatsl.py" "$d" "$d/out" 6400 --fork \
      > "$d/run.log" 2>&1
    tail -1 "$d/run.log"
  done
done
echo "all temporal runs done"
