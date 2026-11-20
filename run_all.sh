#!/bin/bash
# Run the complete analysis: TabuRL-FS (3 classifiers) -> full-pipeline permutation test -> Table 1 / Figures 2-4 -> 100-run recurrence / Figure 5
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
N_JOBS=${N_JOBS:-8}
DEVICE=${DEVICE:-cuda:0}
mkdir -p logs

for clf in LinearSVC LogReg SVM_RBF; do
  $PY -u run_stroke.py select --clf "$clf" --n_jobs "$N_JOBS" --device "$DEVICE" 2>&1 | tee "logs/select_${clf}.log"
done

$PY -u run_stroke.py permutation --n_jobs "$N_JOBS" --device "$DEVICE" 2>&1 | tee logs/permutation.log

$PY -u run_stroke.py evaluate 2>&1 | tee logs/evaluate.log

$PY -u run_stroke.py recurrence --n_jobs "$N_JOBS" --device "$DEVICE" 2>&1 | tee logs/recurrence.log
