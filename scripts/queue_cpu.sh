#!/usr/bin/env bash
# CPU queue: waits until the baseline grid search has logged all configs, then fits the chosen config
# on the fixed 50k subsample and on all non-val folds.
cd "$(dirname "$0")/../src"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
until [ "$(grep -c macro_f1 ../logs/baseline_search.log)" -ge 10 ]; do sleep 30; done
CFG='{"feat": "word", "model": "lr", "C": 16, "wng": [1, 3]}'
[ -f ../preds/baseline_val.npy ] || python baseline.py final --cfg "$CFG" > ../logs/baseline_final_sub.log 2>&1
python baseline.py final --cfg "$CFG" --full > ../logs/baseline_final_full.log 2>&1
echo cpu queue done
