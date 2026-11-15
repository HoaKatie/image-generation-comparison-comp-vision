#!/usr/bin/env bash
# Main experiment (H1-H3): VAE vs DCGAN with the shared default settings, several seeds.
#
#   bash scripts/run_main.sh                # seeds 0 1 2
#   SEEDS="1" bash scripts/run_main.sh      # one seed -- e.g. split seeds across teammates,
#                                           # then copy everyone's runs/ folders together
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
SEEDS=${SEEDS:-"0 1 2"}
DATASET=$("$PY" -c "from src.config import load_config; print(load_config('configs/shared.yaml')['dataset'])")

# The evaluation classifier is trained once and shared by every evaluation. Teammates who split
# the seeds must all copy the SAME runs/classifier_<dataset>/ folder before running this script,
# otherwise their classifier-space metrics are not comparable.
if [ ! -f "runs/classifier_${DATASET}/classifier.pt" ]; then
  "$PY" -m src.train_classifier --config configs/classifier.yaml
  rm -rf "runs/real_baseline_${DATASET}"  # it was computed with the previous classifier
fi
# Real-vs-real reference row for the main table (the best score a perfect generator could get).
if [ ! -f "runs/real_baseline_${DATASET}/metrics.json" ]; then
  "$PY" -m src.real_baseline
fi

for seed in $SEEDS; do
  for model in vae gan; do
    "$PY" -m src.train_$model --config configs/$model.yaml --set seed=$seed
    "$PY" -m src.evaluate     --config configs/$model.yaml --set seed=$seed                       # metrics + figures
    "$PY" -m src.evaluate     --config configs/$model.yaml --set seed=$seed --all-epochs --quick  # learning curve
  done
done

"$PY" -m src.analysis
