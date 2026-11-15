#!/usr/bin/env bash
# End-to-end check in ~2 minutes: tests, tiny training runs, evaluation, analysis.
# Writes to runs_smoke/ and results_smoke/ so real results are never touched.
#
#   bash scripts/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUT="output_dir=runs_smoke"

"$PY" -m pytest -q
# Tiny runs: the metric VALUES are meaningless here -- this only checks that everything runs.
"$PY" -m src.train_classifier --config configs/classifier.yaml --set $OUT subset=10000 epochs=1
"$PY" -m src.real_baseline --set $OUT --n-samples 1000 --feature-spaces classifier
for model in vae gan; do
  "$PY" -m src.train_$model --config configs/$model.yaml --set $OUT subset=1000 epochs=2 save_every=1
  "$PY" -m src.evaluate     --config configs/$model.yaml --set $OUT --n-samples 1000 --feature-spaces classifier
  "$PY" -m src.evaluate     --config configs/$model.yaml --set $OUT --all-epochs --quick
done
"$PY" -m src.analysis --runs runs_smoke --out results_smoke
echo "Smoke test passed. Look at results_smoke/ and runs_smoke/*/eval/."
