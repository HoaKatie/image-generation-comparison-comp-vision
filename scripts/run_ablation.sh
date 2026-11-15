#!/usr/bin/env bash
# Ablations (run AFTER scripts/run_main.sh, which provides latent_dim=32 / beta=1 and the classifier):
#   1. latent dimension 8 / 128 for both models
#   2. beta 0.5 / 4 for the VAE (beta-VAE trade-off: sharpness vs. latent smoothness)
#
#   bash scripts/run_ablation.sh              # seed 0 only (cheaper)
#   SEEDS="0 1 2" bash scripts/run_ablation.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
SEEDS=${SEEDS:-"0"}

for seed in $SEEDS; do
  for z in 8 128; do
    for model in vae gan; do
      "$PY" -m src.train_$model --config configs/$model.yaml --set latent_dim=$z seed=$seed
      "$PY" -m src.evaluate     --config configs/$model.yaml --set latent_dim=$z seed=$seed
    done
  done
  for beta in 0.5 4; do
    "$PY" -m src.train_vae --config configs/vae.yaml --set beta=$beta seed=$seed
    "$PY" -m src.evaluate  --config configs/vae.yaml --set beta=$beta seed=$seed
  done
done

"$PY" -m src.analysis
