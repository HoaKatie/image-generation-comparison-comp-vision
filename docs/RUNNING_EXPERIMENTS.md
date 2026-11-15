# Running the experiments: step by step

Follow the steps in order. Times were measured on an Apple-silicon laptop (MPS). A recent
NVIDIA GPU is similar or faster. A CPU-only machine is several times slower, so use Colab
(section 7) if you do not have a GPU.

## 0. Setup (once)

```bash
git clone https://github.com/HoaKatie/image-generation-comparison-comp-vision.git
cd image-generation-comparison-comp-vision
git checkout feature/initial-scaffold        # until this branch is merged
python3 -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Check that everything works (~1 min)

```bash
python -m pytest                 # 49 tests, ~20 s, no download needed
bash scripts/smoke_test.sh       # tests + tiny real runs, ~1 min, downloads Fashion-MNIST (~30 MB)
```

Both must pass before you start the long runs. What the tests check:

| File | Checks |
|---|---|
| `tests/test_models.py` | output shapes; the VAE decoder and the GAN generator are identical (fairness); the reparameterization has the right mean/std |
| `tests/test_losses.py` | closed-form KL, β scales only the KL, GAN losses reward a good discriminator |
| `tests/test_metrics.py` | every metric on inputs with a known answer (identical sets → FID 0, one-cluster generator → recall ≈ 0.5, blur → lower sharpness, …) |
| `tests/test_config.py` | config inheritance, typo protection for `--set`, and **`vae.yaml` / `gan.yaml` never override a shared setting** |
| `tests/test_data.py` | zero-padding (no resize) to 32×32, values in [−1, 1]; the fast evaluation loader gives the same images as training |
| `tests/test_analysis.py` | the report tables use the last full evaluation, mean ± std over seeds, the right runs |
| `tests/test_pipeline.py` | **end to end** on a toy dataset: train classifier → VAE → GAN → evaluate → real baseline → analysis; every output the report needs exists and is in range; the VAE can overfit a small batch |

If you change any code, run `python -m pytest` again before you commit.

## 1b. Explore the finished runs (no GPU needed)

The outputs of the final experiments are in the repo, so you can study the pipeline without
running anything:

| Path | What to look at |
|---|---|
| `runs/<run>/config.yaml` | the exact settings of that run |
| `runs/<run>/train_log.csv` | loss per epoch (GAN: also D(x) and D(G(z))) |
| `runs/<run>/samples/epoch_XXX.png` | the same fixed latent vectors decoded after every epoch: watch the model learn |
| `runs/<run>/eval/epoch_030/` | `metrics.json` and the evaluation figures |
| `runs/<run>/eval/epoch_XXX_quick/` | learning-curve metrics (5,000 samples) |
| `runs/<run>/checkpoints/epoch_030.pt` | the final model (intermediate checkpoints are not in git) |
| `runs/classifier_fashion_mnist/` | the evaluation classifier behind every classifier-space metric |
| `runs/real_baseline_fashion_mnist/` | real-vs-real reference scores |
| `runs/epoch_check/` | the 60-epoch VAE check (section 3) |

To re-run the evaluation of a committed model (downloads the Inception weights once):
`python -m src.evaluate --config configs/vae.yaml --set seed=0`.

> **Careful.** Re-running a *training* script deletes and recreates that run's folder,
> including its committed files. `git restore runs/` brings them back.

## 2. Main experiment: VAE vs. DCGAN, 3 seeds (~1 hour)

```bash
caffeinate -i bash scripts/run_main.sh       # macOS: caffeinate keeps the laptop awake
bash scripts/run_main.sh                     # Linux / Windows (Git Bash)
```

The script runs these steps:

1. Train the evaluation classifier (about 1 min; skipped if it already exists).
2. Compute the real-vs-real baseline (about 1 min).
3. For each seed (0, 1, 2) and each model, train for 30 epochs (VAE about 5 min, GAN about
   8 min), run the full evaluation (about 2 min), and run the learning-curve evaluations
   (about 30 s).
4. Write `results/`.

The first full evaluation downloads the Inception weights (~100 MB).

### Splitting the seeds across teammates

1. The classifier and the baseline are already in the repo (`runs/classifier_fashion_mnist/`,
   `runs/real_baseline_fashion_mnist/`), so `run_main.sh` skips both steps.
2. Everyone runs their seed: `SEEDS="1" bash scripts/run_main.sh`.
3. Commit and push your `runs/<run>/` folders, pull the others', and run
   `python -m src.analysis`.

> **Important.** Everyone must evaluate with the *same* classifier, so never delete or
> retrain `runs/classifier_fashion_mnist/`. A classifier trained separately gives slightly
> different features, and the classifier-space numbers (FID-classifier, precision/recall,
> mode coverage) would no longer be comparable between seeds.

## 3. Look at seed 0 before running more (team decision)

After the first seed finishes:

* Look at `runs/*_seed0/samples/epoch_030.png`. Are the images recognisable clothes? Does
  the GAN repeat the same few items (mode collapse)?
* Check the GAN's `train_log.csv`. If `d_fake` drops to about 0 and stays there, D has won
  and G stopped learning. Report it if it happens.
* Run `python -m src.analysis` and open `results/learning_curve.png`. If FID is still falling
  clearly at epoch 30, increase the epochs **for both models** (`epochs:` in
  `configs/shared.yaml`) and rerun. This is decision 2 in `docs/PROJECT_PLAN.md` §6.

**Decision 2 (done): the 60-epoch check.** The VAE was still improving slowly at epoch 30, so
we trained VAE seed 0 for 60 epochs. The run goes to its own folder because the run name does
not contain the number of epochs: in `runs/` it would overwrite the main seed-0 run.
`src.analysis` does not look inside `runs/epoch_check/`, so `results/` is not affected.

```bash
mkdir -p runs/epoch_check && cp -R runs/classifier_fashion_mnist runs/cache runs/epoch_check/
SET="output_dir=runs/epoch_check epochs=60 seed=0"
python -m src.train_vae --config configs/vae.yaml --set $SET
python -m src.evaluate  --config configs/vae.yaml --set $SET --epoch 30   # same as the main run
python -m src.evaluate  --config configs/vae.yaml --set $SET              # epoch 60
```

Result: FID-clf went from 21.1 to 18.4, against 1.3 for the DCGAN. We kept 30 epochs
(report §4.2).

## 4. Ablations (~45 min)

```bash
caffeinate -i bash scripts/run_ablation.sh   # latent_dim 8 / 128 (both models), beta 0.5 / 4 (VAE)
SEEDS="0 1 2" bash scripts/run_ablation.sh   # optional: 3 seeds (~2.5 h)
```

## 5. Results → report

```bash
python -m src.analysis       # rebuilds every table and figure in results/
bash report/build.sh         # report/report.pdf and report/report.docx
```

1. Copy the numbers from `results/main_results.md` and `results/ablation_*.md` into the
   **[FILL IN]** places in `report/report.md` §4–5.
2. Commit `results/` and `runs/`. The `.gitignore` keeps out the Inception cache and the
   intermediate checkpoints, so each run adds about 8 MB (mostly its final model).

| Output | Use in the report |
|---|---|
| `main_results.md` | Table 3 (main results, with the real-data row) |
| `precision_recall.png`, `mode_coverage.png`, `sharpness.png` | H1 / H2 figures |
| `learning_curve.png`, `training_curves.png` | training stability (H2) |
| `qualitative_samples.png`, `qualitative_success_failure.png` | samples and failure cases |
| `qualitative_interpolation.png` | H3 |
| `qualitative_nearest_neighbors.png`, `qualitative_reconstructions.png` | memorisation / VAE reconstructions |
| `ablation_latent_dim.*`, `ablation_beta.*` | ablation section |

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `Evaluation classifier not found` | `python -m src.train_classifier --config configs/classifier.yaml` |
| MPS / CUDA error | add `--set device=cpu` to the command (slow but works) |
| Inception download fails (firewall, offline) | `python -m src.evaluate ... --feature-spaces classifier` (classifier metrics only) |
| Out of memory | `--set batch_size=64` for **both** models, then rerun everything (it is a shared setting) |
| "Note: replacing previous results" | Expected: retraining a run deletes its old folder |
| A run stopped half-way | rerun the same command; that run starts again from scratch |

## 7. Google Colab (no local GPU)

```python
!git clone https://github.com/HoaKatie/image-generation-comparison-comp-vision.git
%cd image-generation-comparison-comp-vision
!git checkout feature/initial-scaffold
!pip install -q -r requirements.txt
!python -m pytest -q
!bash scripts/run_main.sh
!zip -qr runs_and_results.zip runs results     # then download the zip from the Files panel
```

Choose *Runtime → Change runtime type → GPU* first. Colab sessions disconnect after a while,
so run one seed per session (`!SEEDS="0" bash scripts/run_main.sh`) and download the zip after
each one.
