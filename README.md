# VAE vs. GAN: Fidelity–Diversity Trade-off in Image Generation

COMP 4102 final project. We compare two generative models, a **Variational Autoencoder (VAE)**
and a **Deep Convolutional GAN (DCGAN)**, on Fashion-MNIST.

> **Research question.** With the *same* network architecture, latent size, data and training
> budget, how do a VAE and a GAN trade off **fidelity / sharpness** against **diversity / mode
> coverage**?

| | Hypothesis | Evidence |
|---|---|---|
| **H1** | The GAN produces sharper samples and a lower FID. | FID, precision, Laplacian variance, high-frequency ratio |
| **H2** | The VAE covers the data distribution better (higher recall, less mode collapse) and trains more stably. | recall, class histogram, variance over seeds, D(x)/D(G(z)) curves |
| **H3** | The VAE has a smoother latent space. | interpolation figures, interpolation step CV |

## Start here

New to the repo? Read in this order. It takes about 30 minutes and needs no GPU, because
every result is already committed.

| # | Read | To learn |
|---|---|---|
| 1 | This README | the question, the results at a glance, how fairness is enforced |
| 2 | [docs/PIPELINE.md](docs/PIPELINE.md) | **how the project works, step by step**: what each script does, why, and which saved output to look at |
| 3 | [`results/`](results/) | the figures; start with `qualitative_samples.png` and `precision_recall.png` |
| 4 | [`runs/`](runs/) | one folder per run: its config, loss log, samples after every epoch, evaluation figures |
| 5 | [report/report.pdf](report/report.pdf) | the full write-up, with the explanation of every result |
| 6 | [docs/RUNNING_EXPERIMENTS.md](docs/RUNNING_EXPERIMENTS.md) | how to re-run or extend the experiments |
| 7 | [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) | protocol, metric definitions, rubric mapping, team decisions, task split |

## Results at a glance

Mean over 3 seeds. Full table: [results/main_results.md](results/main_results.md).

| | DCGAN | VAE | Real data |
|---|---|---|---|
| FID, classifier features ↓ | **1.29** | 20.7 | 0.03 |
| FID, Inception features ↓ | **13.3** | 56.9 | 2.4 |
| Precision / recall ↑ | **0.85 / 0.85** | 0.70 / 0.49 | 0.90 / 0.89 |
| Sharpness: Laplacian variance (closest to real is best) | **0.146** | 0.038 | 0.150 |
| Latent smoothness: interpolation CV ↓ | 0.69 | **0.51** | – |

- **H1 supported.** DCGAN samples are as sharp as real images; VAE samples are blurry.
- **H2 not supported.** The DCGAN also covers the data better and never mode-collapsed. The
  VAE's blur costs it diversity as well as fidelity (report §5 explains why).
- **H3 supported.** The VAE's latent space is smoother, with a caveat from the β ablation.

## 🚧 Project status: what is NOT finished yet

| File / item | Status | What is missing |
|---|---|---|
| `results/`, `runs/` | ✅ done | Main experiment (3 seeds) and ablations (seed 0). Every run's logs, samples, metrics, figures and final model are committed. To reproduce: [docs/RUNNING_EXPERIMENTS.md](docs/RUNNING_EXPERIMENTS.md). |
| [`report/report.md`](report/report.md) | ⚠️ **draft, needs team review** | All sections written from `results/` (6 pages + references). Team names are still missing. §5 Limitations assumes "lead with FID-classifier" (decision 3): change it if the team decides otherwise. |
| [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) §6 | ⚠️ open | 2 team decisions left: primary FID and optional extension. Epochs: decided, keep 30. |

Everything else is implemented and tested (`python -m pytest`, 49 tests): data, models,
`vae_loss`, training, evaluation, the real-vs-real baseline, analysis, and the scripts. The
whole pipeline was checked end to end, both on a toy dataset (tests) and on Fashion-MNIST
(smoke test).

## Fair comparison by construction

- **One architecture.** `ConvDecoder` is both the VAE decoder and the GAN generator, and
  `ConvEncoder` is both the VAE encoder and the GAN discriminator
  ([src/models/networks.py](src/models/networks.py)). `tests/test_models.py` checks that they match.
- **One settings file.** Dataset, latent size, batch size, epochs and seed live in
  [configs/shared.yaml](configs/shared.yaml) and are inherited by both model configs.
- **One evaluation.** Both models are scored by the same script with the same latent vectors
  (fixed `--eval-seed`).

Only the training objective and the optimizer settings recommended by each model's paper differ.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The code runs on CUDA, Apple-silicon (MPS) or CPU and picks the device automatically.
Fashion-MNIST (~30 MB) downloads on first use. The FID Inception weights (~100 MB) download on
the first full evaluation.

## Key design choice: the VAE loss

`vae_loss` in [src/losses.py](src/losses.py) sums the squared error over the pixels of each
image (a Gaussian decoder with fixed variance). It does not average it. Averaging would make
the KL term about 1000× stronger (1024 pixels vs. 32 latent dimensions), which is a hidden
β ≈ 1000 and makes the samples much blurrier. With the sum, `beta` is the only weight on the
KL. This is the part of the project most likely to come up in the Q&A. Its tests:
`python -m pytest tests/test_losses.py`.

## Quick check (~2 min)

```bash
bash scripts/smoke_test.sh          # tests + tiny runs; writes runs_smoke/ and results_smoke/
```
                                                            
## Reproduce the experiments

Step-by-step guide (time estimates, splitting seeds across teammates, Colab,
troubleshooting): [docs/RUNNING_EXPERIMENTS.md](docs/RUNNING_EXPERIMENTS.md).

```bash
bash scripts/run_main.sh            # VAE vs DCGAN, seeds 0 1 2  -> runs/, results/
bash scripts/run_ablation.sh        # latent_dim 8/128 and VAE beta 0.5/4
python -m src.analysis              # rebuild results/ tables and figures at any time
```

Long runs can be split across teammates: `SEEDS="1" bash scripts/run_main.sh`, then merge the
`runs/` folders. On Google Colab, clone the repo, `pip install -r requirements.txt`, and run the
same commands.

Each script can also be run on its own. Every script takes the same `--config` / `--set` flags:

```bash
python -m src.train_classifier --config configs/classifier.yaml           # once per dataset
python -m src.real_baseline                                                # real-vs-real reference row
python -m src.train_vae --config configs/vae.yaml --set seed=1 beta=4
python -m src.train_gan --config configs/gan.yaml --set seed=1 latent_dim=8
python -m src.evaluate  --config configs/gan.yaml --set seed=1             # full metrics + figures
python -m src.evaluate  --config configs/gan.yaml --set seed=1 --all-epochs --quick   # learning curve
```

## Outputs

| Path | Content |
|---|---|
| `runs/<run>/train_log.csv` | per-epoch losses (GAN: also D(x), D(G(z))) |
| `runs/<run>/samples/` | samples from the same fixed latent vectors after every epoch |
| `runs/<run>/eval/epoch_XXX/` | `metrics.json` + samples, success / failure cases, interpolation, nearest neighbours, reconstructions |
| `runs/<run>/checkpoints/epoch_030.pt` | the final model (intermediate checkpoints stay local) |
| `results/main_results.md` | main table, mean ± std over seeds |
| `results/*.png` | report figures: precision–recall, mode coverage, sharpness, learning curves, ablations |
| `results/all_runs.csv` | every final metric of every run |

`runs/` **is committed** (about 8 MB per run), so the whole team can inspect every run and load
the final models without retraining. Only the Inception feature cache and the intermediate
checkpoints are git-ignored. `data/` is git-ignored (downloaded automatically).

## Repository layout

```
configs/            shared.yaml (fairness) + vae.yaml, gan.yaml, classifier.yaml
src/
  data.py           Fashion-MNIST / MNIST / CIFAR-10 -> 32x32 in [-1, 1]
  models/           networks.py (shared blocks), vae.py, gan.py, classifier.py (evaluation only)
  losses.py         VAE negative ELBO and GAN losses
  train_vae.py, train_gan.py, train_classifier.py
  metrics.py        FID, precision/recall, mode coverage, sharpness, PSNR, NN, latent smoothness
  evaluate.py       all metrics + qualitative figures for one run
  real_baseline.py  same metrics on real train vs. test images (best achievable scores)
  analysis.py       aggregates all runs into tables and figures
scripts/            run_main.sh, run_ablation.sh, smoke_test.sh
tests/              shape/fairness checks, loss tests, metric sanity checks
runs/               one folder per run: config, loss log, per-epoch samples, metrics, figures, final model
results/            report tables + figures (created by src.analysis, then committed)
report/             report.md (source) -> report.pdf (committed) with build.sh
docs/               PIPELINE.md (start here), RUNNING_EXPERIMENTS.md, PROJECT_PLAN.md
```

## References

- Kingma & Welling (2014). *Auto-Encoding Variational Bayes.* ICLR.
- Goodfellow et al. (2014). *Generative Adversarial Nets.* NeurIPS.
- Radford, Metz & Chintala (2016). *Unsupervised Representation Learning with Deep Convolutional GANs.* ICLR.
- Higgins et al. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR.
- Heusel et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* (FID). NeurIPS.
- Kynkäänniemi et al. (2019). *Improved Precision and Recall Metric for Assessing Generative Models.* NeurIPS.
- Pech-Pacheco et al. (2000). *Diatom autofocusing in brightfield microscopy: a comparative study* (variance of Laplacian). ICPR.
- White (2016). *Sampling Generative Networks* (slerp).
- Karras et al. (2019). *A Style-Based Generator Architecture for GANs* (perceptual path length). CVPR.
- Xiao, Rasul & Vollgraf (2017). *Fashion-MNIST.* arXiv:1708.07747.
- torchmetrics / torch-fidelity for the FID Inception-v3 network.
