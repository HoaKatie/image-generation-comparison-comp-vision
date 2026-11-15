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

The full plan (protocol, metrics, rubric mapping, open decisions, timeline) is in
[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md).

## 🚧 Project status: what is NOT finished yet

| File / item | Status | What is missing |
|---|---|---|
| [`src/losses.py`](src/losses.py): `vae_loss()` | ❌ **TODO** | Raises `NotImplementedError`. Until it is implemented, `src/train_vae.py`, `scripts/run_main.sh`, `scripts/run_ablation.sh` and `scripts/smoke_test.sh` fail, and the 5 tests in `tests/test_losses.py` fail. Everything else passes (18/23 tests). |
| `results/` | ❌ not created | No real experiments have been run yet. Only short smoke runs were used to check the pipeline. |
| [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) §6 | ⚠️ open | Team decisions marked **[DECIDE]**: sum vs. mean in `vae_loss`, number of epochs, primary FID, optional extension. |
| Report | ❌ not started | Outline and figure plan in `docs/PROJECT_PLAN.md` §9. |

Everything else (data, models, GAN + classifier training, evaluation, analysis, scripts, tests)
is implemented and was checked end-to-end on short runs.

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

## ⚠️ Team TODO before the first run

`vae_loss` in [src/losses.py](src/losses.py) is intentionally left for the team to implement
(about 5 lines). It is the part of the project most likely to come up in the Q&A. The comments
in the function describe what it needs and a design decision to discuss. It is done when this passes:

```bash
python -m pytest tests/test_losses.py
```

## Quick check (~2 min)

```bash
bash scripts/smoke_test.sh          # tests + tiny runs; writes runs_smoke/ and results_smoke/
```

## Reproduce the experiments

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
| `results/main_results.md` | main table, mean ± std over seeds |
| `results/*.png` | report figures: precision–recall, mode coverage, sharpness, learning curves, ablations |
| `results/all_runs.csv` | every final metric of every run |

`runs/` and `data/` are git-ignored. Commit `results/` once the final runs are done.

## Repository layout

```
configs/            shared.yaml (fairness) + vae.yaml, gan.yaml, classifier.yaml
src/
  data.py           Fashion-MNIST / MNIST / CIFAR-10 -> 32x32 in [-1, 1]
  models/           networks.py (shared blocks), vae.py, gan.py, classifier.py (evaluation only)
  losses.py         VAE ELBO (team TODO) and GAN losses
  train_vae.py, train_gan.py, train_classifier.py
  metrics.py        FID, precision/recall, mode coverage, sharpness, PSNR, NN, latent smoothness
  evaluate.py       all metrics + qualitative figures for one run
  analysis.py       aggregates all runs into tables and figures
scripts/            run_main.sh, run_ablation.sh, smoke_test.sh
tests/              shape/fairness checks, loss acceptance tests, metric sanity checks
docs/PROJECT_PLAN.md
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
