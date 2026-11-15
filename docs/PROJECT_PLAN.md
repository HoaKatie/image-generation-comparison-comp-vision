# Project plan: VAE vs. GAN

Living document for team discussion. Edit freely. Items marked **[DECIDE]** need a team decision.

## 1. Research question and hypotheses

> With the same architecture, latent size, data and training budget, how do a VAE and a GAN
> trade off **fidelity / sharpness** against **diversity / mode coverage**?

| | Hypothesis | Metric(s) | Figure / table |
|---|---|---|---|
| H1 | GAN samples are sharper and have a lower FID | FID, precision, Laplacian variance, HF ratio | `main_results.md`, `sharpness.png`, `precision_recall.png` |
| H2 | VAE has better coverage (recall), less mode collapse, more stable training | recall, modes covered, class KL, std over seeds | `precision_recall.png`, `mode_coverage.png`, `training_curves.png` |
| H3 | VAE latent space is smoother | interpolation step CV | `interpolation.png` per run, `main_results.md` |

A hypothesis that turns out **false** is still a good result if we explain why. The rubric
rewards analysis, not "winning".

## 2. Fair-comparison protocol

| Shared (configs/shared.yaml + code) | Different (and why) |
|---|---|
| Dataset, preprocessing (pad to 32×32, scale to [-1, 1]) | Training objective: the thing we are studying |
| Image-producing network: `ConvDecoder` | Optimizer settings: each uses its paper's standard (VAE Adam 1e-3; DCGAN Adam 2e-4, β1 = 0.5) |
| Latent size, batch size, epochs, seeds | VAE encoder head outputs 2·latent_dim numbers; GAN discriminator outputs 1 |
| DCGAN weight initialisation | |
| Evaluation script, latent vectors and metrics | |

## 3. Experiment matrix

| # | Experiment | Runs | Seeds | Script |
|---|---|---|---|---|
| E1 | Main comparison (z = 32, β = 1) | VAE, DCGAN | 0, 1, 2 | `scripts/run_main.sh` |
| E2 | Learning curves (every 5 epochs) | same as E1 | 0, 1, 2 | `run_main.sh` (`--all-epochs --quick`) |
| E3 | Latent dimension 8 / 32 / 128 | VAE, DCGAN | 0 (more if time) | `scripts/run_ablation.sh` |
| E4 | β-VAE: β = 0.5 / 1 / 4 | VAE | 0 (more if time) | `scripts/run_ablation.sh` |
| E5 *(optional)* | Second dataset (CIFAR-10): `--set dataset=cifar10` | VAE, DCGAN | 0 | manual |

## 4. Metrics

| Metric | Measures | Better | Notes |
|---|---|---|---|
| FID (Inception) | overall quality + diversity | ↓ | Standard, but Inception was trained on ImageNet photos. Use it to compare our own models, not to compare with numbers in papers. |
| FID (classifier) | same, in a Fashion-MNIST feature space | ↓ | Uses our evaluation classifier. Faster and more domain-appropriate. |
| Precision | fidelity: fakes inside the real manifold | ↑ | Kynkäänniemi et al. 2019, k = 3 |
| Recall | diversity: reals covered by the fake manifold | ↑ | Directly tests H2 |
| Modes covered / class KL | mode collapse | ↑ / ↓ | Generated samples labelled by the evaluation classifier |
| Laplacian variance | sharpness (edges) | closer to real | Classical CV: 2nd-derivative filter (Edges lecture) |
| HF ratio | share of energy at high frequencies | closer to real | Classical CV: Fourier transform (Fourier lecture). **Above real = noise/artefacts, not sharpness.** |
| Interp. step CV | latent smoothness | ↓ | Uneven steps along a slerp path = "jumpy" latent space |
| NN ratio | memorisation | ≈ 1 | median(gen→train) / median(test→train) pixel distance. ≪ 1 means the model copies training data. |
| Recon PSNR | reconstruction quality | ↑ | VAE only. The GAN has no encoder; state this as a limitation. |

Sanity checks for every metric are in `tests/test_metrics.py`. Mention them in the report as
evidence that the evaluation measures what we claim. Real-vs-real check on the test set:
classifier-FID ≈ 0.1, precision/recall ≈ 0.91.

## 5. Rubric mapping (10 marks)

| Component | Marks | Where we address it |
|---|---|---|
| Problem definition & scope | 1 | Research question + hypotheses (README, report §1). One dataset, two models: manageable. |
| Technical approach | 2 | Two models with a shared architecture; ablations; classical-CV sharpness metrics; our own evaluation classifier. |
| Implementation | 2 | `src/` trains, evaluates and analyses end-to-end; `scripts/smoke_test.sh` proves it runs. |
| Experiments & evaluation | 2 | E1–E4, 3 seeds, 10+ quantitative metrics, systematic qualitative figures (random grids, success/failure by classifier confidence, nearest neighbours). |
| Analysis & understanding | 2 | Test H1–H3 explicitly. Explain *why*: Gaussian likelihood → averaging → blur; adversarial loss → sharp but mode dropping. Discuss failure cases and limitations. |
| Report, code, reproducibility | 1 | README, configs saved with every run, fixed seeds, tests, citations. |

## 6. Open decisions **[DECIDE]**

1. **`vae_loss` reconstruction reduction: sum or mean over pixels?** This changes the effective
   KL weight and therefore the blur. Recommendation: sum (Gaussian decoder with fixed variance);
   β then controls the trade-off explicitly. Whatever we choose, we justify it in the report.
2. **Epochs.** Default is 30. Check the learning curves after the first seed. If FID is still
   dropping, increase the epochs for *both* models.
3. **Primary FID.** Inception (standard) or classifier (domain-specific)? We report both, but
   the text should lead with one.
4. **Optional extension.** Pick one if time allows: WGAN-GP (does it fix mode collapse?) *or*
   CIFAR-10 (does the trade-off hold on colour images?).

## 7. Suggested task split

Everyone must be able to explain **both** models and the metrics, because the Q&A can ask any
member (guidelines §10).

| Member | Main responsibility | Also |
|---|---|---|
| A | VAE: `vae_loss` TODO, VAE runs, β ablation | report §2 Method (VAE) |
| B | GAN: GAN runs, latent-dim ablation, training-stability analysis | report §2 Method (GAN) |
| C | Evaluation: metrics review, analysis figures, failure-case selection | report §4–5 |

## 8. Timeline (Demo/Q&A window: Dec 2–11; confirm the submission deadline)

| Dates | Milestone |
|---|---|
| Nov 14–20 | `vae_loss` done, smoke test green, seed 0 of E1 trained and inspected |
| Nov 21–27 | E1 (3 seeds) + E2 done; decide epochs; start E3/E4 |
| Nov 28–Dec 4 | E3/E4 done, `results/` committed, report written |

## 9. Report outline (4–6 pages)

1. **Problem & motivation** (~0.5 p): fidelity vs. diversity; why it matters.
2. **Method** (~1 p): VAE (ELBO, reparameterization trick), GAN (minimax, non-saturating loss),
   why VAEs blur and GANs drop modes.
3. **Implementation & data** (~1 p): shared architecture figure, protocol table (§2), evaluation
   classifier and its accuracy.
4. **Experiments & results** (~1.5 p): `main_results.md`, `precision_recall.png`,
   `mode_coverage.png`, a sample grid per model, ablation figures.
5. **Analysis & discussion** (~1–1.5 p): H1–H3 verdicts, success/failure cases, learning curves,
   limitations (§10).
6. **Conclusion** (~0.25 p).

## 10. Known limitations (for the report)

- FID-Inception on 32×32 grayscale images is outside Inception's training domain. That is why
  we also report the classifier-based metrics.
- The evaluation classifier is imperfect (state its accuracy). Ambiguous samples may be mislabelled.
- The GAN has no encoder, so reconstruction metrics apply to the VAE only.
- One architecture size and one dataset. Conclusions may not transfer to large models
  (e.g. diffusion-era architectures).
