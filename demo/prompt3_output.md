# Study guide: project structure A→Z, final requirements check, and simplification plan

Three parts:

- **Part A** explains every folder and file in the order the code is used, so you can study
  the code. It covers what the config files mean, what the model scripts do, what training
  involves, how the report's `.png` figures are made, and how the Tkinter demo code works.
- **Part B** is a final check of the code, interface, demo and report against the course
  guidelines, with the issues still to fix.
- **Part C** suggests how to **simplify** the code so it looks like something a student in
  this course could build. **No code was changed**; these are suggestions only.

> Companion file: [prompt2_output.md](prompt2_output.md) is the demo-style walkthrough of the
> project (60-second pitch, VAE vs GAN, demo script with numbers, Q&A answers).

---

# Part A. Project structure, A→Z

## A1. Folder map

Line counts are real, measured on branch `feature/demo-app`.

```
image-generation-comparison-comp-vision/
├── configs/                    SETTINGS (YAML). Who trains with what.
│   ├── shared.yaml             settings BOTH models must share (fairness)
│   ├── vae.yaml                VAE-only: lr, beta
│   ├── gan.yaml                GAN-only: lr, adam_beta1
│   └── classifier.yaml         evaluation classifier: epochs, lr
├── src/                        THE PIPELINE (1,678 lines)
│   ├── config.py        45     load YAML, inherit base, apply --set overrides
│   ├── data.py          84     Fashion-MNIST → 32×32 in [-1, 1]
│   ├── utils.py         93     seed, device, run folder names, save grids/CSV/JSON
│   ├── models/
│   │   ├── networks.py  78     ConvDecoder + ConvEncoder (shared by both models)
│   │   ├── vae.py       43     VAE = ConvEncoder + reparameterize + ConvDecoder
│   │   ├── gan.py       16     GAN = ConvDecoder (generator) + ConvEncoder (discriminator)
│   │   ├── classifier.py 36    evaluation CNN (measuring instrument)
│   │   └── __init__.py  65     build_model, save/load checkpoint, load_classifier
│   ├── losses.py        53     vae_loss, discriminator_loss, generator_loss
│   ├── train_classifier.py 65  step 3: train the measuring instrument
│   ├── train_vae.py     85     step 4: train the VAE
│   ├── train_gan.py    102     step 4: train the DCGAN
│   ├── real_baseline.py 72     step 5: real-vs-real "best possible score"
│   ├── metrics.py      198     every metric (FID, P/R, sharpness, ...)
│   ├── evaluate.py     230     step 6: checkpoint → metrics.json + figures
│   └── analysis.py     413     step 7: all metrics.json → results/ tables + .png
├── scripts/                    run_main.sh (33), run_ablation.sh (26), smoke_test.sh (21)
├── tests/                      49 automated tests in 7 files (564 lines)
├── runs/                       OUTPUT of training + evaluation, one folder per run (committed)
├── results/                    OUTPUT of analysis.py: tables (.md) + figures (.png) (committed)
├── report/                     report.md → build.sh → report.pdf / report.docx
├── docs/                       PIPELINE.md, PROJECT_PLAN.md, RUNNING_EXPERIMENTS.md
├── demo/                       Tkinter demo (857 lines incl. tests)
├── data/                       Fashion-MNIST download (git-ignored)
├── requirements.txt            pip packages
└── README.md                   start here
```

## A2. How one experiment flows through the files

```
configs/vae.yaml ──(config.py merges with shared.yaml)──► cfg dict
        │
        ▼
train_vae.py:  data.py loader ─► models.build_model(cfg) ─► loop: losses.vae_loss ─► Adam
        │
        ▼  writes runs/vae_fashion_mnist_z32_beta1_seed0/
        │        config.yaml · train_log.csv · samples/epoch_XXX.png · checkpoints/epoch_030.pt
        ▼
evaluate.py:   load checkpoint ─► generate 10,000 images ─► metrics.py ─► eval/epoch_030/metrics.json + figures
        │
        ▼
analysis.py:   every runs/*/eval/*/metrics.json ─► results/*.md + results/*.png
        │
        ▼
report/report.md (references ../results/*.png) ─► build.sh ─► report.pdf
```

Each step only reads files the previous step wrote. That's why you can rerun one step
without the others, e.g. `python -m src.analysis` rebuilds every figure in seconds.

## A3. The config files: what each key means

Configs are YAML files, i.e. plain `key: value` lines. `src/config.py` reads them:

1. It loads the model file, e.g. `vae.yaml`.
2. That file says `base: shared.yaml`, so it loads `shared.yaml` first and lets
   `vae.yaml` override or add keys.
3. It applies command-line overrides: `--set seed=1 beta=4`. A key that doesn't exist is an
   **error**, so a typo can't be silently ignored.

The merged result is saved with every run as `runs/<run>/config.yaml`.

**`configs/shared.yaml`: settings both models MUST share**

| Key | Value | Meaning |
|---|---|---|
| `dataset` | `fashion_mnist` | which dataset (`mnist` and `cifar10` also work) |
| `data_root` | `data` | where it is downloaded |
| `output_dir` | `runs` | where run folders are written |
| `latent_dim` | `32` | length of the random vector z (size of the "code") |
| `base_channels` | `64` | network width: the decoder uses 256 → 128 → 64 channels (4×, 2×, 1× base) |
| `batch_size` | `128` | images per training step |
| `epochs` | `30` | passes over the 60,000 training images (= 14,040 steps) |
| `seed` | `0` | random seed (we use 0, 1, 2) |
| `save_every` | `5` | save a checkpoint every 5 epochs (for learning curves) |
| `num_workers` | `2` | background processes loading data |
| `device` | `auto` | CUDA → Apple MPS → CPU, whichever exists |
| `subset` | `null` | e.g. `1000` = train on 1,000 images only (smoke test) |

**`configs/vae.yaml`**: `model: vae`, `lr: 1e-3` (Adam learning rate), `beta: 1.0`
(weight of the KL term; 1 = standard VAE, > 1 = β-VAE).

**`configs/gan.yaml`**: `model: gan`, `lr: 2e-4`, `adam_beta1: 0.5`. These are the DCGAN
paper's settings. Adam's β1 = 0.5 (instead of 0.9) gives less momentum, which keeps the
two-player game more stable.

**`configs/classifier.yaml`**: `model: classifier`, `epochs: 8`, `lr: 1e-3`.

**Why this design:** "fairness by construction". `tests/test_config.py` fails if `vae.yaml`
or `gan.yaml` ever overrides a shared key, so the two models can't drift apart by accident.

**Run names** come from the config (`run_name` in `utils.py`):
`vae_fashion_mnist_z32_beta1_seed0`, `gan_fashion_mnist_z32_seed0`. Changing the seed or β
therefore creates a new folder, and rerunning the same config **replaces** its folder.

## A4. Data: `src/data.py`

| Function | What it does |
|---|---|
| `_transform` | Pad 28×28 → 32×32 (`transforms.Pad(2)`), convert to tensor, normalise to [−1, 1] |
| `get_dataset` | torchvision `FashionMNIST(..., download=True)` |
| `get_loader` | a shuffled `DataLoader` for training (optionally only `subset` images) |
| `load_split` | the whole test set as one tensor, read straight from the raw arrays; much faster for evaluation and gives exactly the same values (a test checks this) |
| `class_names` | `['T-shirt/top', 'Trouser', ..., 'Ankle boot']` |

**Why pad, not resize:** resizing interpolates, which blurs the *real* images. The real
images are the reference for sharpness, so blurring them would favour the blurry VAE.

## A5. The models: `src/models/`

**`networks.py`: the two building blocks**

```
ConvDecoder (z → image)                         ConvEncoder (image → vector)
z (32)  ─► ConvT 4×4  ─► 256 ch, 4×4             image 1×32×32 ─► Conv ─► 64 ch, 16×16
        ─► ConvT      ─► 128 ch, 8×8                            ─► Conv ─► 128 ch, 8×8
        ─► ConvT      ─►  64 ch, 16×16                          ─► Conv ─► 256 ch, 4×4
        ─► ConvT      ─►   1 ch, 32×32 ─► tanh                  ─► head Conv 4×4 ─► out_dim numbers
(BatchNorm + ReLU between layers)               (LeakyReLU 0.2; no BatchNorm on the first layer)
```

- `init_dcgan_weights`: weights ~ N(0, 0.02), applied to **both** models.
- `count_parameters`: the decoder has 788,353 parameters.

**`vae.py`**: `encoder = ConvEncoder(out_dim = 2 × 32)`. Its 64 outputs are split into μ (32)
and log σ² (32). `reparameterize` computes `z = μ + exp(0.5·log σ²) · ε`. `forward(x)` returns
`(x̂, μ, log σ²)`, `decode(z)` generates, and `reconstruct(x)` uses μ (no noise).

**`gan.py`**: `generator = ConvDecoder`, `discriminator = ConvEncoder(out_dim = 1)`: one
"real?" logit. It has `decode(z)` too, so evaluation code can treat both models the same way.

**`classifier.py`**: 5 conv layers → 128-d feature vector (`embed`) → 10 class logits. It is
not compared; it measures. `embed()` gives the feature space for FID and precision/recall.

**`__init__.py`**:
- `build_model(cfg)` picks VAE or GAN from `cfg["model"]`.
- `save_checkpoint` stores the weights, the config, the epoch and the training time.
- `load_checkpoint` rebuilds the model from the config **stored in the checkpoint**, so you
  never need to remember which settings a file was trained with.

## A6. Training: what it involves

**The losses (`src/losses.py`)**

```
vae_loss          = Σ_pixels (x − x̂)²   +   β · ½ Σ_j (μ_j² + σ_j² − log σ_j² − 1)
                    └ reconstruction ┘       └ KL to N(0, I), closed form ┘
discriminator_loss = BCE(D(x), 1) + BCE(D(G(z)), 0)          D: real → 1, fake → 0
generator_loss     = BCE(D(G(z)), 1)   = −log D(G(z))        non-saturating
```

**Sum over pixels, not mean.** This is the most important line in the repo. A mean would
make the KL about 1000× stronger (a hidden β ≈ 1000).

**One training step, side by side**

| VAE (`train_vae.py`) | DCGAN (`train_gan.py`) |
|---|---|
| `x̂, μ, logσ² = model(x)` | `z = randn(128, 32)`, `fake = G(z)` |
| `loss = vae_loss(...)` | **D step:** `loss_d = D_loss(D(x), D(fake.detach()))` → `opt_d.step()` |
| `loss.backward()`, `optimizer.step()` | **G step:** `loss_g = G_loss(D(fake))` → `opt_g.step()` |
| 1 network, 1 optimizer | 2 networks, 2 optimizers |

`fake.detach()` matters: the D step must not update G.

**Shared by both scripts:**
1. `set_seed(seed)` for reproducibility.
2. `get_loader(...)`.
3. `build_model(cfg)` + `init_dcgan_weights`.
4. `prepare_run_dir` creates the run folder and saves `config.yaml`.
5. A fixed set of 64 latent vectors (`fixed_noise`, seed 1234) is decoded after **every
   epoch** into `samples/epoch_XXX.png`, so you can flip through them and watch the model
   learn.
6. `train_log.csv` gets one row per epoch: VAE `loss, recon, kl`; GAN `loss_d, loss_g,
   d_real, d_fake`. `d_real` is the mean D(x), and `d_fake` the mean D(G(z)) (0.17–0.32 in
   all our runs = healthy).
7. A checkpoint is saved every 5 epochs; the final one is `epoch_030.pt`.

**`train_classifier.py`** is ordinary supervised learning: cross-entropy for 8 epochs, then
test accuracy is measured (90.7 %). It saves `runs/classifier_fashion_mnist/classifier.pt`.

## A7. Evaluation: `src/evaluate.py` + `src/metrics.py`

`python -m src.evaluate --config configs/gan.yaml --set seed=1`

1. Find `runs/<run>/checkpoints/epoch_030.pt` (or all of them with `--all-epochs`).
2. `Evaluator` loads, **once**, the 10,000 test images and the classifier (Inception lazily).
3. `evaluate(ckpt)`:

| # | Code | Metric function(s) in `metrics.py` | Output key(s) |
|---|---|---|---|
| 0 | `generate(model, 10000, 32, seed=0)` | – | the fake images (same z for both models) |
| 1 | features of real & fake, in classifier and Inception space | `frechet_distance`, `precision_recall` | `fid_*`, `precision_*`, `recall_*` |
| 2 | classifier softmax on fakes | `mode_coverage` | `class_histogram`, `modes_covered`, `class_kl`, `mean_confidence` |
| 3 | fake and real images | `laplacian_variance`, `high_frequency_ratio` | `laplacian_var(_real)`, `hf_ratio(_real)` |
| 4 | 100 slerp paths × 10 frames | `slerp`, `path_smoothness` | `interp_step_cv` |
| 5 | VAE only: reconstruct test images | `psnr` | `recon_mse`, `recon_psnr` |
| 6 | full mode only: nearest training image | `nearest_neighbors` | `memorization_ratio` |
| 7 | figures | `save_grid` | `samples.png`, `most/least_confident.png`, `interpolation.png`, `nearest_neighbors.png`, `reconstructions.png` |

Everything goes into `runs/<run>/eval/epoch_030/metrics.json`. `--quick` (5,000 samples,
classifier space only, no figures) writes `epoch_XXX_quick/` for the learning curves.

**How the key metrics work, in one line each:**
- **FID**: fit a Gaussian (mean + covariance) to real features and one to fake features, then
  measure the distance between them. 0 = identical.
- **Precision / recall**: draw a ball around each real feature that reaches its 3rd nearest
  neighbour. Precision = % of fakes inside some real ball. Recall = the reverse.
- **Laplacian variance**: convolve with a 3×3 Laplacian kernel `[[0,1,0],[1,−4,1],[0,1,0]]`
  and take the variance of the response. Weak edges mean low variance.
- **HF ratio**: take the 2-D FFT and compute energy above 0.25 cycles/pixel ÷ total energy.
- **Class KL**: KL(generated class shares ‖ real class shares). 0 = perfectly balanced.

**`real_baseline.py`** runs the same metrics on 10,000 real **training** images in place of
generated ones. That gives the "Real data" row (the best score a perfect generator could
get). It reuses `Evaluator`.

## A8. How the report figures (`.png`) are generated: `src/analysis.py`

`python -m src.analysis` rebuilds the whole `results/` folder. It works in four stages.

**Stage 1, load.** `load_evals` collects every `runs/*/eval/epoch_*/metrics.json` into one
pandas DataFrame, with one row per evaluation.

**Stage 2, filter.**
- `final_full_evals`: keep the **last non-quick** evaluation of each run.
- `default_mask`: keep the main comparison only (z = 32 and, for the VAE, β = 1).

**Stage 3, tables.** `mean_std_table` groups runs that differ only by seed and formats each
metric as `mean ± std` (sample std, ddof = 1). `write_markdown` writes `main_results.md`,
`ablation_latent_dim.md` and `ablation_beta.md`. If `runs/real_baseline_*/metrics.json`
exists, a "Real data" row is added.

**Stage 4, figures.** Each figure is a short matplotlib function with a shared style
(`_style()`: VAE blue `#2a78d6`, DCGAN orange `#eb6834`, real data = grey dashed line).

| Figure | Function | Built from | How |
|---|---|---|---|
| **`mode_coverage.png`** (the class distribution figure) | `plot_mode_coverage` | `class_histogram` of every main run | see the walkthrough below |
| `precision_recall.png` | `plot_precision_recall` | `precision_*`, `recall_*` | scatter: one dot **per seed** (x = recall, y = precision), one panel per feature space; grey star = real baseline |
| `sharpness.png` | `plot_sharpness` | `laplacian_var`, `hf_ratio`, `*_real` | bars = mean over seeds, error bars = std, dashed line = real value |
| `learning_curve.png` | `plot_learning_curve` | the `--quick` evaluations, epochs 5…30 | line = mean over seeds, shaded band = min–max |
| `training_curves.png` | `plot_training_curves` | each run's `train_log.csv` | VAE recon and KL per epoch; GAN D(x) solid, D(G(z)) dashed; one line per seed |
| `ablation_latent_dim.png`, `ablation_beta.png` | `plot_ablation` | ablation runs | error-bar lines (x = latent_dim on a log₂ axis, or β) |
| `qualitative_*.png` | `plot_qualitative` | the evaluation PNGs of seed 0 | loads both models' `samples.png` (etc.) with `plt.imread` and puts them side by side with captions |

**Walkthrough: how `mode_coverage.png` is built**

```
1. evaluate.py:  probs = classifier(fake).softmax(dim=1);  pred = argmax
                 class_histogram = np.bincount(pred, minlength=10) / 10000   → e.g. [0.152, 0.084, 0.257, ...]
                 saved in metrics.json (one list of 10 shares per run)
2. analysis.py:  for each model, stack the 3 seeds' histograms and take the mean
3.               ax.bar(x ± offset, shares)   → blue bars (VAE) beside orange bars (DCGAN), one pair per class
4.               ax.axhline(1/10, dashed)     → "balanced (real data)" reference at 10 %
5.               x labels = class names, rotated 30°; saved as results/mode_coverage.png
```

Reading it: the VAE's Pullover bar is far above 10 % and its Coat bar far below, while the
DCGAN's bars all sit near the line. That is H2's evidence.

**How the figures get into the report.** `report/report.md` contains lines like
`![Figure 2 ...](../results/precision_recall.png){width=45%}`. `report/build.sh` runs:
1. `pandoc` converts Markdown to HTML, with the images embedded.
2. Headless Google Chrome prints that HTML to `report.pdf`.
3. `pandoc` also writes `report.docx`.

If the runs change: rerun `src.analysis`, update the numbers in `report.md`, then rerun
`build.sh`.

## A9. Scripts and tests

| Script | Does |
|---|---|
| `scripts/run_main.sh` | trains the classifier and real baseline if missing, then for seeds 0 1 2 and both models: train → evaluate → quick-evaluate all epochs; finally `analysis` |
| `scripts/run_ablation.sh` | z = 8 and 128 for both models, β = 0.5 and 4 for the VAE (seed 0); then `analysis` |
| `scripts/smoke_test.sh` | ~2 min: pytest + tiny runs (1,000 images, 2 epochs) into `runs_smoke/`, never touching real results |

**Tests (`python -m pytest`, 49 tests):**

| File | Checks |
|---|---|
| `test_losses.py` | zero loss for a perfect reconstruction, KL closed form, β only scales the KL, D/G losses behave |
| `test_models.py` | output shapes, **VAE decoder ≡ GAN generator**, reparameterize gives the right mean/std |
| `test_metrics.py` | known answers: identical sets give FID 0, half coverage gives recall ≈ 0.5, blur lowers sharpness, ... |
| `test_config.py` | inheritance, typo errors, **shared settings not overridden**, unique run names |
| `test_data.py` | padding, fast loader = slow loader |
| `test_analysis.py` | tables pick the right runs, mean ± std is correct |
| `test_pipeline.py` | the whole pipeline end to end on a toy dataset |

## A10. The Tkinter demo code (`demo/`)

**Flow of a demo session**

```
compare_demo.py main()
   ├─ DemoEngine(...)                ← engine.py: load VAE, DCGAN, classifier, test set ONCE (~3 s)
   └─ loop:
        ask_category()               ← print 1..10 menu, read input(), validate
        engine.compare(class, seed)  ← engine.py: generate → classify → filter → stats
        print_comparison()           ← terminal table + verdict
        show_comparison()            ← gui.py: open Tk window, BLOCK until closed
      (window closed → back to the menu; 0 = quit)
```

**`engine.py`: the logic, no GUI**
- `_sample_class(model, class)` makes `generate(model, 10000, ...)` → classifier softmax →
  `keep = argmax == class` → returns the kept images and their confidence.
- `compare(class, seed)` does the following:
  1. samples both models;
  2. computes `n = min(500, #VAE kept, #GAN kept)` (equal n for fairness);
  3. computes `_stats` (sharpness, detail, spread) for each model and for real images of the
     class;
  4. adds share, confidence, and precision/recall (via `src/metrics.py`).

  It returns a `Comparison` dataclass, which is just a container of results.

**`compare_rules.py`: the if/else rules**
- `METRICS` is a list of `(key, label, "higher" | "closer", format)`.
- `winner(vae, gan, mode, real)` handles the two modes. For **"higher"**, it's a tie if the
  values are within 5 points. For **"closer"**, it compares `|value/real − 1|` and calls it a
  tie if the two distances are within 10 %.
- `table_rows()` produces the table, and `score()` counts the wins.
- `explain()` returns 8 `(heading, paragraph)` pairs, one small function each (`_sharpness`,
  `_precision_recall`, ...). Each is a chain of `if / elif / else` that picks a sentence and
  fills in the numbers.

**`gui.py`: the Tkinter window.** Concepts you need for Q&A:

| Tkinter concept | Where it's used |
|---|---|
| `tk.Tk()` root window + `mainloop()` | `ComparisonWindow.__init__` / `run()`. `mainloop` runs until the window is destroyed |
| `grid(row, column)` layout | `_build()`: row 0 title, 1 subtitle, 2 images, 3 table + text, 4 buttons |
| `ttk.LabelFrame` | boxed sections ("Real test images", "VAE", "DCGAN", "Statistics", ...) |
| `ttk.Label(image=...)` | shows a picture. `to_photo()` converts tensor → PIL image → ×6 or ×2 **nearest-neighbour** enlargement → `ImageTk.PhotoImage` |
| `self.photos = []` | keeps Python references to images. **Classic pitfall:** without it, Python garbage-collects the image and Tk shows a blank box |
| `ttk.Treeview(show="headings")` | the statistics table; `tag_configure` colours rows by winner |
| `tk.Text` + `Scrollbar` + tags | the "What the numbers say" panel; a "heading" tag makes titles bold; `state="disabled"` = read-only |
| `ttk.Button(command=...)`, `root.bind("<KeyPress-n>")` | buttons and keyboard shortcuts N / S / Esc |
| `show(comp)` | fills every widget; called again by "New samples" with seed + 1, so the layout is built once and the content swapped |
| `lift()`, `-topmost`, `focus_force()` | bring the window in front of the terminal (macOS) |

## A11. Suggested order for studying the code (~3 hours)

1. `configs/*.yaml` → `src/config.py` (15 min)
2. `src/data.py` (10 min)
3. `src/models/networks.py` → `vae.py` → `gan.py` (30 min, **most important**)
4. `src/losses.py` (20 min, **most important**; be able to derive the sum-vs-mean argument)
5. `src/train_vae.py` vs `src/train_gan.py` side by side (25 min)
6. `src/metrics.py`: FID, precision/recall, Laplacian, HF ratio (35 min)
7. `src/evaluate.py` `evaluate()` method only (15 min)
8. `src/analysis.py`: `main()`, then `plot_mode_coverage` and `plot_precision_recall` (20 min)
9. `demo/engine.py` → `compare_rules.py` → `gui.py` (30 min)

---

# Part B. Final check against the requirements

## B1. What I checked (2026-11-17)

| Check | Result |
|---|---|
| `python -m pytest` (pipeline) | ✅ 49 passed |
| `python -m pytest demo` | ✅ 8 passed |
| Demo, all 10 categories | ✅ ~1 s each; window opens; New samples / Save / reopen work; window 1170×781 px (fits a 13" laptop) |
| `report/report.pdf` length | ✅ 6 pages + 1 page of references (limit: 4–6 pages **excluding** references) |
| `report.pdf` newer than `report.md` | ✅ in sync (Nov 14 22:27 vs 22:18) |
| Report sections match guidelines §6 | ✅ Problem · Method · Implementation & data · Experiments · Analysis · Conclusion |
| Placeholders | ❌ `author: "TEAM MEMBER NAMES"` in `report/report.md` line 4 |

## B2. Requirements table (COMP 4102 guidelines)

| Guideline | Requirement | Status | Evidence |
|---|---|---|---|
| §3 Problem | clearly defined CV problem, reasonable scope | ✅ | research question + H1–H3 (README, report §1) |
| §3 Method | at least one meaningful CV method/model | ✅ | VAE + DCGAN + evaluation CNN + classical sharpness measures |
| §3 Implementation | working code | ✅ | 49 + 8 tests pass; smoke test; committed checkpoints |
| §3 Evaluation | how well it works **and where it does not** | ✅ | 10+ metrics, real baseline, failure cases (Fig. 4) |
| §3 Investigation | ≥ 1 meaningful comparison / ablation | ✅ | VAE vs DCGAN (3 seeds) + latent-dim and β ablations + 60-epoch check |
| §4 | representative successes **and failures** | ✅ | `most_confident` / `least_confident` figures, report §5 |
| §4 | explain *why* it succeeds or fails | ✅ | report §5: averaging → blur → lost diversity |
| §5 Code | README explaining how to run | ✅ | README "Setup", "Reproduce", "Interactive demo"; docs/PIPELINE.md |
| §5 Report | ~4–6 pages excl. references | ✅ | 6 + 1 |
| §5 Demo/Q&A | be ready for a brief demo and Q&A (Dec 2–11) | ✅ | `demo/` + `prompt2_output.md` (demo script, 17 Q&A answers) |
| §6 | cite methods, datasets, codebases | ✅ | 11 references incl. torchmetrics / torch-fidelity |
| §10 | every member understands the work | ⚠️ | depends on the team: everyone should rehearse the Q&A list |
| §11 AI use | allowed; you must be able to explain everything | ⚠️ | that's what Parts A and C are for |
| Original demo request | menu 1..n in terminal → Tkinter window with 2 models' images, stats, if/else comparison, run notes, own folder + branch | ✅ | `demo/`, branch `feature/demo-app` |

## B3. Rubric self-assessment (10 marks)

| Component | Marks | Assessment |
|---|---|---|
| Problem definition & scope | 1 | Strong: one dataset, two models, three testable hypotheses |
| Technical approach | 2 | Strong: shared architecture, own evaluation classifier, classical CV metrics, ablations. The risk is that it looks *too* polished for a student project (see Part C) |
| Implementation | 2 | Strong: complete, tested, reproducible |
| Experiments & evaluation | 2 | Strong: 3 seeds, 2 ablations, real baseline, successes and failures |
| Analysis & understanding | 2 | Strong *on paper*, because the report explains each mechanism. **The marks depend on the Q&A**: each member must explain it without notes |
| Report, code, reproducibility | 1 | Strong, once the team names are filled in |

## B4. Issues to fix before submission (in priority order)

1. **❌ Team names.** Replace `author: "TEAM MEMBER NAMES"` in `report/report.md`, then run
   `bash report/build.sh` to rebuild the PDF.
2. **⚠️ Open decisions are out of sync.** `docs/PROJECT_PLAN.md` §6 still lists decision 3
   (primary FID) as open, but the report already says "we lead with the classifier space".
   Also, README's "🚧 Project status" says the report is a draft needing team review. After the
   team agrees, mark decision 3 as decided and update the status table.
3. **✅ Demo merged into `main`** (2026-11-17), together with these study notes.
4. **✅ Study notes committed:** `demo/prompt2_output.md` and this file.
5. **⚠️ Stray IDE change.** `src/real_baseline.py` has an uncommitted change: your editor removed
   the blank lines between functions. It's harmless but it is noise. Run
   `git restore src/real_baseline.py` to discard it, or just don't stage it.
6. ℹ️ *Optional.* One sentence in report §3 mentioning the interactive demo (not required).

---

# Part C. Simplification plan (suggestions only; no code was changed)

## C1. Read this first: two rules

1. **The report's numbers come from this exact code.** Simplifying anything that *computes* a
   number (evaluate, metrics, training) means **re-running the experiments** and **updating
   the report**. A full rerun is roughly 1–1.5 hours on your laptop:
   - 6 main runs at ~4–6.5 min each;
   - 6 ablation runs;
   - about 1–2 min of evaluation per run.

   Changes that touch only *structure* (configs, utils, analysis styling, tests, scripts,
   demo, report build) don't change any number.
2. **Simpler code also helps you in the Q&A.** The guidelines grade *understanding* (§7, §11).
   Code you can explain line by line is worth more than code that is more general than the
   project needs. Most of the cuts below remove **generality** (other datasets, options,
   caching, styling), not **science**.

Do it on a new branch (e.g. `simplify`) so `main` stays safe.

## C2. Keep these: they are the science and they earn the marks

| Keep | Why |
|---|---|
| `ConvDecoder` / `ConvEncoder` shared by both models | the whole fairness argument |
| `vae_loss` with **sum** over pixels + its comment | the #1 Q&A question |
| The evaluation classifier | needed for mode coverage, precision/recall and the demo filter |
| Precision / recall (classifier space) | H2, the key finding, depends on it |
| Class histogram (`mode_coverage`) | shows *which* classes the VAE loses |
| Laplacian variance + HF ratio, with real reference values | classical CV from the course; H1 |
| Successes and failures figure | explicitly required by the guidelines (§4) |
| β ablation | demonstrates the blur mechanism (more KL → more blur) |
| A handful of known-answer tests | evidence that the metrics measure what you claim |

## C3. What to cut or simplify, file by file

| Part | Now | Suggested | What to remove / change | Changes report numbers? |
|---|---|---|---|---|
| `src/analysis.py` | 413 | ~120 | Drop the custom style (`_style`, palette constants), the `qualitative_*` montages (put the eval PNGs in the report directly), the multi-dataset loop and the generic ablation plotting. Keep: load JSONs → `groupby().agg(['mean','std'])` → `df.to_markdown()` → 3 plots (P/R scatter, class-histogram bars, sharpness bars). A Jupyter notebook is also fine | No (same values) |
| `demo/` (6 files) | 857 | ~180 in **one file** | See C5 | No |
| `src/evaluate.py` | 230 | ~100 | One function instead of the `Evaluator` class. **Classifier feature space only** (drop `InceptionFeatures`, its cache, and the `torchmetrics`/`torch-fidelity` dependencies). Drop `--quick`/`--all-epochs` (use `train_log.csv` for training curves). Drop the memorisation check. Keep samples, most/least confident and interpolation figures | **Yes**: FID-Inception, NN ratio and the learning curve disappear from the report |
| `src/metrics.py` | 198 | ~110 | Keep `frechet_distance`, `precision_recall`, `mode_coverage`, `laplacian_variance`, `high_frequency_ratio`, `psnr`. Drop `nearest_neighbors`. Optionally drop `slerp` + `path_smoothness` (H3 then becomes a *visual* argument from the interpolation figure; guidelines §4 accept systematic qualitative evaluation) | Only if you drop metrics |
| `src/config.py` + 4 YAML files | 45 + ~40 | ~15 | Replace with one `config.py` dict, e.g. `SHARED = dict(latent_dim=32, batch_size=128, epochs=30)`, `VAE = dict(SHARED, lr=1e-3, beta=1.0)`, `GAN = dict(SHARED, lr=2e-4, beta1=0.5)`, plus `argparse` flags for `--seed` and `--beta`. Fairness is still visible: both dicts start from `SHARED` | No |
| `src/models/` (5 files) | 238 | ~130 in **one `models.py`** | Merge the files. Replace the checkpoint metadata (params count, train time, stored config) with `torch.save(model.state_dict(), path)` | No |
| `train_vae.py` + `train_gan.py` | 187 | ~110 in **one `train.py`** | One script with `--model vae|gan`, a shared loop, and two small `vae_step()` / `gan_step()` functions. Shows the "only the objective differs" point even more clearly | No, if the order of operations stays the same (otherwise tiny RNG differences: rerun to be safe) |
| `src/data.py` | 84 | ~25 | Fashion-MNIST only (drop MNIST/CIFAR-10, `subset`, `dataset_info`). Optionally drop the `load_split` fast path and evaluate through a `DataLoader` | No |
| `src/utils.py` | 93 | ~25 | Keep `set_seed`, `get_device`, `save_grid`. Replace `run_name`/`prepare_run_dir`/`append_csv`/`save_json` with f-strings, `os.makedirs`, `csv`/`json` one-liners | No |
| `src/real_baseline.py` | 72 | ~10 | Fold into `evaluate.py` as a `--real` flag that uses training images instead of generated ones | No |
| `src/train_classifier.py` | 65 | ~45 | Inline the accuracy helper, drop config plumbing | No |
| `scripts/` (3 files) | 80 | ~15 | One `run_all.sh` with a `for seed in 0 1 2` loop; drop the smoke test, or keep it as `--epochs 1` | No |
| `tests/` (7 files) | 564 | ~100 in **one file** | Keep: sum-vs-mean loss test, KL closed form, decoders identical, FID = 0 for identical sets, P/R detects mode collapse, blur lowers sharpness. Drop `test_pipeline.py` (188 lines, end-to-end on a toy dataset), `test_analysis.py`, `test_config.py`, `test_data.py` | No |
| Latent-dim ablation | 4 runs | 0 | The ranking never changed, so it adds little. Keep the β ablation | **Yes**: drop report Table 4 |
| Report build (`build.sh`, `report.css`, HTML/DOCX) | 22 + css | 0 | Write the report in Overleaf (LaTeX) or Google Docs, the usual student workflow. Keep `report.md` only as a draft | No |

**Totals:** about **3,200 lines → about 1,000 lines**. The scientific content stays the same:
the same hypotheses and the same main result.

## C4. A minimal layout after simplification

```
image-generation-comparison-comp-vision/
├── config.py            ~15   SHARED / VAE / GAN dicts
├── data.py              ~25   Fashion-MNIST, pad to 32×32, [-1, 1]
├── models.py           ~130   ConvDecoder, ConvEncoder, VAE, GAN, EvalClassifier
├── losses.py            ~30   vae_loss (sum!), D and G losses
├── train.py            ~110   --model vae|gan --seed N [--beta B]
├── train_classifier.py  ~45
├── metrics.py          ~110   FID, precision/recall, mode coverage, Laplacian, HF ratio, PSNR
├── evaluate.py         ~100   metrics.json + 3 figures (+ --real for the baseline)
├── analysis.ipynb       ~10 cells   tables + 3 figures
├── demo.py             ~180   terminal menu + Tkinter window
├── test_core.py        ~100
├── run_all.sh           ~15
├── requirements.txt           torch, torchvision, numpy, scipy, matplotlib, pandas, pillow
└── README.md
```

This is a typical size for a 2–3 person course project and can be explained file by file in
a 10-minute Q&A.

## C5. What a simplified demo would look like (one file, ~180 lines)

Keep the same idea (menu → classifier filter → two models side by side → stats → if/else
text) and drop the extras:

| Keep | Drop |
|---|---|
| menu `input()` loop | `--no-gui`, `--pool`, `--vae/--gan`, `--device` options (hard-code them) |
| generate 5,000 → classify → keep class | equal-n precision/recall per class, "spread" metric |
| 8 images per model + 8 real images | the "best case" image, Save button, New samples button, keyboard shortcuts |
| 3–4 stats: share, confidence, sharpness, HF ratio (+ real) | the `Comparison`/`ModelResult` dataclasses, the separate rules module |
| 4–5 `if` sentences (sharper? more recognisable? over- or under-produced?) | 8-paragraph explanation, tie margins as constants, caveats paragraph |
| a `tk.Label` grid + one `tk.Text` | `Treeview` table with colours, scrollbar, fonts, bring-to-front code |
| – | `test_demo.py` |

Skeleton (a suggestion to read, not code to paste blindly):

```python
# demo.py  (sketch)
def load_everything():            # VAE, GAN, classifier from runs/, test set
def images_of_class(model, c):    # generate 5000 -> classifier argmax == c -> return images, share, confidence
def stats(images):                # laplacian_variance(...).mean(), high_frequency_ratio(...).mean()
def compare_text(vae, gan, real): # 4-5 if/elif/else sentences
def show_window(...):             # tk.Tk(); 3 rows of ImageTk images; tk.Text with the sentences; mainloop()
while True:                       # print menu, input(), call the functions above
```

## C6. Doing it safely, step by step

1. `git switch -c simplify main`
2. Make the **structure-only** cuts first (config, utils, models merge, train merge, data,
   tests, scripts, analysis, demo). Rerun `pytest` (your trimmed tests) and one short
   training run (`--epochs 1`) to check that nothing broke.
3. Decide as a team which **metric** cuts to make (Inception FID, NN ratio, learning curves,
   latent-dim ablation, interpolation CV). Then rerun the experiments with `run_all.sh`
   (~1–1.5 h).
4. Regenerate `results/`, update the numbers and remove the dropped tables and figures from
   the report. Rebuild it and check that it is still 4–6 pages.
5. Rehearse the demo and Q&A against the **new** code. Every member should be able to walk
   through `train.py` and `losses.py` without notes.

> If time is short before Dec 2–11, the safest high-value option is to change **nothing** in
> `src/` (the report's numbers stay valid) and spend the time on Part A, i.e. understanding
> the code as it is. The guidelines grade understanding, not line count.
