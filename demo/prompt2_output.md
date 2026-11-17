# VAE vs DCGAN: the whole project, told as a demo

A study guide for presenting the COMP 4102 project live. Read it top to bottom once, then
use the **cheat sheet (§9)** and the **demo script (§7.6)** on the day.

All numbers come from the repository: `results/main_results.md`, `report/report.md`, and the
Tkinter demo at its default seed 0. The demo results in §7 were checked at seeds 0, 1 and 2.

> **Course rule to keep in mind** (Guidelines §11): you "should be able to clearly understand,
> explain, justify, and demonstrate the work you submit". §5 says some groups get a brief live
> demo and/or Q&A between **Dec 2 and 11**. This guide prepares both.

---

## Contents

1. [The 60-second pitch](#1-the-60-second-pitch)
2. [Why this project](#2-why-this-project)
3. [VAE vs GAN in plain words](#3-vae-vs-gan-in-plain-words)
4. [Research question, hypotheses, verdicts](#4-research-question-hypotheses-verdicts)
5. [What we built, A→Z, in the order the code is used](#5-what-we-built-az-in-the-order-the-code-is-used)
6. [The results on one page](#6-the-results-on-one-page)
7. [Demoing with the Tkinter window](#7-demoing-with-the-tkinter-window)
8. [Q&A preparation](#8-qa-preparation)
9. [Cheat sheet](#9-cheat-sheet)

---

## 1. The 60-second pitch

Learn this almost word for word. Everything else in this guide expands one sentence of it.

> "People often say VAEs make **blurry but varied** images, while GANs make **sharp images that
> may miss parts of the data**, which is called mode collapse. We tested that claim fairly. We
> trained a VAE and a DCGAN on Fashion-MNIST with the **exact same image-generating network**,
> latent size, data and training budget, so the only real difference is the **training
> objective**.
>
> We measured **fidelity** (does each image look real?) and **diversity** (do the images cover
> the whole dataset?) with FID, precision/recall, classical sharpness measures, and our own
> evaluation classifier, over 3 seeds plus two ablations.
>
> Result: the GAN is sharper, as expected. Surprisingly, it is **also more diverse**, and it
> never mode-collapsed. The VAE's blur costs it diversity too. The VAE keeps one advantage:
> a **smoother latent space**. Let me show you this live, one clothing category at a time."

---

## 2. Why this project

These reasons are grounded in the course guidelines and the report. Add your own personal
reason too (e.g. "I wanted to understand how image generators actually work"). Examiners like
hearing a real motivation.

| Reason | What to say |
|---|---|
| **Fits the brief exactly** | The guidelines list "compare two or more computer vision methods on a dataset" as an acceptable project (§2), and the marking scheme rewards *analysis and understanding*, not novelty. |
| **A famous claim we could actually test** | "VAEs blur, GANs mode-collapse" is textbook folklore. We turned it into three testable hypotheses (H1–H3) with numbers. |
| **A gap in how it is usually compared** | Published comparisons usually give each model its **own tuned network**, so you can't tell whether a difference comes from the *objective* or the *architecture*. We fixed the architecture, so any difference must come from the objective. |
| **It is a vision problem twice** | Generating images is vision. *Evaluating* them is vision too: a generated image has no ground-truth target, so you have to compare whole **distributions** of images. |
| **Uses course material** | The sharpness metrics are classical CV: the **Laplacian** (edge detection lecture) and the **Fourier transform** (frequency lecture). Both models are CNNs. |
| **Manageable scope** | Fashion-MNIST is small (32×32 grayscale). One training run takes 4–6.5 minutes on a laptop GPU, so we could afford 3 seeds, ablations, and a 60-epoch check. Guidelines §12: "a smaller project completed and evaluated carefully is better". |
| **Matters in practice** | Synthetic training data is useless if it is unrealistic (low fidelity) and harmful if it only covers some classes (low diversity). |

---

## 3. VAE vs GAN in plain words

### 3.1 What both models do

Both are **generators**. They take a vector **z** of 32 random numbers drawn from a standard
normal distribution N(0, I) and turn it into a 32×32 image. Different z gives a different
image. In our project the network that does this is **literally the same class** in both
models (`ConvDecoder`). Only the way it is *trained* differs.

### 3.2 VAE: Variational Autoencoder (Kingma & Welling, 2014)

```
 real image x ─► Encoder ─► μ, log σ² ─► z = μ + σ·ε ─► Decoder ─► reconstruction x̂
                                │              (ε ~ N(0,I))                │
                    KL( q(z|x) ‖ N(0,I) )                    ‖x − x̂‖² summed over pixels
                    "keep codes near N(0,I)"                 "reproduce the input"
```

**Analogy:** a compress-and-decompress machine. The **encoder** squeezes an image into a few
numbers, and the **decoder** rebuilds the image from them. The **KL term** forces all the codes
to look like the same standard Gaussian. That is what lets you later feed the decoder *random*
z and get a new image.

**Loss** (`vae_loss` in `src/losses.py`):

```
loss = ‖x − x̂‖²  (summed over the 1,024 pixels)  +  β · KL(q(z|x) ‖ N(0, I))
       └─ reconstruction: look like the input      └─ regulariser: codes stay Gaussian
```

- This is the negative **ELBO** (evidence lower bound) for a Gaussian decoder with fixed variance.
- **Reparameterization trick:** `z = μ + σ·ε`. The randomness lives in ε, so gradients can still
  flow back through μ and σ to the encoder.
- **β** weights the KL term. β = 1 is the standard VAE; β ≠ 1 is a β-VAE (we tested 0.5 and 4).

**Why VAEs are blurry.** For a given z, the output that minimises squared error is the
**average** of all images that could have produced that z. The KL term pushes all codes
toward the same Gaussian, so the codes of different images overlap. The decoder ends up
outputting an average of several plausible images (e.g. shirts with different textures),
and **averaging acts as a low-pass filter, i.e. blur**.

### 3.3 GAN: Generative Adversarial Network (Goodfellow et al., 2014), DCGAN recipe (Radford et al., 2016)

```
 z ~ N(0,I) ─► Generator G ─► fake image ──┐
                                           ├─► Discriminator D ─► P("this is real")
 real image x ─────────────────────────────┘

 D learns: real → 1, fake → 0          G learns: make D answer 1 on fakes
```

**Analogy:** a counterfeiter (G) and a detective (D). The detective learns to spot fakes, and
the counterfeiter learns to fool the detective. Each improves because the other does.

**Losses** (`src/losses.py`):
- D: binary cross-entropy (real → 1, fake → 0).
- G: **non-saturating** loss `−log D(G(z))`. It has the same goal as the original minimax game
  but gives strong gradients early in training, when D easily rejects the fakes.
- DCGAN recipe: strided convolutions, batch norm, ReLU/LeakyReLU, N(0, 0.02) initialisation,
  Adam lr 2e-4 with β1 = 0.5. Each step does one D update and then one G update.

**Why GANs are sharp.** D is itself a CNN, and blur is trivially easy for a CNN to detect. So G
is pushed hard toward crisp edges and texture.

**Why GANs *can* mode-collapse.** G is punished only for images that *look fake*, never for
the parts of the data it *never produces*. Mapping many z values to a few convincing images is
a valid shortcut. In our runs it **did not happen** (§4).

### 3.4 Side by side

| | VAE | GAN (DCGAN) |
|---|---|---|
| Training signal | reconstruct the input + keep codes Gaussian | fool a learned discriminator |
| Networks | encoder + decoder | discriminator + generator |
| Has an encoder (image → z)? | **yes** (can reconstruct) | no |
| Typical images | blurry but varied | sharp, may miss modes |
| Typical failure | blurry blends of two classes | sharp but wrong (texture artefacts, in-between shapes) |
| Latent space | smooth (KL makes neighbouring codes overlap) | less constrained between codes |
| Training | stable (one loss going down) | a two-player game; can be unstable |
| Our training time (30 epochs) | ~4 min | ~6.5 min (two forward/backward passes per step) |

---

## 4. Research question, hypotheses, verdicts

> **With the same image-producing network, latent size, data and training budget, how do a VAE
> and a DCGAN trade off fidelity/sharpness against diversity/mode coverage?**

| | Hypothesis | Main evidence | Verdict |
|---|---|---|---|
| **H1** | The GAN is sharper and has a lower FID | FID, precision, Laplacian variance, HF ratio | ✅ **Supported.** DCGAN sharpness ≈ real (0.146 vs 0.150); VAE 0.038 |
| **H2** | The VAE covers the data better (higher recall, no mode collapse) and trains more stably | recall, class histogram, std over seeds, D(x)/D(G(z)) | ❌ **Not supported.** DCGAN recall 0.85 vs VAE 0.49, all 10 classes covered, D never overpowered G |
| **H3** | The VAE has a smoother latent space | interpolation step CV | ✅ **Supported, with a caveat.** CV 0.51 (VAE) vs 0.69 (GAN), but a larger β made it *less* smooth |

**Say:** "A hypothesis that turns out false is still a result if you can explain why." H2 is our
most interesting finding. The explanation is in §6.3.

---

## 5. What we built, A→Z, in the order the code is used

### 5.1 The pipeline

```
 Step 1-2  Fashion-MNIST ─► src/data.py ─► configs/shared.yaml (+ vae.yaml, gan.yaml)
                                                 │
            ┌────────────────────────────────────┼─────────────────────────┐
            ▼                                    ▼                         ▼
 Step 3  train_classifier.py          Step 4-6  train_vae.py         train_gan.py
         (measuring instrument)                 (ELBO loss)          (adversarial loss)
            │                                    └────────────┬────────────┘
            │                                                 ▼   runs/<run>/checkpoints/
            ├──► Step 7  real_baseline.py          Step 8  evaluate.py + metrics.py
            │            (best possible score)               │   runs/<run>/eval/
            │                  └────────────────┬────────────┘
            │                                   ▼
            │                     Step 9  analysis.py ─► results/ ─► Step 11 report/
            │
            └──────────────────► Step 12  demo/  (classifier + final checkpoints, live)

 Step 10 (around everything): scripts/*.sh run it all; tests/ check every piece (49 tests + 8 demo tests)
```

Every step follows the same pattern: **what it does → why we built it that way → command → output**.

### Step 0: Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch, torchvision, torchmetrics, scipy, ...
```

The code picks the device automatically: CUDA, Apple MPS, or CPU.

### Step 1: Data (`src/data.py`)

- **What:** Fashion-MNIST, 60,000 training and 10,000 test grayscale 28×28 images in **10 balanced
  classes** (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
- **Pad to 32×32, don't resize.** The network doubles the image size at every layer
  (4→8→16→32). Resizing would *interpolate*, which blurs the **real** images. The real images
  are the reference for our sharpness metric, so blurring them would unfairly help the blurry model.
- **Scale to [−1, 1]** to match the `tanh` output of the decoder/generator.
- The models train on the **training** split only. The **test** split is the reference for every metric.

### Step 2: Configuration (`src/config.py`, `configs/`)

- `configs/shared.yaml` holds everything both models must share: dataset, latent size **32**,
  network width, batch size **128**, **30 epochs**, seed.
- `vae.yaml` and `gan.yaml` start with `base: shared.yaml` and add **only** their optimiser
  settings (VAE: Adam 1e-3, β = 1; DCGAN: Adam 2e-4, β1 = 0.5, each from its own paper).
- You can override anything from the command line with `--set seed=1 latent_dim=8`. A typo in
  the key is an error, so it can't be silently ignored.
- **Fairness by construction:** `tests/test_config.py` fails if a model config ever overrides a
  shared setting.

### Step 3: The evaluation classifier (`src/train_classifier.py`, `src/models/classifier.py`)

```bash
python -m src.train_classifier --config configs/classifier.yaml     # 8 epochs → 90.7 % test accuracy
```

- A small CNN trained on **real** images. It is **not** one of the compared methods. It is a
  *measuring instrument*, used in two ways:
  1. **As a labeller.** It labels every generated image, which gives the class histogram and
     shows mode collapse (and powers the demo's category filter).
  2. **As a feature space.** Its 128-d penultimate layer is used for FID and precision/recall.
     The standard Inception network was trained on ImageNet photos, not 32×32 grayscale clothing.
- It is trained once and committed (`runs/classifier_fashion_mnist/classifier.pt`), so **every
  evaluation uses the same instrument**.

### Step 4: The shared architecture (`src/models/networks.py`, `vae.py`, `gan.py`)

| Class | Shape | Used as |
|---|---|---|
| `ConvDecoder` | z → 4×4 → 8×8 → 16×16 → 32×32 (transposed convs, 256/128/64 channels, BatchNorm + ReLU, tanh). **788,353 parameters** | VAE **decoder** *and* GAN **generator** |
| `ConvEncoder` | 32×32 → 16 → 8 → 4 (strided convs, LeakyReLU) + output head | VAE **encoder** (head: 2×32 numbers = μ, log σ²) *and* GAN **discriminator** (head: 1 logit) |

- Both models get the same DCGAN weight initialisation, so neither gets a head start.
- `tests/test_models.py` checks that the two image networks are identical.
- **This is the key fairness point.** Because the image producer is literally the same class,
  any difference in the results comes from the **training objective**.

### Step 5: The losses (`src/losses.py`, 53 lines)

- `vae_loss`: squared error **summed** over pixels + β·KL (closed form).
- `discriminator_loss`, `generator_loss`: binary cross-entropy, non-saturating G loss.
- **The design choice most likely to come up in Q&A: sum, not mean.** A 32×32 image gives
  1,024 squared-error terms, while the KL has only 32. Averaging over pixels would make the KL
  about **1000× stronger**, a hidden β ≈ 1000, and much blurrier samples. With the sum, the
  loss is the true negative ELBO and β is the *only* knob. `tests/test_losses.py` checks this.

### Step 6: Training (`src/train_vae.py`, `src/train_gan.py`)

```bash
python -m src.train_vae --config configs/vae.yaml --set seed=1
python -m src.train_gan --config configs/gan.yaml --set seed=1
```

- 30 epochs × 468 mini-batches of 128 = **14,040 updates** per model.
- VAE step: encode → reparameterize → decode → `vae_loss` → one Adam step.
- GAN step: D step (real → 1, `fake.detach()` → 0), then G step (fool the just-updated D).
- Each run writes `runs/<run>/`:
  - `config.yaml`: the exact settings.
  - `train_log.csv`: losses per epoch. For the GAN it also logs D(x) and D(G(z)), which are
    stability signals.
  - `samples/epoch_001…030.png`: the **same 64 latent vectors** decoded after every epoch, so
    you can watch the model learn.
  - `checkpoints/epoch_030.pt`: the final model (committed).

### Step 7: Real-vs-real baseline (`src/real_baseline.py`)

```bash
python -m src.real_baseline
```

- **Problem:** these metrics have no absolute scale. Is an FID of 5 good?
- **Solution:** run the *same* metric code on 10,000 real **training** images instead of
  generated ones, compared against the test images. That is the best score a perfect generator
  could get. Classifier space: FID 0.03, precision/recall 0.90/0.89. Inception space: FID 2.4,
  P/R 0.78/0.78. This becomes the "Real data" row of every table.

### Step 8: Evaluation (`src/evaluate.py` + `src/metrics.py`)

```bash
python -m src.evaluate --config configs/gan.yaml --set seed=1                       # full
python -m src.evaluate --config configs/gan.yaml --set seed=1 --all-epochs --quick  # learning curve
```

It loads a checkpoint, generates **10,000 images from fixed latent vectors** (the same draws for
both models), and compares them with the 10,000 test images:

| Measures | Metric | Better | Hypothesis |
|---|---|---|---|
| quality + diversity | **FID**: distance between Gaussians fitted to real and fake features (classifier *and* Inception space) | ↓ | H1, H2 |
| fidelity | **Precision**: share of fakes inside the real "manifold" (k = 3 nearest-neighbour balls) | ↑ | H1 |
| diversity | **Recall**: share of reals inside the fake manifold | ↑ | H2 |
| mode collapse | classes covered (≥ 1 % of samples), **class KL** vs the real class distribution | ↑ / ↓ | H2 |
| sharpness | **Laplacian variance** (edge strength) and **HF ratio** (FFT energy above 0.25 cycles/px) | ≈ real | H1 |
| latent smoothness | **interpolation step CV**: evenness of steps along 100 slerp paths of 10 frames | ↓ | H3 |
| reconstruction | **PSNR** (VAE only; the GAN has no encoder) | ↑ | – |
| memorisation | **NN ratio**: distance to the nearest training image, generated vs test images | ≈ 1 | – |

It also saves qualitative figures: `samples.png` (random, not cherry-picked),
`most/least_confident.png` (successes and failures), `interpolation.png`,
`nearest_neighbors.png`, and `reconstructions.png` (VAE only).

**Why "≈ real" for sharpness?** An image with *more* edge or high-frequency energy than a real
photo is **noisy**, not better. So we reward matching the real value, not maximising it.

### Step 9: Aggregation (`src/analysis.py`)

```bash
python -m src.analysis        # rebuilds all of results/ in a few seconds
```

It reads every `metrics.json`, groups runs that differ only by seed, and writes mean ± std
tables and figures: `main_results.md`, `ablation_*.md`, `precision_recall.png`,
`mode_coverage.png`, `sharpness.png`, `learning_curve.png`, and `qualitative_*.png`.

### Step 10: Automation and tests (`scripts/`, `tests/`)

```bash
bash scripts/smoke_test.sh       # ~2 min: tests + tiny runs, proves the pipeline works end to end
bash scripts/run_main.sh         # E1 + E2: both models, seeds 0 1 2, full + learning-curve evaluation
bash scripts/run_ablation.sh     # E3 + E4: latent size 8/128, VAE β 0.5/4
python -m pytest                 # 49 tests
```

The tests check inputs with a **known answer**: identical sets give FID = 0, a generator
covering half the data gets recall ≈ 0.5, and blurring lowers both sharpness measures. **Say:**
"We tested the measuring tools before trusting their measurements."

### Step 11: The report (`report/report.md` → `bash report/build.sh` → `report.pdf`)

Six pages following the guidelines' structure: problem → method → implementation → experiments
→ analysis → conclusion. Every number comes from `results/`.

### Step 12: The interactive demo (`demo/`)

```bash
python demo/compare_demo.py
```

This is the subject of §7. It **trains nothing**: it loads the committed final checkpoints and
the evaluation classifier, and uses the same `src/metrics.py` functions as the report.

| File | Role |
|---|---|
| `demo/compare_demo.py` | entry point: terminal menu, options, terminal printout |
| `demo/engine.py` | loads the models, generates and filters images by class, computes the stats |
| `demo/compare_rules.py` | if/else rules that pick the winner of each row and write the sentences |
| `demo/gui.py` | the Tkinter window |
| `demo/test_demo.py` | 8 tests (`python -m pytest demo -q`) |

### 5.2 The experiments we ran

| Exp. | Question | Result in one line |
|---|---|---|
| **E1** main | VAE vs DCGAN, same settings, 3 seeds | DCGAN wins fidelity **and** diversity; VAE has the smoother latent space |
| **E2** learning curves | Converged? Stable? | DCGAN FID-clf 11.3 → 2.6 by epoch 15, then 1.5 at 30; VAE still slowly improving at 30; D never overpowers G |
| **E3** latent size 8/32/128 | Does z size matter? | The ranking never changes; z = 8 lowers VAE recall (0.40) |
| **E4** β = 0.5/1/4 (VAE) | What does β trade off? | Larger β → blurrier, precision ↑ slightly, recall collapses (0.61 → 0.14) |
| Epoch check (60 ep.) | Is 30 epochs unfair to the VAE? | FID-clf 21.1 → 18.4 only (closes 14 % of the gap), so we kept 30 |

---

## 6. The results on one page

### 6.1 Main table (mean over 3 seeds)

| | DCGAN | VAE | Real data |
|---|---|---|---|
| FID, classifier space ↓ | **1.29** | 20.7 | 0.03 |
| FID, Inception space ↓ | **13.3** | 56.9 | 2.4 |
| Precision / recall ↑ | **0.85 / 0.85** | 0.70 / 0.49 | 0.90 / 0.89 |
| Class KL ↓ (all 10 classes covered by both) | **0.022** | 0.22 | 0.005 |
| Sharpness: Laplacian var. (≈ real) | **0.146** | 0.038 | 0.150 |
| HF ratio (≈ real) | **0.100** | 0.034 | 0.104 |
| Interpolation CV ↓ | 0.69 | **0.51** | – |
| NN ratio (≈ 1 = no copying) | 1.07 | 1.00 | – |

### 6.2 H1, why the GAN is sharper

DCGAN edge statistics match the real images **without exceeding them** (0.146 vs 0.150), so
its sharpness is real detail, not noise. The VAE keeps about ¼ of the real edge energy. The
mechanism: squared error makes the VAE output a *conditional mean*, which erases detail, while
a CNN discriminator spots missing high frequencies immediately. **E4 confirms it:** a larger β
means more averaging and lower sharpness (0.041 → 0.028).

### 6.3 H2, why the GAN is *also* more diverse (the surprise)

- **No mode collapse.** Every DCGAN class gets 6–13 % of samples (ideal: 10 %), and its recall
  is near the real baseline.
- **The VAE covers all 10 classes, but unevenly.** It makes too many **pullovers (25.7 %)** and
  **ankle boots (17.8 %)** and too few **coats (2.3 %)** and **sandals (4.7 %)**. These pairs
  differ mainly in *fine detail*: a coat is a pullover with a zip, and a sandal is an ankle boot
  with gaps between the straps. Once blur removes that detail, the classifier sees the plainer
  class.
- **Blur also explains low recall.** Recall asks whether each real image has a generated image
  nearby in feature space. Real images defined by texture or thin structures have **no blurry
  counterpart nearby**. So in a learned feature space, **blur costs diversity as well as
  fidelity**.
- **Not less stable either:** seed-to-seed std of FID is 3–6 % for both models, and
  D(G(z)) stayed in [0.17, 0.32] in every epoch of every seed.

### 6.4 H3, why the VAE latent space is smoother

The KL term makes the codes of similar images overlap, so nearby z values decode to similar
images (VAE paths morph gradually, e.g. trousers widen into a dress). The GAN loss never
constrains what lies *between* two codes (some DCGAN paths stay still and then jump, e.g.
sneaker → trousers). **Caveat:** a larger β *raised* the CV (0.45 → 0.63), possibly because a
strong KL switches off latent dimensions. That is from one seed, so it remains a hypothesis.

### 6.5 Limitations (say these before you're asked)

- FID-Inception on 32×32 grayscale is far from ImageNet: even real-vs-real P/R is only 0.78
  there. Both feature spaces rank the models the same way, so we lead with the classifier space.
- The classifier is 90.7 % accurate and less confident on blurry images (mean confidence 0.79
  VAE, 0.88 DCGAN, 0.93 real), so class coverage partly measures blur.
- The VAE was still improving slowly at epoch 30. The ablations use one seed. Reconstruction is
  measured for the VAE only.
- One small architecture, one dataset, and paper-default optimisers, so the results may not
  transfer to large models or colour images.
- **Next step:** remove the blur at its source (a learned decoder variance or a perceptual loss)
  and check whether the VAE's recall recovers.

---

## 7. Demoing with the Tkinter window

### 7.1 Before the demo (checklist)

- [ ] `cd image-generation-comparison-comp-vision && source .venv/bin/activate`
- [ ] Run `python demo/compare_demo.py --category 10` once beforehand. The first run downloads
      Fashion-MNIST (~30 MB) and needs internet once.
- [ ] Keep the default **seed 0**. Same seed = same images and numbers as in this guide.
- [ ] Close distracting windows. On macOS the Tkinter window can open *behind* VS Code; look
      for the Python icon in the Dock.
- [ ] Fallback if Tkinter fails: `python demo/compare_demo.py --no-gui` prints the same table and text.

### 7.2 Launch

```bash
python demo/compare_demo.py
```

```
Loading the trained VAE, DCGAN and evaluation classifier ...
Ready (device: mps, dataset: fashion_mnist).

Which category should both models generate?
   1. T-shirt/top   2. Trouser   3. Pullover   4. Dress   5. Coat
   6. Sandal        7. Shirt     8. Sneaker    9. Bag    10. Ankle boot
   0. Quit
Type a number (0-10): 10
```

Close the window (Esc) to return to the menu. On macOS, click the terminal before typing the
next number.

### 7.3 The one idea you MUST explain first: how do we get "ankle boots" from these models?

Both models are **unconditional**: you cannot *ask* them for a boot. They turn random noise into
"some clothing item". So for each model the demo (`demo/engine.py`):

1. generates **10,000** images from random noise,
2. labels every image with the **evaluation classifier** (the same instrument as in the report),
3. keeps the ones labelled as the chosen category,
4. compares both models on the **same number n** of kept images (at most 500). Precision and
   recall depend on sample size, so equal n keeps it fair.

> **Say:** "Our models don't take a class as input, so we let each model generate ten thousand
> images and use our evaluation classifier to pick out the ankle boots. As a bonus, *how often*
> a model makes ankle boots is itself a measurement: real data has 10 % of each class."

### 7.4 Tour of the window

```
┌ Category 10: Ankle boot ─────────────────────────────────────────────────────┐
│ "Each model generated 10,000 images ... kept 1,784 VAE and 950 DCGAN ..."    │  ← the filtering step
│ ┌ Real test images of this class (reference) ────────────────────────────┐  │  ← what "real" looks like
│ ├ VAE (blue) ──────────────────────┬ DCGAN (orange) ──────────────────────┤  │
│ │ [best case, large]  [first 8]    │ [best case, large]  [first 8]        │  │  ← best vs typical
│ ├ Statistics (rows coloured by     ┬ What the numbers say                 ┤  │
│ │ winner: blue VAE, orange DCGAN)  │ 8 short paragraphs                   │  │  ← read these out
│ └──────────────────────────────────┴──────────────────────────────────────┘  │
│ [New samples (N)]  [Save results (S)]  [Close, pick another category (Esc)]  │
└──────────────────────────────────────────────────────────────────────────────┘
```

| Region | What it is | What to say |
|---|---|---|
| **Real strip** | 12 real test images of the class | "This is the target: what real ankle boots look like." |
| **Best case (large)** | the image the classifier is *most sure* about | "Each model's best ankle boot. Even the VAE's best is blurry." |
| **First 8** | the first 8 images generated, **not cherry-picked** | "And this is the *typical* output, not hand-picked." |
| **Statistics** | 7 metrics + winner per row | see §7.5 |
| **What the numbers say** | plain-English text produced by if/else rules (`compare_rules.py`) | read the headings out loud as you go |
| **New samples (N)** | same category, next seed | "The numbers move a little; the conclusion doesn't." |
| **Save results (S)** | writes images + `stats.json` + `comparison.txt` to `demo/outputs/` | handy for slides |

Images are enlarged with **nearest-neighbour** scaling, so you see the real pixels. **Say:**
"Smooth scaling would blur the GAN images and hide exactly the difference we want to show."

### 7.5 How to read each row of the statistics table

| Row | Meaning | Winner rule | How to say it |
|---|---|---|---|
| **Share of samples in class** | % of the 10,000 images labelled as this class (real: 10 %) | closest to real | "How often the model makes this class, a per-class view of mode coverage." |
| **Classifier confidence** | how typical the kept images look to the classifier | higher (tie within 5 points) | "How recognisable they are. It's biased upward because we only kept images already labelled as this class, so compare the models with each other, not with 100 %." |
| **Sharpness: Laplacian var.** | edge strength (2nd-derivative filter) | closest to real (tie within 10 %) | "Classical edge detection. Blurry images have weak edges." |
| **Fine detail: HF energy** | share of Fourier energy at high frequencies | closest to real | "Same story in the frequency domain: blur is a low-pass filter." |
| **Fidelity: precision** | % of generated images that land among real images (feature space) | higher | "Do the fakes look like real ankle boots?" |
| **Diversity: recall** | % of real images covered by the generated set | higher | "Do the fakes cover all the *kinds* of real ankle boots?" |
| **Within-class spread** | average distance between images of the class | closest to real | "Do the samples vary as much as real ones, or do they all look alike?" |

The thresholds are named constants at the top of `demo/compare_rules.py` (`TIE_POINTS = 0.05`,
`TIE_MARGIN = 0.10`, `NEAR_REAL = 0.20`, `LOW_RECALL = 0.25`, `SHARE_LOW/HIGH = 0.6/1.4`). If
asked "why 5 points?": they are simple, transparent conventions for the demo text. The report's
conclusions come from the full evaluation, not from these thresholds.

### 7.6 The demo script (~3 minutes, all numbers at seed 0)

**① Ankle boot (type `10`): H1, the clearest picture.**

| | VAE | DCGAN | Real | Winner |
|---|---|---|---|---|
| Share | 17.8 % | 9.5 % | 10.0 % | DCGAN |
| Confidence | 86.2 % | 92.6 % | 94.0 % | DCGAN |
| Sharpness | 0.0285 | 0.1432 | 0.1386 | DCGAN |
| Fine detail | 0.0247 | 0.0718 | 0.0692 | DCGAN |
| Precision | 62.6 % | 84.4 % | – | DCGAN |
| Recall | 56.6 % | 80.8 % | – | DCGAN |
| Spread | 6.29 | 6.15 | 5.43 | Tie |

> **Say:** "Left is the VAE, right is the DCGAN, and the top row is real ankle boots. The DCGAN
> boots have laces and soles; the VAE boots are soft blobs. The numbers agree: the DCGAN reaches
> **103 %** of real edge strength, the VAE only **21 %**, a **5× difference**. The reason is the
> VAE's loss: squared error is lowest when you predict the **average** of all plausible boots,
> and an average is blurry. The GAN's discriminator rejects blurry images as fake. The DCGAN
> also wins precision *and* recall, so here there's no trade-off at all."
>
> Bonus point: "Notice the VAE makes ankle boots **17.8 %** of the time, almost double the real
> 10 %. When blur erases the straps of a sandal, the classifier sees an ankle boot."

Then press **N**: "New random images, and the numbers barely move" (sharpness 21 → 22 %,
103 → 105 % at seed 1).

**② Bag (type `9`): where the VAE does best. Be careful here.**

| | VAE | DCGAN | Real | Winner |
|---|---|---|---|---|
| Precision | **88.8 %** | 80.8 % | – | VAE |
| Recall | 54.8 % | **88.0 %** | – | DCGAN |
| Spread | 5.31 | 7.52 | 6.69 | Tie |
| Sharpness | 0.0307 | 0.1703 | 0.1718 | DCGAN |

> **Say:** "Bags are the VAE's strongest class. Its precision matches the DCGAN's; at this seed
> it's even slightly higher. That's the classic **fidelity-vs-diversity trade-off**: the VAE
> plays it safe with plain, average-looking bags (spread only **79 %** of real), while the DCGAN
> covers far more of the variety (recall 88 % vs 55 %)."
>
> ⚠️ At seeds 1 and 2 the precision is a **tie**, so say "matches", not "beats", and **don't press
> N on Bag**. DCGAN's recall lead holds at every seed.

**③ Coat (type `5`): mode coverage.**

| | VAE | DCGAN | Real | Winner |
|---|---|---|---|---|
| Share | **2.3 %** | 5.9 % | 10.0 % | DCGAN |
| Precision | 73.2 % | 90.0 % | – | DCGAN |
| Recall | 31.6 % | 75.8 % | – | DCGAN |

> **Say:** "Out of ten thousand VAE images, only **2.3 %** are coats, against 10 % in the real
> data. Why? A coat is basically a pullover with a zip. Blur erases the zip, so the classifier
> calls it a pullover, and the VAE over-produces pullovers (25.7 %). Its coat recall is only
> 32 % against 76 %. This is the surprise of the project: **blur costs diversity too**, so our
> hypothesis H2 was wrong. The DCGAN, which everyone expects to mode-collapse, actually covers
> the data better."
>
> (The window counts both models as under-producing coats, the DCGAN at 5.9 %. Coats are also
> the hardest class for our classifier: only 70 % confidence even on *real* coats.)

**④ Optional, Trouser (type `2`): the VAE's best-looking class.** Its silhouette survives blur.
Confidence and share are ties, and VAE sharpness is 42 % of real here (vs 21 % for boots).
**Say:** "Simple shapes survive averaging; detailed ones don't."

**⑤ Close with the verdict paragraph** at the bottom of "What the numbers say":

> "Across all classes and 3 seeds, the report says: H1 confirmed, the GAN is sharper. H2
> refuted, the GAN is *also* more diverse and never mode-collapsed. H3 confirmed, the VAE has
> the smoother latent space, which this demo doesn't show but our interpolation figures do."

### 7.7 Linking the demo to the report

| What you see in the demo | Where it comes from in the report |
|---|---|
| VAE sharpness ~20–40 % of real per class | Table 3: VAE Laplacian 0.038 vs real 0.150 overall |
| VAE over-produces Pullover/Ankle boot, under-produces Coat/Sandal | §5 H2 paragraph, `results/mode_coverage.png` |
| DCGAN wins recall in every class | Table 3: recall 0.85 vs 0.49, `precision_recall.png` |
| Bag/Trouser: the VAE holds up best | §5 "Successes and failures": the VAE succeeds on trousers, the DCGAN on bags and sandals |

**Consistency check you can mention:** at seed 0 the demo draws **exactly the same 10,000 latent
vectors** as the report's evaluation of the seed-0 models (same `generate()` function, same
seed). So the demo's "Share" row matches the report's class histogram to the decimal: VAE Coat
2.3 %, Ankle boot 17.8 %, DCGAN Coat 5.9 %. **Say:** "The live demo reproduces the report's
numbers. It isn't a separate experiment."

**What the demo adds:** the report shows *averages over all classes*; the demo breaks them down
**per class, live**, so you can see *where* the VAE loses and *why* (fine detail).

### 7.8 What NOT to claim during the demo

- ❌ "The VAE beats the DCGAN on bags." ✅ It **matches** it on precision (it only wins at seed 0).
- ❌ "The DCGAN is 92 % accurate." ✅ Confidence is computed on **already-filtered** images, so it
  is biased upward. Compare models with each other.
- ❌ "These are the project's results." ✅ The demo uses one seed and ≤ 500 images per class.
  The report uses 10,000 samples, all classes, and 3 seeds.
- ❌ "The demo is the project." ✅ It is a **window onto the evaluation**. The guidelines say an
  interface alone is not enough; the evaluation behind it is the work.

---

## 8. Q&A preparation

Short answers to the questions examiners are most likely to ask. The guidelines say **any
member** may be asked, so everyone should know all of these.

**Q1. Why sum the reconstruction error over pixels instead of averaging?**
1,024 pixel terms vs 32 KL terms. Averaging would make the KL ~1000× stronger, a hidden
β ≈ 1000 and much blurrier samples. With the sum it's the true negative ELBO and β is the only
weight. (`src/losses.py`, `tests/test_losses.py`)

**Q2. Why are VAE images blurry?**
Under squared error, the best output for a given z is the *mean* of all plausible images. The
KL term makes their codes overlap, so the decoder averages several images, and averaging is a
low-pass filter.

**Q3. Why didn't the GAN mode-collapse?**
A balanced, simple dataset plus the well-tested DCGAN recipe (non-saturating loss, Adam
2e-4/β1 = 0.5, batch norm). The logs show D never overpowered G: D(G(z)) stayed in
[0.17, 0.32] in every epoch of every seed, so G always received a useful gradient.

**Q4. How do you make sure the comparison is fair?**
The same `ConvDecoder` class is both the VAE decoder and the GAN generator (788k params). One
shared config file holds the data, latent size, batch size, epochs and seeds. Both get the same
initialisation and the same evaluation latent vectors. Unit tests enforce this. Only the
objective and each paper's optimiser settings differ.

**Q5. Why two feature spaces for FID?**
Inception was trained on ImageNet colour photos, while our images are 32×32 grayscale clothing.
Even real-vs-real precision is only 0.78 in Inception space (0.90 in ours). We report both; they
rank the models the same way, and we lead with the classifier space.

**Q6. What exactly are precision and recall here?**
(Kynkäänniemi et al., 2019.) Put a ball around every real sample reaching its 3rd nearest real
neighbour; that union approximates the "real manifold". **Precision** = share of fakes inside it
(fidelity). **Recall** = the same with the roles swapped (diversity). Mode collapse gives high
precision and low recall.

**Q7. Why is "closest to real" the rule for sharpness?**
More edge or high-frequency energy than real images means noise or checkerboard artefacts, not
quality. The DCGAN is within 4 % of real without exceeding it.

**Q8. Your models are unconditional. How does the demo show a specific class?**
Generate 10,000 images, label them with the evaluation classifier, keep the chosen class, and
compare equal-sized sets (§7.3). How often a model produces the class is itself a mode-coverage
measurement.

**Q9. Isn't the classifier biased toward the GAN?**
Partly, and we say so. It is less confident on blurry images (0.79 VAE vs 0.88 DCGAN vs 0.93
real), so class coverage partly measures blur. That is why we also use classifier-free metrics:
Laplacian variance and HF ratio work on raw pixels.

**Q10. Does the GAN just copy training images?**
No. The NN ratio is 1.07: generated images are as far from their nearest training image as
unseen test images are. Copying would give a ratio well below 1. (β = 4 gives 0.80, but that is
because a blurry average image is close in pixels to many training images.)

**Q11. Is 30 epochs unfair to the VAE, which was still improving?**
We checked: 60 epochs moved VAE FID-clf from 21.1 to 18.4, closing only 14 % of the gap to the
DCGAN (1.3). No conclusion changes.

**Q12. What does β do?**
It weights the KL term. Larger β means more averaging: blurrier (0.041 → 0.028), slightly higher
precision (0.64 → 0.72), recall collapses (0.61 → 0.14). It moves the VAE *along* the trade-off
but never toward the DCGAN.

**Q13. What is the reparameterization trick?**
Sampling z ~ N(μ, σ²) is not differentiable. Writing z = μ + σ·ε with ε ~ N(0, I) moves the
randomness into ε, so gradients flow through μ and σ to the encoder.

**Q14. Why the non-saturating generator loss?**
Early on, D rejects fakes easily and log(1 − D(G(z))) saturates (near-zero gradient). Minimising
−log D(G(z)) has the same fixed point but strong gradients exactly when G is bad.

**Q15. What are the main limitations, and what would you do next?**
See §6.5. Next steps: a learned decoder variance or perceptual loss to remove the VAE blur; then
check whether its recall recovers. Optionally WGAN-GP or CIFAR-10 to test whether the result holds
on harder data.

**Q16. Why pad instead of resize to 32×32?**
Resizing interpolates, which blurs the real reference images and would make the blurry model
look closer to real on our sharpness metrics.

**Q17. How do you know your metrics are correct?**
Unit tests with known answers: identical sets give FD = 0, a half-coverage generator gets recall
≈ 0.5, blur lowers sharpness. The real-vs-real baseline shows the best achievable value of each
metric.

---

## 9. Cheat sheet

**Setup:** Fashion-MNIST, 60k train / 10k test, 10 classes, padded 28→32, scaled to [−1, 1].
z = 32, batch 128, 30 epochs (14,040 updates), seeds 0/1/2. Shared `ConvDecoder`, 788,353
params. VAE: Adam 1e-3, β = 1, ~4 min. DCGAN: Adam 2e-4, β1 = 0.5, ~6.5 min. Evaluation
classifier: 90.7 %.

**Headline numbers (DCGAN vs VAE vs real):**
- FID-clf **1.29** vs 20.7 vs 0.03 · FID-Inc **13.3** vs 56.9 vs 2.4
- Precision **0.85** vs 0.70 vs 0.90 · Recall **0.85** vs 0.49 vs 0.89
- Laplacian **0.146** vs 0.038 vs 0.150 · HF **0.100** vs 0.034 vs 0.104
- Interp. CV 0.69 vs **0.51** · NN ratio 1.07 vs 1.00 · Class KL **0.022** vs 0.22

**Verdicts:** H1 ✅ · H2 ❌ (the surprise: blur costs diversity too) · H3 ✅ (with a β caveat)

**VAE class shares:** Pullover 25.7 %, Ankle boot 17.8 % (too many) · Coat 2.3 %, Sandal 4.7 % (too few)

**Demo (seed 0):** Ankle boot sharpness 21 % vs 103 % of real (5×) · Bag precision 89 vs 81
(tie at other seeds!), recall 55 vs 88 · Coat share 2.3 % vs 5.9 %, recall 32 vs 76

**One-sentence takeaways:**
- "Same network, different objective: the objective alone decides sharpness."
- "The GAN didn't collapse; the VAE's blur cost it diversity as well as fidelity."
- "The VAE's advantage is a smooth latent space and an encoder."
- "The demo breaks the report's averages down per class, live."
