# Interactive demo: VAE vs DCGAN, one clothing category at a time

Pick a category in the terminal (for example `10` = Ankle boot). A window then shows images
of that category from **both trained models** side by side, a statistics table, and a
plain-English comparison that you can read out during the presentation.

Nothing is trained. The demo loads the final models that are already committed in `runs/`,
so it starts in a few seconds on any laptop (no GPU needed).

## 1. Run it

From the repository root, in the project's virtual environment (setup: see the main
[README](../README.md#setup); the demo needs no extra packages):

```bash
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python demo/compare_demo.py
```

```
Loading the trained VAE, DCGAN and evaluation classifier ...
Ready (device: mps, dataset: fashion_mnist).

Which category should both models generate?
   1. T-shirt/top
   2. Trouser
   ...
  10. Ankle boot
   0. Quit
Type a number (0-10): 10
```

The window opens. **Close it (or press Esc) to return to the menu** and pick another
category. Type `0` to quit. On macOS, click the terminal again before typing the next number.

| Option | Meaning |
|---|---|
| `--category 10` | skip the menu, show one category, then exit |
| `--no-gui` | print the table and the full explanation in the terminal instead (works without Tkinter) |
| `--seed 3` | different random images (default `0`; same seed → same images every time) |
| `--pool 20000` | images generated per model before filtering (default 10,000) |
| `--device cpu` | force the CPU if the GPU / MPS gives trouble |
| `--vae PATH --gan PATH` | compare other checkpoints, e.g. the seed 1 runs or the β = 4 ablation |

## 2. What the window shows

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Category 10: Ankle boot                                                      │
│ Each model generated 10,000 images ... kept 1,784 VAE and 950 DCGAN ...       │
│ ┌ Real test images of this class (reference) ───────────────────────────────┐ │
│ │ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣ ▣                                                  │ │
│ ├ VAE (beta=1, z=32, epoch 30) ─────┬ DCGAN (z=32, epoch 30) ───────────────┤ │
│ │ ┌──────┐  ▣ ▣ ▣ ▣                 │ ┌──────┐  ▣ ▣ ▣ ▣                     │ │
│ │ │ best │  ▣ ▣ ▣ ▣  first 8        │ │ best │  ▣ ▣ ▣ ▣  first 8            │ │
│ │ └──────┘                          │ └──────┘                              │ │
│ ├ Statistics ───────────────────────┬ What the numbers say ─────────────────┤ │
│ │ Metric        VAE  DCGAN Real Win │ Sharpness                             │ │
│ │ ...                               │ DCGAN images are 5.0x sharper ...     │ │
│ └───────────────────────────────────┴───────────────────────────────────────┘ │
│ [New samples (N)] [Save results (S)] [Close, pick another category (Esc)]    │
└──────────────────────────────────────────────────────────────────────────────┘
```

- **Best**: the image the classifier is most sure about, i.e. each model's best case.
- **First 8**: the first images generated, i.e. the typical case. They are *not*
  cherry-picked. Show both: even the best VAE image is blurry.
- Images are enlarged with nearest-neighbour scaling, so you see the real pixels. Smooth
  scaling would blur the DCGAN images and hide the difference we want to show.
- **New samples (N)** repeats the comparison with the next seed. Use it to show that the
  conclusion does not depend on one lucky draw.
- **Save results (S)** writes the images, `stats.json` and `comparison.txt` to
  `demo/outputs/<category>_seed<k>/` (git-ignored). Handy for slides.

## 3. How it works (and how to explain it)

The VAE and the DCGAN are **unconditional**: they map random noise to "some clothing item",
and you cannot ask them for a boot. So for each model the demo:

1. generates a pool of 10,000 images from random noise,
2. labels every image with the **evaluation classifier** (`runs/classifier_fashion_mnist`,
   90.7 % test accuracy, trained on real images only),
3. keeps the images labelled as the chosen category,
4. compares the two models on the **same number** n of kept images (at most 500).
   Precision and recall depend on sample size, so equal n keeps the comparison fair.

| Row in the table | What it measures | Who wins | Hypothesis |
|---|---|---|---|
| Share of samples in class | how often the model makes this category (real data: 10 %) | closest to real | H2, mode coverage |
| Classifier confidence | how typical the images look to the classifier | higher | – |
| Sharpness: Laplacian variance | edge strength; blur → low | closest to real | H1 |
| Fine detail: high-freq. energy | share of fine texture in the image spectrum | closest to real | H1 |
| Fidelity: precision | share of generated images that land among real images | higher | H1 |
| Diversity: recall | share of real images that the generated set covers | higher | H2 |
| Within-class spread | how different the images are from each other | closest to real | H2 |

"Closest to real" rather than "higher": an image with *more* edge energy than a real photo
is noisy, not better. All metrics come from [src/metrics.py](../src/metrics.py), the same
code that produced the report's numbers. Precision, recall and spread use the classifier's
feature space, as in the report.

The sentences under "What the numbers say" come from simple if/else rules in
[compare_rules.py](compare_rules.py). Every threshold is a named constant at the top of the
file, e.g. `TIE_POINTS = 0.05` (percentages closer than 5 points are a tie).

## 4. Suggested demo script (about 3 minutes)

These numbers are from the default seed 0, and every claim below was checked at seeds 0, 1 and 2.

1. **Ankle boot (10)**: the clearest picture of H1. DCGAN boots are as sharp as real ones
   (103 % of real edge strength), VAE boots are blurry (21 %). The text explains why: the
   VAE's pixel-wise loss is lowest when it predicts the *average* of plausible images.
   Press **N** here: the numbers move by a point or two, and the conclusion does not change.
2. **Bag (9)**: the class where the VAE does best. At seed 0 the window reports a
   *trade-off*: the VAE has higher precision (89 % vs 81 %) while DCGAN covers far more
   variety (recall 88 % vs 55 %), and the VAE's bags look alike (79 % of the real spread).
   Be careful: at seeds 1 and 2 the precision is a tie, so say "the VAE matches DCGAN's
   fidelity on bags", not "the VAE beats it". DCGAN's recall lead holds at every seed.
   Don't press N on this one during the demo.
3. **Coat (5)**: mode coverage. The VAE makes coats for only 2.3 % of its samples (real:
   10 %), and its recall is 32 % against DCGAN's 76 %.
4. Close with the project-level result printed under "Verdict": H1 confirmed, H2 refuted
   (DCGAN is also more diverse), H3 confirmed (VAE has the smoother latent space; not
   shown in this demo).

## 5. Files

| File | Role |
|---|---|
| [compare_demo.py](compare_demo.py) | entry point: terminal menu, options, terminal printout |
| [engine.py](engine.py) | loads the models and the classifier, generates and filters images, computes the stats (no GUI code) |
| [compare_rules.py](compare_rules.py) | if/else rules that turn the stats into the table "Winner" column and the sentences |
| [gui.py](gui.py) | the Tkinter window |
| [test_demo.py](test_demo.py) | tests: `python -m pytest demo -q` |

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `Tkinter is not available` | Tkinter ships with Python, but some installs leave it out. macOS + Homebrew: `brew install python-tk@3.12` (match your Python version). Ubuntu/Debian: `sudo apt install python3-tk`. Windows: re-run the python.org installer and tick "tcl/tk and IDLE". Or use `--no-gui`. Check with `python -c "import tkinter; tkinter._test()"`. |
| `Checkpoint not found` | `git pull`: the final checkpoints are committed in `runs/`. Run the command from the repository root or pass `--vae` / `--gan`. |
| First run is slow or fails offline | Fashion-MNIST (~30 MB) downloads to `data/` on the first run and needs internet once. |
| The window is hidden | It may open behind the terminal or VS Code: check the Dock / taskbar for a Python icon. |
| Errors on the GPU / MPS | `--device cpu` (the demo is fast on a CPU). |
| `ModuleNotFoundError: torch` | The virtual environment is not active: `source .venv/bin/activate`. |
