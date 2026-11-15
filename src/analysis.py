"""Aggregate every evaluation into report-ready tables and figures.

Usage:
    python -m src.analysis                        # reads runs/, writes results/
    python -m src.analysis --runs runs --out results

Tables (Markdown, mean ± std over seeds) and raw numbers (CSV):
    all_runs.csv              final evaluation of every run (numeric, one row per run)
    main_results.md           VAE vs DCGAN at the default configuration          (H1, H2, H3)
    ablation_latent_dim.md    both models, different latent sizes
    ablation_beta.md          VAE only, different beta
Figures:
    precision_recall.png      fidelity vs diversity, one dot per run             (H1, H2)
    mode_coverage.png         class histogram of generated samples               (H2)
    sharpness.png             Laplacian variance & high-frequency ratio vs real  (H1)
    learning_curve.png        FID / recall over training  (needs `evaluate --all-epochs --quick`)
    training_curves.png       losses and D outputs from train_log.csv
    ablation_latent_dim.png, ablation_beta.png
    qualitative_*.png         VAE and DCGAN evaluation grids side by side (samples, successes and
                              failures, interpolation, nearest neighbours, reconstructions)

If runs/real_baseline_<dataset>/metrics.json exists (`python -m src.real_baseline`), the main
table gets a "Real data" row and the precision-recall plot a reference star.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402

# Validated categorical palette: slot 1 (blue) and slot 2 (orange). Real data = neutral gray.
COLORS = {"vae": "#2a78d6", "gan": "#eb6834"}
LABELS = {"vae": "VAE", "gan": "DCGAN"}
MODELS = ["vae", "gan"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

METRIC_NAMES = {
    "fid_inception": "FID-Inception ↓",
    "fid_classifier": "FID-classifier ↓",
    "precision_inception": "Precision-Inc ↑",
    "recall_inception": "Recall-Inc ↑",
    "precision_classifier": "Precision-clf ↑",
    "recall_classifier": "Recall-clf ↑",
    "modes_covered": "Modes covered ↑",
    "class_kl": "Class KL ↓",
    "laplacian_var": "Laplacian var",
    "hf_ratio": "HF ratio",
    "interp_step_cv": "Interp. CV ↓",
    "memorization_ratio": "NN ratio",
    "recon_psnr": "Recon PSNR ↑",
    "train_time_sec": "Train time (s)",
}


def _style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.constrained_layout.use": True,  # keeps suptitle and panel titles apart
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
    })


def _save(fig, path, written):
    fig.savefig(path)
    plt.close(fig)
    written.append(path)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_evals(runs_dir):
    rows = []
    for path in sorted(Path(runs_dir).glob("*/eval/epoch_*/metrics.json")):
        with open(path) as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows)


def final_full_evals(df):
    """The last full (non --quick) evaluation of every run."""
    full = df[~df["quick"].astype(bool)]
    return full.loc[full.groupby("run")["epoch"].idxmax()].reset_index(drop=True)


def default_mask(df, latent_dim, beta):
    return (df["latent_dim"] == latent_dim) & ((df["model"] == "gan") | (df["beta"] == beta))


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _fmt(values):
    values = values.dropna()
    if values.empty:
        return "–"
    if len(values) == 1:
        return f"{values.iloc[0]:.3g}"
    return f"{values.mean():.3g} ± {values.std(ddof=1):.2g}"


def mean_std_table(df, group_cols, metrics):
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = {col: (LABELS.get(k, k) if col == "model" else k) for col, k in zip(group_cols, keys)}
        row["seeds"] = group["seed"].nunique()
        for metric in metrics:
            row[METRIC_NAMES.get(metric, metric)] = _fmt(group[metric]) if metric in group else "–"
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(table, path, title, note, written):
    lines = [f"# {title}", "", note, "",
             "| " + " | ".join(map(str, table.columns)) + " |",
             "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in table.itertuples(index=False)]
    path.write_text("\n".join(lines) + "\n")
    written.append(path)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _present(df, models=MODELS):
    return [m for m in models if m in set(df["model"])]


def plot_precision_recall(main, out, written, baseline=None):
    spaces = [s for s in ("classifier", "inception")
              if f"precision_{s}" in main and main[f"precision_{s}"].notna().any()]
    if not spaces:
        return
    fig, axes = plt.subplots(1, len(spaces), figsize=(3.3 * len(spaces), 3.2), squeeze=False, sharey=True)
    for ax, space in zip(axes[0], spaces):
        for model in _present(main):
            g = main[main["model"] == model]
            ax.scatter(g[f"recall_{space}"], g[f"precision_{space}"], s=45, color=COLORS[model],
                       label=LABELS[model], edgecolor="white", linewidth=1.5, zorder=3)
        if baseline is not None and f"recall_{space}" in baseline:
            ax.scatter(baseline[f"recall_{space}"], baseline[f"precision_{space}"], s=90, marker="*",
                       color=MUTED, label="real data (train vs. test)", edgecolor="white", linewidth=1, zorder=3)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall (diversity)", title=f"{space.capitalize()} features")
    axes[0][0].set_ylabel("Precision (fidelity)")
    axes[0][-1].legend(loc="lower right")
    _save(fig, out / "precision_recall.png", written)


def plot_mode_coverage(main, out, class_labels, written):
    models = _present(main)
    hists = {m: np.mean(np.stack(main[main["model"] == m]["class_histogram"].tolist()), axis=0) for m in models}
    k = len(next(iter(hists.values())))
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    width = 0.8 / len(models)
    x = np.arange(k)
    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, hists[model], width, color=COLORS[model], label=LABELS[model],
               edgecolor="white", linewidth=1)
    ax.axhline(1 / k, color=MUTED, linestyle="--", linewidth=1.2, label="balanced (real data)")
    labels = class_labels if class_labels and len(class_labels) == k else [str(i) for i in range(k)]
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Share of generated samples")
    ax.set_title("Class distribution of generated samples (evaluation classifier)")
    ax.legend(loc="upper left", ncols=len(models) + 1)
    ax.set_ylim(0, max(1.15 * max(h.max() for h in hists.values()), 0.2))  # room for the legend
    _save(fig, out / "mode_coverage.png", written)


def plot_sharpness(main, out, written):
    models = _present(main)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    panels = [("laplacian_var", "Variance of Laplacian"), ("hf_ratio", "High-frequency energy ratio")]
    for ax, (key, title) in zip(axes, panels):
        means = [main.loc[main["model"] == m, key].mean() for m in models]
        stds = [main.loc[main["model"] == m, key].std(ddof=1) for m in models]
        ax.bar(range(len(models)), means, yerr=np.nan_to_num(stds), width=0.55,
               color=[COLORS[m] for m in models], capsize=3, error_kw={"ecolor": MUTED, "elinewidth": 1})
        ax.axhline(main[f"{key}_real"].mean(), color=MUTED, linestyle="--", linewidth=1.2, label="real data")
        ax.set_xticks(range(len(models)), [LABELS[m] for m in models])
        ax.set_title(title)
    axes[0].legend(loc="upper left")
    fig.suptitle("Sharpness of generated images (higher = more fine detail)", fontsize=10)
    _save(fig, out / "sharpness.png", written)


def plot_learning_curve(curve, out, written):
    if curve.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for ax, (key, title) in zip(axes, [("fid_classifier", "FID-classifier ↓"), ("recall_classifier", "Recall ↑")]):
        for model in _present(curve):
            stats = curve[curve["model"] == model].groupby("epoch")[key].agg(["mean", "min", "max"])
            ax.plot(stats.index, stats["mean"], color=COLORS[model], label=LABELS[model], marker="o")
            ax.fill_between(stats.index, stats["min"], stats["max"], color=COLORS[model], alpha=0.15, linewidth=0)
        ax.set(xlabel="Epoch", title=title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if key.startswith("recall"):
            ax.set_ylim(0, 1)
    axes[0].legend()
    fig.suptitle("Quality during training (line = mean over seeds, band = min–max)", fontsize=10)
    _save(fig, out / "learning_curve.png", written)


def plot_training_curves(runs_dir, main, out, written):
    logs = {}
    for run, model in zip(main["run"], main["model"]):
        path = Path(runs_dir) / run / "train_log.csv"
        if path.exists():
            logs.setdefault(model, []).append(pd.read_csv(path))
    panels = []
    if "vae" in logs:
        panels += [("vae", "recon", "VAE: reconstruction term"), ("vae", "kl", "VAE: KL term")]
    if "gan" in logs:
        panels += [("gan", None, "DCGAN: discriminator outputs")]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 2.8), squeeze=False)
    for ax, (model, key, title) in zip(axes[0], panels):
        for i, log in enumerate(logs[model]):
            first = i == 0
            if model == "vae":
                ax.plot(log["epoch"], log[key], color=COLORS[model], linewidth=1.5)
            else:
                ax.plot(log["epoch"], log["d_real"], color=COLORS[model], linewidth=1.5,
                        label="D(x)" if first else None)
                ax.plot(log["epoch"], log["d_fake"], color=COLORS[model], linewidth=1.5, linestyle="--",
                        label="D(G(z))" if first else None)
        if model == "gan":
            ax.set_ylim(0, 1)
            ax.legend()
        ax.set(xlabel="Epoch", title=title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle("Training dynamics (one line per seed)", fontsize=10)
    _save(fig, out / "training_curves.png", written)


QUALITATIVE_FIGURES = {
    # output file: [(figure saved by src.evaluate, panel caption), ...]  -> one row per entry
    "qualitative_samples.png": [("samples.png", "random samples")],
    "qualitative_success_failure.png": [("most_confident.png", "most confident (successes)"),
                                        ("least_confident.png", "least confident (failures)")],
    "qualitative_interpolation.png": [("interpolation.png", "slerp interpolation, one path per row")],
    "qualitative_nearest_neighbors.png": [("nearest_neighbors.png", "generated (top) vs. nearest train (bottom)")],
    "qualitative_reconstructions.png": [("reconstructions.png", "test image (top) vs. reconstruction (bottom)")],
}


def plot_qualitative(runs_dir, main, out, written, panel_width=3.4):
    """Put the evaluation figures of both models side by side (lowest seed of each model).

    runs/ is git-ignored, so this is how the qualitative figures reach results/ and the report.
    The grids come from fixed latent vectors (--eval-seed), so they are NOT cherry-picked.
    """
    picks = {}
    for model in _present(main):
        row = main[main["model"] == model].sort_values("seed").iloc[0]
        eval_dir = Path(runs_dir) / row["run"] / "eval" / f"epoch_{int(row['epoch']):03d}"
        picks[model] = (eval_dir, int(row["seed"]))
    for name, rows in QUALITATIVE_FIGURES.items():
        # columns = models that actually have this figure (reconstructions: VAE only)
        cols = [m for m in picks if all((picks[m][0] / f).exists() for f, _ in rows)]
        if not cols:
            continue
        images = [[plt.imread(picks[m][0] / f) for m in cols] for f, _ in rows]
        aspect = max(img.shape[0] / img.shape[1] for row in images for img in row)
        fig, axes = plt.subplots(len(rows), len(cols), squeeze=False,
                                 figsize=(panel_width * len(cols), (panel_width * aspect + 0.35) * len(rows)))
        for r, (_, caption) in enumerate(rows):
            for c, model in enumerate(cols):
                ax = axes[r][c]
                ax.imshow(images[r][c], interpolation="nearest")
                ax.set_axis_off()
                ax.set_title(f"{LABELS[model]} (seed {picks[model][1]}): {caption}", fontsize=8)
        _save(fig, out / name, written)


def load_baseline(runs_dir, dataset):
    path = Path(runs_dir) / f"real_baseline_{dataset}" / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def baseline_row(baseline, metrics):
    row = {"model": "Real data (train vs. test)", "seeds": "–"}
    for metric in metrics:
        row[METRIC_NAMES.get(metric, metric)] = f"{baseline[metric]:.3g}" if metric in baseline else "–"
    return row


def plot_ablation(df, x_key, panels, title, path, written, log_x=True):
    models = _present(df)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 2.8), squeeze=False)
    for ax, (key, label) in zip(axes[0], panels):
        for model in models:
            stats = df[df["model"] == model].groupby(x_key)[key].agg(["mean", "std"]).sort_index()
            ax.errorbar(stats.index, stats["mean"], yerr=stats["std"].fillna(0), color=COLORS[model],
                        label=LABELS[model], marker="o", capsize=3, elinewidth=1)
        if log_x:
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(df[x_key].unique()), [f"{v:g}" for v in sorted(df[x_key].unique())])
        ax.set(xlabel=x_key, title=label)
        if key.startswith(("recall", "precision")):
            ax.set_ylim(0, 1)  # fractions: never show a negative axis
    if len(models) > 1:
        axes[0][0].legend()
    fig.suptitle(title, fontsize=10)
    _save(fig, path, written)


# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--configs", default="configs", help="used to find the default latent_dim / beta")
    return parser.parse_args()


def main():
    args = parse_args()
    _style()
    shared = load_config(Path(args.configs) / "shared.yaml")
    default_z, default_beta = shared["latent_dim"], load_config(Path(args.configs) / "vae.yaml")["beta"]

    df = load_evals(args.runs)
    if df.empty:
        raise SystemExit(f"No evaluations found under {args.runs}/*/eval/. Run src.evaluate first.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    for dataset, ddf in df.groupby("dataset"):
        dout = out if df["dataset"].nunique() == 1 else out / dataset
        dout.mkdir(parents=True, exist_ok=True)
        final = final_full_evals(ddf)
        final.drop(columns=["class_histogram"]).to_csv(dout / "all_runs.csv", index=False)
        written.append(dout / "all_runs.csv")

        # ---- Main comparison --------------------------------------------- #
        main_df = final[default_mask(final, default_z, default_beta)]
        baseline = load_baseline(args.runs, dataset)
        if not main_df.empty:
            table = mean_std_table(main_df, ["model"], list(METRIC_NAMES))
            if baseline is not None:
                table = pd.concat([table, pd.DataFrame([baseline_row(baseline, METRIC_NAMES)])], ignore_index=True)
            write_markdown(
                table, dout / "main_results.md",
                f"Main results: {dataset}, latent_dim={default_z}, VAE beta={default_beta:g}",
                f"Mean ± std over seeds. The last row runs the same metrics on 10,000 real TRAINING images "
                f"instead of generated ones (`python -m src.real_baseline`): the best score a perfect generator "
                f"could expect. Evaluation classifier test accuracy = "
                f"{main_df['eval_classifier_accuracy'].iloc[0]:.3f}.",
                written)
            plot_precision_recall(main_df, dout, written, baseline)
            plot_sharpness(main_df, dout, written)
            plot_training_curves(args.runs, main_df, dout, written)
            plot_qualitative(args.runs, main_df, dout, written)
            try:
                from src.data import class_names
                labels = class_names(dataset, shared["data_root"])
            except Exception:  # dataset not downloaded here: fall back to class indices
                labels = None
            plot_mode_coverage(main_df, dout, labels, written)

        # ---- Learning curve (quick evaluations of intermediate checkpoints) #
        quick = ddf[ddf["quick"].astype(bool)]
        plot_learning_curve(quick[default_mask(quick, default_z, default_beta)], dout, written)

        # ---- Ablations ----------------------------------------------------- #
        z_sweep = final[(final["model"] == "gan") | (final["beta"] == default_beta)]
        if z_sweep["latent_dim"].nunique() > 1:
            write_markdown(mean_std_table(z_sweep, ["model", "latent_dim"], list(METRIC_NAMES)),
                           dout / "ablation_latent_dim.md", f"Ablation: latent dimension ({dataset})",
                           "Mean ± std over seeds.", written)
            plot_ablation(z_sweep, "latent_dim",
                          [("fid_classifier", "FID-classifier ↓"), ("recall_classifier", "Recall ↑"),
                           ("laplacian_var", "Laplacian variance")],
                          "Effect of latent dimension", dout / "ablation_latent_dim.png", written)

        b_sweep = final[(final["model"] == "vae") & (final["latent_dim"] == default_z)]
        if b_sweep["beta"].nunique() > 1:
            write_markdown(mean_std_table(b_sweep, ["beta"], list(METRIC_NAMES)),
                           dout / "ablation_beta.md", f"Ablation: VAE beta ({dataset})",
                           "Mean ± std over seeds.", written)
            plot_ablation(b_sweep, "beta",
                          [("fid_classifier", "FID-classifier ↓"), ("laplacian_var", "Laplacian variance"),
                           ("interp_step_cv", "Interp. CV ↓")],
                          "Effect of beta (VAE)", dout / "ablation_beta.png", written)

    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
