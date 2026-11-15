"""End-to-end test of the whole pipeline on a tiny synthetic dataset.

    train_classifier -> train_vae -> train_gan -> evaluate (full + --all-epochs --quick)
    -> real_baseline -> analysis

The real scripts are run through their command-line entry points. Only the data loading is
replaced, so the test needs no download, runs on the CPU in a few seconds, and checks that
every output the report relies on is produced with sensible values. The metric VALUES are
meaningless here (2 epochs on toy data). Correctness of the metrics themselves is tested in
test_metrics.py.
"""
import json
import sys

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import analysis, evaluate, real_baseline, train_classifier, train_gan, train_vae
from src.losses import vae_loss
from src.models import build_model, load_checkpoint

EPOCHS = 2
COMMON = ["device=cpu", "num_workers=0", "base_channels=16", "batch_size=64", "save_every=1"]
VAE_RUN, GAN_RUN = "vae_fashion_mnist_z32_beta1_seed0", "gan_fashion_mnist_z32_seed0"


def synthetic_split(n, seed):
    """Class k = a bright horizontal bar at a class-specific height, plus a little noise."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(n) % 10
    images = torch.full((n, 1, 32, 32), -1.0)
    for k in range(10):
        images[labels == k, :, 2 + 3 * k: 4 + 3 * k, 4:28] = 1.0
    return (images + 0.1 * torch.randn(images.shape, generator=g)).clamp(-1, 1), labels


SPLITS = {"train": synthetic_split(256, seed=0), "test": synthetic_split(200, seed=1)}


def fake_load_split(name, split, root="data"):
    return SPLITS[split]


def fake_get_loader(name, split, batch_size, num_workers=2, root="data", subset=None, seed=0, shuffle=None):
    shuffle = split == "train" if shuffle is None else shuffle
    return DataLoader(TensorDataset(*SPLITS[split]), batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def run_cli(mp, entry_point, *args):
    mp.setattr(sys, "argv", ["prog", *args])
    entry_point()


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Run every stage once; the tests below inspect the outputs."""
    root = tmp_path_factory.mktemp("pipeline")
    runs, results = root / "runs", root / "results"
    common = [f"output_dir={runs}", *COMMON]
    with pytest.MonkeyPatch.context() as mp:
        for module in (train_vae, train_gan, train_classifier):
            mp.setattr(module, "get_loader", fake_get_loader)
        mp.setattr(train_classifier, "load_split", fake_load_split)
        mp.setattr(evaluate, "load_split", fake_load_split)
        mp.setattr("src.data.class_names", lambda *a, **k: [f"class {i}" for i in range(10)])

        # 10 epochs x 4 batches: enough for the toy classes (100% from epoch ~9), so the
        # mode-coverage numbers below are measured with a working classifier.
        run_cli(mp, train_classifier.main, "--config", "configs/classifier.yaml", "--set", *common, "epochs=10")
        for model, module in (("vae", train_vae), ("gan", train_gan)):
            config = f"configs/{model}.yaml"
            run_cli(mp, module.main, "--config", config, "--set", *common, f"epochs={EPOCHS}")
            run_cli(mp, evaluate.main, "--config", config, "--set", *common,
                    "--feature-spaces", "classifier", "--n-samples", "200")
            run_cli(mp, evaluate.main, "--config", config, "--set", *common, "--all-epochs", "--quick")
        run_cli(mp, real_baseline.main, "--set", *common, "--feature-spaces", "classifier")
        run_cli(mp, analysis.main, "--runs", str(runs), "--out", str(results))
    return runs, results


def _metrics(runs, run, epoch=EPOCHS, quick=False):
    path = runs / run / "eval" / (f"epoch_{epoch:03d}" + ("_quick" if quick else "")) / "metrics.json"
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
def test_classifier_learns_the_toy_classes(pipeline):
    runs, _ = pipeline
    acc = json.loads((runs / "classifier_fashion_mnist" / "metrics.json").read_text())["test_accuracy"]
    assert acc > 0.5  # chance = 0.1; the toy classes are trivially separable


@pytest.mark.parametrize("run,columns", [(VAE_RUN, ["loss", "recon", "kl"]),
                                         (GAN_RUN, ["loss_d", "loss_g", "d_real", "d_fake"])])
def test_training_writes_logs_samples_and_checkpoints(pipeline, run, columns):
    runs, _ = pipeline
    log = pd.read_csv(runs / run / "train_log.csv")
    assert log["epoch"].tolist() == list(range(1, EPOCHS + 1))
    assert set(columns) <= set(log.columns)
    assert log[columns].notna().all().all()
    for epoch in range(1, EPOCHS + 1):
        assert (runs / run / "samples" / f"epoch_{epoch:03d}.png").exists()
        assert (runs / run / "checkpoints" / f"epoch_{epoch:03d}.pt").exists()
    assert (runs / run / "config.yaml").exists()
    if "d_real" in columns:  # probabilities
        assert log[["d_real", "d_fake"]].stack().between(0, 1).all()


def test_checkpoints_reload_with_identical_decoders(pipeline):
    """Fairness survives saving/loading: both checkpoints contain the same-size image network."""
    runs, _ = pipeline
    _, _, vae_meta = load_checkpoint(runs / VAE_RUN / "checkpoints" / f"epoch_{EPOCHS:03d}.pt", "cpu")
    gan, cfg, gan_meta = load_checkpoint(runs / GAN_RUN / "checkpoints" / f"epoch_{EPOCHS:03d}.pt", "cpu")
    assert vae_meta["params_decoder"] == gan_meta["params_decoder"]
    assert vae_meta["epoch"] == gan_meta["epoch"] == EPOCHS
    assert gan.decode(torch.randn(3, cfg["latent_dim"])).shape == (3, 1, 32, 32)


@pytest.mark.parametrize("run", [VAE_RUN, GAN_RUN])
def test_full_evaluation_reports_every_metric_in_range(pipeline, run):
    runs, _ = pipeline
    m = _metrics(runs, run)
    assert m["n_samples"] == 200 and m["quick"] is False
    assert m["fid_classifier"] >= 0
    for key in ("precision_classifier", "recall_classifier", "mean_confidence"):
        assert 0 <= m[key] <= 1, key
    assert 0 <= m["modes_covered"] <= 10
    assert sum(m["class_histogram"]) == pytest.approx(1.0)
    assert m["class_kl"] >= -1e-9
    assert m["laplacian_var"] >= 0 and 0 <= m["hf_ratio"] <= 1
    assert m["interp_step_cv"] >= 0 and m["memorization_ratio"] > 0
    assert ("recon_psnr" in m) == run.startswith("vae")  # reconstruction is VAE-only

    figures = ["samples", "most_confident", "least_confident", "interpolation", "nearest_neighbors"]
    figures += ["reconstructions"] if run.startswith("vae") else []
    for name in figures:
        assert (runs / run / "eval" / f"epoch_{EPOCHS:03d}" / f"{name}.png").exists(), name


@pytest.mark.parametrize("run", [VAE_RUN, GAN_RUN])
def test_quick_evaluations_cover_every_checkpoint_without_overwriting(pipeline, run):
    runs, _ = pipeline
    for epoch in range(1, EPOCHS + 1):
        m = _metrics(runs, run, epoch, quick=True)
        assert m["quick"] is True and m["epoch"] == epoch
        assert not (runs / run / "eval" / f"epoch_{epoch:03d}_quick" / "samples.png").exists()
    assert _metrics(runs, run)["quick"] is False  # the full evaluation is still there


def test_real_baseline_is_better_than_barely_trained_generators(pipeline):
    runs, _ = pipeline
    real = json.loads((runs / "real_baseline_fashion_mnist" / "metrics.json").read_text())
    assert real["modes_covered"] == 10
    for run in (VAE_RUN, GAN_RUN):
        assert real["fid_classifier"] < _metrics(runs, run)["fid_classifier"]


def test_analysis_builds_the_report_tables_and_figures(pipeline):
    _, results = pipeline
    table = (results / "main_results.md").read_text()
    for row in ("| VAE |", "| DCGAN |", "| Real data (train vs. test) |"):
        assert row in table
    assert len(pd.read_csv(results / "all_runs.csv")) == 2
    for name in ("precision_recall", "sharpness", "training_curves", "mode_coverage", "learning_curve",
                 "qualitative_samples", "qualitative_success_failure", "qualitative_interpolation",
                 "qualitative_nearest_neighbors", "qualitative_reconstructions"):
        assert (results / f"{name}.png").exists(), name


# --------------------------------------------------------------------------- #
def test_vae_can_overfit_a_small_batch():
    """Classic sanity check: loss, reparameterization and gradients are wired up correctly
    if the VAE can drive the loss on a fixed small batch far below its starting value."""
    torch.manual_seed(0)
    x, _ = synthetic_split(32, seed=0)
    model = build_model({"model": "vae", "dataset": "fashion_mnist", "latent_dim": 8, "base_channels": 16})
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(60):  # loss ratio is ~0.13 after 60 steps; 0.25 leaves a safety margin
        x_recon, mu, logvar = model(x)
        loss, _, _ = vae_loss(x, x_recon, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.25 * losses[0]
