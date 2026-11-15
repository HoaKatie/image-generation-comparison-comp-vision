"""Sanity checks: every metric behaves as expected on inputs where we KNOW the answer.

These are worth mentioning in the report -- they are evidence that our
evaluation measures what we claim it measures.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src import metrics as M


def _rng(seed=0):
    return torch.Generator().manual_seed(seed)


# ---- Fréchet distance ----------------------------------------------------- #
def test_frechet_distance_identical_is_zero():
    a = np.random.default_rng(0).normal(size=(2000, 8))
    assert M.frechet_distance(a, a) == pytest.approx(0.0, abs=1e-6)


def test_frechet_distance_mean_shift():
    a = np.random.default_rng(0).normal(size=(2000, 8))
    # same covariance, mean shifted by 1 in each of 8 dims -> FD = ||delta mu||^2 = 8
    assert M.frechet_distance(a, a + 1.0) == pytest.approx(8.0, rel=1e-3)


# ---- Precision / recall --------------------------------------------------- #
def test_precision_recall_identical_sets():
    a = torch.randn(1000, 4, generator=_rng())
    assert M.precision_recall(a, a) == (1.0, 1.0)


def test_precision_recall_detects_mode_collapse():
    """Real data has two clusters; the 'generator' only produces one of them."""
    g = _rng()
    cluster_a = torch.randn(1000, 2, generator=g)
    cluster_b = torch.randn(1000, 2, generator=g) + 20.0
    real = torch.cat([cluster_a, cluster_b])
    fake = torch.randn(2000, 2, generator=g)  # only cluster A
    precision, recall = M.precision_recall(real, fake)
    assert precision > 0.95             # every fake looks real...
    assert 0.4 < recall < 0.6           # ...but half of the real data is never produced


# ---- Mode coverage -------------------------------------------------------- #
def test_mode_coverage_balanced_vs_collapsed():
    reference = np.repeat(np.arange(10), 100)
    balanced = M.mode_coverage(reference, 10, reference)
    assert balanced["modes_covered"] == 10
    assert balanced["class_kl"] == pytest.approx(0.0, abs=1e-6)

    collapsed = M.mode_coverage(np.full(1000, 3), 10, reference)
    assert collapsed["modes_covered"] == 1
    assert collapsed["class_kl"] == pytest.approx(math.log(10), rel=1e-4)


# ---- Sharpness ------------------------------------------------------------ #
def _blur(images, k=5):
    return F.avg_pool2d(F.pad(images, (k // 2,) * 4, mode="reflect"), k, stride=1)


def test_blur_lowers_both_sharpness_measures():
    sharp = torch.rand(16, 1, 32, 32, generator=_rng()) * 2 - 1
    blurry = _blur(sharp)
    assert (M.laplacian_variance(blurry) < M.laplacian_variance(sharp)).all()
    assert (M.high_frequency_ratio(blurry) < M.high_frequency_ratio(sharp)).all()


def test_sharpness_works_for_rgb():
    rgb = torch.rand(4, 3, 32, 32, generator=_rng()) * 2 - 1
    assert M.laplacian_variance(rgb).shape == (4,)
    assert M.high_frequency_ratio(rgb).shape == (4,)


# ---- PSNR / nearest neighbours ------------------------------------------- #
def test_psnr_known_value():
    x = torch.zeros(2, 1, 8, 8)
    y = x + 0.2  # 0.1 in [0, 1] units -> MSE = 0.01 -> PSNR = 20 dB
    assert M.psnr(x, y) == pytest.approx(torch.tensor([20.0, 20.0]), abs=1e-4)


def test_nearest_neighbors_finds_exact_copy():
    reference = torch.randn(50, 1, 8, 8, generator=_rng())
    dists, idxs = M.nearest_neighbors(reference[[7, 21]], reference)
    assert idxs.tolist() == [7, 21]
    # cdist uses the fast ||a||^2 + ||b||^2 - 2ab formula, so "0" comes out as ~1e-3.
    # Irrelevant in practice: real nearest-neighbour distances are around 5-10.
    assert dists.max().item() == pytest.approx(0.0, abs=1e-2)


# ---- Latent space --------------------------------------------------------- #
def test_slerp_endpoints_and_shape():
    z0, z1 = torch.randn(3, 16, generator=_rng(0)), torch.randn(3, 16, generator=_rng(1))
    path = M.slerp(z0, z1, torch.linspace(0, 1, 5))
    assert path.shape == (3, 5, 16)
    assert torch.allclose(path[:, 0], z0, atol=1e-5)
    assert torch.allclose(path[:, -1], z1, atol=1e-5)


def test_path_smoothness_prefers_even_steps():
    t = torch.linspace(0, 1, 10).view(1, 10, 1)
    even = t.expand(4, 10, 2).clone()                 # constant speed
    jumpy = (t > 0.5).float().expand(4, 10, 2).clone()  # stays still, then one big jump
    assert M.path_smoothness(even) < 1e-5
    assert M.path_smoothness(jumpy) > 1.0
