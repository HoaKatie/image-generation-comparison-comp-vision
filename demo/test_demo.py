"""Tests for the demo.  Run from the repository root:   pytest demo -q

The rule tests use made-up numbers, so they run instantly and need no data or checkpoints.
The last test runs the real engine on a small pool and is skipped if the checkpoints or
Fashion-MNIST are not available locally.
"""
from pathlib import Path

import pytest
import torch

from demo import compare_rules as rules
from demo.engine import DEFAULT_GAN, DEFAULT_VAE, ROOT, Comparison, ModelResult

BASE = {"share": 0.10, "confidence": 0.90, "sharpness": 0.15, "detail": 0.10,
        "precision": 0.80, "recall": 0.80, "spread": 6.0}


def make_comparison(vae=None, gan=None, n_vae=300, n_gan=300):
    """A Comparison with identical models (all ties) unless `vae` / `gan` override some stats."""
    def result(name, overrides, n):
        return ModelResult(name, "test", torch.zeros(0, 1, 32, 32), torch.zeros(0), n, {**BASE, **(overrides or {})})
    real_stats = {**BASE, "precision": None, "recall": None}
    return Comparison(9, "Ankle boot", 0, 10000, min(n_vae, n_gan), 0.907, torch.zeros(0, 1, 32, 32),
                      real_stats, result("VAE", vae, n_vae), result("DCGAN", gan, n_gan))


def paragraph(comp, heading):
    return dict(rules.explain(comp))[heading]


# --------------------------------------------------------------------------- rules
def test_winner_higher_is_better_with_tie_margin():
    assert rules.winner(0.90, 0.70, "higher") == "VAE"
    assert rules.winner(0.80, 0.83, "higher") == "Tie"      # within 5 percentage points
    assert rules.winner(None, 0.83, "higher") == "n/a"


def test_winner_closer_rewards_matching_real_not_maximising():
    # DCGAN has twice the real edge energy (noise), the VAE matches real: VAE should win.
    assert rules.winner(0.15, 0.30, "closer", real=0.15) == "VAE"
    assert rules.winner(0.03, 0.14, "closer", real=0.15) == "DCGAN"


def test_identical_models_are_all_ties():
    comp = make_comparison()
    assert rules.score(comp) == {"VAE": 0, "DCGAN": 0, "Tie": len(rules.METRICS)}
    assert "no clear winner" in paragraph(comp, "Verdict")


def test_blurry_vae_is_explained():
    comp = make_comparison(vae={"sharpness": 0.03}, gan={"sharpness": 0.15})
    text = paragraph(comp, "Sharpness")
    assert "DCGAN images are 5.0x sharper" in text and "AVERAGE" in text


def test_fidelity_diversity_trade_off_is_detected():
    comp = make_comparison(vae={"precision": 0.90, "recall": 0.50}, gan={"precision": 0.70, "recall": 0.80})
    assert "Classic trade-off: VAE is more faithful" in paragraph(comp, "Fidelity vs diversity")


def test_low_recall_and_rare_class_are_flagged():
    comp = make_comparison(vae={"recall": 0.10, "share": 0.02})
    assert "partial mode collapse" in paragraph(comp, "Fidelity vs diversity")
    assert "VAE rarely makes Ankle boot images" in paragraph(comp, "How often each model makes this class")


def test_too_few_samples_skips_precision_recall():
    comp = make_comparison(vae={"precision": None, "recall": None}, gan={"precision": None, "recall": None})
    assert "Too few images" in paragraph(comp, "Fidelity vs diversity")
    assert all(row[-1] == "n/a" for row in rules.table_rows(comp) if "precision" in row[0] or "recall" in row[0])


# --------------------------------------------------------------------------- engine (real models)
@pytest.mark.skipif(not (Path(DEFAULT_VAE).exists() and Path(DEFAULT_GAN).exists()
                         and (ROOT / "data" / "FashionMNIST").exists()),
                    reason="needs the committed checkpoints and a downloaded Fashion-MNIST")
def test_engine_end_to_end_small_pool():
    from demo.engine import DemoEngine
    engine = DemoEngine(pool_size=500)
    comp = engine.compare(1, seed=0)                    # Trouser
    assert comp.class_name == "Trouser"
    assert comp.n_compared == min(500, comp.vae.n_found, comp.gan.n_found)
    for result in (comp.vae, comp.gan):
        assert result.images.shape[1:] == (1, 32, 32)
        assert 0 <= result.stats["share"] <= 1
    assert len(rules.table_rows(comp)) == len(rules.METRICS)
    # Same seed -> exactly the same images (the demo is reproducible for rehearsals).
    again = engine.compare(1, seed=0)
    assert torch.equal(comp.gan.images, again.gan.images)
