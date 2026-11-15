"""Acceptance tests for the losses.

The vae_loss tests FAIL until the team implements the TODO in src/losses.py.
They pin down the required behaviour without deciding the open design choice
(sum vs mean over pixels for the reconstruction term).
"""
import pytest
import torch

from src.losses import discriminator_loss, generator_loss, vae_loss

B, D = 8, 16


@pytest.fixture
def x():
    return torch.rand(B, 1, 32, 32, generator=torch.Generator().manual_seed(0)) * 2 - 1


def test_vae_loss_returns_scalars(x):
    loss, recon, kl = vae_loss(x, x * 0.5, torch.randn(B, D), torch.randn(B, D))
    assert loss.dim() == recon.dim() == kl.dim() == 0


def test_perfect_reconstruction_and_prior_posterior_give_zero(x):
    loss, recon, kl = vae_loss(x, x, torch.zeros(B, D), torch.zeros(B, D))
    assert recon.item() == pytest.approx(0.0, abs=1e-6)
    assert kl.item() == pytest.approx(0.0, abs=1e-6)


def test_kl_closed_form(x):
    # mu = 1, sigma = 1: KL = 0.5 * mu^2 = 0.5 per latent dim -> 0.5 * D per image (summed over dims,
    # averaged over the batch)
    _, _, kl = vae_loss(x, x, torch.ones(B, D), torch.zeros(B, D))
    assert kl.item() == pytest.approx(0.5 * D, rel=1e-5)


def test_beta_only_scales_kl(x):
    mu, logvar = torch.randn(B, D), torch.randn(B, D)
    loss1, recon1, kl1 = vae_loss(x, x * 0.5, mu, logvar, beta=1.0)
    loss4, recon4, kl4 = vae_loss(x, x * 0.5, mu, logvar, beta=4.0)
    assert recon1.item() == pytest.approx(recon4.item())
    assert (loss4 - loss1).item() == pytest.approx(3 * kl1.item(), rel=1e-5)


def test_recon_grows_with_error(x):
    mu, logvar = torch.zeros(B, D), torch.zeros(B, D)
    _, small, _ = vae_loss(x, x + 0.1, mu, logvar)
    _, large, _ = vae_loss(x, x + 0.5, mu, logvar)
    assert large > small > 0


def test_gan_losses_reward_a_good_discriminator():
    confident_real, confident_fake = torch.full((B, 1), 10.0), torch.full((B, 1), -10.0)
    assert discriminator_loss(confident_real, confident_fake).item() < 1e-3
    # If D confidently rejects the fakes, the generator loss is large
    assert generator_loss(confident_fake).item() > 5
