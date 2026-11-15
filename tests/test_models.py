import math

import pytest
import torch

from src.models import build_model
from src.models.networks import count_parameters


def _cfg(model, dataset="fashion_mnist", latent_dim=16):
    return {"model": model, "dataset": dataset, "latent_dim": latent_dim, "base_channels": 16}


@pytest.mark.parametrize("dataset,channels", [("fashion_mnist", 1), ("cifar10", 3)])
def test_vae_shapes(dataset, channels):
    vae = build_model(_cfg("vae", dataset))
    x = torch.rand(4, channels, 32, 32) * 2 - 1
    recon, mu, logvar = vae(x)
    assert recon.shape == x.shape
    assert mu.shape == logvar.shape == (4, 16)
    assert recon.abs().max() <= 1.0  # tanh output


@pytest.mark.parametrize("dataset,channels", [("fashion_mnist", 1), ("cifar10", 3)])
def test_gan_shapes(dataset, channels):
    gan = build_model(_cfg("gan", dataset))
    fake = gan.decode(torch.randn(4, 16))
    assert fake.shape == (4, channels, 32, 32)
    assert gan.discriminator(fake).shape == (4, 1)


def test_vae_decoder_and_gan_generator_are_identical():
    """Fairness: the network that produces images is the same architecture and size in both models."""
    vae, gan = build_model(_cfg("vae")), build_model(_cfg("gan"))
    assert str(vae.decoder) == str(gan.generator)
    assert count_parameters(vae.decoder) == count_parameters(gan.generator)


def test_reparameterize_has_requested_mean_and_std():
    vae = build_model(_cfg("vae"))
    mu = torch.full((20000, 16), 2.0)
    logvar = torch.full((20000, 16), math.log(0.25))  # sigma = 0.5
    z = vae.reparameterize(mu, logvar)
    assert z.mean().item() == pytest.approx(2.0, abs=0.01)
    assert z.std().item() == pytest.approx(0.5, abs=0.01)
