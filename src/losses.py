"""Training objectives for the two compared methods."""
import torch
import torch.nn.functional as F


def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    """Negative ELBO for a VAE with a Gaussian decoder and a N(0, I) prior.

    Args:
        x:          real images, shape (B, C, H, W), values in [-1, 1]
        x_recon:    decoder output, same shape, values in [-1, 1] (tanh)
        mu, logvar: encoder outputs, shape (B, latent_dim)
        beta:       weight on the KL term (1 = standard VAE, >1 = beta-VAE)

    Returns:
        (loss, recon, kl): three scalar (0-dim) tensors, each AVERAGED OVER THE BATCH.
        `loss` is what we backpropagate; `recon` and `kl` are logged separately so
        the training curves show the trade-off between them.
    """
    # TODO(team): implement the VAE objective (~5 lines). Acceptance tests: tests/test_losses.py
    #
    #   1. recon: how far x_recon is from x (per image, then averaged over the batch).
    #   2. kl:    closed-form KL( N(mu, sigma^2) || N(0, I) ), summed over latent dims:
    #               KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    #             then averaged over the batch.
    #   3. loss = recon + beta * kl
    #
    # Design decision to discuss as a team (and justify in the report):
    #   SUM vs MEAN over pixels for the reconstruction term. A 32x32 image has
    #   1024 pixel terms, the KL has only latent_dim terms. Taking the mean over
    #   pixels makes the KL ~1000x stronger relative to reconstruction, which
    #   changes how blurry the samples are -- i.e. it changes the very thing we
    #   are measuring. Summing corresponds to a Gaussian decoder with fixed variance.
    raise NotImplementedError("vae_loss is a team TODO -- see the comments in src/losses.py")


def discriminator_loss(real_logits, fake_logits):
    """D is a binary classifier: real -> 1, fake -> 0 (Goodfellow et al., 2014)."""
    real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return real + fake


def generator_loss(fake_logits):
    """Non-saturating generator loss: maximise log D(G(z)) instead of minimising log(1 - D(G(z))).

    Same fixed point as the original minimax game, but gives strong gradients
    early in training when D easily rejects the generator's samples.
    """
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
