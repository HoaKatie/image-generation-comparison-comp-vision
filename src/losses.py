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
    batch = x.size(0)

    # Reconstruction: squared error SUMMED over all pixels of an image, then averaged
    # over the batch. This is -log p(x|z) of a Gaussian decoder with a fixed variance
    # (sigma^2 = 1/2 on the [-1, 1] scale), up to an additive constant.
    #
    # Why SUM and not MEAN over pixels: a 32x32 image gives 1024 pixel terms while the
    # KL has only latent_dim terms. Averaging over pixels would silently make the KL
    # ~1000x stronger, i.e. a hidden beta of ~1000, and the samples would be far
    # blurrier. With the sum, the ELBO is the real one and `beta` is the ONLY knob for
    # the reconstruction/KL trade-off, which is what the beta ablation varies.
    recon = F.mse_loss(x_recon, x, reduction="sum") / batch

    # Closed-form KL( N(mu, diag(sigma^2)) || N(0, I) ) (Kingma & Welling 2014, App. B),
    # summed over latent dims, averaged over the batch.
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch

    return recon + beta * kl, recon, kl


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
