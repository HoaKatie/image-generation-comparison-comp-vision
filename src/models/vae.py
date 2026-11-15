"""Variational Autoencoder (Kingma & Welling, 2014)."""
import torch
import torch.nn as nn

from src.models.networks import ConvDecoder, ConvEncoder


class VAE(nn.Module):
    def __init__(self, latent_dim, channels, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        # The encoder outputs 2 numbers per latent dimension: mean and log-variance.
        self.encoder = ConvEncoder(channels, 2 * latent_dim, base)
        self.decoder = ConvDecoder(latent_dim, channels, base)

    def encode(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterization trick: z = mu + sigma * eps, eps ~ N(0, I).

        The randomness lives in eps, so gradients can flow back through mu and
        sigma to the encoder. Sampling z ~ N(mu, sigma^2) directly would not be
        differentiable.
        """
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def decode(self, z):
        """Common interface with the GAN: latent vector -> image."""
        return self.decoder(z)

    @torch.no_grad()
    def reconstruct(self, x):
        mu, _ = self.encode(x)
        return self.decoder(mu)
