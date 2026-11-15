"""Deep Convolutional GAN (Goodfellow et al., 2014; Radford et al., 2016)."""
import torch.nn as nn

from src.models.networks import ConvDecoder, ConvEncoder


class GAN(nn.Module):
    def __init__(self, latent_dim, channels, base=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = ConvDecoder(latent_dim, channels, base)   # same class as the VAE decoder
        self.discriminator = ConvEncoder(channels, 1, base)         # outputs one real/fake logit

    def decode(self, z):
        """Common interface with the VAE: latent vector -> image."""
        return self.generator(z)
