"""Shared convolutional building blocks (DCGAN-style, Radford et al., 2016).

    ConvDecoder  = VAE decoder   = GAN generator
    ConvEncoder  = VAE encoder   = GAN discriminator   (only the output size differs)

Because both generative models use exactly the same image-producing network,
differences in our results can be attributed to the TRAINING OBJECTIVE
(ELBO vs. adversarial loss) rather than to model capacity.
tests/test_models.py checks this.
"""
import torch.nn as nn


class ConvDecoder(nn.Module):
    """z (B, latent_dim) -> image (B, C, 32, 32) with values in [-1, 1]."""

    def __init__(self, latent_dim, out_channels, base=64):
        super().__init__()
        self.net = nn.Sequential(
            # project z to a 4x4 feature map
            nn.ConvTranspose2d(latent_dim, base * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(base, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class ConvEncoder(nn.Module):
    """image (B, C, 32, 32) -> vector (B, out_dim)."""

    def __init__(self, in_channels, out_dim, base=64):
        super().__init__()
        self.features = nn.Sequential(
            # 32x32 -> 16x16 (no BatchNorm on the first layer, as in DCGAN)
            nn.Conv2d(in_channels, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 -> 4x4
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 4x4 -> 1x1
        self.head = nn.Conv2d(base * 4, out_dim, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        return self.head(self.features(x)).flatten(1)


def init_dcgan_weights(module):
    """DCGAN initialisation, applied to BOTH models so neither gets a head start."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())
