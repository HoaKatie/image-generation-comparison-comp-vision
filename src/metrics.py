"""Evaluation metrics.

Each function is short and self-contained so every team member can explain it
in the Q&A. Sanity checks for all of them live in tests/test_metrics.py.

Which hypothesis each metric tests (see docs/PROJECT_PLAN.md):
    frechet_distance        overall quality (fidelity + diversity together)
    precision_recall        fidelity (precision) vs diversity (recall)  -> H1, H2
    mode_coverage           mode collapse, using the evaluation classifier -> H2
    laplacian_variance,
    high_frequency_ratio    sharpness / blur, classical CV measures      -> H1
    psnr                    reconstruction quality (VAE only)
    nearest_neighbors       memorisation check
    slerp, path_smoothness  latent space smoothness                      -> H3
"""
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg


# --------------------------------------------------------------------------- #
# Distribution-level metrics (operate on feature vectors)
# --------------------------------------------------------------------------- #
def frechet_distance(feats_a, feats_b):
    """Fréchet distance between Gaussians fitted to two feature sets (Heusel et al., 2017).

        FD = ||mu_a - mu_b||^2 + Tr(S_a + S_b - 2 (S_a S_b)^(1/2))

    With Inception features this is the usual FID. Lower is better.
    """
    a = np.asarray(feats_a, dtype=np.float64)
    b = np.asarray(feats_b, dtype=np.float64)
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)

    with warnings.catch_warnings():
        # ReLU features often have "dead" dimensions with zero variance, which makes the
        # covariance singular. sqrtm still returns the correct PSD square root; the
        # finiteness check below catches the cases where it does not.
        warnings.simplefilter("ignore", linalg.LinAlgWarning)
        covmean = linalg.sqrtm(cov_a @ cov_b)
    if not np.isfinite(covmean).all():  # numerically singular product: add a tiny ridge
        ridge = 1e-6 * np.eye(cov_a.shape[0])
        covmean = linalg.sqrtm((cov_a + ridge) @ (cov_b + ridge))
    covmean = np.real(covmean)  # discard tiny imaginary parts from numerical error

    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a) + np.trace(cov_b) - 2 * np.trace(covmean))


def _knn_radius(feats, k, batch=1000):
    """Distance from each point to its k-th nearest neighbour within the same set."""
    radii = []
    for chunk in feats.split(batch):
        dists = torch.cdist(chunk, feats)
        radii.append(dists.kthvalue(k + 1, dim=1).values)  # k+1: skip the distance to itself (0)
    return torch.cat(radii)


def _fraction_inside(queries, support, support_radii, batch=1000):
    """Fraction of query points that fall inside at least one k-NN ball of the support set."""
    inside = []
    for chunk in queries.split(batch):
        dists = torch.cdist(chunk, support)
        inside.append((dists <= support_radii.unsqueeze(0)).any(dim=1))
    return torch.cat(inside).float().mean().item()


def precision_recall(real_feats, fake_feats, k=3):
    """Improved precision & recall (Kynkäänniemi et al., 2019).

    The real (or fake) "manifold" is approximated by a k-NN ball around every sample.
      precision = fraction of FAKE samples inside the REAL manifold  -> fidelity
      recall    = fraction of REAL samples inside the FAKE manifold  -> diversity / coverage
    A sharp model with mode collapse: high precision, low recall.
    A blurry model that covers everything: lower precision, high recall.
    """
    real = torch.as_tensor(real_feats, dtype=torch.float32)
    fake = torch.as_tensor(fake_feats, dtype=torch.float32)
    precision = _fraction_inside(fake, real, _knn_radius(real, k))
    recall = _fraction_inside(real, fake, _knn_radius(fake, k))
    return precision, recall


def mode_coverage(pred_labels, num_classes, reference_labels, min_share=0.01):
    """Class distribution of generated samples, as labelled by the evaluation classifier.

    modes_covered: number of classes that receive at least `min_share` of the samples.
    class_kl:      KL(p_generated || p_real). 0 = classes as balanced as in the real
                   data; grows when some classes are over/under-produced (mode collapse).
    """
    p = np.bincount(pred_labels, minlength=num_classes) / len(pred_labels)
    q = np.bincount(reference_labels, minlength=num_classes) / len(reference_labels)
    eps = 1e-10
    kl = float(np.sum(p * np.log((p + eps) / (q + eps))))
    return {"modes_covered": int((p >= min_share).sum()), "class_kl": kl, "class_histogram": p.tolist()}


# --------------------------------------------------------------------------- #
# Image-level metrics (operate on images in [-1, 1])
# --------------------------------------------------------------------------- #
def _gray01(images):
    """(B, C, H, W) in [-1, 1] -> (B, 1, H, W) grayscale in [0, 1]."""
    x = (images.clamp(-1, 1) + 1) / 2
    if x.size(1) == 3:
        weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
        x = (x * weights).sum(dim=1, keepdim=True)
    return x


LAPLACIAN_KERNEL = torch.tensor([[0.0, 1.0, 0.0],
                                 [1.0, -4.0, 1.0],
                                 [0.0, 1.0, 0.0]])


def laplacian_variance(images):
    """Per-image variance of the Laplacian response (Pech-Pacheco et al., 2000).

    The Laplacian is a second-derivative filter that responds to edges. Blurry
    images have weak edges -> small, uniform response -> low variance.
    Higher = sharper. Returns a (B,) tensor.
    """
    gray = _gray01(images)
    kernel = LAPLACIAN_KERNEL.to(gray).view(1, 1, 3, 3)
    response = F.conv2d(gray, kernel)  # 'valid' convolution: no border artefacts
    return response.flatten(1).var(dim=1)


def high_frequency_ratio(images, cutoff=0.25):
    """Per-image fraction of spectral energy above a radial frequency `cutoff`.

    Frequencies are in cycles/pixel (Nyquist = 0.5). Blurring is a low-pass
    filter, so blurry images keep less energy at high frequencies.
    Higher = more fine detail. Returns a (B,) tensor.
    """
    gray = _gray01(images)
    gray = gray - gray.mean(dim=(-2, -1), keepdim=True)  # remove the DC component
    power = torch.fft.fft2(gray).abs() ** 2
    h, w = gray.shape[-2:]
    fy = torch.fft.fftfreq(h, device=gray.device).view(-1, 1)
    fx = torch.fft.fftfreq(w, device=gray.device).view(1, -1)
    high = (fy ** 2 + fx ** 2).sqrt() > cutoff
    return power[..., high].flatten(1).sum(1) / power.flatten(1).sum(1).clamp_min(1e-12)


def psnr(x, y):
    """Peak signal-to-noise ratio in dB, per image, computed in [0, 1] (so MAX = 1)."""
    mse = (((x.clamp(-1, 1) - y.clamp(-1, 1)) / 2) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


def nearest_neighbors(queries, reference, batch=500):
    """For every query image: L2 distance (pixel space) to its nearest reference image, and its index.

    Used to check memorisation: if generated images are much closer to the
    training set than unseen TEST images are, the model is copying training data.
    """
    ref = reference.flatten(1)
    dists, idxs = [], []
    for chunk in queries.flatten(1).split(batch):
        nearest = torch.cdist(chunk, ref).min(dim=1)
        dists.append(nearest.values)
        idxs.append(nearest.indices)
    return torch.cat(dists), torch.cat(idxs)


# --------------------------------------------------------------------------- #
# Latent space
# --------------------------------------------------------------------------- #
def slerp(z0, z1, t):
    """Spherical interpolation between latent batches z0, z1 (P, D) at times t (S,) -> (P, S, D).

    Under a Gaussian prior, almost all samples lie near a sphere of radius sqrt(D).
    Linear interpolation cuts through the low-norm region near the origin that the
    model rarely sees; slerp stays on the sphere (White, 2016).
    """
    omega = torch.acos((F.normalize(z0, dim=-1) * F.normalize(z1, dim=-1)).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7))
    omega = omega.view(-1, 1, 1)
    t = t.view(1, -1, 1)
    sin_omega = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / sin_omega) * z0.unsqueeze(1) + \
           (torch.sin(t * omega) / sin_omega) * z1.unsqueeze(1)


def path_smoothness(path_feats):
    """Coefficient of variation of step sizes along interpolation paths (P, S, F) -> float.

    For each path we measure the feature-space distance between consecutive
    frames. A smooth latent space changes at a steady rate (all steps similar
    -> low CV); a "jumpy" one stays put and then changes abruptly (high CV).
    A simple proxy inspired by perceptual path length (Karras et al., 2019).
    Lower = smoother.
    """
    steps = (path_feats[:, 1:] - path_feats[:, :-1]).norm(dim=-1)  # (P, S-1)
    cv = steps.std(dim=1) / steps.mean(dim=1).clamp_min(1e-12)
    return cv.mean().item()
