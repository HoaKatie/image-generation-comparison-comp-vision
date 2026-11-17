"""The non-GUI part of the demo: load the trained models, make images of one class, measure them.

Our VAE and DCGAN are UNCONDITIONAL: they map random noise z to "some clothing item"
and we cannot ask them for a specific class. To show e.g. only ankle boots we:

    1. generate a large pool of images from each model           (default 10,000)
    2. label every image with the evaluation classifier           (runs/classifier_fashion_mnist)
    3. keep only the images labelled with the chosen class
    4. compare both models on the SAME number n of kept images    (fair sample size)

How often a model lands on the chosen class ("share") is itself a result: the real
test set has 10 % of every class, so a share far from 10 % means the model over- or
under-produces that kind of clothing (mode coverage, hypothesis H2).

Nothing here trains or changes the models. Every metric reuses src/metrics.py, the
same code that produced the numbers in the report.
"""
from dataclasses import dataclass
from pathlib import Path

import torch

from src import metrics as M
from src.data import class_names, load_split
from src.evaluate import batched, generate
from src.models import load_checkpoint, load_classifier
from src.utils import get_device

ROOT = Path(__file__).resolve().parents[1]           # repository root
DEFAULT_VAE = ROOT / "runs/vae_fashion_mnist_z32_beta1_seed0/checkpoints/epoch_030.pt"
DEFAULT_GAN = ROOT / "runs/gan_fashion_mnist_z32_seed0/checkpoints/epoch_030.pt"

MAX_COMPARE = 500   # at most this many images per model go into the statistics
MIN_FOR_PR = 10     # precision/recall (k=3 nearest neighbours) needs at least this many images


@dataclass
class ModelResult:
    """What one generative model produced for the chosen class."""
    name: str               # "VAE" or "DCGAN"
    description: str        # e.g. "beta=1, z=32, epoch 30"
    images: torch.Tensor    # kept images of the class, in GENERATION order (not cherry-picked), [-1, 1]
    confidence: torch.Tensor  # classifier probability of the class, one per kept image
    n_found: int            # how many of the pool were labelled as the class
    stats: dict             # see DemoEngine._stats

    @property
    def best_image(self):
        """The sample the classifier is MOST sure about: the model's best case for this class."""
        return self.images[self.confidence.argmax()]


@dataclass
class Comparison:
    """Everything the GUI (or the terminal) needs to show one comparison."""
    class_idx: int
    class_name: str
    seed: int
    pool_size: int
    n_compared: int          # images per model used for the statistics (same for both models)
    classifier_accuracy: float
    real_images: torch.Tensor
    real_stats: dict
    vae: ModelResult
    gan: ModelResult


class DemoEngine:
    """Loads both models, the evaluation classifier and the real test set ONCE, then answers
    `compare(class_idx)` requests quickly (a few seconds on a laptop CPU)."""

    def __init__(self, vae_ckpt=DEFAULT_VAE, gan_ckpt=DEFAULT_GAN, pool_size=10000, device="auto"):
        self.device = get_device(device)
        self.pool_size = pool_size
        self.vae, vae_cfg, vae_meta = self._load(vae_ckpt, "vae")
        self.gan, gan_cfg, gan_meta = self._load(gan_ckpt, "gan")
        if vae_cfg["dataset"] != gan_cfg["dataset"]:
            raise SystemExit("The two checkpoints were trained on different datasets.")
        self.dataset = vae_cfg["dataset"]
        self.latent_dims = {"VAE": vae_cfg["latent_dim"], "DCGAN": gan_cfg["latent_dim"]}
        self.descriptions = {
            "VAE": f"beta={vae_cfg['beta']:g}, z={vae_cfg['latent_dim']}, epoch {vae_meta['epoch']}",
            "DCGAN": f"z={gan_cfg['latent_dim']}, epoch {gan_meta['epoch']}",
        }

        # runs/<run>/checkpoints/epoch_030.pt  ->  runs/  (where the classifier also lives)
        output_dir = Path(vae_ckpt).resolve().parents[2]
        self.classifier, self.classifier_acc = load_classifier(self.dataset, output_dir, self.device)

        data_root = ROOT / "data"   # downloaded automatically the first time
        self.class_names = class_names(self.dataset, data_root)
        self.real_x, self.real_y = load_split(self.dataset, "test", data_root)
        # Classify / embed the real test set once; every comparison reuses it.
        self.real_probs = batched(self.classifier, self.real_x, self.device).softmax(dim=1)
        self.real_feats = batched(self.classifier.embed, self.real_x, self.device)

    def _load(self, path, expected):
        path = Path(path)
        if not path.exists():
            raise SystemExit(f"Checkpoint not found: {path}\n"
                             "Run `git pull` (checkpoints are committed in runs/) or pass --vae / --gan.")
        model, cfg, meta = load_checkpoint(path, self.device)
        if cfg["model"] != expected:
            raise SystemExit(f"{path} is a {cfg['model'].upper()} checkpoint, expected a {expected.upper()} one.")
        return model, cfg, meta

    # ------------------------------------------------------------------ #
    def compare(self, class_idx, seed=0):
        # Both models get the SAME latent vectors (same seed); their latent spaces are
        # unrelated, so this only makes the run reproducible, it does not pair images.
        kept = {name: self._sample_class(model, name, class_idx, seed)
                for name, model in (("VAE", self.vae), ("DCGAN", self.gan))}

        is_class = self.real_y == class_idx
        real_images = self.real_x[is_class]
        real_feats = self.real_feats[is_class]
        n = min(MAX_COMPARE, len(real_images), *(len(images) for images, _ in kept.values()))

        # Real reference: means do not depend on sample size, so use up to MAX_COMPARE real images
        # (stable even when a model found only a few images of this class).
        n_real = min(MAX_COMPARE, len(real_images))
        real_stats = self._stats(real_images[:n_real], real_feats[:n_real])
        real_stats["share"] = is_class.float().mean().item()
        real_stats["confidence"] = self.real_probs[is_class, class_idx].mean().item()
        real_stats["precision"] = real_stats["recall"] = None   # only defined for generated images

        results = {}
        for name, (images, confidence) in kept.items():
            feats = batched(self.classifier.embed, images[:n], self.device)
            stats = self._stats(images[:n], feats)
            stats["share"] = len(images) / self.pool_size
            stats["confidence"] = confidence[:n].mean().item() if n else 0.0
            if n >= MIN_FOR_PR:
                stats["precision"], stats["recall"] = M.precision_recall(real_feats[:n], feats)
            else:
                stats["precision"] = stats["recall"] = None
            results[name] = ModelResult(name, self.descriptions[name], images, confidence, len(images), stats)

        return Comparison(class_idx, self.class_names[class_idx], seed, self.pool_size, n,
                          self.classifier_acc, real_images, real_stats, results["VAE"], results["DCGAN"])

    @torch.no_grad()
    def _sample_class(self, model, name, class_idx, seed):
        """Steps 1-3: generate a pool, classify it, keep the images of `class_idx`."""
        pool = generate(model, self.pool_size, self.latent_dims[name], self.device, seed)
        probs = batched(self.classifier, pool, self.device).softmax(dim=1)
        keep = probs.argmax(dim=1) == class_idx
        return pool[keep], probs[keep, class_idx]

    @staticmethod
    def _stats(images, feats):
        """Image-level statistics shared by generated and real images (all from src/metrics.py)."""
        if len(images) == 0:
            return {"sharpness": 0.0, "detail": 0.0, "spread": 0.0}
        return {
            # Laplacian variance: edge strength. Blurry images -> low value.
            "sharpness": M.laplacian_variance(images).mean().item(),
            # Share of spectral energy at high frequencies: fine detail / texture.
            "detail": M.high_frequency_ratio(images).mean().item(),
            # Mean pairwise distance in classifier feature space: how different the images
            # of this class are from each other (low = samples look alike).
            "spread": torch.pdist(feats).mean().item() if len(feats) > 1 else 0.0,
        }
