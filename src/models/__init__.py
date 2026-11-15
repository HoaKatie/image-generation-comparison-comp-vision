"""Model construction and checkpoint I/O."""
from pathlib import Path

import torch

from src.data import dataset_info
from src.models.classifier import EvalClassifier
from src.models.gan import GAN
from src.models.networks import count_parameters
from src.models.vae import VAE

__all__ = ["VAE", "GAN", "EvalClassifier", "build_model", "save_checkpoint", "load_checkpoint", "load_classifier"]


def build_model(cfg):
    channels = dataset_info(cfg["dataset"])["channels"]
    if cfg["model"] == "vae":
        return VAE(cfg["latent_dim"], channels, cfg["base_channels"])
    if cfg["model"] == "gan":
        return GAN(cfg["latent_dim"], channels, cfg["base_channels"])
    raise ValueError(f"Unknown generative model {cfg['model']!r}")


def save_checkpoint(path, model, cfg, epoch, train_time_sec):
    torch.save(
        {"state_dict": model.state_dict(), "config": cfg, "epoch": epoch, "train_time_sec": train_time_sec},
        path,
    )


def load_checkpoint(path, device):
    """Returns (model in eval mode, config, metadata)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    sampler = model.decoder if cfg["model"] == "vae" else model.generator
    meta = {
        "epoch": ckpt["epoch"],
        "train_time_sec": ckpt["train_time_sec"],
        "params_total": count_parameters(model),
        "params_decoder": count_parameters(sampler),
    }
    return model, cfg, meta


def classifier_path(dataset, output_dir):
    return Path(output_dir) / f"classifier_{dataset}" / "classifier.pt"


def load_classifier(dataset, output_dir, device):
    """Returns (classifier in eval mode, its test accuracy)."""
    path = classifier_path(dataset, output_dir)
    if not path.exists():
        raise SystemExit(
            f"Evaluation classifier not found at {path}.\n"
            f"Train it first:  python -m src.train_classifier --set dataset={dataset}"
        )
    ckpt = torch.load(path, map_location=device, weights_only=True)
    info = dataset_info(dataset)
    clf = EvalClassifier(info["channels"], info["num_classes"]).to(device)
    clf.load_state_dict(ckpt["state_dict"])
    clf.eval()
    return clf, ckpt["test_accuracy"]
