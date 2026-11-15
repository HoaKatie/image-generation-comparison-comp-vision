"""Small helpers shared by the training and evaluation scripts."""
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.utils import save_image


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS


def get_device(preferred="auto"):
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_name(cfg):
    """Unique, human-readable name for one training run, e.g. vae_fashion_mnist_z32_beta1_seed0."""
    if cfg["model"] == "classifier":
        return f"classifier_{cfg['dataset']}"
    name = f"{cfg['model']}_{cfg['dataset']}_z{cfg['latent_dim']}"
    if cfg["model"] == "vae":
        name += f"_beta{float(cfg['beta']):g}"
    return f"{name}_seed{cfg['seed']}"


def run_dir_for(cfg):
    return Path(cfg["output_dir"]) / run_name(cfg)


def prepare_run_dir(cfg):
    """Create a fresh run directory (old results of the SAME run are replaced)."""
    run_dir = run_dir_for(cfg)
    if run_dir.exists():
        print(f"Note: replacing previous results in {run_dir}")
        shutil.rmtree(run_dir)
    (run_dir / "samples").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    return run_dir


def fixed_noise(n, latent_dim, seed=1234):
    """The same latent vectors every epoch, so sample grids are comparable over time."""
    return torch.randn(n, latent_dim, generator=torch.Generator().manual_seed(seed))


def to_unit_range(x):
    """[-1, 1] (model space) -> [0, 1] (image space)."""
    return (x.clamp(-1, 1) + 1) / 2


def save_grid(images, path, nrow=8):
    save_image(to_unit_range(images.detach().cpu()), path, nrow=nrow, padding=2)


def append_csv(path, row):
    path = Path(path)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _to_builtin(obj):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    raise TypeError(f"Cannot serialise {type(obj)}")


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_builtin)
