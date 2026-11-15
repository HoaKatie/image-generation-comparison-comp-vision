"""Train a VAE (or beta-VAE) with the shared convolutional architecture.

Usage:
    python -m src.train_vae --config configs/vae.yaml
    python -m src.train_vae --config configs/vae.yaml --set seed=1 beta=4 latent_dim=8

Outputs (runs/<run_name>/):
    config.yaml        exact settings used
    train_log.csv      per-epoch loss / recon / kl / elapsed time
    samples/           sample grid from FIXED latent vectors after every epoch
    checkpoints/       model every `save_every` epochs (for learning curves)
"""
import argparse
import time

import torch

from src.config import load_config
from src.data import get_loader
from src.losses import vae_loss
from src.models import build_model, save_checkpoint
from src.models.networks import count_parameters, init_dcgan_weights
from src.utils import append_csv, fixed_noise, get_device, prepare_run_dir, save_grid, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VAE")
    parser.add_argument("--config", default="configs/vae.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    return parser.parse_args()


def train(cfg):
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    loader = get_loader(cfg["dataset"], "train", cfg["batch_size"], cfg["num_workers"],
                        cfg["data_root"], cfg["subset"], seed=cfg["seed"])

    model = build_model(cfg).to(device)
    model.apply(init_dcgan_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    run_dir = prepare_run_dir(cfg)
    z_fixed = fixed_noise(64, cfg["latent_dim"]).to(device)
    print(f"[vae] {run_dir.name} | params={count_parameters(model):,} | device={device}")

    start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        sums = torch.zeros(3, device=device)
        for x, _ in loader:
            x = x.to(device)
            x_recon, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x, x_recon, mu, logvar, beta=cfg["beta"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sums += torch.stack([loss.detach(), recon.detach(), kl.detach()])

        loss, recon, kl = (sums / len(loader)).tolist()
        elapsed = time.time() - start
        append_csv(run_dir / "train_log.csv",
                   {"epoch": epoch, "loss": loss, "recon": recon, "kl": kl, "elapsed_sec": round(elapsed, 1)})
        print(f"[vae] epoch {epoch:3d}/{cfg['epochs']}  loss={loss:.3f}  recon={recon:.3f}  "
              f"kl={kl:.3f}  ({elapsed:.0f}s)")

        model.eval()
        with torch.no_grad():
            save_grid(model.decode(z_fixed), run_dir / "samples" / f"epoch_{epoch:03d}.png")
        if epoch % cfg["save_every"] == 0 or epoch == cfg["epochs"]:
            save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, cfg, epoch, elapsed)
    return run_dir


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    if cfg["model"] != "vae":
        raise SystemExit(f"{args.config} is a '{cfg['model']}' config, expected 'vae'")
    train(cfg)


if __name__ == "__main__":
    main()
