"""Train a DCGAN with the shared convolutional architecture.

Usage:
    python -m src.train_gan --config configs/gan.yaml
    python -m src.train_gan --config configs/gan.yaml --set seed=1 latent_dim=8

Outputs (runs/<run_name>/): same layout as src/train_vae.py. train_log.csv has
    loss_d, loss_g   discriminator / generator losses
    d_real, d_fake   mean D(x) and mean D(G(z)) -- useful to diagnose instability:
                     d_fake -> 0 means D wins completely and G stops learning.
"""
import argparse
import time

import torch

from src.config import load_config
from src.data import get_loader
from src.losses import discriminator_loss, generator_loss
from src.models import build_model, save_checkpoint
from src.models.networks import count_parameters, init_dcgan_weights
from src.utils import append_csv, fixed_noise, get_device, prepare_run_dir, save_grid, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DCGAN")
    parser.add_argument("--config", default="configs/gan.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    return parser.parse_args()


def train(cfg):
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    loader = get_loader(cfg["dataset"], "train", cfg["batch_size"], cfg["num_workers"],
                        cfg["data_root"], cfg["subset"], seed=cfg["seed"])

    model = build_model(cfg).to(device)
    model.apply(init_dcgan_weights)
    G, D = model.generator, model.discriminator
    betas = (cfg["adam_beta1"], 0.999)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg["lr"], betas=betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg["lr"], betas=betas)

    run_dir = prepare_run_dir(cfg)
    z_fixed = fixed_noise(64, cfg["latent_dim"]).to(device)
    print(f"[gan] {run_dir.name} | params={count_parameters(model):,} | device={device}")

    start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        sums = torch.zeros(4, device=device)
        for x, _ in loader:
            x = x.to(device)
            z = torch.randn(x.size(0), cfg["latent_dim"], device=device)
            fake = G(z)

            # (1) Discriminator step: push D(x) -> 1 and D(G(z)) -> 0.
            #     fake.detach(): this step must not update the generator.
            real_logits = D(x)
            fake_logits = D(fake.detach())
            loss_d = discriminator_loss(real_logits, fake_logits)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # (2) Generator step: fool the (just updated) discriminator.
            loss_g = generator_loss(D(fake))
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            sums += torch.stack([loss_d.detach(), loss_g.detach(),
                                 torch.sigmoid(real_logits).mean().detach(),
                                 torch.sigmoid(fake_logits).mean().detach()])

        loss_d, loss_g, d_real, d_fake = (sums / len(loader)).tolist()
        elapsed = time.time() - start
        append_csv(run_dir / "train_log.csv",
                   {"epoch": epoch, "loss_d": loss_d, "loss_g": loss_g, "d_real": d_real, "d_fake": d_fake,
                    "elapsed_sec": round(elapsed, 1)})
        print(f"[gan] epoch {epoch:3d}/{cfg['epochs']}  loss_d={loss_d:.3f}  loss_g={loss_g:.3f}  "
              f"D(x)={d_real:.2f}  D(G(z))={d_fake:.2f}  ({elapsed:.0f}s)")

        model.eval()
        with torch.no_grad():
            save_grid(model.decode(z_fixed), run_dir / "samples" / f"epoch_{epoch:03d}.png")
        if epoch % cfg["save_every"] == 0 or epoch == cfg["epochs"]:
            save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, cfg, epoch, elapsed)
    return run_dir


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    if cfg["model"] != "gan":
        raise SystemExit(f"{args.config} is a '{cfg['model']}' config, expected 'gan'")
    train(cfg)


if __name__ == "__main__":
    main()
