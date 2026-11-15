"""Train the evaluation-only classifier (run once per dataset, before src.evaluate).

Usage:
    python -m src.train_classifier --config configs/classifier.yaml
"""
import argparse

import torch
import torch.nn.functional as F

from src.config import load_config
from src.data import dataset_info, get_loader, load_split
from src.models import EvalClassifier
from src.utils import get_device, run_dir_for, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train the evaluation classifier")
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    return parser.parse_args()


@torch.no_grad()
def accuracy(model, images, labels, device, batch=1000):
    model.eval()
    correct = 0
    for x, y in zip(images.split(batch), labels.split(batch)):
        correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    return correct / len(labels)


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    info = dataset_info(cfg["dataset"])

    loader = get_loader(cfg["dataset"], "train", cfg["batch_size"], cfg["num_workers"],
                        cfg["data_root"], cfg["subset"], seed=cfg["seed"])
    test_x, test_y = load_split(cfg["dataset"], "test", cfg["data_root"])

    model = EvalClassifier(info["channels"], info["num_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for x, y in loader:
            loss = F.cross_entropy(model(x.to(device)), y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        acc = accuracy(model, test_x, test_y, device)
        print(f"[classifier] epoch {epoch}/{cfg['epochs']}  test accuracy = {acc:.4f}")

    out_dir = run_dir_for(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "test_accuracy": acc}, out_dir / "classifier.pt")
    save_json({"dataset": cfg["dataset"], "epochs": cfg["epochs"], "test_accuracy": acc}, out_dir / "metrics.json")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
