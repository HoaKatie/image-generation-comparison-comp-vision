"""Real-vs-real reference values: the best score each metric can reach.

Generative-model metrics have no absolute scale (is an FID of 5 good or bad?). To give
them one, we run the SAME metric code on two sets of REAL images: 10,000 random
training images play the role of "generated" samples and are compared with the 10,000
test images that src.evaluate uses as the reference. A perfect generator would score
about this; FID is not exactly 0 because of finite-sample noise. src.analysis adds the
result as a "Real data" row to the main table.

    python -m src.real_baseline                       # dataset from configs/shared.yaml

Writes runs/real_baseline_<dataset>/metrics.json. Needs the evaluation classifier.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from src import metrics as M
from src.config import load_config
from src.evaluate import Evaluator, batched
from src.utils import get_device, save_json


def baseline_path(dataset, output_dir):
    return Path(output_dir) / f"real_baseline_{dataset}" / "metrics.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/shared.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--n-samples", type=int, default=10000, help="same as src.evaluate")
    parser.add_argument("--feature-spaces", nargs="+", default=["classifier", "inception"],
                        choices=["classifier", "inception"])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    device = get_device(cfg["device"])
    ev = Evaluator(cfg, SimpleNamespace(eval_seed=0), device)

    n = min(args.n_samples, len(ev.test_x))
    pick = torch.randperm(len(ev.train_x), generator=torch.Generator().manual_seed(0))[:n]
    stand_in = ev.train_x[pick]  # real training images standing in for "generated" ones

    res = {"run": f"real_baseline_{cfg['dataset']}", "model": "real", "dataset": cfg["dataset"],
           "n_samples": n, "eval_classifier_accuracy": ev.classifier_acc}
    for space in args.feature_spaces:
        f_ref = ev.real_features(space, n)
        f_other = batched(ev.extractor(space), stand_in, device)
        res[f"fid_{space}"] = M.frechet_distance(f_ref.numpy(), f_other.numpy())
        res[f"precision_{space}"], res[f"recall_{space}"] = M.precision_recall(f_ref, f_other)

    confidence, pred = batched(ev.classifier, stand_in, device).softmax(dim=1).max(dim=1)
    res.update(M.mode_coverage(pred.numpy(), ev.num_classes, ev.test_y.numpy()))
    res["mean_confidence"] = confidence.mean().item()
    res["laplacian_var"] = M.laplacian_variance(stand_in).mean().item()
    res["hf_ratio"] = M.high_frequency_ratio(stand_in).mean().item()

    path = baseline_path(cfg["dataset"], cfg["output_dir"])
    save_json(res, path)
    print("\n== real data: train vs. test ==")
    for key in ("fid_classifier", "precision_classifier", "recall_classifier", "fid_inception",
                "precision_inception", "recall_inception", "modes_covered", "class_kl"):
        if key in res:
            print(f"  {key:22s} {res[key]:.4f}")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
