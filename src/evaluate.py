"""Evaluate a trained VAE or GAN.

Uses the same --config / --set flags as training, so the run directory is found
automatically:
    python -m src.evaluate --config configs/gan.yaml --set seed=1
    python -m src.evaluate --config configs/vae.yaml --all-epochs --quick     # learning curve

Writes runs/<run>/eval/epoch_XXX/metrics.json (or epoch_XXX_quick/ with --quick), plus these
figures (unless --quick):
    samples.png            64 random samples from a fixed seed (NOT cherry-picked)
    most_confident.png     samples the evaluation classifier is most sure about   (successes)
    least_confident.png    samples the evaluation classifier is least sure about  (failure cases)
    interpolation.png      slerp between random latent pairs, one pair per row
    nearest_neighbors.png  top row: generated sample, bottom row: closest TRAINING image
    reconstructions.png    VAE only. Top row: real test image, bottom row: reconstruction
"""
import argparse
import time
from pathlib import Path

import torch

from src import metrics as M
from src.config import load_config
from src.data import dataset_info, load_split
from src.models import load_checkpoint, load_classifier
from src.utils import get_device, run_dir_for, run_name, save_grid, save_json

QUICK_SAMPLES = 5000      # --quick: fewer samples, classifier feature space only, no figures
N_MEMORIZATION = 1000     # generated / test images checked against the whole training set
N_PATHS, N_STEPS = 100, 10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="same overrides as training")
    parser.add_argument("--epoch", type=int, default=None, help="checkpoint epoch to evaluate (default: last)")
    parser.add_argument("--all-epochs", action="store_true", help="evaluate every saved checkpoint")
    parser.add_argument("--quick", action="store_true", help=f"{QUICK_SAMPLES} samples, classifier features only, no figures")
    parser.add_argument("--n-samples", type=int, default=10000, help="capped at the test-set size")
    parser.add_argument("--feature-spaces", nargs="+", default=["classifier", "inception"],
                        choices=["classifier", "inception"])
    parser.add_argument("--eval-seed", type=int, default=0, help="seed for the latent vectors used in evaluation")
    return parser.parse_args()


@torch.no_grad()
def batched(fn, x, device, batch=500):
    """Apply fn to x in mini-batches on `device`; results are gathered on the CPU."""
    return torch.cat([fn(chunk.to(device)).cpu() for chunk in x.split(batch)])


@torch.no_grad()
def generate(model, n, latent_dim, device, seed):
    z = torch.randn(n, latent_dim, generator=torch.Generator().manual_seed(seed))
    return batched(model.decode, z, device)


class InceptionFeatures:
    """2048-d pool features of the standard FID Inception-v3 (weights via torchmetrics / torch-fidelity)."""

    def __init__(self, device):
        from torchmetrics.image.fid import NoTrainInceptionV3
        self.net = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()

    def __call__(self, x):
        x = (x.clamp(-1, 1) + 1) / 2
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  # grayscale -> RGB
        return self.net((x * 255).round().to(torch.uint8)).float()


class Evaluator:
    """Everything shared across checkpoints: real data, the evaluation classifier, real-image features."""

    def __init__(self, cfg, args, device):
        self.cfg, self.args, self.device = cfg, args, device
        self.num_classes = dataset_info(cfg["dataset"])["num_classes"]
        self.test_x, self.test_y = load_split(cfg["dataset"], "test", cfg["data_root"])
        self._train_x = None
        self.classifier, self.classifier_acc = load_classifier(cfg["dataset"], cfg["output_dir"], device)
        self._extractors = {"classifier": self.classifier.embed}
        self._real_feats = {}

    @property
    def train_x(self):
        if self._train_x is None:
            self._train_x, _ = load_split(self.cfg["dataset"], "train", self.cfg["data_root"])
        return self._train_x

    def extractor(self, space):
        if space not in self._extractors:
            self._extractors[space] = InceptionFeatures(self.device)
        return self._extractors[space]

    def real_features(self, space, n):
        """Features of the first n real test images (Inception features are cached on disk)."""
        if (space, n) not in self._real_feats:
            cache = Path(self.cfg["output_dir"]) / "cache" / f"{space}_{self.cfg['dataset']}_test{n}.pt"
            if space == "inception" and cache.exists():
                feats = torch.load(cache, weights_only=True)
            else:
                feats = batched(self.extractor(space), self.test_x[:n], self.device)
                if space == "inception":
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(feats, cache)
            self._real_feats[(space, n)] = feats
        return self._real_feats[(space, n)]

    @torch.no_grad()
    def interpolation_paths(self, model, latent_dim):
        g = torch.Generator().manual_seed(self.args.eval_seed + 1)
        z0 = torch.randn(N_PATHS, latent_dim, generator=g)
        z1 = torch.randn(N_PATHS, latent_dim, generator=g)
        z = M.slerp(z0, z1, torch.linspace(0, 1, N_STEPS))  # (P, S, D)
        images = batched(model.decode, z.flatten(0, 1), self.device)
        return images.view(N_PATHS, N_STEPS, *images.shape[1:])

    def evaluate(self, ckpt_path):
        args, quick = self.args, self.args.quick
        model, cfg, meta = load_checkpoint(ckpt_path, self.device)
        n = min(QUICK_SAMPLES if quick else args.n_samples, len(self.test_x))
        # Quick (learning-curve) evaluations get their own folder so they never overwrite a full one.
        out_dir = Path(ckpt_path).parent.parent / "eval" / (f"epoch_{meta['epoch']:03d}" + ("_quick" if quick else ""))
        start = time.time()

        fake = generate(model, n, cfg["latent_dim"], self.device, args.eval_seed)
        real = self.test_x[:n]
        res = {
            "run": run_name(cfg), "model": cfg["model"], "dataset": cfg["dataset"],
            "latent_dim": cfg["latent_dim"], "beta": cfg.get("beta"), "seed": cfg["seed"],
            **meta, "n_samples": n, "quick": quick, "eval_classifier_accuracy": self.classifier_acc,
        }

        # 1. Fidelity & diversity in feature space(s)                                   (H1, H2)
        for space in (["classifier"] if quick else args.feature_spaces):
            f_real = self.real_features(space, n)
            f_fake = batched(self.extractor(space), fake, self.device)
            res[f"fid_{space}"] = M.frechet_distance(f_real.numpy(), f_fake.numpy())
            res[f"precision_{space}"], res[f"recall_{space}"] = M.precision_recall(f_real, f_fake)

        # 2. Mode coverage, using the evaluation classifier                              (H2)
        probs = batched(self.classifier, fake, self.device).softmax(dim=1)
        confidence, pred = probs.max(dim=1)
        res.update(M.mode_coverage(pred.numpy(), self.num_classes, self.test_y.numpy()))
        res["mean_confidence"] = confidence.mean().item()

        # 3. Sharpness, with real images as the reference                               (H1)
        res["laplacian_var"] = M.laplacian_variance(fake).mean().item()
        res["laplacian_var_real"] = M.laplacian_variance(real).mean().item()
        res["hf_ratio"] = M.high_frequency_ratio(fake).mean().item()
        res["hf_ratio_real"] = M.high_frequency_ratio(real).mean().item()

        # 4. Latent space smoothness                                                     (H3)
        paths = self.interpolation_paths(model, cfg["latent_dim"])
        path_feats = batched(self.classifier.embed, paths.flatten(0, 1), self.device)
        res["interp_step_cv"] = M.path_smoothness(path_feats.view(N_PATHS, N_STEPS, -1))

        # 5. Reconstruction -- VAE only (a GAN has no encoder; state this as a limitation)
        recon = None
        if cfg["model"] == "vae":
            recon = batched(model.reconstruct, real, self.device)
            res["recon_mse"] = (((recon - real) / 2) ** 2).mean().item()  # in [0, 1] units
            res["recon_psnr"] = M.psnr(recon, real).mean().item()

        if not quick:
            # 6. Memorisation: generated->train distance vs. unseen test->train distance
            d_fake, nn_idx = M.nearest_neighbors(fake[:N_MEMORIZATION], self.train_x)
            d_test, _ = M.nearest_neighbors(self.test_x[:N_MEMORIZATION], self.train_x)
            res["nn_dist_generated"] = d_fake.median().item()
            res["nn_dist_test"] = d_test.median().item()
            res["memorization_ratio"] = res["nn_dist_generated"] / res["nn_dist_test"]

            # 7. Figures for systematic qualitative evaluation
            self.save_figures(out_dir, fake, confidence, paths, nn_idx, recon)

        res["eval_time_sec"] = round(time.time() - start, 1)
        save_json(res, out_dir / "metrics.json")
        self.print_summary(res, out_dir)
        return res

    def save_figures(self, out_dir, fake, confidence, paths, nn_idx, recon):
        out_dir.mkdir(parents=True, exist_ok=True)
        save_grid(fake[:64], out_dir / "samples.png")
        order = confidence.argsort(descending=True)
        save_grid(fake[order[:32]], out_dir / "most_confident.png")
        save_grid(fake[order[-32:]], out_dir / "least_confident.png")
        save_grid(paths[:8].flatten(0, 1), out_dir / "interpolation.png", nrow=N_STEPS)
        save_grid(torch.cat([fake[:8], self.train_x[nn_idx[:8]]]), out_dir / "nearest_neighbors.png")
        if recon is not None:
            save_grid(torch.cat([self.test_x[:8], recon[:8]]), out_dir / "reconstructions.png")

    @staticmethod
    def print_summary(res, out_dir):
        keys = ["fid_classifier", "precision_classifier", "recall_classifier", "fid_inception",
                "precision_inception", "recall_inception", "modes_covered", "class_kl",
                "laplacian_var", "laplacian_var_real", "hf_ratio", "hf_ratio_real",
                "interp_step_cv", "recon_psnr", "memorization_ratio"]
        print(f"\n== {res['run']}  epoch {res['epoch']}  ({res['eval_time_sec']}s) ==")
        for key in keys:
            if key in res:
                value = res[key]
                print(f"  {key:22s} {value:.4f}" if isinstance(value, float) else f"  {key:22s} {value}")
        print(f"  -> {out_dir / 'metrics.json'}")


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    device = get_device(cfg["device"])

    ckpt_dir = run_dir_for(cfg) / "checkpoints"
    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not checkpoints:
        raise SystemExit(f"No checkpoints in {ckpt_dir}. Train this configuration first.")
    if args.all_epochs:
        selected = checkpoints
    elif args.epoch is not None:
        selected = [ckpt_dir / f"epoch_{args.epoch:03d}.pt"]
    else:
        selected = [checkpoints[-1]]

    evaluator = Evaluator(cfg, args, device)
    for ckpt in selected:
        evaluator.evaluate(ckpt)


if __name__ == "__main__":
    main()
