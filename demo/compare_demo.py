"""VAE vs DCGAN interactive demo: pick a clothing category in the terminal, compare both models in a window.

HOW TO RUN (from the repository root, with the project's virtual environment):

    python demo/compare_demo.py                  # menu in the terminal -> window per category
    python demo/compare_demo.py --category 10    # skip the menu: show "Ankle boot" once, then exit
    python demo/compare_demo.py --no-gui         # terminal only (no Tkinter needed)
    python demo/compare_demo.py --help           # all options

WHAT TO NOTE
  * Nothing is trained here. It loads the final checkpoints that are already committed:
        runs/vae_fashion_mnist_z32_beta1_seed0/checkpoints/epoch_030.pt
        runs/gan_fashion_mnist_z32_seed0/checkpoints/epoch_030.pt
        runs/classifier_fashion_mnist/classifier.pt      (evaluation classifier)
  * Fashion-MNIST is downloaded to data/ on the first run (needs internet once).
  * Both models are unconditional: they cannot be ASKED for a class. The demo generates
    --pool images per model and keeps the ones the classifier labels as your category
    (see demo/engine.py). The comparison sentences come from simple if/else rules in
    demo/compare_rules.py.
  * Same --seed => same images and numbers every time, so you can rehearse the demo.
    "New samples" in the window moves to the next seed.
  * Close the window to get back to the menu. On macOS, click the terminal again before
    typing the next number.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # lets `python demo/compare_demo.py` import `src` and `demo`

from demo import compare_rules as rules  # noqa: E402  (must come after the sys.path line)
from demo.engine import DEFAULT_GAN, DEFAULT_VAE, DemoEngine  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vae", default=DEFAULT_VAE, help="VAE checkpoint (default: main run, seed 0, epoch 30)")
    parser.add_argument("--gan", default=DEFAULT_GAN, help="DCGAN checkpoint (default: main run, seed 0, epoch 30)")
    parser.add_argument("--pool", type=int, default=10000, help="images generated per model before filtering")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random noise vectors")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--category", type=int, default=None, help="category number: skip the menu, run once")
    parser.add_argument("--no-gui", action="store_true", help="print the comparison in the terminal only")
    return parser.parse_args()


def ask_category(names):
    """Show the numbered menu and return a 0-based class index, or None to quit."""
    print("\nWhich category should both models generate?")
    for i, name in enumerate(names, start=1):
        print(f"  {i:2d}. {name}")
    print("   0. Quit")
    while True:
        try:
            answer = input(f"Type a number (0-{len(names)}): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if answer in ("0", "q", "quit", "exit"):
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return int(answer) - 1
        print(f"  '{answer}' is not a valid choice. Please type a number from the list.")


def print_comparison(comp, full):
    """Terminal version of the window: the statistics table, then the verdict (or every paragraph)."""
    print(f"\n=== Category {comp.class_idx + 1}: {comp.class_name}  "
          f"(seed {comp.seed}, n = {comp.n_compared} images per model) ===")
    print(f"  {'Metric':42s} {'VAE':>8s} {'DCGAN':>8s} {'Real':>8s}  Winner")
    for label, vae, gan, real, won in rules.table_rows(comp):
        print(f"  {label:42s} {vae:>8s} {gan:>8s} {real:>8s}  {won}")
    for heading, paragraph in rules.explain(comp):
        if full or heading == "Verdict":
            print(f"\n{heading}\n  {paragraph}")


def main():
    args = parse_args()
    if not args.no_gui:
        try:
            from demo.gui import show_comparison   # imported here so --no-gui works without Tkinter
        except ImportError as err:
            raise SystemExit(f"Tkinter is not available ({err}). Install it (see demo/README.md) "
                             f"or run with --no-gui.")

    print("Loading the trained VAE, DCGAN and evaluation classifier ...")
    engine = DemoEngine(args.vae, args.gan, pool_size=args.pool, device=args.device)
    print(f"Ready (device: {engine.device}, dataset: {engine.dataset}).")

    while True:
        if args.category is not None:
            if not 1 <= args.category <= len(engine.class_names):
                raise SystemExit(f"--category must be between 1 and {len(engine.class_names)}")
            class_idx = args.category - 1
        else:
            class_idx = ask_category(engine.class_names)
            if class_idx is None:
                break

        print(f"Generating {args.pool:,} images per model and keeping the '{engine.class_names[class_idx]}' ones ...")
        comparison = engine.compare(class_idx, seed=args.seed)
        print_comparison(comparison, full=args.no_gui)
        if not args.no_gui:
            print("\nThe comparison window is open. Close it to pick another category.")
            show_comparison(engine, comparison)

        if args.category is not None:   # one-shot mode
            break
    print("Bye!")


if __name__ == "__main__":
    main()
