"""Tkinter window for one VAE vs DCGAN comparison.

Layout (top to bottom):
    title + one-line summary
    real test images of the chosen class           (the reference)
    VAE panel | DCGAN panel                        (best sample, large + first 8 samples)
    statistics table | "What the numbers say" text (same data as demo/compare_rules.py)
    buttons: New samples · Save results · Close

Keyboard shortcuts: N = new samples, S = save, Esc or Q = close.
Closing the window returns to the category menu in the terminal.
"""
import re
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import torch
from PIL import Image, ImageTk
from torchvision.utils import make_grid

from demo import compare_rules as rules
from demo.engine import ROOT
from src.utils import save_grid, save_json, to_unit_range

COLORS = {"VAE": "#1d4ed8", "DCGAN": "#c2410c", "Tie": "#374151", "n/a": "#6b7280"}
BEST_SCALE = 6      # best sample: 32 px -> 192 px
GRID_SCALE = 2      # small grids: 32 px -> 64 px per image
N_GRID = 8          # images in each small model grid
N_REAL = 12         # real images in the reference strip
OUTPUT_DIR = ROOT / "demo" / "outputs"   # "Save results" writes here (git-ignored)


def to_photo(images, scale, nrow=None):
    """Tensor image(s) in [-1, 1] -> Tk image, enlarged with NEAREST so no smoothing is added.

    (Smooth upscaling would blur the DCGAN images and hide the very sharpness difference
    we want to show.)
    """
    if images.dim() == 3:                       # one image (C, H, W)
        grid = to_unit_range(images)
    else:                                       # a batch (B, C, H, W) -> one grid image
        grid = make_grid(to_unit_range(images), nrow=nrow or len(images), padding=2, pad_value=1.0)
    array = (grid[0] * 255).round().to(torch.uint8).cpu().numpy()
    image = Image.fromarray(array)              # 2-D uint8 array -> grayscale image
    image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    return ImageTk.PhotoImage(image)


class ComparisonWindow:
    def __init__(self, engine, comparison):
        self.engine = engine
        self.comp = comparison
        self.root = tk.Tk()
        self.root.title("VAE vs DCGAN demo")
        self.photos = []   # Tk forgets images that Python garbage-collects, so keep references here

        base = tkfont.nametofont("TkDefaultFont")
        self.font_title = base.copy()
        self.font_title.configure(size=base.cget("size") + 6, weight="bold")
        self.font_bold = base.copy()
        self.font_bold.configure(weight="bold")

        self._build()
        self.show(comparison)
        self._bring_to_front()

    # ------------------------------------------------------------------ layout
    def _build(self):
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)
        pad = {"padx": 10, "pady": 4}

        self.title = ttk.Label(root, font=self.font_title)
        self.title.grid(row=0, column=0, sticky="w", **pad)
        self.subtitle = ttk.Label(root, wraplength=1050, justify="left")
        self.subtitle.grid(row=1, column=0, sticky="w", padx=10)

        # --- images ------------------------------------------------------
        images = ttk.Frame(root)
        images.grid(row=2, column=0, sticky="ew", **pad)
        images.columnconfigure((0, 1), weight=1)

        real = ttk.LabelFrame(images, text="Real test images of this class (reference)")
        real.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.real_label = ttk.Label(real)
        self.real_label.pack(padx=6, pady=4)

        self.panels = {}
        for col, name in enumerate(rules.MODELS):
            header = tk.Label(images, font=self.font_bold, fg=COLORS[name])
            frame = ttk.LabelFrame(images, labelwidget=header)
            frame.grid(row=1, column=col, sticky="nsew", padx=(0 if col == 0 else 6, 0))
            best = ttk.Label(frame)
            best.grid(row=0, column=0, rowspan=2, padx=6, pady=4)
            best_caption = ttk.Label(frame, justify="center")
            best_caption.grid(row=2, column=0)
            grid = ttk.Label(frame)
            grid.grid(row=0, column=1, padx=6, pady=4, sticky="s")
            ttk.Label(frame, text=f"First {N_GRID} samples (typical case, not cherry-picked)").grid(
                row=1, column=1, sticky="n")
            self.panels[name] = {"header": header, "best": best, "caption": best_caption, "grid": grid}

        # --- statistics table + explanation ------------------------------
        bottom = ttk.Frame(root)
        bottom.grid(row=3, column=0, sticky="nsew", **pad)
        bottom.columnconfigure(1, weight=1)
        bottom.rowconfigure(0, weight=1)

        table_frame = ttk.LabelFrame(bottom, text="Statistics")
        table_frame.grid(row=0, column=0, sticky="nsew")
        columns = ("metric", "VAE", "DCGAN", "real", "winner")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=len(rules.METRICS))
        for column, title, width, anchor in (("metric", "Metric", 250, "w"), ("VAE", "VAE", 75, "e"),
                                             ("DCGAN", "DCGAN", 75, "e"), ("real", "Real", 75, "e"),
                                             ("winner", "Winner", 70, "center")):
            self.table.heading(column, text=title)
            self.table.column(column, width=width, anchor=anchor, stretch=False)
        for name, color in COLORS.items():
            self.table.tag_configure(name, foreground=color)
        self.table.pack(padx=6, pady=(4, 0))
        ttk.Label(table_frame, justify="left", wraplength=540, text=(
            f"↑ higher is better (tie if within {rules.TIE_POINTS * 100:.0f} percentage points).\n"
            f"≈ real: the model closest to the Real column wins (tie if within {rules.TIE_MARGIN:.0%})."
        )).pack(padx=6, pady=4, anchor="w")

        text_frame = ttk.LabelFrame(bottom, text="What the numbers say")
        text_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        self.text = tk.Text(text_frame, wrap="word", width=60, height=11, relief="flat",
                            padx=8, pady=6, spacing3=6, font=tkfont.nametofont("TkDefaultFont"))
        scroll = ttk.Scrollbar(text_frame, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.text.tag_configure("heading", font=self.font_bold, spacing1=4)

        # --- buttons -----------------------------------------------------
        buttons = ttk.Frame(root)
        buttons.grid(row=4, column=0, sticky="ew", padx=10, pady=(4, 10))
        ttk.Button(buttons, text="New samples (N)", command=self.new_samples).pack(side="left")
        ttk.Button(buttons, text="Save results (S)", command=self.save).pack(side="left", padx=6)
        ttk.Button(buttons, text="Close, pick another category (Esc)", command=self.root.destroy).pack(side="left")
        self.status = ttk.Label(buttons, foreground="#4b5563")
        self.status.pack(side="left", padx=12)

        root.bind("<KeyPress-n>", lambda _: self.new_samples())
        root.bind("<KeyPress-s>", lambda _: self.save())
        root.bind("<Escape>", lambda _: self.root.destroy())
        root.bind("<KeyPress-q>", lambda _: self.root.destroy())

    def _bring_to_front(self):
        # Without this the window can open BEHIND the terminal (especially on macOS).
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(300, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()

    # ------------------------------------------------------------------ content
    def show(self, comp):
        """Fill every widget with the data of `comp` (also used after 'New samples')."""
        self.comp = comp
        self.photos.clear()
        self.root.title(f"VAE vs DCGAN: {comp.class_name}")
        self.title.configure(text=f"Category {comp.class_idx + 1}: {comp.class_name}")
        self.subtitle.configure(text=(
            f"Each model generated {comp.pool_size:,} images from random noise (seed {comp.seed}). "
            f"The classifier kept {comp.vae.n_found:,} VAE and {comp.gan.n_found:,} DCGAN images of this "
            f"class; the statistics compare n = {comp.n_compared} images per model."))

        self.real_label.configure(image=self._photo(comp.real_images[:N_REAL], GRID_SCALE))
        for name, result in (("VAE", comp.vae), ("DCGAN", comp.gan)):
            panel = self.panels[name]
            panel["header"].configure(text=f" {name}  ({result.description}) ")
            if result.n_found == 0:
                panel["best"].configure(image="", text=f"No {comp.class_name}\nin {comp.pool_size:,} samples")
                panel["grid"].configure(image="")
                panel["caption"].configure(text="")
                continue
            panel["best"].configure(image=self._photo(result.best_image, BEST_SCALE), text="")
            panel["caption"].configure(
                text=f"Best case (classifier confidence {result.confidence.max().item():.0%})")
            panel["grid"].configure(image=self._photo(result.images[:N_GRID], GRID_SCALE, nrow=4))

        self.table.delete(*self.table.get_children())
        for row in rules.table_rows(comp):
            self.table.insert("", "end", values=row, tags=(row[-1],))

        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        for heading, paragraph in rules.explain(comp):
            self.text.insert("end", heading + "\n", "heading")
            self.text.insert("end", paragraph + "\n")
        self.text.configure(state="disabled")   # read-only, but still selectable / scrollable

    def _photo(self, images, scale, nrow=None):
        photo = to_photo(images, scale, nrow)
        self.photos.append(photo)
        return photo

    # ------------------------------------------------------------------ buttons
    def new_samples(self):
        """Same category, next seed: shows how stable the numbers are."""
        self.status.configure(text="Generating new samples ...")
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            self.show(self.engine.compare(self.comp.class_idx, self.comp.seed + 1))
            self.status.configure(text=f"Seed {self.comp.seed}")
        finally:
            self.root.configure(cursor="")

    def save(self):
        """Write the images, the statistics and the explanation to demo/outputs/<class>_seed<k>/."""
        comp = self.comp
        slug = re.sub(r"[^a-z0-9]+", "_", comp.class_name.lower()).strip("_")
        out = OUTPUT_DIR / f"{slug}_seed{comp.seed}"
        out.mkdir(parents=True, exist_ok=True)
        save_grid(comp.real_images[:64], out / "real.png")
        for name, result in (("vae", comp.vae), ("gan", comp.gan)):
            if result.n_found:
                save_grid(result.images[:64], out / f"{name}_samples.png")
                save_grid(result.best_image.unsqueeze(0), out / f"{name}_best.png", nrow=1)
        save_json({
            "category": comp.class_name, "seed": comp.seed, "pool_size": comp.pool_size,
            "n_compared": comp.n_compared, "found": {"VAE": comp.vae.n_found, "DCGAN": comp.gan.n_found},
            "stats": {"VAE": comp.vae.stats, "DCGAN": comp.gan.stats, "Real": comp.real_stats},
            "winners": rules.winners(comp), "score": rules.score(comp),
        }, out / "stats.json")
        (out / "comparison.txt").write_text(
            "\n\n".join(f"{heading}\n{paragraph}" for heading, paragraph in rules.explain(comp)) + "\n")
        self.status.configure(text=f"Saved to {out.relative_to(ROOT)}")

    def run(self):
        self.root.mainloop()


def show_comparison(engine, comparison):
    """Open the window and block until it is closed."""
    ComparisonWindow(engine, comparison).run()
