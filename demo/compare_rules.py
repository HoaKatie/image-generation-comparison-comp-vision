"""Turn the numbers of one Comparison into a table and plain-English sentences.

This is plain if/else logic on purpose, so anyone can read it during the demo or the Q&A.
Every threshold is a named constant below: change one and both the table and the text follow.

Two kinds of metric:
    "higher"  - higher is better               (classifier confidence, precision, recall)
    "closer"  - closest to the REAL value wins (sharpness, detail, spread, share)
                e.g. a model with MORE edge energy than real photos is not "sharper and better",
                it is noisy, so we reward matching the real data, not maximising the number.
"""

TIE_MARGIN = 0.10          # "closer" metrics: distances to real within 10 % of each other are a tie
TIE_POINTS = 0.05          # "higher" metrics (all percentages): within 5 percentage points is a tie
NEAR_REAL = 0.20           # within 20 % of the real value counts as "about the same as real"
LOW_RECALL = 0.25          # recall below 25 % = covers only a small part of the real variety
SHARE_LOW, SHARE_HIGH = 0.6, 1.4   # class share below 60 % / above 140 % of the real share

MODELS = ("VAE", "DCGAN")

# (stats key, table label, how to judge, value format)
METRICS = [
    ("share",      "Share of samples in class (≈ real)",   "closer", "{:.1%}"),
    ("confidence", "Classifier confidence (↑)",            "higher", "{:.1%}"),
    ("sharpness",  "Sharpness: Laplacian var. (≈ real)",   "closer", "{:.4f}"),
    ("detail",     "Fine detail: high-freq. energy (≈ real)", "closer", "{:.4f}"),
    ("precision",  "Fidelity: precision (↑)",              "higher", "{:.1%}"),
    ("recall",     "Diversity: recall (↑)",                "higher", "{:.1%}"),
    ("spread",     "Within-class spread (≈ real)",         "closer", "{:.2f}"),
]

# Fixed context from report/report.md (all classes, 3 seeds), shown after the per-class verdict.
PROJECT_RESULT = (
    "Project-level result (all 10 classes, 3 seeds, see report/report.md): DCGAN is sharper and "
    "more faithful (H1 confirmed) and, unexpectedly, also more diverse (H2 refuted: no mode "
    "collapse). The VAE has the smoother latent space (H3 confirmed; not shown in this demo)."
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def distance_to_real(value, real):
    """0 = identical to the real data, 0.3 = 30 % above or below it."""
    return abs(value / real - 1) if real else float("inf")


def fmt(value, pattern):
    return "—" if value is None else pattern.format(value)


def winner(vae, gan, mode, real=None):
    """Return "VAE", "DCGAN", "Tie" or "n/a" for one metric."""
    if vae is None or gan is None:
        return "n/a"
    if mode == "higher":
        if abs(vae - gan) < TIE_POINTS:
            return "Tie"
        return "VAE" if vae > gan else "DCGAN"
    # mode == "closer": compare how far each model is from the real value
    d_vae, d_gan = distance_to_real(vae, real), distance_to_real(gan, real)
    if abs(d_vae - d_gan) < TIE_MARGIN:
        return "Tie"
    return "VAE" if d_vae < d_gan else "DCGAN"


def winners(comp):
    """{metric key: winner} for every row of the table."""
    return {key: winner(comp.vae.stats[key], comp.gan.stats[key], mode, comp.real_stats[key])
            for key, _, mode, _ in METRICS}


def table_rows(comp):
    """Rows for the statistics table: (label, VAE, DCGAN, Real, winner)."""
    won = winners(comp)
    return [(label, fmt(comp.vae.stats[key], pattern), fmt(comp.gan.stats[key], pattern),
             fmt(comp.real_stats[key], pattern), won[key])
            for key, label, _, pattern in METRICS]


def score(comp):
    """How many metrics each side won: {"VAE": a, "DCGAN": b, "Tie": c} (n/a rows are skipped)."""
    counts = {"VAE": 0, "DCGAN": 0, "Tie": 0}
    for result in winners(comp).values():
        if result in counts:
            counts[result] += 1
    return counts


# --------------------------------------------------------------------------- #
# The explanation, one function per paragraph
# --------------------------------------------------------------------------- #
def explain(comp):
    """List of (heading, paragraph) pairs describing the comparison in plain English."""
    won = winners(comp)
    return [
        ("What you are looking at", _setup(comp)),
        ("Sharpness", _sharpness(comp, won)),
        ("Recognisability", _confidence(comp, won)),
        ("Fidelity vs diversity", _precision_recall(comp, won)),
        ("Variety inside the class", _spread(comp)),
        ("How often each model makes this class", _share(comp)),
        ("Verdict", _verdict(comp)),
        ("Caveats", _caveats(comp)),
    ]


def _setup(comp):
    return (f"Each model turned {comp.pool_size:,} random noise vectors into images. The evaluation "
            f"classifier labelled {comp.vae.n_found:,} VAE images and {comp.gan.n_found:,} DCGAN images "
            f"as '{comp.class_name}'. Statistics use the first n = {comp.n_compared} of each, the same "
            f"number for both models so the comparison is fair. The large picture is each model's best "
            f"case (highest classifier confidence); the small grid is its typical case (the first "
            f"images generated, not cherry-picked).")


def _sharpness(comp, won):
    real = comp.real_stats["sharpness"]
    vae, gan = comp.vae.stats["sharpness"], comp.gan.stats["sharpness"]
    vae_pct, gan_pct = vae / real, gan / real
    if abs(vae_pct - 1) < NEAR_REAL and abs(gan_pct - 1) < NEAR_REAL:
        text = (f"Both models are about as sharp as real {comp.class_name} images "
                f"(VAE {vae_pct:.0%}, DCGAN {gan_pct:.0%} of the real edge strength).")
    elif won["sharpness"] == "DCGAN":
        text = (f"DCGAN images are {gan / max(vae, 1e-12):.1f}x sharper than VAE images: DCGAN reaches "
                f"{gan_pct:.0%} of the real edge strength, the VAE only {vae_pct:.0%}. This is the typical "
                f"VAE blur: its pixel-wise reconstruction loss is lowest when it predicts the AVERAGE of all "
                f"plausible images, and an average is blurry. The DCGAN discriminator, in contrast, "
                f"rejects blurry images as fake.")
    elif won["sharpness"] == "VAE":
        text = (f"The VAE is closer to real sharpness here (VAE {vae_pct:.0%}, DCGAN {gan_pct:.0%} of real). "
                f"This is unusual for our models; check the images for noise in the DCGAN samples.")
    else:
        text = f"Both models have similar sharpness (VAE {vae_pct:.0%}, DCGAN {gan_pct:.0%} of real)."

    for name, pct in (("VAE", vae_pct), ("DCGAN", gan_pct)):
        if pct > 1 + NEAR_REAL:
            text += (f" {name} has MORE edge energy than real images, which usually means noise or "
                     f"checkerboard artefacts rather than real detail.")

    if won["detail"] == won["sharpness"] and won["detail"] in MODELS:
        text += f" The fine-detail (high-frequency) measure agrees: {won['detail']} is closer to real."
    elif won["detail"] in MODELS:
        text += f" The fine-detail (high-frequency) measure disagrees: there {won['detail']} is closer to real."
    return text


def _confidence(comp, won):
    vae, gan = comp.vae.stats["confidence"], comp.gan.stats["confidence"]
    real = comp.real_stats["confidence"]
    if won["confidence"] in MODELS:
        best, other = (("VAE", vae), ("DCGAN", gan)) if won["confidence"] == "VAE" else (("DCGAN", gan), ("VAE", vae))
        text = (f"{best[0]} images look more like a typical {comp.class_name} to the classifier "
                f"({best[1]:.0%} vs {other[1]:.0%} confidence; real images: {real:.0%}).")
    else:
        text = (f"The classifier recognises both models' images about equally well "
                f"(VAE {vae:.0%}, DCGAN {gan:.0%}; real images: {real:.0%}).")
    return text + (" Note: only images already labelled as this class are counted, so these numbers are "
                   "biased upwards. Compare the two models with each other, not with 100 %.")


def _precision_recall(comp, won):
    p, r = won["precision"], won["recall"]
    if p == "n/a":
        return (f"Too few images of this class (n < 10) to measure precision and recall. That a model "
                f"almost never produces {comp.class_name} images is itself a sign of a missing mode.")
    vs, gs = comp.vae.stats, comp.gan.stats
    numbers = (f"precision VAE {vs['precision']:.0%} / DCGAN {gs['precision']:.0%}, "
               f"recall VAE {vs['recall']:.0%} / DCGAN {gs['recall']:.0%}")
    if p == r and p in MODELS:
        text = (f"{p} wins on BOTH fidelity and diversity ({numbers}). For {comp.class_name} there is no "
                f"fidelity-vs-diversity trade-off: one model is simply better.")
        if p == "DCGAN":
            text += " This matches the project result where H2 (the GAN would be less diverse) was refuted."
    elif p in MODELS and r in MODELS:
        text = (f"Classic trade-off: {p} is more faithful (its images sit inside the real data more often), "
                f"while {r} covers more of the real variety ({numbers}).")
    elif p in MODELS:
        text = f"{p} is more faithful while diversity is about the same ({numbers})."
    elif r in MODELS:
        text = f"{r} covers more of the real variety while fidelity is about the same ({numbers})."
    else:
        text = f"Both models have similar fidelity and diversity for this class ({numbers})."

    for name, stats in (("VAE", vs), ("DCGAN", gs)):
        if stats["recall"] < LOW_RECALL:
            text += (f" {name}'s recall is only {stats['recall']:.0%}: its {comp.class_name} images cover a "
                     f"small part of what real ones look like (partial mode collapse inside this class).")
    return text


def _spread(comp):
    real = comp.real_stats["spread"]
    sentences = []
    for name, stats in (("VAE", comp.vae.stats), ("DCGAN", comp.gan.stats)):
        ratio = stats["spread"] / real if real else 0.0
        if ratio < 1 - NEAR_REAL:
            sentences.append(f"{name} images are less varied than real ones ({ratio:.0%} of the real spread): "
                             f"many samples look alike.")
        elif ratio > 1 + NEAR_REAL:
            sentences.append(f"{name} images are more spread out than real ones ({ratio:.0%} of the real "
                             f"spread): some are probably odd or borderline images.")
        else:
            sentences.append(f"{name} is about as varied as the real data ({ratio:.0%} of the real spread).")
    return " ".join(sentences)


def _share(comp):
    real = comp.real_stats["share"]
    sentences = []
    for name, stats in (("VAE", comp.vae.stats), ("DCGAN", comp.gan.stats)):
        share = stats["share"]
        if share < SHARE_LOW * real:
            sentences.append(f"{name} rarely makes {comp.class_name} images ({share:.1%} of its images vs "
                             f"{real:.0%} in the real data): it under-represents this mode.")
        elif share > SHARE_HIGH * real:
            sentences.append(f"{name} over-produces {comp.class_name} ({share:.1%} of its images vs "
                             f"{real:.0%} in the real data).")
        else:
            sentences.append(f"{name} makes {comp.class_name} images about as often as the real data does "
                             f"({share:.1%} vs {real:.0%}).")
    sentences.append("The models are unconditional, so this share is the only way they 'choose' a class; "
                     "it is a per-class view of mode coverage.")
    return " ".join(sentences)


def _verdict(comp):
    counts = score(comp)
    total = sum(counts.values())
    text = (f"Score for {comp.class_name}: DCGAN {counts['DCGAN']}, VAE {counts['VAE']}, "
            f"ties {counts['Tie']} (out of {total} metrics). ")
    if counts["DCGAN"] > counts["VAE"]:
        text += f"In this run DCGAN produces the more realistic {comp.class_name} images."
    elif counts["VAE"] > counts["DCGAN"]:
        text += f"In this run the VAE produces the more realistic {comp.class_name} images."
    else:
        text += f"There is no clear winner for {comp.class_name} in this run."
    return text + "\n\n" + PROJECT_RESULT


def _caveats(comp):
    return (f"The classifier is {comp.classifier_accuracy:.1%} accurate on real test images, so a few kept "
            f"images may belong to another class. These numbers come from one seed ({comp.seed}) and "
            f"n = {comp.n_compared} images per model; press 'New samples' to see how much they move. "
            f"The report's numbers use all classes, 10,000 samples and 3 seeds.")
