"""Tiny YAML config loader with inheritance and command-line overrides.

A model config may declare `base: shared.yaml`. The shared file is loaded first
and the model file overrides it. Every setting that must be identical for the
VAE and the GAN lives in configs/shared.yaml, so the two models cannot drift
apart by accident.

Overrides use the same syntax for every script:
    python -m src.train_vae --config configs/vae.yaml --set seed=1 beta=4
"""
from pathlib import Path

import yaml


def _parse_value(raw):
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        # YAML reads "1e-4" as a string; we want it as a float.
        try:
            return float(value)
        except ValueError:
            return value
    return value


def load_config(path, overrides=None):
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    base = cfg.pop("base", None)
    if base is not None:
        merged = load_config(path.parent / base)
        merged.update(cfg)
        cfg = merged

    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        if key not in cfg:
            raise KeyError(f"Unknown config key {key!r}. Known keys: {sorted(cfg)}")
        cfg[key] = _parse_value(raw)
    return cfg
