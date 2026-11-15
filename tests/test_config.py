"""Config loading, run naming, and the fairness guarantee of the real config files."""
from pathlib import Path

import pytest
import torch
import yaml

from src.config import load_config
from src.utils import fixed_noise, run_name

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _write(path, data):
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def cfg_dir(tmp_path):
    _write(tmp_path / "shared.yaml", {"latent_dim": 32, "epochs": 30, "seed": 0, "lr": 0.1})
    _write(tmp_path / "model.yaml", {"base": "shared.yaml", "model": "vae", "lr": 1.0e-3})
    return tmp_path


def test_base_file_is_inherited_and_overridden(cfg_dir):
    cfg = load_config(cfg_dir / "model.yaml")
    assert cfg["latent_dim"] == 32 and cfg["model"] == "vae"  # inherited + own key
    assert cfg["lr"] == pytest.approx(1e-3)                    # model file wins over shared
    assert "base" not in cfg


def test_overrides_are_parsed_to_the_right_type(cfg_dir):
    cfg = load_config(cfg_dir / "model.yaml", ["seed=2", "lr=1e-4", "latent_dim=8"])
    assert cfg["seed"] == 2 and isinstance(cfg["seed"], int)
    assert cfg["lr"] == pytest.approx(1e-4) and isinstance(cfg["lr"], float)  # YAML alone gives a string
    assert cfg["latent_dim"] == 8


def test_typo_in_override_is_an_error(cfg_dir):
    # A silently ignored typo (latnet_dim=8) would produce a mislabelled experiment.
    with pytest.raises(KeyError):
        load_config(cfg_dir / "model.yaml", ["latnet_dim=8"])
    with pytest.raises(ValueError):
        load_config(cfg_dir / "model.yaml", ["latent_dim"])


def test_vae_and_gan_configs_share_every_common_setting():
    """Fairness: neither model config may override a setting from shared.yaml."""
    shared = load_config(CONFIGS / "shared.yaml")
    vae = load_config(CONFIGS / "vae.yaml")
    gan = load_config(CONFIGS / "gan.yaml")
    for key, value in shared.items():
        assert vae[key] == value, f"vae.yaml overrides shared setting {key!r}"
        assert gan[key] == value, f"gan.yaml overrides shared setting {key!r}"


def test_run_names_are_unique_per_setting():
    base = {"model": "vae", "dataset": "fashion_mnist", "latent_dim": 32, "beta": 1.0, "seed": 0}
    assert run_name(base) == "vae_fashion_mnist_z32_beta1_seed0"
    variants = [base, {**base, "beta": 0.5}, {**base, "seed": 1}, {**base, "latent_dim": 8},
                {**base, "model": "gan"}]
    assert len({run_name(v) for v in variants}) == len(variants)  # no two runs share a folder


def test_fixed_noise_is_reproducible():
    assert torch.equal(fixed_noise(8, 32), fixed_noise(8, 32))
    assert not torch.equal(fixed_noise(8, 32, seed=1), fixed_noise(8, 32, seed=2))
