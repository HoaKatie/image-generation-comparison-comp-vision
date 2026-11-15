"""Preprocessing: 28x28 -> 32x32 by zero-padding (no resizing), values in [-1, 1]."""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import _transform, dataset_info, get_dataset, load_split

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def test_padding_keeps_the_image_and_adds_a_black_border():
    white = Image.fromarray(np.full((28, 28), 255, dtype=np.uint8))
    x = _transform("fashion_mnist")(white)
    assert x.shape == (1, 32, 32)
    assert x.min() >= -1 and x.max() <= 1
    # The original pixels are untouched (no interpolation blur) ...
    assert torch.all(x[:, 2:30, 2:30] == 1.0)
    # ... and the 2-pixel border is black (0 -> -1 after scaling).
    border = torch.ones(32, 32, dtype=torch.bool)
    border[2:30, 2:30] = False
    assert torch.all(x[:, border] == -1.0)


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError):
        dataset_info("imagenet")


@pytest.mark.skipif(not (DATA_ROOT / "FashionMNIST" / "raw").exists(),
                    reason="Fashion-MNIST not downloaded yet (any training script downloads it)")
def test_fast_loader_matches_the_transform_pipeline():
    """load_split (used by evaluation) must give exactly the images that training sees."""
    images, labels = load_split("fashion_mnist", "test", DATA_ROOT)
    dataset = get_dataset("fashion_mnist", "test", DATA_ROOT)
    assert images.shape == (10000, 1, 32, 32)
    for i in (0, 1234, 9999):
        x, y = dataset[i]
        assert torch.allclose(images[i], x, atol=1e-6)
        assert labels[i].item() == y
