"""Dataset loading.

Every image is 32x32 and scaled to [-1, 1], which matches the tanh output of the
shared decoder/generator. Datasets are downloaded automatically by torchvision.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGE_SIZE = 32
_DATASETS = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}
_CHANNELS = {"mnist": 1, "fashion_mnist": 1, "cifar10": 3}
_NEEDS_PADDING = {"mnist", "fashion_mnist"}  # 28x28 -> 32x32


def _check(name):
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {sorted(_DATASETS)}")


def dataset_info(name):
    _check(name)
    return {"channels": _CHANNELS[name], "num_classes": 10, "image_size": IMAGE_SIZE}


def _transform(name):
    channels = _CHANNELS[name]
    steps = []
    if name in _NEEDS_PADDING:
        # Zero-pad instead of resizing: interpolation would blur the REAL images
        # and bias our sharpness comparison in favour of the blurrier model.
        steps.append(transforms.Pad(2))
    steps += [transforms.ToTensor(), transforms.Normalize([0.5] * channels, [0.5] * channels)]
    return transforms.Compose(steps)


def get_dataset(name, split, root="data"):
    _check(name)
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    return _DATASETS[name](root=root, train=split == "train", download=True, transform=_transform(name))


def get_loader(name, split, batch_size, num_workers=2, root="data", subset=None, seed=0, shuffle=None):
    dataset = get_dataset(name, split, root)
    if subset:
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
        dataset = Subset(dataset, order[:subset].tolist())
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def load_split(name, split, root="data"):
    """Whole split as in-memory tensors (images in [-1, 1], labels), for evaluation.

    Reads the raw uint8 arrays directly, which is much faster than going through
    the per-image PIL transforms, and produces exactly the same values.
    """
    dataset = get_dataset(name, split, root)
    images = torch.as_tensor(dataset.data)
    images = images.unsqueeze(1) if images.ndim == 3 else images.permute(0, 3, 1, 2)
    images = images.float() / 255.0
    if name in _NEEDS_PADDING:
        images = F.pad(images, (2, 2, 2, 2))
    labels = torch.as_tensor(dataset.targets)
    return images * 2 - 1, labels


def class_names(name, root="data"):
    return list(get_dataset(name, "test", root).classes)
