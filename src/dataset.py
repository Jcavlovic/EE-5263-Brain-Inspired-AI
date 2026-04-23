"""LFW gender-classification dataset.

Images are converted to grayscale, center-cropped to the face region, resized,
and returned as flattened float tensors suitable for an MLP.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from labels import Sample


@dataclass
class DatasetConfig:
    image_size: int = 64           # final square edge length
    center_crop: int = 150         # crop from the 250x250 raw image
    balance: bool = True           # downsample majority class
    test_fraction: float = 0.2
    seed: int = 42


def _default_transform(cfg: DatasetConfig) -> transforms.Compose:
    # Use torch.flatten (a module-level function) rather than a lambda so that
    # the transform pipeline can be pickled by DataLoader workers under the
    # `forkserver` start method used by Python >= 3.14.
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(cfg.center_crop),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),                    # [0, 1], shape (1, H, W)
        transforms.Lambda(torch.flatten),         # shape (H*W,)
    ])


class LFWGenderDataset(Dataset):
    def __init__(self, samples: list[Sample], cfg: DatasetConfig):
        self.samples = samples
        self.cfg = cfg
        self.transform = _default_transform(cfg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        with Image.open(s.path) as img:
            x = self.transform(img.convert("RGB"))
        y = torch.tensor(float(s.label), dtype=torch.float32)
        return x, y


def balance_and_split(
    samples: list[Sample],
    cfg: DatasetConfig,
) -> tuple[list[Sample], list[Sample]]:
    """Split by identity (never put the same person in train and test), then
    downsample the majority class to match the minority for training balance.
    """
    rng = random.Random(cfg.seed)

    by_person: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_person[s.person].append(s)

    # Split at the identity level so no subject leaks across splits.
    people = sorted(by_person.keys())
    rng.shuffle(people)
    n_test_people = max(1, int(len(people) * cfg.test_fraction))
    test_people = set(people[:n_test_people])

    train: list[Sample] = []
    test: list[Sample] = []
    for person, items in by_person.items():
        (test if person in test_people else train).extend(items)

    if cfg.balance:
        train = _downsample(train, rng)
        test = _downsample(test, rng)

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _downsample(items: list[Sample], rng: random.Random) -> list[Sample]:
    males = [s for s in items if s.label == 0]
    females = [s for s in items if s.label == 1]
    n = min(len(males), len(females))
    if n == 0:
        return items
    rng.shuffle(males)
    rng.shuffle(females)
    return males[:n] + females[:n]
