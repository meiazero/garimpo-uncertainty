"""Torch access to the cell store, with the role selection done by the caller."""

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data

import garimpo_uncertainty.data
import garimpo_uncertainty.training.augmentations


class AugmentedDraws(torch.utils.data.Sampler):
    """`samples` cells drawn as RandomSampler draws them, each with what its augmentations drew.

    The augmentations draw from the sampler's own generator, in the main process, so the draws
    follow the cells and not the worker that loads them: the number of workers leaves the weights
    alone.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        samples: int,
        augmentations: tuple[garimpo_uncertainty.training.augmentations.Augmentation, ...],
    ) -> None:
        self.dataset, self.samples, self.augmentations = dataset, samples, augmentations

    def __len__(self) -> int:
        return self.samples

    def __iter__(self) -> Iterator[tuple[int, tuple]]:
        # One draw from torch's generator per pass, as RandomSampler takes its own seed.
        generator = torch.Generator().manual_seed(
            int(torch.empty((), dtype=torch.int64).random_().item())
        )
        cells = torch.utils.data.RandomSampler(
            self.dataset, num_samples=self.samples, generator=generator
        )
        for index in cells:
            yield index, tuple(augmentation.draw(generator) for augmentation in self.augmentations)


class CellDataset(torch.utils.data.Dataset):
    """Yields (image, mask, cell_index): BOA reflectance and its mask, through `transform`."""

    def __init__(
        self,
        root: Path,
        cells: pd.DataFrame,
        transform: garimpo_uncertainty.training.augmentations.Transform | None,
    ) -> None:
        self.root = root
        self.cell_index = cells.cell_index.to_numpy()
        self.offsets = cells.boa_offset.to_numpy(dtype=np.float32)
        self.transform = transform
        # Opened in __getitem__, once per worker: a map held here would be pickled by value when
        # workers spawn. SCL stays out: nothing in training reads it.
        self.images, self.masks = None, None

    def __len__(self) -> int:
        return len(self.cell_index)

    def __getitem__(self, drawn: int | tuple[int, tuple]) -> tuple[torch.Tensor, torch.Tensor, int]:
        index, draws = drawn if isinstance(drawn, tuple) else (drawn, ())
        if self.images is None:
            self.images = np.load(self.root / "images.npy", mmap_mode="r")
            self.masks = np.load(self.root / "masks.npy", mmap_mode="r")
        cell_index = int(self.cell_index[index])
        raw = self.images[cell_index].astype(np.float32)
        # Baseline 04.00 shifts DN by the offset; nodata 0 would go negative, hence the floor.
        reflectance = np.maximum(raw + self.offsets[index], 0)
        image = reflectance / garimpo_uncertainty.data.REFLECTANCE_SCALE
        mask = (self.masks[cell_index][None] > 0).astype(np.float32)
        if self.transform is not None:
            image, mask = self.transform(image, mask, draws)
        return torch.from_numpy(image.copy()), torch.from_numpy(mask.copy()), cell_index


def loader(
    root: Path,
    cells: pd.DataFrame,
    normalization: dict[str, list[float]] | None,
    batch_size: int,
    num_workers: int,
    *,
    samples: int | None = None,
    augmentations: tuple[garimpo_uncertainty.training.augmentations.Augmentation, ...] = (),
) -> torch.utils.data.DataLoader:
    """In order, or training's draws: `samples` cells in back-to-back shuffles, augmented.

    A fixed number of samples gives every training fraction the same updates per epoch.
    """
    transform = garimpo_uncertainty.training.augmentations.Transform(normalization, augmentations)
    dataset = CellDataset(root, cells, transform)
    sampler = None if samples is None else AugmentedDraws(dataset, samples, augmentations)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def batches(loader: torch.utils.data.DataLoader) -> Iterator:
    """Start iterating the loader without losing a Ctrl+C that lands while it forks its workers.

    Python raises such a Ctrl+C inside an at-fork callback, where it only prints it as unraisable
    and the stage carries on; it is kept here and raised again once the workers exist.
    """
    swallowed: list[KeyboardInterrupt] = []
    previous = sys.unraisablehook

    def keep(unraisable: object) -> None:  # sys.UnraisableHookArgs exists only in typeshed
        if isinstance(unraisable.exc_value, KeyboardInterrupt):
            swallowed.append(unraisable.exc_value)
        else:
            previous(unraisable)

    sys.unraisablehook = keep
    try:
        iterator = iter(loader)
    finally:
        sys.unraisablehook = previous
    if swallowed:
        raise swallowed[0]
    return iterator


def identity_normalization() -> dict[str, list[float]]:
    """Return statistics that leave reflectance untouched, as the Sentinel-2 MoCo weights expect."""
    n_bands = len(garimpo_uncertainty.data.BANDS)
    return {"mean": [0.0] * n_bands, "std": [1.0] * n_bands}


def band_statistics(
    root: Path, cells: pd.DataFrame, batch_size: int, num_workers: int
) -> dict[str, list[float]]:
    """Per-band mean and std of training reflectance; only the training role may reach this."""
    total, squares, count = None, None, 0
    for images, _, _ in batches(loader(root, cells, None, batch_size, num_workers)):
        pixels = images.to(torch.float64).transpose(0, 1).flatten(1)
        total = pixels.sum(1) if total is None else total + pixels.sum(1)
        squares = (pixels**2).sum(1) if squares is None else squares + (pixels**2).sum(1)
        count += pixels.shape[1]
    mean = total / count
    std = (squares / count - mean**2).clamp(min=0).sqrt()
    if (std == 0).any():
        raise ValueError(f"a band is constant over the training cells: std={std.tolist()}")
    return {"mean": mean.tolist(), "std": std.tolist()}
