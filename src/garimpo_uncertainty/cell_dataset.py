"""Torch access to the cell store, with the role selection done by the caller."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data

import garimpo_uncertainty.data


class CellDataset(torch.utils.data.Dataset):
    """Yields (image, mask, cell_index); image is BOA reflectance, z-scored when stats are given."""

    def __init__(
        self,
        root: Path,
        cells: pd.DataFrame,
        normalization: dict[str, list[float]] | None,
        *,
        augment: bool = False,
    ) -> None:
        self.root = root
        self.cell_index = cells.cell_index.to_numpy()
        self.offsets = cells.boa_offset.to_numpy(dtype=np.float32)
        self.augment = augment
        # Opened in __getitem__, once per worker: a map held here would be pickled by value when
        # workers spawn. SCL stays out: nothing in training reads it.
        self.images, self.masks = None, None
        self.mean, self.std = None, None
        if normalization is not None:
            self.mean = np.array(normalization["mean"], dtype=np.float32).reshape(-1, 1, 1)
            self.std = np.array(normalization["std"], dtype=np.float32).reshape(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.cell_index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.images is None:
            self.images = np.load(self.root / "images.npy", mmap_mode="r")
            self.masks = np.load(self.root / "masks.npy", mmap_mode="r")
        cell_index = int(self.cell_index[index])
        raw = self.images[cell_index].astype(np.float32)
        # Baseline 04.00 shifts DN by the offset; nodata 0 would go negative, hence the floor.
        reflectance = np.maximum(raw + self.offsets[index], 0)
        image = reflectance / garimpo_uncertainty.data.REFLECTANCE_SCALE
        if self.mean is not None:
            image = (image - self.mean) / self.std
        mask = (self.masks[cell_index][None] > 0).astype(np.float32)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                image, mask = image[..., ::-1], mask[..., ::-1]
            if torch.rand(1).item() > 0.5:
                image, mask = image[..., ::-1, :], mask[..., ::-1, :]
        return torch.from_numpy(image.copy()), torch.from_numpy(mask.copy()), cell_index


def loader(
    root: Path,
    cells: pd.DataFrame,
    normalization: dict[str, list[float]] | None,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool = False,
    augment: bool = False,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        CellDataset(root, cells, normalization, augment=augment),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def identity_normalization() -> dict[str, list[float]]:
    """Return statistics that leave reflectance untouched, as the Sentinel-2 MoCo weights expect."""
    n_bands = len(garimpo_uncertainty.data.BANDS)
    return {"mean": [0.0] * n_bands, "std": [1.0] * n_bands}


def band_statistics(
    root: Path, cells: pd.DataFrame, batch_size: int, num_workers: int
) -> dict[str, list[float]]:
    """Per-band mean and std of training reflectance; only the training role may reach this."""
    total, squares, count = None, None, 0
    for images, _, _ in loader(root, cells, None, batch_size, num_workers):
        pixels = images.to(torch.float64).transpose(0, 1).flatten(1)
        total = pixels.sum(1) if total is None else total + pixels.sum(1)
        squares = (pixels**2).sum(1) if squares is None else squares + (pixels**2).sum(1)
        count += pixels.shape[1]
    mean = total / count
    std = (squares / count - mean**2).clamp(min=0).sqrt()
    if (std == 0).any():
        raise ValueError(f"a band is constant over the training cells: std={std.tolist()}")
    return {"mean": mean.tolist(), "std": std.tolist()}
