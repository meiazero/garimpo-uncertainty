"""Torch access to the cells, with the role selection done by the caller."""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.utils.data

import garimpo_uncertainty.data


class CellDataset(torch.utils.data.Dataset):
    """Yields (image, mask, cell_id); image is BOA reflectance, z-scored when stats are given."""

    def __init__(
        self,
        root: Path,
        cells: pd.DataFrame,
        normalization: dict[str, list[float]] | None,
        augment: bool = False,
    ) -> None:
        self.root = root
        self.paths = cells.path.tolist()
        self.offsets = cells.boa_offset.tolist()
        self.cell_ids = cells.cell_id.tolist()
        self.augment = augment
        self.mean, self.std = None, None
        if normalization is not None:
            self.mean = np.array(normalization["mean"], dtype=np.float32).reshape(-1, 1, 1)
            self.std = np.array(normalization["std"], dtype=np.float32).reshape(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        with rasterio.open(self.root / self.paths[index]) as source:
            stack = source.read().astype(np.float32)
        n_bands = len(garimpo_uncertainty.data.BANDS)
        # Baseline 04.00 shifts DN by the offset; nodata 0 would go negative, hence the floor.
        reflectance = np.maximum(stack[:n_bands] + self.offsets[index], 0)
        image = reflectance / garimpo_uncertainty.data.REFLECTANCE_SCALE
        if self.mean is not None:
            image = (image - self.mean) / self.std
        mask_band = garimpo_uncertainty.data.MASK_BAND
        mask = (stack[mask_band - 1 : mask_band] > 0).astype(np.float32)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                image, mask = image[..., ::-1], mask[..., ::-1]
            if torch.rand(1).item() > 0.5:
                image, mask = image[..., ::-1, :], mask[..., ::-1, :]
        return (
            torch.from_numpy(image.copy()),
            torch.from_numpy(mask.copy()),
            self.cell_ids[index],
        )


def loader(
    root: Path,
    cells: pd.DataFrame,
    normalization: dict[str, list[float]] | None,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    augment: bool = False,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        CellDataset(root, cells, normalization, augment),
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
