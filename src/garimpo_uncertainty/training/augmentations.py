"""Training augmentations by name, and the transform that applies them before normalization.

An augmentation draws its randomness in `draw`, in the main process with the sampler's generator,
and applies it in `__call__`, in a worker. So a cell gets the same augmentation whatever worker
loads it, and the number of workers leaves the trained weights alone.
"""

import json
from typing import Any, override

import numpy as np
import torch

import garimpo_uncertainty.data
import garimpo_uncertainty.registry

AUGMENTATION_REGISTRY = garimpo_uncertainty.registry.Registry("augmentation")
FLIP_PROBABILITY = 0.5
# Albumentations seeds numpy's and Python's generators, which take any non-negative 32-bit int.
SEED_BOUND = 2**32
# A transform with p=0.5 escapes this many probes once in 2**16 configs; __call__ still refuses it.
PROBES = 16


class Augmentation:
    def draw(self, generator: torch.Generator) -> Any:
        raise NotImplementedError

    def __call__(
        self, image: np.ndarray, mask: np.ndarray, drawn: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


@AUGMENTATION_REGISTRY.register("flips")
class Flips(Augmentation):
    """A horizontal and a vertical flip, each with probability one half."""

    @override
    def draw(self, generator: torch.Generator) -> tuple[bool, bool]:
        horizontal, vertical = (torch.rand(2, generator=generator) > FLIP_PROBABILITY).tolist()
        return horizontal, vertical

    @override
    def __call__(
        self, image: np.ndarray, mask: np.ndarray, drawn: tuple[bool, bool]
    ) -> tuple[np.ndarray, np.ndarray]:
        horizontal, vertical = drawn
        if horizontal:
            image, mask = image[..., ::-1], mask[..., ::-1]
        if vertical:
            image, mask = image[..., ::-1, :], mask[..., ::-1, :]
        return image, mask


@AUGMENTATION_REGISTRY.register("albumentations")
class Albumentations(Augmentation):
    """A pipeline in Albumentations' serialized form, `A.to_dict(pipeline)["transform"]`."""

    def __init__(self, transform: dict[str, Any]) -> None:
        # Imported here, so only a config that names it loads OpenCV into every worker.
        import albumentations  # noqa: PLC0415

        if "ToTensorV2" in json.dumps(transform):
            raise ValueError("albumentations: leave out ToTensorV2; the loader makes the tensors")
        try:
            self.pipeline = albumentations.from_dict({"transform": transform})
        except KeyError as missing:  # from_dict names only the key it lacked
            raise ValueError(f"albumentations: not a serialized pipeline, no {missing}") from None
        except AttributeError:  # from_dict calls .get on a step that is a string or a list
            raise ValueError(
                "albumentations: not a serialized pipeline, a step is no dict"
            ) from None
        self.check_output()

    def check_output(self) -> None:
        """Run a few draws on a blank cell, so a wrong pipeline fails before training starts."""
        side, bands = garimpo_uncertainty.data.CELL_PX, len(garimpo_uncertainty.data.BANDS)
        image = np.zeros((bands, side, side), np.float32)
        for seed in range(PROBES):
            self(image, np.zeros((1, side, side), np.float32), seed)

    @override
    def draw(self, generator: torch.Generator) -> int:
        return int(torch.randint(SEED_BOUND, (), generator=generator))

    @override
    def __call__(
        self, image: np.ndarray, mask: np.ndarray, drawn: int
    ) -> tuple[np.ndarray, np.ndarray]:
        self.pipeline.set_random_seed(drawn)
        # Albumentations reads channels last, and a mask without channels.
        found = self.pipeline(image=image.transpose(1, 2, 0), mask=mask[0])
        cell, label = found["image"].transpose(2, 0, 1), found["mask"][None]
        # The loader stacks cells of one shape, and the model reads float32.
        if cell.dtype != np.float32 or cell.shape != image.shape or label.shape != mask.shape:
            raise ValueError(
                "albumentations: the pipeline must keep a float32 cell and its mask as they came, "
                f"{list(image.shape)} and {list(mask.shape)}, not {cell.dtype} "
                f"{list(cell.shape)} and {list(label.shape)}"
            )
        if not np.isfinite(cell).all():
            raise ValueError("albumentations: the pipeline must keep the cell's values finite")
        if not ((label == 0) | (label == 1)).all():
            raise ValueError(
                "albumentations: the pipeline must keep the mask's 0 and 1, the classes the losses "
                f"read, not {np.unique(label)[:8].tolist()}"
            )
        return cell, label


class Transform:
    """Applies each augmentation to a cell's reflectance, then z-scores it by the given statistics.

    Albumentations reads a float image as values in [0, 1] and clips some results to it, which
    reflectance nearly is and a z-score is not. A flip commutes with the z-score, bit for bit.
    ponytail: a step that clips also saturates the few pixels brighter than reflectance 1; one
    that must keep them needs the cell scaled into [0, 1] around it.
    """

    def __init__(
        self,
        normalization: dict[str, list[float]] | None,
        augmentations: tuple[Augmentation, ...] = (),
    ) -> None:
        self.augmentations = augmentations
        self.mean, self.std = None, None
        if normalization is not None:
            self.mean = np.array(normalization["mean"], dtype=np.float32).reshape(-1, 1, 1)
            self.std = np.array(normalization["std"], dtype=np.float32).reshape(-1, 1, 1)

    def __call__(
        self, image: np.ndarray, mask: np.ndarray, draws: tuple = ()
    ) -> tuple[np.ndarray, np.ndarray]:
        for augmentation, drawn in zip(self.augmentations, draws, strict=True):
            image, mask = augmentation(image, mask, drawn)
        if self.mean is not None:
            image = (image - self.mean) / self.std
        return image, mask
