"""COMPASS's search spaces at one layer, shared by the stages that fit, sweep, score and bound them.

COMPASS-L perturbs the logits along ones; COMPASS-J perturbs the layer's features along each cell's
direction, the PCA or SVM projection of that cell's Jacobian of the soft area.
"""

from collections.abc import Callable
from typing import NamedTuple

import compass2d.compass
import compass2d.directions
import compass2d.metrics
import numpy as np
import torch

import garimpo_uncertainty.data
import garimpo_uncertainty.intervals
import garimpo_uncertainty.models.common


def soft_area(temperature: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the cell's mined area in ha with each pixel's sigmoid at `temperature`."""
    pixel_ha = garimpo_uncertainty.data.PIXEL_HA
    return lambda logits: compass2d.metrics.soft_area(logits, pixel_ha, temperature)


def jacobian(
    decode: Callable[[torch.Tensor], torch.Tensor], z: torch.Tensor, temperature: float
) -> np.ndarray:
    """Return each cell's Jacobian of the soft area at `temperature`, which sets its direction."""
    return compass2d.directions.jacobian_sum(decode, z, soft_area(temperature)).cpu().numpy()


def channel_deltas(
    jacobians: np.ndarray, fitted: compass2d.directions.Directions, z: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return each direction's per-cell perturbation of `z`, by name of DIRECTIONS."""
    deltas = {}
    for direction in garimpo_uncertainty.intervals.DIRECTIONS:
        delta = compass2d.directions.cell_deltas(direction, jacobians.astype(np.float64), fitted)
        deltas[direction] = compass2d.compass.channel_delta(torch.from_numpy(delta.copy()).to(z))
    return deltas


class Space(NamedTuple):
    """What one COMPASS method perturbs: the point, its direction, and how to read an area."""

    decode: Callable[[torch.Tensor], torch.Tensor]
    metric: Callable[[torch.Tensor], torch.Tensor]
    z: torch.Tensor
    delta: torch.Tensor

    def endpoints(self, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return compass2d.compass.endpoints(self.decode, self.metric, self.z, self.delta, beta)

    def shifted(self, beta: torch.Tensor) -> torch.Tensor:
        return compass2d.compass.shifted(self.decode, self.metric, self.z, self.delta, beta)


class Split(NamedTuple):
    """One batch's logits, and its features `z` at a layer with the decoder `decode` after it."""

    logits: torch.Tensor
    z: torch.Tensor
    decode: Callable[[torch.Tensor], torch.Tensor]


def split(model: torch.nn.Module, features: list[torch.Tensor], layer: str) -> Split:
    """Split the batch at the logits, for COMPASS-L, and at `layer`, for COMPASS-J."""
    logits, _ = garimpo_uncertainty.models.common.split_at(
        model, garimpo_uncertainty.models.common.LOGITS, features
    )
    return Split(logits, *garimpo_uncertainty.models.common.split_at(model, layer, features))


def spaces(
    split: Split, jacobians: np.ndarray, fitted: compass2d.directions.Directions, temperature: float
) -> dict[str, Space]:
    """Return the space of each method of intervals.COMPASS_METHODS on one batch, in that order."""
    pixel_ha = garimpo_uncertainty.data.PIXEL_HA
    found = {
        "compass_l": Space(
            lambda z: z,
            lambda output: compass2d.metrics.hard_area(output, pixel_ha),
            split.logits,
            torch.ones_like(split.logits),
        )
    }
    for direction, delta in channel_deltas(jacobians, fitted, split.z).items():
        found[f"compass_j_{direction}"] = Space(
            split.decode, soft_area(temperature), split.z, delta
        )
    return found
