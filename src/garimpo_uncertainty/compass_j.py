"""COMPASS's search spaces at one layer, shared by the stages that fit, sweep, score and bound them.

COMPASS-L perturbs the logits along ones; COMPASS-J perturbs the layer's features along each cell's
direction, the PCA or SVM projection of that cell's Jacobian of the soft area.
"""

import math
from collections.abc import Callable
from typing import NamedTuple

import compass2d.compass
import compass2d.directions
import compass2d.metrics
import numpy as np
import torch

import garimpo_uncertainty.data
import garimpo_uncertainty.intervals
import garimpo_uncertainty.models


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


class Memo:
    """A space's metric at every signed beta each cell was decoded at, so none is decoded twice.

    The decoder sees beta cast to z's float32, so betas apart only past float32 give the same input,
    and in eval mode a cell's output ignores the rest of its batch: a remembered value is the bit.
    Betas match by their bits, so a NaN finds itself and -0.0 stays apart from 0.0.
    """

    def __init__(self, space: Space) -> None:
        self.space = space
        cells = space.z.shape[0]
        self.betas = torch.empty((cells, 0), dtype=space.z.dtype, device=space.z.device)
        self.values = torch.empty((cells, 0), dtype=torch.float64, device=space.z.device)
        self.decodes = 0

    def _matches(self, betas: torch.Tensor) -> torch.Tensor:
        return betas.view(torch.int32)[:, :, None] == self.betas.view(torch.int32)[:, None, :]

    def __call__(self, betas: torch.Tensor) -> torch.Tensor:
        """Metric at z + beta * delta for each cell's row of (cells, k) signed betas.

        One decode moves each cell by its own beta, so it takes one unseen beta from every cell
        that has one; a cell with none repeats a seen beta, whose value it already holds.
        """
        betas = betas.to(self.space.z)  # float32, so its bits view as int32
        while (unseen := ~self._matches(betas).any(2)).any():
            beta = betas.gather(1, unseen.to(torch.int8).argmax(1, keepdim=True)).squeeze(1)
            self.values = torch.cat([self.values, self.space.shifted(beta)[:, None]], 1)
            self.betas = torch.cat([self.betas, beta[:, None]], 1)
            self.decodes += 1
        return self.values.gather(1, self._matches(betas).to(torch.int8).argmax(2))


def _covered(areas: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Whether each search covers the target, from the areas at -s, s, -l and u of `search`."""
    down, up, lower, upper = areas.unbind(1)
    symmetric = (torch.minimum(down, up) <= target) & (target <= torch.maximum(down, up))
    return torch.stack([symmetric, lower <= target, upper >= target])


@torch.no_grad()
def search(
    space: Space, target: torch.Tensor, beta_range: float, threshold: float
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(score, saturated) of the symmetric, lower and upper searches, from one shared Memo.

    Equal to the bit to compass2d's binary_search_scores and asymmetric_binary_search_scores: the
    same bisection, run in lockstep. Its four signed betas per step and cell repeat each other, and
    past float32's precision earlier steps, so about 40% of the decodes go.
    """
    memo = Memo(space)
    low = torch.zeros((3, *target.shape), dtype=torch.float64, device=target.device)
    high = torch.full_like(low, beta_range)
    for _ in range(math.ceil(math.log2(beta_range / threshold))):
        middle = (low + high) / 2
        symmetric, lower, upper = middle
        areas = memo(torch.stack([-symmetric, symmetric, -lower, upper], 1))
        covered = _covered(areas, target)
        high = torch.where(covered, middle, high)
        low = torch.where(covered, low, middle)
    return [(score, score >= beta_range) for score in high]


class Split(NamedTuple):
    """One batch's logits, and its features `z` at a layer with the decoder `decode` after it."""

    logits: torch.Tensor
    z: torch.Tensor
    decode: Callable[[torch.Tensor], torch.Tensor]


def split(model: torch.nn.Module, features: list[torch.Tensor], layer: str) -> Split:
    """Split the batch at the logits, for COMPASS-L, and at `layer`, for COMPASS-J."""
    logits, _ = model.split_at(garimpo_uncertainty.models.LOGITS, features)
    return Split(logits, *model.split_at(layer, features))


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
