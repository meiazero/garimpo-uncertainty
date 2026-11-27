"""Validation metrics by name, each accumulated over batches of predicted and reference masks.

The masks are boolean (B, 1, H, W). A metric with nothing to measure is NaN: Dice and recall
without reference mining, precision without predicted mining.
"""

import numpy as np
import torch

import garimpo_uncertainty.registry

METRIC_REGISTRY = garimpo_uncertainty.registry.Registry("metric")


def ratio(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


class Metric:
    def update(self, predictions: torch.Tensor, references: torch.Tensor) -> None:
        raise NotImplementedError

    def compute(self) -> float:
        raise NotImplementedError


@METRIC_REGISTRY.register("dice")
class MeanCellDice(Metric):
    """The mean Dice over cells with mining; an empty cell has no Dice to average."""

    def __init__(self) -> None:
        self.scores: list[float] = []

    def update(self, predictions: torch.Tensor, references: torch.Tensor) -> None:
        intersection = (predictions & references).flatten(1).sum(1)
        totals = predictions.flatten(1).sum(1) + references.flatten(1).sum(1)
        has_mining = references.flatten(1).any(1)
        self.scores += (2 * intersection / totals.clamp(min=1))[has_mining].tolist()

    def compute(self) -> float:
        return float(np.mean(self.scores)) if self.scores else float("nan")


class PooledPixels(Metric):
    """Counts every pixel of every cell, so a large cell weighs more than a small one."""

    def __init__(self) -> None:
        self.true_positive = self.false_positive = self.false_negative = 0

    def update(self, predictions: torch.Tensor, references: torch.Tensor) -> None:
        self.true_positive += int((predictions & references).sum())
        self.false_positive += int((predictions & ~references).sum())
        self.false_negative += int((~predictions & references).sum())


@METRIC_REGISTRY.register("precision")
class Precision(PooledPixels):
    def compute(self) -> float:
        return ratio(self.true_positive, self.true_positive + self.false_positive)


@METRIC_REGISTRY.register("recall")
class Recall(PooledPixels):
    def compute(self) -> float:
        return ratio(self.true_positive, self.true_positive + self.false_negative)
