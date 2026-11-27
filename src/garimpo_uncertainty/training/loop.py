"""The training loop, in a fixed order; checkpoints, logs and early stopping are callbacks.

Each epoch runs `train_step` over the training draws, `val_step` over the validation cells and over
a training subset in eval mode, then every callback's `on_epoch_end`, in order.
"""

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

import garimpo_uncertainty.cell_dataset
import garimpo_uncertainty.pipeline.runner
import garimpo_uncertainty.training.metrics

NO_BEST_DICE = -1.0


def dice_score(record: dict) -> float:
    """Return the epoch's val_dice as the best checkpoint ranks it: a NaN never wins."""
    return NO_BEST_DICE if np.isnan(record["val_dice"]) else record["val_dice"]


@dataclass
class TrainingState:
    """What the epochs share, and what last.pt restores after an interruption."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.Module
    postprocess: Callable[[torch.Tensor], torch.Tensor]
    metric_names: list[str]
    metadata: dict
    arguments: dict[str, str]
    history: list[dict]
    best_dice: float
    first_epoch: int
    run_dir: Path  # where the checkpoints go
    # Set by a callback to end the training after this epoch, as early stopping does.
    stopped: bool = False
    # What callbacks keep across a resume, by callback name.
    callbacks: dict[str, dict] = field(default_factory=dict)


@dataclass
class Loaders:
    train: torch.utils.data.DataLoader
    validation: torch.utils.data.DataLoader
    # Training cells in eval mode, unaugmented: with val_dice, it tells underfitting from
    # overfitting, which train_loss cannot, being augmented and in training mode.
    train_eval: torch.utils.data.DataLoader


class Callback:
    """Runs after each epoch's validation, once its record is in the history."""

    # A callback that changes the state must run before the checkpoint, so last.pt keeps it.
    changes_state = False

    def on_fit_start(self, trainer: "Trainer") -> None:
        """Run once, before the first epoch of this call to fit."""

    def on_epoch_end(self, state: TrainingState, record: dict) -> None:
        raise NotImplementedError


class Trainer:
    def __init__(
        self, state: TrainingState, device: str, epochs: int, callbacks: list[Callback]
    ) -> None:
        self.state, self.device, self.epochs, self.callbacks = state, device, epochs, callbacks

    def fit(self, loaders: Loaders) -> list[int] | None:
        """Train up to `epochs`; return the shape of the last batch, None if no epoch ran."""
        batch_shape = None
        for callback in self.callbacks:
            callback.on_fit_start(self)
        for epoch in range(self.state.first_epoch, self.epochs):
            if self.state.stopped:
                break
            record, batch_shape = self.run_epoch(loaders, epoch)
            self.state.history.append(record)
            for callback in self.callbacks:
                callback.on_epoch_end(self.state, record)
        return batch_shape

    def run_epoch(self, loaders: Loaders, epoch: int) -> tuple[dict, list[int] | None]:
        """Train, validate and score the training subset once; return the record and last shape."""
        garimpo_uncertainty.pipeline.runner.take_gpu_peak_bytes()
        self.state.model.train()
        epoch_start = time.monotonic()
        fitted, batch_shape = self.fit_epoch(loaders.train, epoch_start)
        train_end = time.monotonic()
        cells = fitted.pop("cells")
        train_peak = garimpo_uncertainty.pipeline.runner.take_gpu_peak_bytes()
        evaluated = self.evaluate_epoch(loaders)
        record = {
            "epoch": epoch,
            **fitted,
            **evaluated,
            "seconds": round(time.monotonic() - epoch_start, 3),
            "train_seconds": train_end - epoch_start,
            "train_cells_per_second": cells / (train_end - epoch_start),
            "train_peak_gpu_bytes": train_peak,
        }
        return record, batch_shape

    def fit_epoch(
        self, loader: torch.utils.data.DataLoader, started: float
    ) -> tuple[dict, list[int] | None]:
        """Return the loss, gradient norms, updates, cells and data wait, and the last shape."""
        losses, grad_norms = [], []
        cells = 0
        data_wait = 0.0
        batch_shape = None
        # The wait starts with the epoch's clock, so the two timings share one origin.
        ready = started
        for images, masks, _ in garimpo_uncertainty.cell_dataset.batches(loader):
            data_wait += time.monotonic() - ready
            loss, grad_norm = self.train_step(images, masks)
            losses.append(loss)
            grad_norms.append(grad_norm)
            cells += len(images)
            batch_shape = list(images.shape)
            ready = time.monotonic()
        fitted = {
            "train_loss": float(np.mean(losses)),
            "grad_norm_mean": float(np.mean(grad_norms)),
            # np.max, not max: a NaN norm, the sign of a diverging run, must not hide behind a
            # number.
            "grad_norm_max": float(np.max(grad_norms)) if grad_norms else float("nan"),
            "lr": self.state.optimizer.param_groups[0]["lr"],
            "updates": len(losses),
            "cells": cells,
            "train_data_wait_seconds": round(data_wait, 3),
        }
        return fitted, batch_shape

    def train_step(self, images: torch.Tensor, masks: torch.Tensor) -> tuple[float, float]:
        """One update; return its loss and the gradient norm before the step."""
        state = self.state
        state.optimizer.zero_grad()
        loss = state.loss_fn(state.model(images.to(self.device)), masks.to(self.device))
        loss.backward()
        # Read only, before the step: a rising norm is the first sign of a diverging run.
        gradients = [p.grad for p in state.model.parameters() if p.grad is not None]
        grad_norm = torch.nn.utils.get_total_norm(gradients).item()
        state.optimizer.step()
        return loss.item(), grad_norm

    def evaluate_epoch(self, loaders: Loaders) -> dict:
        """Score the validation cells, then the training subset, in eval mode, with their times."""
        validation_start = time.monotonic()
        validated = self.validate(loaders.validation, "val")
        validation_end = time.monotonic()
        validation_peak = garimpo_uncertainty.pipeline.runner.take_gpu_peak_bytes()
        # Each pass over a loader draws a seed from torch's CPU generator. Restoring it afterwards
        # keeps the training's random stream, and with it the trained weights, as without this.
        with torch.random.fork_rng(devices=[]):
            scored = self.validate(loaders.train_eval, "train_eval")
        return {
            **validated,
            **scored,
            "validation_seconds": validation_end - validation_start,
            "train_eval_seconds": time.monotonic() - validation_end,
            "validation_peak_gpu_bytes": validation_peak,
        }

    @torch.no_grad()
    def validate(self, loader: torch.utils.data.DataLoader, prefix: str) -> dict[str, float]:
        """Return the loss and every chosen metric in eval mode, in one pass over every cell.

        `val_dice` picks the best checkpoint. The loss weighs each batch by its cells.
        """
        self.state.model.eval()
        registry = garimpo_uncertainty.training.metrics.METRIC_REGISTRY
        metrics = {name: registry.build(name) for name in self.state.metric_names}
        weighted_loss = 0.0
        for images, masks, _ in garimpo_uncertainty.cell_dataset.batches(loader):
            weighted_loss += self.val_step(images, masks, metrics.values()) * len(images)
        loss = garimpo_uncertainty.training.metrics.ratio(weighted_loss, len(loader.dataset))
        scores = {f"{prefix}_{name}": metric.compute() for name, metric in metrics.items()}
        # In name order, as main wrote history.csv, whatever order METRICS lists.
        return dict(sorted(({f"{prefix}_loss": loss} | scores).items()))

    def val_step(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        metrics: Iterable[garimpo_uncertainty.training.metrics.Metric],
    ) -> float:
        """Score one batch into each metric; return its mean loss."""
        logits = self.state.model(images.to(self.device))
        references = masks.to(self.device)
        loss = self.state.loss_fn(logits, references).item()
        predictions = self.state.postprocess(logits)
        for metric in metrics:
            metric.update(predictions, references > 0)
        return loss
