"""What runs around the training loop, each callback picked by name from CALLBACKS.

Every run needs `checkpoint`, whose best_model.pt the later stages read; `train` refuses a config
without it, or with a callback that changes the state after it.
"""

import math
from typing import override

import numpy as np
import pandas as pd
import torch

import garimpo_uncertainty.artifacts
import garimpo_uncertainty.checkpoints
import garimpo_uncertainty.files
import garimpo_uncertainty.pipeline.runner
import garimpo_uncertainty.registry
import garimpo_uncertainty.training.loop

CALLBACK_REGISTRY = garimpo_uncertainty.registry.Registry("callback")
TrainingState = garimpo_uncertainty.training.loop.TrainingState


@CALLBACK_REGISTRY.register("checkpoint")
class CheckpointCallback(garimpo_uncertainty.training.loop.Callback):
    """Writes history.csv and last.pt every epoch, and best_model.pt when val_dice improved."""

    @override
    def on_epoch_end(self, state: TrainingState, record: dict) -> None:
        run = garimpo_uncertainty.artifacts.RunPaths.of(state.run_dir)
        epoch, dice = record["epoch"], record["val_dice"]
        # Every epoch, so a job killed at its limit still reports what it measured.
        with garimpo_uncertainty.files.atomic(run.history) as temporary:
            pd.DataFrame(state.history).to_csv(temporary, index=False)
        score = garimpo_uncertainty.training.loop.NO_BEST_DICE if np.isnan(dice) else dice
        weights = {k: v.cpu() for k, v in state.model.state_dict().items()}
        if score > state.best_dice:
            state.best_dice = score
            garimpo_uncertainty.checkpoints.save(
                run.best_model,
                {"model": weights, "meta": state.metadata, "epoch": epoch, "val_dice": dice},
            )
        garimpo_uncertainty.checkpoints.save(
            run.last_checkpoint,
            {
                "model": weights,
                "optimizer": state.optimizer.state_dict(),
                "meta": state.metadata,
                "arguments": state.arguments,
                "epoch": epoch,
                "best_dice": state.best_dice,
                "history": state.history,
                "stopped": state.stopped,
                "callbacks": state.callbacks,
                "rng": torch.get_rng_state(),
                # The DeepLabV3 head's dropout draws on the GPU, so a resume must restore that too;
                # only the device in use, so a run from a 2-GPU host resumes on a 1-GPU runtime.
                "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
            },
        )


@CALLBACK_REGISTRY.register("console_log")
class ConsoleLogCallback(garimpo_uncertainty.training.loop.Callback):
    """Prints one line per epoch, as training logs show it; history.csv keeps every value."""

    def on_fit_start(self, trainer: garimpo_uncertainty.training.loop.Trainer) -> None:
        self.epochs, self.previous_best = trainer.epochs, trainer.state.best_dice

    def on_epoch_end(self, state: TrainingState, record: dict) -> None:
        improved = state.best_dice > self.previous_best
        self.previous_best = state.best_dice
        remaining = self.epochs - record["epoch"] - 1
        eta = remaining * np.mean([epoch["seconds"] for epoch in state.history])
        peak = record["train_peak_gpu_bytes"]
        gpu = "" if peak is None else f" · {peak / 1024**3:.1f} GB"
        duration = garimpo_uncertainty.pipeline.runner.duration
        garimpo_uncertainty.pipeline.runner.show(
            f"epoch {record['epoch'] + 1}/{self.epochs} · loss {record['train_loss']:.4f} · "
            f"val_loss {record['val_loss']:.4f} · val_dice {record['val_dice']:.4f}"
            f"{' *' if improved else ''} · {duration(record['seconds'])}{gpu} · "
            f"eta {duration(eta)}"
        )


@CALLBACK_REGISTRY.register("early_stopping")
class EarlyStopping(garimpo_uncertainty.training.loop.Callback):
    """Stops after `patience` epochs in which `monitor` did not improve by more than `min_delta`.

    A NaN never improves. The protocol trains a fixed number of updates at every fraction, so
    configs/base.json leaves this out.
    """

    changes_state = True

    def __init__(
        self, patience: int, min_delta: float = 0.0, monitor: str = "val_dice", mode: str = "max"
    ) -> None:
        if not isinstance(patience, int) or patience < 1:
            raise ValueError(f"early_stopping: patience must be a positive integer, got {patience}")
        if mode not in {"max", "min"}:
            raise ValueError(f"early_stopping: mode must be max or min, got {mode!r}")
        if not min_delta >= 0:
            raise ValueError(f"early_stopping: min_delta must be 0 or more, got {min_delta}")
        self.patience, self.min_delta, self.monitor, self.mode = patience, min_delta, monitor, mode

    def improves(self, value: float, best: float | None) -> bool:
        if math.isnan(value):
            return False
        if best is None:
            return True
        return (
            value > best + self.min_delta if self.mode == "max" else value < best - self.min_delta
        )

    def on_epoch_end(self, state: TrainingState, record: dict) -> None:
        kept = state.callbacks.setdefault("early_stopping", {"best": None, "waited": 0})
        value = float(record[self.monitor])
        if self.improves(value, kept["best"]):
            kept["best"], kept["waited"] = value, 0
        else:
            kept["waited"] += 1
        if kept["waited"] >= self.patience:
            state.stopped = True
