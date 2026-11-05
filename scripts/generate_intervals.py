"""Stage: intervals of every combination on test cells; COMPASS rows run the decoder, never train.

Output-space rows are arithmetic on persisted terms. Writes intervals/intervals.npz: lower, upper
and quantile as [K, N] arrays, K indexing COMBINATIONS and N the test cells.
"""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import compass2d.compass
import compass2d.directions
import compass2d.metrics
import numpy as np
import pandas as pd
import torch

import garimpo_uncertainty.cell_dataset
import garimpo_uncertainty.checkpoints
import garimpo_uncertainty.data
import garimpo_uncertainty.intervals
import garimpo_uncertainty.models
import garimpo_uncertainty.stages


class SearchSpace(NamedTuple):
    """What one COMPASS method perturbs: the point, its direction, and how to read an area."""

    decode: Callable[[torch.Tensor], torch.Tensor]
    metric: Callable[[torch.Tensor], torch.Tensor]
    z: torch.Tensor
    delta: torch.Tensor

    def endpoints(self, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return compass2d.compass.endpoints(self.decode, self.metric, self.z, self.delta, beta)


def search_spaces(
    model: torch.nn.Module,
    features: list[torch.Tensor],
    jacobians: np.ndarray,
    fitted: compass2d.directions.Directions,
    args: argparse.Namespace,
) -> dict[str, SearchSpace]:
    """Return each COMPASS method's search space on this batch."""
    pixel_ha = garimpo_uncertainty.data.PIXEL_HA

    def soft(logits: torch.Tensor) -> torch.Tensor:
        return compass2d.metrics.soft_area(logits, pixel_ha, args.temperature)

    logits, _ = garimpo_uncertainty.models.split_at(
        model, garimpo_uncertainty.models.LOGITS, features
    )
    z, decode = garimpo_uncertainty.models.split_at(model, args.layer, features)
    spaces = {
        "compass_l": SearchSpace(
            lambda zi: zi,
            lambda output: compass2d.metrics.hard_area(output, pixel_ha),
            logits,
            torch.ones_like(logits),
        )
    }
    for direction in ("pca", "svm"):
        delta = compass2d.directions.cell_deltas(direction, jacobians, fitted)
        spaces[f"compass_j_{direction}"] = SearchSpace(
            decode,
            soft,
            z,
            compass2d.compass.channel_delta(torch.from_numpy(delta.copy()).to(z)),
        )
    return spaces


@torch.no_grad()
def compass_endpoints(
    model: torch.nn.Module,
    quantile: np.ndarray,
    cells: pd.DataFrame,
    jacobians: np.ndarray,
    fitted: compass2d.directions.Directions,
    *,
    root: Path,
    metadata: dict,
    args: argparse.Namespace,
    device: str,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    """Fill the COMPASS rows of lower and upper in one pass: one read and one encoder per batch."""
    rows = [
        (k, method)
        for k, (method, _) in enumerate(garimpo_uncertainty.intervals.COMBINATIONS)
        if method in garimpo_uncertainty.intervals.COMPASS_METHODS
    ]
    loader = garimpo_uncertainty.cell_dataset.loader(
        root, cells, metadata["normalization"], args.batch_size, args.num_workers
    )
    first = 0
    for images, _, _ in loader:
        last = first + len(images)
        features = garimpo_uncertainty.models.encode(model, images.to(device))
        batch_jacobians = jacobians[first:last].astype(np.float64)
        spaces = search_spaces(model, features, batch_jacobians, fitted, args)
        for k, method in rows:
            beta = torch.as_tensor(quantile[k, first:last], device=device)
            low, high = spaces[method].endpoints(beta)
            lower[k, first:last], upper[k, first:last] = low.cpu().numpy(), high.cpu().numpy()
        first = last


def read_jacobians(run_dir: Path, cell_index: np.ndarray) -> np.ndarray:
    with np.load(run_dir / "scores" / "scores.npz") as stored:
        return stored["jacobians"][
            garimpo_uncertainty.data.positions_of(stored["cell_index"], cell_index)
        ]


def generate(args: argparse.Namespace, root: Path, device: str, output: Path) -> dict:
    predictions = garimpo_uncertainty.data.read_predictions(root, args.run_dir)
    model, metadata = garimpo_uncertainty.checkpoints.load_model(
        args.checkpoint, predictions, device
    )
    fitted = compass2d.directions.Directions.load(args.run_dir / "directions" / f"{args.layer}.npz")
    with np.load(args.run_dir / "calibration" / "quantiles.npz") as stored:
        cell_index = stored["cell_index"]
        quantile = stored["quantile"]
        alpha = stored["alpha"]
        # Terms are NaN on COMPASS rows, which compass_endpoints then fills.
        lower = stored["lower_term"] - quantile * stored["scale"]
        upper = stored["upper_term"] + quantile * stored["scale"]
    test = predictions.iloc[
        garimpo_uncertainty.data.positions_of(predictions.cell_index.to_numpy(), cell_index)
    ]
    compass_endpoints(
        model,
        quantile,
        test,
        read_jacobians(args.run_dir, cell_index),
        fitted,
        root=root,
        metadata=metadata,
        args=args,
        device=device,
        lower=lower,
        upper=upper,
    )
    lower, upper = garimpo_uncertainty.intervals.clip_to_cell(lower, upper, test.cell_ha.to_numpy())
    garimpo_uncertainty.stages.write_arrays(
        output,
        cell_index=cell_index,
        lower=lower,
        upper=upper,
        quantile=quantile,
        alpha=alpha,
        layer=np.array(args.layer),
    )
    return {"lower": list(lower.shape), "test_cells": len(cell_index)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--temperature", type=float, default=compass2d.metrics.UPSTREAM_TEMPERATURE)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    garimpo_uncertainty.stages.log_to_stdout()

    root = garimpo_uncertainty.data.data_root()
    device = args.device or garimpo_uncertainty.checkpoints.default_device()
    run_dir = args.run_dir
    output = run_dir / "intervals" / "intervals.npz"
    inputs = [
        args.checkpoint,
        run_dir / "predictions" / "cells.npz",
        run_dir / "calibration" / "quantiles.npz",
        run_dir / "scores" / "scores.npz",
        run_dir / "directions" / f"{args.layer}.npz",
    ]
    garimpo_uncertainty.stages.run_stage(
        run_dir / "status",
        "generate_intervals",
        [output],
        garimpo_uncertainty.stages.fingerprint(
            args, inputs, hashed=garimpo_uncertainty.data.run_identity(root, run_dir)
        ),
        lambda: generate(args, root, device, output),
        force=args.force,
    )


if __name__ == "__main__":
    main()
