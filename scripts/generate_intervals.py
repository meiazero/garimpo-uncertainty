"""Stage: intervals of every combination on test cells; COMPASS rows run the decoder, never train.

Output-space rows are arithmetic on persisted terms. Writes intervals/intervals.npz: lower, upper
and quantile as [A, K, N] arrays, A indexing ALPHAS, K COMBINATIONS and N the test cells.
"""

import argparse
import itertools
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, Self

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
import garimpo_uncertainty.models.common
import garimpo_uncertainty.stages

COMPASS_ROWS = (
    garimpo_uncertainty.intervals.COMPASS_METHODS + garimpo_uncertainty.intervals.ASYMMETRIC_METHODS
)


class SearchSpace(NamedTuple):
    """What one COMPASS method perturbs: the point, its direction, and how to read an area."""

    decode: Callable[[torch.Tensor], torch.Tensor]
    metric: Callable[[torch.Tensor], torch.Tensor]
    z: torch.Tensor
    delta: torch.Tensor

    def endpoints(self, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return compass2d.compass.endpoints(self.decode, self.metric, self.z, self.delta, beta)

    def shifted(self, beta: torch.Tensor) -> torch.Tensor:
        return compass2d.compass.shifted(self.decode, self.metric, self.z, self.delta, beta)


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

    logits, _ = garimpo_uncertainty.models.common.split_at(
        model, garimpo_uncertainty.models.common.LOGITS, features
    )
    z, decode = garimpo_uncertainty.models.common.split_at(model, args.layer, features)
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


class Calibrated(NamedTuple):
    """Quantiles and, per level and row, the censoring point and side ratio, from quantiles.npz.

    censor_at is NaN on output-space rows and side_ratio on every row but the asymmetric ones.
    """

    quantile: np.ndarray
    censor_at: np.ndarray
    side_ratio: np.ndarray

    @classmethod
    def read(cls, run_dir: Path) -> Self:
        with np.load(run_dir / "calibration" / "quantiles.npz") as stored:
            return cls(stored["quantile"], stored["censor_at"], stored["side_ratio"])


def row_endpoints(
    space: SearchSpace, method: str, beta: torch.Tensor, ratio: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric endpoints, or [m(z - beta delta), m(z + ratio beta delta)] for an a_ row."""
    if method in garimpo_uncertainty.intervals.ASYMMETRIC_METHODS:
        return space.shifted(-beta), space.shifted(ratio * beta)
    return space.endpoints(beta)


@torch.no_grad()
def compass_endpoints(
    model: torch.nn.Module,
    calibrated: Calibrated,
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
    """Fill the COMPASS rows of [A, K, N] lower and upper in one pass.

    One read and one encoder per batch, whatever the number of levels.
    """
    rows = [
        (k, method)
        for k, (method, _) in enumerate(garimpo_uncertainty.intervals.COMBINATIONS)
        if method in COMPASS_ROWS
    ]
    loader = garimpo_uncertainty.cell_dataset.loader(
        root, cells, metadata["normalization"], args.batch_size, args.num_workers
    )
    first = 0
    for images, _, _ in garimpo_uncertainty.cell_dataset.batches(loader):
        last = first + len(images)
        features = garimpo_uncertainty.models.common.encode(model, images.to(device))
        batch_jacobians = jacobians[first:last].astype(np.float64)
        spaces = search_spaces(model, features, batch_jacobians, fitted, args)
        for a, (k, method) in itertools.product(range(len(calibrated.quantile)), rows):
            beta = torch.as_tensor(calibrated.quantile[a, k, first:last], device=device)
            # Censored rows become the whole cell in widen_saturated; 0 keeps inf from the decoder.
            beta = torch.where(beta >= float(calibrated.censor_at[a, k]), 0.0, beta)
            low, high = row_endpoints(
                spaces[method.removeprefix("a_")], method, beta, float(calibrated.side_ratio[a, k])
            )
            lower[a, k, first:last] = low.cpu().numpy()
            upper[a, k, first:last] = high.cpu().numpy()
        first = last


def widen_saturated(
    calibrated: Calibrated, cell_ha: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> None:
    """Turn every COMPASS interval whose quantile reached its censoring point into the whole cell.

    Such a quantile is censored: the true one may be any larger, so nothing narrower is supported.
    Output-space rows have no censoring point (NaN), so the comparison leaves them alone.
    """
    trivial = calibrated.quantile >= calibrated.censor_at[:, :, None]
    lower[trivial] = 0.0
    upper[:] = np.where(trivial, cell_ha, upper)


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
    calibrated = Calibrated.read(args.layer_dir)
    with np.load(args.layer_dir / "calibration" / "quantiles.npz") as stored:
        cell_index = stored["cell_index"]
        alpha = stored["alpha"]
        # Terms are NaN on COMPASS rows, which compass_endpoints then fills.
        lower = stored["lower_term"] - calibrated.quantile * stored["scale"]
        upper = stored["upper_term"] + calibrated.quantile * stored["scale"]
    test = predictions.iloc[
        garimpo_uncertainty.data.positions_of(predictions.cell_index.to_numpy(), cell_index)
    ]
    compass_endpoints(
        model,
        calibrated,
        test,
        read_jacobians(args.layer_dir, cell_index),
        fitted,
        root=root,
        metadata=metadata,
        args=args,
        device=device,
        lower=lower,
        upper=upper,
    )
    widen_saturated(calibrated, test.cell_ha.to_numpy(), lower, upper)
    lower, upper = garimpo_uncertainty.intervals.clip_to_cell(lower, upper, test.cell_ha.to_numpy())
    garimpo_uncertainty.stages.write_arrays(
        output,
        cell_index=cell_index,
        lower=lower,
        upper=upper,
        quantile=calibrated.quantile,
        alpha=alpha,
        layer=np.array(args.layer),
    )
    return {"lower": list(lower.shape), "test_cells": len(cell_index)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--layer", required=True)
    garimpo_uncertainty.stages.add_layer_dir(parser)
    parser.add_argument("--temperature", type=float, default=compass2d.metrics.UPSTREAM_TEMPERATURE)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.layer_dir = args.layer_dir or args.run_dir
    garimpo_uncertainty.stages.log_to_stdout()

    root = garimpo_uncertainty.data.data_root()
    device = args.device or garimpo_uncertainty.checkpoints.default_device()
    run_dir, layer_dir = args.run_dir, args.layer_dir
    output = layer_dir / "intervals" / "intervals.npz"
    inputs = [
        args.checkpoint,
        run_dir / "predictions" / "cells.npz",
        layer_dir / "calibration" / "quantiles.npz",
        layer_dir / "scores" / "scores.npz",
        run_dir / "directions" / f"{args.layer}.npz",
    ]
    garimpo_uncertainty.stages.run_stage(
        layer_dir / "status",
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
