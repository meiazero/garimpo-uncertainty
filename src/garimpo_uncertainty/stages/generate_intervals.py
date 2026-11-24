"""Stage: intervals of every combination on test cells; COMPASS rows run the decoder, never train.

Output-space rows are arithmetic on persisted terms. Writes intervals/intervals.npz: lower, upper
and quantile as [A, K, N] arrays, A indexing ALPHAS, K COMBINATIONS and N the test cells.
"""

import argparse
import itertools
from pathlib import Path
from typing import NamedTuple, Self

import compass2d.directions
import compass2d.metrics
import numpy as np
import pandas as pd
import torch

import garimpo_uncertainty.artifacts
import garimpo_uncertainty.cell_dataset
import garimpo_uncertainty.checkpoints
import garimpo_uncertainty.compass_j
import garimpo_uncertainty.data
import garimpo_uncertainty.files
import garimpo_uncertainty.intervals
import garimpo_uncertainty.models.common
import garimpo_uncertainty.pipeline.runner

COMPASS_ROWS = (
    garimpo_uncertainty.intervals.COMPASS_METHODS + garimpo_uncertainty.intervals.ASYMMETRIC_METHODS
)


class Calibrated(NamedTuple):
    """Quantiles and, per level and row, the censoring point and side ratio, from quantiles.npz.

    censor_at is NaN on output-space rows and side_ratio on every row but the asymmetric ones.
    """

    quantile: np.ndarray
    censor_at: np.ndarray
    side_ratio: np.ndarray

    @classmethod
    def read(cls, path: Path) -> Self:
        with np.load(path) as stored:
            return cls(stored["quantile"], stored["censor_at"], stored["side_ratio"])


def row_endpoints(
    space: garimpo_uncertainty.compass_j.Space, method: str, beta: torch.Tensor, ratio: float
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
        spaces = garimpo_uncertainty.compass_j.spaces(
            garimpo_uncertainty.compass_j.split(model, features, args.layer),
            jacobians[first:last],
            fitted,
            args.temperature,
        )
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


def read_jacobians(scores_path: Path, cell_index: np.ndarray) -> np.ndarray:
    with np.load(scores_path) as stored:
        return stored["jacobians"][
            garimpo_uncertainty.data.positions_of(stored["cell_index"], cell_index)
        ]


def generate(
    args: argparse.Namespace, root: Path, device: str, run: garimpo_uncertainty.artifacts.RunPaths
) -> dict:
    predictions = garimpo_uncertainty.data.read_predictions(root, run.root)
    model, metadata = garimpo_uncertainty.checkpoints.load_model(
        args.checkpoint, predictions, device
    )
    fitted = compass2d.directions.Directions.load(run.directions(args.layer))
    calibrated = Calibrated.read(run.quantiles)
    with np.load(run.quantiles) as stored:
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
        read_jacobians(run.scores, cell_index),
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
    garimpo_uncertainty.files.write_arrays(
        run.intervals,
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
    garimpo_uncertainty.pipeline.runner.add_layer_dir(parser)
    parser.add_argument("--temperature", type=float, default=compass2d.metrics.UPSTREAM_TEMPERATURE)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device")
    parser.add_argument(
        "--num-workers", type=int, default=garimpo_uncertainty.pipeline.runner.available_cpus()
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    garimpo_uncertainty.pipeline.runner.log_to_stdout()

    root = garimpo_uncertainty.data.data_root()
    device = args.device or garimpo_uncertainty.checkpoints.default_device()
    run = garimpo_uncertainty.artifacts.RunPaths.of(args.run_dir, args.layer_dir)
    inputs = [
        args.checkpoint,
        run.predictions,
        run.quantiles,
        run.scores,
        run.directions(args.layer),
    ]
    garimpo_uncertainty.pipeline.runner.run_stage(
        run.layer_status,
        "generate_intervals",
        [run.intervals],
        garimpo_uncertainty.pipeline.runner.fingerprint(
            args, inputs, hashed=garimpo_uncertainty.data.run_identity(root, run.root)
        ),
        lambda: generate(args, root, device, run),
        force=args.force,
    )
