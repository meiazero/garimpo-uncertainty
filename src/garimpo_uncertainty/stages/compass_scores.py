"""Stage: COMPASS non-conformity scores on validation, calibration and test cells.

Writes scores/scores.npz: symmetric and one-sided (Algorithm 6) scores and saturation flags, one row
per COMPASS method, plus per-cell Jacobians (PCA directions, interval stage) and summed features
(density ratio), all by position.
"""

import argparse
from pathlib import Path
from typing import NamedTuple

import compass2d.compass
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


class BatchScores(NamedTuple):
    """One batch: symmetric score and flag, then each one-sided score and flag, per method."""

    score: np.ndarray
    saturated: np.ndarray
    lower: np.ndarray
    lower_saturated: np.ndarray
    upper: np.ndarray
    upper_saturated: np.ndarray
    jacobians: np.ndarray
    feature_sums: np.ndarray


def batch_spaces(
    model: torch.nn.Module,
    features: list[torch.Tensor],
    fitted: compass2d.directions.Directions,
    args: argparse.Namespace,
) -> tuple[list[tuple[garimpo_uncertainty.compass_j.Space, float, float]], np.ndarray, np.ndarray]:
    """Return each space in COMPASS_METHODS order with its limits, the Jacobians, the features."""
    split = garimpo_uncertainty.compass_j.split(model, features, args.layer)
    jacobian = garimpo_uncertainty.compass_j.jacobian(
        split.decode, split.z, args.direction_temperature
    )
    spaces = garimpo_uncertainty.compass_j.spaces(split, jacobian, fitted, args.temperature)
    logits_limits = (args.beta_range_logits, args.search_threshold_logits)
    features_limits = (args.beta_range_features, args.search_threshold_features)
    searched = [
        (spaces[method], *(logits_limits if method == "compass_l" else features_limits))
        for method in garimpo_uncertainty.intervals.COMPASS_METHODS
    ]
    return searched, jacobian, split.z.flatten(2).sum(2).cpu().numpy()


def search_space(
    space: garimpo_uncertainty.compass_j.Space,
    beta_range: float,
    threshold: float,
    target: torch.Tensor,
) -> list[torch.Tensor]:
    """Symmetric score and flag, then lower score and flag, then upper score and flag."""
    searched = (space.decode, space.metric, space.z, space.delta, target)
    limits = (beta_range, threshold)
    lower, upper = compass2d.compass.asymmetric_binary_search_scores(*searched, *limits)
    return [*compass2d.compass.binary_search_scores(*searched, *limits), *lower, *upper]


def score_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    masks: torch.Tensor,
    fitted: compass2d.directions.Directions,
    args: argparse.Namespace,
) -> BatchScores:
    """Search each method's scores on one batch, sharing one encoder pass; one row per method."""
    target = masks.to(torch.float64).flatten(1).sum(1) * garimpo_uncertainty.data.PIXEL_HA
    features = garimpo_uncertainty.models.common.encode(model, x)
    spaces, jacobian, feature_sums = batch_spaces(model, features, fitted, args)
    found = [search_space(*space, target) for space in spaces]
    columns = [np.stack([row[c].cpu().numpy() for row in found]) for c in range(6)]
    return BatchScores(*columns, jacobian, feature_sums)


def score_cells(
    model: torch.nn.Module,
    cells: pd.DataFrame,
    root: Path,
    metadata: dict,
    args: argparse.Namespace,
    *,
    device: str,
) -> dict[str, np.ndarray]:
    """Return arrays aligned with `cells`; scores have one row per COMPASS_METHODS entry."""
    run = garimpo_uncertainty.artifacts.RunPaths.of(args.run_dir)
    fitted = compass2d.directions.Directions.load(run.directions(args.layer))
    methods = garimpo_uncertainty.intervals.COMPASS_METHODS
    arrays = {
        name: np.empty((len(methods), len(cells)), dtype=bool if "saturated" in name else float)
        for name in BatchScores._fields[:6]
    }
    jacobians, feature_sums = [], []
    loader = garimpo_uncertainty.cell_dataset.loader(
        root, cells, metadata["normalization"], args.batch_size, args.num_workers
    )
    first = 0
    for images, masks, _ in garimpo_uncertainty.cell_dataset.batches(loader):
        last = first + len(images)
        batch = score_batch(model, images.to(device), masks.to(device), fitted, args)
        for name, array in arrays.items():
            array[:, first:last] = getattr(batch, name)
        jacobians.append(batch.jacobians)
        feature_sums.append(batch.feature_sums)
        first = last
    return {
        "cell_index": cells.cell_index.to_numpy(),
        **arrays,
        "beta_range": np.array(
            [args.beta_range_logits, args.beta_range_features, args.beta_range_features]
        ),
        "jacobians": np.concatenate(jacobians),
        "feature_sums": np.concatenate(feature_sums),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--layer", required=True)
    garimpo_uncertainty.pipeline.runner.add_layer_dir(parser)
    parser.add_argument("--beta-range-logits", type=float, required=True)
    parser.add_argument("--beta-range-features", type=float, required=True)
    parser.add_argument("--search-threshold-logits", type=float, required=True)
    parser.add_argument("--search-threshold-features", type=float, required=True)
    parser.add_argument("--temperature", type=float, default=compass2d.metrics.UPSTREAM_TEMPERATURE)
    parser.add_argument(
        "--direction-temperature",
        type=float,
        required=True,
        help="sigmoid temperature of the area whose Jacobian sets the direction; at 1e4 the "
        "gradient vanishes beyond |logit| ~ 5e-3, so cells predicted empty get the mean direction",
    )
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
    inputs = [args.checkpoint, run.directions(args.layer)]

    def work() -> dict:
        cells = garimpo_uncertainty.data.read_run_cells(root, args.run_dir)
        model, metadata = garimpo_uncertainty.checkpoints.load_model(args.checkpoint, cells, device)
        # Validation cells fix the asymmetric rows' side ratio, apart from calibration.
        evaluated = cells[cells.role.isin(garimpo_uncertainty.data.EVALUATED_ROLES)]
        arrays = score_cells(model, evaluated, root, metadata, args, device=device)
        garimpo_uncertainty.files.write_arrays(run.scores, **arrays)
        return {
            "jacobians": list(arrays["jacobians"].shape),
            "feature_sums": list(arrays["feature_sums"].shape),
            "saturated": dict(
                zip(
                    garimpo_uncertainty.intervals.COMPASS_METHODS,
                    arrays["saturated"].sum(axis=1).tolist(),
                    strict=True,
                )
            ),
        }

    garimpo_uncertainty.pipeline.runner.run_stage(
        run.layer_status,
        "compass_scores",
        [run.scores],
        garimpo_uncertainty.pipeline.runner.fingerprint(
            args, inputs, hashed=garimpo_uncertainty.data.run_identity(root, args.run_dir)
        ),
        work,
        force=args.force,
    )
