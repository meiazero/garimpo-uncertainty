"""Stage: COMPASS non-conformity scores on validation, calibration and test cells.

Writes scores/scores.npz: symmetric and one-sided (Algorithm 6) scores and saturation flags, one row
per COMPASS method, plus per-cell Jacobians (PCA directions, interval stage) and summed features
(density ratio), all by position.
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


class Space(NamedTuple):
    """What one COMPASS method perturbs, and how far and how finely its search goes."""

    decode: Callable[[torch.Tensor], torch.Tensor]
    metric: Callable[[torch.Tensor], torch.Tensor]
    z: torch.Tensor
    delta: torch.Tensor
    beta_range: float
    threshold: float


def batch_spaces(
    model: torch.nn.Module,
    features: list[torch.Tensor],
    fitted: compass2d.directions.Directions,
    args: argparse.Namespace,
) -> tuple[list[Space], np.ndarray, np.ndarray]:
    """Return the spaces in COMPASS_METHODS order, the Jacobians and the summed features."""
    pixel_ha = garimpo_uncertainty.data.PIXEL_HA

    def soft(logits: torch.Tensor) -> torch.Tensor:
        return compass2d.metrics.soft_area(logits, pixel_ha, args.temperature)

    logits, _ = garimpo_uncertainty.models.split_at(
        model, garimpo_uncertainty.models.LOGITS, features
    )
    spaces = [
        Space(
            lambda z: z,
            lambda output: compass2d.metrics.hard_area(output, pixel_ha),
            logits,
            torch.ones_like(logits),
            args.beta_range_logits,
            args.search_threshold_logits,
        )
    ]
    z, decode = garimpo_uncertainty.models.split_at(model, args.layer, features)
    jacobian = compass2d.directions.jacobian_sum(
        decode,
        z,
        lambda logits: compass2d.metrics.soft_area(
            logits, garimpo_uncertainty.data.PIXEL_HA, args.direction_temperature
        ),
    )
    jacobian = jacobian.cpu().numpy()
    for direction in ("pca", "svm"):
        delta = compass2d.directions.cell_deltas(direction, jacobian.astype(np.float64), fitted)
        spaces.append(
            Space(
                decode,
                soft,
                z,
                compass2d.compass.channel_delta(torch.from_numpy(delta.copy()).to(z)),
                args.beta_range_features,
                args.search_threshold_features,
            )
        )
    return spaces, jacobian, z.flatten(2).sum(2).cpu().numpy()


def search_space(space: Space, target: torch.Tensor) -> list[torch.Tensor]:
    """Symmetric score and flag, then lower score and flag, then upper score and flag."""
    searched = (space.decode, space.metric, space.z, space.delta, target)
    limits = (space.beta_range, space.threshold)
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
    features = garimpo_uncertainty.models.encode(model, x)
    spaces, jacobian, feature_sums = batch_spaces(model, features, fitted, args)
    found = [search_space(space, target) for space in spaces]
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
    fitted = compass2d.directions.Directions.load(args.run_dir / "directions" / f"{args.layer}.npz")
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
    for images, masks, _ in loader:
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
    # Upstream's active runner values for the range, and the COMPASSJ default for the threshold.
    parser.add_argument("--beta-range-logits", type=float, default=100.0)
    parser.add_argument("--beta-range-features", type=float, default=1000.0)
    parser.add_argument("--search-threshold-logits", type=float, default=1e-2)
    parser.add_argument("--search-threshold-features", type=float, default=1e-8)
    parser.add_argument("--temperature", type=float, default=compass2d.metrics.UPSTREAM_TEMPERATURE)
    parser.add_argument(
        "--direction-temperature",
        type=float,
        default=compass2d.metrics.UPSTREAM_TEMPERATURE,
        help="sigmoid temperature of the area whose Jacobian sets the direction; at 1e4 the "
        "gradient vanishes beyond |logit| ~ 5e-3, so cells predicted empty get the mean direction",
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    garimpo_uncertainty.stages.log_to_stdout()

    root = garimpo_uncertainty.data.data_root()
    device = args.device or garimpo_uncertainty.checkpoints.default_device()
    scores_path = args.run_dir / "scores" / "scores.npz"
    inputs = [args.checkpoint, args.run_dir / "directions" / f"{args.layer}.npz"]

    def work() -> dict:
        cells = garimpo_uncertainty.data.read_run_cells(root, args.run_dir)
        model, metadata = garimpo_uncertainty.checkpoints.load_model(args.checkpoint, cells, device)
        # Validation cells fix the asymmetric rows' side ratio, apart from calibration.
        evaluated = cells[cells.role.isin(["validation", "calibration", "test"])]
        arrays = score_cells(model, evaluated, root, metadata, args, device=device)
        garimpo_uncertainty.stages.write_arrays(scores_path, **arrays)
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

    garimpo_uncertainty.stages.run_stage(
        args.run_dir / "status",
        "compass_scores",
        [scores_path],
        garimpo_uncertainty.stages.fingerprint(
            args, inputs, hashed=garimpo_uncertainty.data.run_identity(root, args.run_dir)
        ),
        work,
        force=args.force,
    )


if __name__ == "__main__":
    main()
