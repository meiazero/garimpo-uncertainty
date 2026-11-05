"""Stage: COMPASS non-conformity scores on calibration and test cells, through the decoder.

Writes scores/scores.npz: scores and saturation flags, one row per COMPASS method, plus per-cell
Jacobians (PCA directions, interval stage) and summed features (density ratio), all by position.
"""

import argparse
from pathlib import Path

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


def score_cells(
    model: torch.nn.Module,
    cells: pd.DataFrame,
    root: Path,
    metadata: dict,
    args: argparse.Namespace,
    device: str,
) -> dict[str, np.ndarray]:
    """Return arrays aligned with `cells`; scores have one row per COMPASS_METHODS entry."""
    pixel_ha = garimpo_uncertainty.data.PIXEL_HA
    fitted = compass2d.directions.Directions.load(args.run_dir / "directions" / f"{args.layer}.npz")

    def hard(logits: torch.Tensor) -> torch.Tensor:
        return compass2d.metrics.hard_area(logits, pixel_ha)

    def soft(logits: torch.Tensor) -> torch.Tensor:
        return compass2d.metrics.soft_area(logits, pixel_ha, args.temperature)

    methods = garimpo_uncertainty.intervals.COMPASS_METHODS
    score = np.empty((len(methods), len(cells)))
    saturated = np.empty((len(methods), len(cells)), dtype=bool)
    jacobians, feature_sums = [], []
    loader = garimpo_uncertainty.cell_dataset.loader(
        root, cells, metadata["normalization"], args.batch_size, args.num_workers
    )
    first = 0
    for images, masks, _ in loader:
        last = first + len(images)
        x = images.to(device)
        target = masks.to(device).to(torch.float64).flatten(1).sum(1) * pixel_ha
        features = garimpo_uncertainty.models.encode(model, x)
        logits, _ = garimpo_uncertainty.models.split_at(
            model, garimpo_uncertainty.models.LOGITS, features
        )
        found, flags = compass2d.compass.binary_search_scores(
            lambda z: z,
            hard,
            logits,
            torch.ones_like(logits),
            target,
            args.beta_range_logits,
            args.search_threshold_logits,
        )
        score[0, first:last], saturated[0, first:last] = found.cpu().numpy(), flags.cpu().numpy()

        z, decode = garimpo_uncertainty.models.split_at(model, args.layer, features)
        jacobian = compass2d.directions.jacobian_sum(decode, z, soft).cpu().numpy()
        jacobians.append(jacobian)
        feature_sums.append(z.flatten(2).sum(2).cpu().numpy())
        for row, direction in ((1, "pca"), (2, "svm")):
            delta = compass2d.directions.cell_deltas(direction, jacobian.astype(np.float64), fitted)
            found, flags = compass2d.compass.binary_search_scores(
                decode,
                soft,
                z,
                compass2d.compass.channel_delta(torch.from_numpy(delta.copy()).to(z)),
                target,
                args.beta_range_features,
                args.search_threshold_features,
            )
            score[row, first:last] = found.cpu().numpy()
            saturated[row, first:last] = flags.cpu().numpy()
        first = last
    return {
        "cell_index": cells.cell_index.to_numpy(),
        "score": score,
        "saturated": saturated,
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
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = garimpo_uncertainty.data.data_root()
    device = args.device or garimpo_uncertainty.checkpoints.default_device()
    scores_path = args.run_dir / "scores" / "scores.npz"
    inputs = [args.checkpoint, args.run_dir / "directions" / f"{args.layer}.npz"]

    def work() -> dict:
        cells = garimpo_uncertainty.data.read_run_cells(root, args.run_dir)
        model, metadata = garimpo_uncertainty.checkpoints.load_model(args.checkpoint, cells, device)
        evaluated = cells[cells.role.isin(["calibration", "test"])]
        arrays = score_cells(model, evaluated, root, metadata, args, device)
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
