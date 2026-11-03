"""Stage: COMPASS non-conformity scores on calibration and test cells, through the decoder.

Writes scores/compass_scores.csv with a saturation flag per score, and scores/eval_features.npz with
the per-cell Jacobians (PCA directions, interval stage) and summed features (density ratio).
"""

import argparse
from pathlib import Path

import compass2d.compass
import compass2d.decomposition
import compass2d.directions
import compass2d.metrics
import numpy as np
import pandas as pd
import torch

import garimpo_uncertainty.cell_dataset
import garimpo_uncertainty.checkpoints
import garimpo_uncertainty.data
import garimpo_uncertainty.stages


def score_cells(
    model: torch.nn.Module,
    cells: pd.DataFrame,
    root: Path,
    metadata: dict,
    args: argparse.Namespace,
    device: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    pixel_ha = garimpo_uncertainty.data.PIXEL_HA
    fitted = compass2d.directions.Directions.load(args.run_dir / "directions" / f"{args.layer}.npz")

    def hard(logits: torch.Tensor) -> torch.Tensor:
        return compass2d.metrics.hard_area(logits, pixel_ha)

    def soft(logits: torch.Tensor) -> torch.Tensor:
        return compass2d.metrics.soft_area(logits, pixel_ha, args.temperature)

    rows, jacobians, feature_sums = [], [], []

    def record(method: str, cell_ids: list[str], scores, saturated, beta_range: float) -> None:
        rows.extend(
            {
                "cell_id": cell_id,
                "method": method,
                "score": score,
                "saturated": flag,
                "beta_range": beta_range,
            }
            for cell_id, score, flag in zip(
                cell_ids, scores.cpu().tolist(), saturated.cpu().tolist(), strict=True
            )
        )

    loader = garimpo_uncertainty.cell_dataset.loader(
        root, cells, metadata["normalization"], args.batch_size, args.num_workers
    )
    for images, masks, cell_ids in loader:
        x = images.to(device)
        target = masks.to(device).to(torch.float64).flatten(1).sum(1) * pixel_ha
        with torch.no_grad():
            logits = model(x)
        scores, saturated = compass2d.compass.binary_search_scores(
            lambda z: z,
            hard,
            logits,
            torch.ones_like(logits),
            target,
            args.beta_range_logits,
            args.search_threshold_logits,
        )
        record("compass_l", cell_ids, scores, saturated, args.beta_range_logits)

        z = compass2d.decomposition.encode(model, args.layer, x)

        def decode(zi: torch.Tensor, x: torch.Tensor = x) -> torch.Tensor:
            return compass2d.decomposition.decode(model, args.layer, x, zi)

        jacobian = compass2d.directions.jacobian_sum(decode, z, soft).cpu().numpy()
        jacobians.append(jacobian)
        feature_sums.append(z.flatten(2).sum(2).cpu().numpy())
        for direction in ("pca", "svm"):
            delta = compass2d.directions.cell_deltas(direction, jacobian.astype(np.float64), fitted)
            scores, saturated = compass2d.compass.binary_search_scores(
                decode,
                soft,
                z,
                compass2d.compass.channel_delta(torch.from_numpy(delta.copy()).to(z)),
                target,
                args.beta_range_features,
                args.search_threshold_features,
            )
            record(f"compass_j_{direction}", cell_ids, scores, saturated, args.beta_range_features)
    return pd.DataFrame(rows), np.concatenate(jacobians), np.concatenate(feature_sums)


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
    scores_path = args.run_dir / "scores" / "compass_scores.csv"
    features_path = args.run_dir / "scores" / "eval_features.npz"
    inputs = [
        args.checkpoint,
        root / "manifest.csv",
        args.run_dir / "directions" / f"{args.layer}.npz",
    ]

    def work() -> dict:
        manifest = garimpo_uncertainty.data.read_manifest(root)
        model, metadata = garimpo_uncertainty.checkpoints.load_model(
            args.checkpoint, manifest, device
        )
        cells = manifest[manifest.role.isin(["calibration", "test"])]
        scores, jacobians, feature_sums = score_cells(model, cells, root, metadata, args, device)
        with garimpo_uncertainty.stages.atomic(scores_path) as temporary:
            scores.to_csv(temporary, index=False)
        with garimpo_uncertainty.stages.atomic(features_path) as temporary:
            with temporary.open("wb") as file:
                np.savez(
                    file,
                    cell_id=cells.cell_id.to_numpy(dtype=str),
                    jacobians=jacobians,
                    feature_sums=feature_sums,
                )
        return {
            "jacobians": list(jacobians.shape),
            "feature_sums": list(feature_sums.shape),
            "saturated": scores.groupby("method").saturated.sum().to_dict(),
        }

    garimpo_uncertainty.stages.run_stage(
        args.run_dir / "status",
        "compass_scores",
        [scores_path, features_path],
        args.force,
        garimpo_uncertainty.stages.fingerprint(args, inputs),
        work,
    )


if __name__ == "__main__":
    main()
