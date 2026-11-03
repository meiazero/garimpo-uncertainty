"""Stage: intervals of every combination on test cells; COMPASS rows run the decoder, never train.

Output-space rows are arithmetic on persisted terms. Writes intervals/intervals.csv.
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
import garimpo_uncertainty.intervals
import garimpo_uncertainty.stages


@torch.no_grad()
def compass_endpoints(
    model: torch.nn.Module,
    method: str,
    quantiles: pd.Series,
    cells: pd.DataFrame,
    jacobians: dict[str, np.ndarray],
    fitted: compass2d.directions.Directions,
    root: Path,
    metadata: dict,
    args: argparse.Namespace,
    device: str,
) -> tuple[dict, dict]:
    pixel_ha = garimpo_uncertainty.data.PIXEL_HA
    lower, upper = {}, {}
    loader = garimpo_uncertainty.cell_dataset.loader(
        root, cells, metadata["normalization"], args.batch_size, args.num_workers
    )
    for images, _, cell_ids in loader:
        x = images.to(device)
        beta = torch.as_tensor(quantiles.loc[list(cell_ids)].to_numpy(), device=device)
        if method == "compass_l":
            z = model(x)
            low, high = compass2d.compass.endpoints(
                lambda zi: zi,
                lambda logits: compass2d.metrics.hard_area(logits, pixel_ha),
                z,
                torch.ones_like(z),
                beta,
            )
        else:
            z = compass2d.decomposition.encode(model, args.layer, x)
            batch_jacobians = np.stack([jacobians[c] for c in cell_ids]).astype(np.float64)
            direction = method.removeprefix("compass_j_")
            delta = compass2d.directions.cell_deltas(direction, batch_jacobians, fitted)
            low, high = compass2d.compass.endpoints(
                lambda zi, x=x: compass2d.decomposition.decode(model, args.layer, x, zi),
                lambda logits: compass2d.metrics.soft_area(logits, pixel_ha, args.temperature),
                z,
                compass2d.compass.channel_delta(torch.from_numpy(delta.copy()).to(z)),
                beta,
            )
        lower.update(zip(cell_ids, low.cpu().tolist(), strict=True))
        upper.update(zip(cell_ids, high.cpu().tolist(), strict=True))
    return lower, upper


def generate(args: argparse.Namespace, root: Path, device: str, output: Path) -> dict:
    run_dir = args.run_dir
    manifest = garimpo_uncertainty.data.read_manifest(root)
    model, metadata = garimpo_uncertainty.checkpoints.load_model(args.checkpoint, manifest, device)
    cells = pd.read_csv(run_dir / "predictions" / "cells.csv").set_index("cell_id")
    quantiles = pd.read_csv(run_dir / "calibration" / "quantiles.csv")
    alpha = pd.read_csv(run_dir / "calibration" / "summary.csv").alpha.iloc[0]
    with np.load(run_dir / "scores" / "eval_features.npz") as stored:
        jacobians = dict(zip(stored["cell_id"].tolist(), stored["jacobians"], strict=True))
    fitted = compass2d.directions.Directions.load(run_dir / "directions" / f"{args.layer}.npz")

    test_ids = quantiles.cell_id.drop_duplicates()
    test_manifest = manifest.set_index("cell_id").loc[test_ids].reset_index()
    frames = []
    for (method, calibration), rows in quantiles.groupby(["method", "calibration"], sort=False):
        rows = rows.set_index("cell_id")
        if method in garimpo_uncertainty.intervals.COMPASS_METHODS:
            low, high = compass_endpoints(
                model,
                method,
                rows.quantile_value,
                test_manifest,
                jacobians,
                fitted,
                root,
                metadata,
                args,
                device,
            )
            lower = rows.index.map(low).to_numpy(dtype=float)
            upper = rows.index.map(high).to_numpy(dtype=float)
        else:
            lower = (rows.lower_term - rows.quantile_value * rows.scale).to_numpy()
            upper = (rows.upper_term + rows.quantile_value * rows.scale).to_numpy()
        one = cells.loc[rows.index]
        lower, upper = garimpo_uncertainty.intervals.clip_to_cell(
            lower, upper, one.cell_ha.to_numpy()
        )
        frames.append(
            pd.DataFrame(
                {
                    "model": metadata["arch"],
                    "layer": args.layer,
                    "alpha": alpha,
                    "method": method,
                    "calibration": calibration,
                    "cell_id": rows.index,
                    "group": rows.group.to_numpy(),
                    "quantile_value": rows.quantile_value.to_numpy(),
                    "lower": lower,
                    "upper": upper,
                    "v": one.v.to_numpy(),
                    "vhat": one.vhat.to_numpy(),
                    "cell_ha": one.cell_ha.to_numpy(),
                }
            )
        )
    intervals = pd.concat(frames, ignore_index=True)
    with garimpo_uncertainty.stages.atomic(output) as temporary:
        intervals.to_csv(temporary, index=False)
    return {"rows": len(intervals), "test_cells": len(test_ids), "combinations": len(frames)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--temperature", type=float, default=compass2d.metrics.UPSTREAM_TEMPERATURE)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = garimpo_uncertainty.data.data_root()
    device = args.device or garimpo_uncertainty.checkpoints.default_device()
    run_dir = args.run_dir
    output = run_dir / "intervals" / "intervals.csv"
    inputs = [
        args.checkpoint,
        root / "manifest.csv",
        run_dir / "predictions" / "cells.csv",
        run_dir / "calibration" / "quantiles.csv",
        run_dir / "scores" / "eval_features.npz",
    ]
    garimpo_uncertainty.stages.run_stage(
        run_dir / "status",
        "generate_intervals",
        [output],
        args.force,
        garimpo_uncertainty.stages.fingerprint(args, inputs),
        lambda: generate(args, root, device, output),
    )


if __name__ == "__main__":
    main()
