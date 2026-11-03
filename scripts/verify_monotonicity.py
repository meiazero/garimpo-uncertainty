"""Stage: Algorithm 2 sweep per layer and direction; writes monotonicity_<use>.csv.

--use layer informs the choice of layer, so it reads the validation role only, never test.
--use endpoint checks the chosen layer on the cells the endpoint search actually runs on.
The sweep sees only its grid points: a response that turns between two of them passes.
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

ROLES_BY_USE = {"layer": ("validation",), "endpoint": ("calibration", "test")}


def summarize(
    flags: list[compass2d.compass.Monotonicity], layer: str, direction: str, role: str
) -> dict:
    ascending = torch.cat([f.ascending for f in flags]).cpu().numpy()
    any_direction = torch.cat([f.any_direction for f in flags]).cpu().numpy()
    flat = torch.cat([f.flat for f in flags]).cpu().numpy()
    gain = torch.cat([f.gain for f in flags]).cpu().numpy()
    return {
        "layer": layer,
        "direction": direction,
        "role": role,
        "n_cells": len(ascending),
        "fraction_ascending": float(ascending.mean()),
        "fraction_monotonic": float(any_direction.mean()),
        "fraction_flat": float(flat.mean()),
        "mean_gain_ha": float(gain.mean()),
    }


def sweep_role(
    model: torch.nn.Module,
    role: str,
    cells: pd.DataFrame,
    root: Path,
    metadata: dict,
    args: argparse.Namespace,
    device: str,
) -> list[dict]:
    pixel_ha = garimpo_uncertainty.data.PIXEL_HA

    def hard(logits: torch.Tensor) -> torch.Tensor:
        return compass2d.metrics.hard_area(logits, pixel_ha)

    def soft(logits: torch.Tensor) -> torch.Tensor:
        return compass2d.metrics.soft_area(logits, pixel_ha, args.temperature)

    fitted = {
        layer: compass2d.directions.Directions.load(args.run_dir / "directions" / f"{layer}.npz")
        for layer in args.layers
    }
    uniform = []
    per_direction: dict[tuple[str, str], list] = {}
    loader = garimpo_uncertainty.cell_dataset.loader(
        root, cells, metadata["normalization"], args.batch_size, args.num_workers
    )
    for images, _, _ in loader:
        x = images.to(device)
        with torch.no_grad():
            logits = model(x)
        uniform.append(
            compass2d.compass.verify_monotonicity(
                lambda z: z, hard, logits, torch.ones_like(logits), args.beta_max_logits, args.steps
            )
        )
        for layer in args.layers:
            z = compass2d.decomposition.encode(model, layer, x)

            def decode(zi: torch.Tensor, layer: str = layer, x: torch.Tensor = x) -> torch.Tensor:
                return compass2d.decomposition.decode(model, layer, x, zi)

            jacobian = compass2d.directions.jacobian_sum(decode, z, soft).cpu().numpy()
            for direction in ("pca", "svm"):
                delta = compass2d.directions.cell_deltas(
                    direction, np.asarray(jacobian, dtype=np.float64), fitted[layer]
                )
                channel_delta = compass2d.compass.channel_delta(
                    torch.from_numpy(delta.copy()).to(z)
                )
                per_direction.setdefault((layer, direction), []).append(
                    compass2d.compass.verify_monotonicity(
                        decode, soft, z, channel_delta, args.beta_max_features, args.steps
                    )
                )
    rows = [summarize(uniform, compass2d.decomposition.LOGITS, "uniform", role)]
    rows += [
        summarize(flags, layer, direction, role)
        for (layer, direction), flags in per_direction.items()
    ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--use", choices=list(ROLES_BY_USE), required=True)
    parser.add_argument("--layers", nargs="+", required=True)
    parser.add_argument("--beta-max-logits", type=float, required=True)
    parser.add_argument("--beta-max-features", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--cells-per-role", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=compass2d.metrics.UPSTREAM_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = garimpo_uncertainty.data.data_root()
    device = args.device or garimpo_uncertainty.checkpoints.default_device()
    output = args.run_dir / f"monotonicity_{args.use}.csv"
    inputs = [args.checkpoint, root / "manifest.csv", args.run_dir / "directions/diagnostics.csv"]
    inputs += [args.run_dir / "directions" / f"{layer}.npz" for layer in args.layers]

    def work() -> dict:
        manifest = garimpo_uncertainty.data.read_manifest(root)
        model, metadata = garimpo_uncertainty.checkpoints.load_model(
            args.checkpoint, manifest, device
        )
        rows = []
        for role in ROLES_BY_USE[args.use]:
            cells = manifest[manifest.role == role]
            subset = cells.sample(min(args.cells_per_role, len(cells)), random_state=args.seed)
            rows += sweep_role(model, role, subset, root, metadata, args, device)
        with garimpo_uncertainty.stages.atomic(output) as temporary:
            pd.DataFrame(rows).to_csv(temporary, index=False)
        return {"rows": len(rows), "cells_per_role": args.cells_per_role}

    garimpo_uncertainty.stages.run_stage(
        args.run_dir / "status",
        f"{args.use}_monotonicity",
        [output],
        args.force,
        garimpo_uncertainty.stages.fingerprint(args, inputs),
        work,
    )


if __name__ == "__main__":
    main()
