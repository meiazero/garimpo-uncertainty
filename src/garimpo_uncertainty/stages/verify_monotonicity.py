"""Stage: Algorithm 2 sweep per layer and direction; writes monotonicity_<use>.csv.

--use layer informs the choice of layer, so it reads the validation role only, never test;
--use endpoint checks the chosen layer on the cells the endpoint search actually runs on.
The sweep sees only its grid points: a response that turns between two of them passes.
"""

import argparse
from pathlib import Path

import compass2d.compass
import compass2d.directions
import compass2d.metrics
import pandas as pd
import torch

import garimpo_uncertainty.artifacts
import garimpo_uncertainty.cell_dataset
import garimpo_uncertainty.checkpoints
import garimpo_uncertainty.compass_j
import garimpo_uncertainty.data
import garimpo_uncertainty.files
import garimpo_uncertainty.models.common
import garimpo_uncertainty.pipeline.runner

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


def hard_area(logits: torch.Tensor) -> torch.Tensor:
    return compass2d.metrics.hard_area(logits, garimpo_uncertainty.data.PIXEL_HA)


def sweep_layer(
    model: torch.nn.Module,
    layer: str,
    features: list[torch.Tensor],
    fitted: compass2d.directions.Directions,
    args: argparse.Namespace,
) -> dict[str, compass2d.compass.Monotonicity]:
    """Sweep one batch through one layer, one result per direction."""
    z, decode = garimpo_uncertainty.models.common.split_at(model, layer, features)
    jacobian = garimpo_uncertainty.compass_j.jacobian(decode, z, args.direction_temperature)
    soft = garimpo_uncertainty.compass_j.soft_area(args.temperature)
    return {
        direction: compass2d.compass.verify_monotonicity(
            decode, soft, z, delta, args.beta_max_features, args.steps
        )
        for direction, delta in garimpo_uncertainty.compass_j.channel_deltas(
            jacobian, fitted, z
        ).items()
    }


def sweep_role(
    model: torch.nn.Module,
    role: str,
    cells: pd.DataFrame,
    root: Path,
    metadata: dict,
    *,
    args: argparse.Namespace,
    device: str,
) -> list[dict]:
    run = garimpo_uncertainty.artifacts.RunPaths.of(args.run_dir)
    fitted = {
        layer: compass2d.directions.Directions.load(run.directions(layer)) for layer in args.layers
    }
    uniform = []
    per_direction: dict[tuple[str, str], list] = {}
    loader = garimpo_uncertainty.cell_dataset.loader(
        root, cells, metadata["normalization"], args.batch_size, args.num_workers
    )
    for images, _, _ in garimpo_uncertainty.cell_dataset.batches(loader):
        features = garimpo_uncertainty.models.common.encode(model, images.to(device))
        logits, _ = garimpo_uncertainty.models.common.split_at(
            model, garimpo_uncertainty.models.common.LOGITS, features
        )
        uniform.append(
            compass2d.compass.verify_monotonicity(
                lambda z: z,
                hard_area,
                logits,
                torch.ones_like(logits),
                args.beta_max_logits,
                args.steps,
            )
        )
        for layer in args.layers:
            swept = sweep_layer(model, layer, features, fitted[layer], args)
            for direction, flags in swept.items():
                per_direction.setdefault((layer, direction), []).append(flags)
    rows = [summarize(uniform, garimpo_uncertainty.models.common.LOGITS, "uniform", role)]
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
    parser.add_argument(
        "--direction-temperature",
        type=float,
        required=True,
        help="sigmoid temperature of the area whose Jacobian sets the direction; at 1e4 the "
        "gradient vanishes beyond |logit| ~ 5e-3, so cells predicted empty get the mean direction",
    )
    parser.add_argument("--seed", type=int, required=True)
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
    run = garimpo_uncertainty.artifacts.RunPaths.of(args.run_dir)
    output = run.monotonicity(args.use)
    inputs = [args.checkpoint, run.direction_diagnostics]
    inputs += [run.directions(layer) for layer in args.layers]

    def work() -> dict:
        cells = garimpo_uncertainty.data.read_run_cells(root, args.run_dir)
        model, metadata = garimpo_uncertainty.checkpoints.load_model(args.checkpoint, cells, device)
        rows = []
        for role in ROLES_BY_USE[args.use]:
            role_cells = cells[cells.role == role]
            subset = role_cells.sample(
                min(args.cells_per_role, len(role_cells)), random_state=args.seed
            )
            rows += sweep_role(model, role, subset, root, metadata, args=args, device=device)
        with garimpo_uncertainty.files.atomic(output) as temporary:
            pd.DataFrame(rows).to_csv(temporary, index=False)
        return {"rows": len(rows), "cells_per_role": args.cells_per_role}

    garimpo_uncertainty.pipeline.runner.run_stage(
        run.status,
        f"{args.use}_monotonicity",
        [output],
        garimpo_uncertainty.pipeline.runner.fingerprint(
            args, inputs, hashed=garimpo_uncertainty.data.run_identity(root, args.run_dir)
        ),
        work,
        force=args.force,
    )
