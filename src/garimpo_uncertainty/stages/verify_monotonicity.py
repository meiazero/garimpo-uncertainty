"""Stage: Algorithm 2 sweep per layer and direction; writes monotonicity_<use>.csv and the curves.

--use layer informs the choice of layer, so it reads the validation role only, never test;
--use endpoint checks the chosen layer on the cells the endpoint search actually runs on.
The sweep sees only its grid points: a response that turns between two of them passes.
"""

import argparse
from pathlib import Path

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
import garimpo_uncertainty.models
import garimpo_uncertainty.pipeline.runner

ROLES_BY_USE = {"layer": ("validation",), "endpoint": ("calibration", "test")}
Swept = tuple[compass2d.compass.Monotonicity, torch.Tensor]


def sweep(
    decode: compass2d.compass.Decode,
    metric: compass2d.compass.Metric,
    z: torch.Tensor,
    delta: torch.Tensor,
    *,
    beta_max: float,
    steps: int,
) -> Swept:
    """Run Algorithm 2 and keep its curve, area per beta and cell, which it computes and drops.

    verify_monotonicity calls the metric once per grid point, from -beta_max to beta_max.
    """
    curve = []

    def recorded(logits: torch.Tensor) -> torch.Tensor:
        curve.append(metric(logits))
        return curve[-1]

    flags = compass2d.compass.verify_monotonicity(decode, recorded, z, delta, beta_max, steps)
    return flags, torch.stack(curve)


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
) -> dict[str, Swept]:
    """Sweep one batch through one layer, one result per direction."""
    z, decode = model.split_at(layer, features)
    jacobian = garimpo_uncertainty.compass_j.jacobian(decode, z, args.direction_temperature)
    soft = garimpo_uncertainty.compass_j.soft_area(args.temperature)
    return {
        direction: sweep(decode, soft, z, delta, beta_max=args.beta_max_features, steps=args.steps)
        for direction, delta in garimpo_uncertainty.compass_j.channel_deltas(
            jacobian, fitted, z
        ).items()
    }


def show_worst(rows: list[dict]) -> None:
    """Report the least monotonic sweep, the one the layer rule is about."""
    worst = min(rows, key=lambda row: row["fraction_ascending"])
    garimpo_uncertainty.pipeline.runner.result(
        f"worst ascending {worst['fraction_ascending']:.0%} ({worst['role']}, {worst['layer']}, "
        f"{worst['direction']})"
    )


def sweep_role(
    model: torch.nn.Module,
    role: str,
    cells: pd.DataFrame,
    root: Path,
    metadata: dict,
    *,
    args: argparse.Namespace,
    device: str,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Return the role's summary rows and its curves, keyed role:layer:direction, cells in order."""
    run = garimpo_uncertainty.artifacts.RunPaths.of(args.run_dir)
    fitted = {
        layer: compass2d.directions.Directions.load(run.directions(layer)) for layer in args.layers
    }
    logits_layer = garimpo_uncertainty.models.LOGITS
    swept: dict[tuple[str, str], list[Swept]] = {}
    loader = garimpo_uncertainty.cell_dataset.loader(
        root, cells, metadata["normalization"], args.batch_size, args.num_workers
    )
    for images, _, _ in garimpo_uncertainty.cell_dataset.batches(loader):
        features = model.encode(images.to(device))
        logits, _ = model.split_at(logits_layer, features)
        swept.setdefault((logits_layer, "uniform"), []).append(
            sweep(
                lambda z: z,
                hard_area,
                logits,
                torch.ones_like(logits),
                beta_max=args.beta_max_logits,
                steps=args.steps,
            )
        )
        for layer in args.layers:
            for direction, result in sweep_layer(
                model, layer, features, fitted[layer], args
            ).items():
                swept.setdefault((layer, direction), []).append(result)
    rows = [
        summarize([flags for flags, _ in results], layer, direction, role)
        for (layer, direction), results in swept.items()
    ]
    curves = {
        f"{role}:{layer}:{direction}": torch.cat([c for _, c in results], 1).cpu().numpy()
        for (layer, direction), results in swept.items()
    }
    return rows, curves | {f"{role}:cell_index": cells.cell_index.to_numpy()}


def sweep_roles(
    model: torch.nn.Module,
    cells: pd.DataFrame,
    root: Path,
    metadata: dict,
    *,
    args: argparse.Namespace,
    device: str,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Sweep a seeded sample of each role the use reads."""
    rows, curves = [], {}
    for role in ROLES_BY_USE[args.use]:
        role_cells = cells[cells.role == role]
        subset = role_cells.sample(
            min(args.cells_per_role, len(role_cells)), random_state=args.seed
        )
        role_rows, role_curves = sweep_role(
            model, role, subset, root, metadata, args=args, device=device
        )
        rows += role_rows
        curves |= role_curves
    return rows, curves


def write_curves(path: Path, curves: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    """Store the curves with the grid each was swept on: logits and features take other ranges."""
    garimpo_uncertainty.files.write_arrays(
        path,
        **curves,
        beta_logits=np.linspace(-args.beta_max_logits, args.beta_max_logits, args.steps),
        beta_features=np.linspace(-args.beta_max_features, args.beta_max_features, args.steps),
    )


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
    output, curves_output = run.monotonicity(args.use), run.monotonicity_curves(args.use)
    inputs = [args.checkpoint, run.direction_diagnostics]
    inputs += [run.directions(layer) for layer in args.layers]

    def work() -> dict:
        cells = garimpo_uncertainty.data.read_run_cells(root, args.run_dir)
        model, metadata = garimpo_uncertainty.checkpoints.load_model(args.checkpoint, cells, device)
        rows, curves = sweep_roles(model, cells, root, metadata, args=args, device=device)
        with garimpo_uncertainty.files.atomic(output) as temporary:
            pd.DataFrame(rows).to_csv(temporary, index=False)
        write_curves(curves_output, curves, args)
        show_worst(rows)
        return {"rows": len(rows), "cells_per_role": args.cells_per_role}

    garimpo_uncertainty.pipeline.runner.run_stage(
        run.status,
        f"{args.use}_monotonicity",
        # ponytail: the curves stay out of the outputs that decide a skip, so a run finished before
        # they existed still skips instead of rewriting the table and refusing choose_layer. Such a
        # run has no curves; list them here once no such run is kept.
        [output],
        garimpo_uncertainty.pipeline.runner.fingerprint(
            args, inputs, hashed=garimpo_uncertainty.data.run_identity(root, args.run_dir)
        ),
        work,
        force=args.force,
    )
