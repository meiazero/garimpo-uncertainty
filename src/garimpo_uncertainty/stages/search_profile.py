"""Stage search_profile: measure the COMPASS-J beta search on the GPU, so its figures draw on a CPU.

On a sample of calibration cells and at every layer with directions, it keeps the interval after
each bisection step, and times the search and the interval it gives at each threshold of
THRESHOLDS. The search settings are the run's own, read from its compass_scores status. Nothing
the pipeline reports changes: the result goes to search_profile.npz, read only by `garimpo figures`.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import compass2d.compass
import compass2d.directions
import numpy as np
import torch

import garimpo_uncertainty.artifacts
import garimpo_uncertainty.cell_dataset
import garimpo_uncertainty.checkpoints
import garimpo_uncertainty.compass_j
import garimpo_uncertainty.data
import garimpo_uncertainty.files
import garimpo_uncertainty.pipeline.runner

logger = logging.getLogger(__name__)

PROFILE = "search_profile.npz"
DIRECTIONS = {"pca": "compass_j_pca", "svm": "compass_j_svm"}
THRESHOLDS = (1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e-1)


def search_settings(run: garimpo_uncertainty.artifacts.RunPaths) -> dict:
    """Return the arguments compass_scores ran with, so the profile searches as the run did."""
    status = json.loads((run.status / "compass_scores.json").read_text(encoding="utf-8"))
    return status["completed_fingerprint"]["arguments"]


def bisection_path(
    space: garimpo_uncertainty.compass_j.Space, target: torch.Tensor, steps: int, beta_range: float
) -> tuple[np.ndarray, np.ndarray]:
    """Run the symmetric search of compass2d, keeping the interval's areas after every step."""
    low = torch.zeros_like(target, dtype=torch.float64)
    high = torch.full_like(target, beta_range, dtype=torch.float64)
    # high only ever takes middle, so its interval is middle's or the one it already had.
    high_lower, high_upper = space.endpoints(high)
    lowers, uppers = [], []
    for _ in range(steps):
        middle = (low + high) / 2
        lower, upper = space.endpoints(middle)
        covered = (lower <= target) & (target <= upper)
        high = torch.where(covered, middle, high)
        low = torch.where(covered, low, middle)
        high_lower = torch.where(covered, lower, high_lower)
        high_upper = torch.where(covered, upper, high_upper)
        lowers.append(high_lower.cpu().numpy())
        uppers.append(high_upper.cpu().numpy())
    return np.stack(lowers), np.stack(uppers)


def timed_search(
    space: garimpo_uncertainty.compass_j.Space,
    target: torch.Tensor,
    beta_range: float,
    threshold: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Seconds of one symmetric search, and the interval its scores give each cell."""
    synchronize = torch.cuda.synchronize if target.is_cuda else lambda: None
    synchronize()
    start = time.perf_counter()
    score, _ = compass2d.compass.binary_search_scores(
        space.decode, space.metric, space.z, space.delta, target, beta_range, threshold
    )
    synchronize()
    seconds = time.perf_counter() - start
    lower, upper = space.endpoints(score)
    return seconds, lower.cpu().numpy(), upper.cpu().numpy()


def profile_space(
    space: garimpo_uncertainty.compass_j.Space, target: torch.Tensor, settings: dict
) -> dict[str, np.ndarray]:
    """Return one direction's bisection path, and its time and interval at every threshold."""
    beta_range = settings["beta_range_features"]
    steps = int(np.ceil(np.log2(beta_range / settings["search_threshold_features"])))
    path_lower, path_upper = bisection_path(space, target, steps, beta_range)
    timed = [timed_search(space, target, beta_range, t) for t in THRESHOLDS]
    return {
        "path_lower": path_lower,
        "path_upper": path_upper,
        "seconds": np.array([seconds for seconds, _, _ in timed]),
        "lower": np.stack([lower for _, lower, _ in timed]),
        "upper": np.stack([upper for _, _, upper in timed]),
    }


def profile_batch(
    model: torch.nn.Module, images: torch.Tensor, masks: torch.Tensor, layers: dict, settings: dict
) -> dict[str, np.ndarray]:
    """Return every array of one batch, keyed layer:direction:name."""
    target = masks.to(torch.float64).flatten(1).sum(1) * garimpo_uncertainty.data.PIXEL_HA
    features = model.encode(images)
    found = {}
    for layer, fitted in layers.items():
        split = garimpo_uncertainty.compass_j.split(model, features, layer)
        jacobian = garimpo_uncertainty.compass_j.jacobian(
            split.decode, split.z, settings["direction_temperature"]
        )
        spaces = garimpo_uncertainty.compass_j.spaces(
            split, jacobian, fitted, settings["temperature"]
        )
        for direction, method in DIRECTIONS.items():
            arrays = profile_space(spaces[method], target, settings)
            found |= {f"{layer}:{direction}:{name}": array for name, array in arrays.items()}
    return found


def profile_cells(
    run: garimpo_uncertainty.artifacts.RunPaths, sample: object, args: argparse.Namespace
) -> dict[str, np.ndarray]:
    """Profile every layer with directions on the sample, batch by batch, joined along cells."""
    root = garimpo_uncertainty.data.data_root()
    cells = garimpo_uncertainty.data.read_run_cells(root, run.root)
    model, metadata = garimpo_uncertainty.checkpoints.load_model(
        args.checkpoint, cells, args.device
    )
    settings = search_settings(run)
    layers = {
        path.stem: compass2d.directions.Directions.load(path)
        for path in sorted((run.root / "directions").glob("*.npz"))
    }
    loader = garimpo_uncertainty.cell_dataset.loader(
        root, sample, metadata["normalization"], args.batch_size, args.num_workers
    )
    found: dict[str, list] = {}
    for images, masks, _ in garimpo_uncertainty.cell_dataset.batches(loader):
        batch = profile_batch(
            model, images.to(args.device), masks.to(args.device), layers, settings
        )
        for key, array in batch.items():
            found.setdefault(key, []).append(array)
    # Seconds add up across batches; every other array joins along its cells.
    return {
        key: np.sum(values, 0) if key.endswith(":seconds") else np.concatenate(values, -1)
        for key, values in found.items()
    }


def profile(args: argparse.Namespace, run: garimpo_uncertainty.artifacts.RunPaths) -> dict:
    cells = garimpo_uncertainty.data.read_run_cells(
        garimpo_uncertainty.data.data_root(), args.run_dir
    )
    calibration = cells[cells.role == "calibration"]
    sample = calibration.sample(min(args.cells, len(calibration)), random_state=args.seed)
    sample = sample.sort_values("cell_index")
    garimpo_uncertainty.files.write_arrays(
        run.root / PROFILE,
        **profile_cells(run, sample, args),
        cell_index=sample.cell_index.to_numpy(),
        thresholds=np.array(THRESHOLDS),
        device=np.array(str(args.device)),
    )
    return {"cells": len(sample), "device": str(args.device)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=64, help="calibration cells measured")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.device = args.device or garimpo_uncertainty.checkpoints.default_device()
    garimpo_uncertainty.pipeline.runner.log_to_stdout()

    run = garimpo_uncertainty.artifacts.RunPaths.of(args.run_dir)
    # The chosen layer's scores stand for the search settings, which compass_scores recorded.
    inputs = [args.checkpoint, run.scores, *sorted((run.root / "directions").glob("*.npz"))]
    garimpo_uncertainty.pipeline.runner.run_stage(
        run.status,
        "search_profile",
        [run.root / PROFILE],
        garimpo_uncertainty.pipeline.runner.fingerprint(
            args,
            inputs,
            hashed=garimpo_uncertainty.data.run_identity(
                garimpo_uncertainty.data.data_root(), run.root
            ),
        ),
        lambda: profile(args, run),
        force=args.force,
    )
