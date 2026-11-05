"""Stage: runs a checkpoint over every cell and persists what the metrics need.

predictions/cells.npz has one column per area term, aligned with the run's cell table. Validation,
calibration and test also keep per-pixel probabilities as uint8 (p * 255) in predictions/<role>.npy.
Training cells keep only their columns: only the CQR regressors read them, and they use the areas.
"""

import argparse
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import compass2d.metrics
import numpy as np
import pandas as pd
import torch

import garimpo_uncertainty.cell_dataset
import garimpo_uncertainty.checkpoints
import garimpo_uncertainty.data
import garimpo_uncertainty.evaluation
import garimpo_uncertainty.stages

PERSISTED_ROLES = ("validation", "calibration", "test")
AREA_COLUMNS = ("v", "vhat", "p_sum", "pixel_var", "cell_ha")


@contextmanager
def probability_writer(
    path: Path | None, n_cells: int
) -> Iterator[Callable[[int, torch.Tensor], None]]:
    """Yield a function that stores a batch at a position, as uint8, in a temporary file.

    The file takes its pixel shape from the first batch and replaces `path` on a clean exit; with no
    `path` the function stores nothing.
    """
    if path is None:
        yield lambda _first, _probabilities: None
        return
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    stored = None

    def write(first: int, probabilities: torch.Tensor) -> None:
        nonlocal stored
        if stored is None:
            shape = (n_cells, *probabilities.shape[-2:])
            stored = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=shape)
        scaled = probabilities[:, 0] * garimpo_uncertainty.evaluation.PROBABILITY_SCALE
        stored[first : first + len(probabilities)] = scaled.round().to(torch.uint8).cpu().numpy()

    try:
        yield write
        if stored is not None:
            stored.flush()
            del stored
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fill_area_columns(
    columns: dict[str, np.ndarray],
    positions: np.ndarray,
    masks: torch.Tensor,
    logits: torch.Tensor,
    probabilities: torch.Tensor,
) -> None:
    pixel_ha = garimpo_uncertainty.data.PIXEL_HA
    height, width = logits.shape[-2:]
    columns["v"][positions] = masks.to(torch.float64).flatten(1).sum(1).numpy() * pixel_ha
    columns["vhat"][positions] = compass2d.metrics.hard_area(logits, pixel_ha).cpu().numpy()
    columns["p_sum"][positions] = probabilities.flatten(1).sum(1).cpu().numpy() * pixel_ha
    columns["pixel_var"][positions] = (
        (probabilities * (1 - probabilities)).flatten(1).sum(1).cpu().numpy()
    )
    columns["cell_ha"][positions] = height * width * pixel_ha


@torch.no_grad()
def infer_role(
    model: torch.nn.Module,
    role_cells: pd.DataFrame,
    root: Path,
    metadata: dict,
    args: argparse.Namespace,
    *,
    device: str,
    columns: dict[str, np.ndarray],
    positions: np.ndarray,
    probabilities_path: Path | None,
) -> float:
    """Fill `columns` at `positions`, and return the seconds spent waiting for data."""
    loader = garimpo_uncertainty.cell_dataset.loader(
        root, role_cells, metadata["normalization"], args.batch_size, args.num_workers
    )
    first = 0
    data_wait = 0.0
    with probability_writer(probabilities_path, len(role_cells)) as write_probabilities:
        ready = time.monotonic()
        for images, masks, _ in loader:
            data_wait += time.monotonic() - ready
            logits = model(images.to(device))
            probabilities = torch.sigmoid(logits).to(torch.float64)
            batch = positions[first : first + len(images)]
            fill_area_columns(columns, batch, masks, logits, probabilities)
            write_probabilities(first, probabilities)
            first += len(images)
            ready = time.monotonic()
    return data_wait


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    garimpo_uncertainty.stages.log_to_stdout()

    root = garimpo_uncertainty.data.data_root()
    device = args.device or garimpo_uncertainty.checkpoints.default_device()
    prediction_dir = args.run_dir / "predictions"
    cells_path = prediction_dir / "cells.npz"
    arrays = {role: prediction_dir / f"{role}.npy" for role in PERSISTED_ROLES}

    def work() -> dict:
        cells = garimpo_uncertainty.data.read_run_cells(root, args.run_dir)
        model, metadata = garimpo_uncertainty.checkpoints.load_model(args.checkpoint, cells, device)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        columns = {name: np.empty(len(cells)) for name in AREA_COLUMNS}
        data_wait, role_seconds, role_peaks = {}, {}, {}
        for role in garimpo_uncertainty.data.ROLES:
            garimpo_uncertainty.stages.take_gpu_peak_bytes()
            role_start = time.monotonic()
            wait = infer_role(
                model,
                cells[cells.role == role],
                root,
                metadata,
                args,
                device=device,
                columns=columns,
                positions=np.flatnonzero(cells.role == role),
                probabilities_path=arrays.get(role),
            )
            role_seconds[role] = time.monotonic() - role_start
            role_peaks[role] = garimpo_uncertainty.stages.take_gpu_peak_bytes()
            data_wait[role] = round(wait, 3)
        garimpo_uncertainty.stages.write_arrays(
            cells_path, cell_index=cells.cell_index.to_numpy(), **columns
        )
        shapes = {role: list(np.load(path, mmap_mode="r").shape) for role, path in arrays.items()}
        return shapes | {
            "cells_per_role": cells.role.value_counts().to_dict(),
            "data_wait_seconds": data_wait,
            "seconds_per_role": role_seconds,
            "peak_gpu_bytes_per_role": role_peaks,
        }

    garimpo_uncertainty.stages.run_stage(
        args.run_dir / "status",
        "infer",
        [cells_path, *arrays.values()],
        garimpo_uncertainty.stages.fingerprint(
            args,
            [args.checkpoint],
            hashed=garimpo_uncertainty.data.run_identity(root, args.run_dir),
        ),
        work,
        force=args.force,
    )


if __name__ == "__main__":
    main()
