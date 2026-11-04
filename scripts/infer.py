"""Stage: runs a checkpoint over every cell and persists what the metrics need.

predictions/cells.npz has one column per area term, aligned with the run's cell table. Validation,
calibration and test also keep per-pixel probabilities as uint8 (p * 255) in predictions/<role>.npy.
Training cells keep only their columns: only the CQR regressors read them, and they use the areas.
"""

import argparse
import os
import time
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


@torch.no_grad()
def infer_role(
    model: torch.nn.Module,
    role_cells: pd.DataFrame,
    root: Path,
    metadata: dict,
    args: argparse.Namespace,
    device: str,
    columns: dict[str, np.ndarray],
    positions: np.ndarray,
    probabilities_path: Path | None,
) -> float:
    """Fill `columns` at `positions`, and return the seconds spent waiting for data."""
    pixel_ha = garimpo_uncertainty.data.PIXEL_HA
    stored = None
    temporary = None
    loader = garimpo_uncertainty.cell_dataset.loader(
        root, role_cells, metadata["normalization"], args.batch_size, args.num_workers
    )
    first = 0
    data_wait = 0.0
    ready = time.monotonic()
    try:
        for images, masks, _ in loader:
            data_wait += time.monotonic() - ready
            logits = model(images.to(device))
            probabilities = torch.sigmoid(logits).to(torch.float64)
            height, width = logits.shape[-2:]
            batch = positions[first : first + len(images)]
            columns["v"][batch] = masks.to(torch.float64).flatten(1).sum(1).numpy() * pixel_ha
            columns["vhat"][batch] = compass2d.metrics.hard_area(logits, pixel_ha).cpu().numpy()
            columns["p_sum"][batch] = probabilities.flatten(1).sum(1).cpu().numpy() * pixel_ha
            columns["pixel_var"][batch] = (
                (probabilities * (1 - probabilities)).flatten(1).sum(1).cpu().numpy()
            )
            columns["cell_ha"][batch] = height * width * pixel_ha
            if probabilities_path is not None:
                if stored is None:
                    temporary = probabilities_path.with_name(
                        f"{probabilities_path.name}.{os.getpid()}.tmp"
                    )
                    stored = np.lib.format.open_memmap(
                        temporary, mode="w+", dtype=np.uint8, shape=(len(role_cells), height, width)
                    )
                scaled = probabilities[:, 0] * garimpo_uncertainty.evaluation.PROBABILITY_SCALE
                stored[first : first + len(images)] = scaled.round().to(torch.uint8).cpu().numpy()
            first += len(images)
            ready = time.monotonic()
        if stored is not None:
            stored.flush()
            del stored
            temporary.replace(probabilities_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
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
        data_wait = {
            role: round(
                infer_role(
                    model,
                    cells[cells.role == role],
                    root,
                    metadata,
                    args,
                    device,
                    columns,
                    np.flatnonzero(cells.role == role),
                    arrays.get(role),
                ),
                3,
            )
            for role in garimpo_uncertainty.data.ROLES
        }
        garimpo_uncertainty.stages.write_arrays(
            cells_path, cell_index=cells.cell_index.to_numpy(), **columns
        )
        shapes = {role: list(np.load(path, mmap_mode="r").shape) for role, path in arrays.items()}
        return shapes | {
            "cells_per_role": cells.role.value_counts().to_dict(),
            "data_wait_seconds": data_wait,
        }

    garimpo_uncertainty.stages.run_stage(
        args.run_dir / "status",
        "infer",
        [cells_path, *arrays.values()],
        args.force,
        garimpo_uncertainty.stages.fingerprint(
            args,
            [args.checkpoint],
            hashed=garimpo_uncertainty.data.run_identity(root, args.run_dir),
        ),
        work,
    )


if __name__ == "__main__":
    main()
