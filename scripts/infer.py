"""Stage: runs a checkpoint over every cell and persists what the metrics need.

predictions/cells.csv has one row per cell of every role. Validation, calibration and test also keep
per-pixel probabilities as uint8 (p * 255) in predictions/<role>.npy: 262,144 bytes per 512 px cell,
about 0.74 GB per model on the full split (2,816 cells). The 16,048 training cells keep only their
row, 4.2 GB saved; only the CQR regressors read them, and they use the areas.
"""

import argparse
import os
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


@torch.no_grad()
def infer_role(
    model: torch.nn.Module,
    role_cells: pd.DataFrame,
    root: Path,
    metadata: dict,
    args: argparse.Namespace,
    device: str,
    probabilities_path: Path | None,
) -> pd.DataFrame:
    pixel_ha = garimpo_uncertainty.data.PIXEL_HA
    rows = []
    stored = None
    temporary = None
    loader = garimpo_uncertainty.cell_dataset.loader(
        root, role_cells, metadata["normalization"], args.batch_size, args.num_workers
    )
    next_index = 0
    try:
        for images, masks, cell_ids in loader:
            logits = model(images.to(device))
            probabilities = torch.sigmoid(logits).to(torch.float64)
            vhat = compass2d.metrics.hard_area(logits, pixel_ha).cpu().numpy()
            p_sum = probabilities.flatten(1).sum(1).cpu().numpy() * pixel_ha
            pixel_var = (probabilities * (1 - probabilities)).flatten(1).sum(1).cpu().numpy()
            v = masks.to(torch.float64).flatten(1).sum(1).numpy() * pixel_ha
            height, width = logits.shape[-2:]
            rows += [
                {
                    "cell_id": cell_id,
                    "v": v[i],
                    "vhat": vhat[i],
                    "p_sum": p_sum[i],
                    "pixel_var": pixel_var[i],
                    "cell_ha": height * width * pixel_ha,
                }
                for i, cell_id in enumerate(cell_ids)
            ]
            if probabilities_path is not None:
                if stored is None:
                    temporary = probabilities_path.with_name(
                        f"{probabilities_path.name}.{os.getpid()}.tmp"
                    )
                    stored = np.lib.format.open_memmap(
                        temporary, mode="w+", dtype=np.uint8, shape=(len(role_cells), height, width)
                    )
                scaled = probabilities[:, 0] * garimpo_uncertainty.evaluation.PROBABILITY_SCALE
                stored[next_index : next_index + len(cell_ids)] = (
                    scaled.round().to(torch.uint8).cpu().numpy()
                )
                next_index += len(cell_ids)
        if stored is not None:
            stored.flush()
            del stored
            temporary.replace(probabilities_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    identity = role_cells[["cell_id", "tile_id", "role", "group"]]
    return identity.merge(pd.DataFrame(rows), on="cell_id", validate="one_to_one")


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
    cells_path = prediction_dir / "cells.csv"
    arrays = {role: prediction_dir / f"{role}.npy" for role in PERSISTED_ROLES}

    def work() -> dict:
        manifest = garimpo_uncertainty.data.read_manifest(root)
        model, metadata = garimpo_uncertainty.checkpoints.load_model(
            args.checkpoint, manifest, device
        )
        prediction_dir.mkdir(parents=True, exist_ok=True)
        frames = [
            infer_role(
                model,
                manifest[manifest.role == role],
                root,
                metadata,
                args,
                device,
                arrays.get(role),
            )
            for role in garimpo_uncertainty.data.ROLES
        ]
        cells = pd.concat(frames, ignore_index=True)
        with garimpo_uncertainty.stages.atomic(cells_path) as temporary:
            cells.to_csv(temporary, index=False)
        shapes = {role: list(np.load(path, mmap_mode="r").shape) for role, path in arrays.items()}
        return shapes | {"cells_per_role": cells.role.value_counts().to_dict()}

    garimpo_uncertainty.stages.run_stage(
        args.run_dir / "status",
        "infer",
        [cells_path, *arrays.values()],
        args.force,
        garimpo_uncertainty.stages.fingerprint(args, [args.checkpoint, root / "manifest.csv"]),
        work,
    )


if __name__ == "__main__":
    main()
