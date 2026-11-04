"""Stage: fits the PCA and SVM directions per layer on the training role only.

Writes directions/<layer>.npz and directions/diagnostics.csv with explained variance and
|cos(v1, w)| per layer, the evidence for choosing the layer.
"""

import argparse
from pathlib import Path

import compass2d.directions
import compass2d.metrics
import numpy as np
import pandas as pd
import torch

import garimpo_uncertainty.cell_dataset
import garimpo_uncertainty.checkpoints
import garimpo_uncertainty.data
import garimpo_uncertainty.models
import garimpo_uncertainty.stages


def fit_layer(
    model: torch.nn.Module,
    layer: str,
    train_cells: pd.DataFrame,
    root: Path,
    metadata: dict,
    args: argparse.Namespace,
    device: str,
) -> tuple[compass2d.directions.Directions, dict]:
    generator = torch.Generator().manual_seed(args.seed)
    loader = garimpo_uncertainty.cell_dataset.loader(
        root, train_cells, metadata["normalization"], args.batch_size, args.num_workers
    )

    def metric(logits: torch.Tensor) -> torch.Tensor:
        pixel_ha = garimpo_uncertainty.data.PIXEL_HA
        return compass2d.metrics.soft_area(logits, pixel_ha, args.temperature)

    jacobians, features, labels = [], [], []
    feature_shape = None
    for images, masks, _ in loader:
        x, mask = images.to(device), masks.to(device)
        encoded = garimpo_uncertainty.models.encode(model, x)
        z, decode = garimpo_uncertainty.models.split_at(model, layer, encoded)
        feature_shape = list(z.shape[1:])
        jacobian = compass2d.directions.jacobian_sum(decode, z, metric)
        jacobians.append(jacobian.cpu().numpy())
        pixels, targets = compass2d.directions.sample_pixels(
            z, mask, args.svm_pixels_per_cell, generator
        )
        features.append(pixels.cpu().numpy())
        labels.append(targets.cpu().numpy())
    jacobian_matrix = np.concatenate(jacobians).astype(np.float64)
    pixel_matrix = np.concatenate(features).astype(np.float64)
    pixel_labels = np.concatenate(labels).astype(int)

    pca = compass2d.directions.fit_pca(jacobian_matrix, args.n_components)
    mean_jacobian = jacobian_matrix.mean(axis=0)
    class_weight = None if args.svm_class_weight == "none" else args.svm_class_weight
    w = compass2d.directions.fit_svm(
        pixel_matrix,
        pixel_labels,
        mean_jacobian,
        args.svm_c,
        class_weight,
        args.svm_max_iter,
        args.seed,
    )
    fitted = compass2d.directions.Directions(
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_explained_variance=pca.explained_variance_ratio_,
        svm_w=w,
    )
    diagnostics = {
        "layer": layer,
        "feature_shape": str(feature_shape),
        "n_train_cells": len(jacobian_matrix),
        "n_zero_jacobians": int((np.abs(jacobian_matrix).sum(axis=1) == 0).sum()),
        "pca_explained_variance": float(pca.explained_variance_ratio_[0]),
        "abs_cos_v1_w": compass2d.directions.abs_cosine(pca.components_[0], w),
        "svm_mean_jacobian_projection": float(w @ mean_jacobian),
        "svm_pixels": len(pixel_labels),
        "svm_positive_fraction": float(pixel_labels.mean()),
    }
    return fitted, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", required=True)
    parser.add_argument("--svm-c", type=float, required=True)
    parser.add_argument("--svm-class-weight", choices=["none", "balanced"], required=True)
    parser.add_argument("--svm-pixels-per-cell", type=int, required=True)
    parser.add_argument("--svm-max-iter", type=int, default=1000, help="sklearn's default")
    parser.add_argument("--n-components", type=int, default=1, help="paper: L = 1 for area")
    parser.add_argument("--temperature", type=float, default=compass2d.metrics.UPSTREAM_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = garimpo_uncertainty.data.data_root()
    device = args.device or garimpo_uncertainty.checkpoints.default_device()
    directions_dir = args.run_dir / "directions"
    diagnostics_path = directions_dir / "diagnostics.csv"
    layer_paths = {layer: directions_dir / f"{layer}.npz" for layer in args.layers}

    def work() -> dict:
        cells = garimpo_uncertainty.data.read_run_cells(root, args.run_dir)
        model, metadata = garimpo_uncertainty.checkpoints.load_model(args.checkpoint, cells, device)
        available = garimpo_uncertainty.models.feature_layers(model)
        unknown = set(args.layers) - set(available)
        if unknown:
            raise ValueError(f"unknown layers {sorted(unknown)}; available: {available}")
        train_cells = cells[cells.role == "train"]
        rows = []
        for layer, path in layer_paths.items():
            fitted, row = fit_layer(model, layer, train_cells, root, metadata, args, device)
            with garimpo_uncertainty.stages.atomic(path) as temporary:
                fitted.save(temporary)
            rows.append(row)
        with garimpo_uncertainty.stages.atomic(diagnostics_path) as temporary:
            pd.DataFrame(rows).to_csv(temporary, index=False)
        shapes = {row["layer"]: row["feature_shape"] for row in rows}
        return shapes | {"available_layers": available}

    garimpo_uncertainty.stages.run_stage(
        args.run_dir / "status",
        "fit_directions",
        [diagnostics_path, *layer_paths.values()],
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
