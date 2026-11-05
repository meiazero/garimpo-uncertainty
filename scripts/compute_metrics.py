"""Stage: metrics and bootstrap from persisted predictions, intervals and dataset masks.

Imports neither torch nor the model. A new metric or another boundary F1 tolerance only reruns this.
Writes metrics/cell_metrics.npz and metrics/{results,width_comparisons,coverage_comparisons,
q1_spearman}.csv.
"""

import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

import garimpo_uncertainty.data
import garimpo_uncertainty.evaluation
import garimpo_uncertainty.stages


class Intervals(NamedTuple):
    cell_index: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    layer: str
    alpha: float


def read_intervals(run_dir: Path) -> Intervals:
    with np.load(run_dir / "intervals" / "intervals.npz") as stored:
        return Intervals(
            stored["cell_index"],
            stored["lower"],
            stored["upper"],
            str(stored["layer"]),
            float(stored["alpha"]),
        )


def read_saturation(run_dir: Path) -> dict[str, np.ndarray]:
    with np.load(run_dir / "calibration" / "quantiles.npz") as stored:
        return {
            name: stored[name]
            for name in (
                "n_calibration_scores",
                "n_saturated_calibration",
                "quantile_saturated_fraction",
            )
        }


def aligned_cells(
    test_cells: pd.DataFrame, segmentation: dict[str, np.ndarray], cell_index: np.ndarray
) -> garimpo_uncertainty.evaluation.EvaluatedCells:
    """Reorder the test cells and their segmentation metrics as the intervals' cell index."""
    at = garimpo_uncertainty.data.positions_of(test_cells.cell_index.to_numpy(), cell_index)
    evaluated = test_cells.iloc[at]
    return garimpo_uncertainty.evaluation.EvaluatedCells(
        evaluated.v.to_numpy(),
        evaluated.vhat.to_numpy(),
        evaluated.group.to_numpy(),
        {name: values[at] for name, values in segmentation.items()},
    )


def metric_tables(
    args: argparse.Namespace,
    metadata: dict,
    test_cells: pd.DataFrame,
    segmentation: dict[str, np.ndarray],
) -> dict[str, pd.DataFrame]:
    intervals = read_intervals(args.run_dir)
    saturation = read_saturation(args.run_dir)
    cells = aligned_cells(test_cells, segmentation, intervals.cell_index)
    identity = {"model": metadata["variant"], "train_fraction": metadata["train_fraction"]}
    results, widths, coverages = garimpo_uncertainty.evaluation.summarize(
        intervals.lower,
        intervals.upper,
        cells,
        saturation=saturation,
        identity=identity,
        layer=intervals.layer,
        alpha=intervals.alpha,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    spearman = garimpo_uncertainty.evaluation.spearman_by_model(
        cells, identity=identity, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    results.insert(5, "boundary_tolerance_px", args.boundary_tolerance_px)
    spearman.insert(2, "boundary_tolerance_px", args.boundary_tolerance_px)
    return {
        "results": results,
        "width_comparisons": widths,
        "coverage_comparisons": coverages,
        "q1_spearman": spearman,
    }


def compute(
    args: argparse.Namespace, root: Path, cell_metrics_path: Path, outputs: dict[str, Path]
) -> dict:
    cells = garimpo_uncertainty.data.read_predictions(root, args.run_dir)
    metadata = garimpo_uncertainty.data.read_run_metadata(args.run_dir)
    garimpo_uncertainty.data.check_split(metadata, cells)
    test_cells = cells[cells.role == "test"]
    segmentation = garimpo_uncertainty.evaluation.segmentation_metrics(
        root,
        test_cells.cell_index.to_numpy(),
        args.run_dir / "predictions",
        args.boundary_tolerance_px,
    )
    frames = metric_tables(args, metadata, test_cells, segmentation)
    garimpo_uncertainty.stages.write_arrays(
        cell_metrics_path, cell_index=test_cells.cell_index.to_numpy(), **segmentation
    )
    for name, frame in frames.items():
        with garimpo_uncertainty.stages.atomic(outputs[name]) as temporary:
            frame.to_csv(temporary, index=False)
    return {name: list(frame.shape) for name, frame in frames.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--boundary-tolerance-px",
        type=float,
        required=True,
        help="must be chosen, never defaulted",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000, help="resamples of the group bootstrap"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    garimpo_uncertainty.stages.log_to_stdout()

    root = garimpo_uncertainty.data.data_root()
    run_dir = args.run_dir
    metrics_dir = run_dir / "metrics"
    tables = ("results", "width_comparisons", "coverage_comparisons", "q1_spearman")
    outputs = {name: metrics_dir / f"{name}.csv" for name in tables}
    cell_metrics_path = metrics_dir / "cell_metrics.npz"
    inputs = [
        run_dir / "predictions" / "cells.npz",
        run_dir / "predictions" / "test.npy",
        run_dir / "intervals" / "intervals.npz",
        run_dir / "calibration" / "quantiles.npz",
    ]

    garimpo_uncertainty.stages.run_stage(
        run_dir / "status",
        "compute_metrics",
        [cell_metrics_path, *outputs.values()],
        garimpo_uncertainty.stages.fingerprint(
            args, inputs, hashed=garimpo_uncertainty.data.run_identity(root, run_dir)
        ),
        lambda: compute(args, root, cell_metrics_path, outputs),
        force=args.force,
    )


if __name__ == "__main__":
    main()
