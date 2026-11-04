"""Stage: metrics and bootstrap from persisted predictions, intervals and dataset masks.

Imports neither torch nor the model. A new metric or another boundary F1 tolerance only reruns this.
Writes metrics/cell_metrics.npz and metrics/{results,width_comparisons,coverage_comparisons,
q1_spearman}.csv.
"""

import argparse
from pathlib import Path

import numpy as np

import garimpo_uncertainty.data
import garimpo_uncertainty.evaluation
import garimpo_uncertainty.stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--boundary-tolerance-px",
        type=float,
        required=True,
        help="undefined in the proposal; must be chosen, never defaulted",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="proposal: 1,000 resamples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

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

    def work() -> dict:
        cells = garimpo_uncertainty.data.read_predictions(root, run_dir)
        metadata = garimpo_uncertainty.data.read_run_metadata(run_dir)
        garimpo_uncertainty.data.check_split(metadata, cells)
        test_cells = cells[cells.role == "test"]
        segmentation = garimpo_uncertainty.evaluation.segmentation_metrics(
            root,
            test_cells.cell_index.to_numpy(),
            run_dir / "predictions",
            args.boundary_tolerance_px,
        )
        with np.load(run_dir / "intervals" / "intervals.npz") as stored:
            cell_index, lower, upper = stored["cell_index"], stored["lower"], stored["upper"]
            layer, alpha = str(stored["layer"]), float(stored["alpha"])
        with np.load(run_dir / "calibration" / "quantiles.npz") as stored:
            saturation = {
                name: stored[name]
                for name in (
                    "n_calibration_scores",
                    "n_saturated_calibration",
                    "quantile_saturated_fraction",
                )
            }
        at = garimpo_uncertainty.data.positions_of(test_cells.cell_index.to_numpy(), cell_index)
        evaluated = test_cells.iloc[at]
        results, widths, coverages = garimpo_uncertainty.evaluation.summarize(
            lower,
            upper,
            evaluated.v.to_numpy(),
            evaluated.vhat.to_numpy(),
            evaluated.group.to_numpy(),
            {name: values[at] for name, values in segmentation.items()},
            saturation,
            {"model": metadata["variant"], "train_fraction": metadata["train_fraction"]},
            layer,
            alpha,
            args.n_bootstrap,
            args.seed,
        )
        spearman = garimpo_uncertainty.evaluation.spearman_by_model(
            evaluated.v.to_numpy(),
            evaluated.vhat.to_numpy(),
            evaluated.group.to_numpy(),
            {name: values[at] for name, values in segmentation.items()},
            {"model": metadata["variant"], "train_fraction": metadata["train_fraction"]},
            args.n_bootstrap,
            args.seed,
        )
        results.insert(5, "boundary_tolerance_px", args.boundary_tolerance_px)
        spearman.insert(2, "boundary_tolerance_px", args.boundary_tolerance_px)
        frames = {
            "results": results,
            "width_comparisons": widths,
            "coverage_comparisons": coverages,
            "q1_spearman": spearman,
        }
        garimpo_uncertainty.stages.write_arrays(
            cell_metrics_path, cell_index=test_cells.cell_index.to_numpy(), **segmentation
        )
        for name, frame in frames.items():
            with garimpo_uncertainty.stages.atomic(outputs[name]) as temporary:
                frame.to_csv(temporary, index=False)
        return {name: list(frame.shape) for name, frame in frames.items()}

    garimpo_uncertainty.stages.run_stage(
        run_dir / "status",
        "compute_metrics",
        [cell_metrics_path, *outputs.values()],
        args.force,
        garimpo_uncertainty.stages.fingerprint(
            args, inputs, hashed=garimpo_uncertainty.data.run_identity(root, run_dir)
        ),
        work,
    )


if __name__ == "__main__":
    main()
