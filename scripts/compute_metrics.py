"""Stage: metrics and bootstrap from persisted predictions, intervals and dataset masks.

Imports neither torch nor the model. A new metric or another boundary F1 tolerance only reruns this.
Writes metrics/{cell_metrics,results,width_comparisons,coverage_comparisons,q1_spearman}.csv.
"""

import argparse
from pathlib import Path

import pandas as pd

import garimpo_uncertainty.data
import garimpo_uncertainty.evaluation
import garimpo_uncertainty.stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    names = ("cell_metrics", "results", "width_comparisons", "coverage_comparisons", "q1_spearman")
    outputs = {name: metrics_dir / f"{name}.csv" for name in names}
    inputs = [
        root / "manifest.csv",
        run_dir / "predictions" / "cells.csv",
        run_dir / "predictions" / "test.npy",
        run_dir / "intervals" / "intervals.csv",
        run_dir / "calibration" / "summary.csv",
    ]

    def work() -> dict:
        manifest = garimpo_uncertainty.data.read_manifest(root)
        garimpo_uncertainty.data.check_split(
            garimpo_uncertainty.data.read_run_metadata(run_dir), manifest
        )
        cell_metrics = garimpo_uncertainty.evaluation.segmentation_metrics(
            root, manifest, run_dir / "predictions", args.boundary_tolerance_px
        )
        intervals = pd.read_csv(run_dir / "intervals" / "intervals.csv")
        results, widths, coverages = garimpo_uncertainty.evaluation.summarize(
            intervals,
            cell_metrics,
            pd.read_csv(run_dir / "calibration" / "summary.csv"),
            args.n_bootstrap,
            args.seed,
        )
        spearman = garimpo_uncertainty.evaluation.spearman_by_model(
            intervals, cell_metrics, args.n_bootstrap, args.seed
        )
        results.insert(4, "boundary_tolerance_px", args.boundary_tolerance_px)
        spearman.insert(1, "boundary_tolerance_px", args.boundary_tolerance_px)
        tables = {
            "cell_metrics": cell_metrics,
            "results": results,
            "width_comparisons": widths,
            "coverage_comparisons": coverages,
            "q1_spearman": spearman,
        }
        for name, frame in tables.items():
            with garimpo_uncertainty.stages.atomic(outputs[name]) as temporary:
                frame.to_csv(temporary, index=False)
        return {name: list(frame.shape) for name, frame in tables.items()}

    garimpo_uncertainty.stages.run_stage(
        run_dir / "status",
        "compute_metrics",
        list(outputs.values()),
        args.force,
        garimpo_uncertainty.stages.fingerprint(args, inputs),
        work,
    )


if __name__ == "__main__":
    main()
