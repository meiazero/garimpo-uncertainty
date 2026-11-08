"""Stage: metrics and bootstrap from persisted predictions, intervals and dataset masks.

Imports neither torch nor the model. A new metric or another boundary F1 tolerance only reruns this.
Writes metrics/cell_metrics.npz and metrics/{results,width_comparisons,coverage_comparisons,
q1_spearman,coverage_by_stratum,score_relation}.csv; every interval table holds each level of
ALPHAS.
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
    alphas: np.ndarray


def read_intervals(run_dir: Path) -> Intervals:
    with np.load(run_dir / "intervals" / "intervals.npz") as stored:
        return Intervals(
            stored["cell_index"],
            stored["lower"],
            stored["upper"],
            str(stored["layer"]),
            stored["alpha"],
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


def tables_at_level(
    args: argparse.Namespace,
    intervals: Intervals,
    a: int,
    cells: garimpo_uncertainty.evaluation.EvaluatedCells,
    identity: dict,
) -> dict[str, pd.DataFrame]:
    """Return the interval tables at one level of ALPHAS, each row carrying that alpha."""
    saturation = read_saturation(args.run_dir)
    alpha = float(intervals.alphas[a])
    at_level = identity | {"alpha": alpha}
    results, widths, coverages = garimpo_uncertainty.evaluation.summarize(
        intervals.lower[a],
        intervals.upper[a],
        cells,
        saturation={name: values[a] for name, values in saturation.items()},
        identity=at_level,
        layer=intervals.layer,
        alpha=alpha,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    by_stratum = garimpo_uncertainty.evaluation.coverage_by_stratum(
        intervals.lower[a],
        intervals.upper[a],
        cells,
        identity=at_level,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    return {
        "results": results,
        "width_comparisons": widths,
        "coverage_comparisons": coverages,
        "coverage_by_stratum": by_stratum,
    }


def metric_tables(
    args: argparse.Namespace,
    metadata: dict,
    test_cells: pd.DataFrame,
    segmentation: dict[str, np.ndarray],
) -> dict[str, pd.DataFrame]:
    intervals = read_intervals(args.run_dir)
    cells = aligned_cells(test_cells, segmentation, intervals.cell_index)
    identity = {"model": metadata["variant"], "train_fraction": metadata["train_fraction"]}
    levels = [
        tables_at_level(args, intervals, a, cells, identity) for a in range(len(intervals.alphas))
    ]
    frames = {
        name: pd.concat([level[name] for level in levels], ignore_index=True) for name in levels[0]
    }
    spearman = garimpo_uncertainty.evaluation.spearman_by_model(
        cells, identity=identity, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    after_calibration = frames["results"].columns.get_loc("calibration") + 1
    frames["results"].insert(after_calibration, "boundary_tolerance_px", args.boundary_tolerance_px)
    spearman.insert(2, "boundary_tolerance_px", args.boundary_tolerance_px)
    return frames | {"q1_spearman": spearman}


def score_relation(run_dir: Path, cells: pd.DataFrame, identity: dict) -> pd.DataFrame:
    """COMPASS scores against split-conformal scores on the calibration cells."""
    calibration = cells[cells.role == "calibration"]
    with np.load(run_dir / "scores" / "scores.npz") as stored:
        at = garimpo_uncertainty.data.positions_of(
            stored["cell_index"], calibration.cell_index.to_numpy()
        )
        compass_scores, saturated = stored["score"][:, at], stored["saturated"][:, at]
    split_scores = np.abs(calibration.v - calibration.vhat).to_numpy()
    table = garimpo_uncertainty.evaluation.score_relation(split_scores, compass_scores, saturated)
    return table.assign(**identity)


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
    identity = {"model": metadata["variant"], "train_fraction": metadata["train_fraction"]}
    frames["score_relation"] = score_relation(args.run_dir, cells, identity)
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
    tables = (
        "results",
        "width_comparisons",
        "coverage_comparisons",
        "q1_spearman",
        "coverage_by_stratum",
        "score_relation",
    )
    outputs = {name: metrics_dir / f"{name}.csv" for name in tables}
    cell_metrics_path = metrics_dir / "cell_metrics.npz"
    inputs = [
        run_dir / "predictions" / "cells.npz",
        run_dir / "predictions" / "test.npy",
        run_dir / "intervals" / "intervals.npz",
        run_dir / "calibration" / "quantiles.npz",
        run_dir / "scores" / "scores.npz",
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
