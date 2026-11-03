"""Stage: per-test-cell quantile of every method x calibration situation; no model, no GPU.

Reads persisted areas and scores. Writes calibration/quantiles.csv (quantile and output-space terms
per test cell) and calibration/summary.csv (saturation per row of the results table).
"""

import argparse
from pathlib import Path

import compass2d.calibration
import numpy as np
import pandas as pd

import garimpo_uncertainty.intervals
import garimpo_uncertainty.stages


def quantile_per_cell(
    calibration: str,
    calibration_scores: np.ndarray,
    calibration_groups: np.ndarray,
    test_scores: np.ndarray,
    test_groups: np.ndarray,
    density_ratio: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if calibration == "external":
        value = compass2d.calibration.group_quantile(calibration_scores, calibration_groups, alpha)
        return np.full(len(test_scores), value)
    if calibration == "external_weighted":
        value = compass2d.calibration.group_quantile(
            calibration_scores, calibration_groups, alpha, density_ratio
        )
        return np.full(len(test_scores), value)
    if calibration == "own_region":
        return compass2d.calibration.leave_one_group_out_quantiles(test_scores, test_groups, alpha)
    raise ValueError(f"unknown calibration: {calibration!r}")


def calibrate(args: argparse.Namespace, quantiles_path: Path, summary_path: Path) -> dict:
    run_dir = args.run_dir
    cells = pd.read_csv(run_dir / "predictions" / "cells.csv")
    if args.cells == "nonempty":
        cells = cells[cells.v > 0]
    train = cells[cells.role == "train"]
    calibration_cells = cells[cells.role == "calibration"].reset_index(drop=True)
    test_cells = cells[cells.role == "test"].reset_index(drop=True)

    compass_scores = pd.read_csv(run_dir / "scores" / "compass_scores.csv")
    score_table = compass_scores.pivot(index="cell_id", columns="method", values="score")
    saturated_table = compass_scores.pivot(index="cell_id", columns="method", values="saturated")
    beta_ranges = compass_scores.groupby("method").beta_range.first()

    with np.load(run_dir / "scores" / "eval_features.npz") as stored:
        positions = {str(cell_id): i for i, cell_id in enumerate(stored["cell_id"])}
        feature_sums = stored["feature_sums"]
    density_ratio = compass2d.calibration.density_ratio(
        feature_sums[[positions[c] for c in calibration_cells.cell_id]],
        feature_sums[[positions[c] for c in test_cells.cell_id]],
        args.seed,
        args.lightgbm_jobs,
    )

    cqr_models = garimpo_uncertainty.intervals.fit_cqr(train, args.alpha, args.seed)
    calibration_groups = calibration_cells.group.to_numpy()
    test_groups = test_cells.group.to_numpy()
    identity = test_cells[["cell_id", "group"]]

    frames, summary = [], []
    for method, calibration in garimpo_uncertainty.intervals.COMBINATIONS:
        row = {"method": method, "calibration": calibration, "alpha": args.alpha}
        if method == "independent_pixels":
            lower, upper = garimpo_uncertainty.intervals.independent_pixel_terms(
                test_cells, args.alpha
            )
            quantile, scale = np.zeros(len(test_cells)), np.ones(len(test_cells))
            row |= {"n_calibration_scores": 0}
        elif method in garimpo_uncertainty.intervals.COMPASS_METHODS:
            calibration_scores = score_table.loc[calibration_cells.cell_id, method].to_numpy()
            test_scores = score_table.loc[test_cells.cell_id, method].to_numpy()
            quantile = quantile_per_cell(
                calibration,
                calibration_scores,
                calibration_groups,
                test_scores,
                test_groups,
                density_ratio,
                args.alpha,
            )
            lower = upper = scale = np.full(len(test_cells), np.nan)
            used = test_cells if calibration == "own_region" else calibration_cells
            flags = saturated_table.loc[used.cell_id, method].astype(bool)
            row |= {
                "n_calibration_scores": len(flags),
                "n_saturated_calibration": int(flags.sum()),
                "quantile_saturated_fraction": float(np.mean(quantile >= beta_ranges[method])),
            }
        else:
            lower, upper, scale = garimpo_uncertainty.intervals.method_terms(
                method, test_cells, args.score_floor_ha, cqr_models
            )
            own_lower, own_upper, own_scale = garimpo_uncertainty.intervals.method_terms(
                method, calibration_cells, args.score_floor_ha, cqr_models
            )
            quantile = quantile_per_cell(
                calibration,
                garimpo_uncertainty.intervals.scores(
                    calibration_cells.v.to_numpy(), own_lower, own_upper, own_scale
                ),
                calibration_groups,
                garimpo_uncertainty.intervals.scores(test_cells.v.to_numpy(), lower, upper, scale),
                test_groups,
                density_ratio,
                args.alpha,
            )
            used = test_cells if calibration == "own_region" else calibration_cells
            row |= {"n_calibration_scores": len(used)}
        frames.append(
            identity.assign(
                method=method,
                calibration=calibration,
                quantile_value=quantile,
                lower_term=lower,
                upper_term=upper,
                scale=scale,
            )
        )
        summary.append(row)

    with garimpo_uncertainty.stages.atomic(quantiles_path) as temporary:
        pd.concat(frames, ignore_index=True).to_csv(temporary, index=False)
    with garimpo_uncertainty.stages.atomic(summary_path) as temporary:
        pd.DataFrame(summary).to_csv(temporary, index=False)
    return {
        "calibration_cells": len(calibration_cells),
        "test_cells": len(test_cells),
        "rows": len(summary),
        "density_ratio_max": float(density_ratio.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--cells",
        choices=["all", "nonempty"],
        required=True,
        help="population of the intervals: every cell, or cells whose reference has mining",
    )
    parser.add_argument(
        "--score-floor-ha",
        type=float,
        required=True,
        help="denominator floor of the relative and local scores",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="proposal: 90 percent coverage")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lightgbm-jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    quantiles_path = run_dir / "calibration" / "quantiles.csv"
    summary_path = run_dir / "calibration" / "summary.csv"
    inputs = [
        run_dir / "predictions" / "cells.csv",
        run_dir / "scores" / "compass_scores.csv",
        run_dir / "scores" / "eval_features.npz",
    ]
    garimpo_uncertainty.stages.run_stage(
        run_dir / "status",
        "calibrate",
        [quantiles_path, summary_path],
        args.force,
        garimpo_uncertainty.stages.fingerprint(args, inputs),
        lambda: calibrate(args, quantiles_path, summary_path),
    )


if __name__ == "__main__":
    main()
