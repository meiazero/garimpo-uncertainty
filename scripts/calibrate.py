"""Stage: per-test-cell quantile of every method x calibration situation; no model, no GPU.

Reads persisted areas and scores. Writes calibration/quantiles.npz: quantile and output-space terms
as [K, N], K indexing COMBINATIONS and N the test cells (NaN terms on COMPASS rows, whose endpoints
come from the decoder), and the score counts and saturation of each row as [K].
"""

import argparse
from pathlib import Path

import compass2d.calibration
import numpy as np

import garimpo_uncertainty.data
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


def calibrate(args: argparse.Namespace, root: Path, quantiles_path: Path) -> dict:
    run_dir = args.run_dir
    cells = garimpo_uncertainty.data.read_predictions(root, run_dir)
    if args.cells == "nonempty":
        cells = cells[cells.v > 0]
    train = cells[cells.role == "train"]
    calibration_cells = cells[cells.role == "calibration"].reset_index(drop=True)
    test_cells = cells[cells.role == "test"].reset_index(drop=True)

    with np.load(run_dir / "scores" / "scores.npz") as stored:
        calibration_at = garimpo_uncertainty.data.positions_of(
            stored["cell_index"], calibration_cells.cell_index.to_numpy()
        )
        test_at = garimpo_uncertainty.data.positions_of(
            stored["cell_index"], test_cells.cell_index.to_numpy()
        )
        score = stored["score"]
        saturated = stored["saturated"]
        beta_range = stored["beta_range"]
        density_ratio = compass2d.calibration.density_ratio(
            stored["feature_sums"][calibration_at],
            stored["feature_sums"][test_at],
            args.seed,
            args.lightgbm_jobs,
        )

    cqr_models = garimpo_uncertainty.intervals.fit_cqr(train, args.alpha, args.seed)
    calibration_groups = calibration_cells.group.to_numpy()
    test_groups = test_cells.group.to_numpy()

    shape = (len(garimpo_uncertainty.intervals.COMBINATIONS), len(test_cells))
    quantile = np.empty(shape)
    lower_term, upper_term, scale = np.full(shape, np.nan), np.full(shape, np.nan), np.empty(shape)
    n_scores = np.zeros(shape[0], dtype=int)
    n_saturated, saturated_fraction = np.full(shape[0], np.nan), np.full(shape[0], np.nan)
    for k, (method, calibration) in enumerate(garimpo_uncertainty.intervals.COMBINATIONS):
        if method == "independent_pixels":
            lower_term[k], upper_term[k] = garimpo_uncertainty.intervals.independent_pixel_terms(
                test_cells, args.alpha
            )
            quantile[k], scale[k] = 0.0, 1.0
        elif method in garimpo_uncertainty.intervals.COMPASS_METHODS:
            m = garimpo_uncertainty.intervals.COMPASS_METHODS.index(method)
            quantile[k] = quantile_per_cell(
                calibration,
                score[m, calibration_at],
                calibration_groups,
                score[m, test_at],
                test_groups,
                density_ratio,
                args.alpha,
            )
            scale[k] = np.nan
            flags = saturated[m, test_at if calibration == "own_region" else calibration_at]
            n_scores[k] = len(flags)
            n_saturated[k] = flags.sum()
            saturated_fraction[k] = np.mean(quantile[k] >= beta_range[m])
        else:
            lower_term[k], upper_term[k], scale[k] = garimpo_uncertainty.intervals.method_terms(
                method, test_cells, args.score_floor_ha, cqr_models
            )
            own_lower, own_upper, own_scale = garimpo_uncertainty.intervals.method_terms(
                method, calibration_cells, args.score_floor_ha, cqr_models
            )
            quantile[k] = quantile_per_cell(
                calibration,
                garimpo_uncertainty.intervals.scores(
                    calibration_cells.v.to_numpy(), own_lower, own_upper, own_scale
                ),
                calibration_groups,
                garimpo_uncertainty.intervals.scores(
                    test_cells.v.to_numpy(), lower_term[k], upper_term[k], scale[k]
                ),
                test_groups,
                density_ratio,
                args.alpha,
            )
            n_scores[k] = len(test_cells if calibration == "own_region" else calibration_cells)

    garimpo_uncertainty.stages.write_arrays(
        quantiles_path,
        cell_index=test_cells.cell_index.to_numpy(),
        quantile=quantile,
        lower_term=lower_term,
        upper_term=upper_term,
        scale=scale,
        alpha=np.array(args.alpha),
        n_calibration_scores=n_scores,
        n_saturated_calibration=n_saturated,
        quantile_saturated_fraction=saturated_fraction,
    )
    return {
        "calibration_cells": len(calibration_cells),
        "test_cells": len(test_cells),
        "rows": shape[0],
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
    parser.add_argument("--alpha", type=float, default=0.1, help="0.1 is 90 percent coverage")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lightgbm-jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    garimpo_uncertainty.stages.log_to_stdout()

    root = garimpo_uncertainty.data.data_root()
    run_dir = args.run_dir
    quantiles_path = run_dir / "calibration" / "quantiles.npz"
    inputs = [run_dir / "predictions" / "cells.npz", run_dir / "scores" / "scores.npz"]
    garimpo_uncertainty.stages.run_stage(
        run_dir / "status",
        "calibrate",
        [quantiles_path],
        garimpo_uncertainty.stages.fingerprint(
            args, inputs, hashed=garimpo_uncertainty.data.run_identity(root, run_dir)
        ),
        lambda: calibrate(args, root, quantiles_path),
        force=args.force,
    )


if __name__ == "__main__":
    main()
