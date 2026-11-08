"""Stage: per-test-cell quantile of every method x calibration situation; no model, no GPU.

Reads persisted areas and scores. Writes calibration/quantiles.npz: quantile and output-space terms
as [A, K, N], A indexing ALPHAS, K COMBINATIONS and N the test cells (NaN terms on COMPASS rows,
whose endpoints come from the decoder), and the score counts, saturation and share of infinite
quantiles of each row as [A, K].
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import compass2d.calibration
import numpy as np
import pandas as pd

import garimpo_uncertainty.data
import garimpo_uncertainty.intervals
import garimpo_uncertainty.stages


def quantile_rule(
    calibration: str, alpha: float, seed: int
) -> Callable[[np.ndarray, np.ndarray], float]:
    """CDF pooling, the main rule, or repeated subsampling for the sensitivity rows."""
    if calibration.endswith("_subsampled"):
        return lambda scores, groups: compass2d.calibration.subsampled_quantile(
            scores, groups, alpha, garimpo_uncertainty.intervals.SUBSAMPLES, seed
        )
    return lambda scores, groups: compass2d.calibration.group_quantile(scores, groups, alpha)


def quantile_per_cell(
    calibration: str,
    calibration_scores: np.ndarray,
    calibration_groups: np.ndarray,
    test_scores: np.ndarray,
    test_groups: np.ndarray,
    *,
    calibration_ratio: np.ndarray,
    test_ratio: np.ndarray,
    alpha: float,
    seed: int,
) -> np.ndarray:
    rule = quantile_rule(calibration, alpha, seed)
    situation = calibration.removesuffix("_subsampled")
    if situation == "external":
        return np.full(len(test_scores), rule(calibration_scores, calibration_groups))
    if calibration == "external_weighted":
        return compass2d.calibration.group_weighted_quantiles(
            calibration_scores, calibration_groups, calibration_ratio, test_ratio, alpha
        )
    if situation == "own_region":
        return compass2d.calibration.leave_one_group_out_quantiles(test_scores, test_groups, rule)
    raise ValueError(f"unknown calibration: {calibration!r}")


@dataclass(frozen=True)
class CalibrationProblem:
    """One run's conformal problem: both populations, the shift between them, the target level."""

    calibration: pd.DataFrame
    test: pd.DataFrame
    calibration_ratio: np.ndarray
    test_ratio: np.ndarray
    alpha: float
    seed: int

    def quantile(
        self, calibration: str, calibration_scores: np.ndarray, test_scores: np.ndarray
    ) -> np.ndarray:
        return quantile_per_cell(
            calibration,
            calibration_scores,
            self.calibration.group.to_numpy(),
            test_scores,
            self.test.group.to_numpy(),
            calibration_ratio=self.calibration_ratio,
            test_ratio=self.test_ratio,
            alpha=self.alpha,
            seed=self.seed,
        )


@dataclass(frozen=True)
class CompassScores:
    """Persisted COMPASS scores, one row per method, and where each population sits in them."""

    score: np.ndarray
    saturated: np.ndarray
    beta_range: np.ndarray
    calibration_at: np.ndarray
    test_at: np.ndarray


@dataclass(frozen=True)
class Rows:
    """The arrays of quantiles.npz filled here: [K, N] terms and [K] score counts."""

    quantile: np.ndarray
    lower_term: np.ndarray
    upper_term: np.ndarray
    scale: np.ndarray
    n_scores: np.ndarray
    n_saturated: np.ndarray
    saturated_fraction: np.ndarray
    infinite_fraction: np.ndarray

    @classmethod
    def allocate(cls, n_rows: int, n_cells: int) -> Self:
        shape = (n_rows, n_cells)
        return cls(
            quantile=np.empty(shape),
            lower_term=np.full(shape, np.nan),
            upper_term=np.full(shape, np.nan),
            scale=np.empty(shape),
            n_scores=np.zeros(n_rows, dtype=int),
            n_saturated=np.full(n_rows, np.nan),
            saturated_fraction=np.full(n_rows, np.nan),
            infinite_fraction=np.zeros(n_rows),
        )


def read_scores(
    calibration_cells: pd.DataFrame, test_cells: pd.DataFrame, args: argparse.Namespace
) -> tuple[CompassScores, np.ndarray, np.ndarray]:
    """Return the scores and the density ratio of test over calibration features at both."""
    with np.load(args.run_dir / "scores" / "scores.npz") as stored:
        calibration_at = garimpo_uncertainty.data.positions_of(
            stored["cell_index"], calibration_cells.cell_index.to_numpy()
        )
        test_at = garimpo_uncertainty.data.positions_of(
            stored["cell_index"], test_cells.cell_index.to_numpy()
        )
        scores = CompassScores(
            stored["score"], stored["saturated"], stored["beta_range"], calibration_at, test_at
        )
        calibration_ratio, test_ratio = compass2d.calibration.density_ratio(
            stored["feature_sums"][calibration_at],
            stored["feature_sums"][test_at],
            calibration_cells.group.to_numpy(),
            test_cells.group.to_numpy(),
            n_folds=garimpo_uncertainty.intervals.DENSITY_RATIO_FOLDS,
            seed=args.seed,
            n_jobs=args.lightgbm_jobs,
        )
    return scores, calibration_ratio, test_ratio


def fill_compass_row(
    rows: Rows,
    k: int,
    method: str,
    calibration: str,
    *,
    scores: CompassScores,
    problem: CalibrationProblem,
) -> None:
    m = garimpo_uncertainty.intervals.COMPASS_METHODS.index(method)
    rows.quantile[k] = problem.quantile(
        calibration, scores.score[m, scores.calibration_at], scores.score[m, scores.test_at]
    )
    rows.scale[k] = np.nan
    at = scores.test_at if calibration.startswith("own_region") else scores.calibration_at
    flags = scores.saturated[m, at]
    rows.n_scores[k] = len(flags)
    rows.n_saturated[k] = flags.sum()
    rows.saturated_fraction[k] = np.mean(rows.quantile[k] >= scores.beta_range[m])


def fill_output_space_row(
    rows: Rows,
    k: int,
    method: str,
    calibration: str,
    *,
    problem: CalibrationProblem,
    score_floor_ha: float,
    cqr_models: tuple,
) -> None:
    rows.lower_term[k], rows.upper_term[k], rows.scale[k] = (
        garimpo_uncertainty.intervals.method_terms(method, problem.test, score_floor_ha, cqr_models)
    )
    own_lower, own_upper, own_scale = garimpo_uncertainty.intervals.method_terms(
        method, problem.calibration, score_floor_ha, cqr_models
    )
    rows.quantile[k] = problem.quantile(
        calibration,
        garimpo_uncertainty.intervals.scores(
            problem.calibration.v.to_numpy(), own_lower, own_upper, own_scale
        ),
        garimpo_uncertainty.intervals.scores(
            problem.test.v.to_numpy(), rows.lower_term[k], rows.upper_term[k], rows.scale[k]
        ),
    )
    own_region = calibration.startswith("own_region")
    rows.n_scores[k] = len(problem.test if own_region else problem.calibration)


def rows_at(
    problem: CalibrationProblem,
    scores: CompassScores,
    train_cells: pd.DataFrame,
    args: argparse.Namespace,
) -> Rows:
    """Every method x calibration row at problem.alpha; CQR fits its quantiles at that level."""
    cqr_models = garimpo_uncertainty.intervals.fit_cqr(train_cells, problem.alpha, args.seed)
    combinations = garimpo_uncertainty.intervals.COMBINATIONS
    rows = Rows.allocate(len(combinations), len(problem.test))
    for k, (method, calibration) in enumerate(combinations):
        if method == "independent_pixels":
            rows.lower_term[k], rows.upper_term[k] = (
                garimpo_uncertainty.intervals.independent_pixel_terms(problem.test, problem.alpha)
            )
            rows.quantile[k], rows.scale[k] = 0.0, 1.0
        elif method in garimpo_uncertainty.intervals.COMPASS_METHODS:
            fill_compass_row(rows, k, method, calibration, scores=scores, problem=problem)
        else:
            fill_output_space_row(
                rows,
                k,
                method,
                calibration,
                problem=problem,
                score_floor_ha=args.score_floor_ha,
                cqr_models=cqr_models,
            )
    rows.infinite_fraction[:] = np.isinf(rows.quantile).mean(axis=1)
    return rows


def calibrate(args: argparse.Namespace, root: Path, quantiles_path: Path) -> dict:
    cells = garimpo_uncertainty.data.read_predictions(root, args.run_dir)
    if args.cells == "nonempty":
        cells = cells[cells.v > 0]
    calibration_cells = cells[cells.role == "calibration"].reset_index(drop=True)
    test_cells = cells[cells.role == "test"].reset_index(drop=True)
    scores, calibration_ratio, test_ratio = read_scores(calibration_cells, test_cells, args)
    by_alpha = [
        rows_at(
            CalibrationProblem(
                calibration_cells, test_cells, calibration_ratio, test_ratio, alpha, args.seed
            ),
            scores,
            cells[cells.role == "train"],
            args,
        )
        for alpha in garimpo_uncertainty.intervals.ALPHAS
    ]

    def stacked(field: str) -> np.ndarray:
        return np.stack([getattr(rows, field) for rows in by_alpha])

    garimpo_uncertainty.stages.write_arrays(
        quantiles_path,
        cell_index=test_cells.cell_index.to_numpy(),
        quantile=stacked("quantile"),
        lower_term=stacked("lower_term"),
        upper_term=stacked("upper_term"),
        scale=stacked("scale"),
        alpha=np.array(garimpo_uncertainty.intervals.ALPHAS),
        n_calibration_scores=stacked("n_scores"),
        n_saturated_calibration=stacked("n_saturated"),
        quantile_saturated_fraction=stacked("saturated_fraction"),
        infinite_quantile_fraction=stacked("infinite_fraction"),
        density_ratio_calibration=calibration_ratio,
        density_ratio_test=test_ratio,
    )
    masses = compass2d.calibration.group_masses(
        calibration_cells.group.to_numpy(), calibration_ratio
    )
    return {
        "calibration_cells": len(calibration_cells),
        "test_cells": len(test_cells),
        "rows": len(garimpo_uncertainty.intervals.COMBINATIONS),
        "alphas": list(garimpo_uncertainty.intervals.ALPHAS),
        "density_ratio_max": float(max(calibration_ratio.max(), test_ratio.max())),
        # Below a handful, the weighted rows rest on almost no calibration group.
        "weighted_group_ess": compass2d.calibration.effective_sample_size(masses),
        "calibration_groups": len(masses),
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
