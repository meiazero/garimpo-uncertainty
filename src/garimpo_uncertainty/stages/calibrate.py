"""Stage: per-test-cell quantile of every method x calibration situation; no model, no GPU.

Reads persisted areas and scores. Writes calibration/quantiles.npz: quantile and output-space terms
as [A, K, N], A indexing ALPHAS, K COMBINATIONS and N the test cells (NaN terms on COMPASS rows,
whose endpoints come from the decoder), and the score counts, saturation and share of infinite
quantiles of each row as [A, K].
"""

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Self

import compass2d.calibration
import compass2d.compass
import numpy as np
import pandas as pd
import torch

import garimpo_uncertainty.artifacts
import garimpo_uncertainty.data
import garimpo_uncertainty.files
import garimpo_uncertainty.intervals
import garimpo_uncertainty.pipeline.runner


def quantile_rule(
    calibration: str, alpha: float, seed: int
) -> Callable[[np.ndarray, np.ndarray], float]:
    """CDF pooling, the main rule, or repeated subsampling for the sensitivity rows.

    No score bounds nothing: a Mondrian stratum can leave a held-out group alone, which gets inf.
    """
    if calibration.endswith("_subsampled"):

        def rule(scores: np.ndarray, groups: np.ndarray) -> float:
            return compass2d.calibration.subsampled_quantile(
                scores, groups, alpha, garimpo_uncertainty.intervals.SUBSAMPLES, seed
            )

    else:

        def rule(scores: np.ndarray, groups: np.ndarray) -> float:
            return compass2d.calibration.group_quantile(scores, groups, alpha)

    return lambda scores, groups: rule(scores, groups) if len(scores) else np.inf


class Populations(NamedTuple):
    """The scores and groups each situation calibrates on, and the density ratio of each cell."""

    calibration_scores: np.ndarray
    calibration_groups: np.ndarray
    test_scores: np.ndarray
    test_groups: np.ndarray
    calibration_ratio: np.ndarray
    test_ratio: np.ndarray


Rule = Callable[[np.ndarray, np.ndarray], float]


def external_quantiles(rule: Rule, populations: Populations, alpha: float) -> np.ndarray:  # noqa: ARG001 - one signature for every situation
    scores, groups = populations.calibration_scores, populations.calibration_groups
    return np.full(len(populations.test_scores), rule(scores, groups))


def weighted_quantiles(rule: Rule, populations: Populations, alpha: float) -> np.ndarray:  # noqa: ARG001 - the weighting is its own rule
    return compass2d.calibration.group_weighted_quantiles(
        populations.calibration_scores,
        populations.calibration_groups,
        populations.calibration_ratio,
        populations.test_ratio,
        alpha,
    )


def own_region_quantiles(rule: Rule, populations: Populations, alpha: float) -> np.ndarray:  # noqa: ARG001
    return compass2d.calibration.leave_one_group_out_quantiles(
        populations.test_scores, populations.test_groups, rule
    )


# Each calibration situation by name; a _subsampled one swaps the rule, not the situation.
SITUATIONS: dict[str, Callable[[Rule, Populations, float], np.ndarray]] = {
    "external": external_quantiles,
    "external_weighted": weighted_quantiles,
    "own_region": own_region_quantiles,
}


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
    situation = calibration.removesuffix("_subsampled")
    if situation not in SITUATIONS:
        raise ValueError(f"unknown calibration: {calibration!r}")
    populations = Populations(
        calibration_scores,
        calibration_groups,
        test_scores,
        test_groups,
        calibration_ratio,
        test_ratio,
    )
    return SITUATIONS[situation](quantile_rule(calibration, alpha, seed), populations, alpha)


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
        if calibration.endswith("_mondrian"):
            return self.mondrian(
                calibration.removesuffix("_mondrian"), calibration_scores, test_scores
            )
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

    def mondrian(
        self, situation: str, calibration_scores: np.ndarray, test_scores: np.ndarray
    ) -> np.ndarray:
        """Each stratum of predicted mining calibrated alone (Vovk et al., 2005)."""
        rule = quantile_rule(situation, self.alpha, self.seed)
        calibration_strata = garimpo_uncertainty.intervals.mondrian_stratum(self.calibration)
        test_strata = garimpo_uncertainty.intervals.mondrian_stratum(self.test)
        test_groups = self.test.group.to_numpy()
        if situation == "external":
            return compass2d.calibration.mondrian_quantiles(
                calibration_scores,
                self.calibration.group.to_numpy(),
                calibration_strata,
                test_strata,
                rule,
            )
        quantiles = np.empty(len(test_scores))
        for stratum in np.unique(test_strata):
            inside = test_strata == stratum
            quantiles[inside] = compass2d.calibration.leave_one_group_out_quantiles(
                test_scores[inside], test_groups[inside], rule
            )
        return quantiles


# A scores.npz from before the one-sided scores fails here by name, not by position.
STORED_SCORES = (
    "score",
    "saturated",
    "lower",
    "lower_saturated",
    "upper",
    "upper_saturated",
    "beta_range",
)


@dataclass(frozen=True)
class CompassScores:
    """Persisted COMPASS scores, one row per method, and where each population sits in them."""

    score: np.ndarray
    saturated: np.ndarray
    lower: np.ndarray
    lower_saturated: np.ndarray
    upper: np.ndarray
    upper_saturated: np.ndarray
    beta_range: np.ndarray
    calibration_at: np.ndarray
    test_at: np.ndarray
    validation_at: np.ndarray
    validation_groups: np.ndarray


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
    side_ratio: np.ndarray
    censor_at: np.ndarray

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
            side_ratio=np.full(n_rows, np.nan),
            censor_at=np.full(n_rows, np.nan),
        )


def read_scores(
    cells: pd.DataFrame, args: argparse.Namespace
) -> tuple[CompassScores, np.ndarray, np.ndarray]:
    """Return the scores and the density ratio of test over calibration features at both."""
    by_role = {role: cells[cells.role == role] for role in garimpo_uncertainty.data.EVALUATED_ROLES}
    calibration_cells, test_cells = by_role["calibration"], by_role["test"]
    run = garimpo_uncertainty.artifacts.RunPaths.of(args.run_dir, args.layer_dir)
    with np.load(run.scores) as stored:
        at = {
            role: garimpo_uncertainty.data.positions_of(
                stored["cell_index"], frame.cell_index.to_numpy()
            )
            for role, frame in by_role.items()
        }
        calibration_at, test_at = at["calibration"], at["test"]
        scores = CompassScores(
            **{name: stored[name] for name in STORED_SCORES},
            calibration_at=calibration_at,
            test_at=test_at,
            validation_at=at["validation"],
            validation_groups=by_role["validation"].group.to_numpy(),
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


@dataclass(frozen=True)
class RowInputs:
    """What any row of quantiles.npz may read; each filler takes what its method needs."""

    problem: CalibrationProblem
    scores: CompassScores
    score_floor_ha: float
    cqr_models: tuple


def fill_independent_row(
    rows: Rows,
    k: int,
    method: str,  # noqa: ARG001 - one signature for every filler
    calibration: str,  # noqa: ARG001
    inputs: RowInputs,
) -> None:
    rows.lower_term[k], rows.upper_term[k] = garimpo_uncertainty.intervals.independent_pixel_terms(
        inputs.problem.test, inputs.problem.alpha
    )
    rows.quantile[k], rows.scale[k] = 0.0, 1.0


def fill_compass_row(rows: Rows, k: int, method: str, calibration: str, inputs: RowInputs) -> None:
    scores, problem = inputs.scores, inputs.problem
    m = garimpo_uncertainty.intervals.COMPASS_METHODS.index(method)
    rows.quantile[k] = problem.quantile(
        calibration, scores.score[m, scores.calibration_at], scores.score[m, scores.test_at]
    )
    rows.scale[k] = np.nan
    at = scores.test_at if calibration.startswith("own_region") else scores.calibration_at
    flags = scores.saturated[m, at]
    rows.n_scores[k] = len(flags)
    rows.n_saturated[k] = flags.sum()
    rows.censor_at[k] = scores.beta_range[m]
    rows.saturated_fraction[k] = np.mean(rows.quantile[k] >= rows.censor_at[k])


def side_ratio(scores: CompassScores, m: int, alpha: float) -> float:
    """Upper over lower one-sided quantile at 1 - alpha / 2, pooled over validation groups.

    Validation cells keep the ratio apart from calibration, as joint_score requires.
    """
    at, groups = scores.validation_at, scores.validation_groups
    upper = compass2d.calibration.group_quantile(scores.upper[m, at], groups, alpha / 2)
    lower = compass2d.calibration.group_quantile(scores.lower[m, at], groups, alpha / 2)
    return upper / lower


def fill_asymmetric_row(
    rows: Rows, k: int, method: str, calibration: str, inputs: RowInputs
) -> None:
    scores, problem = inputs.scores, inputs.problem
    m = garimpo_uncertainty.intervals.COMPASS_METHODS.index(method.removeprefix("a_"))
    ratio = side_ratio(scores, m, problem.alpha)
    joint, flags = (
        found.numpy()
        for found in compass2d.compass.joint_score(
            (torch.from_numpy(scores.lower[m]), torch.from_numpy(scores.lower_saturated[m])),
            (torch.from_numpy(scores.upper[m]), torch.from_numpy(scores.upper_saturated[m])),
            ratio,
        )
    )
    rows.quantile[k] = problem.quantile(
        calibration, joint[scores.calibration_at], joint[scores.test_at]
    )
    rows.scale[k] = np.nan
    at = scores.test_at if calibration.startswith("own_region") else scores.calibration_at
    rows.n_scores[k] = len(at)
    rows.n_saturated[k] = flags[at].sum()
    rows.side_ratio[k] = ratio
    # The upper side is censored at beta_range / ratio (joint_score).
    rows.censor_at[k] = min(scores.beta_range[m], scores.beta_range[m] / ratio)
    rows.saturated_fraction[k] = np.mean(rows.quantile[k] >= rows.censor_at[k])


def fill_output_space_row(
    rows: Rows, k: int, method: str, calibration: str, inputs: RowInputs
) -> None:
    problem, score_floor_ha, cqr_models = inputs.problem, inputs.score_floor_ha, inputs.cqr_models
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


# The filler of each method's rows; every other method is an output-space one.
FILLERS = (
    {"independent_pixels": fill_independent_row}
    | dict.fromkeys(garimpo_uncertainty.intervals.COMPASS_METHODS, fill_compass_row)
    | dict.fromkeys(garimpo_uncertainty.intervals.ASYMMETRIC_METHODS, fill_asymmetric_row)
)


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
    inputs = RowInputs(problem, scores, args.score_floor_ha, cqr_models)
    for k, (method, calibration) in enumerate(combinations):
        FILLERS.get(method, fill_output_space_row)(rows, k, method, calibration, inputs)
    rows.infinite_fraction[:] = np.isinf(rows.quantile).mean(axis=1)
    return rows


def calibrate(args: argparse.Namespace, root: Path, quantiles_path: Path) -> dict:
    cells = garimpo_uncertainty.data.read_predictions(root, args.run_dir)
    if args.cells == "nonempty":
        cells = cells[cells.v > 0]
    calibration_cells = cells[cells.role == "calibration"].reset_index(drop=True)
    test_cells = cells[cells.role == "test"].reset_index(drop=True)
    scores, calibration_ratio, test_ratio = read_scores(cells, args)
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

    garimpo_uncertainty.files.write_arrays(
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
        side_ratio=stacked("side_ratio"),
        censor_at=stacked("censor_at"),
        density_ratio_calibration=calibration_ratio,
        density_ratio_test=test_ratio,
    )
    masses = compass2d.calibration.group_masses(
        calibration_cells.group.to_numpy(), calibration_ratio
    )
    ess = compass2d.calibration.effective_sample_size(masses)
    infinite = stacked("infinite_fraction") > 0
    garimpo_uncertainty.pipeline.runner.result(
        f"{len(calibration_cells)} cells in {len(masses)} groups · weighted ESS {ess:.1f} groups · "
        f"infinite quantiles {infinite.sum()}/{infinite.size}"
    )
    return {
        "calibration_cells": len(calibration_cells),
        "test_cells": len(test_cells),
        "rows": len(garimpo_uncertainty.intervals.COMBINATIONS),
        "alphas": list(garimpo_uncertainty.intervals.ALPHAS),
        "density_ratio_max": float(max(calibration_ratio.max(), test_ratio.max())),
        # Below a handful, the weighted rows rest on almost no calibration group.
        "weighted_group_ess": ess,
        "calibration_groups": len(masses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    garimpo_uncertainty.pipeline.runner.add_layer_dir(parser)
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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lightgbm-jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    # An infinite floor turns every relative margin into NaN.
    if not (math.isfinite(args.score_floor_ha) and args.score_floor_ha > 0):
        parser.error(f"--score-floor-ha must be positive and finite, got {args.score_floor_ha}")
    garimpo_uncertainty.pipeline.runner.log_to_stdout()

    root = garimpo_uncertainty.data.data_root()
    run = garimpo_uncertainty.artifacts.RunPaths.of(args.run_dir, args.layer_dir)
    garimpo_uncertainty.pipeline.runner.run_stage(
        run.layer_status,
        "calibrate",
        [run.quantiles],
        garimpo_uncertainty.pipeline.runner.fingerprint(
            args,
            [run.predictions, run.scores],
            hashed=garimpo_uncertainty.data.run_identity(root, run.root),
        ),
        lambda: calibrate(args, root, run.quantiles),
        force=args.force,
    )
