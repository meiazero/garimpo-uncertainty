"""The numbers of a finished run, recomputed apart from the pipeline."""

import argparse
import json
import runpy
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import compass2d.calibration
import compass2d.compass
import compass2d.directions
import compass2d.metrics
import numpy as np
import pandas as pd
import pytest
import torch

import garimpo_uncertainty.cell_dataset
import garimpo_uncertainty.checkpoints
import garimpo_uncertainty.data
import garimpo_uncertainty.evaluation
import garimpo_uncertainty.intervals
import garimpo_uncertainty.models.common
from tests.e2e import support

ALPHA = 0.1
ALPHAS = support.ALPHAS
SUBSAMPLES = garimpo_uncertainty.intervals.SUBSAMPLES
# calibrate.py's default, which the shared runs keep.
CALIBRATE_SEED = 0


@dataclass(frozen=True)
class Evaluation:
    arch: str
    run_dir: Path
    layer: str
    data_root: Path
    cells: pd.DataFrame
    all_results: pd.DataFrame
    test: pd.DataFrame
    interval_index: np.ndarray
    all_lower: np.ndarray
    all_upper: np.ndarray
    all_quantile: np.ndarray

    # The scenarios below read the paper's main level; the alpha tests read every level.
    @property
    def results(self) -> pd.DataFrame:
        return self.all_results[self.all_results.alpha == ALPHA].reset_index(drop=True)

    @property
    def lower(self) -> np.ndarray:
        return self.all_lower[ALPHAS.index(ALPHA)]

    @property
    def upper(self) -> np.ndarray:
        return self.all_upper[ALPHAS.index(ALPHA)]

    @property
    def quantile(self) -> np.ndarray:
        return self.all_quantile[ALPHAS.index(ALPHA)]

    @property
    def v(self) -> np.ndarray:
        return self.test.v.to_numpy()

    @property
    def vhat(self) -> np.ndarray:
        return self.test.vhat.to_numpy()

    @property
    def groups(self) -> np.ndarray:
        return self.test.group.to_numpy()

    def row(self, method: str, calibration: str) -> int:
        return garimpo_uncertainty.intervals.COMBINATIONS.index((method, calibration))


@pytest.fixture(scope="module")
def evaluation(finished_run: support.FinishedRun, data_root: Path) -> Evaluation:
    arch, run_dir, layer = finished_run
    cells = garimpo_uncertainty.data.read_predictions(data_root, run_dir)
    with np.load(run_dir / "intervals" / "intervals.npz") as stored:
        interval_index, lower, upper = stored["cell_index"], stored["lower"], stored["upper"]
        quantile = stored["quantile"]
        assert tuple(stored["alpha"]) == ALPHAS
    return Evaluation(
        arch,
        run_dir,
        layer,
        data_root,
        cells,
        pd.read_csv(run_dir / "metrics" / "results.csv"),
        cells[cells.role == "test"].reset_index(drop=True),
        interval_index,
        lower,
        upper,
        quantile,
    )


def pooled_quantile(scores: np.ndarray, groups: np.ndarray, level: float) -> float:
    """Brute-force Dunn et al. pooling: return the smallest score whose mean ECDF reaches level."""
    for candidate in np.sort(scores):
        ecdfs = [np.mean(scores[groups == group] <= candidate) for group in np.unique(groups)]
        if np.mean(ecdfs) >= level - 1e-12:
            return float(candidate)
    return float(scores.max())


def brute_force_metrics(
    lower: float,
    upper: float,
    v: float,
    alpha: float,
) -> tuple[float, float, float]:
    """Return covered, width and interval score of one cell, with no array arithmetic."""
    covered = 1.0 if lower <= v <= upper else 0.0
    width = upper - lower
    penalty = (2 / alpha) * (max(lower - v, 0.0) + max(v - upper, 0.0))
    return covered, width, width + penalty


def mean_of_group_means(values: list[float], groups: np.ndarray) -> float:
    """Average per-group means, skipping NaN inside a group and groups left with no value."""
    by_group: dict[int, list[float]] = {}
    for value, group in zip(values, groups.tolist(), strict=True):
        if not np.isnan(value):
            by_group.setdefault(group, []).append(value)
    return sum(sum(kept) / len(kept) for kept in by_group.values()) / len(by_group)


def brute_force_dice(root: Path, run_dir: Path, test_index: np.ndarray) -> list[float]:
    masks = np.load(root / "masks.npy", mmap_mode="r")
    probabilities = np.load(run_dir / "predictions" / "test.npy", mmap_mode="r")
    values = []
    for position, cell_index in enumerate(test_index.tolist()):
        reference = masks[cell_index] > 0
        prediction = probabilities[position] >= 128
        if reference.any():
            overlap = int((reference & prediction).sum())
            values.append(2 * overlap / (int(reference.sum()) + int(prediction.sum())))
        else:
            values.append(float("nan"))
    return values


def test_results_table_lists_every_combination(evaluation: Evaluation) -> None:
    results = evaluation.results
    keys = set(zip(results.method, results.calibration, strict=True))
    assert keys == support.EXPECTED_ROWS
    assert len(results) == len(support.EXPECTED_ROWS)
    assert (results.model == support.MODEL_NAMES[evaluation.arch]).all()
    assert (results.train_fraction == 1.0).all()
    assert list(zip(results.method, results.calibration, strict=True)) == list(
        garimpo_uncertainty.intervals.COMBINATIONS
    )


def test_intervals_stay_inside_the_cell(evaluation: Evaluation) -> None:
    assert np.array_equal(evaluation.interval_index, evaluation.test.cell_index.to_numpy())
    assert (evaluation.lower >= 0).all()
    assert (evaluation.lower <= evaluation.upper).all()
    assert (evaluation.upper <= evaluation.test.cell_ha.to_numpy() + 1e-6).all()


def test_every_alpha_has_every_row(evaluation: Evaluation) -> None:
    results = evaluation.all_results
    assert tuple(results.alpha.unique()) == ALPHAS
    for alpha in ALPHAS:
        at = results[results.alpha == alpha]
        assert list(zip(at.method, at.calibration, strict=True)) == list(
            garimpo_uncertainty.intervals.COMBINATIONS
        ), alpha
    assert evaluation.all_lower.shape[0] == len(ALPHAS)


def test_array_metrics_match_one_computation_per_cell(evaluation: Evaluation) -> None:
    """Array metrics in results.csv against one Python computation per cell, by group means.

    Every level, so an interval built at one alpha and scored at another shows.
    """
    v, groups = evaluation.v, evaluation.groups
    dice_by_cell = brute_force_dice(
        evaluation.data_root, evaluation.run_dir, evaluation.interval_index
    )
    for a, alpha in enumerate(ALPHAS):
        lower, upper = evaluation.all_lower[a], evaluation.all_upper[a]
        at = evaluation.all_results[evaluation.all_results.alpha == alpha]
        for k, row in enumerate(at.itertuples()):
            per_cell = [
                brute_force_metrics(float(lower[k, i]), float(upper[k, i]), float(v[i]), alpha)
                for i in range(len(v))
            ]
            covered, width, score = zip(*per_cell, strict=True)
            assert np.isclose(row.picp, mean_of_group_means(list(covered), groups)), (alpha, k)
            assert np.isclose(row.mpiw, mean_of_group_means(list(width), groups)), (alpha, k)
            assert np.isclose(row.interval_score, mean_of_group_means(list(score), groups))
            assert np.isclose(row.dice, mean_of_group_means(dice_by_cell, groups)), (alpha, k)
            assert row.picp_ci_low <= row.picp_ci_high
            assert row.n_groups == len(np.unique(groups))


def test_quantiles_grow_as_alpha_shrinks(evaluation: Evaluation) -> None:
    """A higher level can only raise a quantile of the same scores.

    CQR is left out: each level fits its own quantile regressors, so its scores differ.
    """
    order = np.argsort(ALPHAS)
    for k, (method, calibration) in enumerate(garimpo_uncertainty.intervals.COMBINATIONS):
        if method == "cqr":
            continue
        by_level = evaluation.all_quantile[order, k]
        # Compared, not subtracted: weighted quantiles may be inf at several levels.
        assert (by_level[1:] <= by_level[:-1] + 1e-9).all(), (method, calibration)
    simple = evaluation.row("conformal_simple", "external")
    assert (
        evaluation.all_quantile[order[0], simple][0] > evaluation.all_quantile[order[-1], simple][0]
    )


@pytest.mark.parametrize("alpha", support.ALPHAS)
def test_every_level_is_calibrated_at_its_own_alpha(evaluation: Evaluation, alpha: float) -> None:
    """Pooled quantile, CQR regressors and the control, each recomputed at this level."""
    level = ALPHAS.index(alpha)
    intervals = garimpo_uncertainty.intervals
    calibration = evaluation.cells[evaluation.cells.role == "calibration"]
    groups = calibration.group.to_numpy()
    simple = evaluation.row("conformal_simple", "external")
    scores = np.abs(calibration.v - calibration.vhat).to_numpy()
    assert np.allclose(
        evaluation.all_quantile[level, simple], pooled_quantile(scores, groups, 1 - alpha)
    )
    models = intervals.fit_cqr(evaluation.cells[evaluation.cells.role == "train"], alpha, 0)
    low, high, _ = intervals.method_terms("cqr", calibration, 0.01, models)
    cqr_scores = np.maximum(low - calibration.v.to_numpy(), calibration.v.to_numpy() - high)
    cqr = evaluation.row("cqr", "external")
    assert np.allclose(
        evaluation.all_quantile[level, cqr], pooled_quantile(cqr_scores, groups, 1 - alpha)
    )
    control = evaluation.row("independent_pixels", "none")
    lower, upper = intervals.clip_to_cell(
        *intervals.independent_pixel_terms(evaluation.test, alpha),
        evaluation.test.cell_ha.to_numpy(),
    )
    assert np.allclose(evaluation.all_lower[level, control], lower)
    assert np.allclose(evaluation.all_upper[level, control], upper)


def test_quantile_weighs_groups_equally(evaluation: Evaluation) -> None:
    """The big calibration group makes this bite."""
    calibration = evaluation.cells[evaluation.cells.role == "calibration"]
    scores = np.abs(calibration.v - calibration.vhat).to_numpy()
    expected = pooled_quantile(scores, calibration.group.to_numpy(), 1 - ALPHA)
    simple = evaluation.row("conformal_simple", "external")
    assert np.allclose(evaluation.quantile[simple], expected)
    # Output-space rows are plain arithmetic on the quantile, recomputed here apart from the stage.
    cell_ha = evaluation.test.cell_ha.to_numpy()
    assert np.allclose(evaluation.lower[simple], np.clip(evaluation.vhat - expected, 0, cell_ha))
    assert np.allclose(evaluation.upper[simple], np.clip(evaluation.vhat + expected, 0, cell_ha))


def brute_force_weighted_quantile(
    scores: np.ndarray, groups: np.ndarray, ratio: np.ndarray, test_ratio: float, level: float
) -> float:
    """Tibshirani et al. (2019), Corollary 1 with groups as units, one candidate at a time."""
    sizes = {group: int((groups == group).sum()) for group in np.unique(groups)}
    masses = np.array([r / sizes[g] for r, g in zip(ratio, groups, strict=True)])
    total = masses.sum() + test_ratio
    for candidate in np.sort(scores):
        if masses[scores <= candidate].sum() >= level * total - 1e-12 * total:
            return float(candidate)
    return float("inf")


def test_weighted_rows_put_the_test_cell_at_infinity(evaluation: Evaluation) -> None:
    """Each test cell gets its own quantile, with its own density ratio as mass at +inf."""
    with np.load(evaluation.run_dir / "calibration" / "quantiles.npz") as stored:
        ratio, test_ratio = stored["density_ratio_calibration"], stored["density_ratio_test"]
        infinite = stored["infinite_quantile_fraction"][ALPHAS.index(ALPHA)]
    calibration = evaluation.cells[evaluation.cells.role == "calibration"]
    scores = np.abs(calibration.v - calibration.vhat).to_numpy()
    groups = calibration.group.to_numpy()
    k = evaluation.row("conformal_simple", "external_weighted")
    expected = [
        brute_force_weighted_quantile(scores, groups, ratio, r, 1 - ALPHA) for r in test_ratio
    ]
    assert np.allclose(evaluation.quantile[k], expected)
    assert infinite[k] == np.mean(np.isinf(expected))
    assert (ratio > 0).all()
    assert (test_ratio > 0).all()


# support.run_scores_and_intervals passes it to calibrate.py.
SCORE_FLOOR_HA = 0.01


def test_relative_scores_floor_their_denominator(evaluation: Evaluation) -> None:
    """Cells predicted empty have no area to divide by; the floor must stand in for it.

    The local score's denominator never falls under the floor in these runs; test_stages.py
    covers it.
    """
    calibration = evaluation.cells[evaluation.cells.role == "calibration"]
    assert (calibration.vhat == 0).any(), "the floor must bite"
    scale = np.maximum(calibration.vhat.to_numpy(), SCORE_FLOOR_HA)
    scores = np.abs(calibration.v - calibration.vhat).to_numpy() / scale
    expected = pooled_quantile(scores, calibration.group.to_numpy(), 1 - ALPHA)
    k = evaluation.row("conformal_relative", "external")
    assert np.allclose(evaluation.quantile[k], expected)
    margin = expected * np.maximum(evaluation.vhat, SCORE_FLOOR_HA)
    cell_ha = evaluation.test.cell_ha.to_numpy()
    assert np.allclose(evaluation.lower[k], np.clip(evaluation.vhat - margin, 0, cell_ha))
    assert np.allclose(evaluation.upper[k], np.clip(evaluation.vhat + margin, 0, cell_ha))


def test_weighted_group_ess_is_reported(evaluation: Evaluation) -> None:
    """Kish's ESS of the calibration groups, each weighing the mean density ratio of its cells."""
    with np.load(evaluation.run_dir / "calibration" / "quantiles.npz") as stored:
        ratio = stored["density_ratio_calibration"]
    groups = evaluation.cells[evaluation.cells.role == "calibration"].group.to_numpy()
    masses = [ratio[groups == group].mean() for group in np.unique(groups)]
    expected = sum(masses) ** 2 / sum(mass**2 for mass in masses)
    status = json.loads((evaluation.run_dir / "status" / "calibrate.json").read_text())
    assert np.isclose(status["shapes"]["weighted_group_ess"], expected)
    assert status["shapes"]["calibration_groups"] == len(masses)
    assert expected < len(masses), "equal masses would not tell group weights from cell weights"


def test_mondrian_rows_calibrate_each_predicted_stratum_alone(evaluation: Evaluation) -> None:
    """Calibrate each stratum of predicted mining alone, external and own region.

    A stratum with no other score gives inf.
    """
    calibration = evaluation.cells[evaluation.cells.role == "calibration"]
    scores = np.abs(calibration.v - calibration.vhat).to_numpy()
    groups = calibration.group.to_numpy()
    strata = (calibration.vhat.to_numpy() > 0).astype(int)
    test_scores = np.abs(evaluation.v - evaluation.vhat)
    test_strata = (evaluation.vhat > 0).astype(int)
    external = evaluation.quantile[evaluation.row("conformal_simple", "external_mondrian")]
    own = evaluation.quantile[evaluation.row("conformal_simple", "own_region_mondrian")]
    for i in range(len(evaluation.v)):
        same = strata == test_strata[i]
        expected = pooled_quantile(scores[same], groups[same], 1 - ALPHA) if same.any() else np.inf
        assert np.isclose(external[i], expected), i
        others = (test_strata == test_strata[i]) & (evaluation.groups != evaluation.groups[i])
        expected = (
            pooled_quantile(test_scores[others], evaluation.groups[others], 1 - ALPHA)
            if others.any()
            else np.inf
        )
        assert np.isclose(own[i], expected), i
    assert set(test_strata.tolist()) == {0, 1}, "the fixture needs both strata"


def test_asymmetric_rows_calibrate_one_joint_score(evaluation: Evaluation) -> None:
    """Recompute the side ratio on validation and the joint quantile on calibration.

    Ratio: one-sided quantiles at 1 - alpha / 2; then max(lower, upper / ratio) at 1 - alpha.
    """
    cells = evaluation.cells
    validation, calibration = cells[cells.role == "validation"], cells[cells.role == "calibration"]
    with np.load(evaluation.run_dir / "scores" / "scores.npz") as stored:
        index = stored["cell_index"].tolist()
        lower, upper = stored["lower"][0], stored["upper"][0]
    at_validation = [index.index(i) for i in validation.cell_index]
    at_calibration = [index.index(i) for i in calibration.cell_index]
    with np.load(evaluation.run_dir / "calibration" / "quantiles.npz") as stored:
        ratio = stored["side_ratio"][ALPHAS.index(ALPHA)]
    k = evaluation.row("a_compass_l", "external")
    level = 1 - ALPHA / 2
    groups = validation.group.to_numpy()
    expected_ratio = pooled_quantile(upper[at_validation], groups, level) / pooled_quantile(
        lower[at_validation], groups, level
    )
    assert np.isclose(ratio[k], expected_ratio)
    joint = np.maximum(lower[at_calibration], upper[at_calibration] / expected_ratio)
    expected = pooled_quantile(joint, calibration.group.to_numpy(), 1 - ALPHA)
    assert np.allclose(evaluation.quantile[k], expected)
    assert np.isnan(ratio[evaluation.row("compass_l", "external")])


def test_own_region_calibration_excludes_the_evaluated_group(evaluation: Evaluation) -> None:
    groups = evaluation.groups
    own = evaluation.quantile[evaluation.row("conformal_simple", "own_region")]
    test_scores = np.abs(evaluation.v - evaluation.vhat)
    for group in np.unique(groups):
        others = groups != group
        expected = pooled_quantile(test_scores[others], groups[others], 1 - ALPHA)
        assert np.allclose(own[groups == group], expected), group


def test_subsampled_rows_draw_from_their_own_population(evaluation: Evaluation) -> None:
    """External draws from the calibration groups, own region from every test group but one."""
    subsampled = compass2d.calibration.subsampled_quantile
    calibration = evaluation.cells[evaluation.cells.role == "calibration"]
    calibration_scores = np.abs(calibration.v - calibration.vhat).to_numpy()
    external = evaluation.quantile[evaluation.row("conformal_simple", "external_subsampled")]
    expected = subsampled(
        calibration_scores, calibration.group.to_numpy(), ALPHA, SUBSAMPLES, CALIBRATE_SEED
    )
    assert np.allclose(external, expected)
    groups = evaluation.groups
    own = evaluation.quantile[evaluation.row("conformal_simple", "own_region_subsampled")]
    test_scores = np.abs(evaluation.v - evaluation.vhat)
    for group in np.unique(groups):
        others = groups != group
        expected = subsampled(
            test_scores[others], groups[others], ALPHA, SUBSAMPLES, CALIBRATE_SEED
        )
        assert np.allclose(own[groups == group], expected), group


@pytest.mark.parametrize("method", ["conformal_simple", "compass_l"])
def test_conformal_does_not_collapse_on_exchangeable_regions(
    evaluation: Evaluation, method: str
) -> None:
    results = evaluation.results
    picp = results[(results.method == method) & (results.calibration == "external")].picp.item()
    assert picp >= 0.6, (method, picp)


def test_monotone_logits_hold_the_point_estimate(evaluation: Evaluation) -> None:
    row = evaluation.row("compass_l", "external")
    lower, upper, vhat = evaluation.lower[row], evaluation.upper[row], evaluation.vhat
    assert ((lower <= vhat + 1e-6) & (vhat <= upper + 1e-6)).all()


def test_monotonicity_tables(evaluation: Evaluation) -> None:
    monotonicity = pd.read_csv(evaluation.run_dir / "monotonicity_layer.csv")
    # Adding a constant to every logit can only grow the area, whatever the model.
    uniform = monotonicity[monotonicity.direction == "uniform"]
    assert (uniform.fraction_ascending == 1.0).all()
    assert {"pca", "svm"} <= set(monotonicity.direction)
    # Choosing the layer must never read the test role.
    assert set(monotonicity.role) == {"validation"}
    endpoint = pd.read_csv(evaluation.run_dir / "monotonicity_endpoint.csv")
    assert set(endpoint.role) == {"calibration", "test"}


def test_direction_diagnostics(evaluation: Evaluation) -> None:
    diagnostics = pd.read_csv(evaluation.run_dir / "directions" / "diagnostics.csv")
    assert diagnostics.abs_cos_v1_w.between(0, 1 + 1e-9).all()
    assert diagnostics.pca_explained_variance.between(0, 1, inclusive="right").all()
    assert (diagnostics.svm_mean_jacobian_projection > 0).all()
    assert (diagnostics.n_train_cells == (evaluation.cells.role == "train").sum()).all()

    # w shares the scale convention of the PCA direction, so one beta_range serves both.
    with np.load(evaluation.run_dir / "directions" / f"{diagnostics.layer.iloc[0]}.npz") as fitted:
        assert np.isclose(fitted["svm_w"].std(ddof=1), 1.0)


def test_bootstrap_resamples_groups_and_pairs_rows_on_one_resample(evaluation: Evaluation) -> None:
    """The run's test groups, with intervals built so each bootstrap mistake widens a known CI.

    Each group covers every other cell, so only a cell resample moves coverage off 0.5. Widths
    differ by group and the second row is the first widened by 1 ha on each side, so only
    unpaired resamples move the width difference off -2.
    """
    test = evaluation.test
    sizes = test.groupby("group").size()
    assert (sizes % 2 == 0).all(), "every group must cover exactly half its cells"
    covered = (test.groupby("group").cumcount() % 2 == 0).to_numpy()
    n_rows = len(garimpo_uncertainty.intervals.COMBINATIONS)
    lower = np.broadcast_to(np.where(covered, 0.0, 0.5), (n_rows, len(test))).copy()
    upper = lower + 1 + np.unique(test.group, return_inverse=True)[1]
    lower[1], upper[1] = lower[0] - 1, upper[0] + 1
    nothing = np.full(len(test), np.nan)
    cells = garimpo_uncertainty.evaluation.EvaluatedCells(
        np.zeros(len(test)),
        np.zeros(len(test)),
        test.group.to_numpy(),
        test.region.to_numpy(),
        {"dice": nothing, "boundary_f1": nothing, "boundary_by_rule": np.zeros(len(test), bool)},
    )
    results, widths, _ = garimpo_uncertainty.evaluation.summarize(
        lower,
        upper,
        cells,
        saturation={},
        identity={},
        layer="none",
        alpha=ALPHA,
        n_bootstrap=200,
        seed=0,
    )
    assert results.picp[0] == results.picp_ci_low[0] == results.picp_ci_high[0] == 0.5
    first, second = garimpo_uncertainty.intervals.COMBINATIONS[:2]
    pair = widths[
        (widths.method_a == first[0])
        & (widths.calibration_a == first[1])
        & (widths.method_b == second[0])
        & (widths.calibration_b == second[1])
    ].iloc[0]
    assert pair.mean_difference == pair.ci_low == pair.ci_high == -2


def comparisons_at(run_dir: Path, name: str, alpha: float) -> pd.DataFrame:
    table = pd.read_csv(run_dir / "metrics" / f"{name}.csv")
    return table[table.alpha == alpha]


def test_comparisons_never_cross_levels(evaluation: Evaluation) -> None:
    for name in ("width_comparisons", "coverage_comparisons"):
        table = pd.read_csv(evaluation.run_dir / "metrics" / f"{name}.csv")
        pairs = len(list(combinations(support.EXPECTED_ROWS, 2)))
        assert table.groupby("alpha").size().to_dict() == dict.fromkeys(ALPHAS, pairs), name


def results_at(evaluation: Evaluation, alpha: float) -> pd.DataFrame:
    return evaluation.all_results[evaluation.all_results.alpha == alpha].set_index(
        ["method", "calibration"]
    )


@pytest.mark.parametrize("alpha", support.ALPHAS)
def test_width_comparisons_subtract_the_results(evaluation: Evaluation, alpha: float) -> None:
    """Every level against its own rows, so a comparison built from another level shows."""
    comparisons = comparisons_at(evaluation.run_dir, "width_comparisons", alpha)
    assert len(comparisons) == len(list(combinations(support.EXPECTED_ROWS, 2)))
    rows = results_at(evaluation, alpha).mpiw
    for pair in comparisons.itertuples():
        difference = rows[(pair.method_a, pair.calibration_a)]
        difference -= rows[(pair.method_b, pair.calibration_b)]
        assert np.isclose(pair.mean_difference, difference)
        assert pair.ci_low <= pair.ci_high


@pytest.mark.parametrize("alpha", support.ALPHAS)
def test_coverage_comparisons_subtract_the_results(evaluation: Evaluation, alpha: float) -> None:
    comparisons = comparisons_at(evaluation.run_dir, "width_comparisons", alpha)
    coverage = comparisons_at(evaluation.run_dir, "coverage_comparisons", alpha)
    assert len(coverage) == len(comparisons)
    picp = results_at(evaluation, alpha).picp
    for pair in coverage.itertuples():
        difference = picp[(pair.method_a, pair.calibration_a)]
        difference -= picp[(pair.method_b, pair.calibration_b)]
        assert np.isclose(pair.mean_difference, difference)


def has_boundary(mask: np.ndarray) -> bool:
    """Find a mining pixel with a 4-neighbor outside the mining area; the cell edge is none."""
    padded = np.pad(mask, 1, constant_values=True)
    neighbors = [padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]]
    return any((mask & ~neighbor).any() for neighbor in neighbors)


def brute_force_boundary_by_rule(root: Path, run_dir: Path, test_index: np.ndarray) -> int:
    """Count the cells with mining whose reference or prediction has no boundary to match."""
    masks = np.load(root / "masks.npy", mmap_mode="r")
    probabilities = np.load(run_dir / "predictions" / "test.npy", mmap_mode="r")
    count = 0
    for position, cell_index in enumerate(test_index.tolist()):
        reference = masks[cell_index] > 0
        prediction = probabilities[position] >= 128
        if reference.any() and not (has_boundary(reference) and has_boundary(prediction)):
            count += 1
    return count


def test_q1_spearman_on_cells_with_mining(evaluation: Evaluation) -> None:
    """Q1 recomputed apart from the pipeline."""
    run_dir, v, vhat = evaluation.run_dir, evaluation.v, evaluation.vhat
    spearman = pd.read_csv(run_dir / "metrics" / "q1_spearman.csv")
    assert len(spearman) == 4
    assert (spearman.model == support.MODEL_NAMES[evaluation.arch]).all()
    segmentation = support.read_cell_metrics(run_dir)
    assert np.array_equal(segmentation["cell_index"], evaluation.interval_index)
    dice_by_cell = brute_force_dice(evaluation.data_root, run_dir, evaluation.interval_index)
    assert np.array_equal(segmentation["dice"], dice_by_cell, equal_nan=True)
    mining = v > 0
    error_by_cell = pd.Series(np.abs(vhat - v)[mining])
    expected = error_by_cell.corr(pd.Series(segmentation["dice"][mining]), method="spearman")
    row = spearman[
        (spearman.area_error == "abs_error_ha") & (spearman.segmentation_quality == "dice")
    ].iloc[0]
    assert np.isclose(row.spearman, expected, equal_nan=True)
    boundary_cells = brute_force_boundary_by_rule(
        evaluation.data_root, run_dir, evaluation.interval_index
    )
    assert boundary_cells > 0, "the fixture needs a cell without a boundary"
    assert (evaluation.results.n_cells_boundary_by_rule == boundary_cells).all()


def area_bin(v: float, edges: list[float]) -> int:
    """0 for a cell without mining, else 1 to 5 by the quintile edges of the cells with mining."""
    if v == 0:
        return 0
    return 1 + sum(v > edge for edge in edges)


def test_coverage_by_stratum_matches_one_computation_per_cell(evaluation: Evaluation) -> None:
    """Coverage and width per stratum, never recalibrated.

    By reference area, adapted from Cheung et al. (2026), Figure 7, whose bins the paper does not
    define, with the cells without mining apart; and by
    what the model predicts, the only stratum known when an interval is used.
    """
    table = pd.read_csv(evaluation.run_dir / "metrics" / "coverage_by_stratum.csv")
    v, vhat, groups = evaluation.v, evaluation.vhat, evaluation.groups
    lower, upper = evaluation.lower, evaluation.upper
    edges = np.quantile(v[v > 0], [0.2, 0.4, 0.6, 0.8]).tolist()
    regions = evaluation.test.region.to_numpy()
    names = sorted(set(regions.tolist()))
    strata = {
        "reference_area": (np.array([area_bin(float(value), edges) for value in v]), v),
        "predicted_mining": ((vhat > 0).astype(int), vhat),
        "region": (np.array([names.index(region) for region in regions]), v),
    }
    assert (v[strata["reference_area"][0] == 0] == 0).all()
    at = table[table.alpha == ALPHA]
    assert set(at.partition) == set(strata)
    for k, (method, calibration) in enumerate(garimpo_uncertainty.intervals.COMBINATIONS):
        for partition, (labels, by) in strata.items():
            rows = at[
                (at.method == method)
                & (at.calibration == calibration)
                & (at.partition == partition)
            ]
            assert rows.n_cells.sum() == len(v), (method, calibration, partition)
            for row in rows.itertuples():
                inside = np.flatnonzero(labels == row.stratum)
                covered = [float(lower[k, i] <= v[i] <= upper[k, i]) for i in inside]
                width = [float(upper[k, i] - lower[k, i]) for i in inside]
                assert row.n_cells == len(inside)
                assert row.n_groups == len(np.unique(groups[inside]))
                assert np.isclose(row.picp, mean_of_group_means(covered, groups[inside]))
                assert np.isclose(row.mpiw, mean_of_group_means(width, groups[inside]))
                assert row.low_ha == by[inside].min()
                assert row.high_ha == by[inside].max()
                if partition == "region":
                    assert row.label == names[row.stratum]
    assert set(table.alpha) == set(ALPHAS)


def test_score_relation_is_a_log_log_fit_on_calibration(evaluation: Evaluation) -> None:
    """Cheung et al. (2026), Figure 4: COMPASS scores against split-conformal scores.

    Zero scores have no logarithm and saturated ones are censored at beta_range, so both leave.
    """
    table = pd.read_csv(evaluation.run_dir / "metrics" / "score_relation.csv").set_index("method")
    calibration = evaluation.cells[evaluation.cells.role == "calibration"]
    with np.load(evaluation.run_dir / "scores" / "scores.npz") as stored:
        at = garimpo_uncertainty.data.positions_of(
            stored["cell_index"], calibration.cell_index.to_numpy()
        )
        compass_scores, saturated = stored["score"][:, at], stored["saturated"][:, at]
    split_scores = np.abs(calibration.v - calibration.vhat).to_numpy()
    assert set(table.index) == set(garimpo_uncertainty.intervals.COMPASS_METHODS)
    for m, method in enumerate(garimpo_uncertainty.intervals.COMPASS_METHODS):
        kept = (split_scores > 0) & (compass_scores[m] > 0) & ~saturated[m]
        row = table.loc[method]
        assert row.n_cells == kept.sum(), method
        assert row.n_saturated == saturated[m].sum(), method
        x = np.log(split_scores[kept])
        if np.unique(x).size < 2:
            assert np.isnan(row.slope), method
            assert np.isnan(row.intercept), method
            continue
        slope, intercept = np.polyfit(x, np.log(compass_scores[m][kept]), 1)
        assert np.isclose(row.slope, slope), method
        assert np.isclose(row.intercept, intercept), method


@dataclass(frozen=True)
class CompassInputs:
    model: torch.nn.Module
    metadata: dict
    test: pd.DataFrame
    jacobians: np.ndarray
    fitted: compass2d.directions.Directions
    compass_rows: list[tuple[int, str]]

    @property
    def rows(self) -> list[int]:
        return [k for k, _ in self.compass_rows]


@pytest.fixture(scope="module")
def compass_inputs(evaluation: Evaluation) -> CompassInputs:
    intervals = garimpo_uncertainty.intervals
    predictions = evaluation.cells
    run_dir = evaluation.run_dir
    model, metadata = garimpo_uncertainty.checkpoints.load_model(
        run_dir / "best_model.pt", predictions, "cpu"
    )
    test = predictions.iloc[
        np.searchsorted(predictions.cell_index.to_numpy(), evaluation.interval_index)
    ]
    with np.load(run_dir / "scores" / "scores.npz") as stored:
        jacobians = stored["jacobians"][
            np.searchsorted(stored["cell_index"], evaluation.interval_index)
        ]
    fitted = compass2d.directions.Directions.load(
        run_dir / "directions" / f"{evaluation.layer}.npz"
    )
    compass_rows = [
        (k, method)
        for k, (method, _) in enumerate(intervals.COMBINATIONS)
        if method in intervals.COMPASS_METHODS + intervals.ASYMMETRIC_METHODS
    ]
    return CompassInputs(model, metadata, test, jacobians, fitted, compass_rows)


# These recompute the stage in this process, on CPU; exact equality holds only on one device.
same_device_as_the_stage = pytest.mark.skipif(
    support.DEVICE != "cpu", reason="the stage ran on the GPU, this process recomputes on CPU"
)


@same_device_as_the_stage
@pytest.mark.parametrize("alpha", support.ALPHAS)
def test_each_compass_row_alone_matches_the_one_pass(
    evaluation: Evaluation, compass_inputs: CompassInputs, alpha: float
) -> None:
    """Evaluate each COMPASS row with its own loader against the stage's one pass.

    Exact equality, no tolerance: same weights, same batches, same CPU code, so any difference is a
    change in what is computed, and the rewrite promised none.
    """
    intervals = garimpo_uncertainty.intervals
    pixel_ha = garimpo_uncertainty.data.PIXEL_HA
    model, test = compass_inputs.model, compass_inputs.test
    compass_rows = compass_inputs.compass_rows
    assert len(compass_rows) == 31
    level = ALPHAS.index(alpha)

    def soft(logits: torch.Tensor) -> torch.Tensor:
        return compass2d.metrics.soft_area(logits, pixel_ha, compass2d.metrics.UPSTREAM_TEMPERATURE)

    def hard(logits: torch.Tensor) -> torch.Tensor:
        return compass2d.metrics.hard_area(logits, pixel_ha)

    alone_lower = np.empty((len(compass_rows), len(test)))
    alone_upper = np.empty((len(compass_rows), len(test)))
    with np.load(evaluation.run_dir / "scores" / "scores.npz") as stored:
        beta_range = dict(zip(intervals.COMPASS_METHODS, stored["beta_range"], strict=True))
    with np.load(evaluation.run_dir / "calibration" / "quantiles.npz") as stored:
        side_ratio = stored["side_ratio"][level]
    whole_cell = np.zeros((len(compass_rows), len(test)), dtype=bool)
    with torch.no_grad():
        for row, (k, method) in enumerate(compass_rows):
            first = 0
            for images, _, _ in garimpo_uncertainty.cell_dataset.loader(
                evaluation.data_root, test, compass_inputs.metadata["normalization"], 8, 0
            ):
                last = first + len(images)
                features = garimpo_uncertainty.models.common.encode(model, images)
                quantile = evaluation.all_quantile[level, k, first:last]
                base = method.removeprefix("a_")
                asymmetric = base != method
                ratio = float(side_ratio[k])
                # Censored quantiles give the whole cell, inf included; an asymmetric row's upper
                # side is censored at beta_range / ratio.
                censor = min(beta_range[base], beta_range[base] / ratio) if asymmetric else None
                whole_cell[row, first:last] = quantile >= (censor or beta_range[base])
                beta = torch.as_tensor(np.where(whole_cell[row, first:last], 0.0, quantile))
                if base == "compass_l":
                    logits, _ = garimpo_uncertainty.models.common.split_at(
                        model, garimpo_uncertainty.models.common.LOGITS, features
                    )
                    space = (lambda zi: zi, hard, logits, torch.ones_like(logits))
                else:
                    z, decode = garimpo_uncertainty.models.common.split_at(
                        model, evaluation.layer, features
                    )
                    direction = base.removeprefix("compass_j_")
                    deltas = compass2d.directions.cell_deltas(
                        direction,
                        compass_inputs.jacobians[first:last].astype(np.float64),
                        compass_inputs.fitted,
                    )
                    delta = compass2d.compass.channel_delta(torch.from_numpy(deltas.copy()).to(z))
                    space = (decode, soft, z, delta)
                if asymmetric:
                    low = compass2d.compass.shifted(*space, -beta)
                    high = compass2d.compass.shifted(*space, ratio * beta)
                else:
                    low, high = compass2d.compass.endpoints(*space, beta)
                alone_lower[row, first:last] = low.numpy()
                alone_upper[row, first:last] = high.numpy()
                first = last
    cell_ha = np.broadcast_to(test.cell_ha.to_numpy(), whole_cell.shape)
    alone_lower[whole_cell], alone_upper[whole_cell] = 0.0, cell_ha[whole_cell]
    alone_lower, alone_upper = intervals.clip_to_cell(
        alone_lower, alone_upper, test.cell_ha.to_numpy()
    )
    assert np.array_equal(alone_lower, evaluation.all_lower[level, compass_inputs.rows])
    assert np.array_equal(alone_upper, evaluation.all_upper[level, compass_inputs.rows])


def test_loaded_model_scores_each_cell_independently_of_its_batch(
    compass_inputs: CompassInputs,
) -> None:
    """jacobian_sum takes per-cell gradients from one backward pass over a batch.

    That holds only if no cell's output depends on another's, which batch norm in train mode
    breaks; the stages get their model from load_model alone, so it must leave eval mode on.
    """
    model = compass_inputs.model
    assert not any(module.training for module in model.modules())
    assert not any(parameter.requires_grad for parameter in model.parameters())
    torch.manual_seed(0)
    x = torch.randn(3, compass_inputs.metadata["in_channels"], support.CELL_PX, support.CELL_PX)
    with torch.no_grad():
        together = model(x)
        alone = torch.cat([model(x[i : i + 1]) for i in range(len(x))])
    assert torch.allclose(together, alone, atol=1e-5)


def test_empty_cell_gets_the_scaled_mean_direction(compass_inputs: CompassInputs) -> None:
    """A cell predicted empty has a zero Jacobian; its PCA direction must be finite, never NaN.

    With no mean either, the direction is zero, and dividing by its spread must raise instead.
    """
    fitted = compass_inputs.fitted
    jacobians = compass_inputs.jacobians.copy()
    jacobians[0] = 0
    deltas = compass2d.directions.cell_deltas("pca", jacobians, fitted)
    assert np.isfinite(deltas).all()
    np.testing.assert_allclose(deltas[0], fitted.pca_mean / fitted.pca_mean.std(ddof=1))

    no_mean = compass2d.directions.Directions(
        fitted.pca_components,
        np.zeros_like(fitted.pca_mean),
        fitted.pca_explained_variance,
        fitted.svm_w,
    )
    with pytest.raises(ValueError, match="constant across channels"):
        compass2d.directions.cell_deltas("pca", jacobians, no_mean)


@same_device_as_the_stage
def test_stage_reads_each_cell_once_and_encodes_each_batch_once(
    evaluation: Evaluation, compass_inputs: CompassInputs
) -> None:
    """The stage's own function, in this process, so its reads and encoder passes can be counted."""
    model, test = compass_inputs.model, compass_inputs.test
    stage = runpy.run_path(str(support.ROOT / "scripts" / "generate_intervals.py"))
    first_layer = model.encoder.conv1
    encoder_passes: list[int] = []
    handle = first_layer.register_forward_hook(lambda *_: encoder_passes.append(1))
    reads: list[int] = []
    read_cell = garimpo_uncertainty.cell_dataset.CellDataset.__getitem__

    def counted_read(self: garimpo_uncertainty.cell_dataset.CellDataset, index: int) -> tuple:
        reads.append(index)
        return read_cell(self, index)

    args = argparse.Namespace(
        temperature=compass2d.metrics.UPSTREAM_TEMPERATURE,
        batch_size=8,
        num_workers=0,
        layer=evaluation.layer,
    )
    lower = np.full_like(evaluation.all_lower, np.nan)
    upper = np.full_like(evaluation.all_upper, np.nan)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(garimpo_uncertainty.cell_dataset.CellDataset, "__getitem__", counted_read)
        calibrated = stage["Calibrated"].read(evaluation.run_dir)
        stage["compass_endpoints"](
            model,
            calibrated,
            test,
            compass_inputs.jacobians,
            compass_inputs.fitted,
            root=evaluation.data_root,
            metadata=compass_inputs.metadata,
            args=args,
            device="cpu",
            lower=lower,
            upper=upper,
        )
    handle.remove()
    assert sorted(reads) == list(range(len(test)))
    assert len(encoder_passes) == int(np.ceil(len(test) / 8))
    rows = compass_inputs.rows
    stage["widen_saturated"](calibrated, test.cell_ha.to_numpy(), lower, upper)
    clipped_lower, clipped_upper = garimpo_uncertainty.intervals.clip_to_cell(
        lower[:, rows], upper[:, rows], test.cell_ha.to_numpy()
    )
    assert np.array_equal(clipped_lower, evaluation.all_lower[:, rows])
    assert np.array_equal(clipped_upper, evaluation.all_upper[:, rows])


def test_boundary_f1_without_a_boundary() -> None:
    whole = np.ones((8, 8), dtype=bool)
    empty = np.zeros((8, 8), dtype=bool)
    partial = empty.copy()
    partial[2:6, 2:6] = True
    assert garimpo_uncertainty.evaluation.boundary_f1(whole, whole, 1) == 1.0
    assert garimpo_uncertainty.evaluation.boundary_f1(empty, whole, 1) == 0.0
    assert garimpo_uncertainty.evaluation.boundary_f1(partial, whole, 1) == 0.0
