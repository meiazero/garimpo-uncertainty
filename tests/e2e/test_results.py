"""The numbers of a finished run, recomputed apart from the pipeline."""

import argparse
import runpy
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

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
import garimpo_uncertainty.models
from tests.e2e import support

ALPHA = 0.1


@dataclass(frozen=True)
class Evaluation:
    arch: str
    run_dir: Path
    layer: str
    data_root: Path
    cells: pd.DataFrame
    results: pd.DataFrame
    test: pd.DataFrame
    interval_index: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    quantile: np.ndarray

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


def test_array_metrics_match_one_computation_per_cell(evaluation: Evaluation) -> None:
    """Array metrics in results.csv against one Python computation per cell, by group means."""
    v, groups, lower, upper = evaluation.v, evaluation.groups, evaluation.lower, evaluation.upper
    dice_by_cell = brute_force_dice(
        evaluation.data_root, evaluation.run_dir, evaluation.interval_index
    )
    for k, row in enumerate(evaluation.results.itertuples()):
        per_cell = [
            brute_force_metrics(float(lower[k, i]), float(upper[k, i]), float(v[i]), ALPHA)
            for i in range(len(v))
        ]
        covered, width, score = zip(*per_cell, strict=True)
        assert np.isclose(row.picp, mean_of_group_means(list(covered), groups)), k
        assert np.isclose(row.mpiw, mean_of_group_means(list(width), groups)), k
        assert np.isclose(row.interval_score, mean_of_group_means(list(score), groups)), k
        assert np.isclose(row.dice, mean_of_group_means(dice_by_cell, groups)), k
        assert row.picp_ci_low <= row.picp_ci_high
        assert row.n_groups == len(np.unique(groups))


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


def test_own_region_calibration_excludes_the_evaluated_group(evaluation: Evaluation) -> None:
    groups = evaluation.groups
    own = evaluation.quantile[evaluation.row("conformal_simple", "own_region")]
    test_scores = np.abs(evaluation.v - evaluation.vhat)
    for group in np.unique(groups):
        others = groups != group
        expected = pooled_quantile(test_scores[others], groups[others], 1 - ALPHA)
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


def test_width_comparisons_subtract_the_results(evaluation: Evaluation) -> None:
    comparisons = pd.read_csv(evaluation.run_dir / "metrics" / "width_comparisons.csv")
    assert len(comparisons) == len(list(combinations(support.EXPECTED_ROWS, 2)))
    rows = evaluation.results.set_index(["method", "calibration"]).mpiw
    for pair in comparisons.itertuples():
        difference = rows[(pair.method_a, pair.calibration_a)]
        difference -= rows[(pair.method_b, pair.calibration_b)]
        assert np.isclose(pair.mean_difference, difference)
        assert pair.ci_low <= pair.ci_high


def test_coverage_comparisons_subtract_the_results(evaluation: Evaluation) -> None:
    comparisons = pd.read_csv(evaluation.run_dir / "metrics" / "width_comparisons.csv")
    coverage = pd.read_csv(evaluation.run_dir / "metrics" / "coverage_comparisons.csv")
    assert len(coverage) == len(comparisons)
    picp = evaluation.results.set_index(["method", "calibration"]).picp
    for pair in coverage.itertuples():
        difference = picp[(pair.method_a, pair.calibration_a)]
        difference -= picp[(pair.method_b, pair.calibration_b)]
        assert np.isclose(pair.mean_difference, difference)


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
    boundary_cells = segmentation["boundary_by_rule"].sum()
    assert (evaluation.results.n_cells_boundary_by_rule == boundary_cells).all()


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
        if method in intervals.COMPASS_METHODS
    ]
    return CompassInputs(model, metadata, test, jacobians, fitted, compass_rows)


def test_each_compass_row_alone_matches_the_one_pass(
    evaluation: Evaluation, compass_inputs: CompassInputs
) -> None:
    """Evaluate each COMPASS row with its own loader against the stage's one pass.

    Exact equality, no tolerance: same weights, same batches, same CPU code, so any difference is a
    change in what is computed, and the rewrite promised none.
    """
    intervals = garimpo_uncertainty.intervals
    pixel_ha = garimpo_uncertainty.data.PIXEL_HA
    model, test = compass_inputs.model, compass_inputs.test
    compass_rows = compass_inputs.compass_rows
    assert len(compass_rows) == 7

    def soft(logits: torch.Tensor) -> torch.Tensor:
        return compass2d.metrics.soft_area(logits, pixel_ha, compass2d.metrics.UPSTREAM_TEMPERATURE)

    def hard(logits: torch.Tensor) -> torch.Tensor:
        return compass2d.metrics.hard_area(logits, pixel_ha)

    alone_lower = np.empty((len(compass_rows), len(test)))
    alone_upper = np.empty((len(compass_rows), len(test)))
    with torch.no_grad():
        for row, (k, method) in enumerate(compass_rows):
            first = 0
            for images, _, _ in garimpo_uncertainty.cell_dataset.loader(
                evaluation.data_root, test, compass_inputs.metadata["normalization"], 8, 0
            ):
                last = first + len(images)
                features = garimpo_uncertainty.models.encode(model, images)
                beta = torch.as_tensor(evaluation.quantile[k, first:last])
                if method == "compass_l":
                    logits, _ = garimpo_uncertainty.models.split_at(
                        model, garimpo_uncertainty.models.LOGITS, features
                    )
                    low, high = compass2d.compass.endpoints(
                        lambda zi: zi, hard, logits, torch.ones_like(logits), beta
                    )
                else:
                    z, decode = garimpo_uncertainty.models.split_at(
                        model, evaluation.layer, features
                    )
                    direction = method.removeprefix("compass_j_")
                    deltas = compass2d.directions.cell_deltas(
                        direction,
                        compass_inputs.jacobians[first:last].astype(np.float64),
                        compass_inputs.fitted,
                    )
                    delta = compass2d.compass.channel_delta(torch.from_numpy(deltas.copy()).to(z))
                    low, high = compass2d.compass.endpoints(decode, soft, z, delta, beta)
                alone_lower[row, first:last] = low.numpy()
                alone_upper[row, first:last] = high.numpy()
                first = last
    alone_lower, alone_upper = intervals.clip_to_cell(
        alone_lower, alone_upper, test.cell_ha.to_numpy()
    )
    assert np.array_equal(alone_lower, evaluation.lower[compass_inputs.rows])
    assert np.array_equal(alone_upper, evaluation.upper[compass_inputs.rows])


def test_stage_reads_each_cell_once_and_encodes_each_batch_once(
    evaluation: Evaluation, compass_inputs: CompassInputs
) -> None:
    """The stage's own function, in this process, so its reads and encoder passes can be counted."""
    model, test = compass_inputs.model, compass_inputs.test
    stage = runpy.run_path(str(support.ROOT / "scripts" / "generate_intervals.py"))
    first_layer = model.down[0] if evaluation.arch == "convnet" else model.encoder.conv1
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
    lower = np.full_like(evaluation.lower, np.nan)
    upper = np.full_like(evaluation.upper, np.nan)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(garimpo_uncertainty.cell_dataset.CellDataset, "__getitem__", counted_read)
        stage["compass_endpoints"](
            model,
            evaluation.quantile,
            test,
            compass_inputs.jacobians,
            compass_inputs.fitted,
            evaluation.data_root,
            compass_inputs.metadata,
            args,
            "cpu",
            lower,
            upper,
        )
    handle.remove()
    assert sorted(reads) == list(range(len(test)))
    assert len(encoder_passes) == int(np.ceil(len(test) / 8))
    rows = compass_inputs.rows
    clipped_lower, clipped_upper = garimpo_uncertainty.intervals.clip_to_cell(
        lower[rows], upper[rows], test.cell_ha.to_numpy()
    )
    assert np.array_equal(clipped_lower, evaluation.lower[rows])
    assert np.array_equal(clipped_upper, evaluation.upper[rows])


def test_boundary_f1_without_a_boundary() -> None:
    whole = np.ones((8, 8), dtype=bool)
    empty = np.zeros((8, 8), dtype=bool)
    partial = empty.copy()
    partial[2:6, 2:6] = True
    assert garimpo_uncertainty.evaluation.boundary_f1(whole, whole, 1) == 1.0
    assert garimpo_uncertainty.evaluation.boundary_f1(empty, whole, 1) == 0.0
    assert garimpo_uncertainty.evaluation.boundary_f1(partial, whole, 1) == 0.0
