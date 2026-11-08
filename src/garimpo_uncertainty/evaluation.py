"""Metrics on persisted predictions and intervals; no model, no GPU, no torch.

Each group counts once: per-group means are averaged, matching the group-weighted calibration, and
the bootstrap resamples whole groups with one index matrix shared by every metric and method.
"""

import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.stats

import garimpo_uncertainty.intervals

PROBABILITY_SCALE = 255
HARD_THRESHOLD_U8 = 128  # round(0.5 * 255), so u8 >= 128 is p >= 0.5
CONFIDENCE = (2.5, 97.5)
MIN_PAIRS_FOR_CORRELATION = 2


@dataclass(frozen=True)
class EvaluatedCells:
    """The test cells under evaluation, aligned: areas, group, region and segmentation quality."""

    v: np.ndarray
    vhat: np.ndarray
    group: np.ndarray
    region: np.ndarray
    segmentation: dict[str, np.ndarray]


def dice(prediction: np.ndarray, reference: np.ndarray) -> float:
    total = prediction.sum() + reference.sum()
    return 2 * float((prediction & reference).sum()) / float(total) if total else float("nan")


def _boundary(mask: np.ndarray) -> np.ndarray:
    # border_value=1: the cell edge is a cut of the image, not a boundary of the mining area.
    return mask & ~scipy.ndimage.binary_erosion(mask, border_value=1)


def boundary_f1(prediction: np.ndarray, reference: np.ndarray, tolerance_px: float) -> float:
    """F1 of boundary pixels matched within a tolerance (Csurka et al., 2013)."""
    predicted, actual = _boundary(prediction), _boundary(reference)
    # A mask filling the cell has no boundary, and NaN would drop those cells from the means.
    # Without a boundary to match, F1 is 1 for identical masks and 0 otherwise; results.csv counts
    # these cells.
    if not predicted.any() and not actual.any():
        return float(np.array_equal(prediction, reference))
    if not predicted.any() or not actual.any():
        return 0.0
    near_actual = scipy.ndimage.distance_transform_edt(~actual) <= tolerance_px
    near_predicted = scipy.ndimage.distance_transform_edt(~predicted) <= tolerance_px
    precision = (predicted & near_actual).sum() / predicted.sum()
    recall = (actual & near_predicted).sum() / actual.sum()
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def segmentation_metrics(
    root: Path, cell_index: np.ndarray, prediction_dir: Path, tolerance_px: float
) -> dict[str, np.ndarray]:
    """Per-cell Dice and boundary F1 on test cells with mining, from the persisted probabilities."""
    masks = np.load(root / "masks.npy", mmap_mode="r")
    probabilities = np.load(prediction_dir / "test.npy", mmap_mode="r")
    dice_values = np.full(len(cell_index), np.nan)
    boundary_values = np.full(len(cell_index), np.nan)
    by_rule = np.zeros(len(cell_index), dtype=bool)
    for position, index in enumerate(cell_index):
        reference = masks[index] > 0
        if reference.any():
            prediction = probabilities[position] >= HARD_THRESHOLD_U8
            dice_values[position] = dice(prediction, reference)
            boundary_values[position] = boundary_f1(prediction, reference, tolerance_px)
            by_rule[position] = not (_boundary(prediction).any() and _boundary(reference).any())
    return {"dice": dice_values, "boundary_f1": boundary_values, "boundary_by_rule": by_rule}


def _group_means(values: np.ndarray, membership: np.ndarray) -> np.ndarray:
    """Return per-group means of [..., N] values, skipping NaN; membership is [G, N]."""
    present = ~np.isnan(values)
    totals = np.where(present, values, 0.0) @ membership.T
    with np.errstate(invalid="ignore"):
        return totals / (present.astype(float) @ membership.T)


def _paired_differences(
    values: np.ndarray, indices: np.ndarray, identity: dict[str, object]
) -> pd.DataFrame:
    """Differences between rows on the same group resamples, so the comparison stays paired."""
    rows = []
    combos = garimpo_uncertainty.intervals.COMBINATIONS
    for a, b in combinations(range(len(combos)), 2):
        low, high = np.percentile((values[a] - values[b])[indices].mean(axis=1), CONFIDENCE)
        rows.append(
            identity
            | {
                "method_a": combos[a][0],
                "calibration_a": combos[a][1],
                "method_b": combos[b][0],
                "calibration_b": combos[b][1],
                "mean_difference": float((values[a] - values[b]).mean()),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return pd.DataFrame(rows)


def _spearman(pairs: np.ndarray) -> float:
    kept = pairs[~np.isnan(pairs).any(axis=1)]
    if len(kept) < MIN_PAIRS_FOR_CORRELATION:
        return float("nan")
    return float(scipy.stats.spearmanr(kept[:, 0], kept[:, 1]).statistic)


def _spearman_interval(
    pair: np.ndarray, positions: list[np.ndarray], indices: np.ndarray
) -> tuple[float, float]:
    resampled = [
        _spearman(pair[np.concatenate([positions[i] for i in chosen])]) for chosen in indices
    ]
    low, high = np.nanpercentile(resampled, CONFIDENCE)
    return float(low), float(high)


def _area_errors(v: np.ndarray, vhat: np.ndarray) -> dict[str, np.ndarray]:
    abs_error_ha = np.abs(vhat - v)
    return {"abs_error_ha": abs_error_ha, "abs_error_percent": 100 * abs_error_ha / v}


def spearman_by_model(
    cells: EvaluatedCells, *, identity: dict[str, object], n_bootstrap: int, seed: int
) -> pd.DataFrame:
    """Q1: rank correlation of area error with Dice and boundary F1, on cells with mining.

    Intervals resample whole groups.
    """
    mining = cells.v > 0
    errors = _area_errors(cells.v[mining], cells.vhat[mining])
    kept_groups = cells.group[mining]
    groups = np.unique(kept_groups)
    positions = [np.flatnonzero(kept_groups == one) for one in groups]
    indices = np.random.default_rng(seed).integers(0, len(groups), (n_bootstrap, len(groups)))
    rows = []
    with warnings.catch_warnings():
        # A constant column, such as a resample where Dice never varies, gives NaN.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for error_name, error in errors.items():
            for quality in ("dice", "boundary_f1"):
                pair = np.column_stack([error, cells.segmentation[quality][mining]])
                low, high = _spearman_interval(pair, positions, indices)
                rows.append(
                    identity
                    | {
                        "area_error": error_name,
                        "segmentation_quality": quality,
                        "spearman": _spearman(pair),
                        "ci_low": low,
                        "ci_high": high,
                        "n_cells": len(pair),
                        "n_groups": len(groups),
                    }
                )
    return pd.DataFrame(rows)


def _metric_columns(
    lower: np.ndarray, upper: np.ndarray, cells: EvaluatedCells, alpha: float
) -> dict[str, np.ndarray]:
    """Return per-cell values of every reported metric, in the order of the results table."""
    width = upper - lower
    penalty = (2 / alpha) * (np.clip(lower - cells.v, 0, None) + np.clip(cells.v - upper, 0, None))
    error = cells.vhat - cells.v
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = np.where(cells.v > 0, np.abs(error) / cells.v, np.nan)
    return {
        "picp": ((cells.v >= lower) & (cells.v <= upper)).astype(float),
        "mpiw": width,
        "interval_score": width + penalty,
        "mae": np.abs(error),
        "bias": error,
        "relative_error": relative,
        "dice": cells.segmentation["dice"],
        "boundary_f1": cells.segmentation["boundary_f1"],
    }


def _means_and_intervals(
    columns: dict[str, np.ndarray], membership: np.ndarray, indices: np.ndarray
) -> dict[str, np.ndarray]:
    """Group-weighted mean and bootstrap interval of each column, one value per combination."""
    n_combinations = len(garimpo_uncertainty.intervals.COMBINATIONS)
    table = {}
    with warnings.catch_warnings():
        # Groups with no mining cell give NaN in the segmentation metrics; nanmean skips them.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for name, values in columns.items():
            means = np.broadcast_to(
                _group_means(values, membership), (n_combinations, len(membership))
            )
            low, high = np.nanpercentile(np.nanmean(means[:, indices], axis=2), CONFIDENCE, axis=1)
            table |= {
                name: np.nanmean(means, axis=1),
                f"{name}_ci_low": low,
                f"{name}_ci_high": high,
            }
    return table


def summarize(
    lower: np.ndarray,
    upper: np.ndarray,
    cells: EvaluatedCells,
    *,
    saturation: dict[str, np.ndarray],
    identity: dict[str, object],
    layer: str,
    alpha: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One row per method x calibration of one run, plus paired width and coverage differences.

    lower and upper are [K, N], K indexing COMBINATIONS. Q3 reads the coverage differences of a
    method between its calibration situations.
    """
    columns = _metric_columns(lower, upper, cells, alpha)
    groups = np.unique(cells.group)
    membership = (cells.group[None, :] == groups[:, None]).astype(float)
    indices = np.random.default_rng(seed).integers(0, len(groups), (n_bootstrap, len(groups)))
    combos = garimpo_uncertainty.intervals.COMBINATIONS
    table: dict[str, object] = {
        **identity,
        "layer": layer,
        "method": [method for method, _ in combos],
        "calibration": [calibration for _, calibration in combos],
        "alpha": alpha,
        "n_cells": len(cells.v),
        "n_groups": len(groups),
        "n_cells_boundary_by_rule": int(cells.segmentation["boundary_by_rule"].sum()),
    }
    results = pd.DataFrame(table | _means_and_intervals(columns, membership, indices) | saturation)
    return (
        results,
        _paired_differences(_group_means(columns["mpiw"], membership), indices, identity),
        _paired_differences(_group_means(columns["picp"], membership), indices, identity),
    )


AREA_QUANTILES = (0.2, 0.4, 0.6, 0.8)


def area_bins(v: np.ndarray) -> np.ndarray:
    """0 for cells without mining, 1 to 5 by quintile of the reference area of those with it."""
    if not (v > 0).any():
        return np.zeros(len(v), dtype=int)
    edges = np.quantile(v[v > 0], AREA_QUANTILES)
    return np.where(v > 0, 1 + np.searchsorted(edges, v, side="left"), 0)


def strata(cells: EvaluatedCells) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Each partition's stratum per cell, the area that defines it, and each stratum's label.

    The held-out region is the shift itself, so its coverage is the one the guarantee is about.
    """
    regions, region_codes = np.unique(cells.region, return_inverse=True)
    return {
        "reference_area": (area_bins(cells.v), cells.v, np.arange(6).astype(str)),
        "predicted_mining": ((cells.vhat > 0).astype(int), cells.vhat, np.array(["0", "1"])),
        "region": (region_codes, cells.v, regions),
    }


def coverage_by_stratum(
    lower: np.ndarray,
    upper: np.ndarray,
    cells: EvaluatedCells,
    *,
    identity: dict[str, object],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """PICP and MPIW per stratum, group-weighted, from intervals calibrated once on every cell.

    By reference area, adapted from Cheung et al. (2026), Figure 7, whose bins it does not
    define; by predicted mining, the only stratum known when an interval is used; by held-out
    region. The guarantee is marginal, so a stratum may fall short.
    """
    combos = garimpo_uncertainty.intervals.COMBINATIONS
    covered = ((cells.v >= lower) & (cells.v <= upper)).astype(float)
    frames = []
    for partition, (labels, by, names) in strata(cells).items():
        for stratum in np.unique(labels):
            inside = labels == stratum
            groups = np.unique(cells.group[inside])
            membership = (cells.group[inside][None, :] == groups[:, None]).astype(float)
            indices = np.random.default_rng(seed).integers(
                0, len(groups), (n_bootstrap, len(groups))
            )
            columns = {"picp": covered[:, inside], "mpiw": (upper - lower)[:, inside]}
            table = {
                **identity,
                "method": [method for method, _ in combos],
                "calibration": [calibration for _, calibration in combos],
                "partition": partition,
                "stratum": int(stratum),
                "label": str(names[stratum]),
                "low_ha": float(by[inside].min()),
                "high_ha": float(by[inside].max()),
                "n_cells": int(inside.sum()),
                "n_groups": len(groups),
            }
            means = _means_and_intervals(columns, membership, indices)
            frames.append(pd.DataFrame(table | means))
    return pd.concat(frames, ignore_index=True)


def score_relation(
    split_scores: np.ndarray, compass_scores: np.ndarray, saturated: np.ndarray
) -> pd.DataFrame:
    """Cheung et al. (2026), Figure 4: log-log fit of COMPASS scores on split-conformal scores.

    A slope below 1 compresses the tail of the scores, which is why their quantile is smaller. Zero
    scores have no logarithm, and saturated ones are censored at beta_range, so both are left out.
    """
    rows = []
    for m, method in enumerate(garimpo_uncertainty.intervals.COMPASS_METHODS):
        kept = (split_scores > 0) & (compass_scores[m] > 0) & ~saturated[m]
        x, y = np.log(split_scores[kept]), np.log(compass_scores[m][kept])
        # Every score saturated leaves nothing to fit; NaN says so instead of failing the stage.
        fitted = len(x) >= MIN_PAIRS_FOR_CORRELATION
        slope, intercept = np.polyfit(x, y, 1) if fitted else (np.nan, np.nan)
        rows.append(
            {
                "method": method,
                "slope": float(slope),
                "intercept": float(intercept),
                "spearman": _spearman(np.column_stack([x, y])),
                "n_cells": int(kept.sum()),
                "n_saturated": int(saturated[m].sum()),
            }
        )
    return pd.DataFrame(rows)
