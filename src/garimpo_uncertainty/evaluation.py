"""Metrics on persisted predictions and intervals; no model, no GPU, no torch.

Each group counts once: per-group means are averaged, matching the group-weighted calibration, and
the bootstrap resamples whole groups with one index matrix shared by every metric and method.
"""

import warnings
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


def spearman_by_model(
    v: np.ndarray,
    vhat: np.ndarray,
    group: np.ndarray,
    segmentation: dict[str, np.ndarray],
    identity: dict[str, object],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Q1: rank correlation of area error with Dice and boundary F1, on cells with mining.

    Intervals resample whole groups.
    """
    mining = v > 0
    abs_error_ha = np.abs(vhat[mining] - v[mining])
    errors = {"abs_error_ha": abs_error_ha, "abs_error_percent": 100 * abs_error_ha / v[mining]}
    kept_groups = group[mining]
    groups = np.unique(kept_groups)
    positions = [np.flatnonzero(kept_groups == one) for one in groups]
    indices = np.random.default_rng(seed).integers(0, len(groups), (n_bootstrap, len(groups)))
    rows = []
    with warnings.catch_warnings():
        # A constant column, such as a resample where Dice never varies, gives NaN.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for error_name, error in errors.items():
            for quality in ("dice", "boundary_f1"):
                pair = np.column_stack([error, segmentation[quality][mining]])
                resampled = [
                    _spearman(pair[np.concatenate([positions[i] for i in chosen])])
                    for chosen in indices
                ]
                low, high = np.nanpercentile(resampled, CONFIDENCE)
                rows.append(
                    identity
                    | {
                        "area_error": error_name,
                        "segmentation_quality": quality,
                        "spearman": _spearman(pair),
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "n_cells": len(pair),
                        "n_groups": len(groups),
                    }
                )
    return pd.DataFrame(rows)


def summarize(
    lower: np.ndarray,
    upper: np.ndarray,
    v: np.ndarray,
    vhat: np.ndarray,
    group: np.ndarray,
    segmentation: dict[str, np.ndarray],
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
    width = upper - lower
    covered = ((v >= lower) & (v <= upper)).astype(float)
    penalty = (2 / alpha) * (np.clip(lower - v, 0, None) + np.clip(v - upper, 0, None))
    error = vhat - v
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = np.where(v > 0, np.abs(error) / v, np.nan)
    columns = {
        "picp": covered,
        "mpiw": width,
        "interval_score": width + penalty,
        "mae": np.abs(error),
        "bias": error,
        "relative_error": relative,
        "dice": segmentation["dice"],
        "boundary_f1": segmentation["boundary_f1"],
    }
    groups = np.unique(group)
    membership = (group[None, :] == groups[:, None]).astype(float)
    indices = np.random.default_rng(seed).integers(0, len(groups), (n_bootstrap, len(groups)))
    combos = garimpo_uncertainty.intervals.COMBINATIONS
    table: dict[str, object] = {
        **identity,
        "layer": layer,
        "method": [method for method, _ in combos],
        "calibration": [calibration for _, calibration in combos],
        "alpha": alpha,
        "n_cells": len(v),
        "n_groups": len(groups),
        "n_cells_boundary_by_rule": int(segmentation["boundary_by_rule"].sum()),
    }
    with warnings.catch_warnings():
        # Groups with no mining cell give NaN in the segmentation metrics; nanmean skips them.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for name, values in columns.items():
            means = np.broadcast_to(_group_means(values, membership), (len(combos), len(groups)))
            low, high = np.nanpercentile(np.nanmean(means[:, indices], axis=2), CONFIDENCE, axis=1)
            table |= {
                name: np.nanmean(means, axis=1),
                f"{name}_ci_low": low,
                f"{name}_ci_high": high,
            }

    results = pd.DataFrame(table | saturation)
    return (
        results,
        _paired_differences(_group_means(width, membership), indices, identity),
        _paired_differences(_group_means(covered, membership), indices, identity),
    )
