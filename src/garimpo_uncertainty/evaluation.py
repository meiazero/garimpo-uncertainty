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

import garimpo_uncertainty.data

PROBABILITY_SCALE = 255
HARD_THRESHOLD_U8 = 128  # round(0.5 * 255), so u8 >= 128 is p >= 0.5
CONFIDENCE = (2.5, 97.5)
INTERVAL_METRICS = ("picp", "mpiw", "interval_score")
POINT_METRICS = ("mae", "bias", "relative_error", "dice", "boundary_f1")


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
    root: Path, manifest: pd.DataFrame, prediction_dir: Path, tolerance_px: float
) -> pd.DataFrame:
    """Per-cell Dice and boundary F1 on test cells with mining, from the persisted probabilities."""
    cells = pd.read_csv(prediction_dir / "cells.csv")
    test = cells[cells.role == "test"].reset_index(drop=True)
    probabilities = np.load(prediction_dir / "test.npy", mmap_mode="r")
    paths = manifest.set_index("cell_id").path
    rows = []
    for index, cell_id in enumerate(test.cell_id):
        reference = garimpo_uncertainty.data.read_reference_mask(root, paths[cell_id])
        prediction = probabilities[index] >= HARD_THRESHOLD_U8
        has_mining = reference.any()
        rows.append(
            {
                "cell_id": cell_id,
                "dice": dice(prediction, reference) if has_mining else float("nan"),
                "boundary_f1": (
                    boundary_f1(prediction, reference, tolerance_px) if has_mining else float("nan")
                ),
                "boundary_by_rule": bool(
                    has_mining and not (_boundary(prediction).any() and _boundary(reference).any())
                ),
            }
        )
    return pd.DataFrame(rows)


def add_cell_metrics(intervals: pd.DataFrame, alpha: float) -> pd.DataFrame:
    frame = intervals.copy()
    frame["width"] = frame.upper - frame.lower
    frame["covered"] = (frame.v >= frame.lower) & (frame.v <= frame.upper)
    penalty = (2 / alpha) * (
        (frame.lower - frame.v).clip(lower=0) + (frame.v - frame.upper).clip(lower=0)
    )
    frame["interval_score"] = frame.width + penalty
    frame["error"] = frame.vhat - frame.v
    frame["abs_error"] = frame.error.abs()
    frame["relative_abs_error"] = frame.abs_error / frame.v.where(frame.v > 0)
    return frame


def _group_means(frame: pd.DataFrame, column: str, groups: np.ndarray) -> np.ndarray:
    return frame.groupby("group")[column].mean().reindex(groups).to_numpy(dtype=float)


def _paired_differences(values: dict, indices: np.ndarray, model: str) -> pd.DataFrame:
    """Differences between rows on the same group resamples, so the comparison stays paired."""
    rows = []
    for (key_a, a), (key_b, b) in combinations(values.items(), 2):
        low, high = np.percentile((a - b)[indices].mean(axis=1), CONFIDENCE)
        rows.append(
            {
                "model": model,
                "method_a": key_a[0],
                "calibration_a": key_a[1],
                "method_b": key_b[0],
                "calibration_b": key_b[1],
                "mean_difference": float((a - b).mean()),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return pd.DataFrame(rows)


def _spearman(pairs: np.ndarray) -> float:
    kept = pairs[~np.isnan(pairs).any(axis=1)]
    if len(kept) < 2:
        return float("nan")
    return float(scipy.stats.spearmanr(kept[:, 0], kept[:, 1]).statistic)


def spearman_by_model(
    intervals: pd.DataFrame, segmentation: pd.DataFrame, n_bootstrap: int, seed: int
) -> pd.DataFrame:
    """Q1: rank correlation of area error with Dice and boundary F1, on cells with mining.

    Point estimates, so one interval row per cell is enough. Intervals resample whole groups.
    """
    cells = intervals.drop_duplicates("cell_id")[["model", "cell_id", "group", "v", "vhat"]]
    cells = cells.merge(segmentation, on="cell_id")
    cells = cells[cells.v > 0].reset_index(drop=True)
    cells["abs_error_ha"] = (cells.vhat - cells.v).abs()
    cells["abs_error_percent"] = 100 * cells.abs_error_ha / cells.v
    groups = np.sort(cells.group.unique())
    positions = [np.flatnonzero(cells.group == group) for group in groups]
    indices = np.random.default_rng(seed).integers(0, len(groups), (n_bootstrap, len(groups)))
    rows = []
    with warnings.catch_warnings():
        # A constant column, such as a resample where Dice never varies, gives NaN.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for error in ("abs_error_ha", "abs_error_percent"):
            for quality in ("dice", "boundary_f1"):
                pair = cells[[error, quality]].to_numpy()
                resampled = [
                    _spearman(pair[np.concatenate([positions[i] for i in chosen])])
                    for chosen in indices
                ]
                low, high = np.nanpercentile(resampled, CONFIDENCE)
                rows.append(
                    {
                        "model": cells.model.iloc[0],
                        "area_error": error,
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
    intervals: pd.DataFrame,
    segmentation: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One row per model x method x calibration, plus paired width and coverage differences.

    Q3 reads the coverage differences of a method between its calibration situations.
    """
    alpha = float(intervals.alpha.iloc[0])
    frame = add_cell_metrics(intervals, alpha).merge(segmentation, on="cell_id", how="left")
    groups = np.sort(frame.group.unique())
    indices = np.random.default_rng(seed).integers(0, len(groups), (n_bootstrap, len(groups)))
    columns = {
        "picp": "covered",
        "mpiw": "width",
        "interval_score": "interval_score",
        "mae": "abs_error",
        "bias": "error",
        "relative_error": "relative_abs_error",
        "dice": "dice",
        "boundary_f1": "boundary_f1",
    }
    rows, widths, coverages = [], {}, {}
    with warnings.catch_warnings():
        # Groups with no mining cell give NaN in the segmentation metrics; nanmean skips them.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for (method, calibration), one in frame.groupby(["method", "calibration"], sort=False):
            row = {
                "model": one.model.iloc[0],
                "layer": one.layer.iloc[0],
                "method": method,
                "calibration": calibration,
                "alpha": alpha,
                "n_cells": len(one),
                "n_groups": len(groups),
                "n_cells_boundary_by_rule": int(one.boundary_by_rule.sum()),
            }
            for name, column in columns.items():
                means = _group_means(one, column, groups)
                resampled = np.nanmean(means[indices], axis=1)
                low, high = np.nanpercentile(resampled, CONFIDENCE)
                row |= {
                    name: float(np.nanmean(means)),
                    f"{name}_ci_low": float(low),
                    f"{name}_ci_high": float(high),
                }
            rows.append(row)
            widths[(method, calibration)] = _group_means(one, "width", groups)
            coverages[(method, calibration)] = _group_means(one, "covered", groups)

    saturation = calibration_summary.drop(columns="alpha")
    results = pd.DataFrame(rows).merge(saturation, on=["method", "calibration"], how="left")
    model = frame.model.iloc[0]
    return (
        results,
        _paired_differences(widths, indices, model),
        _paired_differences(coverages, indices, model),
    )
