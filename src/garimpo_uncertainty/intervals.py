"""Output-space interval methods, sharing one normalized score.

Every method is described by a lower term, an upper term and a scale; the score is
max(lower - v, v - upper) / scale and the interval is [lower - q scale, upper + q scale]. Simple,
relative and local take lower = upper = estimated area.
"""

import statistics

import numpy as np
import pandas as pd
import sklearn.ensemble

import garimpo_uncertainty.data

COMBINATIONS = (  # tab:combinacoes, plus the SVM direction with the same calibrations as PCA
    ("independent_pixels", "none"),
    ("conformal_simple", "external"),
    ("conformal_simple", "external_weighted"),
    ("conformal_simple", "own_region"),
    ("conformal_relative", "external"),
    ("conformal_relative", "own_region"),
    ("conformal_local", "external"),
    ("conformal_local", "own_region"),
    ("cqr", "external"),
    ("cqr", "own_region"),
    ("compass_l", "external"),
    ("compass_l", "external_weighted"),
    ("compass_l", "own_region"),
    ("compass_j_pca", "external"),
    ("compass_j_pca", "own_region"),
    ("compass_j_svm", "external"),
    ("compass_j_svm", "own_region"),
)
# Sensitivity analysis: the pooled external and own-region rows again, by repeated subsampling.
COMBINATIONS += tuple(
    (method, f"{calibration}_subsampled")
    for method, calibration in COMBINATIONS
    if calibration in {"external", "own_region"}
)
SUBSAMPLES = 100  # B in Dunn et al. (2022), Section 4.4
COMPASS_METHODS = ("compass_l", "compass_j_pca", "compass_j_svm")  # the rows of scores/scores.npz
# Upstream's values, also in Cheung et al. (2026), Appendix E.
CQR_PARAMETERS = {
    "learning_rate": 0.1,
    "n_estimators": 50,
    "max_depth": 3,
    "min_samples_leaf": 1,
    "min_samples_split": 9,
}


def fit_cqr(
    train: pd.DataFrame, alpha: float, seed: int
) -> tuple[sklearn.ensemble.GradientBoostingRegressor, sklearn.ensemble.GradientBoostingRegressor]:
    """Quantile regressors of the reference area on the estimate, as upstream's output-space CQR."""
    models = [
        sklearn.ensemble.GradientBoostingRegressor(
            loss="quantile", alpha=level, random_state=seed, **CQR_PARAMETERS
        ).fit(train[["vhat"]].to_numpy(), train.v.to_numpy())
        for level in (alpha / 2, 1 - alpha / 2)
    ]
    return models[0], models[1]


def uncertainty_ha(cells: pd.DataFrame) -> np.ndarray:
    """Sum of p(1 - p) in hectares; it concentrates where the model hesitates, the borders."""
    return cells.pixel_var.to_numpy() * garimpo_uncertainty.data.PIXEL_HA


def independent_pixel_terms(cells: pd.DataFrame, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Control: Poisson-binomial mean and normal approximation, treating pixels as independent."""
    z = statistics.NormalDist().inv_cdf(1 - alpha / 2)
    half_width = z * garimpo_uncertainty.data.PIXEL_HA * np.sqrt(cells.pixel_var.to_numpy())
    center = cells.p_sum.to_numpy()
    return center - half_width, center + half_width


def method_terms(
    method: str, cells: pd.DataFrame, score_floor_ha: float, cqr_models: tuple
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vhat = cells.vhat.to_numpy()
    if method == "conformal_simple":
        return vhat, vhat, np.ones_like(vhat)
    if method == "conformal_relative":
        return vhat, vhat, np.maximum(vhat, score_floor_ha)
    if method == "conformal_local":
        return vhat, vhat, np.maximum(uncertainty_ha(cells), score_floor_ha)
    if method == "cqr":
        lower = cqr_models[0].predict(cells[["vhat"]].to_numpy())
        upper = cqr_models[1].predict(cells[["vhat"]].to_numpy())
        return lower, upper, np.ones_like(vhat)
    raise ValueError(f"not an output-space method: {method!r}")


def scores(v: np.ndarray, lower: np.ndarray, upper: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.maximum(lower - v, v - upper) / scale


def clip_to_cell(
    lower: np.ndarray, upper: np.ndarray, cell_ha: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Area lives in [0, cell]; a negative CQR margin may cross, which collapses to a point."""
    lower = np.clip(lower, 0, cell_ha)
    upper = np.clip(upper, 0, cell_ha)
    return lower, np.maximum(lower, upper)
