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

# The directions compass2d registers, in its order; written out, as importing compass2d loads torch,
# which compute_metrics must not; tests/test_training_parts.py holds the two equal.
DIRECTIONS = ("pca", "svm")
JACOBIAN_METHODS = tuple(f"compass_j_{direction}" for direction in DIRECTIONS)
COMPASS_METHODS = ("compass_l", *JACOBIAN_METHODS)  # the rows of scores/scores.npz
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
    *(
        (method, situation)
        for method in COMPASS_METHODS
        for situation in ("external", "external_weighted", "own_region")
    ),
)
# Mondrian calibration by predicted mining, the stratum known when an interval is used: one quantile
# for cells predicted empty, one for the rest. Each is the pooled group quantile, without classic
# Mondrian CP's finite-sample correction, so a small stratum can undercover (Dunn et al., 2022).
MONDRIAN_METHODS = (
    "conformal_simple",
    "conformal_local",
    "cqr",
    *COMPASS_METHODS,
)
COMBINATIONS += tuple(
    (method, f"{situation}_mondrian")
    for method in MONDRIAN_METHODS
    for situation in ("external", "own_region")
)
# Asymmetric COMPASS (Cheung et al., 2026, Algorithm 6) through one joint score: each a_ row reads
# the one-sided scores of the COMPASS method named after the prefix.
ASYMMETRIC_METHODS = tuple(f"a_{method}" for method in COMPASS_METHODS)
COMBINATIONS += tuple(
    (method, situation) for method in ASYMMETRIC_METHODS for situation in ("external", "own_region")
)
# Sensitivity analysis: the pooled external and own-region rows again, by repeated subsampling.
# The weighted quantile has no subsampled form, so only these two calibrations get one.
SUBSAMPLED_CALIBRATIONS = ("external", "own_region")
COMBINATIONS += tuple(
    (method, f"{calibration}_subsampled")
    for method, calibration in COMBINATIONS
    if calibration in SUBSAMPLED_CALIBRATIONS
)
SUBSAMPLES = 100  # B in Dunn et al. (2022), Section 4.4
DENSITY_RATIO_FOLDS = 5  # group folds of the cross-fitted domain classifier
ALPHAS = (0.05, 0.1, 0.15)  # Cheung et al. (2026), Table 1
LAYER_METHODS = (*JACOBIAN_METHODS, *(f"a_{method}" for method in JACOBIAN_METHODS))


def depends_on_layer(results: pd.DataFrame) -> pd.Series:
    """Rows whose intervals change with the COMPASS-J layer (Cheung et al., 2026, Table 4).

    The weighted calibrations count too: their density ratio reads the layer's feature sums.
    """
    return results.method.isin(LAYER_METHODS) | results.calibration.str.contains("weighted")


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


def mondrian_stratum(cells: pd.DataFrame) -> np.ndarray:
    """1 where the model predicts mining, 0 elsewhere."""
    return (cells.vhat.to_numpy() > 0).astype(int)


def uncertainty_ha(cells: pd.DataFrame) -> np.ndarray:
    """Sum of p(1 - p) in hectares; it concentrates where the model hesitates, the borders."""
    return cells.pixel_var.to_numpy() * garimpo_uncertainty.data.PIXEL_HA


def independent_pixel_terms(cells: pd.DataFrame, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Control: Poisson-binomial mean and normal approximation, treating pixels as independent."""
    z = statistics.NormalDist().inv_cdf(1 - alpha / 2)
    half_width = z * garimpo_uncertainty.data.PIXEL_HA * np.sqrt(cells.pixel_var.to_numpy())
    center = cells.p_sum.to_numpy()
    return center - half_width, center + half_width


Terms = tuple[np.ndarray, np.ndarray, np.ndarray]


def simple_terms(cells: pd.DataFrame, _floor: float, _cqr_models: tuple) -> Terms:
    vhat = cells.vhat.to_numpy()
    return vhat, vhat, np.ones_like(vhat)


def relative_terms(cells: pd.DataFrame, floor: float, _cqr_models: tuple) -> Terms:
    vhat = cells.vhat.to_numpy()
    return vhat, vhat, np.maximum(vhat, floor)


def local_terms(cells: pd.DataFrame, floor: float, _cqr_models: tuple) -> Terms:
    vhat = cells.vhat.to_numpy()
    return vhat, vhat, np.maximum(uncertainty_ha(cells), floor)


def cqr_terms(cells: pd.DataFrame, _floor: float, cqr_models: tuple) -> Terms:
    lower = cqr_models[0].predict(cells[["vhat"]].to_numpy())
    upper = cqr_models[1].predict(cells[["vhat"]].to_numpy())
    return lower, upper, np.ones_like(cells.vhat.to_numpy())


OUTPUT_SPACE_TERMS = {
    "conformal_simple": simple_terms,
    "conformal_relative": relative_terms,
    "conformal_local": local_terms,
    "cqr": cqr_terms,
}


def method_terms(
    method: str, cells: pd.DataFrame, score_floor_ha: float, cqr_models: tuple
) -> Terms:
    """Return the lower term, the upper term and the scale of an output-space method."""
    if method not in OUTPUT_SPACE_TERMS:
        raise ValueError(f"not an output-space method: {method!r}")
    return OUTPUT_SPACE_TERMS[method](cells, score_floor_ha, cqr_models)


def scores(v: np.ndarray, lower: np.ndarray, upper: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.maximum(lower - v, v - upper) / scale


def clip_to_cell(
    lower: np.ndarray, upper: np.ndarray, cell_ha: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Area lives in [0, cell]; a negative CQR margin may cross, which collapses to a point."""
    lower = np.clip(lower, 0, cell_ha)
    upper = np.clip(upper, 0, cell_ha)
    return lower, np.maximum(lower, upper)
