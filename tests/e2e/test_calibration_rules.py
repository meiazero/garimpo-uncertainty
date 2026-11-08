"""The two group-level quantile rules on synthetic scores, with no pipeline around them."""

import json

import compass2d.calibration
import numpy as np
import pytest

import garimpo_uncertainty.intervals
from tests.e2e import support

ARTIFACTS = support.ROOT / "tests" / "artifacts" / "calibration_rules"
ALPHA = 0.1
SUBSAMPLES = garimpo_uncertainty.intervals.SUBSAMPLES


def subsampled(scores: np.ndarray, groups: np.ndarray, seed: int = 0) -> float:
    return compass2d.calibration.subsampled_quantile(scores, groups, ALPHA, SUBSAMPLES, seed)


def dunn_subsampled(drawn: np.ndarray, alpha: float) -> float:
    """Brute-force Dunn et al. (2022), §4.3, on subsamples already drawn, one row each."""
    k = drawn.shape[1]
    accepted = [
        s
        for s in np.unique(drawn)
        if np.mean([(1 + np.sum(row >= s)) / (k + 1) for row in drawn]) >= alpha - 1e-12
    ]
    return float(max(accepted))


def test_one_cell_per_group_is_split_conformal_with_its_correction() -> None:
    """Every subsample is then the whole sample, so the rule must reduce to split conformal."""
    scores = np.random.default_rng(0).permutation(np.arange(1.0, 26.0))
    groups = np.arange(25)
    assert subsampled(scores, groups) == dunn_subsampled(scores[None, :], ALPHA)
    # (k + 1)(1 - alpha) = 23.4 gives the 24th smallest; pooling, with no +1, stops at 0.9 x 25.
    assert subsampled(scores, groups) == 24.0
    assert compass2d.calibration.group_quantile(scores, groups, ALPHA) == 23.0


def test_draws_one_cell_per_group_not_per_cell() -> None:
    """30 cells at 1000 in one group: drawing by cell would put about 14 of them in each sample."""
    scores = np.concatenate([np.full(30, 1000.0), np.ones(24)])
    groups = np.concatenate([np.zeros(30, dtype=int), np.arange(1, 25)])
    assert subsampled(scores, groups) == 1.0


def test_too_few_groups_refuse_instead_of_an_unbounded_interval() -> None:
    """At k <= 1/alpha - 1 every score above the sample is accepted, so the interval has no end."""
    with pytest.raises(ValueError, match="groups"):
        subsampled(np.arange(9.0), np.arange(9))
    assert subsampled(np.arange(10.0), np.arange(10)) == 9.0


def test_every_cell_of_a_group_can_be_drawn() -> None:
    """Each group holds a 1 then a 10: drawing only a group's first cell would give 1."""
    scores = np.tile([1.0, 10.0], 25)
    assert subsampled(scores, np.repeat(np.arange(25), 2)) == 10.0


def test_the_quantile_is_a_score_not_an_average_of_quantiles() -> None:
    rng = np.random.default_rng(2)
    scores = rng.exponential(size=500)
    assert subsampled(scores, np.repeat(np.arange(25), 20)) in scores


def test_the_seed_alone_fixes_the_draws() -> None:
    """Seeds 0 and 1 give different quantiles on these scores, so a fixed seed would show."""
    scores = np.random.default_rng(1).exponential(size=500)
    groups = np.repeat(np.arange(25), 20)
    np.random.seed(123)  # noqa: NPY002 - the legacy global state must not reach the draws
    state = np.random.get_state()  # noqa: NPY002
    first = subsampled(scores, groups, seed=0)
    assert np.array_equal(np.random.get_state()[1], state[1])  # noqa: NPY002
    np.random.seed(456)  # noqa: NPY002
    assert subsampled(scores, groups, seed=0) == first
    assert subsampled(scores, groups, seed=1) != first


def test_coverage_of_a_new_group() -> None:
    """Dunn et al.'s balanced simulation at this experiment's k: group means drawn from N(0, 1).

    Only subsampling's 1 - 2 alpha is guaranteed; both coverages go to the artifact for reading.
    """
    rng = np.random.default_rng(0)
    k, n, repetitions = 25, 20, 2000
    covered = {"pooled": 0, "subsampled": 0}
    for repetition in range(repetitions):
        means = rng.normal(size=k + 1)
        scores = np.abs(means[:k, None] + rng.normal(size=(k, n))).ravel()
        groups = np.repeat(np.arange(k), n)
        new = abs(means[k] + rng.normal())
        pooled = compass2d.calibration.group_quantile(scores, groups, ALPHA)
        covered["pooled"] += new <= pooled
        covered["subsampled"] += new <= subsampled(scores, groups, seed=repetition)
    coverage = {rule: count / repetitions for rule, count in covered.items()}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "coverage_k25.json").write_text(
        json.dumps({"k": k, "n_per_group": n, "repetitions": repetitions, **coverage}, indent=2)
    )
    standard_error = np.sqrt(0.8 * 0.2 / repetitions)
    assert coverage["subsampled"] >= 1 - 2 * ALPHA - 3 * standard_error, coverage
