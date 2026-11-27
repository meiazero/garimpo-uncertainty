"""Which calibration names calibrate accepts, apart from the E2E runs that compute them."""

import numpy as np
import pytest

import garimpo_uncertainty.stages.calibrate


@pytest.mark.parametrize("calibration", ["external_weighted_subsampled", "nearby"])
def test_a_calibration_with_no_quantile_rule_is_refused(calibration: str) -> None:
    scores, groups = np.ones(4), np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match=f"unknown calibration: '{calibration}'"):
        garimpo_uncertainty.stages.calibrate.quantile_per_cell(
            calibration,
            scores,
            groups,
            scores,
            groups,
            calibration_ratio=np.ones(4),
            test_ratio=np.ones(4),
            alpha=0.1,
            seed=0,
        )
