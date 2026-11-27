"""The tuning's verdict and its run of sections, on rows written by hand; the timings need a GPU."""

from pathlib import Path

import pytest

import garimpo_uncertainty.tuning.tune
import garimpo_uncertainty.tuning.verdict


def train_row(workers: int, wait: float, *, stable: bool = True) -> dict:
    return {
        "workers": workers,
        "seconds": {"median": 1.0, "low": 0.9, "high": 1.1},
        "seconds_per_update": 0.4,
        "data_wait_share": wait,
        "gpu_busy": 0.9,
        "stable": stable,
        "weights_match_0_workers": True,
    }


def test_the_fewest_workers_that_keep_training_fed_size_the_run() -> None:
    verdict = garimpo_uncertainty.tuning.verdict.Verdict([], updates=9000, evaluated_cells=16)
    verdict.train("unet", [train_row(0, 0.3), train_row(2, 0.05), train_row(4, 0.01)])
    assert verdict.failures == []
    assert verdict.findings[0].endswith("limited by data loading, waiting 30% of the time")
    assert verdict.findings[-1] == "train unet: 2 workers keep it fed; 9000 updates take ~1.0 h"


def test_lost_bits_and_starved_loops_fail_the_tuning() -> None:
    verdict = garimpo_uncertainty.tuning.verdict.Verdict([], updates=1, evaluated_cells=1)
    verdict.train("unet", [train_row(0, 0.5, stable=False), train_row(2, 0.4)])
    verdict.loader(
        [
            {"reads_as": "stages", "workers": 0, "bitwise": True, "cells_per_second": 10.0},
            {"reads_as": "train", "workers": 2, "bitwise": False, "cells_per_second": 20.0},
        ]
    )
    assert verdict.failures == [
        "train unet: two runs at [0] workers end with other weights under deterministic algorithms",
        "train unet: data loading starves the loop at every worker count",
        "loader: train at 2 workers read other bytes",
    ]


def test_a_section_that_fails_is_noted_and_the_rest_still_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def no_gpu(form: dict, out: Path) -> dict:
        raise SystemExit("No GPU")

    sections = {"hardware": no_gpu, "lightgbm": lambda form, out: {"lightgbm": {"rows": []}}}
    monkeypatch.setattr(garimpo_uncertainty.tuning.sections, "SECTIONS", sections)
    results, failed = garimpo_uncertainty.tuning.tune.run_sections({}, tmp_path)
    assert results == {"lightgbm": {"rows": []}}
    assert failed == ["hardware: SystemExit: No GPU"]
    verdict = garimpo_uncertainty.tuning.verdict.judge(results, failed, ["unet"], 1, 1)
    assert "hardware: SystemExit: No GPU" in verdict.failures
    assert "train-unet: no result, the section crashed or never ran" in verdict.failures
