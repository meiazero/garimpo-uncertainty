"""The tuning's verdict and its run of sections, on rows written by hand; the timings need a GPU."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

import garimpo_uncertainty.tuning.sections
import garimpo_uncertainty.tuning.tune
import garimpo_uncertainty.tuning.verdict
import garimpo_uncertainty.tuning.workloads


def train_row(workers: int, wait: float, *, stable: bool = True, matches: bool = True) -> dict:
    return {
        "workers": workers,
        "seconds": {"median": 1.0, "low": 0.9, "high": 1.1},
        "seconds_per_update": 0.4,
        "data_wait_share": wait,
        "gpu_busy": 0.9,
        "stable": stable,
        "weights_match_0_workers": matches,
    }


def test_the_fewest_workers_that_keep_training_fed_size_the_run() -> None:
    verdict = garimpo_uncertainty.tuning.verdict.Verdict([], updates=9000, evaluated_cells=16)
    verdict.train("unet", [train_row(0, 0.3), train_row(2, 0.05), train_row(4, 0.01)])
    assert verdict.failures == []
    assert verdict.findings[0].endswith("limited by data loading, waiting 30% of the time")
    assert verdict.findings[-1] == "train unet: 2 workers keep it fed; 9000 updates take ~1.0 h"


def test_results_json_holds_no_nan_for_a_gpu_use_never_sampled() -> None:
    results = {"train-unet": {"rows": [train_row(0, 0.0) | {"gpu_busy": float("nan")}]}}
    written = json.dumps(garimpo_uncertainty.tuning.tune.json_safe(results), allow_nan=False)
    assert json.loads(written)["train-unet"]["rows"][0]["gpu_busy"] is None


def test_configs_that_cannot_be_read_fail_the_tuning_with_a_verdict(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def unreadable() -> tuple[int, int]:
        raise TypeError("unsupported operand type(s) for *: 'NoneType' and 'int'")

    monkeypatch.setattr(garimpo_uncertainty.tuning.tune, "full_run_size", unreadable)
    monkeypatch.setattr("sys.argv", ["garimpo tune"])
    with pytest.raises(SystemExit) as exited:
        garimpo_uncertainty.tuning.tune.main()
    assert exited.value.code == 1
    assert "TUNING: FAIL: the configs cannot be read: TypeError" in capsys.readouterr().out


@pytest.mark.parametrize("epochs", ["100", None, 0, True])
def test_the_run_size_needs_positive_integers(
    monkeypatch: pytest.MonkeyPatch, epochs: object
) -> None:
    full = {"EPOCHS": epochs, "STEPS_PER_EPOCH": 10}
    monkeypatch.setattr(garimpo_uncertainty.tuning.workloads, "read_json", lambda path: full)
    with pytest.raises(ValueError, match="must be positive integers"):
        garimpo_uncertainty.tuning.tune.full_run_size()


def test_lost_bits_and_starved_loops_fail_the_tuning() -> None:
    verdict = garimpo_uncertainty.tuning.verdict.Verdict([], updates=1, evaluated_cells=1)
    verdict.train("unet", [train_row(0, 0.5), train_row(2, 0.4, stable=False)])
    verdict.loader(
        [
            {
                "reads_as": "stages",
                "workers": 0,
                "bitwise": True,
                "stable": True,
                "cells_per_second": 10.0,
            },
            {
                "reads_as": "train",
                "workers": 2,
                "bitwise": False,
                "stable": True,
                "cells_per_second": 20.0,
            },
        ]
    )
    assert verdict.failures == [
        "train unet: two runs at [2] workers end with other weights under deterministic algorithms",
        "train unet: data loading starves the loop at every worker count that keeps the weights",
        "loader: train at 2 workers read other bytes",
    ]


@pytest.mark.parametrize(
    ("rows", "sized_by"),
    [
        ([train_row(0, 0.3), train_row(2, 0.0, stable=False), train_row(4, 0.0)], 4),
        # Two workers match 0 workers' weights, but those changed between two runs.
        ([train_row(0, 0.0, stable=False), train_row(2, 0.0)], None),
    ],
)
def test_an_unstable_worker_count_never_sizes_the_run(rows: list[dict], sized_by: int) -> None:
    verdict = garimpo_uncertainty.tuning.verdict.Verdict([], updates=3600, evaluated_cells=1)
    verdict.train("unet", rows)
    sized = [line for line in verdict.findings if "keep it fed" in line]
    assert sized == (
        []
        if sized_by is None
        else [f"train unet: {sized_by} workers keep it fed; 3600 updates take ~0.4 h"]
    )


def test_workers_that_move_the_weights_fail_the_tuning() -> None:
    verdict = garimpo_uncertainty.tuning.verdict.Verdict([], updates=1, evaluated_cells=1)
    verdict.train("unet", [train_row(0, 0.0), train_row(2, 0.0, matches=False)])
    assert verdict.failures == ["train unet: [2] workers end with other weights than 0 workers"]


def test_a_lost_reference_or_an_unstable_batch_fails_the_tuning() -> None:
    verdict = garimpo_uncertainty.tuning.verdict.Verdict([], updates=1, evaluated_cells=1)
    verdict.precision(
        "unet", [{"layer": "l", "candidate": "strict_fp32", "failed": "out of memory"}]
    )
    seconds = {"median": 1.0, "low": 0.9, "high": 1.1}
    fast = {"median": 0.5, "low": 0.4, "high": 0.6}
    batch = {"batch_size": 16, "seconds": seconds, "stable": True, "bitwise": True}
    batch |= {"layer": "l", "gpu_busy": 0.5, "gpu_memory_gb": 1.0}
    unstable = batch | {"batch_size": 32, "seconds": fast, "stable": False}
    verdict.batch("unet", {"reference": 16, "rows": [batch, unstable]}, memory=24)
    assert verdict.failures == [
        "precision unet l: strict float32 failed",
        "batch unet l: batch 32 changed its bits between two runs",
    ]
    assert verdict.findings[-1].endswith("no larger batch is worth its memory")


def test_an_unstable_candidate_or_a_failed_worker_count_fails_the_tuning() -> None:
    verdict = garimpo_uncertainty.tuning.verdict.Verdict([], updates=1, evaluated_cells=1)
    seconds = {"median": 1.0, "low": 0.9, "high": 1.1}
    strict = {"layer": "l", "candidate": "strict_fp32", "seconds": seconds, "stable": True}
    strict |= {"bitwise_with_strict_fp32": True}
    tuned = strict | {"candidate": "strict_fp32_autotuned", "stable": False}
    verdict.precision("unet", [strict, tuned])
    verdict.train("unet", [train_row(0, 0.0), {"workers": 2, "failed": "out of memory"}])
    assert verdict.failures == [
        "precision unet l strict_fp32_autotuned: other bits twice",
        "train unet: 2 workers failed, so their weights went unchecked",
    ]


def test_a_loader_or_lightgbm_row_that_differs_between_two_runs_fails_the_tuning() -> None:
    verdict = garimpo_uncertainty.tuning.verdict.Verdict([], updates=1, evaluated_cells=1)
    read = {"reads_as": "stages", "workers": 0, "bitwise": True, "stable": False}
    verdict.loader([read | {"cells_per_second": 10.0}])
    seconds = {"median": 1.0, "low": 0.9, "high": 1.1}
    ratio = {"channels": 16, "jobs": 1, "bitwise": True, "stable": False, "seconds": seconds}
    verdict.lightgbm([ratio])
    assert verdict.failures == [
        "loader: stages at 0 workers read other bytes between two runs",
        "lightgbm 16 channels: 1 threads gave other ratios between two runs",
    ]
    assert not [line for line in verdict.findings if line.startswith("lightgbm")]


def test_an_architecture_done_before_another_fails_keeps_its_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def search(form: dict, out: Path) -> Iterator[dict]:
        yield {"search-unet": {"rows": []}}
        raise RuntimeError("convnet ran out of memory")

    monkeypatch.setattr(garimpo_uncertainty.tuning.sections, "SECTIONS", {"search": search})
    results, failed = garimpo_uncertainty.tuning.tune.run_sections({}, tmp_path)
    assert results == {"search-unet": {"rows": []}}
    assert failed == ["search: RuntimeError: convnet ran out of memory"]


def test_a_section_that_fails_is_noted_and_the_rest_still_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def no_gpu(form: dict, out: Path) -> Iterator[dict]:
        raise SystemExit("No GPU")
        yield {}

    def lightgbm(form: dict, out: Path) -> Iterator[dict]:
        yield {"lightgbm": {"rows": []}}

    sections = {"hardware": no_gpu, "lightgbm": lightgbm}
    monkeypatch.setattr(garimpo_uncertainty.tuning.sections, "SECTIONS", sections)
    results, failed = garimpo_uncertainty.tuning.tune.run_sections({}, tmp_path)
    assert results == {"lightgbm": {"rows": []}}
    assert failed == ["hardware: SystemExit: No GPU"]
    verdict = garimpo_uncertainty.tuning.verdict.judge(results, failed, ["unet"], 1, 1)
    assert "hardware: SystemExit: No GPU" in verdict.failures
    assert "train-unet: no result, the section crashed or never ran" in verdict.failures
