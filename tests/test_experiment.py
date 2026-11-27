"""Which scales `garimpo experiment` runs, and which runs `garimpo figures --scale` draws."""

import types
from pathlib import Path

import pytest

import garimpo_uncertainty.figures.make
import garimpo_uncertainty.pipeline.experiment
import garimpo_uncertainty.pipeline.status

experiment = garimpo_uncertainty.pipeline.experiment


@pytest.mark.parametrize(
    ("passes", "interrupted", "expected_runs", "ran"),
    [
        # The pilot passes on this code, so the full runs follow, forced from calibrate.
        (["pilot", "full"], False, [("pilot", []), ("full", ["--force-from", "calibrate"])], 2),
        # A pilot that does not pass keeps the full runs waiting.
        (["pilot", "pilot"], False, [("pilot", [])], 1),
        # An interrupted pilot stops everything after it.
        (["pilot"], True, [("pilot", [])], 1),
        # A pilot already passed: only the full runs.
        (["full"], False, [("full", ["--force-from", "calibrate"])], 1),
    ],
)
def test_the_full_runs_start_only_once_the_pilot_passed_on_this_code(
    monkeypatch: pytest.MonkeyPatch,
    passes: list[str],
    interrupted: bool,
    expected_runs: list,
    ran: int,
) -> None:
    answers, runs = iter(passes), []
    monkeypatch.setattr(garimpo_uncertainty.pipeline.status, "next_scale", lambda: next(answers))

    def run_scale(scale: str, options: list[str]) -> bool:
        runs.append((scale, options))
        return not interrupted

    monkeypatch.setattr(experiment, "run_scale", run_scale)
    assert len(experiment.run_scales("calibrate")) == ran
    assert runs == expected_runs


def test_figures_at_a_scale_draw_only_the_runs_with_intervals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    done = tmp_path / "unet_scratch-f0.01"
    (done / "intervals").mkdir(parents=True)
    (done / "intervals" / "intervals.npz").touch()
    configs = {
        "unet_scratch": types.SimpleNamespace(run_dir=str(done)),
        "convnet": types.SimpleNamespace(run_dir=str(tmp_path / "convnet-f0.01")),
    }
    monkeypatch.setattr(
        garimpo_uncertainty.pipeline.status, "load", lambda scale, model: configs[model]
    )
    found = garimpo_uncertainty.figures.make.runs_at("pilot", ["unet_scratch", "convnet"])
    assert found == [done]
    assert "Left out convnet" in capsys.readouterr().out
