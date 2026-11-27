"""Which scales `garimpo experiment` runs, and which runs `garimpo figures --scale` draws."""

import os
import signal
import sys
import threading
import time
import types
from pathlib import Path

import pytest

import garimpo_uncertainty.figures.make
import garimpo_uncertainty.pipeline.experiment
import garimpo_uncertainty.pipeline.status

experiment = garimpo_uncertainty.pipeline.experiment


@pytest.mark.parametrize(
    ("passes", "codes", "expected_runs", "result"),
    [
        # The pilot passes on this code, so the full runs follow, forced from calibrate.
        (["pilot", "full"], [0, 0], [("pilot", []), ("full", ["--force-from", "calibrate"])], 0),
        # A pilot that does not pass keeps the full runs waiting, and fails the command.
        (["pilot", "pilot"], [0], [("pilot", [])], 1),
        # An interrupted pilot stops everything after it.
        (["pilot"], [130], [("pilot", [])], 130),
        # A pilot already passed: only the full runs, whose failure is the command's.
        (["full"], [1], [("full", ["--force-from", "calibrate"])], 1),
    ],
)
def test_the_full_runs_start_only_once_the_pilot_passed_on_this_code(
    monkeypatch: pytest.MonkeyPatch,
    passes: list[str],
    codes: list[int],
    expected_runs: list,
    result: int,
) -> None:
    answers, ends, runs = iter(passes), iter(codes), []
    monkeypatch.setattr(garimpo_uncertainty.pipeline.status, "next_scale", lambda: next(answers))

    def run_scale(scale: str, options: list[str]) -> int:
        runs.append((scale, options))
        return next(ends)

    monkeypatch.setattr(experiment, "run_scale", run_scale)
    ran, code = experiment.run_scales("calibrate")
    assert runs == expected_runs
    assert ran == [scale for scale, _ in expected_runs]
    assert code == result


def test_an_interrupt_reaches_the_job_which_ends_on_its_own(tmp_path: Path) -> None:
    """Colab interrupts this process alone; the job must get the SIGINT and stop its own way."""
    ready = tmp_path / "ready"
    job = (
        "import pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGINT, lambda *_: sys.exit(130))\n"
        f"pathlib.Path({str(ready)!r}).touch()\n"
        "time.sleep(60)\n"
    )

    def interrupt_once_ready() -> None:
        while not ready.exists():
            time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=interrupt_once_ready, daemon=True).start()
    assert experiment.run_model([sys.executable, "-c", job]) == 130


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
