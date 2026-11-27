"""Which scales `garimpo experiment` runs, and which runs `garimpo figures --scale` draws."""

import os
import signal
import subprocess
import sys
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


def test_the_models_run_in_main_s_order_with_a_new_one_last() -> None:
    models = ["convnet", "new", "unet_imagenet", "unet_scratch", "unet_sentinel2"]
    assert experiment.run_order(models) == [*experiment.ORDER, "new"]


@pytest.mark.parametrize("ends_with", [130, 0])
@pytest.mark.parametrize("whole_group", [False, True])
def test_an_interrupt_reaches_the_job_once(
    tmp_path: Path, whole_group: bool, ends_with: int
) -> None:
    """Colab interrupts garimpo experiment alone, a terminal its whole group; the job sees one.

    The job counts the SIGINTs it got, as a second would kill a stage mid-stop, and ends
    `ends_with`: a job that finished 0 just then still stops the models after it.
    """
    ready, seen = tmp_path / "ready", tmp_path / "seen"
    job = (
        "import pathlib, signal, sys, time\n"
        "seen = []\n"
        "signal.signal(signal.SIGINT, lambda *_: seen.append(1))\n"
        f"pathlib.Path({str(ready)!r}).touch()\n"
        "while not seen:\n"
        "    time.sleep(0.01)\n"
        "time.sleep(0.5)\n"
        f"pathlib.Path({str(seen)!r}).write_text(str(len(seen)))\n"
        f"sys.exit({ends_with})\n"
    )
    parent = (
        "import sys\n"
        "import garimpo_uncertainty.pipeline.experiment as experiment\n"
        f"sys.exit(experiment.run_model([sys.executable, '-c', {job!r}]))\n"
    )
    process = subprocess.Popen([sys.executable, "-c", parent], start_new_session=True)  # noqa: S603
    deadline = time.monotonic() + 30
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    if whole_group:
        os.killpg(process.pid, signal.SIGINT)
    else:
        os.kill(process.pid, signal.SIGINT)
    assert process.wait(timeout=30) == 130
    assert seen.read_text() == "1"


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
    found = garimpo_uncertainty.figures.make.runs_at(
        "pilot", ["unet_scratch", "convnet"], finished=False
    )
    assert found == [done]
    assert "Left out convnet" in capsys.readouterr().out
    # The other figures read the stages after the intervals, which this run has not done.
    assert garimpo_uncertainty.figures.make.runs_at("pilot", ["unet_scratch"], finished=True) == []
    assert "has no every stage ok yet" in capsys.readouterr().out
