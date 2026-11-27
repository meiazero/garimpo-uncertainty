"""Which scales `garimpo experiment` runs, and which runs `garimpo figures --scale` draws."""

import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

import garimpo_uncertainty.config
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
    codes = experiment.run_scales("calibrate")
    assert runs == expected_runs
    assert list(codes) == [scale for scale, _ in expected_runs]
    assert max(codes.values()) == result


def test_only_a_scale_whose_runs_all_passed_is_merged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment, "run_scales", lambda force_from: {"pilot": 0, "full": 1})
    monkeypatch.setattr(garimpo_uncertainty.pipeline.status, "print_status", lambda *_: None)
    monkeypatch.setattr(signal, "signal", lambda *_: None)
    monkeypatch.setattr(sys, "argv", ["garimpo experiment"])
    merged = []

    def merge(words: list[str], check: bool) -> types.SimpleNamespace:
        merged.append(words[-1])
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(experiment.subprocess, "run", merge)
    with pytest.raises(SystemExit) as exited:
        experiment.main()
    assert merged == ["pilot"]
    assert exited.value.code == 1


def test_each_job_runs_the_code_this_interpreter_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `garimpo` found on PATH may be another install, or none."""
    commands = []
    monkeypatch.setattr(experiment, "run_model", lambda command: commands.append(command) or 0)
    monkeypatch.setattr(garimpo_uncertainty.config, "models", lambda configs: ["convnet"])
    assert experiment.run_scale("pilot", []) == 0
    assert commands == [[sys.executable, "-m", "garimpo_uncertainty", "run", "pilot", "convnet"]]


def test_the_models_run_in_main_s_order_with_a_new_one_last() -> None:
    models = ["convnet", "new", "unet_imagenet", "unet_scratch", "unet_sentinel2"]
    expected = ["unet_scratch", "unet_imagenet", "unet_sentinel2", "convnet", "new"]
    assert experiment.run_order(models) == expected


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


def test_a_second_interrupt_reaches_the_job_as_sigterm(tmp_path: Path) -> None:
    """Colab sends SIGTERM soon after the first SIGINT; the job must get it to stop at once."""
    ready, seen = tmp_path / "ready", tmp_path / "seen"
    job = (
        "import pathlib, signal, sys, time\n"
        "seen = []\n"
        "signal.signal(signal.SIGINT, lambda *_: seen.append('int'))\n"
        "signal.signal(signal.SIGTERM, lambda *_: seen.append('term'))\n"
        f"pathlib.Path({str(ready)!r}).touch()\n"
        "deadline = time.monotonic() + 10\n"
        "while 'term' not in seen and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        f"pathlib.Path({str(seen)!r}).write_text(','.join(seen))\n"
        "sys.exit(130)\n"
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
    os.kill(process.pid, signal.SIGINT)
    time.sleep(0.05)
    os.kill(process.pid, signal.SIGTERM)
    assert process.wait(timeout=30) == 130
    assert seen.read_text() == "int,term"


def test_an_interrupt_while_the_job_starts_still_reaches_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = tmp_path / "seen"
    job = (
        "import pathlib, signal, sys, time\n"
        "seen = []\n"
        "signal.signal(signal.SIGINT, lambda *_: seen.append('int'))\n"
        "deadline = time.monotonic() + 10\n"
        "while not seen and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        f"pathlib.Path({str(seen)!r}).write_text(','.join(seen))\n"
    )
    start = subprocess.Popen

    def interrupted_start(*args: object, **kwargs: object) -> subprocess.Popen:
        # The handler is in place; the interrupt lands before the job can be signalled.
        signal.raise_signal(signal.SIGINT)
        process = start(*args, **kwargs)
        time.sleep(1)  # the job installs its handler first, as garimpo run does
        return process

    monkeypatch.setattr(experiment.subprocess, "Popen", interrupted_start)
    assert experiment.run_model([sys.executable, "-c", job]) == experiment.STOPPED
    assert seen.read_text() == "int"


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
