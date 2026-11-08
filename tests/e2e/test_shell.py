"""The bash layer in slurm/: full.sbatch, pilot.sbatch and stages.sh, run as a job runs them.

Failures guarded here: a failing stage that lets later stages run on missing or stale inputs; a
value left empty in the env file reaching a stage as an empty argument after training; a flag name
or variable miswired in stages.sh; the chosen layer not reaching the calibrated stages; FORCE_FROM
redoing too much, too little, or nothing at all on a typo; the EXIT trap losing the exit code.

The scripts `cd` to their parent, so they run from SHELL_ROOT, which links the real slurm/, scripts/
and environment and holds a synthetic outputs/split.csv, a copy of the data root and the runs.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pytest

from tests.e2e import support

SHELL_ROOT = support.ROOT / "tests" / "artifacts" / "shell"
DATA = SHELL_ROOT / "data"
EXAMPLE = support.ROOT / "slurm" / "full.env.example"
LAYERS = ["decoder.blocks.4", "decoder.blocks.3"]
# full.env.example's values stay unless filled or shrunk here, so the example itself is under test.
FILLED = {
    "ARCH": "unet",
    "ENCODER_WEIGHTS": "none",
    "RUN_DIR": "runs/unet_scratch-f1.0",
    "TRAIN_FRACTION": "1.0",
    # The shared runs' training, which the synthetic tiles are known to survive.
    "EPOCHS": "3",
    "STEPS_PER_EPOCH": support.STEPS_PER_EPOCH,
    "BATCH_SIZE": "8",
    "LR": "1e-3",
    "WEIGHT_DECAY": "0",
    # Apart from BATCH_SIZE and NUM_WORKERS, so a swapped variable shows in the trace.
    "EVAL_BATCH_SIZE": "5",
    "LAYERS": f"({' '.join(LAYERS)})",
    "MIN_FRACTION_ASCENDING": "0",
    "SVM_C": "1.0",
    "SVM_CLASS_WEIGHT": "balanced",
    "SVM_PIXELS_PER_CELL": "50",
    "SWEEP_STEPS": "9",
    "SWEEP_CELLS": "6",
    # Few updates leave every Jacobian zero at the upstream 10,000 (docs/failure-modes.md, 40).
    "DIRECTION_TEMPERATURE": "1",
    "SEARCH_THRESHOLD_FEATURES": "1e-3",
    "SCORE_FLOOR_HA": "0.01",
    "NUM_WORKERS": "2",
}
STAGE_ORDER = (
    "train",
    "infer",
    "fit_directions",
    "layer_monotonicity",
    "choose_layer",
    "endpoint_monotonicity",
    "compass_scores",
    "calibrate",
    "generate_intervals",
    "compute_metrics",
)
EVALUATION_SCRIPTS = (
    "infer",
    "fit_directions",
    "verify_monotonicity",
    "compass_scores",
    "generate_intervals",
)


def example_names() -> list[str]:
    return re.findall(r"^([A-Z_]+)=", EXAMPLE.read_text(), flags=re.MULTILINE)


def write_env(name: str, **changes: str) -> Path:
    """Write the example followed by the test values; a later assignment wins, as in bash."""
    values = FILLED | changes
    path = SHELL_ROOT / f"{name}.env"
    lines = [f"{key}={value}" for key, value in values.items()]
    path.write_text(EXAMPLE.read_text() + "\n# test values\n" + "\n".join(lines) + "\n")
    return path


def run_job(*command: str, timeout: float = 600, **variables: str) -> subprocess.CompletedProcess:
    """Run a job script from SHELL_ROOT on CPU, with the data root and any job variables."""
    env = {
        **os.environ,
        "GARIMPO_DATA_DIR": str(DATA),
        "CELL_PX": str(support.CELL_PX),
        "CUDA_VISIBLE_DEVICES": "",
        # bash words its own errors in the locale; the tests match the English ones.
        "LC_ALL": "C",
        **variables,
    }
    return subprocess.run(  # noqa: S603 - arguments come from the tests
        ["bash", *command],  # noqa: S607 - bash from PATH, as a job finds it
        check=False,
        cwd=SHELL_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def status_mtimes(run_dir: Path) -> dict[str, int]:
    return {path.stem: path.stat().st_mtime_ns for path in (run_dir / "status").glob("*.json")}


def traced_commands(stderr: str) -> dict[str, list[list[str]]]:
    """Map each script to the argument lists `bash -x` shows it received, in order."""
    commands: dict[str, list[list[str]]] = {}
    for line in stderr.splitlines():
        words = line.split()
        if line.startswith("+") and "python" in words:
            scripts = [w for w in words if w.startswith("scripts/") and w.endswith(".py")]
            if scripts:
                commands.setdefault(Path(scripts[0]).stem, []).append(words)
    return commands


def value_of(words: list[str], flag: str) -> str:
    return words[words.index(flag) + 1]


def values_of(words: list[str], flag: str, count: int) -> list[str]:
    start = words.index(flag) + 1
    return words[start : start + count]


@pytest.fixture(scope="module")
def shell_root(data_root: Path) -> Path:
    SHELL_ROOT.mkdir(parents=True)
    for name in ("slurm", "scripts", "pyproject.toml", "uv.lock", ".venv"):
        (SHELL_ROOT / name).symlink_to(support.ROOT / name)
    (SHELL_ROOT / "outputs").mkdir()
    shutil.copy(support.SPLIT_PATH, SHELL_ROOT / "outputs" / "split.csv")
    counts = pd.read_csv(support.SPLIT_PATH).groupby(["scale", "role"]).size()
    tiles = {f"{scale}/{role}": int(n) for (scale, role), n in counts.items() if role != "dropped"}
    summary = {"tiles_by_scale_role": tiles}
    (SHELL_ROOT / "outputs" / "split_summary.json").write_text(json.dumps(summary))
    # Validation writes its status into the data root, which the shared one must never get.
    shutil.copytree(data_root, DATA)
    shutil.rmtree(DATA / "weights", ignore_errors=True)
    return SHELL_ROOT


class PassedJob(NamedTuple):
    run_dir: Path
    result: subprocess.CompletedProcess


@pytest.fixture(scope="module")
def passed_job(shell_root: Path) -> PassedJob:
    env = write_env("passed")
    result = run_job("-x", "slurm/full.sbatch", str(env))
    (shell_root / "passed.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return PassedJob(shell_root / FILLED["RUN_DIR"], result)


def test_full_job_runs_every_stage_and_passes_the_check(passed_job: PassedJob) -> None:
    assert "PIPELINE CHECK: PASS" in passed_job.result.stdout
    statuses = {
        stage: json.loads((passed_job.run_dir / "status" / f"{stage}.json").read_text())["status"]
        for stage in STAGE_ORDER
    }
    assert statuses == dict.fromkeys(STAGE_ORDER, "ok")


def test_full_job_passes_each_value_to_its_flag(passed_job: PassedJob) -> None:
    """A swapped variable runs fine and is wrong, so only the arguments themselves show it."""
    commands = traced_commands(passed_job.result.stderr)
    (train,) = commands["train"]
    assert value_of(train, "--batch-size") == FILLED["BATCH_SIZE"]
    assert value_of(train, "--encoder-weights") == FILLED["ENCODER_WEIGHTS"]
    for script in EVALUATION_SCRIPTS:
        for words in commands[script]:
            assert value_of(words, "--batch-size") == FILLED["EVAL_BATCH_SIZE"], script
            assert value_of(words, "--num-workers") == FILLED["NUM_WORKERS"], script
    (calibrate,) = commands["calibrate"]
    assert value_of(calibrate, "--lightgbm-jobs") == FILLED["NUM_WORKERS"]

    layer = support.chosen_layer(passed_job.run_dir)
    layer_sweep, endpoint_sweep = commands["verify_monotonicity"]
    assert values_of(commands["fit_directions"][0], "--layers", len(LAYERS)) == LAYERS
    assert values_of(layer_sweep, "--layers", len(LAYERS)) == LAYERS
    assert value_of(endpoint_sweep, "--layers") == layer
    assert value_of(commands["compass_scores"][0], "--layer") == layer
    assert value_of(commands["generate_intervals"][0], "--layer") == layer


def test_forced_rerun_redoes_only_what_it_names(passed_job: PassedJob) -> None:
    """FORCE_FROM redoes its stage and later ones, FORCE_VALIDATE the validation; none retrains."""
    run_dir = passed_job.run_dir
    validation = DATA / "status" / "fraction-1.0-seed-0" / "validate_data.json"
    with support.restored_after(run_dir):
        before = status_mtimes(run_dir)
        validated = validation.stat().st_mtime_ns
        checkpoint = (run_dir / "best_model.pt").stat().st_mtime_ns
        result = run_job(
            "slurm/full.sbatch",
            str(write_env("passed")),
            FORCE_FROM="calibrate",
            FORCE_VALIDATE="1",
        )
        after = status_mtimes(run_dir)
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
        assert "PIPELINE CHECK: PASS" in result.stdout
        redone = {stage for stage in after if after[stage] != before[stage]}
        assert redone == {"calibrate", "generate_intervals", "compute_metrics"}
        assert validation.stat().st_mtime_ns != validated
        assert (run_dir / "best_model.pt").stat().st_mtime_ns == checkpoint


def test_unknown_force_from_stops_the_job_and_fails_the_check(passed_job: PassedJob) -> None:
    """A typo must not run the stages unforced, which would skip and read as a successful redo."""
    before = status_mtimes(passed_job.run_dir)
    result = run_job("slurm/full.sbatch", str(write_env("passed")), FORCE_FROM="calibration")
    assert result.returncode == 1
    assert "FORCE_FROM=calibration is not one of" in result.stderr
    assert "job exit code 1" in result.stdout
    assert "PIPELINE CHECK: FAIL" in result.stdout
    assert status_mtimes(passed_job.run_dir) == before


@pytest.mark.parametrize(
    ("job", "run_dir"),
    [("pilot", "runs/pilot/unet-imagenet"), ("full", "runs/unet_imagenet-f1.0")],
)
def test_failed_stage_stops_the_job(shell_root: Path, job: str, run_dir: str) -> None:
    """ImageNet weights never downloaded: training fails, and nothing may run after it."""
    env = write_env("failing", ENCODER_WEIGHTS="imagenet", RUN_DIR=run_dir)
    command = ["slurm/pilot.sbatch"] if job == "pilot" else ["slurm/full.sbatch", str(env)]
    result = run_job(*command, ENCODER_WEIGHTS="imagenet")
    (shell_root / f"failed_{job}.log").write_text(result.stdout + result.stderr)
    assert result.returncode != 0
    assert "failing stage: train (failed)" in result.stdout
    assert "PIPELINE CHECK: FAIL" in result.stdout
    assert {path.stem for path in (shell_root / run_dir / "status").glob("*.json")} == {"train"}


@pytest.mark.parametrize("name", example_names())
def test_full_job_refuses_an_empty_value_before_any_stage(shell_root: Path, name: str) -> None:
    """Every value of the example, left empty, stops the job before training, not after."""
    run_dir = f"runs/empty_{name.lower()}"
    changes = {name: "()" if name == "LAYERS" else ""}
    env = write_env(f"empty_{name}", **({"RUN_DIR": run_dir} | changes))
    result = run_job("slurm/full.sbatch", str(env), timeout=30)
    assert result.returncode != 0
    assert ("LAYERS is empty" if name == "LAYERS" else f"{name}: parameter null") in result.stderr
    assert not (shell_root / run_dir).exists()
    assert "PIPELINE CHECK" not in result.stdout


def test_full_job_needs_an_env_file(shell_root: Path) -> None:
    result = run_job("slurm/full.sbatch", timeout=30)
    assert result.returncode != 0
    assert "usage: bash slurm/full.sbatch <values.env>" in result.stderr


def test_convnet_needs_no_encoder_weights(shell_root: Path) -> None:
    """Reaching the FORCE_FROM check shows the values passed; the typo then stops the job."""
    env = write_env("convnet", ARCH="convnet", ENCODER_WEIGHTS="", RUN_DIR="runs/convnet-f1.0")
    result = run_job("slurm/full.sbatch", str(env), FORCE_FROM="none")
    assert result.returncode == 1
    assert "ENCODER_WEIGHTS" not in result.stderr
    assert "FORCE_FROM=none is not one of" in result.stderr
