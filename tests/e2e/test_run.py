"""scripts/run.py, the runner of one model through every stage, run as Colab runs it.

Failures guarded here: a failing stage that lets later stages run on missing or stale inputs; a
value left empty or misspelled in the JSON config reaching a stage as an empty argument after
training, or a versioned config the runner would refuse; a config value passed to the wrong flag;
the chosen layer not reaching the calibrated stages; --force-from redoing too much, too little, or
nothing at all on a typo; a stage's exit code lost before the pipeline check; a stage's lines
missing from the log, or a new attempt overwriting the earlier ones; Ctrl+C leaving the stage
running, or skipping the check.

The runner reads its scripts and outputs from the working directory, so it runs from RUN_ROOT,
which links the real scripts/ and holds a synthetic outputs/split.csv, a copy of the data root and
the runs.
"""

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pytest

import garimpo_uncertainty.models.common
import garimpo_uncertainty.run_config
import garimpo_uncertainty.stages
from tests.e2e import support

RUN_ROOT = support.ROOT / "tests" / "artifacts" / "run"
DATA = RUN_ROOT / "data"
# The versioned config, so its own values are under test wherever FILLED leaves them.
BASE = support.ROOT / "configs" / "full" / "unet_scratch.json"
CONFIGS = sorted((support.ROOT / "configs").glob("*/*.json"))
# Ordered so the chosen layer is not the first: passing LAYERS[0] for it must show in the commands.
LAYERS = ["decoder.blocks.3", "decoder.blocks.4"]
# Not one config value: paths, the layer lists and the check's input, checked apart or not at all.
NOT_FROM_CONFIG = {
    "--checkpoint",
    "--layers",
    "--layer",
    "--use",
    "--report",
    "--split-summary",
    "--job-exit-code",
}
# Flags whose value is not the config key of their own name in capitals.
NAMED_OTHERWISE = {
    "--beta-max-features": "BETA_RANGE_FEATURES",
    "--beta-max-logits": "BETA_RANGE_LOGITS",
    "--cell-px": "GARIMPO_TEST_CELL_PX",
    "--cells-per-role": "SWEEP_CELLS",
    "--lightgbm-jobs": "NUM_WORKERS",
    "--steps": "SWEEP_STEPS",
}
FILLED = {
    "ARCH": "unet",
    "ENCODER_WEIGHTS": "none",
    "RUN_DIR": "runs/unet_scratch-f1.0",
    "TRAIN_FRACTION": 1.0,
    # Not SEED's 0, so swapping the two shows; at fraction 1.0 every seed selects every tile.
    "SUBSET_SEED": 7,
    # The shared runs' training, which the synthetic tiles are known to survive.
    "EPOCHS": 3,
    "STEPS_PER_EPOCH": int(support.STEPS_PER_EPOCH),
    "BATCH_SIZE": 8,
    "LR": 1e-3,
    # Not 0, which SEED holds, so a swap shows in the commands.
    "WEIGHT_DECAY": 1e-6,
    # Apart from BATCH_SIZE and NUM_WORKERS, so a swapped value shows in the commands.
    "EVAL_BATCH_SIZE": 5,
    "LAYERS": LAYERS,
    # Every value distinct from every other, so a wrong key shows: 0.0 is not SEED's 0.
    "MIN_FRACTION_ASCENDING": 0.0,
    "SVM_C": 1.5,
    "SVM_CLASS_WEIGHT": "balanced",
    "SVM_PIXELS_PER_CELL": 50,
    "SWEEP_STEPS": 9,
    "SWEEP_CELLS": 6,
    # Few updates leave every Jacobian zero at the upstream 10,000 (docs/failure-modes.md, 40).
    "DIRECTION_TEMPERATURE": 1,
    "SEARCH_THRESHOLD_FEATURES": 2e-3,
    # Not 0.01, which SEARCH_THRESHOLD_LOGITS holds in the base config.
    "SCORE_FLOOR_HA": 0.02,
}
# What the pilot may change: its size and length, and the layer threshold, so that a model trained
# for a few updates still reaches the last stage.
PILOT_ONLY = {"RUN_DIR", "TRAIN_FRACTION", "EPOCHS", "STEPS_PER_EPOCH", "MIN_FRACTION_ASCENDING"}
# An execution setting, not part of the config: Colab leaves it at the runner's default.
NUM_WORKERS = "2"
ATTEMPT = "=== attempt started "


def base_names() -> list[str]:
    return list(json.loads(BASE.read_text(encoding="utf-8")))


def write_config(name: str, **changes: object) -> Path:
    """Write the base config with the test values over it; a change to None writes null."""
    values = json.loads(BASE.read_text(encoding="utf-8")) | FILLED | changes
    path = RUN_ROOT / "configs" / f"{name}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(values, indent=2))
    return path


def log_of(config: Path) -> Path:
    """Return the log the runner names after the config's folder and file: pilot-unet_scratch."""
    return RUN_ROOT / "outputs" / "logs" / f"{config.parent.name}-{config.stem}.log"


def runner_command(*arguments: str) -> list[str]:
    return [sys.executable, "scripts/run.py", *arguments]


def runner_env() -> dict[str, str]:
    return {
        **os.environ,
        "GARIMPO_DATA_DIR": str(DATA),
        "GARIMPO_TEST_CELL_PX": str(support.CELL_PX),
        "CUDA_VISIBLE_DEVICES": "",
    }


def run_job(*arguments: str, timeout: float = 600) -> subprocess.CompletedProcess:
    """Run the runner from RUN_ROOT on CPU, with the data root and the test's worker count."""
    return subprocess.run(  # noqa: S603 - arguments come from the tests
        runner_command(*arguments, "--num-workers", NUM_WORKERS),
        check=False,
        cwd=RUN_ROOT,
        env=runner_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def status_mtimes(run_dir: Path) -> dict[str, int]:
    return {path.stem: path.stat().st_mtime_ns for path in (run_dir / "status").glob("*.json")}


def started_commands(output: str) -> dict[str, list[list[str]]]:
    """Map each script to the argument lists the runner logged when it started it, in order."""
    commands: dict[str, list[list[str]]] = {}
    for line in output.splitlines():
        _, marker, command = line.partition("=== start ")
        if not marker:
            continue
        words = shlex.split(command.split(": ", 1)[1])
        script = next(w for w in words if w.startswith("scripts/") and w.endswith(".py"))
        commands.setdefault(Path(script).stem, []).append(words)
    return commands


def values_of(words: list[str], flag: str) -> list[str]:
    """Every word after `flag` up to the next flag, so an extra value shows too."""
    start = words.index(flag) + 1
    end = next((i for i in range(start, len(words)) if words[i].startswith("--")), len(words))
    return words[start:end]


@pytest.fixture(scope="module")
def run_root(data_root: Path) -> Path:
    RUN_ROOT.mkdir(parents=True)
    (RUN_ROOT / "scripts").symlink_to(support.ROOT / "scripts")
    (RUN_ROOT / "outputs").mkdir()
    shutil.copy(support.SPLIT_PATH, RUN_ROOT / "outputs" / "split.csv")
    counts = pd.read_csv(support.SPLIT_PATH).groupby(["scale", "role"]).size()
    tiles = {f"{scale}/{role}": int(n) for (scale, role), n in counts.items() if role != "dropped"}
    summary = {"tiles_by_scale_role": tiles}
    (RUN_ROOT / "outputs" / "split_summary.json").write_text(json.dumps(summary))
    # Validation writes its status into the data root, which the shared one must never get.
    shutil.copytree(data_root, DATA)
    shutil.rmtree(DATA / "weights", ignore_errors=True)
    return RUN_ROOT


class PassedJob(NamedTuple):
    config: Path
    run_dir: Path
    result: subprocess.CompletedProcess


@pytest.fixture(scope="module")
def passed_job(run_root: Path) -> PassedJob:
    config = write_config("passed")
    result = run_job(str(config))
    assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-4000:]
    return PassedJob(config, run_root / FILLED["RUN_DIR"], result)


def test_full_job_runs_every_stage_and_passes_the_check(passed_job: PassedJob) -> None:
    assert "PIPELINE CHECK: PASS" in passed_job.result.stdout
    statuses = {
        stage: json.loads((passed_job.run_dir / "status" / f"{stage}.json").read_text())["status"]
        for stage in garimpo_uncertainty.stages.STAGE_ORDER
    }
    assert statuses == dict.fromkeys(garimpo_uncertainty.stages.STAGE_ORDER, "ok")


def test_log_holds_every_line_the_job_printed(passed_job: PassedJob) -> None:
    """The log file is what Colab keeps in Drive; a buffered stage would leave its lines out."""
    log = log_of(passed_job.config).read_text(encoding="utf-8")
    for line in passed_job.result.stdout.splitlines():
        assert line in log, line
    for stage in ("validate_data", *garimpo_uncertainty.stages.STAGE_ORDER, "pipeline_check"):
        assert f"=== end {stage}: exit 0 in " in log, stage
    # The stages' own lines, not only the runner's: the training's line per epoch among them.
    assert "train: ok in" in log
    assert f"epoch {FILLED['EPOCHS']}/{FILLED['EPOCHS']}: train_loss " in log
    assert "PIPELINE CHECK: PASS" in log


def expected_key(flag: str, script: str) -> str:
    """Return the config key a flag must receive: from its own name, never from the runner."""
    if flag == "--batch-size":
        return "BATCH_SIZE" if script == "train" else "EVAL_BATCH_SIZE"
    return NAMED_OTHERWISE.get(flag, flag.removeprefix("--").upper().replace("-", "_"))


def test_full_job_passes_each_value_to_its_flag(passed_job: PassedJob) -> None:
    """Every flag that takes a value receives the config value its name implies.

    The key comes from the flag's name or NAMED_OTHERWISE, written here, so passing one value for
    another shows; the values are distinct, so the commands tell them apart.
    """
    config = json.loads(passed_job.config.read_text(encoding="utf-8"))
    values = {key: str(value) for key, value in config.items() if not isinstance(value, list)}
    values |= {"NUM_WORKERS": NUM_WORKERS, "GARIMPO_TEST_CELL_PX": str(support.CELL_PX)}
    # A wrong key holding the same value would pass, so no two may share one.
    by_value: dict[str, set[str]] = {}
    for name, value in values.items():
        by_value.setdefault(value, set()).add(name)
    shared = {value: names for value, names in by_value.items() if len(names) > 1}
    assert not shared, shared

    commands = started_commands(passed_job.result.stdout)
    received = set()
    for script, runs in commands.items():
        for words in runs:
            for flag in (w for w in words if w.startswith("--") and w not in NOT_FROM_CONFIG):
                given = values_of(words, flag)
                if not given:
                    continue  # a switch such as --force
                name = expected_key(flag, script)
                assert name in values, f"{script} {flag}: no config key {name}"
                assert given == [values[name]], (script, flag, name)
                received.add(name)
    assert received == set(values), f"never passed to a stage: {sorted(set(values) - received)}"

    layer = support.chosen_layer(passed_job.run_dir)
    assert layer != LAYERS[0], "the run must choose a layer other than the first"
    layer_sweep, endpoint_sweep = commands["verify_monotonicity"]
    assert values_of(commands["fit_directions"][0], "--layers") == LAYERS
    assert values_of(layer_sweep, "--layers") == LAYERS
    assert values_of(endpoint_sweep, "--layers") == [layer]
    assert values_of(commands["compass_scores"][0], "--layer") == [layer]
    assert values_of(commands["generate_intervals"][0], "--layer") == [layer]


def test_forced_rerun_redoes_only_what_it_names(passed_job: PassedJob) -> None:
    """--force-from redoes its stage and later ones, --force-validate the validation; none retrains.

    The rerun appends a second attempt to the log, under the first.
    """
    run_dir = passed_job.run_dir
    validation = DATA / "status" / "fraction-1.0-seed-7" / "validate_data.json"
    with support.restored_after(run_dir):
        before = status_mtimes(run_dir)
        validated = validation.stat().st_mtime_ns
        checkpoint = (run_dir / "best_model.pt").stat().st_mtime_ns
        result = run_job(str(passed_job.config), "--force-from", "calibrate", "--force-validate")
        after = status_mtimes(run_dir)
        assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-4000:]
        assert "PIPELINE CHECK: PASS" in result.stdout
        redone = {stage for stage in after if after[stage] != before[stage]}
        assert redone == {"calibrate", "generate_intervals", "compute_metrics"}
        assert validation.stat().st_mtime_ns != validated
        assert (run_dir / "best_model.pt").stat().st_mtime_ns == checkpoint
    log = log_of(passed_job.config).read_text(encoding="utf-8")
    first, second = log.split(ATTEMPT)[1:3]
    assert "PIPELINE CHECK: PASS" in first
    assert "PIPELINE CHECK: PASS" in second


def test_unknown_force_from_stops_the_job_before_any_stage(passed_job: PassedJob) -> None:
    """A typo must not run the stages unforced, which would skip and read as a successful redo."""
    before = status_mtimes(passed_job.run_dir)
    result = run_job(str(passed_job.config), "--force-from", "calibration", timeout=30)
    assert result.returncode == 2
    assert "invalid choice: 'calibration'" in result.stderr
    assert "=== start" not in result.stdout
    assert status_mtimes(passed_job.run_dir) == before


def test_failed_stage_stops_the_job(run_root: Path) -> None:
    """ImageNet weights never downloaded: training fails, and nothing may run after it."""
    run_dir = "runs/unet_imagenet-f1.0"
    config = write_config("failing", ENCODER_WEIGHTS="imagenet", RUN_DIR=run_dir)
    result = run_job(str(config))
    assert result.returncode != 0
    assert "failing stage: train (failed)" in result.stdout
    assert f"job exit code {result.returncode}" in result.stdout
    assert {path.stem for path in (run_root / run_dir / "status").glob("*.json")} == {"train"}
    log = log_of(config).read_text(encoding="utf-8")
    assert "=== end train: exit 1 in " in log
    assert "=== start infer" not in log
    assert "PIPELINE CHECK: FAIL" in log


def test_ctrl_c_stops_the_stage_and_still_runs_the_check(run_root: Path) -> None:
    """Colab's interrupt reaches the runner alone: the stage must stop too, and the check run."""
    run_dir = "runs/interrupted"
    config = write_config("interrupted", RUN_DIR=run_dir)
    with subprocess.Popen(  # noqa: S603 - arguments come from the tests
        runner_command(str(config), "--num-workers", NUM_WORKERS),
        cwd=RUN_ROOT,
        env=runner_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as process:
        assert process.stdout is not None
        lines = []
        for line in process.stdout:
            lines.append(line)
            if "=== start train" in line:
                process.send_signal(signal.SIGINT)
                break
        output = "".join(lines) + process.communicate(timeout=300)[0]
    assert process.returncode == 130, output[-8000:]
    assert "interrupted by Ctrl+C" in output
    assert "PIPELINE CHECK: FAIL" in output
    assert "=== start infer" not in output
    train = run_root / run_dir / "status" / "train.json"
    # Absent when the signal came during the imports, before the stage wrote its first status.
    assert not train.exists() or json.loads(train.read_text())["status"] == "interrupted"


@pytest.mark.parametrize("name", base_names())
def test_full_job_refuses_an_empty_value_before_any_stage(run_root: Path, name: str) -> None:
    """Every value of the base config, left null, stops the job before training, not after."""
    run_dir = f"runs/empty_{name.lower()}"
    changes = {name: [] if name == "LAYERS" else None}
    config = write_config(f"empty_{name}", **({"RUN_DIR": run_dir} | changes))
    result = run_job(str(config), timeout=30)
    assert result.returncode != 0
    assert f"{name}: missing or empty" in result.stdout
    assert not (run_root / run_dir).exists()
    assert "=== start" not in result.stdout


def test_full_job_refuses_a_misspelled_key_before_any_stage(run_root: Path) -> None:
    """A misspelled key would otherwise leave its value at nothing, read as no setting at all.

    The reason reaches the log too, since Colab shows the log's last attempt.
    """
    run_dir = "runs/misspelled"
    config = write_config("misspelled", RUN_DIR=run_dir, SWEEP_STEP=9)
    result = run_job(str(config), timeout=30)
    assert result.returncode != 0
    assert "SWEEP_STEP: unknown key" in result.stdout
    assert "SWEEP_STEP: unknown key" in log_of(config).read_text(encoding="utf-8")
    assert not (run_root / run_dir).exists()


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: f"{path.parent.name}/{path.stem}")
def test_versioned_configs_are_accepted_and_name_model_layers(path: Path) -> None:
    """A layer the model lacks would stop fit_directions only after the whole training."""
    config = json.loads(path.read_text(encoding="utf-8"))
    assert garimpo_uncertainty.run_config.problems(config) == []
    model = garimpo_uncertainty.models.common.build(config["ARCH"], in_channels=6)
    assert set(config["LAYERS"]) <= set(garimpo_uncertainty.models.common.feature_layers(model))


@pytest.mark.parametrize(
    "name", sorted(path.stem for path in support.ROOT.glob("configs/full/*.json"))
)
def test_pilot_config_differs_from_the_full_one_only_in_scale(name: str) -> None:
    """The pilot checks the code the full run will run, so only its size and length may differ."""
    full, pilot = (
        json.loads((support.ROOT / "configs" / scale / f"{name}.json").read_text(encoding="utf-8"))
        for scale in ("full", "pilot")
    )
    differing = {key for key in full.keys() | pilot.keys() if full.get(key) != pilot.get(key)}
    assert differing == PILOT_ONLY, differing
    assert pilot["RUN_DIR"] == f"runs/pilot/{name}-f0.01"
    assert full["RUN_DIR"] == f"runs/{name}-f1.0"


@pytest.mark.parametrize("floor", [0, -0.01, float("nan"), float("inf"), "0.01ha"])
def test_full_job_refuses_a_bad_score_floor_before_any_stage(run_root: Path, floor: object) -> None:
    """`calibrate.py` refuses these too, but only after training; the job must not get that far."""
    run_dir = f"runs/floor_{str(floor).replace('.', '_')}"
    config = write_config(f"floor_{floor}", RUN_DIR=run_dir, SCORE_FLOOR_HA=floor)
    result = run_job(str(config), timeout=60)
    assert result.returncode != 0
    assert f"SCORE_FLOOR_HA must be a positive finite number, got '{floor}'" in result.stdout
    assert not (run_root / run_dir).exists()
    assert "=== start" not in result.stdout


def test_full_job_needs_a_config(run_root: Path) -> None:
    result = run_job(timeout=30)
    assert result.returncode == 2
    assert "the following arguments are required: config" in result.stderr
