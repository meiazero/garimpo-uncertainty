"""`garimpo run`, the job of one model through every stage, run as Colab runs it.

Failures guarded here: a failing stage that lets later stages run on missing or stale inputs; a
value left empty or misspelled in the JSON config reaching a stage as an empty argument after
training, or a versioned config the runner would refuse; a config value passed to the wrong flag;
the chosen layer not reaching the calibrated stages; a versioned run dir or log name drifting from
what recorded runs and their fingerprints say; --force-from redoing too much, too little, or
nothing at all on a typo; a stage's exit code lost before the pipeline check; a stage's lines
missing from the log, or a new attempt overwriting the earlier ones; Ctrl+C leaving the stage
running, losing the lines it prints as it stops, or skipping the check; a job lost between two
stages leaving the later ones ok.

The runner reads its configs and outputs from the working directory, so it runs from RUN_ROOT,
which holds configs of its own, a synthetic outputs/split.csv, a copy of the data root and the runs.
"""

import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pytest

import garimpo_uncertainty.config
import garimpo_uncertainty.models.common
import garimpo_uncertainty.pipeline.runner
from tests.e2e import support

RUN_ROOT = support.ROOT / "tests" / "artifacts" / "run"
DATA = RUN_ROOT / "data"
CONFIGS = support.ROOT / "configs"
# The versioned base, so its own values are under test wherever FILLED leaves them.
BASE = CONFIGS / "base.json"
MODELS = sorted(path.stem for path in (CONFIGS / "models").glob("*.json"))
SCALES = sorted(path.stem for path in (CONFIGS / "scales").glob("*.json"))
# The scale every job here runs at, in RUN_ROOT's own configs.
SCALE = "test"
# Ordered so the chosen layer is not the first: passing LAYERS[0] for it must show in the commands.
LAYERS = ["decoder.blocks.3", "decoder.blocks.4"]
# Not one config value: paths, the layer lists and the check's input, checked apart or not at all.
NOT_FROM_CONFIG = {
    "--checkpoint",
    "--layers",
    "--layer",
    "--layer-dir",
    "--use",
    "--report",
    "--split-summary",
    "--job-exit-code",
}
# Flags whose value is not the config key of their own name in capitals.
NAMED_OTHERWISE = {
    "--beta-max-features": "BETA_RANGE_FEATURES",
    "--beta-max-logits": "BETA_RANGE_LOGITS",
    "--cells-per-role": "SWEEP_CELLS",
    "--lightgbm-jobs": "NUM_WORKERS",
    "--steps": "SWEEP_STEPS",
}
FILLED = {
    "ARCH": "unet",
    "ENCODER_WEIGHTS": "none",
    "RUNS_DIR": "runs",
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
    # The synthetic store's cells, not the real store's 512 pixels.
    "CELL_PX": support.CELL_PX,
}
# An execution setting, not part of the config: Colab leaves it at the runner's default.
NUM_WORKERS = "2"
ATTEMPT = "=== attempt started "


def write_configs(configs: Path, model: str, **changes: object) -> None:
    """Write base, `model` and the test scale with FILLED over the versioned base; None is null.

    Each change goes to the file its key belongs in, or to the model file when it belongs nowhere.
    """
    values = json.loads(BASE.read_text(encoding="utf-8")) | FILLED | changes
    homes = {
        "models": (
            *garimpo_uncertainty.config.MODEL_KEYS,
            garimpo_uncertainty.config.ENCODER_WEIGHTS,
        ),
        "scales": garimpo_uncertainty.config.SCALE_KEYS,
        "base": garimpo_uncertainty.config.BASE_KEYS,
    }
    placed = {key for keys in homes.values() for key in keys}
    files = {
        configs / "base.json": homes["base"],
        configs / "scales" / f"{SCALE}.json": homes["scales"],
        configs / "models" / f"{model}.json": (*homes["models"], *(set(values) - placed)),
    }
    for path, keys in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({key: values[key] for key in keys if key in values}, indent=2))


def write_config(model: str, **changes: object) -> str:
    """Write a model of RUN_ROOT's configs with `changes` to its keys; return its name."""
    write_configs(RUN_ROOT / "configs", model, **changes)
    return model


def run_dir_of(model: str) -> Path:
    return (
        RUN_ROOT
        / garimpo_uncertainty.config.RunConfig.load(RUN_ROOT / "configs", SCALE, model).run_dir
    )


def log_of(model: str) -> Path:
    """Return the log the runner names after the scale and the model: pilot-unet_scratch."""
    return RUN_ROOT / "outputs" / "logs" / f"{SCALE}-{model}.log"


def runner_command(*arguments: str) -> list[str]:
    return support.garimpo("run", *arguments)


def runner_env() -> dict[str, str]:
    return {
        **os.environ,
        "GARIMPO_DATA_DIR": str(DATA),
        "CUDA_VISIBLE_DEVICES": "",
    }


def run_job(*arguments: str, timeout: float = 600) -> subprocess.CompletedProcess:
    """Run the runner from RUN_ROOT on CPU, with the data root and the test's worker count."""
    return subprocess.run(  # noqa: S603 - arguments come from the tests
        runner_command(SCALE, *arguments, "--num-workers", NUM_WORKERS),
        check=False,
        cwd=RUN_ROOT,
        env=runner_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def status_mtimes(run_dir: Path) -> dict[str, int]:
    """Map each status file to its mtime, an other layer's as `<stage> [<layer>]`."""
    mtimes = {path.stem: path.stat().st_mtime_ns for path in (run_dir / "status").glob("*.json")}
    for path in (run_dir / "layers").glob("*/status/*.json"):
        mtimes[f"{path.stem} [{path.parent.parent.name}]"] = path.stat().st_mtime_ns
    return mtimes


def started_commands(output: str) -> dict[str, list[list[str]]]:
    """Map each script to the argument lists the runner logged when it started it, in order."""
    commands: dict[str, list[list[str]]] = {}
    for line in output.splitlines():
        _, marker, command = line.partition("=== start ")
        if not marker:
            continue
        words = shlex.split(command.split(": ", 1)[1])
        command_name = words[words.index("garimpo_uncertainty") + 1]
        commands.setdefault(command_name, []).append(words)
    return commands


def values_of(words: list[str], flag: str) -> list[str]:
    """Every word after `flag` up to the next flag, so an extra value shows too."""
    start = words.index(flag) + 1
    end = next((i for i in range(start, len(words)) if words[i].startswith("--")), len(words))
    return words[start:end]


@pytest.fixture(scope="module")
def run_root(data_root: Path) -> Path:
    RUN_ROOT.mkdir(parents=True)
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
    model: str
    run_dir: Path
    result: subprocess.CompletedProcess


@pytest.fixture(scope="module")
def passed_job(run_root: Path) -> PassedJob:
    model = write_config("unet_scratch")
    result = run_job(model)
    assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-4000:]
    return PassedJob(model, run_dir_of(model), result)


def test_full_job_runs_every_stage_and_passes_the_check(passed_job: PassedJob) -> None:
    assert "PIPELINE CHECK: PASS" in passed_job.result.stdout
    statuses = {
        stage: json.loads((passed_job.run_dir / "status" / f"{stage}.json").read_text())["status"]
        for stage in garimpo_uncertainty.pipeline.runner.STAGE_ORDER
    }
    assert statuses == dict.fromkeys(garimpo_uncertainty.pipeline.runner.STAGE_ORDER, "ok")


def test_log_holds_every_line_the_job_printed(passed_job: PassedJob) -> None:
    """The log file is what Colab keeps in Drive, so every line the job printed must be in it."""
    log = log_of(passed_job.model).read_text(encoding="utf-8")
    for line in passed_job.result.stdout.splitlines():
        assert line in log, line
    for stage in (
        "validate_data",
        *garimpo_uncertainty.pipeline.runner.STAGE_ORDER,
        "pipeline_check",
    ):
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
    config = garimpo_uncertainty.config.RunConfig.load(
        RUN_ROOT / "configs", SCALE, passed_job.model
    )
    # RUNS_DIR reaches the stages only through RUN_DIR, which the runner derives from it.
    listed = {key: value for key, value in config.values.items() if key != "RUNS_DIR"}
    values = {key: str(value) for key, value in listed.items() if not isinstance(value, list)}
    values |= {"RUN_DIR": config.run_dir, "NUM_WORKERS": NUM_WORKERS}
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
    # Then the same four stages once more, on the layer not chosen, into a directory of its own.
    (other,) = set(LAYERS) - {layer}
    other_dir = f"{config.run_dir}/layers/{other}"
    for script in garimpo_uncertainty.pipeline.runner.LAYER_STAGES:
        main, again = commands[script]
        assert "--layer-dir" not in main, script
        assert values_of(again, "--layer-dir") == [other_dir], script
    assert values_of(commands["compass_scores"][1], "--layer") == [other]
    assert values_of(commands["generate_intervals"][1], "--layer") == [other]


def test_full_job_scores_every_other_layer_apart(passed_job: PassedJob) -> None:
    """Each other layer gets its own results, read from the same predictions as the chosen one.

    The output-space rows are arithmetic on the shared predictions, so they must match the chosen
    layer's to the bit; the COMPASS rows run the model in another process, whose last bits may
    differ on a hybrid CPU (support.one_core), so only COMPASS-J, which reads the layer, must move.
    """
    run_dir = passed_job.run_dir
    layer = support.chosen_layer(run_dir)
    (other,) = set(LAYERS) - {layer}
    other_dir = run_dir / "layers" / other
    for stage in garimpo_uncertainty.pipeline.runner.LAYER_STAGES:
        assert json.loads((other_dir / "status" / f"{stage}.json").read_text())["status"] == "ok"
    chosen = pd.read_csv(run_dir / "metrics" / "results.csv")
    again = pd.read_csv(other_dir / "metrics" / "results.csv")
    assert set(chosen.layer) == {layer}
    assert set(again.layer) == {other}
    keys = ["alpha", "method", "calibration"]
    chosen, again = chosen.set_index(keys).sort_index(), again.set_index(keys).sort_index()
    assert chosen.index.equals(again.index)
    methods = chosen.index.get_level_values("method")
    calibrations = chosen.index.get_level_values("calibration")
    # The weighted rows read the summed features of the layer through the density ratio.
    shared = ~methods.str.contains("compass") & ~calibrations.str.contains("weighted")
    layered = methods.str.contains("compass_j")
    assert shared.any()
    assert layered.any()
    columns = ["picp", "mpiw", "n_cells"]
    pd.testing.assert_frame_equal(chosen[shared][columns], again[shared][columns], check_exact=True)
    assert not chosen[layered][["mpiw"]].equals(again[layered][["mpiw"]])


def pipeline_check(run_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    command = support.garimpo("check", "--run-dir", str(run_dir))
    command += ["--train-fraction", str(FILLED["TRAIN_FRACTION"])]
    command += ["--subset-seed", str(FILLED["SUBSET_SEED"]), *extra]
    return subprocess.run(  # noqa: S603 - arguments come from the test
        command, check=False, cwd=RUN_ROOT, env=runner_env(), capture_output=True, text=True
    )


def test_pipeline_check_names_another_layer_that_never_ran(passed_job: PassedJob) -> None:
    """A session lost after the chosen layer's stages leaves the other layers' tables absent."""
    run_dir = passed_job.run_dir
    (other,) = set(LAYERS) - {support.chosen_layer(run_dir)}
    assert pipeline_check(run_dir, "--layers", *LAYERS).returncode == 0
    with support.restored_after(run_dir):
        shutil.rmtree(run_dir / "layers" / other)
        report = pipeline_check(run_dir, "--layers", *LAYERS)
    assert report.returncode == 1
    assert f"compass_scores [{other}]" in report.stdout.split("PIPELINE CHECK: FAIL")[1]


def test_pipeline_check_reads_another_layers_results(passed_job: PassedJob) -> None:
    """An other layer's table is checked as the chosen one's: a NaN width fails, naming it."""
    run_dir = passed_job.run_dir
    (other,) = set(LAYERS) - {support.chosen_layer(run_dir)}
    results = run_dir / "layers" / other / "metrics" / "results.csv"
    with support.restored_after(run_dir):
        table = pd.read_csv(results)
        table.loc[table.method == "compass_j_pca", "mpiw"] = float("nan")
        table.to_csv(results, index=False)
        report = pipeline_check(run_dir, "--layers", *LAYERS)
    assert report.returncode == 1
    verdict = report.stdout.split("PIPELINE CHECK: FAIL")[1]
    assert f"results contain NaN in PICP or MPIW [{other}]" in verdict


@pytest.mark.parametrize(("content", "cause"), [(None, "FileNotFoundError"), ("[]", "KeyError")])
def test_pipeline_check_fails_without_the_chosen_layer(
    passed_job: PassedJob, content: str | None, cause: str
) -> None:
    """Every status ok, but chosen_layer.json gone or garbled: the check still ends in a verdict."""
    run_dir = passed_job.run_dir
    with support.restored_after(run_dir):
        chosen = run_dir / "chosen_layer.json"
        if content is None:
            chosen.unlink()
        else:
            chosen.write_text(content)
        report = pipeline_check(run_dir, "--layers", *LAYERS)
    assert report.returncode == 1
    failing = report.stdout.split("--- failing stage: chosen_layer.json (unreadable)\n")[1]
    assert failing.startswith(f"{run_dir / 'chosen_layer.json'}: {cause}")
    assert "chosen_layer.json" in report.stdout.split("PIPELINE CHECK: FAIL")[1]


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
        result = run_job(passed_job.model, "--force-from", "calibrate", "--force-validate")
        after = status_mtimes(run_dir)
        assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-4000:]
        assert "PIPELINE CHECK: PASS" in result.stdout
        redone = {stage for stage in after if after[stage] != before[stage]}
        (other,) = set(LAYERS) - {support.chosen_layer(run_dir)}
        forced = ("calibrate", "generate_intervals", "compute_metrics")
        assert redone == {*forced, *(f"{stage} [{other}]" for stage in forced)}
        assert validation.stat().st_mtime_ns != validated
        assert (run_dir / "best_model.pt").stat().st_mtime_ns == checkpoint
    log = log_of(passed_job.model).read_text(encoding="utf-8")
    first, second = log.split(ATTEMPT)[1:3]
    assert "PIPELINE CHECK: PASS" in first
    assert "PIPELINE CHECK: PASS" in second


def test_a_job_lost_after_a_stage_merges_nothing_until_rerun(
    passed_job: PassedJob, tmp_path: Path
) -> None:
    """A lost runtime ends the job between two stages, after a copy to Drive took its statuses.

    The later stages must not read ok, or the merge takes their tables of the earlier layer; the
    next attempt must skip them back to ok, since their inputs did not change.
    """
    run_dir = passed_job.run_dir
    choose = started_commands(passed_job.result.stdout)["choose_layer"][0]
    order = garimpo_uncertainty.pipeline.runner.STAGE_ORDER
    later = order[order.index("choose_layer") + 1 :]
    with support.restored_after(run_dir):
        subprocess.run(  # noqa: S603 - the runner's own command
            [*choose, "--force"], check=True, cwd=RUN_ROOT, env=runner_env(), capture_output=True
        )
        statuses = {s: json.loads((run_dir / "status" / f"{s}.json").read_text()) for s in later}
        assert {s: r["status"] for s, r in statuses.items()} == dict.fromkeys(later, "stale")
        # The other layers' stages read the same run, so they cannot vouch for their tables either.
        (other_status,) = (run_dir / "layers").glob("*/status")
        for stage in garimpo_uncertainty.pipeline.runner.LAYER_STAGES:
            record = json.loads((other_status / f"{stage}.json").read_text())
            assert record["status"] == "stale", stage
        merge = support.garimpo("merge", "--runs", str(run_dir))
        refused = subprocess.run(  # noqa: S603 - arguments come from the test
            [*merge, "--out", str(tmp_path / "merged")],
            check=False,
            cwd=RUN_ROOT,
            env=runner_env(),
            capture_output=True,
            text=True,
        )
        assert refused.returncode != 0
        assert f"{later[0]} is stale" in refused.stderr
        assert not (tmp_path / "merged").exists()
        check = support.garimpo("check", "--run-dir", str(run_dir))
        check += ["--train-fraction", str(FILLED["TRAIN_FRACTION"])]
        report = subprocess.run(  # noqa: S603 - arguments come from the test
            [*check, "--subset-seed", str(FILLED["SUBSET_SEED"])],
            check=False,
            cwd=RUN_ROOT,
            env=runner_env(),
            capture_output=True,
            text=True,
        )
        assert report.returncode == 1
        assert f"--- failing stage: {later[0]} (stale)\nchoose_layer ran again after it" in (
            report.stdout
        )

        result = run_job(passed_job.model)
        assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-4000:]
        assert "PIPELINE CHECK: PASS" in result.stdout
        for stage in (*later, "choose_layer"):
            assert f"{stage}: outputs exist, skipping" in result.stdout, stage


def test_unknown_force_from_stops_the_job_before_any_stage(passed_job: PassedJob) -> None:
    """A typo must not run the stages unforced, which would skip and read as a successful redo."""
    before = status_mtimes(passed_job.run_dir)
    result = run_job(passed_job.model, "--force-from", "calibration", timeout=30)
    assert result.returncode == 2
    assert "invalid choice: 'calibration'" in result.stderr
    assert "=== start" not in result.stdout
    assert status_mtimes(passed_job.run_dir) == before


def test_failed_stage_stops_the_job(run_root: Path) -> None:
    """ImageNet weights never downloaded: training fails, and nothing may run after it."""
    model = write_config("failing", ENCODER_WEIGHTS="imagenet")
    result = run_job(model)
    assert result.returncode != 0
    assert "failing stage: train (failed)" in result.stdout
    assert f"job exit code {result.returncode}" in result.stdout
    assert {path.stem for path in (run_dir_of(model) / "status").glob("*.json")} == {"train"}
    log = log_of(model).read_text(encoding="utf-8")
    assert "=== end train: exit 1 in " in log
    assert "=== start infer" not in log
    assert "PIPELINE CHECK: FAIL" in log


def test_ctrl_c_stops_the_stage_and_still_runs_the_check(run_root: Path) -> None:
    """Colab's interrupt reaches the runner alone: the stage must stop too, and the check run.

    The signal comes once training prints, inside its stage; the stage's own traceback, printed
    after the signal, must still reach the log.
    """
    model = write_config("interrupted")
    with subprocess.Popen(  # noqa: S603 - arguments come from the tests
        runner_command(SCALE, model, "--num-workers", NUM_WORKERS),
        cwd=RUN_ROOT,
        env=runner_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            if " training on " in line:
                process.send_signal(signal.SIGINT)
                break
        process.communicate(timeout=300)
    assert process.returncode == 130
    log = log_of(model).read_text(encoding="utf-8")
    train = json.loads((run_dir_of(model) / "status" / "train.json").read_text())
    assert train["status"] == "interrupted", log[-8000:]
    assert "KeyboardInterrupt" in log
    assert "=== end train: exit " in log
    assert "interrupted by Ctrl+C" in log
    assert "=== start infer" not in log
    assert "PIPELINE CHECK: FAIL" in log


def refusal(tmp_path: Path, **changes: object) -> str:
    """Return why the configs with `changes` cannot run, as the runner would print it."""
    write_configs(tmp_path, "model", **changes)
    with pytest.raises(garimpo_uncertainty.config.ConfigError) as refused:
        garimpo_uncertainty.config.RunConfig.load(tmp_path, SCALE, "model")
    return str(refused.value)


@pytest.mark.parametrize("name", garimpo_uncertainty.config.REQUIRED)
def test_config_refuses_an_empty_value(tmp_path: Path, name: str) -> None:
    """Every value, left null, refuses the config: the job would otherwise fail after training."""
    assert f"{name}: missing or empty" in refusal(
        tmp_path, **{name: [] if name == "LAYERS" else None}
    )


def test_config_refuses_a_misspelled_or_misplaced_key(tmp_path: Path) -> None:
    """A misspelled key leaves its value at nothing; one in another file has two values."""
    assert "SWEEP_STEP: unknown key" in refusal(tmp_path, SWEEP_STEP=9)
    model = {"ARCH": "unet", "ENCODER_WEIGHTS": "none", "LAYERS": LAYERS, "SVM_C": 2.0}
    (tmp_path / "models" / "model.json").write_text(json.dumps(model))
    with pytest.raises(
        garimpo_uncertainty.config.ConfigError, match=r"SVM_C: belongs in .*base\.json"
    ):
        garimpo_uncertainty.config.RunConfig.load(tmp_path, SCALE, "model")


@pytest.mark.parametrize("floor", [0, -0.01, float("nan"), float("inf"), "0.01ha"])
def test_config_refuses_a_bad_score_floor(tmp_path: Path, floor: object) -> None:
    """`calibrate.py` refuses these too, but only after training."""
    expected = f"SCORE_FLOOR_HA must be a positive finite number, got '{floor}'"
    assert expected in refusal(tmp_path, SCORE_FLOOR_HA=floor)


def test_full_job_refuses_a_config_before_any_stage(run_root: Path) -> None:
    """A refused config stops the job before training, and its reason reaches the log too.

    Colab shows the log's last attempt, so the reason must be there and not only in the cell.
    """
    model = write_config("misspelled", SWEEP_STEP=9, LAYERS=[])
    result = run_job(model, timeout=30)
    assert result.returncode != 0
    for reason in ("SWEEP_STEP: unknown key", "LAYERS: missing or empty"):
        assert reason in result.stdout
        assert reason in log_of(model).read_text(encoding="utf-8")
    assert not (RUN_ROOT / "runs" / "misspelled-f1.0").exists()
    assert "=== start" not in result.stdout


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("model", MODELS)
def test_versioned_configs_are_accepted_and_name_model_layers(scale: str, model: str) -> None:
    """A layer the model lacks would stop fit_directions only after the whole training."""
    config = garimpo_uncertainty.config.RunConfig.load(CONFIGS, scale, model)
    built = garimpo_uncertainty.models.common.build(config["ARCH"], in_channels=6)
    assert set(config["LAYERS"]) <= set(garimpo_uncertainty.models.common.feature_layers(built))


@pytest.mark.parametrize("model", MODELS)
def test_versioned_run_dirs_and_logs_are_those_recorded(model: str) -> None:
    """Recorded runs and their fingerprints name these directories, and Drive keeps these logs."""
    for scale, run_dir in (("full", f"runs/{model}-f1.0"), ("pilot", f"runs/pilot/{model}-f0.01")):
        config = garimpo_uncertainty.config.RunConfig.load(CONFIGS, scale, model)
        assert config.run_dir == run_dir
        assert config.log_name == f"{scale}-{model}"
    assert SCALES == ["full", "pilot"]


def test_full_job_needs_a_scale_and_a_model(run_root: Path) -> None:
    result = subprocess.run(  # noqa: S603 - arguments come from the test
        runner_command(),
        check=False,
        cwd=RUN_ROOT,
        env=runner_env(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "the following arguments are required: scale, model" in result.stderr


def test_status_chooses_the_scale_and_reads_each_last_verdict(tmp_path: Path) -> None:
    """`auto` turns full once a full run saved last.pt; only the last attempt's verdict counts.

    A log not written since the session started was restored from Drive: its run never started.
    """
    shutil.copytree(CONFIGS, tmp_path / "configs")
    failed = "PIPELINE CHECK: FAIL at ['train']"

    def status(*arguments: str) -> str:
        command = support.garimpo("status", *arguments)
        found = subprocess.run(  # noqa: S603 - arguments come from the test
            command, check=True, cwd=tmp_path, capture_output=True, text=True
        )
        return found.stdout

    assert status("auto", "--choose").splitlines() == ["pilot"]
    last = tmp_path / "runs" / "convnet-f1.0" / "last.pt"
    last.parent.mkdir(parents=True)
    last.touch()
    assert status("auto", "--choose").splitlines() == ["full"]
    logs = tmp_path / "outputs" / "logs"
    logs.mkdir(parents=True)
    passed = f"{ATTEMPT}1 ===\n00:00:01 PIPELINE CHECK: PASS (every stage ran)\n"
    (logs / "full-convnet.log").write_text(f"{passed}{ATTEMPT}2 ===\n00:00:02 {failed}\n")
    # The second attempt died before its check: the first one's PASS must not stand for it.
    (logs / "full-unet_imagenet.log").write_text(f"{passed}{ATTEMPT}2 ===\n00:00:02 epoch 1/2\n")
    report = status("full")
    assert f"full-convnet: 00:00:02 {failed}" in report
    assert "full-unet_imagenet: NO VERDICT" in report
    assert "full-unet_scratch: NO LOG" in report
    assert "full-convnet: NOT RUN" in status("full", "--since", str(time.time() + 60))
