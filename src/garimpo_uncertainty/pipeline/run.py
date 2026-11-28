"""Run one model through every stage at one scale, every value from configs/ (docs/decisions.md).

    garimpo run pilot unet_scratch

The full scale trains on 100% of the training tiles; the pilot trains the same models on 1%, with a
short training, to check the code first. Run it from the repository root, one run per GPU.

Each stage runs in its own process. The terminal shows one numbered line per stage with its
duration and key numbers, and while it runs its epochs or progress; a failed stage adds its last
lines. The log, outputs/logs/<scale>-<model>.log, holds all that and every other line, each stage's
command included, all with the UTC time; while a stage runs, a line every 30 seconds adds its time
so far and the GPU's use, so a session that ends mid-stage leaves when and where it stopped.
Each attempt is appended under the earlier ones. Finished stages skip themselves, and the pipeline
check runs last, after a failed or interrupted stage too.
"""

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import NoReturn

import garimpo_uncertainty.artifacts
import garimpo_uncertainty.config
import garimpo_uncertainty.data
import garimpo_uncertainty.files
import garimpo_uncertainty.pipeline.commands
import garimpo_uncertainty.pipeline.runner

STAGE_ORDER = garimpo_uncertainty.pipeline.runner.STAGE_ORDER
LAYER_STAGES = garimpo_uncertainty.pipeline.runner.LAYER_STAGES
# Up to choose_layer, then on the chosen layer: endpoint_monotonicity and LAYER_STAGES.
DIAGNOSTICS = STAGE_ORDER[: STAGE_ORDER.index("choose_layer") + 1]
CALIBRATED = ("endpoint_monotonicity", *LAYER_STAGES)
# The longest stage name, endpoint_monotonicity, so the stages' lines align.
NAME_WIDTH = 21
# The lines of a failed stage the terminal repeats; the log has them all.
FAILURE_LINES = 30
# A session lost mid-stage leaves, as its last alive line, when and where it stopped.
HEARTBEAT_SECONDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("scale", help="a file of configs/scales/, such as pilot or full")
    parser.add_argument("model", help="a file of configs/models/, such as unet_scratch")
    parser.add_argument(
        "--force-from",
        choices=STAGE_ORDER,
        help="redo this stage and those after it, after a code fix; train runs only when named",
    )
    parser.add_argument("--force-validate", action="store_true", help="check the data again")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=garimpo_uncertainty.pipeline.runner.available_cpus(),
        help="data loader processes and LightGBM threads; default: the CPUs this process may use",
    )
    return parser.parse_args()


class Log:
    """Write each line with the UTC time to the run's log, and the shown ones to the terminal."""

    def __init__(self, path: Path, total: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.total = total
        self.started = 0
        self.passed = 0
        self.label = ""
        self.outcome = ""
        self.tail: deque[str] = deque(maxlen=FAILURE_LINES)
        # Held while an alive line is written, so none follows its stage's end.
        self.beating = threading.Lock()

    def write(self, text: str, *, shown: bool = True) -> None:
        line = f"{time.strftime('%H:%M:%S', time.gmtime())} {text}"
        if shown:
            print(line, flush=True)
        try:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
        except OSError as error:
            # A full disk must not stop the run: the terminal still shows what matters.
            print(f"could not append to {self.path}: {error}", flush=True)

    def copy_lines(self, process: subprocess.Popen) -> None:
        runner = garimpo_uncertainty.pipeline.runner
        for raw in process.stdout:
            line = raw.rstrip("\n")
            self.tail.append(line)
            if line.startswith(runner.RESULT):
                self.outcome = line.removeprefix(runner.RESULT)
            elif line.startswith(runner.SHOWN):
                self.write(f"{self.label} {line.removeprefix(runner.SHOWN)}")
            else:
                self.write(line, shown=False)

    def run(self, name: str, command: list[str]) -> int:
        """Run one stage, logging its lines as they come; return its exit code."""
        self.started += 1
        self.label = f"[{self.started:>2}/{self.total}] {name:<{NAME_WIDTH}}"
        self.outcome = ""
        self.write(f"=== start {name}: {shlex.join(command)}", shown=False)
        self.tail.clear()
        start = time.monotonic()
        # Unbuffered, or a stage's lines would reach the log only when its process ends.
        env = os.environ | {"PYTHONUNBUFFERED": "1"}
        stopped = threading.Event()
        threading.Thread(target=self.beat, args=(name, start, stopped), daemon=True).start()
        try:
            with subprocess.Popen(  # noqa: S603 - the commands below, from a checked config
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                env=env,
            ) as process:
                try:
                    self.copy_lines(process)
                except KeyboardInterrupt:
                    self.stop(process)
                    self.silence(stopped)
                    self.ended(name, process, start)
                    raise
        finally:
            self.silence(stopped)
        return self.ended(name, process, start)

    def beat(self, name: str, start: float, stopped: threading.Event) -> None:
        while not stopped.wait(HEARTBEAT_SECONDS):
            took = garimpo_uncertainty.pipeline.runner.duration(time.monotonic() - start)
            # Read before the lock: a slow nvidia-smi must not hold up the stage's end.
            line = f"=== alive {name}: {took}{gpu_use()}"
            with self.beating:
                if not stopped.is_set():
                    self.write(line, shown=False)

    def silence(self, stopped: threading.Event) -> None:
        """Stop the stage's alive lines; an alive line being written finishes first."""
        with self.beating:
            stopped.set()

    def stop(self, process: subprocess.Popen) -> None:
        """Stop the stage after Ctrl+C and keep its last lines; a second interrupt kills it."""
        # Colab's interrupt reaches this process alone; the stage must stop as well.
        process.send_signal(signal.SIGINT)
        try:
            # Read on: a stage printing more than the pipe holds would block and never end.
            self.copy_lines(process)
        except KeyboardInterrupt:
            process.kill()
        process.wait()

    def ended(self, name: str, process: subprocess.Popen, start: float) -> int:
        """Log the stage's exit code and duration; return the code, positive after a signal."""
        seconds = time.monotonic() - start
        self.write(f"=== end {name}: exit {process.returncode} in {seconds:.1f} s", shown=False)
        took = garimpo_uncertainty.pipeline.runner.duration(seconds)
        if process.returncode == 0:
            self.passed += 1
            self.write(
                f"{self.label} ok {took:>7}" + (f" · {self.outcome}" if self.outcome else "")
            )
            return 0
        self.write(f"{self.label} FAILED after {took}, exit {process.returncode}; its last lines:")
        for line in self.tail:
            self.write(f"    | {line}")
        self.write(f"    the whole output is in {self.path}")
        # A stage killed by signal N reports -N; the shell's 128 + N keeps the job's code positive.
        return process.returncode if process.returncode >= 0 else 128 - process.returncode


def gpu_use() -> str:
    """Return the GPU's use and memory as nvidia-smi gives them, or nothing without a GPU."""
    if (nvidia_smi := shutil.which("nvidia-smi")) is None:
        return ""
    query = ["--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"]
    try:
        done = subprocess.run(  # noqa: S603 - a fixed query
            [nvidia_smi, *query], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    gpus = "; ".join(line.strip() for line in done.stdout.splitlines() if line.strip())
    return f" · GPU {gpus}" if done.returncode == 0 and gpus else ""


def command(stage: str, values: dict, *extra: str) -> list[str]:
    command = garimpo_uncertainty.pipeline.commands.STAGES[stage].command
    arguments = garimpo_uncertainty.pipeline.commands.arguments(stage, values)
    return [sys.executable, "-m", "garimpo_uncertainty", command, *arguments, *extra]


def job_values(config: garimpo_uncertainty.config.RunConfig, num_workers: int) -> dict:
    """Return the config's values and the job's own: where the run is, and how many workers."""
    selection = garimpo_uncertainty.data.selection_name(
        config["TRAIN_FRACTION"], config["SUBSET_SEED"], config["EVALUATION_TILES"]
    )
    return config.values | {
        garimpo_uncertainty.config.ENCODER_WEIGHTS: config.values.get(
            garimpo_uncertainty.config.ENCODER_WEIGHTS
        ),
        "RUN_DIR": config.run_dir,
        "CHECKPOINT": garimpo_uncertainty.artifacts.RunPaths.of(Path(config.run_dir)).best_model,
        "NUM_WORKERS": num_workers,
        "VALIDATION_REPORT": f"outputs/data_validation-{selection}.json",
        "SPLIT_SUMMARY": "outputs/split_summary.json",
        "LAYER_DIR": None,
    }


def run_in_order(log: Log, stages: tuple[str, ...], values: dict, force_from: str | None) -> int:
    """Run each stage until one fails; return its exit code, or 0."""
    for name in stages:
        forced = force_from is not None and (
            STAGE_ORDER.index(name) >= STAGE_ORDER.index(force_from)
        )
        words = ["--force"] if forced else []
        # A pilot's training only checks the code, so its forced redo drops even an unfinished one.
        if forced and name == "train" and values["PILOT"]:
            words.append("--discard-unfinished")
        if code := log.run(name, command(name, values, *words)):
            return code
    return 0


def stage_count(layers: int) -> int:
    """Return how many stages a job runs, the data check and the pipeline check included."""
    return 1 + len(DIAGNOSTICS) + len(CALIBRATED) + len(LAYER_STAGES) * (layers - 1) + 2


def write_heading(log: Log, args: argparse.Namespace, config: dict) -> None:
    """Say what this attempt trains and which layers it compares."""
    log.write(
        f"train: {float(config['TRAIN_FRACTION']):.0%} of the tiles · {config['EPOCHS']} epochs x "
        f"{config['STEPS_PER_EPOCH']} updates x {config['BATCH_SIZE']} cells · lr {config['LR']} · "
        f"{args.num_workers} workers"
    )
    log.write(f"layers: {', '.join(config['LAYERS'])}; the best one first, then the others")
    if args.force_from or args.force_validate:
        log.write(f"forced: from {args.force_from or 'no stage'}; data check {args.force_validate}")


def run_stages(log: Log, values: dict, force_from: str | None) -> int:
    """Run the stages up to the layer choice, the chosen layer's, the other layers', the profile.

    The chosen layer's results come first, so a session lost during the other layers keeps them.
    Cheung et al. (2026), Table 4, compare the interval width across layers; the chosen layer's
    files stay at the run's root, so nothing reading a run's results changes.
    """
    if code := run_in_order(log, DIAGNOSTICS, values, force_from):
        return code
    run = garimpo_uncertainty.artifacts.RunPaths.of(Path(values["RUN_DIR"]))
    try:
        layer = garimpo_uncertainty.artifacts.read_chosen_layer(run.chosen_layer)["layer"]
    except garimpo_uncertainty.artifacts.UNREADABLE as error:
        log.write(f"Could not read the chosen layer from {run.chosen_layer}: {error!r}")
        return 1
    log.write(f"── chosen layer: {layer} ──")
    return (
        run_in_order(log, CALIBRATED, values | {"LAYER": layer}, force_from)
        or run_other_layers(log, values, layer, force_from)
        or run_in_order(log, ("search_profile",), values, profile_force(force_from))
    )


def profile_force(force_from: str | None) -> str | None:
    """Redo the profile only from a stage it reads: never for calibration, intervals or metrics."""
    return None if force_from in LAYER_STAGES[1:] else force_from


def run_other_layers(log: Log, values: dict, chosen: str, force_from: str | None) -> int:
    """Run the layer's stages on every other layer of LAYERS, each into layers/<layer>/."""
    run = garimpo_uncertainty.artifacts.RunPaths.of(Path(values["RUN_DIR"]))
    others = [name for name in values["LAYERS"] if name != chosen]
    for number, layer in enumerate(others, start=1):
        log.write(f"── other layer {number}/{len(others)}: {layer} ──")
        where = {"LAYER": layer, "LAYER_DIR": run.other_layer(layer).layer_root}
        if code := run_in_order(log, LAYER_STAGES, values | where, force_from):
            return code
    return 0


def commit() -> str:
    try:
        found = subprocess.run(
            ["git", "log", "-1", "--format=%h %cs %s"],  # noqa: S607 - git from PATH
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    return found.stdout.strip() or "unknown"


def interrupt(_signum: int, _frame: object) -> NoReturn:
    raise KeyboardInterrupt


def start_attempt(args: argparse.Namespace) -> tuple["Log", garimpo_uncertainty.config.RunConfig]:
    """Open the log under a new attempt and say what it runs; exit if the config is refused."""
    name = garimpo_uncertainty.config.log_name(args.scale, args.model)
    try:
        config = garimpo_uncertainty.config.RunConfig.load(Path("configs"), args.scale, args.model)
    except garimpo_uncertainty.config.ConfigError as error:
        config, problems = None, error.problems
    log = Log(
        Path("outputs/logs") / f"{name}.log", stage_count(len(config["LAYERS"])) if config else 0
    )
    started = time.strftime("%Y-%m-%d %H:%M", time.gmtime())
    log.write(f"{garimpo_uncertainty.pipeline.runner.ATTEMPT}{started} UTC ===")
    log.write(f"{args.model} · {args.scale} · {log.total} stages · commit {commit()}")
    if config is None:
        for line in problems:
            log.write(line)
        sys.exit(1)
    write_heading(log, args, config.values)
    return log, config


# The pilot only checks the code, so a pilot not trained to the end by this code is redone whole.
PILOT = "pilot"


def trained_here(status: dict) -> bool:
    """Tell whether train's status is a finished training by the code checked out, unedited."""
    now = garimpo_uncertainty.files.code_versions()
    clean = not any(now[key] or status.get(key) for key in ("root_dirty", "compass_dirty"))
    commits = all(status.get(key) == now[key] for key in ("root_commit", "compass_commit"))
    return status.get("status") == "ok" and clean and commits


def redo_from(args: argparse.Namespace, config: garimpo_uncertainty.config.RunConfig) -> str | None:
    """Return --force-from, or train for a pilot whose training did not finish on this code.

    A stage that fails rewrites its status with the code of the moment, so only a finished
    training's status names the code that trained it.
    """
    if args.force_from or args.scale != PILOT:
        return args.force_from
    status = garimpo_uncertainty.artifacts.RunPaths.of(Path(config.run_dir)).status
    trained = garimpo_uncertainty.pipeline.runner.read_status(status, "train")
    return None if trained_here(trained) else "train"


def run_job(
    log: Log, args: argparse.Namespace, config: garimpo_uncertainty.config.RunConfig, values: dict
) -> int:
    """Check the data, then run every stage; return the first nonzero exit code, or 0."""
    force_from = redo_from(args, config)
    if force_from != args.force_from:
        log.write(f"the pilot's training did not finish on this code: running it from {force_from}")
    validate = command("validate_data", values, *(["--force"] if args.force_validate else []))
    return log.run("validate_data", validate) or run_stages(log, values, force_from)


def main() -> None:
    # Colab's second interrupt is SIGTERM, and SIGKILL 0.5 s later: kill the stage before that.
    signal.signal(signal.SIGTERM, interrupt)
    args = parse_args()
    start = time.monotonic()
    log, config = start_attempt(args)
    values = job_values(config, args.num_workers) | {"PILOT": args.scale == PILOT}
    interrupted = False
    try:
        code = run_job(log, args, config, values)
    except KeyboardInterrupt:
        interrupted, code = True, 130
    finally:
        # The verdict is the attempt's record, so an interrupt from here on, such as Colab's
        # SIGTERM after the stage already stopped, is ignored. A handler, unlike SIG_IGN, is not
        # passed on: a terminal's Ctrl+C to the whole group still stops the check, which then fails.
        for kind in (signal.SIGINT, signal.SIGTERM):
            signal.signal(kind, lambda *_: None)
    if interrupted:
        log.write("interrupted by Ctrl+C; run again to resume")
    # The exit code goes to the check because an error before run_stage leaves every status ok.
    check = log.run("pipeline_check", command("pipeline_check", values | {"JOB_EXIT_CODE": code}))
    took = garimpo_uncertainty.pipeline.runner.duration(time.monotonic() - start)
    verdict = "PASS" if code == check == 0 else "FAIL"
    log.write(f"=== {args.model}: {log.passed}/{log.total} stages ok in {took} · {verdict}")
    # Overwritten each attempt, never deleted: a deleted file comes back with the next restore.
    garimpo_uncertainty.files.write_json(
        garimpo_uncertainty.artifacts.RunPaths.of(Path(config.run_dir)).verdict,
        {"verdict": verdict, **garimpo_uncertainty.files.code_versions()},
    )
    # A stage's exit code wins; a failed check alone still fails the job.
    sys.exit(code or check)
