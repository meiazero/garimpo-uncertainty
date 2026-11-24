"""Run one model through every stage at one scale, every value from configs/ (docs/decisions.md).

    uv run --no-sync python scripts/run.py pilot unet_scratch

The full scale trains on 100% of the training tiles; the pilot trains the same models on 1%, with a
short training, to check the code first. Run it from the repository root, one run per GPU.

Each stage runs in its own process. Every line it prints, and each stage's command, exit code and
duration, goes to the terminal and to outputs/logs/<scale>-<model>.log, with the UTC time. Each
attempt is appended under the earlier ones. Finished stages skip themselves, and the pipeline
check runs last, after a failed or interrupted stage too.
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn

import garimpo_uncertainty.artifacts
import garimpo_uncertainty.commands
import garimpo_uncertainty.config
import garimpo_uncertainty.data
import garimpo_uncertainty.stages

STAGE_ORDER = garimpo_uncertainty.stages.STAGE_ORDER
# The run notebook's print_verdicts splits the log on this marker to read the last attempt only.
ATTEMPT = "=== attempt started "


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
        default=garimpo_uncertainty.stages.available_cpus(),
        help="data loader processes and LightGBM threads; default: the CPUs this process may use",
    )
    return parser.parse_args()


class Log:
    """Print each line with the UTC time and append it to the run's log file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, text: str) -> None:
        line = f"{time.strftime('%H:%M:%S', time.gmtime())} {text}"
        print(line, flush=True)
        try:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
        except OSError as error:
            # A full disk must not stop the run: the terminal still shows every line.
            print(f"could not append to {self.path}: {error}", flush=True)

    def copy_lines(self, process: subprocess.Popen) -> None:
        for line in process.stdout:
            self.write(line.rstrip("\n"))

    def run(self, name: str, command: list[str]) -> int:
        """Run one stage, logging its lines as they come; return its exit code."""
        self.write(f"=== start {name}: {shlex.join(command)}")
        start = time.monotonic()
        # Unbuffered, or a stage's lines would reach the log only when its process ends.
        env = os.environ | {"PYTHONUNBUFFERED": "1"}
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
                self.ended(name, process, start)
                raise
        return self.ended(name, process, start)

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
        self.write(f"=== end {name}: exit {process.returncode} in {seconds:.1f} s")
        # A stage killed by signal N reports -N; the shell's 128 + N keeps the job's code positive.
        return process.returncode if process.returncode >= 0 else 128 - process.returncode


def command(stage: str, values: dict, *extra: str) -> list[str]:
    module = garimpo_uncertainty.commands.STAGES[stage].module
    arguments = garimpo_uncertainty.commands.arguments(stage, values)
    return [sys.executable, f"scripts/{module}.py", *arguments, *extra]


def job_values(config: garimpo_uncertainty.config.RunConfig, num_workers: int) -> dict:
    """Return the config's values and the job's own: where the run is, and how many workers."""
    selection = garimpo_uncertainty.data.selection_name(
        config["TRAIN_FRACTION"], config["SUBSET_SEED"]
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
        if code := log.run(name, command(name, values, *(["--force"] if forced else []))):
            return code
    return 0


def run_stages(log: Log, values: dict, force_from: str | None) -> int:
    """Run the stages up to the layer choice, the others on the chosen layer, then other layers.

    The chosen layer's results come first, so a session lost during the other layers keeps them.
    Cheung et al. (2026), Table 4, compare the interval width across layers; the chosen layer's
    files stay at the run's root, so nothing reading a run's results changes.
    """
    diagnostics = STAGE_ORDER[: STAGE_ORDER.index("choose_layer") + 1]
    if code := run_in_order(log, diagnostics, values, force_from):
        return code
    run = garimpo_uncertainty.artifacts.RunPaths.of(Path(values["RUN_DIR"]))
    try:
        layer = json.loads(run.chosen_layer.read_text(encoding="utf-8"))["layer"]
    except (OSError, ValueError, KeyError) as error:
        log.write(f"Could not read the chosen layer from {run.chosen_layer}: {error!r}")
        return 1
    log.write(f"chosen layer: {layer}")
    calibrated = ("endpoint_monotonicity", *garimpo_uncertainty.stages.LAYER_STAGES)
    return run_in_order(log, calibrated, values | {"LAYER": layer}, force_from) or (
        run_other_layers(log, values, layer, force_from)
    )


def run_other_layers(log: Log, values: dict, chosen: str, force_from: str | None) -> int:
    """Run the layer's stages on every other layer of LAYERS, each into layers/<layer>/."""
    run = garimpo_uncertainty.artifacts.RunPaths.of(Path(values["RUN_DIR"]))
    for layer in (name for name in values["LAYERS"] if name != chosen):
        where = {"LAYER": layer, "LAYER_DIR": run.other_layer(layer).layer_root}
        stages = garimpo_uncertainty.stages.LAYER_STAGES
        if code := run_in_order(log, stages, values | where, force_from):
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


def main() -> None:
    # Colab's second interrupt is SIGTERM, and SIGKILL 0.5 s later: kill the stage before that.
    signal.signal(signal.SIGTERM, interrupt)
    args = parse_args()
    log = Log(
        Path("outputs/logs") / f"{garimpo_uncertainty.config.log_name(args.scale, args.model)}.log"
    )
    log.write(f"{ATTEMPT}{time.strftime('%Y-%m-%d %H:%M', time.gmtime())} UTC ===")
    log.write(f"config {args.scale} {args.model}; commit {commit()}")
    log.write(
        f"force from {args.force_from or 'nothing'}; force validate {args.force_validate}; "
        f"{args.num_workers} workers"
    )
    try:
        config = garimpo_uncertainty.config.RunConfig.load(Path("configs"), args.scale, args.model)
    except garimpo_uncertainty.config.ConfigError as error:
        for line in error.problems:
            log.write(line)
        sys.exit(1)

    values = job_values(config, args.num_workers)
    try:
        validate = command("validate_data", values, *(["--force"] if args.force_validate else []))
        code = log.run("validate_data", validate) or run_stages(log, values, args.force_from)
    except KeyboardInterrupt:
        log.write("interrupted by Ctrl+C; run again to resume")
        code = 130
    # The exit code goes to the check because an error before run_stage leaves every status ok.
    check = log.run("pipeline_check", command("pipeline_check", values | {"JOB_EXIT_CODE": code}))
    # A stage's exit code wins; a failed check alone still fails the job.
    sys.exit(code or check)


if __name__ == "__main__":
    main()
