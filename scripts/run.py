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
import garimpo_uncertainty.config
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
        default=len(os.sched_getaffinity(0)),
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


def script(name: str, *arguments: str) -> list[str]:
    return [sys.executable, f"scripts/{name}.py", *arguments]


def validate_command(config: garimpo_uncertainty.config.RunConfig, *, force: bool) -> list[str]:
    fraction, seed = config["TRAIN_FRACTION"], config["SUBSET_SEED"]
    command = script(
        "validate_data",
        "--train-fraction",
        str(fraction),
        "--subset-seed",
        str(seed),
        "--report",
        f"outputs/data_validation-fraction-{fraction}-seed-{seed}.json",
        "--split-summary",
        "outputs/split_summary.json",
    )
    if force:
        command.append("--force")
    # For the tests' synthetic store only; a real store has validate_data.py's default.
    if cell_px := os.environ.get("GARIMPO_TEST_CELL_PX"):
        command += ["--cell-px", cell_px]
    return command


class Stages:
    """The commands of a run's stages, in STAGE_ORDER, from its config."""

    def __init__(self, config: garimpo_uncertainty.config.RunConfig, num_workers: int) -> None:
        self.config = config
        self.run = garimpo_uncertainty.artifacts.RunPaths.of(Path(config.run_dir))
        self.common = [
            "--checkpoint",
            str(self.run.best_model),
            "--run-dir",
            config.run_dir,
            "--batch-size",
            self.value("EVAL_BATCH_SIZE"),
            "--num-workers",
            str(num_workers),
        ]
        # The sweep's beta_max is the search range, so it covers the domain the search walks.
        self.sweep = [
            *self.common,
            *self.flags("--beta-max-logits", "BETA_RANGE_LOGITS"),
            *self.flags("--beta-max-features", "BETA_RANGE_FEATURES"),
            *self.flags("--steps", "SWEEP_STEPS"),
            *self.flags("--cells-per-role", "SWEEP_CELLS"),
            *self.flags("--seed", "SEED"),
            *self.flags("--direction-temperature", "DIRECTION_TEMPERATURE"),
        ]
        self.num_workers = str(num_workers)

    def value(self, key: str) -> str:
        return str(self.config[key])

    def flags(self, flag: str, key: str) -> list[str]:
        return [flag, self.value(key)]

    def diagnostics(self) -> list[tuple[str, list[str]]]:
        """Return the stages up to the layer, which choose_layer.py picks on the validation role."""
        weights = garimpo_uncertainty.config.ENCODER_WEIGHTS
        unset = garimpo_uncertainty.config.is_empty(self.config.values.get(weights))
        # Passed whenever set, so train.py alone decides what fits the architecture.
        encoder = [] if unset else self.flags("--encoder-weights", weights)
        layers = ["--layers", *self.config["LAYERS"]]
        train = [
            *self.flags("--arch", "ARCH"),
            *encoder,
            "--run-dir",
            self.config.run_dir,
            *self.flags("--train-fraction", "TRAIN_FRACTION"),
            *self.flags("--subset-seed", "SUBSET_SEED"),
            *self.flags("--epochs", "EPOCHS"),
            *self.flags("--steps-per-epoch", "STEPS_PER_EPOCH"),
            *self.flags("--batch-size", "BATCH_SIZE"),
            *self.flags("--lr", "LR"),
            *self.flags("--weight-decay", "WEIGHT_DECAY"),
            *self.flags("--loss", "LOSS"),
            *self.flags("--seed", "SEED"),
            "--num-workers",
            self.num_workers,
        ]
        directions = [
            *self.common,
            *layers,
            *self.flags("--svm-c", "SVM_C"),
            *self.flags("--svm-class-weight", "SVM_CLASS_WEIGHT"),
            *self.flags("--svm-pixels-per-cell", "SVM_PIXELS_PER_CELL"),
            *self.flags("--seed", "SEED"),
            *self.flags("--direction-temperature", "DIRECTION_TEMPERATURE"),
        ]
        choose = [
            "--run-dir",
            self.config.run_dir,
            *self.flags("--min-fraction-ascending", "MIN_FRACTION_ASCENDING"),
        ]
        return [
            ("train", script("train", *train)),
            ("infer", script("infer", *self.common)),
            ("fit_directions", script("fit_directions", *directions)),
            (
                "layer_monotonicity",
                script("verify_monotonicity", *self.sweep, "--use", "layer", *layers),
            ),
            ("choose_layer", script("choose_layer", *choose)),
        ]

    def calibrated(self, layer: str) -> list[tuple[str, list[str]]]:
        """Return the stages after the layer choice, each on the chosen layer."""
        endpoint = script(
            "verify_monotonicity", *self.sweep, "--use", "endpoint", "--layers", layer
        )
        return [("endpoint_monotonicity", endpoint), *self.on_layer(layer)]

    def other_layers(self, chosen: str) -> list[tuple[str, list[str]]]:
        """Return the layer's stages on every other layer of LAYERS, each into layers/<layer>/.

        Cheung et al. (2026), Table 4, compare the interval width across layers. The chosen layer's
        files stay where they were, so nothing reading a run's results changes.
        """
        return [
            stage
            for layer in self.config["LAYERS"]
            if layer != chosen
            for stage in self.on_layer(
                layer, "--layer-dir", str(self.run.other_layer(layer).layer_root)
            )
        ]

    def on_layer(self, layer: str, *where: str) -> list[tuple[str, list[str]]]:
        """Return the stages of LAYER_STAGES on one layer; `where` sends their files elsewhere."""
        scores = [
            *self.common,
            "--layer",
            layer,
            *self.flags("--beta-range-logits", "BETA_RANGE_LOGITS"),
            *self.flags("--beta-range-features", "BETA_RANGE_FEATURES"),
            *self.flags("--search-threshold-logits", "SEARCH_THRESHOLD_LOGITS"),
            *self.flags("--search-threshold-features", "SEARCH_THRESHOLD_FEATURES"),
            *self.flags("--direction-temperature", "DIRECTION_TEMPERATURE"),
        ]
        calibrate = [
            "--run-dir",
            self.config.run_dir,
            *self.flags("--cells", "CELLS"),
            *self.flags("--score-floor-ha", "SCORE_FLOOR_HA"),
            *self.flags("--seed", "SEED"),
            "--lightgbm-jobs",
            self.num_workers,
        ]
        metrics = [
            "--run-dir",
            self.config.run_dir,
            *self.flags("--boundary-tolerance-px", "BOUNDARY_TOLERANCE_PX"),
            *self.flags("--seed", "SEED"),
        ]
        intervals = [*self.common, "--layer", layer]
        return [
            ("compass_scores", script("compass_scores", *scores, *where)),
            ("calibrate", script("calibrate", *calibrate, *where)),
            ("generate_intervals", script("generate_intervals", *intervals, *where)),
            ("compute_metrics", script("compute_metrics", *metrics, *where)),
        ]


def run_in_order(log: Log, commands: list[tuple[str, list[str]]], force_from: str | None) -> int:
    """Run each stage until one fails; return its exit code, or 0."""
    for name, command in commands:
        forced = force_from is not None and (
            STAGE_ORDER.index(name) >= STAGE_ORDER.index(force_from)
        )
        if code := log.run(name, [*command, "--force"] if forced else command):
            return code
    return 0


def run_stages(log: Log, stages: Stages, force_from: str | None) -> int:
    """Run the stages up to the layer choice, the others on the chosen layer, then other layers.

    The chosen layer's results come first, so a session lost during the other layers keeps them.
    """
    if code := run_in_order(log, stages.diagnostics(), force_from):
        return code
    chosen = stages.run.chosen_layer
    try:
        layer = json.loads(chosen.read_text(encoding="utf-8"))["layer"]
    except (OSError, ValueError, KeyError) as error:
        log.write(f"Could not read the chosen layer from {chosen}: {error!r}")
        return 1
    log.write(f"chosen layer: {layer}")
    return run_in_order(log, stages.calibrated(layer), force_from) or run_in_order(
        log, stages.other_layers(layer), force_from
    )


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
    log = Log(Path("outputs/logs") / f"{args.scale}-{args.model}.log")
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

    try:
        code = log.run(
            "validate_data", validate_command(config, force=args.force_validate)
        ) or run_stages(log, Stages(config, args.num_workers), args.force_from)
    except KeyboardInterrupt:
        log.write("interrupted by Ctrl+C; run again to resume")
        code = 130
    # The exit code goes to the check because an error before run_stage leaves every status ok.
    check = log.run(
        "pipeline_check",
        script(
            "pipeline_check",
            "--run-dir",
            config.run_dir,
            "--train-fraction",
            str(config["TRAIN_FRACTION"]),
            "--subset-seed",
            str(config["SUBSET_SEED"]),
            "--job-exit-code",
            str(code),
            "--layers",
            *config["LAYERS"],
        ),
    )
    # A stage's exit code wins; a failed check alone still fails the job.
    sys.exit(code or check)


if __name__ == "__main__":
    main()
