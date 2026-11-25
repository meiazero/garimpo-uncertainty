"""Runs each pipeline stage once: skip finished work, record status, timing, shapes, peak memory.

No torch import here, so CPU-only stages stay free of it.
"""

import json
import logging
import os
import resource
import socket
import sys
import time
import traceback
from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NoReturn

import garimpo_uncertainty.artifacts
import garimpo_uncertainty.files

logger = logging.getLogger(__name__)

# `report` names an output: another job's path for the same selection must redo it, not refuse.
# `layer_dir` holds the status itself, so no fingerprint is ever compared across two of them.
IGNORED_ARGUMENTS = {
    "force",
    "device",
    "num_workers",
    "batch_size",
    "workers",
    "lightgbm_jobs",
    "report",
    "layer_dir",
}
MISSING_INPUT = "missing"
NOT_RUN = "not_run"
# A job's log lines that `garimpo status` reads back: each attempt starts with ATTEMPT, and the
# check ends it with one of VERDICTS.
ATTEMPT = "=== attempt started "
PASS, FAIL = "PIPELINE CHECK: PASS", "PIPELINE CHECK: FAIL"
VERDICTS = (PASS, FAIL)
# A stage's lines that `garimpo run` shows in the terminal, as it runs and once it ends; every other
# line goes to the log only.
SHOWN, RESULT = "» ", "»= "
MINUTE, HOUR = 60, 3600
PROGRESS_EVERY_SECONDS = MINUTE
PROGRESS_BAR = 20
# The one stage every fraction shares, with its status in the data root instead of a run.
DOWNLOAD_SCENES = "download_scenes"
# The stages that depend on the COMPASS-J layer; a job also runs them on each other layer.
LAYER_STAGES = ("compass_scores", "calibrate", "generate_intervals", "compute_metrics")
# The order a job runs a model's stages in, after validate_data; --force-from names one.
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


def show(text: str) -> None:
    """Print a line `garimpo run` shows while the stage runs: an epoch or the progress."""
    print(f"{SHOWN}{text}", flush=True)


def result(text: str) -> None:
    """Print the stage's outcome in a few numbers, which `garimpo run` puts on its closing line."""
    print(f"{RESULT}{text}", flush=True)


def duration(seconds: float) -> str:
    """Return a duration as people read it: 35.4 s, 13m45s, 1h02m."""
    whole = round(seconds)
    if whole < MINUTE:
        return f"{seconds:.1f} s"
    if whole < HOUR:
        return f"{whole // MINUTE}m{whole % MINUTE:02d}s"
    return f"{whole // HOUR}h{whole % HOUR // MINUTE:02d}m"


class Progress:
    """Show how far a long loop is, at most once a minute, with the time left at its pace so far."""

    def __init__(self, what: str, total: int) -> None:
        self.what, self.total, self.done = what, total, 0
        self.start = self.last = time.monotonic()

    def advance(self, count: int) -> None:
        self.done += count
        now = time.monotonic()
        if now - self.last < PROGRESS_EVERY_SECONDS or self.done >= self.total:
            return
        self.last = now
        elapsed = now - self.start
        left = elapsed / self.done * (self.total - self.done)
        filled = PROGRESS_BAR * self.done // self.total
        bar = "=" * filled + ">" + "." * (PROGRESS_BAR - filled - 1)
        show(
            f"{self.done}/{self.total} {self.what} [{bar}] {self.done / self.total:.0%} · "
            f"{duration(elapsed)} · eta {duration(left)}"
        )


def read_status(status_dir: Path, stage: str) -> dict:
    """Return a stage's last status record, or one saying it never ran."""
    path = status_dir / f"{stage}.json"
    return json.loads(path.read_text()) if path.exists() else {"stage": stage, "status": NOT_RUN}


def available_cpus() -> int:
    """Return the CPUs this process may use: the default of every stage's workers."""
    return len(os.sched_getaffinity(0))


def add_layer_dir(parser: ArgumentParser) -> None:
    """Add --layer-dir to a stage of LAYER_STAGES; RunPaths falls back to --run-dir without it."""
    parser.add_argument(
        "--layer-dir",
        type=Path,
        help="where this layer's scores, calibration, intervals, metrics and status go, for a "
        "layer other than the chosen one (Cheung et al., 2026, Table 4); default: --run-dir",
    )


def log_to_stdout(log_file: Path | None = None) -> None:
    """Show this package's messages as bare stdout lines, and append them timestamped to `log_file`.

    Only this package: INFO on the root logger would add the dependencies' messages to the job log.
    Handlers are replaced, not added, so a main run twice in one process does not double each line.
    """
    package = logging.getLogger("garimpo_uncertainty")
    for handler in package.handlers:
        handler.close()
    package.handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        to_file = logging.FileHandler(log_file)
        to_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        package.handlers.append(to_file)
    package.setLevel(logging.INFO)
    package.propagate = False


def fingerprint(
    args: Namespace,
    inputs: list[Path],
    keep: Iterable[str] = (),
    hashed: dict[str, str | None] | None = None,
) -> dict:
    """Arguments that change a result plus the identity of input files, comparable after JSON.

    `hashed` holds inputs known by content, with None when absent: the shared store, of which a run
    reads only its own slots. Commits stay out on purpose: a pull must not invalidate training, and
    after a code fix --force-from redoes the stage and those after it by name.
    """
    ignored = IGNORED_ARGUMENTS - set(keep)
    arguments = {k: v for k, v in vars(args).items() if k not in ignored}
    files = {
        str(p): f"{p.stat().st_size}:{p.stat().st_mtime_ns}" if p.exists() else MISSING_INPUT
        for p in inputs
    }
    files |= {path: value or MISSING_INPUT for path, value in (hashed or {}).items()}
    return json.loads(json.dumps({"arguments": arguments, "inputs": files}, default=str))


# Restarting torch's count per section would otherwise shrink the peak the stage reports.
GPU_PEAK = {"highest_bytes": 0}


def take_gpu_peak_bytes() -> int | None:
    """Return the GPU peak since the last call and restart the count; None if no GPU is in use."""
    torch = sys.modules.get("torch")
    # Not is_available: a `--device cpu` run on a GPU host would record a peak of zero bytes.
    if torch is None or not torch.cuda.is_initialized():
        return None
    peak = torch.cuda.max_memory_allocated()
    GPU_PEAK["highest_bytes"] = max(GPU_PEAK["highest_bytes"], peak)
    torch.cuda.reset_peak_memory_stats()
    return peak


def _peak_memory() -> dict:
    # ru_maxrss is in KiB on Linux; children covers DataLoader workers.
    rss = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
    peak = {"peak_rss_gb": round(rss / 1024**2, 3)}
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        highest = max(GPU_PEAK["highest_bytes"], torch.cuda.max_memory_allocated())
        peak["peak_gpu_gb"] = round(highest / 1024**3, 3)
    return peak


def _identity(name: str) -> dict:
    # The code each stage ran, since a session may pull between stages of one run.
    return {
        "stage": name,
        "host": socket.gethostname(),
        **garimpo_uncertainty.files.code_versions(),
    }


def fail_stage(status_dir: Path, name: str, message: str) -> NoReturn:
    """Record a failure found before a stage's work, and stop.

    The last completed fingerprint stays, so fixing the cause lets a finished stage skip again.
    """
    completed = read_status(status_dir, name).get("completed_fingerprint")
    kept = {"completed_fingerprint": completed} if completed else {}
    garimpo_uncertainty.files.write_json(
        status_dir / f"{name}.json", _identity(name) | kept | {"status": "failed", "error": message}
    )
    raise SystemExit(message)


def _reusable(outputs: list[Path], completed: dict | None, *, force: bool) -> bool:
    """Tell whether finished outputs exist that this run must either skip over or refuse."""
    if force or not outputs or not completed:
        return False
    return all(p.exists() for p in outputs)


def _skip_or_refuse(status_path: Path, name: str, previous: dict, stage_fingerprint: dict) -> None:
    """Skip when the finished outputs came from these inputs; otherwise stop the stage."""
    if previous["completed_fingerprint"] == stage_fingerprint:
        if previous["status"] != "ok":
            restored = {k: v for k, v in previous.items() if k != "error"}
            garimpo_uncertainty.files.write_json(status_path, restored | {"status": "ok"})
        logger.info("%s: outputs exist, skipping (use --force to redo)", name)
        result("skipped: done before with the same inputs")
        return
    # The validation sits outside STAGE_ORDER.
    redo = (
        "--force-validate to redo it"
        if name == "validate_data"
        else f"--force-from {name} to redo this stage and those after it"
    )
    message = f"{name}: outputs exist from other inputs or arguments; use --force, or {redo}"
    garimpo_uncertainty.files.write_json(
        status_path, previous | {"status": "refused", "error": message}
    )
    raise SystemExit(message)


def _mark_later_stale(status_dir: Path, name: str) -> None:
    """Turn every later ok status to stale, since those stages read what this one is about to redo.

    Done before the work, so a job that dies after it leaves no later stage reading ok, also in a
    copy to Drive. The completed fingerprint stays: a later stage whose inputs did not change skips.
    A stage before the layer's ones also marks every other layer's stages, which read its files.
    """
    if name not in STAGE_ORDER:
        return
    later = STAGE_ORDER[STAGE_ORDER.index(name) + 1 :]
    status_dirs = [status_dir]
    if STAGE_ORDER.index(name) < STAGE_ORDER.index(LAYER_STAGES[0]):
        status_dirs += sorted(
            (status_dir.parent / garimpo_uncertainty.artifacts.LAYERS_DIR).glob("*/status")
        )
    for directory, stage in ((d, s) for d in status_dirs for s in later):
        record = read_status(directory, stage)
        if record["status"] == "ok":
            garimpo_uncertainty.files.write_json(
                directory / f"{stage}.json",
                record | {"status": "stale", "error": f"{name} ran again after it"},
            )


def _run_work(
    status_path: Path, name: str, stage_fingerprint: dict, work: Callable[[], dict]
) -> None:
    identity = _identity(name)
    _mark_later_stale(status_path.parent, name)
    garimpo_uncertainty.files.write_json(status_path, identity | {"status": "running"})
    start = time.monotonic()
    try:
        shapes = work()
    except Exception as error:
        record = identity | {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            # The U-Net's deepest layer fails 24 frames below this `try`; COMPASS adds more.
            "traceback": traceback.format_exc(limit=32),
            "seconds": round(time.monotonic() - start, 1),
        }
        garimpo_uncertainty.files.write_json(status_path, record | _peak_memory())
        raise
    except KeyboardInterrupt:
        # Ctrl+C is no failure to debug: the stage stopped where it was and a rerun resumes it.
        seconds = round(time.monotonic() - start, 1)
        stopped = {"status": "interrupted", "error": "Ctrl+C; rerun to resume", "seconds": seconds}
        garimpo_uncertainty.files.write_json(status_path, identity | stopped)
        raise
    record = identity | {
        "status": "ok",
        "completed_fingerprint": stage_fingerprint,
        "shapes": shapes,
        "seconds": round(time.monotonic() - start, 1),
    }
    garimpo_uncertainty.files.write_json(status_path, record | _peak_memory())
    logger.info("%s: ok in %s s, %s", name, record["seconds"], shapes)


def run_stage(
    status_dir: Path,
    name: str,
    outputs: list[Path],
    stage_fingerprint: dict,
    work: Callable[[], dict],
    *,
    force: bool,
) -> None:
    """Run `work` unless its outputs exist from the same inputs; `work` returns shapes to record.

    Outputs from other inputs or arguments stop the stage instead of being reused, so a changed
    tolerance can never silently keep the old table. A refusal is a status of its own, and keeps the
    last completed fingerprint, so it can never turn into a rerun on the next attempt.
    """
    status_path = status_dir / f"{name}.json"
    previous = read_status(status_dir, name)

    absent = [p for p, value in stage_fingerprint["inputs"].items() if value == MISSING_INPUT]
    if absent:
        fail_stage(
            status_dir, name, f"{name}: missing input {absent}; run the stage that writes it first"
        )

    if _reusable(outputs, previous.get("completed_fingerprint"), force=force):
        _skip_or_refuse(status_path, name, previous, stage_fingerprint)
        return
    _run_work(status_path, name, stage_fingerprint, work)
