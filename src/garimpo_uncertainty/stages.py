"""Runs each pipeline stage once: skip finished work, record status, timing, shapes, peak memory.

No torch import here, so CPU-only stages stay free of it.
"""

import json
import logging
import os
import resource
import socket
import subprocess
import sys
import time
import traceback
from argparse import Namespace
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn

import numpy as np

logger = logging.getLogger(__name__)

# `report` names an output: another job's path for the same selection must redo it, not refuse.
IGNORED_ARGUMENTS = {
    "force",
    "device",
    "num_workers",
    "batch_size",
    "workers",
    "lightgbm_jobs",
    "report",
}
MISSING_INPUT = "missing"
# The order scripts/run.py runs a model's stages in, after validate_data; --force-from names one.
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


@contextmanager
def atomic(path: Path) -> Iterator[Path]:
    """Yield a temporary path renamed to `path` on success, so a crash leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        yield temporary
        Path(temporary).replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, content: dict) -> None:
    with atomic(path) as temporary:
        temporary.write_text(json.dumps(content, indent=2, default=str) + "\n")


def write_arrays(path: Path, **arrays: np.ndarray) -> None:
    with atomic(path) as temporary, temporary.open("wb") as file:
        np.savez(file, **arrays)


def _git(directory: str, *args: str) -> str:
    try:
        command = ["git", "-C", directory, *args]
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()  # noqa: S603 - literal git arguments only
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def code_versions() -> dict[str, str | bool]:
    """Return the commits of the package and the submodule, as the job actually saw them."""
    return {
        "root_commit": _git(".", "rev-parse", "HEAD"),
        "root_dirty": bool(
            _git(".", "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=dirty")
        ),
        "compass_commit": _git("compass", "rev-parse", "HEAD"),
        "compass_dirty": bool(_git("compass", "status", "--porcelain")),
    }


def _identity(name: str) -> dict:
    # The code each stage ran, since a session may pull between stages of one run.
    return {
        "stage": name,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        **code_versions(),
    }


def fail_stage(status_dir: Path, name: str, message: str) -> NoReturn:
    """Record a failure found before a stage's work, and stop.

    The last completed fingerprint stays, so fixing the cause lets a finished stage skip again.
    """
    status_path = status_dir / f"{name}.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    completed = previous.get("completed_fingerprint")
    kept = {"completed_fingerprint": completed} if completed else {}
    write_json(status_path, _identity(name) | kept | {"status": "failed", "error": message})
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
            write_json(status_path, restored | {"status": "ok"})
        logger.info("%s: outputs exist, skipping (use --force to redo)", name)
        return
    # The validation sits outside STAGE_ORDER.
    redo = (
        "--force-validate to redo it"
        if name == "validate_data"
        else f"--force-from {name} to redo this stage and those after it"
    )
    message = f"{name}: outputs exist from other inputs or arguments; use --force, or {redo}"
    write_json(status_path, previous | {"status": "refused", "error": message})
    raise SystemExit(message)


def _mark_later_stale(status_dir: Path, name: str) -> None:
    """Turn every later ok status to stale, since those stages read what this one is about to redo.

    Done before the work, so a job that dies after it leaves no later stage reading ok, also in a
    copy to Drive. The completed fingerprint stays: a later stage whose inputs did not change skips.
    """
    if name not in STAGE_ORDER:
        return
    for later in STAGE_ORDER[STAGE_ORDER.index(name) + 1 :]:
        path = status_dir / f"{later}.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        if record.get("status") == "ok":
            write_json(path, record | {"status": "stale", "error": f"{name} ran again after it"})


def _run_work(
    status_path: Path, name: str, stage_fingerprint: dict, work: Callable[[], dict]
) -> None:
    identity = _identity(name)
    _mark_later_stale(status_path.parent, name)
    write_json(status_path, identity | {"status": "running"})
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
        write_json(status_path, record | _peak_memory())
        raise
    except KeyboardInterrupt:
        # Ctrl+C is no failure to debug: the stage stopped where it was and a rerun resumes it.
        seconds = round(time.monotonic() - start, 1)
        stopped = {"status": "interrupted", "error": "Ctrl+C; rerun to resume", "seconds": seconds}
        write_json(status_path, identity | stopped)
        raise
    record = identity | {
        "status": "ok",
        "completed_fingerprint": stage_fingerprint,
        "shapes": shapes,
        "seconds": round(time.monotonic() - start, 1),
    }
    write_json(status_path, record | _peak_memory())
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
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}

    absent = [p for p, value in stage_fingerprint["inputs"].items() if value == MISSING_INPUT]
    if absent:
        fail_stage(
            status_dir, name, f"{name}: missing input {absent}; run the stage that writes it first"
        )

    if _reusable(outputs, previous.get("completed_fingerprint"), force=force):
        _skip_or_refuse(status_path, name, previous, stage_fingerprint)
        return
    _run_work(status_path, name, stage_fingerprint, work)
