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


def log_to_stdout() -> None:
    """Show this package's messages as bare stdout lines, in order with what a script prints.

    Only this package: INFO on the root logger would add the dependencies' messages to the job log.
    Handlers are replaced, not added, so a main run twice in one process does not double each line.
    """
    package = logging.getLogger("garimpo_uncertainty")
    package.handlers = [logging.StreamHandler(sys.stdout)]
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
    after a code fix FORCE_FROM redoes the stage and those after it by name.
    """
    ignored = IGNORED_ARGUMENTS - set(keep)
    arguments = {k: v for k, v in vars(args).items() if k not in ignored}
    files = {
        str(p): f"{p.stat().st_size}:{p.stat().st_mtime_ns}" if p.exists() else MISSING_INPUT
        for p in inputs
    }
    files |= {path: value or MISSING_INPUT for path, value in (hashed or {}).items()}
    return json.loads(json.dumps({"arguments": arguments, "inputs": files}, default=str))


def _peak_memory() -> dict:
    # ru_maxrss is in KiB on Linux; children covers DataLoader workers.
    rss = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
    peak = {"peak_rss_gb": round(rss / 1024**2, 3)}
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        peak["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
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


def _identity(name: str) -> dict:
    return {
        "stage": name,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
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
    completed = previous.get("completed_fingerprint")
    identity = _identity(name)

    absent = [p for p, value in stage_fingerprint["inputs"].items() if value == MISSING_INPUT]
    if absent:
        fail_stage(
            status_dir, name, f"{name}: missing input {absent}; run the stage that writes it first"
        )

    if outputs and completed and all(p.exists() for p in outputs) and not force:
        if completed == stage_fingerprint:
            if previous["status"] != "ok":
                restored = {k: v for k, v in previous.items() if k != "error"}
                write_json(status_path, restored | {"status": "ok"})
            logger.info("%s: outputs exist, skipping (use --force to redo)", name)
            return
        # The validation sits outside STAGE_ORDER in slurm/stages.sh.
        redo = (
            "FORCE_VALIDATE=1 to redo it"
            if name == "validate_data"
            else f"FORCE_FROM={name} to redo this stage and those after it"
        )
        message = f"{name}: outputs exist from other inputs or arguments; use --force, or {redo}"
        write_json(status_path, previous | {"status": "refused", "error": message})
        raise SystemExit(message)

    write_json(status_path, identity | {"status": "running"})
    start = time.monotonic()
    try:
        shapes = work()
    except Exception as error:
        record = identity | {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=8),
            "seconds": round(time.monotonic() - start, 1),
        }
        write_json(status_path, record | _peak_memory())
        raise
    record = identity | {
        "status": "ok",
        "completed_fingerprint": stage_fingerprint,
        "shapes": shapes,
        "seconds": round(time.monotonic() - start, 1),
    }
    write_json(status_path, record | _peak_memory())
    logger.info("%s: ok in %s s, %s", name, record["seconds"], shapes)
