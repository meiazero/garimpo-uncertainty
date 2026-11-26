"""What every test process needs before anything imports torch or starts a worker."""

import os
import shutil
from pathlib import Path

import pytest
import torch

# Workers run torch side by side, so threads per process would oversubscribe. The count also shapes
# the weights: test_fractions' one-tile run fails fit_directions at 2 threads and passes at 1.
os.environ["OMP_NUM_THREADS"] = "1"

# What `garimpo_uncertainty.cli` turns on for every stage: a test that recomputes a stage here
# compares it to the bit with the stage's own output.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(mode=True)

# On a hybrid Intel CPU, oneDNN's dilated convolution, in the ConvNet's encoder, gives other last
# bits on an efficiency core than on a performance core, so two processes compared to the bit could
# disagree. Every process, workers and stages included, inherits this; a narrower taskset stays.
EFFICIENCY_CORES = Path("/sys/devices/cpu_atom/cpus")


def cpu_list(text: str) -> set[int]:
    """Parse a kernel CPU list such as `4-11` or `0,2-3`."""
    cpus = set()
    for part in text.strip().split(","):
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


efficiency_cores = EFFICIENCY_CORES.read_text(encoding="utf-8") if EFFICIENCY_CORES.exists() else ""
if efficiency_cores.strip() and (pinned := cpu_list(efficiency_cores) & os.sched_getaffinity(0)):
    os.sched_setaffinity(0, pinned)

ARTIFACTS = Path(__file__).parent / "artifacts"


def pytest_sessionstart(session: pytest.Session) -> None:
    """Reset once, from the controller: a worker doing it would delete what the others built."""
    config = session.config
    if not hasattr(config, "workerinput") and not config.option.collectonly:
        shutil.rmtree(ARTIFACTS, ignore_errors=True)
        ARTIFACTS.mkdir()
