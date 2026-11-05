"""What every test process needs before anything imports torch or starts a worker."""

import os
import shutil
from pathlib import Path

import pytest

# Workers run torch side by side, so threads per process would oversubscribe. The count also shapes
# the weights: test_fractions' one-tile run fails fit_directions at 2 threads and passes at 1.
os.environ["OMP_NUM_THREADS"] = "1"

ARTIFACTS = Path(__file__).parent / "artifacts"


def pytest_sessionstart(session: pytest.Session) -> None:
    """Reset once, from the controller: a worker doing it would delete what the others built."""
    config = session.config
    if not hasattr(config, "workerinput") and not config.option.collectonly:
        shutil.rmtree(ARTIFACTS, ignore_errors=True)
        ARTIFACTS.mkdir()
