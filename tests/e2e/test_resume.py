"""Training killed after an epoch and run again, as when a Colab session ends mid-training.

A refused resume under other arguments is in test_training.py.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
import torch

from tests.e2e import support

# What the clock changes; every other column must match the uninterrupted run.
TIMINGS = ["seconds", "train_seconds", "validation_seconds", "train_data_wait_seconds"]
LAST_PT_TIMEOUT_SECONDS = 300
# One epoch before the kill, one after it.
EPOCHS = "2"


def training_args(run_dir: Path) -> list[str]:
    return support.train_args("convnet", run_dir, epochs=EPOCHS)


def start_training(run_dir: Path, data_root: Path) -> subprocess.Popen:
    env = {**os.environ, "GARIMPO_DATA_DIR": str(data_root)}
    command = [sys.executable, "scripts/train.py", *training_args(run_dir)]
    return subprocess.Popen(  # noqa: S603 - arguments come from the tests
        command, cwd=support.ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def stop_after_first_epoch(process: subprocess.Popen, run_dir: Path, signal_number: int) -> None:
    """Signal the process as soon as epoch 0's last.pt exists, long before epoch 1 can end."""
    last = run_dir / "last.pt"
    deadline = time.monotonic() + LAST_PT_TIMEOUT_SECONDS
    while not last.exists():
        if process.poll() is not None:
            stderr = process.communicate()[1].decode()
            raise AssertionError(f"train.py exited {process.returncode}:\n{stderr[-4000:]}")
        if time.monotonic() > deadline:
            process.kill()
            raise AssertionError(f"no last.pt after {LAST_PT_TIMEOUT_SECONDS} s")
        time.sleep(0.005)
    process.send_signal(signal_number)
    process.communicate(timeout=60)


def assert_same_training(run_dir: Path, reference_dir: Path) -> None:
    """Exact equality: same seed, one CPU core, so any difference is state the resume lost."""
    history = pd.read_csv(run_dir / "history.csv").drop(columns=TIMINGS)
    expected = pd.read_csv(reference_dir / "history.csv").drop(columns=TIMINGS)
    pd.testing.assert_frame_equal(history, expected, check_exact=True)
    last, expected_last = (
        torch.load(d / "last.pt", weights_only=True) for d in (run_dir, reference_dir)
    )
    # Without the optimizer's moments or the generator, epoch 1 would differ.
    for key in ("model", "rng"):
        torch.testing.assert_close(last[key], expected_last[key], rtol=0, atol=0)
    optimizer, expected_optimizer = last["optimizer"]["state"], expected_last["optimizer"]["state"]
    torch.testing.assert_close(optimizer, expected_optimizer, rtol=0, atol=0)
    best, expected_best = (
        torch.load(d / "best_model.pt", weights_only=True) for d in (run_dir, reference_dir)
    )
    torch.testing.assert_close(best["model"], expected_best["model"], rtol=0, atol=0)
    assert (best["epoch"], best["val_dice"]) == (expected_best["epoch"], expected_best["val_dice"])


@pytest.fixture(scope="module")
def uninterrupted(data_root: Path) -> Path:
    """Train the same command, never interrupted, on the core the interrupted runs use."""
    run_dir = support.RUNS / "convnet_uninterrupted"
    with support.one_core():
        support.run("train.py", *training_args(run_dir), data_root=data_root)
    return run_dir


@pytest.mark.parametrize(
    ("signal_number", "left_status"),
    [(signal.SIGKILL, "running"), (signal.SIGINT, "interrupted")],
    ids=["session_ends", "ctrl_c"],
)
def test_training_resumes_after_the_last_finished_epoch(
    uninterrupted: Path, data_root: Path, tmp_path: Path, signal_number: int, left_status: str
) -> None:
    run_dir = tmp_path / "convnet"
    status = run_dir / "status" / "train.json"
    with support.one_core():
        stop_after_first_epoch(start_training(run_dir, data_root), run_dir, signal_number)
        # Neither is a completed stage, so the rerun must reach the work instead of skipping.
        assert json.loads(status.read_text())["status"] == left_status
        assert torch.load(run_dir / "last.pt", weights_only=True)["epoch"] == 0
        assert len(pd.read_csv(run_dir / "history.csv")) == 1
        args = training_args(run_dir)
        resumed = support.run("train.py", *args, data_root=data_root)
    assert "resuming after epoch 0" in resumed.stdout
    assert json.loads(status.read_text())["status"] == "ok"
    assert pd.read_csv(run_dir / "history.csv").epoch.tolist() == [0, 1]
    # On a GPU, kernels are not deterministic and last.pt keeps only the CPU generator.
    if support.DEVICE == "cpu":
        assert_same_training(run_dir, uninterrupted)
