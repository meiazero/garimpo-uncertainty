"""Training killed after an epoch and run again, as when a Colab session ends mid-training.

A refused resume under other arguments is in test_training.py.
"""

import json
import os
import shutil
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
TIMINGS = [
    "seconds",
    "train_seconds",
    "validation_seconds",
    "train_data_wait_seconds",
    "train_cells_per_second",
]
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
    # On a GPU, kernels are not deterministic.
    if support.DEVICE == "cpu":
        assert_same_training(run_dir, uninterrupted)


def finished_epoch_zero(run_dir: Path, data_root: Path) -> None:
    stop_after_first_epoch(start_training(run_dir, data_root), run_dir, signal.SIGKILL)


def test_a_resume_keeps_the_best_dice_so_far(data_root: Path, tmp_path: Path) -> None:
    """With last.pt's best Dice raised to 1, which no epoch reaches, epoch 1 must not replace it."""
    run_dir = tmp_path / "convnet"
    finished_epoch_zero(run_dir, data_root)
    last = torch.load(run_dir / "last.pt", weights_only=True)
    torch.save(last | {"best_dice": 1.0}, run_dir / "last.pt")
    support.run("train.py", *training_args(run_dir), data_root=data_root)
    assert torch.load(run_dir / "best_model.pt", weights_only=True)["epoch"] == 0


def test_force_refuses_to_discard_an_unfinished_training(data_root: Path, tmp_path: Path) -> None:
    """--force-from train left in a cell must not restart a training a lost session cut short."""
    run_dir = tmp_path / "convnet"
    finished_epoch_zero(run_dir, data_root)
    before = (run_dir / "last.pt").stat().st_mtime_ns
    args = training_args(run_dir)
    forced = support.run("train.py", *args, "--force", data_root=data_root, check=False)
    assert forced.returncode != 0
    status = json.loads((run_dir / "status" / "train.json").read_text())
    assert status["status"] == "failed"
    assert "holds an unfinished training (status running)" in status["error"], status
    assert (run_dir / "last.pt").stat().st_mtime_ns == before
    assert "resuming after epoch 0" in support.run("train.py", *args, data_root=data_root).stdout


def test_forced_redo_of_a_finished_training_never_resumes_from_its_end(
    data_root: Path, tmp_path: Path
) -> None:
    """A forced redo cut short before its first epoch must start over, not resume the old run.

    The run goes to Drive and back between the sessions as the Colab cells copy it, with no
    --delete, so a last.pt removed only locally would return.
    """
    run_dir, drive = tmp_path / "convnet", tmp_path / "drive"
    # Trained here, not copied: a last.pt naming another run_dir would be refused for that instead.
    support.run("train.py", *training_args(run_dir), data_root=data_root)
    shutil.copytree(run_dir, drive)
    status = run_dir / "status" / "train.json"
    env = {**os.environ, "GARIMPO_DATA_DIR": str(data_root)}
    command = [sys.executable, "scripts/train.py", *training_args(run_dir), "--force"]
    process = subprocess.Popen(command, cwd=support.ROOT, env=env)  # noqa: S603 - test arguments
    deadline = time.monotonic() + LAST_PT_TIMEOUT_SECONDS
    while json.loads(status.read_text())["status"] != "running":
        assert process.poll() is None
        assert time.monotonic() < deadline
        time.sleep(0.005)
    process.kill()
    process.wait(timeout=60)
    shutil.copytree(run_dir, drive, dirs_exist_ok=True)
    shutil.rmtree(run_dir)
    shutil.copytree(drive, run_dir)
    rerun = support.run("train.py", *training_args(run_dir), data_root=data_root)
    assert "resuming" not in rerun.stdout
    assert pd.read_csv(run_dir / "history.csv").epoch.tolist() == [0, 1]
