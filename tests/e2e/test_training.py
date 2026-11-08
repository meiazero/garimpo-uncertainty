"""What training writes, and every way it must refuse instead of training on the wrong thing."""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import garimpo_uncertainty.data
from tests.e2e import support


def test_training_artifacts(finished_run: support.FinishedRun) -> None:
    arch, run_dir, _ = finished_run
    history = pd.read_csv(run_dir / "history.csv")
    assert len(history) == 3
    assert {"train_data_wait_seconds", "seconds"} <= set(history.columns)
    # Raw timings, unrounded: the parts of an epoch must fit inside the epoch's rounded total.
    assert (history.train_seconds > 0).all()
    assert (history.validation_seconds > 0).all()
    assert (history.train_seconds + history.validation_seconds <= history.seconds + 0.001).all()
    assert (history.train_data_wait_seconds <= history.train_seconds + 0.001).all()
    infer = json.loads((run_dir / "status" / "infer.json").read_text(encoding="utf-8"))["shapes"]
    assert set(infer["seconds_per_role"]) == set(garimpo_uncertainty.data.ROLES)
    assert all(seconds > 0 for seconds in infer["seconds_per_role"].values())
    assert set(infer["peak_gpu_bytes_per_role"]) == set(garimpo_uncertainty.data.ROLES)
    infer_peaks = list(infer["peak_gpu_bytes_per_role"].values())
    peaks = [*history.train_peak_gpu_bytes, *history.validation_peak_gpu_bytes, *infer_peaks]
    if support.DEVICE == "cpu":
        # Recorded as absent, never as zero.
        assert all(pd.isna(peak) for peak in peaks)
    else:
        # A peak exists only if the stages, given no --device, chose the GPU themselves.
        assert all(peak > 0 for peak in peaks)
        assert json.loads((run_dir / "status" / "train.json").read_text())["peak_gpu_gb"] > 0
    assert (run_dir / "last.pt").exists()
    metadata = garimpo_uncertainty.data.read_run_metadata(run_dir)
    for key in ("arch", "seed", "normalization", "split_hash", "root_commit", "compass_commit"):
        assert key in metadata, key
    assert metadata["train_fraction"] == 1.0
    assert metadata["subset_seed"] == 0
    assert metadata["arch"] == arch
    assert metadata["variant"] == support.MODEL_NAMES[arch]
    assert len(metadata["normalization"]["mean"]) == 6
    assert len(metadata["normalization"]["std"]) == 6


@pytest.fixture(scope="module")
def own_run(data_root: Path) -> Path:
    """Skipping needs the path the stage recorded, so these scenarios share a run of their own."""
    run_dir = support.RUNS / "convnet_refusals"
    support.run("train.py", *support.train_args("convnet", run_dir), data_root=data_root)
    return run_dir


@pytest.fixture
def refusal_run(own_run: Path) -> Iterator[Path]:
    with support.restored_after(own_run):
        yield own_run


def test_another_batch_size_refuses_and_the_original_skips(
    refusal_run: Path, data_root: Path
) -> None:
    """Training's batch size shapes the weights."""
    status = refusal_run / "status" / "train.json"
    args = support.train_args("convnet", refusal_run, batch_size="4")
    refused = support.run("train.py", *args, data_root=data_root, check=False)
    assert refused.returncode != 0
    assert json.loads(status.read_text())["status"] == "refused"
    rerun = support.run(
        "train.py", *support.train_args("convnet", refusal_run), data_root=data_root
    )
    assert "skipping" in rerun.stdout
    assert json.loads(status.read_text())["status"] == "ok"


def test_interrupted_run_refuses_other_arguments(refusal_run: Path, data_root: Path) -> None:
    """An interrupted run has no finished status, so last.pt itself must refuse."""
    status = refusal_run / "status" / "train.json"
    weights = support.file_hashes(refusal_run)["last.pt"]
    status.unlink()
    args = support.train_args("convnet", refusal_run, lr="1e-2")
    resumed = support.run("train.py", *args, data_root=data_root, check=False)
    assert resumed.returncode != 0
    record = json.loads(status.read_text())
    assert record["status"] == "failed"
    assert "['lr']" in record["error"]
    assert support.file_hashes(refusal_run)["last.pt"] == weights


def test_all_nan_dice_fails(data_root: Path) -> None:
    """Give validation cells no mining: NaN every epoch, so no best checkpoint ever exists."""
    no_mining = support.ARTIFACTS / "data_no_validation_mining"
    shutil.copytree(data_root, no_mining)
    split = pd.read_csv(support.SPLIT_PATH)
    cells = garimpo_uncertainty.data.cell_table(no_mining, split, split.tile_id.tolist())
    masks = np.load(no_mining / "masks.npy", mmap_mode="r+")
    masks[cells[cells.role == "validation"].cell_index.to_numpy()] = 0
    masks.flush()
    run_dir = support.RUNS / "convnet_all_nan"
    result = support.run(
        "train.py",
        *support.train_args("convnet", run_dir, epochs="1"),
        data_root=no_mining,
        check=False,
    )
    assert result.returncode != 0
    record = json.loads((run_dir / "status" / "train.json").read_text())
    assert record["status"] == "failed"
    assert "every val_dice is NaN" in record["error"]
    assert not (run_dir / "best_model.pt").exists()


@pytest.mark.parametrize("name", ["reordered", "dropped", "added"])
def test_training_refuses_a_split_that_moved_a_tile(
    data_root: Path, tmp_path: Path, name: str
) -> None:
    """A split that moved a tile must refuse, not read another tile's slots."""
    path, message = support.write_changed_split(name, tmp_path)
    run_dir = tmp_path / "run"
    args = support.train_args("convnet", run_dir, epochs="1", split_path=path)
    assert support.run("train.py", *args, data_root=data_root, check=False).returncode != 0
    record = json.loads((run_dir / "status" / "train.json").read_text())
    assert record["status"] == "failed"
    assert message in record["error"], name
    assert not (run_dir / "best_model.pt").exists()


def encoder_weights_args(case: str, run_dir: Path) -> list[str]:
    if case == "convnet_with_weights":
        return [*support.train_args("convnet", run_dir), "--encoder-weights", "none"]
    if case == "unet_without_weights":
        args = support.train_args("unet", run_dir)
        cut = args.index("--encoder-weights")
        return args[:cut] + args[cut + 2 :]
    # The encoder is fixed, so no run can choose another one.
    return [*support.train_args("unet", run_dir), "--encoder", "resnet50"]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("convnet_with_weights", "invalid for convnet"),
        ("unet_without_weights", "required for unet"),
        ("unet_with_another_encoder", "unrecognized arguments"),
    ],
)
def test_encoder_weights_refusals(data_root: Path, tmp_path: Path, case: str, message: str) -> None:
    args = encoder_weights_args(case, tmp_path / "run")
    result = support.run("train.py", *args, data_root=data_root, check=False)
    assert result.returncode == 2, message
    assert message in result.stderr


def test_imagenet_absent_file_names_the_command(data_root: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    args = [*support.train_args("unet", run_dir, epochs="1"), "--encoder-weights", "imagenet"]
    result = support.run("train.py", *args, data_root=data_root, check=False)
    assert result.returncode != 0
    record = json.loads((run_dir / "status" / "train.json").read_text())
    assert record["status"] == "failed"
    assert "download_data.py weights" in record["error"]
