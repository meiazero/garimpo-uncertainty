"""Training from Sentinel-2 weights: the file's identity, present, absent and replaced."""

import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

import garimpo_uncertainty.data
from tests.e2e import support


@dataclass(frozen=True)
class Sentinel2Run:
    data_root: Path
    run_dir: Path
    weights: Path
    args: list[str]
    other_variant_args: list[str]

    @property
    def status(self) -> Path:
        return self.run_dir / "status" / "train.json"

    def hide_weights(self) -> None:
        self.weights.rename(self.weights.with_suffix(".away"))

    def train(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return support.run("train.py", *args, data_root=self.data_root, check=check)


@pytest.fixture(scope="module")
def trained_sentinel2(data_root: Path) -> Sentinel2Run:
    """Train on a data root of its own, whose weights directory the scenarios below change."""
    root = support.ARTIFACTS / "data_sentinel2"
    shutil.copytree(data_root, root)
    weights = garimpo_uncertainty.data.starting_weights_path(root, "sentinel2_moco")
    support.write_synthetic_sentinel2_weights(weights)
    run_dir = support.RUNS / "unet_sentinel2"
    base = support.train_args("unet", run_dir, epochs="1")
    args = [*base, "--encoder-weights", "sentinel2_moco"]
    support.run("train.py", *args, data_root=root)
    return Sentinel2Run(root, run_dir, weights, args, [*base, "--encoder-weights", "none"])


@pytest.fixture
def sentinel2(trained_sentinel2: Sentinel2Run) -> Iterator[Sentinel2Run]:
    """The 0.5 GB checkpoints stay outside the restore, as nothing here may write them."""
    with (
        support.restored_after(trained_sentinel2.run_dir / "status"),
        support.restored_after(trained_sentinel2.weights.parent),
    ):
        yield trained_sentinel2


def test_sentinel2_checkpoint_records_the_weights(trained_sentinel2: Sentinel2Run) -> None:
    best = torch.load(
        trained_sentinel2.run_dir / "best_model.pt", map_location="cpu", weights_only=True
    )
    metadata = best["meta"]
    assert metadata["normalization"] == {"mean": [0.0] * 6, "std": [1.0] * 6}
    assert metadata["encoder"] == "resnet50"
    assert metadata["encoder_weights"] == "sentinel2_moco"
    assert metadata["variant"] == "unet_sentinel2"
    sha256 = hashlib.sha256(trained_sentinel2.weights.read_bytes()).hexdigest()
    assert metadata["encoder_weights_sha256"] == sha256


def test_later_stage_rebuilds_the_model_without_the_weights_file(sentinel2: Sentinel2Run) -> None:
    sentinel2.hide_weights()
    support.run(
        "infer.py", *support.checkpoint_args(sentinel2.run_dir), data_root=sentinel2.data_root
    )


def test_finished_training_skips_without_the_weights_file(sentinel2: Sentinel2Run) -> None:
    """As on a machine with runs/ but no weights/."""
    sentinel2.hide_weights()
    skipped = sentinel2.train(*sentinel2.args)
    assert "skipping" in skipped.stdout
    assert json.loads(sentinel2.status.read_text())["status"] == "ok"


@pytest.mark.parametrize("weights_file", ["absent", "present"])
def test_another_variant_refuses_in_the_same_run_directory(
    sentinel2: Sentinel2Run, weights_file: str
) -> None:
    if weights_file == "absent":
        sentinel2.hide_weights()
    changed = sentinel2.train(*sentinel2.other_variant_args, check=False)
    assert changed.returncode != 0
    assert json.loads(sentinel2.status.read_text())["status"] == "refused"
    sentinel2.train(*sentinel2.args)
    assert json.loads(sentinel2.status.read_text())["status"] == "ok"


def test_unknown_identity_names_what_restores_it(sentinel2: Sentinel2Run) -> None:
    """With the file and run_meta.json both gone, then back: the completed fingerprint stayed."""
    sentinel2.hide_weights()
    run_meta = sentinel2.run_dir / "run_meta.json"
    run_meta.rename(run_meta.with_suffix(".away"))
    unknown = sentinel2.train(*sentinel2.args, check=False)
    assert unknown.returncode != 0
    record = json.loads(sentinel2.status.read_text())
    assert record["status"] == "failed"
    for name in ("run_meta.json", "download_data.py weights"):
        assert name in record["error"], name
    # Restoring it must skip again rather than retrain.
    run_meta.with_suffix(".away").rename(run_meta)
    assert "skipping" in sentinel2.train(*sentinel2.args).stdout
    assert json.loads(sentinel2.status.read_text())["status"] == "ok"


def test_training_that_must_run_names_the_command_for_the_file(sentinel2: Sentinel2Run) -> None:
    sentinel2.hide_weights()
    fresh_dir = support.RUNS / "unet_sentinel2_no_file"
    fresh = [
        *support.train_args("unet", fresh_dir, epochs="1"),
        "--encoder-weights",
        "sentinel2_moco",
    ]
    assert sentinel2.train(*fresh, check=False).returncode != 0
    record = json.loads((fresh_dir / "status" / "train.json").read_text())
    assert record["status"] == "failed"
    assert "download_data.py weights" in record["error"]


def test_interrupted_run_refuses_other_weights(sentinel2: Sentinel2Run) -> None:
    """An interrupted run has no finished status, so last.pt itself must refuse."""
    sentinel2.status.unlink()
    resumed = sentinel2.train(*sentinel2.other_variant_args, check=False)
    assert resumed.returncode != 0
    record = json.loads(sentinel2.status.read_text())
    assert record["status"] == "failed"
    assert "encoder_weights" in record["error"]


def test_another_file_under_the_same_name_changes_the_identity(sentinel2: Sentinel2Run) -> None:
    support.write_synthetic_sentinel2_weights(sentinel2.weights, seed=1)
    refused = sentinel2.train(*sentinel2.args, check=False)
    assert refused.returncode != 0
    assert json.loads(sentinel2.status.read_text())["status"] == "refused"
