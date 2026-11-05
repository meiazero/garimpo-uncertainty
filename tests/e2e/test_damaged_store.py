"""A store damaged under a finished run: stages may skip, the validation and the report may not."""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

import garimpo_uncertainty.data
from tests.e2e import support

REPORT = support.ARTIFACTS / "validation_damaged.json"


class DamagedRun(NamedTuple):
    root: Path
    run_dir: Path


def assert_report_fails(damaged: DamagedRun, message: str) -> None:
    result = support.pipeline_check(damaged.run_dir, damaged.root, check=False)
    assert result.returncode == 1, result.stderr
    assert message in result.stdout
    assert "PIPELINE CHECK: FAIL" in result.stdout
    assert "PIPELINE CHECK: PASS" not in result.stdout


def problems_after_validating(root: Path) -> str:
    result = support.validate(root, support.SPLIT_PATH, REPORT, check=False)
    assert result.returncode != 0
    return " ".join(json.loads(REPORT.read_text())["problems"])


def move_store_away(root: Path) -> None:
    for name in garimpo_uncertainty.data.CellStore._fields:
        path = root / f"{name}.npy"
        path.rename(path.with_suffix(".away"))


@pytest.fixture(scope="module")
def own_run(data_root: Path) -> DamagedRun:
    root = support.ARTIFACTS / "data_damaged"
    shutil.copytree(data_root, root)
    shutil.rmtree(root / "status")
    support.validate(root, support.SPLIT_PATH, REPORT)
    run_dir = support.RUNS / "convnet_damaged"
    support.finish_run("convnet", run_dir, root)
    return DamagedRun(root, run_dir)


@pytest.fixture
def damaged(own_run: DamagedRun) -> Iterator[DamagedRun]:
    with support.restored_after(own_run.root):
        yield own_run


def test_zeroed_masks_fail_the_report_and_the_validation(damaged: DamagedRun) -> None:
    """The metrics stage skips, as fingerprints read sidecars only: nothing else may."""
    masks_path = damaged.root / "masks.npy"
    healthy = masks_path.read_bytes()
    masks = np.load(masks_path, mmap_mode="r+")
    masks[:] = 0
    masks.flush()
    metrics = support.compute_metrics(damaged.run_dir, support.BOUNDARY_TOLERANCE, damaged.root)
    assert "skipping" in metrics.stdout
    assert_report_fails(damaged, "slot_hash")
    assert "slot_hash" in problems_after_validating(damaged.root)
    masks_path.write_bytes(healthy)
    assert "skipping" not in support.validate(damaged.root, support.SPLIT_PATH, REPORT).stdout
    # The status really failed above, so this is a recovery, not a run that never broke.
    assert "PIPELINE CHECK: PASS" in support.pipeline_check(damaged.run_dir, damaged.root).stdout


def test_deleted_store_with_its_sidecars_fails_the_report(damaged: DamagedRun) -> None:
    move_store_away(damaged.root)
    train_args = support.train_args("convnet", damaged.run_dir)
    assert "skipping" in support.run("train.py", *train_args, data_root=damaged.root).stdout
    assert_report_fails(damaged, "store_missing")
    assert "store_missing" in problems_after_validating(damaged.root)


def test_nothing_vouches_for_the_tiles_without_store_and_sidecars(damaged: DamagedRun) -> None:
    move_store_away(damaged.root)
    (damaged.root / "tiles").rename(damaged.root / "tiles_away")
    assert_report_fails(damaged, "store_missing")
    emptied = support.validate(damaged.root, support.SPLIT_PATH, REPORT, check=False)
    assert emptied.returncode != 0
    assert "missing input" in emptied.stderr


def test_sidecars_gone_with_the_arrays_still_there_fail_the_report(damaged: DamagedRun) -> None:
    tiles_dir = damaged.root / "tiles"
    tiles_dir.rename(damaged.root / "tiles_away")
    assert_report_fails(damaged, "sidecars_missing")
    (damaged.root / "tiles_away").rename(tiles_dir)
    support.validate(damaged.root, support.SPLIT_PATH, REPORT)
    assert "PIPELINE CHECK: PASS" in support.pipeline_check(damaged.run_dir, damaged.root).stdout
