"""The validation stage against defects in the store and the split."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import garimpo_uncertainty.data
from tests.e2e import support


def test_clean_store_has_no_problems(data_root: Path) -> None:
    assert json.loads(support.CLEAN_REPORT.read_text())["problems"] == []


def test_validation_skips_an_intact_store(data_root: Path) -> None:
    result = support.validate(data_root, support.SPLIT_PATH, support.CLEAN_REPORT)
    assert "skipping" in result.stdout


@pytest.fixture(scope="module")
def broken_validation(data_root: Path) -> tuple[int, str]:
    """Validate a copy holding one defect per check; return its exit code and its problems."""
    broken = support.ARTIFACTS / "broken_data"
    shutil.copytree(data_root, broken)
    split = pd.read_csv(support.SPLIT_PATH)
    cells = garimpo_uncertainty.data.cell_table(data_root, split, split.tile_id.tolist())
    # One test tile leaks to calibration, another vanishes with a missing scene, one mask changes.
    test_groups = cells[cells.role == "test"].groupby("group").tile_id.unique()
    leaked, vanished = test_groups.iloc[0][0], test_groups.iloc[1][0]
    split.loc[split.tile_id == leaked, "role"] = "calibration"
    broken_split = support.ARTIFACTS / "split_broken.csv"
    split.to_csv(broken_split, index=False)

    def edit_sidecar(tile_id: int, **changes: object) -> None:
        path = broken / "tiles" / f"{tile_id}.json"
        path.write_text(json.dumps(json.loads(path.read_text()) | changes))

    edit_sidecar(vanished, status="missing_item")
    # A box wider than the window explains its offset, as in tile 684; beyond that it is a defect.
    explained, misaligned = cells[cells.role == "train"].tile_id.unique()[:2]
    edit_sidecar(explained, tile_offset_m=15.6, tile_box_excess_m=31.0)
    edit_sidecar(misaligned, tile_offset_m=26.6, tile_box_excess_m=31.0)
    # A 20 m grid: every cell would cover four times the area the metric assumes.
    coarse = cells[cells.role == "validation"].tile_id.unique()[0]
    transform = json.loads((broken / "tiles" / f"{coarse}.json").read_text())["transform"]
    edit_sidecar(coarse, transform=[20.0, *transform[1:4], -20.0, transform[5]])
    store = garimpo_uncertainty.data.open_store(broken, "r+")
    calibration = cells[cells.role == "calibration"].cell_index.to_numpy()
    bad_cell = calibration[store.stats["mask_ha"][calibration] > 0][0]
    store.masks[bad_cell] = 0
    store.masks.flush()

    report = support.ARTIFACTS / "validation_broken.json"
    result = support.validate(broken, broken_split, report, check=False)
    return result.returncode, " ".join(json.loads(report.read_text())["problems"])


@pytest.mark.parametrize(
    "check",
    [
        "group_leakage",
        "held_out_leak",
        "missing_scenes",
        "partial_group",
        "mask_area",
        "pixel_size",
    ],
)
def test_validation_names_each_defect(broken_validation: tuple[int, str], check: str) -> None:
    returncode, problems = broken_validation
    assert returncode != 0
    assert check in problems, f"{check} not reported: {problems}"


def test_validation_counts_only_the_unexplained_offset(broken_validation: tuple[int, str]) -> None:
    _, problems = broken_validation
    assert "tile_offset: 1 cases" in problems, problems


@pytest.mark.parametrize("name", ["reordered", "dropped", "added"])
def test_validation_refuses_a_split_that_moved_a_tile(
    private_data_root: Path, tmp_path: Path, name: str
) -> None:
    path, message = support.write_changed_split(name, tmp_path)
    report = tmp_path / f"validation_split_{name}.json"
    assert support.validate(private_data_root, path, report, check=False).returncode != 0
    assert message in " ".join(json.loads(report.read_text())["problems"]), name


@pytest.mark.parametrize("defect", ["cell_px", "stats_length"])
def test_validation_names_a_store_of_another_layout(
    private_data_root: Path, tmp_path: Path, defect: str
) -> None:
    """Cells of another size, or a stats array of another store, misstate every cell's area."""
    cell_px = str(support.CELL_PX)
    if defect == "cell_px":
        cell_px, named = (
            str(support.CELL_PX // 2),
            "3 cases, for example ['images', 'masks', 'scl']",
        )
    else:
        stats = np.load(private_data_root / "stats.npy")
        np.save(private_data_root / "stats.npy", np.concatenate([stats, stats[:4]]))
        named = "1 cases, for example ['stats']"
    report = tmp_path / "validation.json"
    result = support.run(
        "validate_data.py",
        "--split",
        str(support.SPLIT_PATH),
        "--train-fraction",
        "1.0",
        "--subset-seed",
        "0",
        "--cell-px",
        cell_px,
        "--report",
        str(report),
        data_root=private_data_root,
        check=False,
    )
    assert result.returncode != 0
    problems = json.loads(report.read_text())["problems"]
    assert f"store_layout: {named}" in problems, problems
