"""Counts and nesting of select_tiles and of the versioned lists, on the real outputs/split.csv."""

import hashlib
import json
import runpy
from pathlib import Path

import pandas as pd
import pytest

import garimpo_uncertainty.data
from tests.e2e import support

LIST_ARTIFACTS = support.ROOT / "tests" / "artifacts" / "lists"
LIST_DIRECTORY = support.ROOT / "outputs" / "tiles"
REAL_SPLIT_PATH = support.ROOT / "outputs" / "split.csv"
# Literal training tiles per fraction on outputs/split.csv, kept apart from the pipeline.
REAL_SPLIT_TRAIN_TILES = {0.01: (1, 9), 0.05: (3, 47), 0.10: (6, 95), 1.0: (56, 947)}
REAL_SPLIT_OTHER_TILES = 176
# Literal list file, tiles and cells per fraction on outputs/split.csv.
REAL_LISTS = {
    0.01: ("tiles_1pct.json", 186, 2_976),
    0.05: ("tiles_5pct.json", 226, 3_616),
    0.10: ("tiles_10pct.json", 277, 4_432),
    1.0: ("tiles_100pct.json", 1_179, 18_864),
}


@pytest.fixture(scope="module")
def real_split() -> pd.DataFrame:
    return pd.read_csv(REAL_SPLIT_PATH)


@pytest.mark.parametrize("fraction", REAL_SPLIT_TRAIN_TILES)
def test_select_tiles_keeps_the_literal_counts(real_split: pd.DataFrame, fraction: float) -> None:
    summary = json.loads((support.ROOT / "outputs" / "split_summary.json").read_text())
    others = real_split[real_split.role.isin(["validation", "calibration", "test"])]
    artisanal, industrial = REAL_SPLIT_TRAIN_TILES[fraction]
    selected = garimpo_uncertainty.data.select_tiles(real_split, fraction, 0)
    assert selected.equals(garimpo_uncertainty.data.select_tiles(real_split, fraction, 0))
    assert "dropped" not in set(selected.role)
    train = selected[selected.role == "train"]
    assert (train.scale == "Artisanal").sum() == artisanal
    assert (train.scale == "Industrial").sum() == industrial
    assert set(selected[selected.role != "train"].tile_id) == set(others.tile_id)
    assert len(selected) == artisanal + industrial + REAL_SPLIT_OTHER_TILES

    # The counts validation expects come from the summary, the tiles from the split.
    scaled = garimpo_uncertainty.data.scale_summary(summary, fraction)["tiles_by_scale_role"]
    for (scale, role), rows in selected.groupby(["scale", "role"]):
        assert scaled[f"{scale}/{role}"] == len(rows), (fraction, scale, role)


def test_selected_training_tiles_nest_up_to_the_whole_role(real_split: pd.DataFrame) -> None:
    nested: set[int] = set()
    for fraction in REAL_SPLIT_TRAIN_TILES:
        selected = garimpo_uncertainty.data.select_tiles(real_split, fraction, 0)
        train = set(selected[selected.role == "train"].tile_id)
        assert nested < train
        nested = train
    assert nested == set(real_split[real_split.role == "train"].tile_id)


def test_subset_seed_changes_the_selection(real_split: pd.DataFrame) -> None:
    select = garimpo_uncertainty.data.select_tiles
    assert set(select(real_split, 0.05, 0).tile_id) != set(select(real_split, 0.05, 1).tile_id)


def test_selection_keeps_a_minimum_where_rounding_gives_zero() -> None:
    """10 artisanal and 5 industrial training tiles."""
    small = garimpo_uncertainty.data.select_tiles(pd.DataFrame(support.group_tiles()), 0.01, 0)
    assert small[small.role == "train"].scale.tolist() == ["Artisanal"]


@pytest.fixture(scope="module")
def list_seed() -> int:
    return json.loads((LIST_DIRECTORY / "tiles_1pct.json").read_text())["subset_seed"]


@pytest.fixture(scope="module")
def regenerated_lists(list_seed: int) -> Path:
    regenerated = LIST_ARTIFACTS / "real"
    arguments = ["lists", "--subset-seed", str(list_seed), "--out", str(regenerated)]
    support.run("download_data.py", *arguments, data_root=regenerated)
    return regenerated


@pytest.mark.parametrize("fraction", REAL_LISTS)
def test_code_reproduces_the_versioned_list(regenerated_lists: Path, fraction: float) -> None:
    """A fresh clone downloads from these files, so the code must still reproduce them exactly."""
    name = REAL_LISTS[fraction][0]
    assert (LIST_DIRECTORY / name).read_bytes() == (regenerated_lists / name).read_bytes(), name


@pytest.mark.parametrize("fraction", REAL_LISTS)
def test_versioned_list_records_its_selection(
    real_split: pd.DataFrame, list_seed: int, fraction: float
) -> None:
    name, n_tiles, n_cells = REAL_LISTS[fraction]
    tile_list = json.loads((LIST_DIRECTORY / name).read_text())
    assert list(tile_list) == ["train_fraction", "subset_seed", "split_sha256", "tiles"]
    assert tile_list["train_fraction"] == fraction
    assert tile_list["subset_seed"] == list_seed
    assert tile_list["split_sha256"] == hashlib.sha256(REAL_SPLIT_PATH.read_bytes()).hexdigest()
    tiles = [(tile["tile_id"], tile["role"]) for tile in tile_list["tiles"]]
    assert len(tiles) == n_tiles
    assert len(tiles) * 16 == n_cells

    # What training selects, what the list says and what the download takes are one set.
    trained = garimpo_uncertainty.data.select_tiles(real_split, fraction, list_seed)
    assert tiles == sorted(zip(trained.tile_id.tolist(), trained.role.tolist(), strict=True))
    listed_tiles = runpy.run_path(str(support.ROOT / "scripts" / "download_data.py"))[
        "listed_tiles"
    ]
    downloaded = listed_tiles(LIST_DIRECTORY / name, REAL_SPLIT_PATH, real_split)
    assert downloaded.tile_id.tolist() == trained.tile_id.tolist()


def test_versioned_lists_nest_and_stand_alone() -> None:
    nested: set[int] = set()
    for name, _, _ in REAL_LISTS.values():
        tiles = json.loads((LIST_DIRECTORY / name).read_text())["tiles"]
        train = {tile["tile_id"] for tile in tiles if tile["role"] == "train"}
        assert nested < train
        nested = train
    assert sorted(path.name for path in LIST_DIRECTORY.iterdir()) == sorted(
        name for name, _, _ in REAL_LISTS.values()
    )
