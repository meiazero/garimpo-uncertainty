"""The fixed-slot cell store and the dataset that reads it."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import garimpo_uncertainty.cell_dataset
import garimpo_uncertainty.data
import garimpo_uncertainty.download
from tests.e2e import support

STORE_ARTIFACTS = support.ROOT / "tests" / "artifacts" / "store"


def filled_store(
    name: str, split: pd.DataFrame, written: list[int]
) -> garimpo_uncertainty.data.CellStore:
    root = STORE_ARTIFACTS / name
    order = garimpo_uncertainty.data.tile_table(split).set_index("tile_id").tile_index
    garimpo_uncertainty.download.create_store(root, len(order), support.TILE_PX, support.CELL_PX)
    store = garimpo_uncertainty.data.open_store(root, "r+")
    for tile_id in written:
        rng = np.random.default_rng(tile_id)
        support.write_synthetic_tile(root, store, tile_id, int(order[tile_id]), rng)
    return garimpo_uncertainty.data.open_store(root)


@pytest.fixture(scope="module")
def split() -> pd.DataFrame:
    return pd.DataFrame(support.group_tiles())


@pytest.fixture(scope="module")
def tile_ids(split: pd.DataFrame) -> list[int]:
    return garimpo_uncertainty.data.select_tiles(split, 1.0, 0).tile_id.tolist()


@pytest.fixture(scope="module")
def ascending(split: pd.DataFrame, tile_ids: list[int]) -> garimpo_uncertainty.data.CellStore:
    return filled_store("ascending", split, tile_ids)


@pytest.fixture(scope="module")
def shuffled(split: pd.DataFrame, tile_ids: list[int]) -> garimpo_uncertainty.data.CellStore:
    order = np.random.default_rng(0).permutation(tile_ids).tolist()
    assert order != tile_ids
    return filled_store("shuffled", split, order)


def test_slots_do_not_depend_on_write_order(
    split: pd.DataFrame,
    tile_ids: list[int],
    ascending: garimpo_uncertainty.data.CellStore,
    shuffled: garimpo_uncertainty.data.CellStore,
) -> None:
    assert len(garimpo_uncertainty.data.tile_table(split)) == len(tile_ids)
    for name in garimpo_uncertainty.data.CellStore._fields:
        assert getattr(ascending, name).tobytes() == getattr(shuffled, name).tobytes(), name
    assert ascending.images.any()


def test_dataset_reads_the_slot_its_cell_index_names(
    split: pd.DataFrame,
    tile_ids: list[int],
    ascending: garimpo_uncertainty.data.CellStore,
    shuffled: garimpo_uncertainty.data.CellStore,
) -> None:
    """Whichever tile reached the disk first."""
    root = STORE_ARTIFACTS / "shuffled"
    cells = garimpo_uncertainty.data.cell_table(root, split, tile_ids)
    dataset = garimpo_uncertainty.cell_dataset.CellDataset(root, cells, None)
    for position in (0, len(cells) // 2, len(cells) - 1):
        image, _, cell_index = dataset[position]
        assert cell_index == cells.cell_index.iloc[position]
        expected = ascending.images[cell_index] / garimpo_uncertainty.data.REFLECTANCE_SCALE
        assert np.allclose(image.numpy(), expected)


def test_one_tile_allocates_the_whole_store(split: pd.DataFrame, tile_ids: list[int]) -> None:
    """Allocation is measured, not asserted: holes depend on the filesystem of tests/artifacts."""
    one_tile = filled_store("one_tile", split, tile_ids[:1])
    measures = {}
    for name in garimpo_uncertainty.data.CellStore._fields:
        status = (STORE_ARTIFACTS / "one_tile" / f"{name}.npy").stat()
        measures[name] = {"apparent_bytes": status.st_size, "allocated": status.st_blocks * 512}
    n_tiles = len(garimpo_uncertainty.data.tile_table(split))
    assert one_tile.images.shape[0] == n_tiles * support.CELLS_PER_TILE
    (STORE_ARTIFACTS / "one_tile_allocation.json").write_text(json.dumps(measures, indent=2))


def test_cell_dataset_applies_the_offset(data_root: Path) -> None:
    """Baseline 04.00 shifts DN; the dataset must undo it and floor at zero."""
    split = pd.read_csv(support.SPLIT_PATH)
    cells = garimpo_uncertainty.data.cell_table(data_root, split, split.tile_id.tolist())
    cells = cells.head(1).assign(boa_offset=-1000)
    image, mask, cell_index = garimpo_uncertainty.cell_dataset.CellDataset(data_root, cells, None)[
        0
    ]
    assert cell_index == cells.cell_index.iloc[0]
    store = garimpo_uncertainty.data.open_store(data_root)
    raw = store.images[cell_index].astype(np.float32)
    assert (raw < 1000).any()
    expected = np.maximum(raw - 1000, 0) / garimpo_uncertainty.data.REFLECTANCE_SCALE
    assert np.allclose(image.numpy(), expected)
    assert np.array_equal(mask[0].numpy() > 0, store.masks[cell_index] > 0)
