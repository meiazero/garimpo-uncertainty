"""Cell store, data root, split selection and the tile and cell tables; no torch, for CPU stages.

One authoritative store: fixed-slot arrays under the data root, addressed by cell_index =
tile_index * cells_per_tile + position. tile_index is a tile's row among the non-dropped tiles of
the split, so a fraction is an index set, and a larger fraction moves nothing.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
RED_INDEX, NIR_INDEX = 2, 3
TILE_PX = 2048
CELL_PX = 512
PIXEL_M = 10.0
PIXEL_HA = 0.01  # 10 m x 10 m
REFLECTANCE_SCALE = 10_000
CLOUD_SCL_CLASSES = (3, 8, 9, 10)  # cloud shadow, medium and high cloud, thin cirrus
HELD_OUT_REGIONS = {"Amazonia BR", "Americas, other"}  # same set as scripts/split/make_split.py
ROLES = ("train", "validation", "calibration", "test")
TRAIN_FRACTIONS = (0.01, 0.05, 0.10, 1.0)
# vector_ha needs the polygon, so no array can recompute it; mask_ha is the writer's claim that
# validate_data.py checks against masks.npy.
CELL_STATS_DTYPE = np.dtype(
    [
        ("mask_ha", "f8"),
        ("vector_ha", "f8"),
    ]
)
NOT_ATTEMPTED = "not_attempted"
SIDECAR_COLUMNS = (
    "status",
    "item",
    "detail",
    "epsg",
    "transform",
    "boa_offset",
    "tile_offset_m",
    "tile_box_excess_m",
    "resampling_20m",
    "scene_cloud",
    "tile_index",
    "cells_per_tile",
    "slots_sha256",
)


class CellStore(NamedTuple):
    images: np.ndarray  # (N, bands, px, px) uint16, raw DN
    masks: np.ndarray  # (N, px, px) uint8
    scl: np.ndarray  # (N, px, px) uint8
    stats: np.ndarray  # (N,) CELL_STATS_DTYPE


def data_root() -> Path:
    default = "~/Experiments/datasets/garimpo"
    return Path(os.environ.get("GARIMPO_DATA_DIR", default)).expanduser()


def starting_weights_path(root: Path, encoder_weights: str | None) -> Path | None:
    """Return the file holding the chosen starting weights, or None when none are chosen."""
    names = {
        "imagenet": "resnet50_imagenet-df1aad85.safetensors",
        "sentinel2_moco": "resnet50_sentinel2_all_moco-df8b932e.pth",
    }
    return root / "weights" / names[encoder_weights] if encoder_weights in names else None


def open_store(root: Path, mode: str = "r") -> CellStore:
    return CellStore(*(np.load(root / f"{name}.npy", mmap_mode=mode) for name in CellStore._fields))


def missing_arrays(root: Path) -> list[str]:
    files = [f"{name}.npy" for name in CellStore._fields]
    return [name for name in files if not (root / name).exists()]


def slots_sha256(store: CellStore, slots: slice) -> str:
    """Hash a tile's slots, so a sidecar vouches for their content, not only for their writing."""
    digest = hashlib.sha256()
    for array in (store.images, store.masks, store.scl):
        digest.update(array[slots])
    return digest.hexdigest()


def subset_size(scale: str, tiles: int, train_fraction: float) -> int:
    """Return how many training tiles of a scale a fraction keeps, at least one artisanal."""
    kept = round(tiles * train_fraction)
    return max(1, kept) if scale == "Artisanal" else kept


def select_tiles(split: pd.DataFrame, train_fraction: float, seed: int) -> pd.DataFrame:
    """Return every evaluation tile and a seeded prefix of each scale's training tiles.

    Whole tiles, not whole groups: groups matter across roles, and only the train role is reduced.
    The shuffles ignore the fraction, so under one seed each subset contains the smaller ones.
    """
    kept = split[split.role != "dropped"]
    train = kept[kept.role == "train"]
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for scale in sorted(train.scale.unique()):
        tiles = rng.permutation(np.sort(train[train.scale == scale].tile_id.unique()))
        chosen += tiles[: subset_size(scale, len(tiles), train_fraction)].tolist()
    return kept[(kept.role != "train") | kept.tile_id.isin(chosen)]


def scale_summary(summary: dict, train_fraction: float) -> dict:
    """Return outputs/split_summary.json with the train role reduced as select_tiles reduces it."""
    counts = {
        name: subset_size(name.split("/")[0], tiles, train_fraction)
        if name.endswith("/train")
        else tiles
        for name, tiles in summary["tiles_by_scale_role"].items()
    }
    return summary | {"tiles_by_scale_role": counts}


def selection_name(train_fraction: float, subset_seed: int) -> str:
    return f"fraction-{train_fraction}-seed-{subset_seed}"


def tile_table(split: pd.DataFrame) -> pd.DataFrame:
    """Return the non-dropped tiles; a row's position is its tile_index, for good."""
    tiles = split[split.role != "dropped"].reset_index(drop=True)
    return tiles.assign(
        tile_index=tiles.index,
        group=tiles.cluster_50km.fillna(-1).astype(int),
        region=tiles.region.fillna(""),
    )


def tile_records(root: Path, split: pd.DataFrame, tile_ids: list[int]) -> pd.DataFrame:
    """Return one row per selected tile: identity from the split, the rest from its sidecar.

    A sidecar's tile_index binds it to the slots it vouches for, so a split that moved the tile
    refuses here instead of reading another tile's slots.
    """
    tiles = tile_table(split)
    tiles = tiles[tiles.tile_id.isin(tile_ids)].reset_index(drop=True)
    sidecars = []
    for tile_id in tiles.tile_id:
        path = root / "tiles" / f"{tile_id}.json"
        sidecars.append(
            json.loads(path.read_text()) if path.exists() else {"status": NOT_ATTEMPTED}
        )
    records = pd.DataFrame(sidecars, columns=list(SIDECAR_COLUMNS))
    moved = tiles.tile_id[(records.status == "ok") & (records.tile_index != tiles.tile_index)]
    if len(moved):
        raise ValueError(
            f"tile_index in the sidecars of {len(moved)} tiles, for example {moved.tolist()[:5]}, "
            "disagrees with this split: restore the split the store was written with, or empty "
            "the store and download again"
        )
    return pd.concat([tiles, records.drop(columns="tile_index")], axis=1)


def cell_table(root: Path, split: pd.DataFrame, tile_ids: list[int]) -> pd.DataFrame:
    """Return the canonical table of a selection: its usable cells, by ascending cell_index."""
    records = tile_records(root, split, tile_ids)
    records = records[records.status == "ok"]
    n_cells = np.load(root / "images.npy", mmap_mode="r").shape[0]
    n_tiles = len(tile_table(split))
    wrong = records[records.cells_per_tile * n_tiles != n_cells]
    if len(wrong):
        raise ValueError(
            f"the store holds {n_cells} cells, but this split's {n_tiles} tiles at "
            f"{int(wrong.cells_per_tile.iloc[0])} cells per tile need "
            f"{n_tiles * int(wrong.cells_per_tile.iloc[0])}: restore the split the store was "
            "written with, or empty the store and download again"
        )
    cells_per_tile = int(records.cells_per_tile.iloc[0]) if len(records) else 0
    columns = ["tile_id", "tile_index", "role", "scale", "region", "group", "boa_offset"]
    cells = records[columns].loc[records.index.repeat(cells_per_tile)].reset_index(drop=True)
    position = np.tile(np.arange(cells_per_tile), len(records))
    return cells.assign(
        position=position,
        cell_index=cells.tile_index.to_numpy() * cells_per_tile + position,
        boa_offset=cells.boa_offset.astype(int),
    )


def positions_of(cell_index: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Return where each wanted value sits in an ascending array; a bare searchsorted would give a
    neighbor's position for an absent one."""
    at = np.searchsorted(cell_index, wanted)
    found = at < len(cell_index)
    found[found] = cell_index[at[found]] == wanted[found]
    if not found.all():
        absent = wanted[~found].tolist()
        raise ValueError(f"{len(absent)} cell_index values are absent, for example {absent[:5]}")
    return at


def tiles_identity(root: Path, split: pd.DataFrame, tile_ids: list[int]) -> dict[str, str | None]:
    """Return the store's identity for a stage: this selection's slots, or None when none exist.

    Hashes the index set, roles, groups and the sidecars that vouch for those slots, so more tiles
    arriving elsewhere in the store never invalidate a finished run.
    """
    directory = root / "tiles"
    tiles = tile_table(split)
    tiles = tiles[tiles.tile_id.isin(tile_ids)]
    table = tiles[["tile_index", "tile_id", "role", "group"]].to_csv(index=False)
    digest = hashlib.sha256(table.encode())
    any_ok = False
    for tile_id in tiles.tile_id:
        path = directory / f"{tile_id}.json"
        text = path.read_text() if path.exists() else "{}"
        if json.loads(text).get("status") == "ok":
            digest.update(text.encode())
            any_ok = True
    return {str(directory): digest.hexdigest() if any_ok else None}


def read_run_metadata(run_dir: Path) -> dict:
    """JSON twin of the checkpoint metadata, so stages with no torch can check the split."""
    return json.loads((run_dir / "run_meta.json").read_text())


def run_identity(root: Path, run_dir: Path) -> dict[str, str | None]:
    """Return tiles_identity for a stage after training, which recorded the run's tiles."""
    path = run_dir / "run_meta.json"
    if not path.exists():
        return {str(path): None}
    metadata = read_run_metadata(run_dir)
    split_path = Path(metadata["split"])
    if not split_path.exists():
        return {str(split_path): None}
    return tiles_identity(root, pd.read_csv(split_path), metadata["tile_ids"])


def read_run_cells(root: Path, run_dir: Path) -> pd.DataFrame:
    metadata = read_run_metadata(run_dir)
    return cell_table(root, pd.read_csv(metadata["split"]), metadata["tile_ids"])


def read_predictions(root: Path, run_dir: Path) -> pd.DataFrame:
    """Return the run's cell table with infer.py's columns, which align with it by position."""
    cells = read_run_cells(root, run_dir)
    with np.load(run_dir / "predictions" / "cells.npz") as stored:
        if not np.array_equal(stored["cell_index"], cells.cell_index.to_numpy()):
            raise ValueError("predictions/cells.npz does not align with this run's cell table")
        names = [name for name in stored.files if name != "cell_index"]
        return cells.assign(**{name: stored[name] for name in names})


def split_hash(cells: pd.DataFrame, train_fraction: float, subset_seed: int) -> str:
    """Return the identity of a run's subset, tiles and roles; detects another data root."""
    table = cells[["tile_id", "role", "group"]].drop_duplicates().sort_values("tile_id")
    content = f"{train_fraction}:{subset_seed}\n{table.to_csv(index=False)}"
    return hashlib.sha256(content.encode()).hexdigest()


def check_split(metadata: dict, cells: pd.DataFrame) -> None:
    current = split_hash(cells, metadata["train_fraction"], metadata["subset_seed"])
    if metadata["split_hash"] != current:
        raise ValueError(
            f"this run's cells have split {current[:12]}, training had "
            f"{metadata['split_hash'][:12]}: wrong GARIMPO_DATA_DIR or another subset"
        )
