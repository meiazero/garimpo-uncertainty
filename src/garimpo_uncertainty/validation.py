"""Checks the cell store against the split.

Failing checks are named in `problems` and fail the stage. Everything under `diagnostics` is
reported with no threshold and cannot fail the stage: clouds, nodata, raster versus vector area,
box excess, NDVI contrast.
"""

import concurrent.futures
import json
from pathlib import Path

import numpy as np
import pandas as pd

import garimpo_uncertainty.data

MAX_EXAMPLES = 5
QUANTILES = (0.0, 0.5, 0.9, 0.99, 1.0)
# Anything else may be an outage, which must not silently remove hundreds of training tiles.
EXCLUDABLE_INDUSTRIAL_STATUSES = ("missing_item", "no_mask_polygon")
DIAGNOSTICS_NOTE = "reported only, with no threshold; these values cannot fail the stage"
# Beyond 11 a value is no Sentinel-2 scene classification class.
MAX_SCL_CLASS = 11


def _problem(check: str, items: list) -> list[str]:
    if not items:
        return []
    return [f"{check}: {len(items)} cases, for example {items[:MAX_EXAMPLES]}"]


def check_split(tiles: pd.DataFrame) -> list[str]:
    """Group integrity of the selection, plus tiles it needs and the store does not vouch for."""
    artisanal = tiles[tiles.scale == "Artisanal"]
    roles = artisanal.groupby("group").role.nunique()
    problems = _problem("group_leakage", roles[roles > 1].index.tolist())

    held_out = artisanal[artisanal.region.isin(garimpo_uncertainty.data.HELD_OUT_REGIONS)]
    problems += _problem("held_out_leak", held_out[held_out.role != "test"].tile_id.tolist())

    incomplete = artisanal[artisanal.status != "ok"]
    problems += _problem("partial_group", sorted(incomplete.group.unique().tolist()))
    absent = tiles[tiles.status == garimpo_uncertainty.data.NOT_ATTEMPTED]
    problems += _problem("not_downloaded", absent.tile_id.tolist())
    return problems


def check_split_counts(tiles: pd.DataFrame, summary: dict) -> list[str]:
    """Compare with the summary, its train role reduced to the fraction by data.scale_summary.

    Industrial tiles were never confirmed available, so one with no scene or no mask counts as
    excluded. Every other tile of the split must still be accounted for.
    """
    excluded = (tiles.scale == "Industrial") & tiles.status.isin(EXCLUDABLE_INDUSTRIAL_STATUSES)
    accounted = tiles[(tiles.status == "ok") | excluded].groupby(["scale", "role"]).size()
    wrong = []
    for name, expected in summary["tiles_by_scale_role"].items():
        if name.endswith("/dropped"):
            continue
        found = int(accounted.get(tuple(name.split("/")), 0))
        if found != expected:
            wrong.append(f"{name}: expected {expected}, accounted for {found}")
    return _problem("split_counts", wrong)


def check_store(store: garimpo_uncertainty.data.CellStore, cell_px: int) -> list[str]:
    n_cells = len(store.images)
    wanted = {
        "images": (np.uint16, (n_cells, len(garimpo_uncertainty.data.BANDS), cell_px, cell_px)),
        "masks": (np.uint8, (n_cells, cell_px, cell_px)),
        "scl": (np.uint8, (n_cells, cell_px, cell_px)),
    }
    wrong = [
        name
        for name, (dtype, shape) in wanted.items()
        if getattr(store, name).dtype != dtype or getattr(store, name).shape != shape
    ]
    if len(store.stats) != n_cells:
        wrong.append("stats")
    return _problem("store_layout", wrong)


def inspect_tile(
    store: garimpo_uncertainty.data.CellStore, slots: slice, boa_offset: int
) -> dict[str, np.ndarray]:
    """Recompute from the arrays what the writer claimed; the slots are the thing being checked."""
    images = store.images[slots].astype(np.float32)
    masks = store.masks[slots]
    scl = store.scl[slots]
    red = np.maximum(images[:, garimpo_uncertainty.data.RED_INDEX] + boa_offset, 0)
    nir = np.maximum(images[:, garimpo_uncertainty.data.NIR_INDEX] + boa_offset, 0)
    ndvi = np.divide(nir - red, nir + red, out=np.zeros_like(nir), where=(nir + red) > 0)
    inside = masks > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        # Mining scars are bare soil: a mask shifted off its scar pushes this toward zero.
        mean_inside = (ndvi * inside).sum(axis=(1, 2)) / inside.sum(axis=(1, 2))
        mean_outside = (ndvi * ~inside).sum(axis=(1, 2)) / (~inside).sum(axis=(1, 2))
    return {
        "mask_ha_recomputed": inside.sum(axis=(1, 2)) * garimpo_uncertainty.data.PIXEL_HA,
        "mask_binary": np.isin(masks, (0, 1)).all(axis=(1, 2)),
        "nodata_fraction": (images == 0).all(axis=1).mean(axis=(1, 2)),
        "cloud_fraction": np.isin(scl, garimpo_uncertainty.data.CLOUD_SCL_CLASSES).mean(
            axis=(1, 2)
        ),
        "scl_values_ok": (scl <= MAX_SCL_CLASS).all(axis=(1, 2)),
        "ndvi_mask_minus_outside": mean_inside - mean_outside,
    }


def check_cells(cells: pd.DataFrame) -> list[str]:
    named = {
        "mask_values": ~cells.mask_binary,
        "scl_values": ~cells.scl_values_ok,
        "mask_area": ~np.isclose(cells.mask_ha, cells.mask_ha_recomputed),
    }
    problems = []
    for check, failed in named.items():
        problems += _problem(check, cells[failed].cell_index.tolist())
    return problems


def check_tiles(tiles: pd.DataFrame) -> list[str]:
    # A box wider than the window sets its corners apart by construction; only the rest is an error.
    # Centered windows bound the rest at half a pixel, so this trips only on a defect in the
    # sidecar or in tile_origin; rotation shows in the tile_box_excess_m diagnostic instead.
    misalignment = tiles.tile_offset_m - tiles.tile_box_excess_m / 2
    pixel_m = tiles["transform"].map(lambda transform: abs(transform[0]))
    named = {
        "pixel_size": pixel_m != garimpo_uncertainty.data.PIXEL_M,
        "tile_offset": misalignment > garimpo_uncertainty.data.PIXEL_M,
    }
    problems = []
    for check, failed in named.items():
        problems += _problem(check, tiles[failed].tile_id.tolist())
    return problems


def unusable_store(root: Path, split: pd.DataFrame, tile_ids: list[int]) -> list[str]:
    """Return why the store cannot be read for this selection at all: an array gone, or a split
    that no longer matches its slots.
    """
    missing = garimpo_uncertainty.data.missing_arrays(root)
    if missing:
        return _problem("store_missing", missing)
    try:
        garimpo_uncertainty.data.cell_table(root, split, tile_ids)
    except ValueError as error:
        return [f"store_binding: {error}"]
    return []


def slot_mismatches(root: Path, tiles: pd.DataFrame, workers: int) -> list[str]:
    """Hash the slots of the given tile records against what each sidecar recorded when written.

    Reads every slot of those tiles and no others, so more tiles arriving elsewhere change nothing.
    """
    store = garimpo_uncertainty.data.open_store(root)
    usable = list(tiles[tiles.status == "ok"].itertuples())

    def matches(tile: tuple) -> bool:
        cells = int(tile.cells_per_tile)
        slots = slice(tile.tile_index * cells, (tile.tile_index + 1) * cells)
        return garimpo_uncertainty.data.slots_sha256(store, slots) == tile.slots_sha256

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        same = list(pool.map(matches, usable))
    wrong = [tile.tile_id for tile, equal in zip(usable, same, strict=True) if not equal]
    return _problem("slot_hash", wrong)


def _quantiles(values: pd.Series) -> dict[str, float]:
    return {f"q{int(q * 100)}": round(float(values.quantile(q)), 4) for q in QUANTILES}


def validate(
    root: Path,
    split: pd.DataFrame,
    tile_ids: list[int],
    cell_px: int,
    summary: dict | None,
    workers: int,
) -> tuple[dict, pd.DataFrame]:
    """Return the report and per-cell findings on the selected tiles; the store may hold more.

    Clouds and nodata are reported, never filtered. A missing artisanal tile fails, since all 232
    were confirmed available; an industrial tile with no scene or no mask is listed and the run
    proceeds without it, and any other industrial failure stops the run.
    """
    unusable = unusable_store(root, split, tile_ids)
    if unusable:
        return {"problems": unusable}, pd.DataFrame()
    tiles = garimpo_uncertainty.data.tile_records(root, split, tile_ids)
    failed = tiles[tiles.status != "ok"]
    artisanal_failed = failed[failed.scale == "Artisanal"]
    industrial_failed = failed[failed.scale == "Industrial"]
    excludable = industrial_failed.status.isin(EXCLUDABLE_INDUSTRIAL_STATUSES)
    problems = _problem("missing_scenes", artisanal_failed.tile_id.tolist())
    problems += _problem(
        "industrial_download_errors", industrial_failed[~excludable].tile_id.tolist()
    )
    problems += check_split(tiles)
    if summary is not None:
        problems += check_split_counts(tiles, summary)

    store = garimpo_uncertainty.data.open_store(root)
    problems += check_store(store, cell_px)
    ok_tiles = tiles[tiles.status == "ok"].reset_index(drop=True)
    problems += check_tiles(ok_tiles)
    problems += slot_mismatches(root, ok_tiles, workers)
    cells = garimpo_uncertainty.data.cell_table(root, split, tile_ids)
    n_per_tile = len(cells) // max(len(ok_tiles), 1)
    tile_chunks = [
        (slice(tile.tile_index * n_per_tile, (tile.tile_index + 1) * n_per_tile), tile.boa_offset)
        for tile in ok_tiles.itertuples()
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        chunks = list(pool.map(lambda chunk: inspect_tile(store, *chunk), tile_chunks))
    if chunks:
        columns = {name: np.concatenate([c[name] for c in chunks]) for name in chunks[0]}
        stats = store.stats[cells.cell_index.to_numpy()]
        cells = cells.assign(**columns, **{name: stats[name] for name in stats.dtype.names})
        problems += check_cells(cells)

    by_role = {}
    if chunks:
        for role, rows in cells.groupby("role"):
            role_tiles = ok_tiles[ok_tiles.role == role]
            by_role[role] = {
                "cells": len(rows),
                "groups": int(rows[rows.group >= 0].group.nunique()),
                "cells_100_percent_nodata": int((rows.nodata_fraction == 1).sum()),
                "cells_100_percent_cloud": int((rows.cloud_fraction == 1).sum()),
                "cloud_fraction": _quantiles(rows.cloud_fraction),
                "nodata_fraction": _quantiles(rows.nodata_fraction),
                "abs_mask_minus_vector_ha": _quantiles((rows.mask_ha - rows.vector_ha).abs()),
                "tile_box_excess_m": _quantiles(role_tiles.tile_box_excess_m),
                "ndvi_mask_minus_outside_median": round(
                    float(rows.ndvi_mask_minus_outside.median()), 4
                ),
            }
    report = {
        "problems": problems,
        "tiles_not_ok": json.loads(
            failed[["tile_id", "scale", "role", "group", "item", "status"]].to_json(
                orient="records"
            )
        ),
        "industrial_tiles_excluded": int(excludable.sum()),
        "diagnostics": {"note": DIAGNOSTICS_NOTE, "by_role": by_role},
    }
    return report, cells
