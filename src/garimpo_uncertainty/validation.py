"""Checks the downloaded cells against the split.

Failing checks are named in `problems` and fail the stage. Everything under `diagnostics` is
reported with no threshold and cannot fail the stage: clouds, nodata, raster versus vector area,
box excess, NDVI contrast.
"""

import concurrent.futures
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

import garimpo_uncertainty.data

MAX_EXAMPLES = 5
QUANTILES = (0.0, 0.5, 0.9, 0.99, 1.0)
# Anything else may be an outage, which must not silently remove hundreds of training tiles.
EXCLUDABLE_INDUSTRIAL_STATUSES = ("missing_item", "no_mask_polygon")
DIAGNOSTICS_NOTE = "reported only, with no threshold; these values cannot fail the stage"


def _problem(check: str, items: list) -> list[str]:
    if not items:
        return []
    return [f"{check}: {len(items)} cases, for example {items[:MAX_EXAMPLES]}"]


def check_split(manifest: pd.DataFrame, report: pd.DataFrame, split: pd.DataFrame) -> list[str]:
    """Group integrity against the versioned split, so a subset must still keep whole groups."""
    artisanal = manifest[manifest.scale == "Artisanal"]
    roles = artisanal.groupby("group").role.nunique()
    problems = _problem("group_leakage", roles[roles > 1].index.tolist())

    split_roles = split.set_index("tile_id").role
    tiles = manifest.drop_duplicates("tile_id")
    wrong_role = tiles[tiles.tile_id.map(split_roles) != tiles.role].tile_id.tolist()
    problems += _problem("role_mismatch", wrong_role)

    held_out = artisanal[artisanal.region.isin(garimpo_uncertainty.data.HELD_OUT_REGIONS)]
    problems += _problem(
        "held_out_leak", held_out[held_out.role != "test"].tile_id.unique().tolist()
    )

    expected = split[split.cluster_50km.isin(report[report.scale == "Artisanal"].group)]
    present = set(manifest.tile_id)
    incomplete = expected[~expected.tile_id.isin(present)]
    problems += _problem("partial_group", sorted(incomplete.cluster_50km.unique().tolist()))
    return problems


def check_split_counts(manifest: pd.DataFrame, report: pd.DataFrame, summary: dict) -> list[str]:
    """Only meaningful on the full dataset; a pilot subset does not pass this argument.

    Industrial tiles were never confirmed available, so one with no scene or no mask counts as
    excluded. Every other tile of the split must still be accounted for.
    """
    counts = manifest.drop_duplicates("tile_id").groupby(["scale", "role"]).size()
    excludable = report.status.isin(EXCLUDABLE_INDUSTRIAL_STATUSES)
    excluded = report[excludable & (report.scale == "Industrial")]
    excluded_counts = excluded.groupby(["scale", "role"]).size()
    wrong = []
    for name, expected in summary["tiles_by_scale_role"].items():
        if name.endswith("/dropped"):
            continue
        key = tuple(name.split("/"))
        found = int(counts.get(key, 0)) + int(excluded_counts.get(key, 0))
        if found != expected:
            wrong.append(f"{name}: expected {expected}, accounted for {found}")
    return _problem("split_counts", wrong)


def inspect_cell(root: Path, cell: dict, cell_px: int) -> dict:
    """Recomputes from the file what the manifest claims; the file is the thing being checked."""
    with rasterio.open(root / cell["path"]) as source:
        stack = source.read()
        result = {
            "cell_id": cell["cell_id"],
            "n_bands": source.count,
            "shape_ok": source.height == cell_px and source.width == cell_px,
            "dtype_ok": all(dtype == "uint16" for dtype in source.dtypes),
            "epsg": source.crs.to_epsg(),
            "pixel_m": abs(source.transform.a),
        }
    n_bands = len(garimpo_uncertainty.data.BANDS)
    mask = stack[garimpo_uncertainty.data.MASK_BAND - 1]
    scl = stack[garimpo_uncertainty.data.SCL_BAND - 1]
    image = stack[:n_bands].astype(np.float32)
    red = np.maximum(image[garimpo_uncertainty.data.RED_INDEX] + cell["boa_offset"], 0)
    nir = np.maximum(image[garimpo_uncertainty.data.NIR_INDEX] + cell["boa_offset"], 0)
    ndvi = np.divide(nir - red, nir + red, out=np.zeros_like(nir), where=(nir + red) > 0)
    inside = mask > 0
    result |= {
        "mask_ha_file": float(inside.sum() * garimpo_uncertainty.data.PIXEL_HA),
        "mask_binary": bool(np.isin(mask, (0, 1)).all()),
        "nodata_fraction": float((image == 0).all(axis=0).mean()),
        "cloud_fraction": float(np.isin(scl, garimpo_uncertainty.data.CLOUD_SCL_CLASSES).mean()),
        "scl_values_ok": bool((scl <= 11).all()),
        # Mining scars are bare soil: a mask shifted off its scar pushes this toward zero.
        "ndvi_mask_minus_outside": float(ndvi[inside].mean() - ndvi[~inside].mean())
        if inside.any() and (~inside).any()
        else float("nan"),
    }
    return result


def check_cells(manifest: pd.DataFrame, inspected: pd.DataFrame) -> list[str]:
    merged = manifest.merge(inspected, on="cell_id", suffixes=("", "_file"))
    # A box wider than the window sets its corners apart by construction; only the rest is an error.
    # Centered windows bound the rest at half a pixel, so this trips only on a defect in the
    # manifest or in tile_origin; rotation shows in the tile_box_excess_m diagnostic instead.
    misalignment = merged.tile_offset_m - merged.tile_box_excess_m / 2
    named = {
        "band_count": merged.n_bands != garimpo_uncertainty.data.N_CELL_BANDS,
        "cell_shape": ~merged.shape_ok,
        "dtype": ~merged.dtype_ok,
        "crs": merged.epsg != merged.epsg_file,
        "pixel_size": merged.pixel_m != garimpo_uncertainty.data.PIXEL_M,
        "mask_values": ~merged.mask_binary,
        "scl_values": ~merged.scl_values_ok,
        "mask_area": ~np.isclose(merged.mask_ha, merged.mask_ha_file),
        "tile_offset": misalignment > garimpo_uncertainty.data.PIXEL_M,
    }
    problems = []
    for check, failed in named.items():
        problems += _problem(check, merged[failed].cell_id.tolist())
    problems += _problem(
        "duplicate_cells", manifest[manifest.cell_id.duplicated()].cell_id.tolist()
    )
    return problems


def _quantiles(values: pd.Series) -> dict[str, float]:
    return {f"q{int(q * 100)}": round(float(values.quantile(q)), 4) for q in QUANTILES}


def validate(
    root: Path, split: pd.DataFrame, cell_px: int, summary: dict | None, workers: int
) -> tuple[dict, pd.DataFrame]:
    """Return the report and per-cell findings. Clouds and nodata are reported, never filtered.

    A missing artisanal tile fails, since all 232 were confirmed available. An industrial tile with
    no scene or no mask is listed and the run proceeds without it; any other industrial failure
    stops the run. Group leakage always fails.
    """
    manifest = garimpo_uncertainty.data.read_manifest(root)
    download_report = pd.read_csv(root / "download_report.csv")

    failed = download_report[download_report.status != "ok"]
    artisanal_failed = failed[failed.scale == "Artisanal"]
    industrial_failed = failed[failed.scale == "Industrial"]
    excludable = industrial_failed.status.isin(EXCLUDABLE_INDUSTRIAL_STATUSES)
    problems = _problem("missing_scenes", artisanal_failed.tile_id.tolist())
    problems += _problem(
        "industrial_download_errors", industrial_failed[~excludable].tile_id.tolist()
    )
    problems += check_split(manifest, download_report, split)
    if summary is not None:
        problems += check_split_counts(manifest, download_report, summary)

    cells = manifest.to_dict("records")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        inspected = pd.DataFrame(list(pool.map(lambda c: inspect_cell(root, c, cell_px), cells)))
    problems += check_cells(manifest, inspected)

    merged = manifest.merge(inspected, on="cell_id", suffixes=("", "_file"))
    by_role = {
        role: {
            "cells": len(rows),
            "groups": int(rows[rows.group >= 0].group.nunique()),
            "cells_100_percent_nodata": int((rows.nodata_fraction_file == 1).sum()),
            "cells_100_percent_cloud": int((rows.cloud_fraction_file == 1).sum()),
            "cloud_fraction": _quantiles(rows.cloud_fraction_file),
            "nodata_fraction": _quantiles(rows.nodata_fraction_file),
            "abs_mask_minus_vector_ha": _quantiles((rows.mask_ha - rows.vector_ha).abs()),
            "tile_box_excess_m": _quantiles(rows.tile_box_excess_m),
            "ndvi_mask_minus_outside_median": round(
                float(rows.ndvi_mask_minus_outside.median()), 4
            ),
        }
        for role, rows in merged.groupby("role")
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
    return report, merged
