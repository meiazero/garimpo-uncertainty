"""Login-node download: Jasansky et al. metadata and Sentinel-2 L2A cells from Planetary Computer.

Resumable: a tile is done once its sidecar says so, and the sidecar is written last. Tiles that
fail stay in the report with their reason; nothing is dropped silently.
"""

import concurrent.futures
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.enums
import rasterio.features
import rasterio.transform
import rasterio.windows
import shapely

import garimpo_uncertainty.data
import garimpo_uncertainty.stages

ZENODO_URL = "https://zenodo.org/records/14195737/files/mining_area_data.zip?download=1"
ZENODO_MD5 = "1259ad8045ae648c547ef3db1fccab3b"  # from scripts/proposal/get_data.sh
GPKG_MEMBER = "mining_area_data.gpkg"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
TOKEN_REFRESH_S = 20 * 60
ASSETS_10M = {"B02", "B03", "B04", "B08"}
BASELINE_WITH_OFFSET = 4.0
BOA_OFFSET_DN = -1000


def gpkg_path(root: Path) -> Path:
    return root / "zenodo" / GPKG_MEMBER


def download_metadata(root: Path) -> None:
    """Stop at the md5: a wrong archive would silently change every mask."""
    archive = root / "zenodo" / "mining_area_data.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not (archive.exists() and _md5(archive) == ZENODO_MD5):
        with garimpo_uncertainty.stages.atomic(archive) as temporary:
            with urllib.request.urlopen(ZENODO_URL, timeout=120) as response:
                temporary.write_bytes(response.read())
            if _md5(temporary) != ZENODO_MD5:
                raise ValueError(f"md5 mismatch for {ZENODO_URL}")
    if not gpkg_path(root).exists():
        with zipfile.ZipFile(archive) as bundle:
            bundle.extract(GPKG_MEMBER, archive.parent)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _get_json(url: str) -> dict:
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            time.sleep(int(error.headers.get("Retry-After") or 2**attempt))
    raise TimeoutError(f"still rate limited: {url}")


class _Token:
    """SAS token shared across threads.

    ponytail: fixed refresh interval instead of parsing `msft:expiry`; assumes tokens outlive it.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.value = ""
        self.fetched_at = float("-inf")

    def get(self) -> str:
        with self.lock:
            if time.monotonic() - self.fetched_at > TOKEN_REFRESH_S:
                self.value = _get_json(TOKEN_URL)["token"]
                self.fetched_at = time.monotonic()
            return self.value


def tile_origin(
    grid: rasterio.Affine, bounds: tuple[float, float, float, float], tile_px: int
) -> tuple[int, int, float, float]:
    """Pixel offset of a tile window centered on the polygon, snapped to the 10 m grid.

    The window never resamples the 10 m bands. Also returns the corner offset in meters and how much
    wider the polygon's box is than the window. A box wider by e meters puts the centered corners
    e / 2 away by construction, which is what the validation discounts before judging misalignment.
    """
    x0, y0, x1, y1 = bounds
    pixel = garimpo_uncertainty.data.PIXEL_M
    col = round(((x0 + x1) / 2 - grid.c) / pixel - tile_px / 2)
    row = round((grid.f - (y0 + y1) / 2) / pixel - tile_px / 2)
    offset = max(abs(grid.c + col * pixel - x0), abs(grid.f - row * pixel - y1))
    window_m = tile_px * pixel
    excess = max(abs((x1 - x0) - window_m), abs((y1 - y0) - window_m))
    return col, row, float(offset), float(excess)


def write_cells(
    root: Path,
    tile_id: int,
    image: np.ndarray,
    scl: np.ndarray,
    geometry: shapely.Geometry,
    transform: rasterio.Affine,
    epsg: int,
    cell_px: int,
) -> list[dict]:
    """Rasterizes the mask on the image's own grid, cuts the tile, returns per-cell statistics.

    Band order is BANDS + mask + SCL, all uint16. Each file is renamed into place when complete.
    """
    height, width = image.shape[1:]
    geometry = shapely.make_valid(geometry)
    mask = np.zeros((height, width), dtype=np.uint8)
    if not geometry.is_empty:
        mask = rasterio.features.rasterize(
            [geometry],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            default_value=1,
            dtype="uint8",
        )
    stack = np.concatenate([image, mask[None], scl[None]]).astype(np.uint16)
    records = []
    for row in range(height // cell_px):
        for col in range(width // cell_px):
            window = rasterio.windows.Window(col * cell_px, row * cell_px, cell_px, cell_px)
            cell = stack[
                :, row * cell_px : (row + 1) * cell_px, col * cell_px : (col + 1) * cell_px
            ]
            cell_transform = rasterio.windows.transform(window, transform)
            relative = Path("cells") / str(tile_id) / f"{row}_{col}.tif"
            _write_cell(root / relative, cell, cell_transform, epsg)
            box = shapely.box(*rasterio.windows.bounds(window, transform))
            n_bands = len(garimpo_uncertainty.data.BANDS)
            records.append(
                {
                    "cell_id": f"{tile_id}_{row}_{col}",
                    "row": row,
                    "col": col,
                    "path": str(relative),
                    "mask_ha": float(cell[n_bands].sum() * garimpo_uncertainty.data.PIXEL_HA),
                    "vector_ha": float(geometry.intersection(box).area / 1e4),
                    "nodata_fraction": float((cell[:n_bands] == 0).all(axis=0).mean()),
                    "cloud_fraction": float(
                        np.isin(
                            cell[n_bands + 1], garimpo_uncertainty.data.CLOUD_SCL_CLASSES
                        ).mean()
                    ),
                }
            )
    return records


def _write_cell(path: Path, cell: np.ndarray, transform: rasterio.Affine, epsg: int) -> None:
    with garimpo_uncertainty.stages.atomic(path) as temporary:
        with rasterio.open(
            temporary,
            "w",
            driver="GTiff",
            height=cell.shape[1],
            width=cell.shape[2],
            count=cell.shape[0],
            dtype="uint16",
            crs=f"EPSG:{epsg}",
            transform=transform,
            compress="zstd",
            predictor=2,
        ) as target:
            target.write(cell)


def write_sidecar(root: Path, tile_id: int, record: dict) -> None:
    garimpo_uncertainty.stages.write_json(root / "tiles" / f"{tile_id}.json", record)


def _signed(href: str, token: _Token) -> str:
    return f"{href}?{token.get()}"


def _read_band(
    href: str,
    bounds: tuple[float, float, float, float],
    tile_px: int,
    resampling: rasterio.enums.Resampling,
) -> np.ndarray:
    with rasterio.open(href) as source:
        window = rasterio.windows.from_bounds(*bounds, transform=source.transform)
        data = source.read(1, window=window, out_shape=(tile_px, tile_px), resampling=resampling)
    return data.astype(np.uint16)


def _boa_offset(item: dict) -> int:
    baseline = item["properties"]["s2:processing_baseline"]
    return BOA_OFFSET_DN if float(baseline) >= BASELINE_WITH_OFFSET else 0


def process_tile(
    root: Path,
    tile_id: int,
    item_id: str,
    tile_geometry: shapely.Geometry,
    mask_geometry: shapely.Geometry,
    token: _Token,
    resampling: rasterio.enums.Resampling,
) -> str:
    """Download one tile. Every outcome ends in its sidecar; the caller reports statuses."""
    record: dict = {"item": item_id, "resampling_20m": resampling.name}
    try:
        item = _get_json(STAC + item_id)
        record |= {
            "scene_cloud": item["properties"].get("eo:cloud_cover"),
            "boa_offset": _boa_offset(item),
        }
        tile_px = garimpo_uncertainty.data.TILE_PX
        env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MAX_RETRY": "5"}
        with rasterio.Env(**env):
            with rasterio.open(_signed(item["assets"]["B02"]["href"], token)) as reference:
                epsg = reference.crs.to_epsg()
                grid, scene_width, scene_height = (
                    reference.transform,
                    reference.width,
                    reference.height,
                )
            bounds_utm = gpd.GeoSeries([tile_geometry], crs=4326).to_crs(epsg).iloc[0].bounds
            col, row, offset, excess = tile_origin(grid, bounds_utm, tile_px)
            inside = (
                col >= 0
                and row >= 0
                and col + tile_px <= scene_width
                and row + tile_px <= scene_height
            )
            if not inside:
                raise ValueError(f"tile window ({col}, {row}) leaves the scene")
            transform = grid * rasterio.Affine.translation(col, row)
            bounds = rasterio.transform.array_bounds(tile_px, tile_px, transform)
            image = np.stack(
                [
                    _read_band(
                        _signed(item["assets"][band]["href"], token),
                        bounds,
                        tile_px,
                        rasterio.enums.Resampling.nearest if band in ASSETS_10M else resampling,
                    )
                    for band in garimpo_uncertainty.data.BANDS
                ]
            )
            scl = _read_band(
                _signed(item["assets"]["SCL"]["href"], token),
                bounds,
                tile_px,
                rasterio.enums.Resampling.nearest,
            )
        mask_utm = gpd.GeoSeries([mask_geometry], crs=4326).to_crs(epsg).iloc[0]
        cells = write_cells(
            root,
            tile_id,
            image,
            scl,
            mask_utm,
            transform,
            epsg,
            garimpo_uncertainty.data.CELL_PX,
        )
        record |= {
            "status": "ok",
            "epsg": epsg,
            "tile_offset_m": offset,
            "tile_box_excess_m": excess,
            "cells": cells,
        }
    except urllib.error.HTTPError as error:
        status = "missing_item" if error.code == 404 else f"http_{error.code}"
        record |= {"status": status, "detail": str(error)}
    # Thread boundary: one bad tile must end in the report, not abort the other tiles.
    except Exception as error:
        record |= {"status": f"error_{type(error).__name__}", "detail": str(error)}
    write_sidecar(root, tile_id, record)
    return record["status"]


def download_scenes(
    root: Path,
    selected: pd.DataFrame,
    workers: int,
    resampling: rasterio.enums.Resampling,
    force: bool = False,
) -> None:
    """Fetch pending tiles; `force` redoes every selected tile, so a cutting fix has a path."""
    tiles = gpd.read_file(gpkg_path(root), layer="tiles").set_index("tile_id")
    masks = gpd.read_file(gpkg_path(root), layer="preferred_polygons").set_index("tile_id").geometry
    if not masks.index.is_unique:
        raise ValueError("preferred_polygons has more than one row for a tile")
    token = _Token()

    def finished_record(tile_id: int) -> dict | None:
        sidecar = root / "tiles" / f"{tile_id}.json"
        record = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        return record if record.get("status") == "ok" else None

    finished = {int(t): finished_record(int(t)) for t in selected.tile_id}
    finished = {tile_id: record for tile_id, record in finished.items() if record}
    mismatched = [
        tile_id
        for tile_id, record in finished.items()
        if record.get("resampling_20m") != resampling.name
    ]
    if mismatched and not force:
        raise ValueError(
            f"{len(mismatched)} tiles were cut with another 20 m resampling than "
            f"{resampling.name}, for example {mismatched[:5]}; use --force to redo them"
        )

    def one(tile_id: int) -> tuple[int, str]:
        if tile_id not in masks.index:
            write_sidecar(root, tile_id, {"status": "no_mask_polygon"})
            return tile_id, "no_mask_polygon"
        tile = tiles.loc[tile_id]
        status = process_tile(
            root, tile_id, tile.s2_tile_id, tile.geometry, masks.loc[tile_id], token, resampling
        )
        return tile_id, status

    pending = [int(t) for t in selected.tile_id if force or int(t) not in finished]
    print(f"{len(selected) - len(pending)} tiles already done, {len(pending)} pending")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for count, (tile_id, status) in enumerate(pool.map(one, pending), start=1):
            print(f"[{count}/{len(pending)}] tile {tile_id}: {status}", flush=True)
    write_manifest(root, selected)


def _write_if_changed(path: Path, frame: pd.DataFrame) -> None:
    """Keep the file's mtime when nothing changed, so a rerun does not refuse later stages."""
    content = frame.to_csv(index=False)
    if path.exists() and path.read_text() == content:
        return
    with garimpo_uncertainty.stages.atomic(path) as temporary:
        temporary.write_text(content)


def write_manifest(root: Path, selected: pd.DataFrame) -> pd.DataFrame:
    """Write manifest.csv, one row per cell, and download_report.csv, one row per expected tile."""
    cells, report = [], []
    for tile in selected.itertuples():
        sidecar = root / "tiles" / f"{tile.tile_id}.json"
        record = (
            json.loads(sidecar.read_text()) if sidecar.exists() else {"status": "not_attempted"}
        )
        group = -1 if pd.isna(tile.cluster_50km) else int(tile.cluster_50km)
        region = "" if pd.isna(tile.region) else tile.region
        identity = {
            "tile_id": tile.tile_id,
            "role": tile.role,
            "scale": tile.scale,
            "region": region,
            "group": group,
        }
        report.append(
            identity
            | {
                "item": record.get("item", ""),
                "status": record["status"],
                "detail": record.get("detail", ""),
            }
        )
        if record["status"] == "ok":
            # resampling_20m makes a forced re-cut change the manifest and reach later stages.
            tile_values = {
                key: record[key]
                for key in (
                    "item",
                    "epsg",
                    "boa_offset",
                    "tile_offset_m",
                    "tile_box_excess_m",
                    "resampling_20m",
                )
            }
            cells += [identity | tile_values | cell for cell in record["cells"]]
    report_frame = pd.DataFrame(report)
    _write_if_changed(root / "download_report.csv", report_frame)
    _write_if_changed(root / "manifest.csv", pd.DataFrame(cells))
    print(report_frame.groupby(["scale", "role", "status"]).size().to_string())
    return report_frame
