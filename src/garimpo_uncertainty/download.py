"""Login-node download: Jasansky et al. metadata and Sentinel-2 L2A cells from Planetary Computer.

Resumable: a tile is done once its sidecar says so, and the sidecar is written after its slots are
flushed. Tiles that fail keep their reason in the sidecar; nothing is dropped silently.
"""

import concurrent.futures
import hashlib
import http
import json
import logging
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import geopandas
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

logger = logging.getLogger(__name__)

ZENODO_URL = "https://zenodo.org/records/14195737/files/mining_area_data.zip?download=1"
ZENODO_MD5 = "1259ad8045ae648c547ef3db1fccab3b"
GPKG_MEMBER = "mining_area_data.gpkg"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"  # noqa: S105 - a URL, not a password
TOKEN_REFRESH_S = 20 * 60
SENTINEL2_MOCO_URL = (
    "https://hf.co/torchgeo/resnet50_sentinel2_all_moco/resolve/"
    "da4f3c9dbe09272eb902f3b37f46635fa4726879/resnet50_sentinel2_all_moco-df8b932e.pth"
)
SENTINEL2_MOCO_HASH = "df8b932e"
IMAGENET_URL = (
    "https://huggingface.co/smp-hub/resnet50.imagenet/resolve/"
    "00cb74e366966d59cd9a35af57e618af9f88efe9/model.safetensors"
)
IMAGENET_HASH = "df1aad85"
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


def download_weights(root: Path, encoder_weights: list[str]) -> None:
    """Fetch the chosen starting weights under the data root, for offline training."""
    import torch.hub  # noqa: PLC0415 - only this command needs torch; the others run without it

    sources = {
        "imagenet": (IMAGENET_URL, IMAGENET_HASH),
        "sentinel2_moco": (SENTINEL2_MOCO_URL, SENTINEL2_MOCO_HASH),
    }
    for choice in encoder_weights:
        target = garimpo_uncertainty.data.starting_weights_path(root, choice)
        if not target.exists():
            url, hash_prefix = sources[choice]
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(url, str(target), hash_prefix=hash_prefix)


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for block in iter(lambda: file.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _get_json(url: str) -> dict:
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - callers fix an https prefix
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != http.HTTPStatus.TOO_MANY_REQUESTS:
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
    """Offset of a window centered on the polygon, on the 10 m grid, so no 10 m band resamples.

    Also returns the corner offset in meters and how much wider the polygon's box is than the
    window. A box wider by e meters puts the centered corners e / 2 away by construction, which is
    what the validation discounts before judging misalignment.
    """
    x0, y0, x1, y1 = bounds
    pixel = garimpo_uncertainty.data.PIXEL_M
    col = round(((x0 + x1) / 2 - grid.c) / pixel - tile_px / 2)
    row = round((grid.f - (y0 + y1) / 2) / pixel - tile_px / 2)
    offset = max(abs(grid.c + col * pixel - x0), abs(grid.f - row * pixel - y1))
    window_m = tile_px * pixel
    excess = max(abs((x1 - x0) - window_m), abs((y1 - y0) - window_m))
    return col, row, float(offset), float(excess)


def create_store(root: Path, n_tiles: int, tile_px: int, cell_px: int) -> None:
    """Allocate every slot of the store once, so slots never move; refuse a store that cannot match.

    open_memmap extends the file by seeking past its end and writing one byte
    (numpy/_core/memmap.py), so unwritten slots stay holes where the filesystem supports them.
    """
    missing = garimpo_uncertainty.data.missing_arrays(root)
    sidecars = (root / "tiles").glob("*.json")
    if missing and any(json.loads(path.read_text()).get("status") == "ok" for path in sidecars):
        raise ValueError(
            f"{missing} are missing from {root} while tiles/ has sidecars with status ok: "
            "recreated, they would serve zeros as those tiles; restore them, or empty the store "
            "and tiles/"
        )
    n_cells = n_tiles * (tile_px // cell_px) ** 2
    pixels = (n_cells, cell_px, cell_px)
    layout = {
        "images": (np.uint16, (n_cells, len(garimpo_uncertainty.data.BANDS), cell_px, cell_px)),
        "masks": (np.uint8, pixels),
        "scl": (np.uint8, pixels),
        "stats": (garimpo_uncertainty.data.CELL_STATS_DTYPE, (n_cells,)),
    }
    root.mkdir(parents=True, exist_ok=True)
    for name, (dtype, shape) in layout.items():
        path = root / f"{name}.npy"
        if not path.exists():
            with garimpo_uncertainty.stages.atomic(path) as temporary:
                np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape).flush()
        elif np.load(path, mmap_mode="r").shape != shape:
            raise ValueError(
                f"{path} holds {np.load(path, mmap_mode='r').shape[0]} cells, but this split's "
                f"{n_tiles} tiles need {n_cells}: restore the split the store was written with, "
                "or empty the store and tiles/"
            )


@dataclass(frozen=True)
class CutTile:
    """A tile's pixels and its mask geometry, both on the tile's own grid."""

    image: np.ndarray
    scl: np.ndarray
    geometry: shapely.Geometry
    transform: rasterio.Affine


def _rasterized_mask(
    geometry: shapely.Geometry, transform: rasterio.Affine, shape: tuple[int, int]
) -> np.ndarray:
    if geometry.is_empty:
        return np.zeros(shape, dtype=np.uint8)
    return rasterio.features.rasterize(
        [geometry],
        out_shape=shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype="uint8",
    )


def _cell_windows(height: int, width: int, cell_px: int) -> list[rasterio.windows.Window]:
    return [
        rasterio.windows.Window(col * cell_px, row * cell_px, cell_px, cell_px)
        for row in range(height // cell_px)
        for col in range(width // cell_px)
    ]


def write_cells(
    store: garimpo_uncertainty.data.CellStore, tile_index: int, tile: CutTile, cell_px: int
) -> dict[str, int | str]:
    """Rasterize the mask on the image's own grid, then fill and flush this tile's slots.

    Returns what the tile's sidecar records to bind itself to those slots.
    """
    height, width = tile.image.shape[1:]
    geometry = shapely.make_valid(tile.geometry)
    mask = _rasterized_mask(geometry, tile.transform, (height, width))
    windows = _cell_windows(height, width, cell_px)
    first_slot = tile_index * len(windows)
    for slot, window in enumerate(windows, start=first_slot):
        rows, cols = window.toslices()
        store.images[slot] = tile.image[:, rows, cols]
        store.masks[slot] = mask[rows, cols]
        store.scl[slot] = tile.scl[rows, cols]
        box = shapely.box(*rasterio.windows.bounds(window, tile.transform))
        store.stats[slot] = (
            mask[rows, cols].sum() * garimpo_uncertainty.data.PIXEL_HA,
            geometry.intersection(box).area / 1e4,
        )
    for array in store:
        array.flush()
    return {
        "tile_index": tile_index,
        "cells_per_tile": len(windows),
        "slots_sha256": garimpo_uncertainty.data.slots_sha256(
            store, slice(first_slot, first_slot + len(windows))
        ),
    }


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


def _check_inside_scene(
    col: int, row: int, tile_px: int, scene_width: int, scene_height: int
) -> None:
    inside = (
        col >= 0 and row >= 0 and col + tile_px <= scene_width and row + tile_px <= scene_height
    )
    if not inside:
        raise ValueError(f"tile window ({col}, {row}) leaves the scene")


@dataclass(frozen=True)
class TileSource:
    """Where one tile comes from: its slot, its Sentinel-2 item and its geometries in WGS 84."""

    tile_id: int
    tile_index: int
    item_id: str
    tile_geometry: shapely.Geometry
    mask_geometry: shapely.Geometry


@dataclass(frozen=True)
class TilePlacement:
    """A tile's window on its scene's 10 m grid, with the offsets the sidecar records."""

    epsg: int
    transform: rasterio.Affine
    offset_m: float
    excess_m: float


def _to_utm(geometry: shapely.Geometry, epsg: int) -> shapely.Geometry:
    return geopandas.GeoSeries([geometry], crs=4326).to_crs(epsg).iloc[0]


def _place_tile(reference_href: str, tile_geometry: shapely.Geometry) -> TilePlacement:
    tile_px = garimpo_uncertainty.data.TILE_PX
    with rasterio.open(reference_href) as reference:
        epsg = reference.crs.to_epsg()
        grid, scene_width, scene_height = reference.transform, reference.width, reference.height
    col, row, offset, excess = tile_origin(grid, _to_utm(tile_geometry, epsg).bounds, tile_px)
    _check_inside_scene(col, row, tile_px, scene_width, scene_height)
    return TilePlacement(epsg, grid * rasterio.Affine.translation(col, row), offset, excess)


def _read_bands(
    item: dict,
    transform: rasterio.Affine,
    token: _Token,
    resampling: rasterio.enums.Resampling,
) -> tuple[np.ndarray, np.ndarray]:
    tile_px = garimpo_uncertainty.data.TILE_PX
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
    return image, scl


def _cut_and_store(
    root: Path,
    store: garimpo_uncertainty.data.CellStore,
    source: TileSource,
    item: dict,
    *,
    token: _Token,
    resampling: rasterio.enums.Resampling,
) -> dict:
    """Read the item's bands into the tile's slots; return what the sidecar records of them."""
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MAX_RETRY": "5"}
    with rasterio.Env(**env):
        placement = _place_tile(_signed(item["assets"]["B02"]["href"], token), source.tile_geometry)
        image, scl = _read_bands(item, placement.transform, token, resampling)
    mask_utm = _to_utm(source.mask_geometry, placement.epsg)
    # A forced re-cut overwrites slots in place: a crash midway must leave no `ok` vouching.
    (root / "tiles" / f"{source.tile_id}.json").unlink(missing_ok=True)
    bound = write_cells(
        store,
        source.tile_index,
        CutTile(image, scl, mask_utm, placement.transform),
        garimpo_uncertainty.data.CELL_PX,
    )
    return bound | {
        "status": "ok",
        "epsg": placement.epsg,
        "transform": list(placement.transform)[:6],
        "tile_offset_m": placement.offset_m,
        "tile_box_excess_m": placement.excess_m,
    }


def process_tile(
    root: Path,
    store: garimpo_uncertainty.data.CellStore,
    source: TileSource,
    token: _Token,
    resampling: rasterio.enums.Resampling,
) -> str:
    """Download one tile. Every outcome ends in its sidecar; the caller reports statuses."""
    record: dict = {"item": source.item_id, "resampling_20m": resampling.name}
    try:
        item = _get_json(STAC + source.item_id)
        record |= {
            "scene_cloud": item["properties"].get("eo:cloud_cover"),
            "boa_offset": _boa_offset(item),
        }
        record |= _cut_and_store(root, store, source, item, token=token, resampling=resampling)
    except urllib.error.HTTPError as error:
        status = "missing_item" if error.code == http.HTTPStatus.NOT_FOUND else f"http_{error.code}"
        record |= {"status": status, "detail": str(error)}
    # Thread boundary: one bad tile must end in the report, not abort the other tiles.
    except Exception as error:  # noqa: BLE001 - the thread boundary stated above
        record |= {"status": f"error_{type(error).__name__}", "detail": str(error)}
    write_sidecar(root, source.tile_id, record)
    return record["status"]


def _read_mask_geometries(root: Path) -> geopandas.GeoSeries:
    masks = (
        geopandas.read_file(gpkg_path(root), layer="preferred_polygons")
        .set_index("tile_id")
        .geometry
    )
    if not masks.index.is_unique:
        raise ValueError("preferred_polygons has more than one row for a tile")
    return masks


def _finished_tiles(root: Path, selected: pd.DataFrame) -> dict[int, dict]:
    """Return the sidecars with status ok of the selected tiles, by tile id."""
    finished = {}
    for tile_id in selected.tile_id.astype(int):
        sidecar = root / "tiles" / f"{tile_id}.json"
        record = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        if record.get("status") == "ok":
            finished[tile_id] = record
    return finished


def _refuse_other_resampling(
    finished: dict[int, dict], resampling: rasterio.enums.Resampling
) -> None:
    mismatched = [
        tile_id
        for tile_id, record in finished.items()
        if record.get("resampling_20m") != resampling.name
    ]
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} tiles were cut with another 20 m resampling than "
            f"{resampling.name}, for example {mismatched[:5]}; use --force to redo them"
        )


def download_scenes(
    root: Path,
    split: pd.DataFrame,
    selected: pd.DataFrame,
    workers: int,
    resampling: rasterio.enums.Resampling,
    *,
    force: bool = False,
) -> None:
    """Fetch pending tiles; `force` redoes every selected tile, so a cutting fix has a path."""
    tiles = geopandas.read_file(gpkg_path(root), layer="tiles").set_index("tile_id")
    masks = _read_mask_geometries(root)
    token = _Token()
    order = garimpo_uncertainty.data.tile_table(split).set_index("tile_id").tile_index
    create_store(
        root, len(order), garimpo_uncertainty.data.TILE_PX, garimpo_uncertainty.data.CELL_PX
    )
    store = garimpo_uncertainty.data.open_store(root, "r+")
    # Every tile, not only the selected ones: a moved tile of another fraction would be overwritten.
    garimpo_uncertainty.data.tile_records(root, split, order.index.tolist())

    finished = _finished_tiles(root, selected)
    if not force:
        _refuse_other_resampling(finished, resampling)

    def one(tile_id: int) -> tuple[int, str]:
        if tile_id not in masks.index:
            write_sidecar(root, tile_id, {"status": "no_mask_polygon"})
            return tile_id, "no_mask_polygon"
        tile = tiles.loc[tile_id]
        source = TileSource(
            tile_id, int(order[tile_id]), tile.s2_tile_id, tile.geometry, masks.loc[tile_id]
        )
        return tile_id, process_tile(root, store, source, token, resampling)

    pending = [int(t) for t in selected.tile_id if force or int(t) not in finished]
    logger.info("%s tiles already done, %s pending", len(selected) - len(pending), len(pending))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for count, (tile_id, status) in enumerate(pool.map(one, pending), start=1):
            logger.info("[%s/%s] tile %s: %s", count, len(pending), tile_id, status)
