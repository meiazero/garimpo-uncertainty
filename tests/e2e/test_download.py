"""The download code on local rasters, with a fake Planetary Computer serving them."""

import hashlib
import json
import runpy
import sys
import urllib.error
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import rasterio.enums
import rasterio.transform
import shapely
import shapely.affinity

import garimpo_uncertainty.data
import garimpo_uncertainty.download
import garimpo_uncertainty.validation
from tests.e2e import support

download = garimpo_uncertainty.download

DOWNLOAD_ARTIFACTS = support.ROOT / "tests" / "artifacts" / "download"
DATA = DOWNLOAD_ARTIFACTS / "data"
LISTS = DOWNLOAD_ARTIFACTS / "lists"
SCENE_ORIGIN = (600_000.0, 9_100_000.0)
SCENE_PX = 2_200
EPSG = support.EPSG
# Same 20,511 m box against a 20,480 m window as tile 684.
TILE_BOX = shapely.box(600_484.5, 9_079_004.5, 620_995.5, 9_099_515.5)
MASK = shapely.box(603_000, 9_090_000, 606_000, 9_093_000)
TILE_PX = garimpo_uncertainty.data.TILE_PX
CELL_PX = garimpo_uncertainty.data.CELL_PX
NEAREST = rasterio.enums.Resampling.nearest
BILINEAR = rasterio.enums.Resampling.bilinear
SPLIT_COUNTS = {"Artisanal/test": 1, "Industrial/train": 1, "Industrial/dropped": 5}
SPLIT = pd.DataFrame(
    {
        "tile_id": range(4),
        "region": ["Amazonia BR", None, "Asia/Oceania", "Asia/Oceania"],
        "cluster_50km": [0.0, None, 1.0, 2.0],
        "role": ["test", "train", "calibration", "train"],
        "scale": ["Artisanal", "Industrial", "Artisanal", "Artisanal"],
    }
)
SELECTED = SPLIT[SPLIT.tile_id < 2]
TILE_IDS = SELECTED.tile_id.tolist()


def write_local_band(path: Path, band: np.ndarray, pixel_m: float) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=band.shape[0],
        width=band.shape[1],
        count=1,
        dtype="uint16",
        crs=f"EPSG:{EPSG}",
        transform=rasterio.transform.from_origin(*SCENE_ORIGIN, pixel_m, pixel_m),
        compress="zstd",
    ) as target:
        target.write(band.astype(np.uint16), 1)


def write_local_scene(directory: Path) -> tuple[dict, np.ndarray]:
    """Write a Sentinel-2-like scene; 10 m values encode their position, so a shift shows.

    The window of the tile below starts at pixel 50. Its first cell is all cloud in SCL and its last
    cell is nodata in every band.
    """
    directory.mkdir(parents=True)
    rows, cols = np.mgrid[0:SCENE_PX, 0:SCENE_PX]
    position = 1 + cols % 251 + 256 * (rows % 251)
    position[1586:2098, 1586:2098] = 0
    _, coarse_cols = np.mgrid[0 : SCENE_PX // 2, 0 : SCENE_PX // 2]
    gradient = 1000 + 20 * coarse_cols
    gradient[791:1051, 791:1051] = 0  # two pixels wider, so bilinear stays zero across the cell
    scl = np.full(coarse_cols.shape, 4)
    scl[25:281, 25:281] = 9
    assets = {}
    for band in (*garimpo_uncertainty.data.BANDS, "SCL"):
        path = directory / f"{band}.tif"
        if band in download.ASSETS_10M:
            write_local_band(path, position, 10.0)
        else:
            write_local_band(path, scl if band == "SCL" else gradient, 20.0)
        assets[band] = {"href": str(path)}
    item = {
        "properties": {"eo:cloud_cover": 1.0, "s2:processing_baseline": "04.00"},
        "assets": assets,
    }
    return item, position


def to_wgs84(geometry: shapely.Geometry) -> shapely.Geometry:
    return gpd.GeoSeries([geometry], crs=EPSG).to_crs(4326).iloc[0]


def write_local_gpkg(
    root: Path, items: list[str], geometries: list[shapely.Geometry], masks: list[shapely.Geometry]
) -> None:
    download.gpkg_path(root).parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {
            "tile_id": range(len(items)),
            "s2_tile_id": items,
            "geometry": [to_wgs84(g) for g in geometries],
        },
        crs=4326,
    ).to_file(download.gpkg_path(root), layer="tiles", driver="GPKG")
    gpd.GeoDataFrame(
        {"tile_id": range(len(masks)), "geometry": [to_wgs84(m) for m in masks]}, crs=4326
    ).to_file(download.gpkg_path(root), layer="preferred_polygons", driver="GPKG")


def fake_get_json(item: dict) -> Callable[[str], dict]:
    def get_json(url: str) -> dict:
        if url == download.TOKEN_URL:
            return {"token": "sig=1"}
        if url == download.STAC + "S2_LOCAL":
            return item
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    return get_json


def serve_local_scene(patch: pytest.MonkeyPatch, item: dict) -> None:
    patch.setattr(download, "_get_json", fake_get_json(item))
    patch.setattr(download, "_signed", lambda href, _token: href)


@pytest.fixture(scope="module")
def scene() -> tuple[dict, np.ndarray]:
    return write_local_scene(DOWNLOAD_ARTIFACTS / "scene")


@pytest.fixture(scope="module")
def downloaded(scene: tuple[dict, np.ndarray]) -> Path:
    """Two tiles downloaded with bilinear resampling: one present, one absent from the catalog."""
    off_scene = shapely.affinity.translate(TILE_BOX, xoff=100_000)
    items = ["S2_LOCAL", "S2_ABSENT", "S2_ABSENT", "S2_LOCAL"]
    write_local_gpkg(DATA, items, [TILE_BOX, TILE_BOX, TILE_BOX, off_scene], [MASK] * 4)
    with pytest.MonkeyPatch.context() as patch:
        serve_local_scene(patch, scene[0])
        download.download_scenes(DATA, SPLIT, SELECTED, 2, BILINEAR)
    return DATA


@pytest.fixture
def scratch_download(
    downloaded: Path, scene: tuple[dict, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    serve_local_scene(monkeypatch, scene[0])
    with support.restored_after(downloaded):
        yield downloaded


def test_tile_origin_snaps_the_box_to_the_grid() -> None:
    grid = rasterio.transform.from_origin(*SCENE_ORIGIN, 10, 10)
    assert download.tile_origin(grid, TILE_BOX.bounds, TILE_PX) == (50, 50, 15.5, 31.0)


def test_boa_offset_follows_the_processing_baseline() -> None:
    assert download._boa_offset({"properties": {"s2:processing_baseline": "04.00"}}) == -1000
    assert download._boa_offset({"properties": {"s2:processing_baseline": "03.01"}}) == 0


def test_signing_appends_the_token(
    scene: tuple[dict, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(download, "_get_json", fake_get_json(scene[0]))
    assert download._signed("a.tif", download._Token()) == "a.tif?sig=1"


def test_10m_bands_are_read_without_resampling(scene: tuple[dict, np.ndarray]) -> None:
    item, position = scene
    first_cell = (600_500.0, 9_094_380.0, 605_620.0, 9_099_500.0)
    exact = download._read_band(item["assets"]["B02"]["href"], first_cell, CELL_PX, NEAREST)
    assert exact.dtype == np.uint16
    assert np.array_equal(exact, position[50:562, 50:562])


def test_20m_bands_land_on_the_10m_grid(scene: tuple[dict, np.ndarray]) -> None:
    item, _ = scene
    first_cell = (600_500.0, 9_094_380.0, 605_620.0, 9_099_500.0)
    b11 = item["assets"]["B11"]["href"]
    stairs = download._read_band(b11, first_cell, CELL_PX, NEAREST)
    ramp = download._read_band(b11, first_cell, CELL_PX, BILINEAR)
    expected = 1000.0 + 10.0 * np.arange(50, 562)
    for band in (stairs, ramp):
        assert band.shape == (CELL_PX, CELL_PX)
        assert np.abs(band[8:-8, 8:-8] - expected[8:-8]).max() <= 20
    assert (stairs != ramp).any()


def test_sidecar_records_the_window_and_the_offsets(downloaded: Path) -> None:
    sidecar = json.loads((downloaded / "tiles" / "0.json").read_text())
    assert sidecar["status"] == "ok"
    assert sidecar["resampling_20m"] == "bilinear"
    assert sidecar["boa_offset"] == -1000
    assert sidecar["tile_offset_m"] > garimpo_uncertainty.data.PIXEL_M
    assert np.isclose(sidecar["tile_offset_m"], 15.5, atol=0.01)
    assert np.isclose(sidecar["tile_box_excess_m"], 31.0, atol=0.01)
    # The tile table georeferences a slot: the window starts 50 pixels into the scene.
    assert sidecar["transform"][2] == SCENE_ORIGIN[0] + 500
    assert sidecar["transform"][5] == SCENE_ORIGIN[1] - 500
    assert json.loads((downloaded / "tiles" / "1.json").read_text())["status"] == "missing_item"


def test_store_holds_the_scene_at_the_window(
    downloaded: Path, scene: tuple[dict, np.ndarray]
) -> None:
    store = garimpo_uncertainty.data.open_store(downloaded)
    assert np.array_equal(store.images[0, 0], scene[1][50:562, 50:562])
    assert (store.scl[0] == 9).all()


def test_real_offset_and_missing_industrial_tile_pass_validation(downloaded: Path) -> None:
    store = garimpo_uncertainty.data.open_store(downloaded)
    written = garimpo_uncertainty.data.cell_table(downloaded, SPLIT, TILE_IDS)
    assert written.cell_index.tolist() == list(range(16))
    assert np.isclose(store.stats["mask_ha"][:16].sum(), MASK.area / 1e4, rtol=0.01)
    report, _ = garimpo_uncertainty.validation.validate(
        downloaded, SPLIT, TILE_IDS, CELL_PX, {"tiles_by_scale_role": SPLIT_COUNTS}, 2
    )
    assert report["problems"] == []
    assert report["industrial_tiles_excluded"] == 1
    assert report["diagnostics"]["by_role"]["test"]["cells_100_percent_cloud"] == 1
    assert report["diagnostics"]["by_role"]["test"]["cells_100_percent_nodata"] == 1


@pytest.mark.parametrize("wrong", [{"Artisanal/test": 2}, {"Industrial/train": 2}])
def test_wrong_counts_fail_the_split_check(downloaded: Path, wrong: dict[str, int]) -> None:
    tiles = garimpo_uncertainty.data.tile_records(downloaded, SPLIT, TILE_IDS)
    problems = garimpo_uncertainty.validation.check_split_counts(
        tiles, {"tiles_by_scale_role": wrong}
    )
    assert problems
    assert problems[0].startswith("split_counts")


def test_outage_is_not_a_missing_tile(scratch_download: Path) -> None:
    """It must stop the run instead of shrinking the training set."""
    industrial = scratch_download / "tiles" / "1.json"
    industrial.write_text(json.dumps(json.loads(industrial.read_text()) | {"status": "http_503"}))
    report, _ = garimpo_uncertainty.validation.validate(
        scratch_download, SPLIT, TILE_IDS, CELL_PX, {"tiles_by_scale_role": SPLIT_COUNTS}, 2
    )
    assert "industrial_download_errors" in " ".join(report["problems"])
    assert "split_counts" in " ".join(report["problems"])
    assert report["industrial_tiles_excluded"] == 0


def test_unchanged_rerun_touches_nothing(scratch_download: Path) -> None:
    """Neither the sidecar nor the identity later stages check."""
    sidecar_path = scratch_download / "tiles" / "0.json"
    identity = garimpo_uncertainty.data.tiles_identity(scratch_download, SPLIT, TILE_IDS)
    sidecar_time = sidecar_path.stat().st_mtime_ns
    download.download_scenes(scratch_download, SPLIT, SELECTED, 2, BILINEAR)
    assert sidecar_path.stat().st_mtime_ns == sidecar_time
    assert garimpo_uncertainty.data.tiles_identity(scratch_download, SPLIT, TILE_IDS) == identity


def test_another_resampling_refuses(scratch_download: Path) -> None:
    sidecar_path = scratch_download / "tiles" / "0.json"
    sidecar_time = sidecar_path.stat().st_mtime_ns
    with pytest.raises(ValueError, match="resampling"):
        download.download_scenes(scratch_download, SPLIT, SELECTED, 2, NEAREST)
    assert sidecar_path.stat().st_mtime_ns == sidecar_time


def test_crash_during_a_recut_leaves_no_sidecar_vouching_for_it(scratch_download: Path) -> None:
    """A crash after a re-cut's slots and before its sidecar; the rerun finishes that tile."""
    root = scratch_download
    sidecar_path = root / "tiles" / "0.json"
    store = garimpo_uncertainty.data.open_store(root)
    identity = garimpo_uncertainty.data.tiles_identity(root, SPLIT, TILE_IDS)
    b02_before, b11_before = np.array(store.images[0, 0]), np.array(store.images[0, 4])

    class Crash(BaseException):
        pass

    write_cells = download.write_cells

    def write_then_crash(*args: object) -> None:
        write_cells(*args)
        raise Crash

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(download, "write_cells", write_then_crash)
        with pytest.raises(Crash):
            download.download_scenes(root, SPLIT, SELECTED, 2, NEAREST, force=True)
    assert (store.images[0, 4] != b11_before).any()
    assert not sidecar_path.exists()
    assert garimpo_uncertainty.data.cell_table(root, SPLIT, TILE_IDS).empty

    # The rerun then finishes that tile, as any pending tile, and the identity moves.
    download.download_scenes(root, SPLIT, SELECTED, 2, NEAREST)
    assert json.loads(sidecar_path.read_text())["resampling_20m"] == "nearest"
    assert garimpo_uncertainty.data.tiles_identity(root, SPLIT, TILE_IDS) != identity
    assert np.array_equal(store.images[0, 0], b02_before)
    assert (store.images[0, 4] != b11_before).any()
    recut = sidecar_path.stat().st_mtime_ns
    download.download_scenes(root, SPLIT, SELECTED, 2, NEAREST, force=True)
    assert sidecar_path.stat().st_mtime_ns != recut
    # The hash must follow the re-cut slots, not stay from the first cut.
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["slots_sha256"] == garimpo_uncertainty.data.slots_sha256(store, slice(0, 16))


def test_missing_artisanal_tile_and_off_scene_window_fail_validation(
    scratch_download: Path,
) -> None:
    root = scratch_download
    download.download_scenes(root, SPLIT, SPLIT, 2, BILINEAR)
    assert json.loads((root / "tiles" / "2.json").read_text())["status"] == "missing_item"
    assert json.loads((root / "tiles" / "3.json").read_text())["status"] == "error_ValueError"
    report, _ = garimpo_uncertainty.validation.validate(
        root, SPLIT, SPLIT.tile_id.tolist(), CELL_PX, None, 2
    )
    problems = " ".join(report["problems"])
    assert "missing_scenes: 2 cases" in problems, problems
    assert "partial_group" in problems, problems


def test_sidecar_binds_its_tile_to_its_slot_and_content(downloaded: Path) -> None:
    store = garimpo_uncertainty.data.open_store(downloaded)
    sidecar = json.loads((downloaded / "tiles" / "0.json").read_text())
    assert sidecar["tile_index"] == 0
    assert sidecar["cells_per_tile"] == 16
    assert sidecar["slots_sha256"] == garimpo_uncertainty.data.slots_sha256(store, slice(0, 16))


@pytest.mark.parametrize("force", [False, True])
def test_lost_array_does_not_come_back_as_zeros(scratch_download: Path, force: bool) -> None:
    """Sidecars still say `ok`, even with --force."""
    lost = scratch_download / "masks.npy"
    lost.rename(lost.with_suffix(".away"))
    with pytest.raises(ValueError, match=r"masks\.npy.*status ok"):
        download.download_scenes(scratch_download, SPLIT, SELECTED, 2, BILINEAR, force=force)
    assert not lost.exists()


def test_array_comes_back_when_no_sidecar_vouches_for_it(scene: tuple[dict, np.ndarray]) -> None:
    """As after a crash during first creation."""
    fresh = DOWNLOAD_ARTIFACTS / "fresh"
    download.create_store(fresh, 1, TILE_PX, CELL_PX)
    (fresh / "masks.npy").unlink()
    download.create_store(fresh, 1, TILE_PX, CELL_PX)
    assert (fresh / "masks.npy").exists()


def test_moved_split_does_not_download_into_another_tile_s_slots(scratch_download: Path) -> None:
    moved = SPLIT.iloc[::-1]
    with pytest.raises(ValueError, match="tile_index"):
        download.download_scenes(scratch_download, moved, moved, 2, BILINEAR)


def test_added_tile_does_not_download_into_another_tile_s_slots(scratch_download: Path) -> None:
    added = pd.concat([SPLIT, SPLIT.tail(1).assign(tile_id=99)])
    with pytest.raises(ValueError, match="holds 64 cells"):
        download.download_scenes(scratch_download, added, added, 2, BILINEAR)


LIST_SPLIT = pd.DataFrame(
    {
        "tile_id": range(5),
        "region": ["Amazonia BR", "Asia/Oceania", "Asia/Oceania", "Asia/Oceania", "Africa"],
        "cluster_50km": [0.0, 1.0, 2.0, 3.0, 4.0],
        "role": ["test", "train", "train", "train", "dropped"],
        "scale": ["Artisanal"] * 4 + ["Industrial"],
    }
)
LIST_SPLIT_PATH = LISTS / "split.csv"
LIST_ROOT = LISTS / "data"


def listed_ids(name: str) -> set[int]:
    tile_list = json.loads((LISTS / name).read_text())
    return {tile["tile_id"] for tile in tile_list["tiles"]}


def run_scenes_command(patch: pytest.MonkeyPatch, root: Path, tile_list: Path) -> None:
    """Through the real command, with a fake Planetary Computer still in place."""
    patch.setenv("GARIMPO_DATA_DIR", str(root))
    arguments = ["--split", str(LIST_SPLIT_PATH), "--tiles", str(tile_list)]
    arguments += ["--resampling-20m", "bilinear"]
    patch.setattr(sys, "argv", ["download_data.py", "scenes", *arguments])
    runpy.run_path(str(support.ROOT / "scripts" / "download_data.py"), run_name="__main__")


def sidecars() -> dict[int, tuple[str, int]]:
    paths = (LIST_ROOT / "tiles").glob("*.json")
    return {int(p.stem): (p.read_text(), p.stat().st_mtime_ns) for p in paths}


def slot_hashes(tile_id: int) -> list[str]:
    store = garimpo_uncertainty.data.open_store(LIST_ROOT)
    slots = slice(tile_id * 16, (tile_id + 1) * 16)  # tile_index == tile_id in this split
    return [hashlib.sha256(array[slots]).hexdigest() for array in store]


def written_tiles(tile_ids: set[int]) -> set[int]:
    store = garimpo_uncertainty.data.open_store(LIST_ROOT)
    return {t for t in tile_ids if store.images[t * 16 : (t + 1) * 16].any()}


@dataclass(frozen=True)
class Incremental:
    """What the small list left behind, measured before the larger list ran over it."""

    small: set[int]
    large: set[int]
    sidecars: dict[int, tuple[str, int]]
    slots: dict[int, list[str]]
    written: set[int]


@pytest.fixture(scope="module")
def incremental(scene: tuple[dict, np.ndarray]) -> Incremental:
    LISTS.mkdir()
    LIST_SPLIT.to_csv(LIST_SPLIT_PATH, index=False)
    arguments = [
        "lists",
        "--split",
        str(LIST_SPLIT_PATH),
        "--subset-seed",
        "0",
        "--out",
        str(LISTS),
    ]
    support.run("download_data.py", *arguments, data_root=LISTS)
    small, large = listed_ids("tiles_1pct.json"), listed_ids("tiles_100pct.json")

    # A different mask per tile, so slots written under another tile's index cannot pass.
    masks = [shapely.affinity.translate(MASK, yoff=1000 * i) for i in range(4)]
    write_local_gpkg(LIST_ROOT, ["S2_LOCAL"] * 4, [TILE_BOX] * 4, masks)
    with pytest.MonkeyPatch.context() as patch:
        serve_local_scene(patch, scene[0])
        run_scenes_command(patch, LIST_ROOT, LISTS / "tiles_1pct.json")
        state = Incremental(
            small,
            large,
            sidecars(),
            {tile_id: slot_hashes(tile_id) for tile_id in small},
            written_tiles(large),
        )
        run_scenes_command(patch, LIST_ROOT, LISTS / "tiles_100pct.json")
    return state


@pytest.fixture
def scratch_lists(
    incremental: Incremental, scene: tuple[dict, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> Iterator[Incremental]:
    serve_local_scene(monkeypatch, scene[0])
    with support.restored_after(LISTS):
        yield incremental


def test_small_list_nests_in_the_large_one(incremental: Incremental) -> None:
    assert incremental.small < incremental.large
    assert incremental.large == {0, 1, 2, 3}


def test_small_list_writes_only_its_tiles(incremental: Incremental) -> None:
    assert set(incremental.sidecars) == incremental.small
    assert incremental.written == incremental.small


def test_larger_list_adds_tiles_and_touches_no_earlier_one(incremental: Incremental) -> None:
    small, large = incremental.small, incremental.large
    after = sidecars()
    assert set(after) == large
    assert {tile_id: after[tile_id] for tile_id in small} == incremental.sidecars
    assert {tile_id: slot_hashes(tile_id) for tile_id in small} == incremental.slots
    assert written_tiles(large) == large
    assert len({tuple(slot_hashes(tile_id)) for tile_id in large}) == len(large)


def test_list_from_another_split_stops_before_creating_a_store(
    scratch_lists: Incremental, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It names other slots."""
    before = sidecars()
    stale = LISTS / "stale"
    stale_split = LIST_SPLIT.assign(role=LIST_SPLIT.role.replace({"test": "validation"}))
    stale_split.to_csv(LIST_SPLIT_PATH, index=False)
    with pytest.raises(ValueError, match="split_sha256"):
        run_scenes_command(monkeypatch, stale, LISTS / "tiles_1pct.json")
    assert not (stale / "images.npy").exists()
    assert json.loads((stale / "status" / "download_scenes.json").read_text())["status"] == "failed"
    assert sidecars() == before


def test_list_naming_tiles_outside_the_split_stops(
    scratch_lists: Incremental, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the current split's hash: tile 4 is dropped and tile 99 was never in it."""
    before = sidecars()
    stale = LISTS / "stale"
    tile_list = json.loads((LISTS / "tiles_1pct.json").read_text())
    tile_list["tiles"] += [{"tile_id": 4, "role": "test"}, {"tile_id": 99, "role": "test"}]
    edited = LISTS / "edited.json"
    edited.write_text(json.dumps(tile_list))
    with pytest.raises(ValueError, match=r"2 tiles .*\[4, 99\]"):
        run_scenes_command(monkeypatch, stale, edited)
    assert not (stale / "images.npy").exists()
    assert sidecars() == before
