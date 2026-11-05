"""Measures the dataset's available units; writes outputs/frame.json and tiles_artisanal.csv."""

import argparse
import json
import pathlib
import re

import geopandas
import numpy as np
import pandas as pd
import pyproj
import shapely

CHIPS_PER_SIDE = 4  # 2048 px tile into 512 px chips, the chip size of the dataset's own code
SEPARATIONS_KM = (25, 50, 100)
AMERICAS_MAX_LON = -30
AFRICA_EUROPE_MAX_LON = 60
TILE_SIDE_MIN_M = 20_000
TILE_SIDE_MAX_M = 21_000


def cluster_labels(tiles: geopandas.GeoDataFrame, separation_km: float) -> np.ndarray:
    """Tiles closer than the separation share a label, so a split never divides a neighborhood."""
    merged = tiles.geometry.buffer(separation_km * 500).union_all()
    parts = list(getattr(merged, "geoms", [merged]))
    tree = shapely.STRtree(parts)
    return np.array([tree.query(p, predicate="intersects")[0] for p in tiles.geometry])


def chip_areas_ha(
    tile: shapely.Polygon, mask: shapely.Geometry, crs: pyproj.CRS
) -> tuple[list[float], float]:
    """Mining area per chip in the tile's UTM zone, where metric areas are valid."""
    series = geopandas.GeoSeries([tile, mask], crs=4326).to_crs(crs)
    x0, y0, x1, y1 = series.iloc[0].bounds
    mask_utm = shapely.make_valid(series.iloc[1])
    xs, ys = np.linspace(x0, x1, CHIPS_PER_SIDE + 1), np.linspace(y0, y1, CHIPS_PER_SIDE + 1)
    areas = [
        mask_utm.intersection(shapely.box(xs[i], ys[j], xs[i + 1], ys[j + 1])).area / 1e4
        for i in range(CHIPS_PER_SIDE)
        for j in range(CHIPS_PER_SIDE)
    ]
    return areas, x1 - x0


def quantiles(values: pd.Series) -> dict[str, float]:
    points = values.quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).round(1)
    return {f"q{int(q * 100)}": float(v) for q, v in points.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpkg", type=pathlib.Path, default="data/zenodo/mining_area_data.gpkg")
    parser.add_argument("--biomes", type=pathlib.Path, default="data/biomes/lm_bioma_250.shp")
    return parser.parse_args()


def add_location_columns(tiles: geopandas.GeoDataFrame, biomes: geopandas.GeoDataFrame) -> None:
    centers = tiles.geometry.representative_point()
    tiles["lon"], tiles["lat"] = centers.x, centers.y
    tiles["biome"] = geopandas.sjoin(
        geopandas.GeoDataFrame(geometry=centers, crs=4326),
        biomes[["Bioma", "geometry"]],
        predicate="within",
        how="left",
    )["Bioma"]
    tiles["image_year"] = [
        int(re.search(r"_(20\d{2})\d{4}T", s).group(1)) for s in tiles.s2_tile_id
    ]


def artisanal_with_regions(tiles: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    artisanal = tiles[tiles.minetype2 == "Artisanal"].copy()
    artisanal["region"] = np.select(
        [
            artisanal.biome == "Amazônia",
            artisanal.lon < AMERICAS_MAX_LON,
            artisanal.lon < AFRICA_EUROPE_MAX_LON,
        ],
        ["Amazonia BR", "Americas, other", "Africa/Europe"],
        "Asia/Oceania",
    )
    return artisanal


def add_cluster_columns(artisanal: geopandas.GeoDataFrame) -> dict[int, dict[str, int]]:
    """Add one label column per separation; return the clusters each region has under each."""
    # One projection for all continents; tropical distances distort about 15%, tolerable for a
    # grouping threshold.
    metric = artisanal.to_crs(6933)
    clusters = {}
    for km in SEPARATIONS_KM:
        artisanal[f"cluster_{km}km"] = cluster_labels(metric, km)
        clusters[km] = artisanal.groupby("region")[f"cluster_{km}km"].nunique().to_dict()
    return clusters


def measure_chips(artisanal: geopandas.GeoDataFrame, masks: geopandas.GeoSeries) -> pd.DataFrame:
    rows = []
    for tile in artisanal.itertuples():
        crs = geopandas.GeoSeries([tile.geometry], crs=4326).estimate_utm_crs()
        areas, side_m = chip_areas_ha(tile.geometry, masks.loc[tile.tile_id], crs)
        rows += [(tile.tile_id, tile.region, area, side_m) for area in areas]
    return pd.DataFrame(rows, columns=["tile_id", "region", "ha", "side_m"])


def frame_report(
    tiles: geopandas.GeoDataFrame,
    artisanal: geopandas.GeoDataFrame,
    clusters: dict[int, dict[str, int]],
    chips: pd.DataFrame,
) -> dict:
    amazon_chips = chips[chips.region == "Amazonia BR"]
    amazon_artisanal = artisanal[artisanal.region == "Amazonia BR"]
    return {
        "tiles_with_split": len(tiles),
        "scale": tiles.minetype2.value_counts().to_dict(),
        "artisanal_by_region": artisanal.region.value_counts().to_dict(),
        "clusters_by_separation_km": clusters,
        "amazon_artisanal_mine_type": amazon_artisanal.minetype1.value_counts().to_dict(),
        "amazon_artisanal_image_year": amazon_artisanal.image_year.value_counts()
        .sort_index()
        .to_dict(),
        "tile_side_m_median": float(chips.side_m.median()),
        "chips_total_all_tiles": len(tiles) * CHIPS_PER_SIDE**2,
        "chips_artisanal": len(chips),
        "chips_amazon": len(amazon_chips),
        "chips_amazon_nonempty": int((amazon_chips.ha > 0).sum()),
        "chip_ha_amazon_nonempty": quantiles(amazon_chips.ha[amazon_chips.ha > 0]),
        "chip_ha_artisanal_nonempty": quantiles(chips.ha[chips.ha > 0]),
    }


def main() -> None:
    args = parse_args()
    tiles = geopandas.read_file(args.gpkg, layer="tiles")
    tiles = tiles[tiles.split.notna()].copy()
    masks = geopandas.read_file(args.gpkg, layer="preferred_polygons").set_index("tile_id").geometry
    biomes = geopandas.read_file(args.biomes).to_crs(4326)

    add_location_columns(tiles, biomes)
    artisanal = artisanal_with_regions(tiles)
    clusters = add_cluster_columns(artisanal)
    chips = measure_chips(artisanal, masks)
    report = frame_report(tiles, artisanal, clusters, chips)
    if not TILE_SIDE_MIN_M < report["tile_side_m_median"] < TILE_SIDE_MAX_M:
        raise ValueError(
            f"tile is not 2048 px at 10 m: median side is {report['tile_side_m_median']} m"
        )

    outputs = pathlib.Path("outputs")
    outputs.mkdir(exist_ok=True)
    (outputs / "frame.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    artisanal.drop(columns="geometry").to_csv(outputs / "tiles_artisanal.csv", index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
