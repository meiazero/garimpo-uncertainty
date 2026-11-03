"""Draws dataset examples for the proposal; writes proposta/exemplos_dataset.pdf and outputs/."""

import csv
import json
import pathlib
import re
import time
import urllib.error
import urllib.request

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.transform
import rasterio.windows
import shapely

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/"
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
CHIPS_PER_SIDE = 4  # same grid as frame.py
CHIP_PX = 512
MAX_SCENE_CLOUD = 1.0
PANELS = (  # (scale, region or None, label)
    ("Artisanal", "Amazonia BR", "Amazônia brasileira\nartesanal"),
    ("Artisanal", "Africa/Europe", "África\nartesanal"),
    ("Artisanal", "Asia/Oceania", "Ásia/Oceania\nartesanal"),
    ("Industrial", None, "Industrial\n(treinamento)"),
)

plt.rcParams.update(
    {"font.family": "serif", "font.serif": ["STIXGeneral"], "font.size": 9, "pdf.fonttype": 42}
)


def get_json(url: str) -> dict:
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            time.sleep(int(error.headers.get("Retry-After") or 2**attempt))
    raise TimeoutError(f"still rate limited: {url}")


def utm_crs(item_id: str) -> int:
    zone, band = re.search(r"_T(\d{2})([C-X])", item_id).groups()
    return (32600 if band >= "N" else 32700) + int(zone)


def chips(tile: shapely.Polygon, mask: shapely.Geometry, crs: int) -> list[dict]:
    series = gpd.GeoSeries([tile, mask], crs=4326).to_crs(crs)
    x0, y0, x1, y1 = series.iloc[0].bounds
    mask_utm = shapely.make_valid(series.iloc[1])
    xs, ys = np.linspace(x0, x1, CHIPS_PER_SIDE + 1), np.linspace(y0, y1, CHIPS_PER_SIDE + 1)
    return [
        {
            "bounds": (xs[i], ys[j], xs[i + 1], ys[j + 1]),
            "area_ha": mask_utm.intersection(shapely.box(xs[i], ys[j], xs[i + 1], ys[j + 1])).area
            / 1e4,
            "mask": mask_utm,
        }
        for i in range(CHIPS_PER_SIDE)
        for j in range(CHIPS_PER_SIDE)
    ]


def rgb(item: dict, crs: int, bounds: tuple[float, ...], token: str) -> np.ndarray:
    bands = []
    for band in ("B04", "B03", "B02"):
        with rasterio.open(f"{item['assets'][band]['href']}?{token}") as source:
            assert source.crs.to_epsg() == crs
            window = rasterio.windows.from_bounds(*bounds, transform=source.transform)
            bands.append(source.read(1, window=window, out_shape=(CHIP_PX, CHIP_PX)).astype(float))
    image = np.dstack(bands)
    low, high = np.percentile(image, (2, 98))  # display stretch only
    return np.clip((image - low) / (high - low), 0, 1)


def pick(scale: str, region: str | None, split: pd.DataFrame, tiles, masks, target: float) -> dict:
    """Return the valid mining chip nearest the Amazon median area, in a nearly cloud-free scene."""
    rows = split[split.scale == scale]
    if region:
        rows = rows[rows.region == region]
    candidates = []
    for tile_id in rows.tile_id:
        tile = tiles.loc[tile_id]
        crs = utm_crs(tile.s2_tile_id)
        candidates += [
            chip | {"tile_id": tile_id, "item_id": tile.s2_tile_id, "crs": crs}
            for chip in chips(tile.geometry, masks.loc[tile_id], crs)
            if chip["area_ha"] > 0
        ]
    for chip in sorted(candidates, key=lambda c: (abs(c["area_ha"] - target), c["tile_id"])):
        try:
            item = get_json(STAC + chip["item_id"])
        except urllib.error.HTTPError as error:
            if error.code == 404:
                continue
            raise
        cloud = item["properties"].get("eo:cloud_cover", 100)
        if cloud <= MAX_SCENE_CLOUD:
            return chip | {"item": item, "cloud": cloud}
    raise LookupError(f"no chip for {scale} {region}")


def main() -> None:
    gpkg = "data/zenodo/mining_area_data.gpkg"
    tiles = gpd.read_file(gpkg, layer="tiles").set_index("tile_id")
    masks = gpd.read_file(gpkg, layer="preferred_polygons").set_index("tile_id").geometry
    split = pd.read_csv("outputs/split.csv")
    frame = json.loads(pathlib.Path("outputs/frame.json").read_text())
    target = frame["chip_ha_amazon_nonempty"]["q50"]

    token = get_json(TOKEN)["token"]
    fig, axes = plt.subplots(2, len(PANELS), figsize=(6.3, 3.5), layout="constrained")
    records = []
    for column, (scale, region, label) in enumerate(PANELS):
        chip = pick(scale, region, split, tiles, masks, target)
        image = rgb(chip["item"], chip["crs"], chip["bounds"], token)
        transform = rasterio.transform.from_bounds(*chip["bounds"], CHIP_PX, CHIP_PX)
        mask = rasterio.features.rasterize(
            [chip["mask"]], out_shape=(CHIP_PX, CHIP_PX), transform=transform
        )
        year = chip["item_id"].split("_")[2][:4]
        area = f"{chip['area_ha']:.1f}".replace(".", ",")
        axes[0, column].imshow(image)
        axes[0, column].set_title(label, fontsize=9)
        axes[1, column].imshow(image)
        axes[1, column].contour(mask, levels=[0.5], colors="yellow", linewidths=0.8)
        axes[1, column].set_xlabel(f"{area} ha, {year}")
        records.append(
            {
                "panel": label.replace("\n", " "),
                "tile_id": chip["tile_id"],
                "item": chip["item_id"],
                "scene_cloud_pct": chip["cloud"],
                "chip_bounds_utm": [round(v, 1) for v in chip["bounds"]],
                "epsg": chip["crs"],
                "mining_area_ha": round(chip["area_ha"], 2),
            }
        )
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig("proposta/exemplos_dataset.pdf")
    fig.savefig("outputs/exemplos_dataset.png", dpi=200)
    with open("outputs/exemplos_dataset.csv", "w", newline="") as target_file:
        writer = csv.DictWriter(target_file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(*records, sep="\n")


if __name__ == "__main__":
    main()
