"""Measures how far the dataset's own masks agree; writes outputs/label_agreement.json."""

import json
import pathlib
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

GPKG = "data/zenodo/mining_area_data.gpkg"
CHIPS_PER_SIDE = 4  # same grid as frame.py


def utm_crs(item_id: str) -> int:
    zone, band = re.search(r"_T(\d{2})([C-X])", item_id).groups()
    return (32600 if band >= "N" else 32700) + int(zone)


def compare(tile: shapely.Polygon, first, second, crs: int) -> dict:
    tile_utm, a, b = gpd.GeoSeries([tile, first, second], crs=4326).to_crs(crs)
    a, b = (shapely.make_valid(g).intersection(tile_utm) for g in (a, b))
    union = a.union(b).area
    x0, y0, x1, y1 = tile_utm.bounds
    xs, ys = np.linspace(x0, x1, CHIPS_PER_SIDE + 1), np.linspace(y0, y1, CHIPS_PER_SIDE + 1)
    boxes = [
        shapely.box(xs[i], ys[j], xs[i + 1], ys[j + 1])
        for i in range(CHIPS_PER_SIDE)
        for j in range(CHIPS_PER_SIDE)
    ]
    chips = [(a.intersection(c).area / 1e4, b.intersection(c).area / 1e4) for c in boxes]
    return {
        "area_a_ha": a.area / 1e4,
        "area_b_ha": b.area / 1e4,
        "iou": a.intersection(b).area / union if union else np.nan,
        "chip_abs_diff_ha": [abs(x - y) for x, y in chips if x > 0 or y > 0],
    }


def summarize(rows: pd.DataFrame) -> dict:
    ratio = rows.area_a_ha / rows.area_b_ha.replace(0, np.nan)
    diffs = pd.Series([d for chip in rows.chip_abs_diff_ha for d in chip])
    points = [0.1, 0.5, 0.9]
    return {
        "tiles": len(rows),
        "iou_q10_q50_q90": rows.iou.quantile(points).round(3).tolist(),
        "area_ratio_a_over_b_q10_q50_q90": ratio.quantile(points).round(3).tolist(),
        "total_area_ratio_a_over_b": round(rows.area_a_ha.sum() / rows.area_b_ha.sum(), 3),
        "nonempty_chips": len(diffs),
        "chip_abs_diff_ha_q50_q90": diffs.quantile([0.5, 0.9]).round(1).tolist(),
    }


def pair(tiles: pd.DataFrame, first: pd.Series, second: pd.Series) -> pd.DataFrame:
    shared = first.dropna().index.intersection(second.dropna().index).intersection(tiles.index)
    rows = [
        {"tile_id": t, "scale": tiles.minetype2[t]}
        | compare(tiles.geometry[t], first[t], second[t], utm_crs(tiles.s2_tile_id[t]))
        for t in shared
    ]
    return pd.DataFrame(rows)


def main() -> None:
    tiles = gpd.read_file(GPKG, layer="tiles").set_index("tile_id")

    def layer(name: str) -> gpd.GeoSeries:
        return gpd.read_file(GPKG, layer=name).set_index("tile_id").geometry

    split = pd.read_csv("outputs/split.csv").set_index("tile_id")

    validated = layer("test_polygons_validated")
    against_validated = pair(tiles, layer("preferred_polygons"), validated)
    against_validated["our_role"] = against_validated.tile_id.map(split.role)
    between_sources = pair(tiles, layer("maus_polygons"), layer("tang_polygons"))

    report = {
        "split_values": tiles.split.value_counts(dropna=False).to_dict(),
        "validated_by_our_role_and_scale": against_validated.groupby(
            ["our_role", "scale"], dropna=False
        )
        .size()
        .rename(lambda x: x)
        .reset_index()
        .astype(str)
        .to_dict("records"),
        "preferred_vs_validated": {
            scale: summarize(rows) for scale, rows in against_validated.groupby("scale")
        },
        "maus_vs_tang": {
            scale: summarize(rows) for scale, rows in between_sources.groupby("scale")
        },
    }
    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    pathlib.Path("outputs/label_agreement.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
