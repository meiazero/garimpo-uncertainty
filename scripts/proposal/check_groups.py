"""Cross-checks the Amazon grouping in a Brazilian projection and summarizes the 50 km groups."""

import json
import pathlib

import geopandas as gpd
import pandas as pd


def main() -> None:
    tiles = gpd.read_file("data/zenodo/mining_area_data.gpkg", layer="tiles")
    artisanal = pd.read_csv("outputs/tiles_artisanal.csv")
    amazon_ids = artisanal[artisanal.region == "Amazonia BR"].tile_id
    amazon = tiles[tiles.tile_id.isin(amazon_ids)].to_crs(5880)

    groups_5880 = {}
    for km in (25, 50, 100):
        merged = amazon.buffer(km * 500).union_all()
        groups_5880[km] = len(getattr(merged, "geoms", [merged]))

    regions_per_group = artisanal.groupby("cluster_50km").region.nunique()
    other_americas = artisanal[artisanal.region == "Americas, other"]
    report = {
        "amazon_groups_epsg5880": groups_5880,
        "groups_50km_total": int(artisanal.cluster_50km.nunique()),
        "groups_50km_by_region": artisanal.groupby("region").cluster_50km.nunique().to_dict(),
        "groups_50km_spanning_regions": int((regions_per_group > 1).sum()),
        "other_americas_lat": [float(other_americas.lat.min()), float(other_americas.lat.max())],
        "other_americas_lon": [float(other_americas.lon.min()), float(other_americas.lon.max())],
    }
    text = json.dumps(report, indent=2)
    pathlib.Path("outputs/groups.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
