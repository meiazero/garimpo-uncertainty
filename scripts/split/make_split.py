"""Fixes the geographic partition; writes outputs/split.csv and split_summary.json."""

import json
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd

SEED = 0
GUARD_M = 50_000
HELD_OUT_REGIONS = {"Amazonia BR", "Americas, other"}


def assign_groups(artisanal: pd.DataFrame) -> dict[int, str]:
    """Whole 50 km groups go to one role, so no neighborhood crosses a partition."""
    rng = np.random.default_rng(SEED)
    roles = {}
    for region, tiles in artisanal.groupby("region"):
        groups = np.sort(tiles.cluster_50km.unique())
        if region in HELD_OUT_REGIONS:
            roles |= {int(group): "test" for group in groups}
            continue
        rng.shuffle(groups)
        n_cal = round(0.5 * len(groups))
        n_val = max(1, round(0.1 * len(groups)))
        roles |= {int(group): "calibration" for group in groups[:n_cal]}
        roles |= {int(group): "validation" for group in groups[n_cal : n_cal + n_val]}
        roles |= {int(group): "train" for group in groups[n_cal + n_val :]}
    return roles


def main() -> None:
    artisanal = pd.read_csv("outputs/tiles_artisanal.csv")
    artisanal["role"] = artisanal.cluster_50km.map(assign_groups(artisanal))

    tiles = gpd.read_file("data/zenodo/mining_area_data.gpkg", layer="tiles")
    tiles = tiles[tiles.split.notna()].to_crs(6933)
    roles = artisanal.set_index("tile_id").role
    held_out = tiles[tiles.tile_id.map(roles).isin(["test", "calibration", "validation"])]
    industrial = tiles[tiles.minetype2 == "Industrial"].copy()
    distance = industrial.geometry.distance(held_out.geometry.union_all())
    industrial["role"] = np.where(distance < GUARD_M, "dropped", "train")

    split = pd.concat(
        [
            artisanal[["tile_id", "region", "cluster_50km", "role"]].assign(scale="Artisanal"),
            pd.DataFrame(industrial[["tile_id", "role"]]).assign(scale="Industrial"),
        ]
    )
    assert split.tile_id.is_unique
    assert (artisanal.groupby("cluster_50km").role.nunique() == 1).all()
    train_groups = artisanal[artisanal.role == "train"].cluster_50km
    test_groups = artisanal[artisanal.role == "test"].cluster_50km
    assert not set(train_groups) & set(test_groups)

    summary = {
        "seed": SEED,
        "tiles_by_scale_role": {
            f"{scale}/{role}": int(n)
            for (scale, role), n in split.groupby(["scale", "role"]).size().items()
        },
        "groups_by_role": artisanal.groupby("role").cluster_50km.nunique().to_dict(),
        "train_chips": int((split.role == "train").sum() * 16),
    }
    outputs = pathlib.Path("outputs")
    split.to_csv(outputs / "split.csv", index=False)
    text = json.dumps(summary, indent=2)
    (outputs / "split_summary.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
