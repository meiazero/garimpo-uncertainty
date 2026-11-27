"""`garimpo data_figures`: figures of the data itself, for the text's description of it.

    garimpo data_figures --out outputs/figures

They read the cell store of GARIMPO_DATA_DIR and the split, never a run, so they draw on a CPU
before any model is trained. Cells are chosen from the reference alone.
"""

import argparse
import math
from pathlib import Path

import geopandas
import matplotlib.patheffects
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp

import garimpo_uncertainty.data
import garimpo_uncertainty.figures.orx_figstyle as style
from garimpo_uncertainty.figures import common
from garimpo_uncertainty.figures.maps import below, scale_bar, true_color

# Natural Earth 1:110m land, v5.1.2 (public domain), only as the map's backdrop.
LAND = Path(__file__).with_name("ne_110m_land.geojson")
# A segment narrower than this share of the longest bar has no room for its count.
LABELED_SHARE = 0.04
ROLE_MARKERS = {"train": "o", "validation": "^", "calibration": "s", "test": "D"}

ROLE_NAMES = {
    "train": "treino",
    "validation": "validação",
    "calibration": "calibração",
    "test": "teste",
}
REGION_NAMES = {
    "Asia/Oceania": "Ásia/Oceania",
    "Africa/Europe": "África/Europa",
    "Amazonia BR": "Amazônia (BR)",
    "Americas, other": "Américas, outras",
}
SCALE_NAMES = {"Artisanal": "artesanal", "Industrial": "industrial"}


def region_name(region: str) -> str:
    return REGION_NAMES.get(region, region)


def read_cells(root: Path, split: pd.DataFrame, train_fraction: float, seed: int) -> pd.DataFrame:
    """Return the selection's cells with each cell's reference area, from the store's stats."""
    tiles = garimpo_uncertainty.data.select_tiles(split, train_fraction, seed)
    cells = garimpo_uncertainty.data.cell_table(root, split, tiles.tile_id.tolist())
    stats = garimpo_uncertainty.data.open_store(root).stats
    return cells.assign(v=stats["mask_ha"][cells.cell_index.to_numpy()])


def split_composition(cells: pd.DataFrame, stem: Path) -> None:
    """Show which regions each role draws on: the test regions appear in no other role.

    Cells per role, stacked by region, with industrial training cells apart, as the split assigns
    whole regions to the test.
    """
    table = (
        cells.assign(
            region=np.where(
                cells.scale == "Industrial", "industrial", cells.region.map(region_name)
            )
        )
        .groupby(["role", "region"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=[r for r in ROLE_NAMES if r in set(cells.role)])
    )
    fig, ax = style.figure_grid(1, 1, width=common.TEXT_WIDTH, ratio=0.4)
    y = np.arange(len(table))
    left = np.zeros(len(table))
    colors = [*style.CYCLE, style.MUTED]
    for k, region in enumerate(table.columns):
        color = style.MUTED if region == "industrial" else colors[k % len(style.CYCLE)]
        bars = ax.barh(
            y, table[region], left=left, color=color, edgecolor="black", linewidth=0.4, label=region
        )
        left += table[region].to_numpy()
        room = LABELED_SHARE * table.sum(axis=1).max()
        ax.bar_label(
            bars,
            [common.number(n, 0) if n >= room else "" for n in table[region]],
            label_type="center",
            fontsize=6,
        )
    for row, total in zip(y, table.sum(axis=1), strict=True):
        ax.text(total, row, f" {common.number(total, 0)}", va="center", fontsize=6)
    ax.set_yticks(y, [ROLE_NAMES[r] for r in table.index])
    ax.invert_yaxis()
    ax.set_xlabel("Células")
    fig.legend(loc="outside lower center", ncol=min(len(table.columns), 5))
    common.save(fig, stem)


def area_distribution(cells: pd.DataFrame, stem: Path) -> None:
    """Show how the reference area spreads: many empty cells, and mining areas over decades.

    (a) Share of cells without mining per role, with n. (b) Empirical distribution of the
    reference area of the cells with mining, per role, on a log axis.
    """
    roles = [r for r in ROLE_NAMES if r in set(cells.role)]
    fig, (left, right) = style.figure_grid(1, 2, width=common.TEXT_WIDTH, ratio=0.38)
    empty = [100 * (cells[cells.role == r].v == 0).mean() for r in roles]
    bars = left.bar(range(len(roles)), empty, color=style.MUTED, edgecolor="black", linewidth=0.4)
    left.bar_label(
        bars,
        [
            f"{common.number(share, 0)}%\nn = {common.number((cells.role == r).sum(), 0)}"
            for share, r in zip(empty, roles, strict=True)
        ],
        fontsize=6,
        padding=1,
    )
    left.set_xticks(range(len(roles)), [ROLE_NAMES[r] for r in roles])
    left.set_ylim(0, 115)
    left.set_ylabel("Células sem mineração (%)")
    for role, color in zip(roles, style.CYCLE, strict=False):
        area = np.sort(cells[(cells.role == role) & (cells.v > 0)].v.to_numpy())
        if len(area):
            right.step(
                area,
                np.arange(1, len(area) + 1) / len(area),
                where="post",
                color=color,
                label=f"{ROLE_NAMES[role]} (n = {common.number(len(area), 0)})",
            )
    right.set_xscale("log")
    right.set_xlabel("Área de referência das células com mineração (ha, log)")
    right.set_ylabel("Fração acumulada")
    right.legend(loc="upper left", fontsize=6)
    style.panel_labels([left, right])
    common.save(fig, stem)


def example_cells(cells: pd.DataFrame, root: Path, stem: Path) -> None:
    """Show what a cell and its reference look like in each region of the evaluation.

    Per region, the evaluation cell with mining at the median reference area: its true color
    and the reference mask over it, with the area under each.
    """
    evaluated = cells[(cells.role != "train") & (cells.v > 0)]
    chosen = []
    for region in sorted(evaluated.region.unique()):
        mining = evaluated[evaluated.region == region].sort_values(["v", "cell_index"])
        chosen.append(mining.iloc[(len(mining) - 1) // 2])
    shown = pd.DataFrame(chosen).reset_index(drop=True)
    masks = np.load(root / "masks.npy", mmap_mode="r")[shown.cell_index.to_numpy()] > 0
    rgb = true_color(root, shown)
    fig, axes = style.figure_grid(
        2, len(shown), width=common.TEXT_WIDTH, ratio=2.3 / max(len(shown), 1), squeeze=False
    )
    for column, image, mask, (_, cell) in zip(axes.T, rgb, masks, shown.iterrows(), strict=True):
        column[0].imshow(image, interpolation="nearest")
        scale_bar(column[0], image.shape[1])
        column[1].imshow(image, interpolation="nearest")
        column[1].contour(mask, levels=[0.5], colors=[style.PALETTE["orange"]], linewidths=0.8)
        below(column[1], f"{ROLE_NAMES[cell.role]}, v = {common.number(cell.v)} ha")
        for ax in column:
            ax.set_axis_off()
    style.panel_labels(axes[0], [region_name(r) for r in shown.region])
    common.save(fig, stem)


def tile_centers(root: Path, split: pd.DataFrame) -> pd.DataFrame:
    """Return every downloaded tile of the split with its center's longitude and latitude."""
    tiles = garimpo_uncertainty.data.tile_records(root, split, split.tile_id.tolist())
    tiles = tiles[tiles.status == "ok"].reset_index(drop=True)
    cell_px = np.load(root / "masks.npy", mmap_mode="r").shape[1]
    half = math.isqrt(int(tiles.cells_per_tile.iloc[0])) * cell_px / 2
    x, y = zip(*(rasterio.Affine(*t) * (half, half) for t in tiles["transform"]), strict=True)
    lon, lat = np.empty(len(tiles)), np.empty(len(tiles))
    for epsg, rows in tiles.groupby("epsg").groups.items():
        at = list(rows)
        lon[at], lat[at] = rasterio.warp.transform(
            f"EPSG:{int(epsg)}", "EPSG:4326", [x[i] for i in at], [y[i] for i in at]
        )
    return tiles.assign(lon=lon, lat=lat)


def tile_map(tiles: pd.DataFrame, stem: Path) -> None:
    """Show where the tiles are: the Americas hold the test tiles and no other role's.

    The artisanal tiles by role, marker and color both, over the industrial training tiles in
    grey, each at its tile's center. Natural Earth's land is the backdrop.
    """
    fig, ax = style.figure_grid(1, 1, width=common.TEXT_WIDTH, ratio=0.5)
    geopandas.read_file(LAND).plot(ax=ax, color="#EFEFEF", edgecolor="none")
    industrial = tiles[tiles.scale == "Industrial"]
    ax.scatter(industrial.lon, industrial.lat, s=2, color=style.MUTED, label="industrial (treino)")
    artisanal = tiles[tiles.scale == "Artisanal"]
    for (role, marker), color in zip(ROLE_MARKERS.items(), style.CYCLE, strict=False):
        shown = artisanal[artisanal.role == role]
        if len(shown):
            ax.scatter(
                shown.lon,
                shown.lat,
                s=9,
                marker=marker,
                color=color,
                edgecolor="black",
                linewidth=0.3,
                label=f"{ROLE_NAMES[role]} (n = {common.number(len(shown), 0)})",
            )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    fig.legend(loc="outside lower center", ncol=5)
    common.save(fig, stem)


def chosen_tile(cells: pd.DataFrame) -> pd.DataFrame:
    """Return, by position, the evaluation tile whose cells with mining are nearest half its cells.

    The lowest tile_id wins a tie, so the choice reads the reference alone.
    """
    evaluated = cells[cells.role != "train"]
    by_tile = evaluated.groupby("tile_id").v
    distance = (by_tile.apply(lambda v: (v > 0).sum()) - by_tile.size() / 2).abs()
    tile_id = distance[distance == distance.min()].index.min()
    return evaluated[evaluated.tile_id == tile_id].sort_values("position")


def mosaic(per_cell: np.ndarray) -> np.ndarray:
    """Return a tile's cells as one image; they run row by row over it (docs/data-layout.md)."""
    side = math.isqrt(len(per_cell))
    rows = [np.concatenate(per_cell[r * side : (r + 1) * side], axis=1) for r in range(side)]
    return np.concatenate(rows, axis=0)


def area_label(v: float) -> str:
    """Return a cell's area in whole hectares; a cell with mining never reads "0 ha"."""
    shown = common.number(v, 0)
    return "< 1 ha" if v > 0 and shown == "0" else f"{shown} ha"


def tile_cells(cells: pd.DataFrame, root: Path, stem: Path) -> None:
    """Show how a tile becomes 16 cells, and that one tile holds cells with and without mining.

    The tile chosen_tile returns: its true color, the reference mask outlined, the cell grid,
    and each cell's reference area at its top left.
    """
    tile = chosen_tile(cells)
    masks = np.load(root / "masks.npy", mmap_mode="r")[tile.cell_index.to_numpy()] > 0
    side, px = math.isqrt(len(tile)), masks.shape[1]
    fig, ax = style.figure_grid(1, 1, width=style.COLUMN, ratio=1.0)
    ax.imshow(mosaic(true_color(root, tile)), interpolation="nearest")
    ax.contour(mosaic(masks), levels=[0.5], colors=[style.PALETTE["orange"]], linewidths=0.7)
    for line in range(1, side):
        ax.axhline(line * px - 0.5, color="white", linewidth=0.6)
        ax.axvline(line * px - 0.5, color="white", linewidth=0.6)
    for position, v in zip(tile.position, tile.v, strict=True):
        row, col = divmod(int(position), side)
        ax.text(
            (col + 0.04) * px,
            (row + 0.06) * px,
            area_label(v),
            color="white",
            fontsize=6,
            va="top",
            # A dark rim keeps the label legible over bare soil, the brightest ground.
            path_effects=[matplotlib.patheffects.withStroke(linewidth=1.5, foreground="black")],
        )
    scale_bar(ax, side * px)
    ax.set_axis_off()
    first = tile.iloc[0]
    below(ax, f"{region_name(first.region)}, {ROLE_NAMES[first.role]}, recorte {first.tile_id}")
    common.save(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True, help="where the PDF and SVG files go")
    parser.add_argument("--split", type=Path, default=garimpo_uncertainty.data.SPLIT_PATH)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--subset-seed", type=int, default=0)
    args = parser.parse_args()
    style.use_style()
    root = garimpo_uncertainty.data.data_root()
    split = pd.read_csv(args.split)
    cells = read_cells(root, split, args.train_fraction, args.subset_seed)
    split_composition(cells, args.out / "split_composition")
    area_distribution(cells, args.out / "area_distribution")
    example_cells(cells, root, args.out / "example_cells")
    tile_map(tile_centers(root, split), args.out / "tile_map")
    tile_cells(cells, root, args.out / "tile_cells")


DATA_FIGURES = (
    "split_composition",
    "area_distribution",
    "example_cells",
    "tile_map",
    "tile_cells",
)
