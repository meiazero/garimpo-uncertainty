"""`garimpo data_figures`: figures of the data itself, for the text's description of it.

    garimpo data_figures --out outputs/figures

They read the cell store of GARIMPO_DATA_DIR and the split, never a run, so they draw on a CPU
before any model is trained. Cells are chosen from the reference alone.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import garimpo_uncertainty.data
import garimpo_uncertainty.figures.orx_figstyle as style
from garimpo_uncertainty.figures import common
from garimpo_uncertainty.figures.maps import below, scale_bar, true_color

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
        ax.bar_label(
            bars,
            [str(n) if n else "" for n in table[region]],
            label_type="center",
            fontsize=6,
        )
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
            f"{common.number(share, 0)}%\nn = {(cells.role == r).sum()}"
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
                label=f"{ROLE_NAMES[role]} (n = {len(area)})",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True, help="where the PDF and SVG files go")
    parser.add_argument("--split", type=Path, default=garimpo_uncertainty.data.SPLIT_PATH)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--subset-seed", type=int, default=0)
    args = parser.parse_args()
    style.use_style()
    root = garimpo_uncertainty.data.data_root()
    cells = read_cells(root, pd.read_csv(args.split), args.train_fraction, args.subset_seed)
    split_composition(cells, args.out / "split_composition")
    area_distribution(cells, args.out / "area_distribution")
    example_cells(cells, root, args.out / "example_cells")


DATA_FIGURES = ("split_composition", "area_distribution", "example_cells")
