"""The qualitative figure: what a test cell, its reference and each model's prediction look like.

Cells are chosen from the reference alone, never from a model's result, so the figure shows the
range of the test set rather than the cases a method handles best.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import garimpo_uncertainty.data
import garimpo_uncertainty.evaluation
import garimpo_uncertainty.figures.orx_figstyle as style
import garimpo_uncertainty.intervals
from garimpo_uncertainty.figures import common
from garimpo_uncertainty.figures.common import Run

# One empty cell, then the mining cells at these quantiles of reference area, the largest last.
AREA_QUANTILES = (0.25, 0.5, 0.75, 1.0)
# The two intervals the text compares cell by cell: the output-space baseline and COMPASS-J.
SHOWN_METHODS = (("conformal_simple", "CS"), ("compass_j_pca", "CJ"))
TRUE_COLOR = tuple(garimpo_uncertainty.data.BANDS.index(band) for band in ("B04", "B03", "B02"))
# The same stretch for every cell shown, so brightness compares across rows.
STRETCH_PERCENTILES = (2, 98)
SCALE_BARS_M = (100, 200, 500, 1000, 2000)
TEXT_LINES_IN, COLORBAR_IN = 0.4, 0.5
KM = 1000


def chosen_cells(test: pd.DataFrame) -> pd.DataFrame:
    """Return the empty test cell first in cell order and the mining cells at AREA_QUANTILES."""
    empty = test[test.v == 0].sort_values("cell_index").head(1)
    mining = test[test.v > 0].sort_values(["v", "cell_index"]).reset_index(drop=True)
    at = [round(q * (len(mining) - 1)) for q in AREA_QUANTILES] if len(mining) else []
    return pd.concat([empty, mining.iloc[sorted(set(at))]], ignore_index=True)


def true_color(root: Path, cells: pd.DataFrame) -> np.ndarray:
    """Return the cells' red, green and blue reflectance, stretched together to [0, 1]."""
    images = np.load(root / "images.npy", mmap_mode="r")
    raw = images[cells.cell_index.to_numpy()][:, TRUE_COLOR].astype(np.float32)
    offsets = cells.boa_offset.to_numpy(dtype=np.float32).reshape(-1, 1, 1, 1)
    reflectance = np.maximum(raw + offsets, 0) / garimpo_uncertainty.data.REFLECTANCE_SCALE
    low, high = np.percentile(reflectance, STRETCH_PERCENTILES, axis=(0, 2, 3), keepdims=True)
    rgb = np.clip((reflectance - low) / np.maximum(high - low, 1e-6), 0, 1)
    return rgb.transpose(0, 2, 3, 1)


def probabilities(run: Run, cells: pd.DataFrame) -> np.ndarray:
    """Return the run's stored test probabilities of the cells, in [0, 1]."""
    test = run.cells[run.cells.role == "test"].reset_index(drop=True)
    rows = garimpo_uncertainty.data.positions_of(
        test.cell_index.to_numpy(), cells.cell_index.to_numpy()
    )
    stored = np.load(run.paths.probabilities("test"), mmap_mode="r")
    return stored[rows] / garimpo_uncertainty.evaluation.PROBABILITY_SCALE


def interval_text(run: Run, cells: pd.DataFrame) -> list[str]:
    """Return per cell the run's predicted area and its two shown intervals, in hectares."""
    stored = common.arrays(run.paths.intervals)
    level = int(np.flatnonzero(np.isclose(stored["alpha"], common.MAIN_ALPHA))[0])
    at = garimpo_uncertainty.data.positions_of(stored["cell_index"], cells.cell_index.to_numpy())
    vhat = run.cells.set_index("cell_index").loc[cells.cell_index, "vhat"].to_numpy()
    combinations = garimpo_uncertainty.intervals.COMBINATIONS
    lines = [[f"$\\hat{{v}}$ = {common.number(value)}"] for value in vhat]
    for method, short in SHOWN_METHODS:
        k = combinations.index((method, common.POOLED))
        lower, upper = stored["lower"][level, k][at], stored["upper"][level, k][at]
        for line, low, high in zip(lines, lower, upper, strict=True):
            line.append(f"{short} [{common.number(low)}; {common.number(high)}]")
    return ["\n".join(line) for line in lines]


def scale_bar(ax: object, pixels: int) -> None:
    """Draw the longest standard length under a third of the cell, at the bottom left."""
    width_m = pixels * garimpo_uncertainty.data.PIXEL_M
    length = max((m for m in SCALE_BARS_M if m <= width_m / 3), default=SCALE_BARS_M[0])
    bar = length / garimpo_uncertainty.data.PIXEL_M
    y = pixels * 0.92
    ax.plot([pixels * 0.06, pixels * 0.06 + bar], [y, y], color="white", linewidth=1.5)
    label = f"{length // KM} km" if length >= KM else f"{length} m"
    ax.text(pixels * 0.06, y - pixels * 0.04, label, color="white", fontsize=6, va="bottom")


def qualitative_cells(runs: list[Run], stem: Path) -> None:
    """Show what the models predict on test cells across the range of reference area.

    Per row one test cell: its true color, its reference mask, and each model's probability with
    the reference outline, the predicted area and the intervals of the conformal simple (CS) and
    COMPASS-J (CJ) under it. The first row is an empty cell; the others are the mining cells at the
    quartiles and the maximum of reference area, chosen before any model is read.
    """
    cells = chosen_cells(runs[0].cells[runs[0].cells.role == "test"])
    root = garimpo_uncertainty.data.data_root()
    masks = np.load(root / "masks.npy", mmap_mode="r")[cells.cell_index.to_numpy()] > 0
    columns = 2 + len(runs)
    # A square panel, three lines of text under it, and the colorbar under the grid.
    height = len(cells) * (common.TEXT_WIDTH / columns + TEXT_LINES_IN) + COLORBAR_IN
    fig, axes = style.figure_grid(
        len(cells), columns, width=common.TEXT_WIDTH, ratio=height / common.TEXT_WIDTH
    )
    axes = np.atleast_2d(axes)
    draw_references(axes, true_color(root, cells), masks, cells.v)
    for j, run in enumerate(runs):
        image = draw_predictions(axes[:, 2 + j], run, cells, masks)
    for ax in axes.ravel():
        ax.set_axis_off()
    style.panel_labels(axes[0], ["Cor real", "Referência", *(run.name for run in runs)])
    for i, ax in enumerate(axes[:, 0]):
        ax.text(-0.06, 0.5, f"({chr(97 + i)})", transform=ax.transAxes, ha="right", va="center")
    colorbar = fig.colorbar(image, ax=axes[:, 2:], location="bottom", shrink=0.6, aspect=40)
    colorbar.set_label("Probabilidade de mineração prevista (contorno branco: referência)")
    common.save(fig, stem)


def draw_references(axes: np.ndarray, rgb: np.ndarray, masks: np.ndarray, v: pd.Series) -> None:
    for row, image, mask, area in zip(axes, rgb, masks, v, strict=True):
        row[0].imshow(image, interpolation="nearest")
        scale_bar(row[0], image.shape[1])
        row[1].imshow(mask, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        # An empty mask is all white; the frame keeps the cell visible against the page.
        row[1].add_patch(
            style.mpl.patches.Rectangle(
                (-0.5, -0.5), *mask.shape[::-1], fill=False, color="black", linewidth=0.5
            )
        )
        below(row[1], f"v = {common.number(area)} ha")


def draw_predictions(
    column: np.ndarray, run: Run, cells: pd.DataFrame, masks: np.ndarray
) -> object:
    """Draw one model's probabilities down a column; return the last image for the colorbar."""
    image = None
    for ax, probability, mask, text in zip(
        column, probabilities(run, cells), masks, interval_text(run, cells), strict=True
    ):
        image = ax.imshow(
            probability, cmap=style.SEQUENTIAL, vmin=0, vmax=1, interpolation="nearest"
        )
        ax.contour(mask, levels=[0.5], colors="white", linewidths=0.6)
        below(ax, text)
    return image


def below(ax: object, text: str) -> None:
    ax.text(0.5, -0.03, text, transform=ax.transAxes, ha="center", va="top", fontsize=6)


FIGURES = {"qualitative_cells": qualitative_cells}
