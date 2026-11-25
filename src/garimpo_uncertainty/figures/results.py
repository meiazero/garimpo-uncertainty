"""The figures of the results chapter: one per research question, then its sensitivity checks.

Every interval is the 95% group bootstrap of compute_metrics, and every model is a single
training, which each caption must say.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import garimpo_uncertainty.figures.orx_figstyle as style
from garimpo_uncertainty.figures import common
from garimpo_uncertainty.figures.common import Run

PICP_LABEL = "PICP (%, meta tracejada)"
MPIW_LABEL = "MPIW (ha, escala log; menor é melhor)"


def model_grid(runs: list[Run], ratio: float, **kwargs: object) -> tuple[object, np.ndarray]:
    """One panel per model, two per row, so four models fill the text width in a square."""
    columns = min(len(runs), 2)
    rows = -(-len(runs) // columns)
    fig, axes = style.figure_grid(rows, columns, width=common.TEXT_WIDTH, ratio=ratio, **kwargs)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(runs) :]:
        ax.set_visible(False)
    return fig, axes[: len(runs)]


def label_models(axes: np.ndarray, runs: list[Run]) -> None:
    """Panel letters with the model's name, since the caption names models by letter."""
    style.panel_labels(axes, [f"({chr(97 + i)}) {run.name}" for i, run in enumerate(runs)])


def errorbar(ax: object, row: pd.Series, x: str, y: str, look: common.Look) -> None:
    """Draw a point with its bootstrap interval on each axis that has one."""
    xerr = interval(row, x)
    yerr = interval(row, y)
    ax.errorbar(
        row[x] * scale(x),
        row[y] * scale(y),
        xerr=None if xerr is None else xerr * scale(x),
        yerr=None if yerr is None else yerr * scale(y),
        linestyle="none",
        markersize=4,
        elinewidth=0.6,
        capsize=0,
        **look.marker_style(),
    )


def scale(column: str) -> float:
    return 100.0 if column == "picp" else 1.0


def interval(row: pd.Series, column: str) -> np.ndarray | None:
    low, high = f"{column}_ci_low", f"{column}_ci_high"
    if low not in row or pd.isna(row[low]):
        return None
    # A bound equal to the estimate, as when every resample covers the whole cell, can land one
    # rounding step past it, and matplotlib refuses a negative error.
    return np.maximum(np.array([[row[column] - row[low]], [row[high] - row[column]]]), 0)


def method_legend(fig: object, methods: list[str], looks: dict[str, common.Look]) -> None:
    """One legend under the panels, in the pipeline's method order."""
    handles = [
        style.plt.Line2D([], [], linestyle="none", markersize=4, **looks[m].marker_style())
        for m in methods
    ]
    labels = [common.method_name(m) for m in methods]
    fig.legend(handles, labels, loc="outside lower center", ncol=4)


def q1_area_error_and_boundary(runs: list[Run], stem: Path) -> None:
    """Show for Q1 whether a cell's area error follows how well its boundary is drawn.

    One point per test cell with reference mining, absolute area error against boundary F1, with
    the Spearman rho and its interval from q1_spearman.csv. It shows the whole spread the single
    correlation of the text summarizes, cells with a near-perfect boundary and a large error
    included, which is the case Q1 asks about.
    """
    fig, axes = model_grid(runs, 0.78, sharex=True, sharey=True)
    for ax, run in zip(axes, runs, strict=True):
        cells = common.arrays(run.paths.cell_metrics)
        predictions = common.arrays(run.paths.predictions)
        at = np.searchsorted(predictions["cell_index"], cells["cell_index"])
        error = np.abs(predictions["vhat"][at] - predictions["v"][at])
        mining = predictions["v"][at] > 0
        ax.scatter(
            cells["boundary_f1"][mining],
            error[mining],
            s=4,
            color=style.PALETTE["blue"],
            alpha=0.5,
            linewidth=0,
        )
        rho = run.table("q1_spearman")
        row = rho[(rho.area_error == "abs_error_ha") & (rho.segmentation_quality == "boundary_f1")]
        if len(row):
            row = row.iloc[0]
            ax.annotate(
                f"$\\rho$ = {common.number(row.spearman)} [{common.number(row.ci_low)}; "
                f"{common.number(row.ci_high)}]\nn = {row.n_cells}",
                xy=(0.03, 0.95),
                xycoords="axes fraction",
                ha="left",
                va="top",
                fontsize=7,
            )
        ax.set_xlim(-0.02, 1.02)
        ax.grid(visible=True, axis="both")
    for ax in axes[-2:]:
        ax.set_xlabel("F1 de fronteira")
    for ax in axes[::2]:
        ax.set_ylabel("Erro absoluto de área (ha)")
    label_models(axes, runs)
    common.save(fig, stem)


def q2_coverage_and_width(runs: list[Run], stem: Path) -> None:
    """Show for Q2 which methods reach the nominal coverage, and how wide those are.

    Coverage against width is a trade-off, so each method is one point in that plane, with its
    bootstrap interval on both axes; the dashed line is the 90% target. A method below the line
    misses the guarantee whatever its width, and among those above it the leftmost is the most
    informative. The table of the text gives the same numbers; the plane gives the ranking.
    """
    fig, axes = model_grid(runs, 0.86, sharex=True, sharey=True)
    methods = common.ordered_methods(runs[0].results)
    looks = common.looks(methods)
    for ax, run in zip(axes, runs, strict=True):
        rows = common.main_rows(run.results)
        common.target_line(ax, 100 * (1 - common.MAIN_ALPHA))
        for _, row in rows.iterrows():
            errorbar(ax, row, "mpiw", "picp", looks[row.method])
        ax.set_xscale("log")
        ax.grid(visible=True, axis="both")
    for ax in axes[-2:]:
        ax.set_xlabel(MPIW_LABEL)
    for ax in axes[::2]:
        ax.set_ylabel(PICP_LABEL)
    label_models(axes, runs)
    method_legend(fig, methods, looks)
    common.save(fig, stem)


# The three calibration situations of Q3, in the order the shift is introduced and then corrected.
SITUATIONS = ("own_region", "external", "external_weighted")
SITUATION_MARKERS = {"own_region": "o", "external": "s", "external_weighted": "^"}


def q3_calibration_situations(runs: list[Run], stem: Path) -> None:
    """Show for Q3 what calibrating in another region costs, and whether weighting helps.

    For each method, its coverage calibrated in the test's own region, in the other regions, and
    in the other regions weighted toward the test's, joined by a line. The drop from the first to
    the second is the cost of the shift; the third shows whether the density ratio recovers it.
    Only the methods the pipeline weights have a third point.
    """
    methods = [m for m in common.ordered_methods(runs[0].results) if m != "independent_pixels"]
    fig, axes = style.figure_grid(
        1,
        len(runs),
        width=common.TEXT_WIDTH,
        ratio=0.12 + 0.05 * len(methods),
        sharey=True,
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for ax, run in zip(axes, runs, strict=True):
        results = run.results[np.isclose(run.results.alpha, common.MAIN_ALPHA)]
        common.target_line(ax, 100 * (1 - common.MAIN_ALPHA), vertical=True)
        for y, method in enumerate(methods):
            rows = results[results.method == method].set_index("calibration")
            present = [s for s in SITUATIONS if s in rows.index]
            ax.plot(
                [100 * rows.loc[s, "picp"] for s in present],
                [y] * len(present),
                color=style.MUTED,
                linewidth=1.0,
                zorder=1,
            )
            for situation in present:
                ax.plot(
                    100 * rows.loc[situation, "picp"],
                    y,
                    linestyle="none",
                    marker=SITUATION_MARKERS[situation],
                    color=common.looks(methods)[method].color,
                    markersize=4,
                    zorder=2,
                )
        ax.set_yticks(range(len(methods)), [common.method_name(m) for m in methods])
        ax.set_xlabel("PICP (%)")
        ax.grid(visible=True, axis="x")
        ax.grid(visible=False, axis="y")
        ax.tick_params(axis="y", length=0)
    # Shared, so inverting one panel inverts them all: the first method on top.
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Método")
    style.panel_labels(axes, [f"({chr(97 + i)}) {run.name}" for i, run in enumerate(runs)])
    handles = [
        style.plt.Line2D(
            [], [], linestyle="none", marker=SITUATION_MARKERS[s], color="black", markersize=4
        )
        for s in SITUATIONS
    ]
    fig.legend(
        handles,
        [f"calibração {common.calibration_name(s)}" for s in SITUATIONS],
        loc="outside lower center",
        ncol=3,
    )
    common.save(fig, stem)


def area_band_ticks(table: pd.DataFrame) -> list[str]:
    """Name each band by the reference areas it holds, the empty cells first."""
    labels = []
    for _, row in table.drop_duplicates("stratum").sort_values("stratum").iterrows():
        labels.append(
            "0"
            if row.high_ha == 0
            else f"{common.number(row.low_ha)}\u2013\n{common.number(row.high_ha)}"
        )
    return labels


def coverage_by_area_band(runs: list[Run], stem: Path) -> None:
    """Show whether the coverage holds for small and large mined areas alike.

    The guarantee is marginal, so it may hold on average and fail in the empty cells or in the
    largest areas. One line per method across the six bands of reference area, the empty cells
    and the quintiles of the rest, at the main level with pooled external calibration.
    """
    fig, axes = model_grid(runs, 0.86, sharey=True)
    methods = common.ordered_methods(runs[0].results)
    looks = common.looks(methods)
    for ax, run in zip(axes, runs, strict=True):
        table = run.table("coverage_by_stratum")
        table = common.main_rows(table[table.partition == "reference_area"])
        common.target_line(ax, 100 * (1 - common.MAIN_ALPHA))
        for method in methods:
            rows = table[table.method == method].sort_values("stratum")
            look = looks[method]
            ax.plot(
                rows.stratum,
                100 * rows.picp,
                linewidth=0.8,
                markersize=3,
                linestyle="--" if look.hollow else "-",
                **look.marker_style(),
            )
        # Tied areas can leave a band empty, so the ticks go at the strata present, not 0 to n - 1.
        ax.set_xticks(sorted(table.stratum.unique()), area_band_ticks(table))
        ax.set_xlabel("Área de referência da faixa (ha)")
    for ax in axes[::2]:
        ax.set_ylabel("PICP (%)")
    label_models(axes, runs)
    method_legend(fig, methods, looks)
    common.save(fig, stem)


FIGURES = {
    "q1_area_error_and_boundary": q1_area_error_and_boundary,
    "q2_coverage_and_width": q2_coverage_and_width,
    "q3_calibration_situations": q3_calibration_situations,
    "coverage_by_area_band": coverage_by_area_band,
}
