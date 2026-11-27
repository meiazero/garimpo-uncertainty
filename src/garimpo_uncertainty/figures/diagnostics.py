"""The figures that explain the results: training, layers, scores, calibration and each cell.

The results figures say what each method achieves; these say why, so a wide interval, a censored
search or a layer that behaves apart can be traced to the stage that produced it.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import garimpo_uncertainty.figures.orx_figstyle as style
import garimpo_uncertainty.intervals
import garimpo_uncertainty.models
from garimpo_uncertainty.figures import common
from garimpo_uncertainty.figures.common import Run
from garimpo_uncertainty.figures.results import (
    label_models,
    method_legend,
    model_grid,
)

logger = logging.getLogger(__name__)
MODEL_MARKERS = ("o", "s", "^", "D", "v", "P")


def rows_by_models(runs: list[Run], columns: int, ratio_per_row: float, **kwargs: object) -> tuple:
    """One row of panels per model, sized to the text width."""
    fig, axes = style.figure_grid(
        len(runs),
        columns,
        width=common.TEXT_WIDTH,
        ratio=ratio_per_row * len(runs),
        squeeze=False,
        **kwargs,
    )
    return fig, axes


def name_rows(axes: np.ndarray, runs: list[Run], titles: list[str]) -> None:
    """Letter every panel, name the columns on the first row and each row's model on its left.

    The model goes into the first column's y label, above the quantity, so a long name never runs
    off a narrow panel.
    """
    labels = [
        f"({chr(97 + i * len(titles) + j)})" + (f" {title}" if i == 0 else "")
        for i in range(len(runs))
        for j, title in enumerate(titles)
    ]
    style.panel_labels(axes, labels)
    for ax, run in zip(axes[:, 0], runs, strict=True):
        quantity = ax.get_ylabel()
        ax.set_ylabel(f"{run.name}\n{quantity}" if quantity else run.name)


def training_curves(runs: list[Run], stem: Path) -> None:
    """Show how each model trained, and that best_model.pt is the epoch with the best Dice.

    Validation Dice and both losses per epoch, one line per model, the selected epoch starred. It
    justifies the checkpoint every later stage reads, and shows whether a model stopped while still
    improving or after it began to fit the training cells better than new ones.
    """
    fig, axes = style.figure_grid(1, 2, width=common.TEXT_WIDTH, ratio=0.4, sharex=True)
    for color, marker, run in zip(style.CYCLE, MODEL_MARKERS, runs, strict=False):
        history = pd.read_csv(run.paths.history)
        epoch = history.epoch + 1
        axes[0].plot(epoch, history.val_dice, color=color, marker=marker, label=run.name)
        best = history.val_dice.idxmax()
        axes[0].plot(
            epoch[best], history.val_dice[best], marker="*", markersize=9, color=color, zorder=3
        )
        axes[1].plot(epoch, history.val_loss, color=color, marker=marker)
        axes[1].plot(epoch, history.train_loss, color=color, linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Dice na validação")
    axes[1].set_ylabel("Perda Dice + BCE (tracejado: treino)")
    for ax in axes:
        ax.set_xlabel("Época")
    style.panel_labels(axes)
    fig.legend(loc="outside lower center", ncol=len(runs))
    common.save(fig, stem)


# The methods drawn cell by cell: the output-space baseline and each COMPASS family's first member.
CELL_METHODS = ("conformal_simple", "compass_l", "compass_j_pca")


def interval_rows(run: Run, alpha: float = common.MAIN_ALPHA) -> tuple[pd.DataFrame, dict]:
    """Return the test cells in order of reference area, and each method's pooled interval."""
    stored = common.arrays(run.paths.intervals)
    level = int(np.flatnonzero(np.isclose(stored["alpha"], alpha))[0])
    cells = run.cells.set_index("cell_index").loc[stored["cell_index"]].reset_index()
    order = np.argsort(cells.v.to_numpy(), kind="stable")
    combinations = garimpo_uncertainty.intervals.COMBINATIONS
    bounds = {}
    for method in CELL_METHODS:
        k = combinations.index((method, common.POOLED))
        bounds[method] = (stored["lower"][level, k][order], stored["upper"][level, k][order])
    return cells.iloc[order].reset_index(drop=True), bounds


def intervals_by_cell(runs: list[Run], stem: Path) -> None:
    """Show where each method's intervals miss: every test cell, in order of reference area.

    One vertical segment per cell from its lower to its upper bound, the reference area as a dot,
    and the cells the interval misses in a second marker. Coverage and width are averages; this
    shows which cells pay for them, the empty cells and the largest areas above all.
    """
    titles = [common.method_name(m) for m in CELL_METHODS]
    fig, axes = rows_by_models(runs, len(CELL_METHODS), 0.3, sharex=True, sharey=True)
    looks = common.looks(list(CELL_METHODS))
    for row, run in zip(axes, runs, strict=True):
        cells, bounds = interval_rows(run)
        rank = np.arange(len(cells))
        for ax, method in zip(row, CELL_METHODS, strict=True):
            lower, upper = bounds[method]
            missed = (cells.v < lower) | (cells.v > upper)
            ax.vlines(rank, lower, upper, color=looks[method].color, linewidth=0.4, alpha=0.6)
            ax.plot(rank[~missed], cells.v[~missed], ".", color="black", markersize=1.5)
            ax.plot(rank[missed], cells.v[missed], "x", color=style.PALETTE["red"], markersize=2.5)
            # Empty cells have no logarithm; symlog keeps them on the axis, linear below 0.1 ha.
            ax.set_yscale("symlog", linthresh=0.1)
    fig.supxlabel(
        "Célula de teste, por área de referência", fontsize=style.mpl.rcParams["axes.labelsize"]
    )
    for ax in axes[:, 0]:
        ax.set_ylabel("Área (ha)")
    name_rows(axes, runs, titles)
    common.save(fig, stem)


def score_pairs(run: Run) -> tuple[np.ndarray, dict, np.ndarray]:
    """Return the calibration cells' split-conformal scores, COMPASS scores and saturation."""
    stored = common.arrays(run.paths.scores)
    calibration = run.cells[run.cells.role == "calibration"]
    at = np.searchsorted(stored["cell_index"], calibration.cell_index.to_numpy())
    split = np.abs(calibration.v - calibration.vhat).to_numpy()
    return split, stored["score"][:, at], stored["saturated"][:, at]


def score_relation(runs: list[Run], stem: Path) -> None:
    """Show why COMPASS intervals are narrower: their scores grow slower than the area error.

    Cheung et al. (2026), Figure 4, on this data: each calibration cell's COMPASS score against its
    split-conformal score, on log axes, with the fitted line of score_relation.csv. A slope below 1
    compresses the tail the quantile is taken from. Censored searches sit on the top edge as
    triangles; zero scores have no logarithm and are counted, not drawn.
    """
    methods = garimpo_uncertainty.intervals.COMPASS_METHODS
    titles = [common.method_name(m) for m in methods]
    # Each column shares its axes: the COMPASS-L beta perturbs logits, COMPASS-J's the features.
    fig, axes = rows_by_models(runs, len(methods), 0.3, sharex="col", sharey="col")
    for row, run in zip(axes, runs, strict=True):
        split, scores, saturated = score_pairs(run)
        fits = run.table("score_relation").set_index("method")
        for ax, m, method in zip(row, range(len(methods)), methods, strict=True):
            draw_scores(ax, split, scores[m], saturated[m], fits.loc[method])
    for ax in axes[-1]:
        ax.set_xlabel(r"Escore conformal $|\hat{v} - v|$ (ha)")
    for ax in axes[0]:
        ax.set_ylabel("Escore do COMPASS (β)")
    name_rows(axes, runs, titles)
    common.save(fig, stem)


def draw_scores(
    ax: object, split: np.ndarray, score: np.ndarray, saturated: np.ndarray, fit: pd.Series
) -> None:
    kept = (split > 0) & (score > 0) & ~saturated
    ax.scatter(split[kept], score[kept], s=3, color=style.PALETTE["blue"], alpha=0.5, linewidth=0)
    censored = saturated & (split > 0)
    if censored.any():
        top = score[kept].max() if kept.any() else 1.0
        ax.plot(split[censored], np.full(censored.sum(), top), "^", color=style.PALETTE["red"])
    if np.isfinite(fit.slope) and kept.any():
        x = np.geomspace(split[kept].min(), split[kept].max(), 50)
        ax.plot(x, np.exp(fit.intercept) * x**fit.slope, color="black", linewidth=0.8)
        ax.annotate(
            f"inclinação {common.number(fit.slope)}\nn = {common.number(kept.sum(), 0)}, zeros = "
            f"{common.number(((split == 0) | (score == 0)).sum(), 0)}, "
            f"censuradas = {common.number(saturated.sum(), 0)}",
            # Lower right: a large conformal score with a small COMPASS one is the rare case, and
            # censored scores fill the top.
            xy=(0.97, 0.04),
            xycoords="axes fraction",
            ha="right",
            fontsize=6,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Narrow panels: a tick per decade would collide, and labelled minor ticks crowd the axis.
    ax.xaxis.set_major_locator(style.mpl.ticker.LogLocator(numticks=4))
    ax.yaxis.set_minor_formatter(style.mpl.ticker.NullFormatter())


def layer_table(run: Run) -> pd.DataFrame:
    """Return per layer its first-component share, validation monotonicity and COMPASS-J rows."""
    diagnostics = pd.read_csv(run.paths.direction_diagnostics).set_index("layer")
    monotonicity = pd.read_csv(run.paths.monotonicity("layer"))
    validation = monotonicity[monotonicity.role == "validation"]
    ascending = validation.pivot_table(
        index="layer", columns="direction", values="fraction_ascending"
    )
    table = diagnostics[["pca_explained_variance"]].join(ascending)
    widths = {
        layer: common.main_rows(pd.read_csv(paths.results)).set_index("method")
        for layer, paths in run.layer_runs().items()
    }
    for method in ("compass_j_pca", "compass_j_svm"):
        for column in ("mpiw", "mpiw_ci_low", "mpiw_ci_high", "picp"):
            table[f"{method}_{column}"] = [
                widths[layer].loc[method, column] if layer in widths else np.nan
                for layer in table.index
            ]
    return table


def layer_comparison(runs: list[Run], stem: Path) -> None:
    """Show why the chosen layer was chosen, and what another layer would have given.

    Per candidate layer: the share of the Jacobians' variance in the first PCA component, the share
    of validation cells whose area grows with beta in each direction, and the width of the
    COMPASS-J intervals at the main level, Cheung et al. (2026), Table 4. The chosen layer's tick
    is bold. A layer that the rule rejects but gives narrower intervals would show here.
    """
    titles = ["1ª componente", "respostas crescentes", "MPIW do COMPASS-J"]
    fig, axes = rows_by_models(runs, len(titles), 0.3)
    for row, run in zip(axes, runs, strict=True):
        table = layer_table(run)
        x = np.arange(len(table))
        row[0].bar(x, 100 * table.pca_explained_variance, color=style.MUTED, edgecolor="black")
        draw_ascending(row[1], x, table)
        draw_layer_widths(row[2], x, table)
        for ax in row:
            mark_chosen(ax, x, table.index, run.chosen_layer)
        row[0].set_ylabel("Variância explicada (%)")
        row[1].set_ylabel("Células crescentes (%)")
        row[2].set_ylabel("MPIW (ha)")
    axes[0, 1].legend()
    name_rows(axes, runs, titles)
    common.save(fig, stem)


def draw_ascending(ax: object, x: np.ndarray, table: pd.DataFrame) -> None:
    for direction, marker in (("pca", "o"), ("svm", "s")):
        if direction in table:
            ax.plot(x, 100 * table[direction], marker=marker, label=direction.upper())


def draw_layer_widths(ax: object, x: np.ndarray, table: pd.DataFrame) -> None:
    for method, look in common.looks(["compass_j_pca", "compass_j_svm"]).items():
        width = table[f"{method}_mpiw"]
        errors = np.maximum(
            [width - table[f"{method}_mpiw_ci_low"], table[f"{method}_mpiw_ci_high"] - width], 0
        )
        ax.errorbar(x, width, yerr=errors, linestyle="none", **look.marker_style())
    # A saturating layer gives whole-cell intervals, decades above the chosen layer's.
    ax.set_yscale("log")


def layer_name(layer: str) -> str:
    """Name a decoder layer as the text does: bloco 0 to 4 for the U-Net, aspp and relu."""
    name = layer.removeprefix("decoder.")
    return f"bloco {name.removeprefix('blocks.')}" if name.startswith("blocks.") else name


def mark_chosen(ax: object, x: np.ndarray, layers: pd.Index, chosen: str) -> None:
    """Name each layer without the shared prefix, the chosen one in bold, on two lines to fit."""
    ax.set_xticks(x, [layer_name(layer).replace(" ", "\n") for layer in layers])
    for tick, layer in zip(ax.get_xticklabels(), layers, strict=True):
        tick.set_fontweight("bold" if layer == chosen else "normal")


# What each sweep perturbs: every logit by the same beta, or the chosen layer along a direction.
SWEEPS = (
    ("uniform", "logits", "COMPASS-L: logits"),
    ("pca", "features", "COMPASS-J: PCA"),
    ("svm", "features", "COMPASS-J: SVM"),
)


def monotonicity_curves(runs: list[Run], stem: Path) -> None:
    """Show what the monotonicity check of Algorithm 2 sees: the area of each cell as beta grows.

    One line per validation cell the layer rule sweeps, at the chosen layer, the cells whose area
    ever falls in a second color. The share of ascending cells is the number layer_comparison
    draws; the curves show where the area saturates and how many cells never move, which a share
    alone hides.
    """
    runs = with_curves(runs)
    if not runs:
        return
    # A row holds the same cells, so one area axis serves its three sweeps.
    fig, axes = rows_by_models(runs, len(SWEEPS), 0.3, sharex="col", sharey="row")
    for row, run in zip(axes, runs, strict=True):
        with np.load(run.paths.monotonicity_curves("layer")) as stored:
            for ax, (direction, grid, _) in zip(row, SWEEPS, strict=True):
                layer = (
                    garimpo_uncertainty.models.LOGITS
                    if direction == "uniform"
                    else run.chosen_layer
                )
                draw_curves(ax, stored[f"beta_{grid}"], stored[f"validation:{layer}:{direction}"])
        row[0].set_ylabel("Área prevista (ha)")
    for ax, (_, grid, _) in zip(axes[-1], SWEEPS, strict=True):
        ax.set_xlabel("β nos logits" if grid == "logits" else "β nas características")
    name_rows(axes, runs, [title for _, _, title in SWEEPS])
    common.save(fig, stem)


def with_curves(runs: list[Run]) -> list[Run]:
    """Keep the runs with curves: one finished before the stage kept them has none."""
    missing = [run.name for run in runs if not run.paths.monotonicity_curves("layer").exists()]
    if missing:
        logger.warning("monotonicity_curves: no curves for %s", ", ".join(missing))
    return [run for run in runs if run.name not in missing]


def draw_curves(ax: object, beta: np.ndarray, curves: np.ndarray) -> None:
    ascending = (np.diff(curves, axis=0) >= 0).all(0)
    ax.plot(beta, curves[:, ascending], color=style.PALETTE["blue"], linewidth=0.5, alpha=0.6)
    ax.plot(beta, curves[:, ~ascending], color=style.PALETTE["red"], linewidth=0.8)
    ax.annotate(
        f"crescentes: {common.number(ascending.sum(), 0)} de {common.number(len(ascending), 0)}",
        xy=(0.03, 0.95),
        xycoords="axes fraction",
        va="top",
        fontsize=6,
    )
    ax.grid(visible=True, axis="both")


def density_ratio(runs: list[Run], stem: Path) -> None:
    """Show how far the weighted calibration leans on a few calibration cells.

    Histograms of the density ratio that weights the external calibration toward the test region,
    for the calibration cells and the test cells, on a log axis. A long right tail means few cells
    carry the weighted quantile; the effective number of groups calibrate.py records says so too.
    """
    fig, axes = model_grid(runs, 0.6, sharex=True)
    for ax, run in zip(axes, runs, strict=True):
        stored = common.arrays(run.paths.quantiles)
        ratios = [stored["density_ratio_calibration"], stored["density_ratio_test"]]
        positive = np.concatenate(ratios)
        positive = positive[positive > 0]
        bins = np.geomspace(positive.min(), positive.max(), 30)
        for values, label, color in zip(
            ratios,
            ("calibração", "teste"),
            (style.PALETTE["blue"], style.PALETTE["orange"]),
            strict=True,
        ):
            ax.hist(values[values > 0], bins=bins, histtype="step", color=color, label=label)
        ess = common.stage_shapes(run.paths.layer_status, "calibrate")["weighted_group_ess"]
        ax.annotate(
            # Top left: the test cells sit right of the calibration ones, where weighting looks.
            f"grupos efetivos: {common.number(ess, 1)}",
            xy=(0.03, 0.95),
            xycoords="axes fraction",
            va="top",
            fontsize=7,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Razão de densidade (escala log)")
        ax.set_ylabel("Células")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    label_models(axes, runs)
    common.save(fig, stem)


REFERENCE = ("conformal_simple", common.POOLED)


def paired_rows(table: pd.DataFrame, alpha: float = common.MAIN_ALPHA) -> pd.DataFrame:
    """Return each pooled method minus the reference; the table stores row a minus row b."""
    level = table[np.isclose(table.alpha, alpha)]
    a = pd.Series(list(zip(level.method_a, level.calibration_a, strict=True)), index=level.index)
    b = pd.Series(list(zip(level.method_b, level.calibration_b, strict=True)), index=level.index)
    as_b = level[(b == REFERENCE) & (a.str[1] == common.POOLED)]
    as_a = level[(a == REFERENCE) & (b.str[1] == common.POOLED)]
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "method": as_b.method_a,
                    "difference": as_b.mean_difference,
                    "low": as_b.ci_low,
                    "high": as_b.ci_high,
                }
            ),
            pd.DataFrame(
                {
                    "method": as_a.method_b,
                    "difference": -as_a.mean_difference,
                    "low": -as_a.ci_high,
                    "high": -as_a.ci_low,
                }
            ),
        ],
        ignore_index=True,
    )


def paired_differences(runs: list[Run], stem: Path) -> None:
    """Show the paired differences of each method from the conformal simple, with their intervals.

    Two overlapping intervals of coverage or width do not mean the methods are alike; the interval
    of the difference, from the same group resamples, does. Coverage in percentage points and width
    in hectares, each method minus the conformal simple, pooled external calibration, main level.
    The independent pixels are left out: they have no calibration, so their difference measures
    calibrating at all rather than the method, and the Q2 figure already shows it.
    """
    titles = ["cobertura (p.p.)", "largura (ha)"]
    fig, axes = rows_by_models(runs, 2, 0.32, sharey=True)
    methods = [
        m
        for m in common.ordered_methods(runs[0].results)
        if m not in (REFERENCE[0], "independent_pixels")
    ]
    for row, run in zip(axes, runs, strict=True):
        for ax, name, factor in zip(row, ("coverage", "width"), (100.0, 1.0), strict=True):
            table = paired_rows(run.table(f"{name}_comparisons")).set_index("method")
            ax.axvline(0, color=style.BASELINE, linewidth=0.8)
            for y, method in enumerate(methods):
                if method in table.index:
                    point = table.loc[method] * factor
                    ax.plot([point.low, point.high], [y, y], color="black", linewidth=0.8)
                    ax.plot(point.difference, y, "o", color="black")
            ax.set_yticks(range(len(methods)), [common.method_name(m) for m in methods])
            ax.grid(visible=True, axis="x")
            ax.tick_params(axis="y", length=0)
        row[0].set_xlabel("Diferença de PICP (p.p.)")
        row[1].set_xlabel("Diferença de MPIW (ha)")
    axes[0, 0].invert_yaxis()
    name_rows(axes, runs, titles)
    common.save(fig, stem)


def coverage_by_level(runs: list[Run], stem: Path) -> None:
    """Show whether each method keeps its coverage at every nominal level, not only at 90%.

    Observed coverage against the nominal 1 - alpha for the three levels of Cheung et al. (2026),
    Table 1, pooled external calibration. Points on the diagonal are calibrated; below it, the
    method promises more than it delivers at that level.
    """
    fig, axes = model_grid(runs, 0.86, sharex=True, sharey=True)
    methods = common.ordered_methods(runs[0].results)
    looks = common.looks(methods)
    levels = 100 * (1 - np.array(garimpo_uncertainty.intervals.ALPHAS))
    for ax, run in zip(axes, runs, strict=True):
        rows = run.results[run.results.calibration.isin(("none", common.POOLED))]
        ax.plot(levels, levels, color=style.BASELINE, linewidth=0.8, linestyle="--")
        for method in methods:
            line = rows[rows.method == method].sort_values("alpha", ascending=False)
            ax.plot(
                100 * (1 - line.alpha),
                100 * line.picp,
                linewidth=0.8,
                **looks[method].marker_style(),
            )
        ax.set_xticks(levels)
    for ax in axes[-2:]:
        ax.set_xlabel("Cobertura nominal (%)")
    for ax in axes[::2]:
        ax.set_ylabel("PICP (%)")
    label_models(axes, runs)
    method_legend(fig, methods, looks)
    common.save(fig, stem)


QUANTILE_RULES = (common.POOLED, f"{common.POOLED}_subsampled")


def quantile_rule(runs: list[Run], stem: Path) -> None:
    """Show whether the choice of quantile rule changes coverage, width or the ranking.

    Each method's coverage and width with the pooled scores, the main rule, and with repeated
    subsampling of one cell per group, the sensitivity rule of docs/design.md, joined by a line.
    A result that holds under both rules does not depend on how groups are pooled.
    """
    titles = ["PICP (%)", "MPIW (ha)"]
    fig, axes = rows_by_models(runs, 2, 0.32, sharey=True)
    methods = [m for m in common.ordered_methods(runs[0].results) if m != "independent_pixels"]
    looks = common.looks(methods)
    for row, run in zip(axes, runs, strict=True):
        results = run.results[np.isclose(run.results.alpha, common.MAIN_ALPHA)]
        for ax, column, factor in zip(row, ("picp", "mpiw"), (100.0, 1.0), strict=True):
            for y, method in enumerate(methods):
                pair = results[results.method == method].set_index("calibration")
                values = [factor * pair.loc[r, column] for r in QUANTILE_RULES if r in pair.index]
                ax.plot(values, [y] * len(values), color=style.MUTED, zorder=1)
                for value, marker in zip(values, ("o", "s"), strict=False):
                    ax.plot(value, y, marker, color=looks[method].color, zorder=2)
            ax.set_yticks(range(len(methods)), [common.method_name(m) for m in methods])
            ax.tick_params(axis="y", length=0)
            ax.grid(visible=True, axis="x")
        common.target_line(row[0], 100 * (1 - common.MAIN_ALPHA), vertical=True)
        row[0].set_xlabel("PICP (%)")
        row[1].set_xlabel("MPIW (ha)")
    axes[0, 0].invert_yaxis()
    handles = [
        style.plt.Line2D([], [], linestyle="none", marker=m, color="black") for m in ("o", "s")
    ]
    fig.legend(
        handles,
        ["agregação dos escores", "subamostragem repetida"],
        loc="outside lower center",
        ncol=2,
    )
    name_rows(axes, runs, titles)
    common.save(fig, stem)


STRATUM_NAMES = {"predicted_mining": {"0": "sem mineração prevista", "1": "com mineração prevista"}}
PARTITIONS = ("region", "predicted_mining")


def coverage_by_region_and_prediction(runs: list[Run], stem: Path) -> None:
    """Show where the marginal guarantee holds: per test region and per predicted mining.

    The guarantee is over all test cells, so a region or the cells predicted empty may fall short.
    Coverage per stratum, one marker per stratum and a row per method, pooled external calibration
    at the main level. Predicted mining is the only stratum known when an interval is used.
    """
    titles = ["por região", "por mineração prevista"]
    fig, axes = rows_by_models(runs, 2, 0.34, sharey=True, sharex=True)
    methods = common.ordered_methods(runs[0].results)
    for row, run in zip(axes, runs, strict=True):
        table = common.main_rows(run.table("coverage_by_stratum"))
        for ax, partition in zip(row, PARTITIONS, strict=True):
            draw_strata(ax, table[table.partition == partition], methods, partition)
        row[0].legend(loc="lower left", fontsize=6)
        row[1].legend(loc="lower left", fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("PICP (%)")
    axes[0, 0].invert_yaxis()
    name_rows(axes, runs, titles)
    common.save(fig, stem)


def draw_strata(ax: object, table: pd.DataFrame, methods: list[str], partition: str) -> None:
    common.target_line(ax, 100 * (1 - common.MAIN_ALPHA), vertical=True)
    names = STRATUM_NAMES.get(partition, {})
    strata = table.drop_duplicates("stratum").sort_values("stratum")
    for marker, color, (_, stratum) in zip(
        MODEL_MARKERS, style.CYCLE, strata.iterrows(), strict=False
    ):
        rows = table[table.stratum == stratum.stratum].set_index("method")
        y = [i for i, m in enumerate(methods) if m in rows.index]
        x = [100 * rows.loc[m, "picp"] for m in methods if m in rows.index]
        label = names.get(str(stratum.label), str(stratum.label))
        ax.plot(
            x,
            y,
            marker,
            color=color,
            linestyle="none",
            label=f"{label} (n = {common.number(stratum.n_cells, 0)})",
        )
    ax.set_yticks(range(len(methods)), [common.method_name(m) for m in methods])
    ax.tick_params(axis="y", length=0)
    ax.grid(visible=True, axis="x")


FIGURES = {
    "training_curves": training_curves,
    "intervals_by_cell": intervals_by_cell,
    "score_relation": score_relation,
    "layer_comparison": layer_comparison,
    "monotonicity_curves": monotonicity_curves,
    "density_ratio": density_ratio,
    "paired_differences": paired_differences,
    "coverage_by_level": coverage_by_level,
    "quantile_rule": quantile_rule,
    "coverage_by_region_and_prediction": coverage_by_region_and_prediction,
}
