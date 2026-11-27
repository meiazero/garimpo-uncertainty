"""The figures of the COMPASS beta search: where it saturates, what it costs, how far it must go.

Saturation and cost come from the status files every run writes. Convergence and the threshold
trade-off come from search_profile.npz, which `garimpo search_profile` writes on request; a run
without it is left out of those two figures, with a warning.
"""

import json
import logging
from pathlib import Path

import numpy as np

import garimpo_uncertainty.data
import garimpo_uncertainty.figures.orx_figstyle as style
import garimpo_uncertainty.intervals
from garimpo_uncertainty.figures import common
from garimpo_uncertainty.figures.common import Run
from garimpo_uncertainty.figures.diagnostics import (
    layer_name,
    mark_chosen,
    name_rows,
    rows_by_models,
)
from garimpo_uncertainty.figures.results import label_models, model_grid
from garimpo_uncertainty.stages.search_profile import DIRECTIONS, PROFILE

logger = logging.getLogger(__name__)
# The stages that repeat per layer and cost most: the search, then the intervals it feeds.
COSTLY_STAGES = (("compass_scores", "escores"), ("generate_intervals", "intervalos"))


# Below ten minutes a total gets a decimal, or a short stage would read as zero.
WHOLE_MINUTES = 10


def status(paths: object, stage: str) -> dict:
    return json.loads((paths.layer_status / f"{stage}.json").read_text(encoding="utf-8"))


def layers_in_order(run: Run) -> list[tuple[str, object]]:
    """Return the run's layers with results in decoder order, so the chosen one keeps its place."""
    return sorted(run.layer_runs().items())


def bar_labels(ax: object, bars: object, digits: int) -> None:
    """Write each bar's value on it; redundant with its height, so it may sit at the 5 pt floor."""
    ax.bar_label(bars, [common.number(v, digits) for v in bars.datavalues], fontsize=5, padding=1)


def saturated_shares(layers: list[tuple[str, object]], method: str) -> list[float]:
    shares = []
    for _, paths in layers:
        shapes = status(paths, "compass_scores")["shapes"]
        shares.append(100 * shapes["saturated"][method] / shapes["jacobians"][0])
    return shares


def saturation_by_layer(runs: list[Run], stem: Path) -> None:
    """Show that the beta search saturates at the layers the rule rejects, not at the chosen one.

    Per layer, the share of validation, calibration and test cells whose COMPASS score reached the
    end of the beta range, from the counts compass_scores records. A saturated score is censored:
    it says the area never reached the cell's reference within the range, not how far it is.
    """
    methods = garimpo_uncertainty.intervals.COMPASS_METHODS
    looks = common.looks(list(methods))
    fig, axes = model_grid(runs, 0.62, sharey=True)
    width = 0.8 / len(methods)
    for ax, run in zip(axes, runs, strict=True):
        layers = layers_in_order(run)
        x = np.arange(len(layers))
        for k, method in enumerate(methods):
            bars = ax.bar(
                x + (k - (len(methods) - 1) / 2) * width,
                saturated_shares(layers, method),
                width,
                color=looks[method].color,
                edgecolor="black",
                linewidth=0.4,
                label=common.method_name(method),
            )
            bar_labels(ax, bars, 0)
        mark_chosen(ax, x, [layer for layer, _ in layers], run.chosen_layer)
        ax.set_ylim(0, 112)
        ax.set_xlabel("Camada (negrito: escolhida)")
    for ax in axes[::2]:
        ax.set_ylabel("Escores saturados (%)")
    label_models(axes, runs)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=len(methods))
    common.save(fig, stem)


def stacked_minutes(ax: object, layers: list[tuple[str, object]]) -> np.ndarray:
    """Stack each costly stage's minutes per layer; return the totals."""
    x = np.arange(len(layers))
    bottom = np.zeros(len(layers))
    shades = (style.PALETTE["blue"], style.MUTED)
    for (stage, label), color, hatch in zip(COSTLY_STAGES, shades, ("", "///"), strict=True):
        minutes = np.array([status(paths, stage)["seconds"] / 60 for _, paths in layers])
        ax.bar(
            x,
            minutes,
            0.6,
            bottom=bottom,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )
        bottom += minutes
    return bottom


def cost_by_layer(runs: list[Run], stem: Path) -> None:
    """Show what evaluating each layer costs: the beta search dominates, and grows with depth.

    Per layer, the minutes of compass_scores and of generate_intervals, stacked, from the seconds
    each stage records. The other stages of a layer take seconds. The cost depends on the GPU.
    """
    fig, axes = model_grid(runs, 0.5, sharey=True)
    for ax, run in zip(axes, runs, strict=True):
        layers = layers_in_order(run)
        x = np.arange(len(layers))
        totals = stacked_minutes(ax, layers)
        bars = ax.bar(x, totals, 0.6, fill=False, linewidth=0)
        bar_labels(ax, bars, 0 if totals.max() >= WHOLE_MINUTES else 1)
        mark_chosen(ax, x, [layer for layer, _ in layers], run.chosen_layer)
        ax.set_xlabel("Camada (negrito: escolhida)")
        ax.margins(y=0.15)
    for ax in axes[::2]:
        ax.set_ylabel("Tempo (min)")
    label_models(axes, runs)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="outside lower center", ncol=2)
    common.save(fig, stem)


def with_profile(runs: list[Run]) -> list[Run]:
    missing = [run.name for run in runs if not (run.paths.root / PROFILE).exists()]
    if missing:
        logger.warning("%s: run `garimpo search_profile` for %s", PROFILE, ", ".join(missing))
    return [run for run in runs if run.name not in missing]


def profiled_layers(stored: dict) -> list[str]:
    return sorted({key.split(":")[0] for key in stored if key.endswith(":seconds")})


def interval_change(stored: dict, key: str) -> np.ndarray:
    """Return how far the interval still moves after each step, in ha: (steps - 1, cells)."""
    lower, upper = stored[f"{key}:path_lower"], stored[f"{key}:path_upper"]
    return np.maximum(np.abs(lower - lower[-1]), np.abs(upper - upper[-1]))[:-1]


def pixel_line(ax: object) -> None:
    """Mark one pixel, the area's resolution: a change below it moves no interval."""
    ax.axhline(
        garimpo_uncertainty.data.PIXEL_HA, color=style.BASELINE, linestyle="--", linewidth=0.8
    )
    ax.set_yscale("symlog", linthresh=garimpo_uncertainty.data.PIXEL_HA)
    ax.grid(visible=True, axis="both")


def draw_convergence(ax: object, stored: dict, direction: str, chosen_layer: str) -> None:
    layers = profiled_layers(stored)
    shades = style.family("blue", len(layers), 0.35) if len(layers) > 1 else [style.PALETTE["blue"]]
    for layer, shade in zip(layers, shades, strict=True):
        change = interval_change(stored, f"{layer}:{direction}")
        step = np.arange(1, len(change) + 1)
        chosen = layer == chosen_layer
        ax.plot(
            step,
            np.median(change, 1),
            color=shade,
            linewidth=1.4 if chosen else 0.8,
            label=layer_name(layer) + (" (escolhida)" if chosen else ""),
        )
        ax.fill_between(step, 0, np.quantile(change, 0.9, 1), color=shade, alpha=0.15, linewidth=0)
    pixel_line(ax)
    ax.xaxis.set_major_locator(style.mpl.ticker.MaxNLocator(integer=True))


def search_convergence(runs: list[Run], stem: Path) -> None:
    """Show after how many bisection steps the COMPASS-J interval stops changing.

    Per layer and direction, the median over the profiled calibration cells of how many hectares
    the interval still moves before the last step, and the 90th percentile as a band. Below one
    pixel (dashed) a step cannot change an interval; every step after that costs time for nothing.
    """
    runs = with_profile(runs)
    if not runs:
        return
    fig, axes = rows_by_models(runs, len(DIRECTIONS), 0.32, sharex=True, sharey=True)
    for row, run in zip(axes, runs, strict=True):
        stored = common.arrays(run.paths.root / PROFILE)
        for ax, direction in zip(row, DIRECTIONS, strict=True):
            draw_convergence(ax, stored, direction, run.chosen_layer)
        row[0].set_ylabel("Mudança restante (ha)")
        row[-1].legend(
            loc="upper right",
            fontsize=6,
            title_fontsize=6,
            title=f"n = {common.number(len(stored['cell_index']), 0)} células",
        )
    for ax in axes[-1]:
        ax.set_xlabel("Passo da bissecção")
    name_rows(axes, runs, [f"COMPASS-J ({d.upper()})" for d in DIRECTIONS])
    common.save(fig, stem)


def draw_tradeoff(ax: object, stored: dict) -> None:
    """One line per layer and direction, the thresholds in order; label the knob's two ends."""
    for layer, color in zip(profiled_layers(stored), style.CYCLE, strict=False):
        for direction, marker in (("pca", "o"), ("svm", "s")):
            key = f"{layer}:{direction}"
            seconds = stored[f"{key}:seconds"]
            lower, upper = stored[f"{key}:lower"], stored[f"{key}:upper"]
            change = np.maximum(np.abs(lower - lower[0]), np.abs(upper - upper[0])).max(1)
            ax.plot(
                seconds / seconds[0],
                change,
                marker=marker,
                color=color,
                linewidth=0.8,
                markersize=3,
                label=f"{layer_name(layer)}, {direction.upper()}",
            )
    for at in (0, -1):
        ax.annotate(
            f"limiar {stored['thresholds'][at]:.0e}".replace("e-", "e\u2212"),
            (seconds[at] / seconds[0], change[at]),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=6,
        )
    pixel_line(ax)


def threshold_tradeoff(runs: list[Run], stem: Path) -> None:
    """Show what a coarser search threshold saves in time and costs in interval precision.

    One point per threshold, per layer and direction: the search time relative to the run's own
    threshold against the largest change of any profiled cell's interval, in hectares. Points join
    in threshold order, the knob's only order; below one pixel (dashed) no interval changes.
    """
    runs = with_profile(runs)
    if not runs:
        return
    fig, axes = model_grid(runs, 0.62, sharey=True)
    for ax, run in zip(axes, runs, strict=True):
        draw_tradeoff(ax, common.arrays(run.paths.root / PROFILE))
        ax.set_xlabel("Tempo relativo ao limiar mais fino")
    for ax in axes[::2]:
        ax.set_ylabel("Mudança máx. (ha)")
    label_models(axes, runs)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
    common.save(fig, stem)


FIGURES = {
    "saturation_by_layer": saturation_by_layer,
    "cost_by_layer": cost_by_layer,
    "search_convergence": search_convergence,
    "threshold_tradeoff": threshold_tradeoff,
}
