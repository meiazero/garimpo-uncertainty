"""The figures of the latent space: the shift as the models see it, and the COMPASS-J geometry.

Everything comes from scores/scores.npz, which keeps per cell the chosen layer's summed features
and Jacobian, from the density ratio calibrate.py fits on those features, and from the intervals.
"""

import json
from pathlib import Path

import compass2d.directions
import matplotlib.colors
import numpy as np

import garimpo_uncertainty.data
import garimpo_uncertainty.figures.orx_figstyle as style
import garimpo_uncertainty.intervals
from garimpo_uncertainty.figures import common
from garimpo_uncertainty.figures.common import Run
from garimpo_uncertainty.figures.diagnostics import layer_name, name_rows, rows_by_models
from garimpo_uncertainty.figures.results import label_models, model_grid

BINS = 5
WEIGHTED = f"{common.POOLED}_weighted"
Z_95 = 1.959964


def weighted_pairs() -> list[tuple[str, str]]:
    """Every method the pipeline weights, pooled and weighted: Q3's comparison under the shift."""
    combinations = garimpo_uncertainty.intervals.COMBINATIONS
    methods = [m for m, c in combinations if c == WEIGHTED]
    return [(m, c) for m in methods for c in (common.POOLED, WEIGHTED)]


def role_rows(run: Run, role: str) -> tuple[object, np.ndarray]:
    """Return a role's calibrated cells in cell order and their rows in scores.npz.

    The cells calibrate.py used, so they align with its density ratios: without the empty ones
    when the run calibrated on nonempty cells only.
    """
    cells = run.cells[run.cells.role == role]
    status = run.paths.layer_status / "calibrate.json"
    arguments = json.loads(status.read_text(encoding="utf-8"))["completed_fingerprint"]["arguments"]
    if arguments["cells"] == "nonempty":
        cells = cells[cells.v > 0]
    cells = cells.reset_index(drop=True)
    with np.load(run.paths.scores) as stored:
        rows = garimpo_uncertainty.data.positions_of(
            stored["cell_index"], cells.cell_index.to_numpy()
        )
    return cells, rows


def wilson(covered: np.ndarray) -> tuple[float, float, float]:
    """Coverage in percent with its 95% Wilson interval, the cells taken as independent."""
    n, share = len(covered), covered.mean()
    centre = (share + Z_95**2 / (2 * n)) / (1 + Z_95**2 / n)
    half = Z_95 * np.sqrt(share * (1 - share) / n + Z_95**2 / (4 * n**2)) / (1 + Z_95**2 / n)
    low, high = centre - half, centre + half
    # At a share of 0 or 1 a bound equals it, and rounding can put it a hair past.
    return 100 * share, 100 * max(share - low, 0.0), 100 * max(high - share, 0.0)


def coverage_by_latent_distance(runs: list[Run], stem: Path) -> None:
    """Show whether coverage falls on the test cells least like the calibration ones.

    The test cells in quintiles of the density ratio calibrate.py fits on the chosen layer's
    features: the fifth holds the cells the calibration represents least. One panel per weighted
    method, its coverage with pooled (solid) and weighted (dashed) calibration per quintile, with a
    95% Wilson interval that treats cells as independent, so narrower than a group bootstrap.
    """
    methods = list(dict.fromkeys(m for m, _ in weighted_pairs()))
    looks = common.looks(methods)
    fig, axes = rows_by_models(runs, len(methods), 0.3, sharex=True, sharey=True)
    for row, run in zip(axes, runs, strict=True):
        test, band, covered = quintile_coverage(run)
        for ax, method in zip(row, methods, strict=True):
            common.target_line(ax, 100 * (1 - common.MAIN_ALPHA))
            for calibration, shift in ((common.POOLED, -0.08), (WEIGHTED, 0.08)):
                points = np.array(
                    [wilson(covered[method, calibration][band == b]) for b in range(BINS)]
                )
                weighted = calibration == WEIGHTED
                ax.errorbar(
                    np.arange(BINS) + shift,
                    points[:, 0],
                    yerr=points[:, 1:].T,
                    linestyle="--" if weighted else "-",
                    linewidth=0.8,
                    markersize=3,
                    elinewidth=0.5,
                    capsize=0,
                    label=f"calibração {common.calibration_name(calibration)}",
                    **common.Look(
                        looks[method].color, looks[method].marker, weighted
                    ).marker_style(),
                )
            ax.set_xticks(range(BINS), [f"{b + 1}º" for b in range(BINS)])
        row[0].set_ylabel("PICP (%)")
        row[-1].annotate(
            f"n = {common.number(len(test) // BINS, 0)} por quintil",
            xy=(0.97, 0.04),
            xycoords="axes fraction",
            ha="right",
            fontsize=6,
        )
    fig.supxlabel(
        "Quintil da razão de densidade no teste (5º: menos representado na calibração)",
        fontsize=style.mpl.rcParams["axes.labelsize"],
    )
    name_rows(axes, runs, [common.method_name(m) for m in methods])
    handles = [
        style.plt.Line2D(
            [], [], color="black", linestyle=style_, marker="o", markerfacecolor=face, markersize=3
        )
        for style_, face in (("-", "black"), ("--", "white"))
    ]
    fig.legend(
        handles,
        [f"calibração {common.calibration_name(c)}" for c in (common.POOLED, WEIGHTED)],
        loc="outside upper center",
        ncol=2,
    )
    common.save(fig, stem)


def quintile_coverage(run: Run) -> tuple[object, np.ndarray, dict]:
    """Return the test cells, their quintile of log density ratio, and what each pair covers."""
    test, _ = role_rows(run, "test")
    ratio = np.log(common.arrays(run.paths.quantiles)["density_ratio_test"])
    # By rank, not by value: tied ratios would otherwise leave a quintile empty.
    band = np.empty(len(ratio), dtype=int)
    band[np.argsort(ratio, kind="stable")] = np.arange(len(ratio)) * BINS // len(ratio)
    lower, upper = main_level_bounds(run, test)
    v = test.v.to_numpy()
    combinations = garimpo_uncertainty.intervals.COMBINATIONS
    covered = {
        pair: (v >= lower[combinations.index(pair)]) & (v <= upper[combinations.index(pair)])
        for pair in weighted_pairs()
    }
    return test, band, covered


def main_level_bounds(run: Run, cells: object) -> tuple[np.ndarray, np.ndarray]:
    """Return each combination's bounds at the main level, (combinations, cells), in order."""
    stored = common.arrays(run.paths.intervals)
    level = int(np.flatnonzero(np.isclose(stored["alpha"], common.MAIN_ALPHA))[0])
    at = garimpo_uncertainty.data.positions_of(stored["cell_index"], cells.cell_index.to_numpy())
    return stored["lower"][level][:, at], stored["upper"][level][:, at]


def principal_plane(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardized rows on their first two principal components, and each one's variance share."""
    scaled = (values - values.mean(0)) / (values.std(0) + 1e-12)
    _, singular, vt = np.linalg.svd(scaled, full_matrices=False)
    return scaled @ vt[:2].T, singular[:2] ** 2 / (singular**2).sum()


def latent_shift_map(runs: list[Run], stem: Path) -> None:
    """Show the geographic shift in the chosen layer's features, and which cells weighting favors.

    Calibration (circles) and test (squares) cells on the first two principal components of their
    summed features, standardized per model. On the right, each calibration cell colored by the
    density ratio that weights it, on one logarithmic scale for every model; a few bright cells
    carrying most of the weight is what a low effective number of groups means.
    """
    fig, axes = rows_by_models(runs, 2, 0.42)
    maps = [latent_points(run) for run in runs]
    ratios = np.concatenate([m[2] for m in maps])
    norm = matplotlib.colors.LogNorm(ratios.min(), ratios.max())
    points = None
    for row, (calibration_xy, test_xy, ratio, share) in zip(axes, maps, strict=True):
        row[0].scatter(
            *calibration_xy.T, s=5, color=style.PALETTE["blue"], linewidth=0, label="calibração"
        )
        row[0].scatter(
            *test_xy.T, s=5, marker="s", color=style.PALETTE["orange"], linewidth=0, label="teste"
        )
        row[1].scatter(*test_xy.T, s=4, marker="s", color=style.MUTED, linewidth=0, zorder=0)
        order = np.argsort(ratio)
        points = row[1].scatter(
            *calibration_xy[order].T,
            s=6,
            c=ratio[order],
            cmap=style.SEQUENTIAL,
            norm=norm,
            linewidth=0,
        )
        for ax in row:
            ax.set_xlabel(f"1ª componente ({common.number(100 * share[0], 0)}%)")
        row[0].set_ylabel(f"2ª componente ({common.number(100 * share[1], 0)}%)")
        row[1].sharex(row[0])
        row[1].sharey(row[0])
    axes[0, 0].legend(
        loc="upper right", fontsize=6, frameon=True, framealpha=1.0, edgecolor="black"
    )
    colorbar = fig.colorbar(points, ax=axes[:, 1], location="bottom", shrink=0.9, aspect=30)
    colorbar.set_label("Razão de densidade (escala log)")
    # Ratios spanning under three decades get labels at 1, 2 and 5 of each, and no minor labels.
    colorbar.ax.xaxis.set_major_locator(style.mpl.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    colorbar.ax.xaxis.set_minor_formatter(style.mpl.ticker.NullFormatter())
    name_rows(axes, runs, ["papel de cada célula", "peso na calibração ponderada"])
    common.save(fig, stem)


def latent_points(run: Run) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    calibration, calibration_at = role_rows(run, "calibration")
    _, test_at = role_rows(run, "test")
    with np.load(run.paths.scores) as stored:
        features = stored["feature_sums"][np.concatenate([calibration_at, test_at])]
    xy, share = principal_plane(features)
    ratio = common.arrays(run.paths.quantiles)["density_ratio_calibration"]
    return xy[: len(calibration)], xy[len(calibration) :], ratio, share


def sensitivity(run: Run, layer: str, paths: object) -> tuple[np.ndarray, dict]:
    """Each cell's first-order area change per unit beta along its PCA direction, |J . delta|."""
    stored = common.arrays(paths.scores)
    fitted = compass2d.directions.Directions.load(run.paths.directions(layer))
    delta = compass2d.directions.cell_deltas("pca", stored["jacobians"], fitted)
    return np.abs((stored["jacobians"] * delta).sum(1)), stored


def draw_sensitivity(ax: object, run: Run, layer: str, paths: object) -> None:
    m = garimpo_uncertainty.intervals.COMPASS_METHODS.index("compass_j_pca")
    speed, stored = sensitivity(run, layer, paths)
    score, saturated = stored["score"][m], stored["saturated"][m]
    kept = (score > 0) & (speed > 0) & ~saturated
    ax.scatter(speed[kept], score[kept], s=3, color=style.PALETTE["blue"], alpha=0.5, linewidth=0)
    censored = saturated & (speed > 0)
    ax.scatter(
        speed[censored], score[censored], s=7, marker="^", color=style.PALETTE["red"], linewidth=0
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(style.mpl.ticker.LogLocator(numticks=4))
    ax.annotate(
        f"{layer_name(layer)}{' (escolhida)' if layer == run.chosen_layer else ''}\n"
        f"cens. {saturated.sum()}, zeros {(score == 0).sum()}",
        xy=(0.03, 0.04),
        xycoords="axes fraction",
        fontsize=6,
    )


def label_lowest_visible(axes: np.ndarray, label: str) -> None:
    """Give each column's lowest visible panel the shared x ticks and label.

    A model with fewer layers hides its last panels, which would otherwise carry them.
    """
    for column in axes.T:
        lowest = [ax for ax in column if ax.get_visible()][-1]
        lowest.xaxis.set_tick_params(labelbottom=True)
        lowest.set_xlabel(label)


def sensitivity_and_saturation(runs: list[Run], stem: Path) -> None:
    """Test whether the cells whose search saturates are those where the area barely responds.

    Per layer, each cell's COMPASS-J (PCA) score against its sensitivity |J . delta|, the area
    change per unit beta the direction promises at the start. Censored scores (triangles) sit at
    the end of the range; zero scores have no logarithm and are counted, not drawn.
    """
    columns = max(len(run.layer_runs()) for run in runs)
    fig, axes = rows_by_models(runs, columns, 0.3, sharex=True, sharey=True)
    for row, run in zip(axes, runs, strict=True):
        layers = sorted(run.layer_runs().items())
        for ax, (layer, paths) in zip(row, layers, strict=False):
            draw_sensitivity(ax, run, layer, paths)
        for ax in row[len(layers) :]:
            ax.set_visible(False)
        row[0].set_ylabel("Escore PCA (β)")
    label_lowest_visible(axes, "|J · δ| (ha/β)")
    # Shared and spanning decades: a tick every second decade, only those inside the limits.
    low, high = np.log10(axes[0, 0].get_ylim())
    axes[0, 0].set_yticks(10.0 ** np.arange(np.ceil(low), np.floor(high) + 1, 2))
    style.panel_labels(axes[np.vectorize(lambda ax: ax.get_visible())(axes)])
    for ax, run in zip(axes[:, 0], runs, strict=True):
        ax.set_ylabel(f"{run.name}\n{ax.get_ylabel()}")
    common.save(fig, stem)


def draw_directions(ax: object, run: Run, plane: np.ndarray, reach: float) -> float:
    """Draw the fitted PCA and SVM directions projected on the plane; return their |cos|."""
    fitted = compass2d.directions.Directions.load(run.paths.directions(run.chosen_layer))
    pca, svm = (v / np.linalg.norm(v) for v in (fitted.pca_components[0], fitted.svm_w))
    for vector, color, name in ((pca, "red", "PCA"), (svm, "orange", "SVM")):
        end = reach * (plane @ vector)
        arrow = {"arrowstyle": "->", "color": style.PALETTE[color], "linewidth": 1.2}
        ax.annotate("", xy=end, xytext=(0, 0), arrowprops=arrow)
        ax.annotate(
            name,
            xy=end,
            xytext=(3, 3),
            textcoords="offset points",
            color=style.PALETTE[color],
            fontsize=7,
        )
    return float(abs(pca @ svm))


def jacobian_geometry(runs: list[Run], stem: Path) -> None:
    """Show that one direction describes how the area responds in every cell of the chosen layer.

    Each cell's Jacobian of the area on the first two principal components of all Jacobians, on
    equal scales, with the fitted PCA and SVM directions projected on the same plane. Cells on one
    line mean one direction serves them all; the angle between the arrows is |cos| in the label.
    """
    fig, axes = model_grid(runs, 0.62)
    for ax, run in zip(axes, runs, strict=True):
        with np.load(run.paths.scores) as stored:
            jacobians = stored["jacobians"]
        centered = jacobians - jacobians.mean(0)
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        share = singular**2 / (singular**2).sum()
        xy = centered @ vt[:2].T
        ax.scatter(*xy.T, s=4, color=style.PALETTE["blue"], alpha=0.5, linewidth=0)
        cosine = draw_directions(ax, run, vt[:2], np.abs(xy).max() * 0.8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(f"1ª componente ({common.number(100 * share[0], 1)}%)")
        ax.set_ylabel(f"2ª componente ({common.number(100 * share[1], 1)}%)")
        ax.annotate(
            f"{layer_name(run.chosen_layer)}; |cos(PCA, SVM)| = {common.number(cosine)}",
            xy=(0.03, 0.04),
            xycoords="axes fraction",
            fontsize=6,
        )
    label_models(axes, runs)
    common.save(fig, stem)


FIGURES = {
    "coverage_by_latent_distance": coverage_by_latent_distance,
    "latent_shift_map": latent_shift_map,
    "sensitivity_and_saturation": sensitivity_and_saturation,
    "jacobian_geometry": jacobian_geometry,
}
