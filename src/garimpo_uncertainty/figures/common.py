"""What every figure reads from a finished run, and how it names and draws each method.

Labels are in Portuguese, the dissertation's language. A method with no entry in METHOD_NAMES keeps
its own name and takes its family's color from its prefix, so a new COMPASS variant appears in
every figure without a change here.
"""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import matplotlib.ticker
import numpy as np
import pandas as pd

import garimpo_uncertainty.artifacts
import garimpo_uncertainty.data
import garimpo_uncertainty.figures.orx_figstyle as style

# UFCdoc.cls: A4, 3 cm and 2 cm side margins, so \textwidth is 16 cm. The style's audit checks a
# figure's width against its TEXT, which is a paper's; this document's replaces it.
TEXT_WIDTH = 16 / 2.54
style.TEXT = TEXT_WIDTH
# The level the text reports first (5-experimentos.tex); the others go to their own figure.
MAIN_ALPHA = 0.1
POOLED = "external"

MODEL_NAMES = {
    "unet_scratch": "U-Net aleatória",
    "unet_imagenet": "U-Net ImageNet",
    "unet_sentinel2": "U-Net MoCo",
    "convnet": "DeepLabV3",
}
METHOD_NAMES = {
    "independent_pixels": "Pixels independentes",
    "conformal_simple": "Conformal simples",
    "conformal_relative": "Conformal relativo",
    "conformal_local": "Conformal local",
    "cqr": "CQR",
    "compass_l": "COMPASS-L",
    "compass_j_pca": "COMPASS-J (PCA)",
    "compass_j_svm": "COMPASS-J (SVM)",
}
CALIBRATION_NAMES = {
    "none": "sem calibração",
    "own_region": "própria região",
    "external": "externa",
    "external_weighted": "externa ponderada",
    "own_region_mondrian": "própria região, Mondrian",
    "external_mondrian": "externa, Mondrian",
    "own_region_subsampled": "própria região, subamostragem",
    "external_subsampled": "externa, subamostragem",
}
ASYMMETRIC = "a_"
# Hue names the family and shade the member, so the output-space baselines, CQR and each COMPASS
# family separate before the legend is read; markers repeat the family for greyscale print.
FAMILIES = (
    ("independent_pixels", "black", "x"),
    ("conformal_", "blue", "o"),
    ("cqr", "orange", "s"),
    ("compass_l", "green", "^"),
    ("compass_j_", "red", "D"),
    ("compass", "purple", "v"),
)


def model_name(model: str) -> str:
    return MODEL_NAMES.get(model, model)


def method_name(method: str) -> str:
    base = method.removeprefix(ASYMMETRIC)
    name = METHOD_NAMES.get(base, base)
    return f"{name} assimétrico" if base != method else name


def calibration_name(calibration: str) -> str:
    return CALIBRATION_NAMES.get(calibration, calibration)


def family_of(method: str) -> tuple[str, str, str]:
    """Return the family prefix, palette color and marker of a method."""
    base = method.removeprefix(ASYMMETRIC)
    return next((family for family in FAMILIES if base.startswith(family[0])), ("", "cyan", "P"))


@dataclass(frozen=True)
class Look:
    color: str
    marker: str
    hollow: bool

    def marker_style(self) -> dict:
        """Asymmetric rows are hollow: same family, other interval shape."""
        face = "white" if self.hollow else self.color
        return {"marker": self.marker, "color": self.color, "markerfacecolor": face}


def looks(methods: list[str]) -> dict[str, Look]:
    """Give each method its family's hue, a shade per member, and hollow markers if asymmetric."""
    members: dict[str, list[str]] = {}
    for method in methods:
        members.setdefault(family_of(method)[0], [])
        base = method.removeprefix(ASYMMETRIC)
        if base not in members[family_of(method)[0]]:
            members[family_of(method)[0]].append(base)
    result = {}
    for method in methods:
        prefix, color, marker = family_of(method)
        bases = members[prefix]
        shades = (
            [style.PALETTE[color]] if len(bases) == 1 else style.family(color, len(bases), 0.45)
        )
        shade = shades[bases.index(method.removeprefix(ASYMMETRIC))]
        result[method] = Look(shade, marker, method.startswith(ASYMMETRIC))
    return result


def ordered_methods(table: pd.DataFrame) -> list[str]:
    """Methods in the pipeline's own order, which is the order of COMBINATIONS in results.csv."""
    return list(dict.fromkeys(table.method))


@dataclass(frozen=True)
class Run:
    """One finished run: its directory, its model, and its tables, read when first used."""

    paths: garimpo_uncertainty.artifacts.RunPaths

    @classmethod
    def of(cls, run_dir: Path) -> "Run":
        return cls(garimpo_uncertainty.artifacts.RunPaths.of(run_dir))

    @cached_property
    def metadata(self) -> dict:
        return json.loads(self.paths.run_meta.read_text(encoding="utf-8"))

    @property
    def model(self) -> str:
        return self.metadata["variant"]

    @property
    def name(self) -> str:
        return model_name(self.model)

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.paths.metric_table(name))

    @cached_property
    def results(self) -> pd.DataFrame:
        return self.table("results")

    @cached_property
    def cells(self) -> pd.DataFrame:
        """The run's cells with infer's areas; reads the cell table of GARIMPO_DATA_DIR."""
        return garimpo_uncertainty.data.read_predictions(
            garimpo_uncertainty.data.data_root(), self.paths.root
        )

    @cached_property
    def chosen_layer(self) -> str:
        return garimpo_uncertainty.artifacts.read_chosen_layer(self.paths.chosen_layer)["layer"]

    def layer_runs(self) -> dict[str, garimpo_uncertainty.artifacts.RunPaths]:
        """Every layer with results: the chosen one at the run's root, the others in layers/."""
        layers = {self.chosen_layer: self.paths}
        for directory in sorted(
            (self.paths.root / garimpo_uncertainty.artifacts.LAYERS_DIR).glob("*")
        ):
            other = self.paths.other_layer(directory.name)
            # layers/<chosen> can only hold results of an attempt that chose another layer.
            if directory.name != self.chosen_layer and other.results.exists():
                layers[directory.name] = other
        return layers


def main_rows(results: pd.DataFrame, alpha: float = MAIN_ALPHA) -> pd.DataFrame:
    """Return the pooled external rows at one level, with the uncalibrated baseline."""
    pooled = results.calibration.isin(("none", POOLED))
    return results[pooled & np.isclose(results.alpha, alpha)]


def target_line(ax: object, coverage: float, *, vertical: bool = False) -> None:
    """Draw the nominal coverage as a grey reference: it is a target, not a compared series."""
    draw = ax.axvline if vertical else ax.axhline
    draw(coverage, color=style.BASELINE, linewidth=0.8, linestyle="--", zorder=1)


def number(value: float, digits: int = 2) -> str:
    """Write a number as the dissertation does: dot between thousands, decimal comma, true minus."""
    english = f"{value:,.{digits}f}"
    return english.translate(str.maketrans({",": ".", ".": ",", "-": "\u2212"}))


class CommaFormatter(matplotlib.ticker.ScalarFormatter):
    """Matplotlib's linear tick labels with a decimal comma and dots between thousands."""

    def __call__(self, x: float, pos: int | None = None) -> str:
        label = super().__call__(x, pos)
        # Rewrite the label, not x: on an axis with an offset the label is x minus that offset.
        plain = label.replace("\u2212", "-")
        try:
            value = float(plain)
        except ValueError:
            return label.replace(".", ",")
        return number(value, len(plain.partition(".")[2]))

    def get_offset(self) -> str:
        return super().get_offset().replace(".", ",")


def decimal_comma(fig: object) -> None:
    """Give every linear axis decimal commas; a log axis writes powers of ten, which have none."""
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if type(axis.get_major_formatter()) is matplotlib.ticker.ScalarFormatter:
                axis.set_major_formatter(CommaFormatter())


def save(fig: object, stem: Path) -> None:
    decimal_comma(fig)
    style.save(fig, str(stem))


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files}


def stage_shapes(status_dir: Path, stage: str) -> dict:
    """Return what a stage recorded about its outputs in its status file."""
    return json.loads((status_dir / f"{stage}.json").read_text(encoding="utf-8"))["shapes"]
