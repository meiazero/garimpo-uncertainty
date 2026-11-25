"""`garimpo figures`: the dissertation's figures, drawn from finished runs.

    garimpo figures runs/full/*-f1.0 --out outputs/figures

Each figure is a function of FIGURES, whose docstring says what claim it makes and which section of
the dissertation it serves; docs/figures.md lists them. Every number comes from the runs' files,
so a figure is redrawn, never edited.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import garimpo_uncertainty.figures.diagnostics
import garimpo_uncertainty.figures.latent
import garimpo_uncertainty.figures.maps
import garimpo_uncertainty.figures.orx_figstyle as style
import garimpo_uncertainty.figures.results
import garimpo_uncertainty.figures.search
from garimpo_uncertainty.figures.common import Run

Figure = Callable[[list[Run], Path], None]
FIGURES: dict[str, Figure] = (
    garimpo_uncertainty.figures.results.FIGURES
    | garimpo_uncertainty.figures.diagnostics.FIGURES
    | garimpo_uncertainty.figures.maps.FIGURES
    | garimpo_uncertainty.figures.latent.FIGURES
    | garimpo_uncertainty.figures.search.FIGURES
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "runs", nargs="+", type=Path, help="finished run directories, one per model"
    )
    parser.add_argument("--out", type=Path, required=True, help="where the PDF and SVG files go")
    parser.add_argument(
        "--only", nargs="+", choices=list(FIGURES), help="draw these figures; default: all"
    )
    args = parser.parse_args()
    style.use_style()
    runs = [Run.of(run_dir) for run_dir in args.runs]
    for name in args.only or FIGURES:
        FIGURES[name](runs, args.out / name)
