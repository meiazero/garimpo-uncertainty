"""`garimpo figures`: the dissertation's figures, drawn from finished runs.

    garimpo figures runs/*-f1.0 --out outputs/figures
    garimpo figures --scale auto --out outputs/figures

With --scale, the runs are those of configs/models/ at that scale whose every stage is ok, or, for
qualitative_cells alone, that have intervals; the figures go to <out>/<scale>, the scale `auto`
resolves to.

Each figure is a function of FIGURES, whose docstring says what claim it makes and which section of
the dissertation it serves; docs/figures.md lists them. Every number comes from the runs' files,
so a figure is redrawn, never edited.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import garimpo_uncertainty.artifacts
import garimpo_uncertainty.config
import garimpo_uncertainty.figures.diagnostics
import garimpo_uncertainty.figures.latent
import garimpo_uncertainty.figures.maps
import garimpo_uncertainty.figures.orx_figstyle as style
import garimpo_uncertainty.figures.results
import garimpo_uncertainty.figures.search
import garimpo_uncertainty.pipeline.merge
import garimpo_uncertainty.pipeline.status
from garimpo_uncertainty.figures.common import Run

Figure = Callable[[list[Run], Path], None]
FIGURES: dict[str, Figure] = (
    garimpo_uncertainty.figures.results.FIGURES
    | garimpo_uncertainty.figures.diagnostics.FIGURES
    | garimpo_uncertainty.figures.maps.FIGURES
    | garimpo_uncertainty.figures.latent.FIGURES
    | garimpo_uncertainty.figures.search.FIGURES
)


# Every other figure reads stages after the intervals: metrics, the cell table, the search.
INTERVALS_ONLY = {"qualitative_cells"}


def runs_at(scale: str, models: list[str], *, finished: bool) -> list[Path]:
    """Return each model's run at `scale` that is `finished` or has intervals; name the rest."""
    found = []
    for model in models:
        config = garimpo_uncertainty.pipeline.status.load(scale, model)
        if config is None:
            continue
        run = Path(config.run_dir)
        if finished:
            statuses = garimpo_uncertainty.pipeline.merge.stage_statuses(run).values()
            ready, lacks = all(status == "ok" for status in statuses), "every stage ok"
        else:
            ready = garimpo_uncertainty.artifacts.RunPaths.of(run).intervals.exists()
            lacks = "intervals"
        if ready:
            found.append(run)
        else:
            print(f"Left out {model}: {run} has no {lacks} yet.")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("runs", nargs="*", type=Path, help="finished run directories")
    parser.add_argument("--scale", help="instead of runs: a file of configs/scales/, or auto")
    parser.add_argument(
        "--models",
        nargs="+",
        default=garimpo_uncertainty.config.models(garimpo_uncertainty.pipeline.status.CONFIGS),
        help="with --scale, the models to draw; default: every file of configs/models/",
    )
    parser.add_argument("--out", type=Path, required=True, help="where the PDF and SVG files go")
    parser.add_argument(
        "--only", nargs="+", choices=list(FIGURES), help="draw these figures; default: all"
    )
    args = parser.parse_args()
    if bool(args.runs) == bool(args.scale):
        parser.error("give either run directories or --scale")
    run_dirs, out = args.runs, args.out
    if args.scale:
        scale = garimpo_uncertainty.pipeline.status.choose_scale(args.scale)
        finished = not set(args.only or FIGURES) <= INTERVALS_ONLY
        run_dirs, out = runs_at(scale, args.models, finished=finished), out / scale
        print(f"Scale {scale}; runs {', '.join(map(str, run_dirs)) or 'none'}.")
    if not run_dirs:
        print("Skipped: no chosen model has a run that holds what these figures read.")
        return
    style.use_style()
    runs = [Run.of(run_dir) for run_dir in run_dirs]
    for name in args.only or FIGURES:
        FIGURES[name](runs, out / name)
