"""`garimpo figures` draws every dissertation figure from the runs the other tests finished."""

from pathlib import Path

import numpy as np

from garimpo_uncertainty.figures.data import DATA_FIGURES
from garimpo_uncertainty.figures.make import FIGURES
from garimpo_uncertainty.stages.search_profile import PROFILE, THRESHOLDS
from tests.e2e import support


def test_every_figure_draws_from_finished_runs(
    unet_run: support.FinishedRun,
    convnet_run: support.FinishedRun,
    data_root: Path,
    tmp_path: Path,
) -> None:
    """Every file a figure reads comes from the stages, the search profile included."""
    for run in (unet_run, convnet_run):
        with np.load(run.run_dir / PROFILE) as profile:
            assert profile["thresholds"].tolist() == list(THRESHOLDS)
            assert len(profile["cell_index"]) > 0

    out = tmp_path / "figures"
    support.run(
        "figures",
        str(unet_run.run_dir),
        str(convnet_run.run_dir),
        "--out",
        str(out),
        data_root=data_root,
    )
    assert {path.stem for path in out.glob("*.pdf")} == set(FIGURES)
    assert {path.stem for path in out.glob("*.svg")} == set(FIGURES)


def test_data_figures_draw_from_the_store_alone(data_root: Path, tmp_path: Path) -> None:
    """The data's figures need no run: the store and the split are enough."""
    out = tmp_path / "data"
    support.run(
        "data_figures", "--out", str(out), "--split", str(support.SPLIT_PATH), data_root=data_root
    )
    assert {path.stem for path in out.glob("*.pdf")} == set(DATA_FIGURES)
