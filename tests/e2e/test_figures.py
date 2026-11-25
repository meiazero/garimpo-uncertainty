"""`garimpo figures` draws every dissertation figure from the runs the other tests finished."""

from pathlib import Path

from garimpo_uncertainty.figures.make import FIGURES
from tests.e2e import support


def test_every_figure_draws_from_finished_runs(
    unet_run: support.FinishedRun, convnet_run: support.FinishedRun, data_root: Path, tmp_path: Path
) -> None:
    support.run(
        "figures",
        str(unet_run.run_dir),
        str(convnet_run.run_dir),
        "--out",
        str(tmp_path),
        data_root=data_root,
    )
    assert {path.stem for path in tmp_path.glob("*.pdf")} == set(FIGURES)
    assert {path.stem for path in tmp_path.glob("*.svg")} == set(FIGURES)
