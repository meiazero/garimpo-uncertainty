"""DataLoader worker processes, which the rest of the suite leaves at 0 and Colab runs at its CPUs.

Each worker pickles the dataset, opens the memmaps itself and draws its seed from the run's
generator. Each comparison runs both sides on one core (support.one_core).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import garimpo_uncertainty.data
from tests.e2e import support

pytestmark = pytest.mark.skipif(
    support.NUM_WORKERS != "0", reason="GARIMPO_TEST_DEVICE=cuda runs every stage with workers"
)


def with_workers(args: list[str]) -> list[str]:
    position = args.index("--num-workers") + 1
    return [*args[:position], "2", *args[position + 1 :]]


def predictions(run_dir: Path) -> dict[str, np.ndarray]:
    directory = run_dir / "predictions"
    with np.load(directory / "cells.npz") as cells:
        arrays = {name: cells[name] for name in cells.files}
    roles = ("validation", "calibration", "test")
    return arrays | {role: np.load(directory / f"{role}.npy") for role in roles}


def test_training_with_workers_is_fixed_by_the_seed(
    convnet_run: support.FinishedRun, data_root: Path, tmp_path: Path
) -> None:
    """Two runs give one model, so worker seeds come from --seed and not from the clock or pid.

    The band statistics sum batches in loader order, so workers must leave them unchanged.
    """
    expected = garimpo_uncertainty.data.read_run_metadata(convnet_run.run_dir)["normalization"]
    weights = []
    for name in ("first", "second"):
        run_dir = tmp_path / name
        args = with_workers(support.train_args("convnet", run_dir, epochs="1"))
        with support.one_core():
            support.run("train.py", *args, data_root=data_root)
        metadata = garimpo_uncertainty.data.read_run_metadata(run_dir)
        assert metadata["normalization"] == expected
        weights.append(torch.load(run_dir / "last.pt", weights_only=True)["model"])
    torch.testing.assert_close(*weights, rtol=0, atol=0)


def test_inference_with_workers_matches_inference_without(
    private_convnet: support.FinishedRun, data_root: Path
) -> None:
    """Out-of-order batches would put one cell's prediction at another's position."""
    args = [*support.checkpoint_args(private_convnet.run_dir), "--force"]
    results = []
    for stage_args in (args, with_workers(args)):
        with support.one_core():
            support.run("infer.py", *stage_args, data_root=data_root)
        results.append(predictions(private_convnet.run_dir))
    without, with_two = results
    assert with_two.keys() == without.keys()
    for name, array in without.items():
        assert np.array_equal(with_two[name], array), name
