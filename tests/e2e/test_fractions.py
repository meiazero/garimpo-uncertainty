"""Two training fractions on one data root, the larger's data arriving after the smaller ran."""

import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import garimpo_uncertainty.cell_dataset
import garimpo_uncertainty.data
from tests.e2e import support

FRACTION_ARTIFACTS = support.ROOT / "tests" / "artifacts" / "fractions"
ROOT = FRACTION_ARTIFACTS / "data"
RUNS = FRACTION_ARTIFACTS / "runs"
SPLIT_PATH = FRACTION_ARTIFACTS / "split.csv"
SUMMARY = FRACTION_ARTIFACTS / "split_summary.json"
SMALL_DIR = RUNS / "convnet-f0.01"
FULL_DIR = RUNS / "convnet-f1.0"
SMALL_STATUS = SMALL_DIR / "status" / "train.json"
SMALL_REPORT = FRACTION_ARTIFACTS / "validation_small.json"


def small_args(fraction: str = "0.01", subset_seed: str = "0") -> list[str]:
    """One tile is 4 cells, so each epoch's 10 updates at batch 2 pass over them 5 times.

    With few updates no logit nears zero, and at the upstream temperature every Jacobian is then
    exactly zero, which fit_directions rightly refuses.
    """
    return support.train_args(
        "convnet",
        SMALL_DIR,
        batch_size="2",
        lr="1e-2",
        epochs="8",
        split_path=SPLIT_PATH,
        fraction=fraction,
        subset_seed=subset_seed,
    )


def finish_run(run_dir: Path) -> None:
    _, layer, other_layer = support.RUN_ORDER["convnet"]
    support.run_stages_after_training(run_dir, layer, [layer, other_layer], ROOT)
    support.compute_metrics(run_dir, support.BOUNDARY_TOLERANCE, ROOT)


def slots_of(cell_index: np.ndarray) -> dict[str, bytes]:
    store = garimpo_uncertainty.data.open_store(ROOT)
    return {name: getattr(store, name)[cell_index].tobytes() for name in store._fields}


@dataclass(frozen=True)
class GrownRoot:
    """What the small run left behind, measured before the larger fraction's data arrived."""

    split: pd.DataFrame
    small: pd.DataFrame
    full: pd.DataFrame
    small_index: np.ndarray
    slots: dict[str, bytes]
    sizes: dict[str, int]
    identity: dict[str, str | None]
    finished: dict[str, str]


@pytest.fixture(scope="module")
def grown_root() -> GrownRoot:
    FRACTION_ARTIFACTS.mkdir(parents=True)
    split = pd.DataFrame(support.group_tiles())
    split.to_csv(SPLIT_PATH, index=False)
    counts = split.groupby(["scale", "role"]).size()
    SUMMARY.write_text(
        json.dumps({"tiles_by_scale_role": {f"{s}/{r}": int(n) for (s, r), n in counts.items()}})
    )
    small = garimpo_uncertainty.data.select_tiles(split, 0.01, 0)
    full = garimpo_uncertainty.data.select_tiles(split, 1.0, 0)
    rng = np.random.default_rng(0)
    support.fill_tiles(ROOT, split, small.tile_id.tolist(), rng)

    support.validate(ROOT, SPLIT_PATH, SMALL_REPORT, fraction="0.01", summary=SUMMARY)
    support.run("train.py", *small_args(), data_root=ROOT)
    finish_run(SMALL_DIR)

    small_ids = small.tile_id.tolist()
    small_index = garimpo_uncertainty.data.cell_table(ROOT, split, small_ids).cell_index.to_numpy()
    grown = GrownRoot(
        split,
        small,
        full,
        small_index,
        slots_of(small_index),
        {path.name: path.stat().st_size for path in ROOT.glob("*.npy")},
        garimpo_uncertainty.data.tiles_identity(ROOT, split, small_ids),
        support.file_hashes(SMALL_DIR),
    )
    support.fill_tiles(ROOT, split, full.tile_id[~full.tile_id.isin(small.tile_id)].tolist(), rng)
    return grown


@pytest.fixture
def scratch_small_run(grown_root: GrownRoot) -> Iterator[GrownRoot]:
    with support.restored_after(SMALL_DIR), support.restored_after(ROOT):
        yield grown_root


def test_larger_fraction_leaves_earlier_slots_byte_identical(grown_root: GrownRoot) -> None:
    small_ids = grown_root.small.tile_id.tolist()
    assert len(grown_root.small_index) == len(grown_root.small) * support.CELLS_PER_TILE
    assert slots_of(grown_root.small_index) == grown_root.slots
    assert {path.name: path.stat().st_size for path in ROOT.glob("*.npy")} == grown_root.sizes
    assert garimpo_uncertainty.data.tiles_identity(ROOT, grown_root.split, small_ids) == (
        grown_root.identity
    )
    grown = garimpo_uncertainty.data.cell_table(
        ROOT, grown_root.split, grown_root.full.tile_id.tolist()
    )
    assert len(grown) > len(grown_root.small_index)


def test_finished_run_cannot_care_that_more_tiles_arrived(grown_root: GrownRoot) -> None:
    assert "skipping" in support.run("train.py", *small_args(), data_root=ROOT).stdout
    assert json.loads(SMALL_STATUS.read_text())["status"] == "ok"
    # A refusal would raise here, and a redone stage would change a file.
    finish_run(SMALL_DIR)
    assert support.file_hashes(SMALL_DIR) == grown_root.finished
    rerun = support.validate(ROOT, SPLIT_PATH, SMALL_REPORT, fraction="0.01", summary=SUMMARY)
    assert "skipping" in rerun.stdout
    # The slot hashes cover this run's tiles only, so the others arriving must keep it passing.
    report_args = ["--run-dir", str(SMALL_DIR), "--split", str(SPLIT_PATH)]
    report_args += ["--train-fraction", "0.01", "--subset-seed", "0"]
    assert (
        "PIPELINE CHECK: PASS"
        in support.run("pipeline_check.py", *report_args, data_root=ROOT).stdout
    )


@pytest.mark.parametrize(
    "changed", [{"fraction": "1.0"}, {"subset_seed": "1"}], ids=["fraction", "subset_seed"]
)
def test_another_selection_refuses_under_the_same_run_dir(
    scratch_small_run: GrownRoot, changed: dict[str, str]
) -> None:
    refused = support.run("train.py", *small_args(**changed), data_root=ROOT, check=False)
    assert refused.returncode != 0
    assert json.loads(SMALL_STATUS.read_text())["status"] == "refused"
    assert "skipping" in support.run("train.py", *small_args(), data_root=ROOT).stdout


def test_interrupted_run_refuses_another_fraction(scratch_small_run: GrownRoot) -> None:
    """An interrupted run has no finished status, so last.pt itself must refuse."""
    SMALL_STATUS.unlink()
    resumed = support.run("train.py", *small_args(fraction="1.0"), data_root=ROOT, check=False)
    assert resumed.returncode != 0
    assert "['train_fraction']" in json.loads(SMALL_STATUS.read_text())["error"]


def test_recut_of_a_tile_of_the_run_refuses(scratch_small_run: GrownRoot) -> None:
    """The identity follows the run's own rows."""
    small = scratch_small_run.small
    tile = int(small[small.role == "train"].tile_id.iloc[0])
    sidecar = ROOT / "tiles" / f"{tile}.json"
    original = sidecar.read_text()
    sidecar.write_text(json.dumps(json.loads(original) | {"resampling_20m": "nearest"}))
    assert support.run("train.py", *small_args(), data_root=ROOT, check=False).returncode != 0
    assert json.loads(SMALL_STATUS.read_text())["status"] == "refused"
    sidecar.write_text(original)
    assert "skipping" in support.run("train.py", *small_args(), data_root=ROOT).stdout


def without_tile(tile_id: int, name: str) -> Path:
    """Copy the data root with a tile's sidecar gone, as if never fetched."""
    damaged = FRACTION_ARTIFACTS / name
    shutil.copytree(ROOT, damaged)
    (damaged / "tiles" / f"{tile_id}.json").unlink()
    return damaged


@pytest.fixture(scope="module")
def no_outside(grown_root: GrownRoot) -> Path:
    train = grown_root.split[
        (grown_root.split.role == "train") & (grown_root.split.scale == "Artisanal")
    ].tile_id
    outside = int(train[~train.isin(grown_root.small.tile_id)].iloc[0])
    return without_tile(outside, "data_no_outside")


def test_tile_outside_the_selection_is_no_concern_of_this_fraction(no_outside: Path) -> None:
    report = FRACTION_ARTIFACTS / "validation_no_outside.json"
    support.validate(no_outside, SPLIT_PATH, report, fraction="0.01", summary=SUMMARY)
    assert json.loads(report.read_text())["problems"] == []


def test_tile_outside_the_selection_concerns_the_full_fraction(no_outside: Path) -> None:
    report = FRACTION_ARTIFACTS / "validation_no_outside_full.json"
    assert (
        support.validate(no_outside, SPLIT_PATH, report, False, "1.0", summary=SUMMARY).returncode
        != 0
    )
    assert "not_downloaded" in " ".join(json.loads(report.read_text())["problems"])


def test_tile_inside_the_selection_must_be_there(grown_root: GrownRoot) -> None:
    train = grown_root.split[
        (grown_root.split.role == "train") & (grown_root.split.scale == "Artisanal")
    ].tile_id
    inside = int(train[train.isin(grown_root.small.tile_id)].iloc[0])
    no_inside = without_tile(inside, "data_no_inside")
    report = FRACTION_ARTIFACTS / "validation_no_inside.json"
    assert (
        support.validate(no_inside, SPLIT_PATH, report, False, "0.01", summary=SUMMARY).returncode
        != 0
    )
    problems = " ".join(json.loads(report.read_text())["problems"])
    for check in ("not_downloaded", "split_counts"):
        assert check in problems, problems


@pytest.fixture(scope="module")
def full_run(grown_root: GrownRoot) -> Path:
    report = FRACTION_ARTIFACTS / "validation_full.json"
    support.validate(ROOT, SPLIT_PATH, report, summary=SUMMARY)
    full_args = support.train_args("convnet", FULL_DIR, split_path=SPLIT_PATH, fraction="1.0")
    support.run("train.py", *full_args, data_root=ROOT)
    finish_run(FULL_DIR)
    return FULL_DIR


def test_every_fraction_is_evaluated_on_the_same_cells(full_run: Path) -> None:
    """Only the training cells differ."""
    small_cells = garimpo_uncertainty.data.read_predictions(ROOT, SMALL_DIR)
    full_cells = garimpo_uncertainty.data.read_predictions(ROOT, full_run)
    small_eval = small_cells[small_cells.role != "train"].cell_index.tolist()
    assert small_eval == full_cells[full_cells.role != "train"].cell_index.tolist()
    small_train = set(small_cells[small_cells.role == "train"].cell_index)
    assert small_train < set(full_cells[full_cells.role == "train"].cell_index)
    assert len(small_train) == support.CELLS_PER_TILE
    diagnostics = pd.read_csv(SMALL_DIR / "directions" / "diagnostics.csv")
    assert (diagnostics.n_train_cells == support.CELLS_PER_TILE).all()


def test_every_fraction_makes_the_same_updates_per_epoch(full_run: Path) -> None:
    """A hundredth of the cells still trains as long between validations."""
    small = pd.read_csv(SMALL_DIR / "history.csv")
    full = pd.read_csv(full_run / "history.csv")
    assert (small.updates == int(support.STEPS_PER_EPOCH)).all()
    assert (full.updates == int(support.STEPS_PER_EPOCH)).all()


def test_normalization_comes_from_the_run_s_own_training_tiles(full_run: Path) -> None:
    small_cells = garimpo_uncertainty.data.read_predictions(ROOT, SMALL_DIR)
    small_meta = garimpo_uncertainty.data.read_run_metadata(SMALL_DIR)
    full_meta = garimpo_uncertainty.data.read_run_metadata(full_run)
    own = garimpo_uncertainty.cell_dataset.band_statistics(
        ROOT, small_cells[small_cells.role == "train"], 8, 0
    )
    assert np.allclose(small_meta["normalization"]["mean"], own["mean"])
    assert np.allclose(small_meta["normalization"]["std"], own["std"])
    assert small_meta["normalization"] != full_meta["normalization"]
    assert small_meta["split_hash"] != full_meta["split_hash"]
    assert small_meta["train_fraction"] == 0.01
    assert full_meta["train_fraction"] == 1.0


@pytest.fixture(scope="module")
def merged_fractions(full_run: Path) -> Path:
    merged = FRACTION_ARTIFACTS / "merged"
    support.run(
        "merge_results.py",
        "--runs",
        str(SMALL_DIR),
        str(full_run),
        "--out",
        str(merged),
        data_root=ROOT,
    )
    return merged


def test_merge_keeps_the_fractions_apart(merged_fractions: Path) -> None:
    results = pd.read_csv(merged_fractions / "results.csv")
    assert results.groupby(["model", "train_fraction"]).size().to_dict() == {
        ("convnet", 0.01): len(support.ALPHAS) * len(support.EXPECTED_ROWS),
        ("convnet", 1.0): len(support.ALPHAS) * len(support.EXPECTED_ROWS),
    }


@pytest.mark.parametrize("table", ["width_comparisons.csv", "coverage_comparisons.csv"])
def test_merge_never_compares_across_fractions(merged_fractions: Path, table: str) -> None:
    pairs = len(support.ALPHAS) * len(list(combinations(support.EXPECTED_ROWS, 2)))
    sizes = pd.read_csv(merged_fractions / table).groupby("train_fraction").size()
    assert sizes.to_dict() == {0.01: pairs, 1.0: pairs}, table


@pytest.mark.parametrize(
    "table",
    [
        "q1_spearman.csv",
        "layer_diagnostics.csv",
        "monotonicity_endpoint.csv",
        "coverage_by_stratum.csv",
        "score_relation.csv",
    ],
)
def test_merged_tables_cover_both_fractions(merged_fractions: Path, table: str) -> None:
    assert set(pd.read_csv(merged_fractions / table).train_fraction) == {0.01, 1.0}, table


def test_merge_refuses_two_runs_of_one_model_at_one_fraction(grown_root: GrownRoot) -> None:
    """They would merge into a single row."""
    twin = RUNS / "convnet-f0.01-twin"
    twin.mkdir()
    shutil.copy(SMALL_DIR / "run_meta.json", twin / "run_meta.json")
    duplicate = support.run(
        "merge_results.py",
        "--runs",
        str(SMALL_DIR),
        str(twin),
        "--out",
        str(FRACTION_ARTIFACTS / "merged_refused"),
        data_root=ROOT,
        check=False,
    )
    assert duplicate.returncode != 0
    assert "same model and train_fraction" in duplicate.stderr
