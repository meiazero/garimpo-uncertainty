"""Rerunning, refusing, merging and reporting the stages after training."""

import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import compass2d.directions
import compass2d.metrics
import numpy as np
import pandas as pd
import pytest
import torch

import garimpo_uncertainty.data
import garimpo_uncertainty.intervals
import garimpo_uncertainty.stages
from tests.e2e import support


def test_stage_peak_survives_per_section_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restarting the GPU count per section must not shrink the peak the stage reports."""
    gib = 1024**3
    counter = {"peak": 3 * gib}

    def restart() -> None:
        counter["peak"] = gib

    cuda = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        max_memory_allocated=lambda: counter["peak"],
        reset_peak_memory_stats=restart,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(garimpo_uncertainty.stages.GPU_PEAK, "highest_bytes", 0)
    sections = []

    def work() -> dict:
        sections.append(garimpo_uncertainty.stages.take_gpu_peak_bytes())
        sections.append(garimpo_uncertainty.stages.take_gpu_peak_bytes())
        return {}

    fingerprint = {"arguments": {}, "inputs": {}}
    garimpo_uncertainty.stages.run_stage(tmp_path, "probe", [], fingerprint, work, force=True)
    status = json.loads((tmp_path / "probe.json").read_text(encoding="utf-8"))
    assert sections == [3 * gib, gib]
    assert status["peak_gpu_gb"] == 3.0


def test_a_cpu_run_on_a_gpu_host_records_no_gpu_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GPU the run never touched must read as absent, not as a peak of zero bytes."""
    cuda = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    assert garimpo_uncertainty.stages.take_gpu_peak_bytes() is None


def test_metrics_stage_requires_a_tolerance(
    finished_run: support.FinishedRun, data_root: Path
) -> None:
    """An undefined boundary tolerance must stop the stage, not be chosen silently."""
    missing = support.run(
        "compute_metrics.py",
        "--run-dir",
        str(finished_run.run_dir),
        data_root=data_root,
        check=False,
    )
    assert missing.returncode == 2


def test_finished_stage_skips_with_another_batch_size(
    unet_run: support.FinishedRun, data_root: Path
) -> None:
    """As a rerun after an OOM does."""
    predictions = unet_run.run_dir / "predictions" / "test.npy"
    before = predictions.stat().st_mtime_ns
    rerun = support.run(
        "infer.py", *support.checkpoint_args(unet_run.run_dir, batch_size="4"), data_root=data_root
    )
    assert "skipping" in rerun.stdout
    assert predictions.stat().st_mtime_ns == before


class FirstTolerance(NamedTuple):
    run_dir: Path
    metrics: dict[str, np.ndarray]


FIRST_TOLERANCE, SECOND_TOLERANCE = 1.0, 5.0


@pytest.fixture
def first_tolerance(private_convnet: support.FinishedRun, data_root: Path) -> FirstTolerance:
    run_dir = private_convnet.run_dir
    support.compute_metrics(run_dir, FIRST_TOLERANCE, data_root, force=True)
    return FirstTolerance(run_dir, support.read_cell_metrics(run_dir))


def metric_files_only(names: set[str]) -> set[str]:
    return {
        name
        for name in names
        if not name.startswith("metrics/") and name != "status/compute_metrics.json"
    }


def test_another_tolerance_refuses_without_force(
    first_tolerance: FirstTolerance, data_root: Path
) -> None:
    """It must not silently keep the old table, and a refusal must never read as a passing run."""
    run_dir, first = first_tolerance
    before = support.file_hashes(run_dir)
    refused = support.compute_metrics(run_dir, SECOND_TOLERANCE, data_root, check=False)
    assert refused.returncode != 0
    assert not metric_files_only(
        {name for name, value in support.file_hashes(run_dir).items() if before.get(name) != value}
    )
    kept = support.read_cell_metrics(run_dir)
    assert np.array_equal(kept["boundary_f1"], first["boundary_f1"], equal_nan=True)
    report = support.pipeline_check(run_dir, data_root, check=False)
    assert report.returncode == 1
    assert "failing stage: compute_metrics (refused)" in report.stdout
    assert "PIPELINE CHECK: PASS" not in report.stdout


def test_another_tolerance_reruns_only_the_metrics_stage(
    first_tolerance: FirstTolerance, data_root: Path
) -> None:
    run_dir = first_tolerance.run_dir
    before = support.file_hashes(run_dir)
    support.compute_metrics(run_dir, SECOND_TOLERANCE, data_root, force=True)
    after = support.file_hashes(run_dir)
    assert after["best_model.pt"] == before["best_model.pt"]
    assert after["last.pt"] == before["last.pt"]
    assert set(after) == set(before)
    changed = {name for name in after if after[name] != before[name]}
    assert not metric_files_only(changed), metric_files_only(changed)
    assert "metrics/results.csv" in changed
    results = pd.read_csv(run_dir / "metrics" / "results.csv")
    assert (results.boundary_tolerance_px == SECOND_TOLERANCE).all()


def test_wider_tolerance_can_only_match_more_boundary(
    first_tolerance: FirstTolerance, data_root: Path
) -> None:
    """Dice must not move at all."""
    run_dir, first = first_tolerance
    support.compute_metrics(run_dir, SECOND_TOLERANCE, data_root, force=True)
    second = support.read_cell_metrics(run_dir)
    defined = ~np.isnan(first["boundary_f1"])
    assert np.array_equal(np.isnan(second["boundary_f1"]), ~defined)
    assert (second["boundary_f1"][defined] >= first["boundary_f1"][defined] - 1e-12).all()
    assert (second["boundary_f1"][defined] > first["boundary_f1"][defined]).any()
    assert np.array_equal(first["dice"], second["dice"], equal_nan=True)


@pytest.fixture(scope="session")
def saturated_run(convnet_run: support.FinishedRun, data_root: Path) -> Path:
    """Build a run whose beta_range is too small to cover anything."""
    saturated = support.RUNS / "convnet_saturated"

    def build() -> None:
        shutil.copytree(convnet_run.run_dir, saturated)
        shutil.rmtree(saturated / "scores")
        shutil.rmtree(saturated / "calibration")
        shutil.rmtree(saturated / "intervals")
        shutil.rmtree(saturated / "metrics")
        for stage in ("compass_scores", "calibrate", "generate_intervals", "compute_metrics"):
            (saturated / "status" / f"{stage}.json").unlink()
        support.run_scores_and_intervals(
            saturated, convnet_run.layer, data_root, beta_range="1e-6", threshold="1e-7"
        )
        support.compute_metrics(saturated, support.BOUNDARY_TOLERANCE, data_root)

    support.build_once("saturated_run", build)
    return saturated


def test_saturation_is_reported_not_hidden(saturated_run: Path, data_root: Path) -> None:
    table = pd.read_csv(saturated_run / "metrics" / "results.csv")
    j_rows = table[table.method.str.startswith("compass_j")]
    assert (j_rows.n_saturated_calibration > 0).all()
    relation = pd.read_csv(saturated_run / "metrics" / "score_relation.csv").set_index("method")
    cells = garimpo_uncertainty.data.read_predictions(data_root, saturated_run)
    calibration = cells[cells.role == "calibration"]
    with np.load(saturated_run / "scores" / "scores.npz") as stored:
        at = garimpo_uncertainty.data.positions_of(
            stored["cell_index"], calibration.cell_index.to_numpy()
        )
        score, saturated = stored["score"][:, at], stored["saturated"][:, at]
    split = np.abs(calibration.v - calibration.vhat).to_numpy()
    for m, method in enumerate(garimpo_uncertainty.intervals.COMPASS_METHODS):
        kept = (split > 0) & (score[m] > 0) & ~saturated[m]
        assert relation.n_cells[method] == kept.sum(), method
        assert relation.n_saturated[method] == saturated[m].sum(), method
    assert relation.n_saturated.sum() > 0
    assert relation.slope[relation.n_cells < 2].isna().all()


def test_saturated_quantiles_give_the_whole_cell(saturated_run: Path, data_root: Path) -> None:
    """Give the whole cell wherever a COMPASS quantile is censored.

    Censored means at beta_range, or at beta_range / ratio on an asymmetric row's upper side.
    """
    cells = garimpo_uncertainty.data.read_predictions(data_root, saturated_run)
    with np.load(saturated_run / "intervals" / "intervals.npz") as stored:
        cell_index, lower, upper = stored["cell_index"], stored["lower"], stored["upper"]
        quantile = stored["quantile"]
    with np.load(saturated_run / "calibration" / "quantiles.npz") as stored:
        side_ratio = stored["side_ratio"]
    with np.load(saturated_run / "scores" / "scores.npz") as stored:
        beta_range = dict(
            zip(garimpo_uncertainty.intervals.COMPASS_METHODS, stored["beta_range"], strict=True)
        )
    cell_ha = cells.set_index("cell_index").cell_ha.loc[cell_index].to_numpy()
    checked = 0
    for k, (method, _) in enumerate(garimpo_uncertainty.intervals.COMBINATIONS):
        base = method.removeprefix("a_")
        if base not in beta_range:
            continue
        censor = np.full(len(quantile), beta_range[base])
        if base != method:
            censor = np.minimum(censor, beta_range[base] / side_ratio[:, k])
        trivial = quantile[:, k] >= censor[:, None]
        checked += int(trivial.sum())
        assert (lower[:, k][trivial] == 0).all(), method
        whole = np.broadcast_to(cell_ha, trivial.shape)[trivial]
        assert np.array_equal(upper[:, k][trivial], whole), method
    assert checked > 0


def test_missing_input_is_recorded(saturated_run: Path, data_root: Path, tmp_path: Path) -> None:
    run_dir = Path(shutil.copytree(saturated_run, tmp_path / "saturated"))
    result = support.run(
        "compass_scores.py",
        *support.checkpoint_args(run_dir),
        "--layer",
        "absent.layer",
        data_root=data_root,
        check=False,
    )
    assert result.returncode != 0
    record = json.loads((run_dir / "status" / "compass_scores.json").read_text())
    assert record["status"] == "failed"
    assert "missing input" in record["error"]
    assert "absent.layer.npz" in record["error"]


@pytest.fixture(scope="session")
def merged_results(
    convnet_run: support.FinishedRun, unet_run: support.FinishedRun, data_root: Path
) -> pd.DataFrame:
    merged = support.ARTIFACTS / "merged"
    support.build_once(
        "merged_results",
        lambda: support.run(
            "merge_results.py",
            "--runs",
            str(convnet_run.run_dir),
            str(unet_run.run_dir),
            "--out",
            str(merged),
            data_root=data_root,
        ),
    )
    return pd.read_csv(merged / "results.csv")


def test_merge_keeps_every_row_of_each_model(merged_results: pd.DataFrame) -> None:
    assert set(merged_results.model) == {"convnet", "unet_scratch"}
    assert len(merged_results) == 2 * len(support.ALPHAS) * len(support.EXPECTED_ROWS)


def test_merge_records_each_runs_chosen_layer(
    merged_results: pd.DataFrame, convnet_run: support.FinishedRun, unet_run: support.FinishedRun
) -> None:
    """The two runs chose different layers, so a layer attached to the wrong run would show."""
    chosen = pd.read_csv(support.ARTIFACTS / "merged" / "chosen_layers.csv")
    assert len(chosen) == len(set(merged_results.model)) == 2
    assert chosen.set_index("model").layer.to_dict() == {
        "convnet": convnet_run.layer,
        "unet_scratch": unet_run.layer,
    }
    assert (chosen.min_fraction_ascending == 0).all()


def test_merge_refuses_two_runs_of_one_variant(
    convnet_run: support.FinishedRun, saturated_run: Path, data_root: Path
) -> None:
    """They would merge into one model in every table."""
    same_variant = support.run(
        "merge_results.py",
        "--runs",
        str(convnet_run.run_dir),
        str(saturated_run),
        "--out",
        str(support.ARTIFACTS / "merged_refused"),
        data_root=data_root,
        check=False,
    )
    assert same_variant.returncode != 0
    assert "same model and train_fraction" in same_variant.stderr


def test_merge_refuses_a_run_whose_stages_are_not_all_ok(
    private_convnet: support.FinishedRun, unet_run: support.FinishedRun, data_root: Path
) -> None:
    """A failed choice leaves the old tables, or brings them back from Drive; none may merge."""
    (private_convnet.run_dir / "status" / "choose_layer.json").write_text(
        json.dumps({"stage": "choose_layer", "status": "failed", "error": "no layer qualifies"})
    )
    out = support.ARTIFACTS / "merged_not_ok"
    refused = support.run(
        "merge_results.py",
        "--runs",
        str(private_convnet.run_dir),
        str(unet_run.run_dir),
        "--out",
        str(out),
        data_root=data_root,
        check=False,
    )
    assert refused.returncode != 0
    assert f"{private_convnet.run_dir}: choose_layer is failed" in refused.stderr
    assert str(unet_run.run_dir) not in refused.stderr
    assert not out.exists()


def test_pipeline_check_passes_on_a_healthy_run(
    unet_run: support.FinishedRun, data_root: Path
) -> None:
    summary = support.pipeline_check(unet_run.run_dir, data_root)
    assert "PIPELINE CHECK: PASS" in summary.stdout
    assert "train_data_wait_seconds" in summary.stdout
    assert "'data_wait_seconds'" in summary.stdout
    (support.ARTIFACTS / "pipeline_check.txt").write_text(summary.stdout)


def test_pipeline_check_fails_on_the_job_exit_code(
    unet_run: support.FinishedRun, data_root: Path
) -> None:
    """Every status is ok here, as after an import or argparse error."""
    crashed = support.pipeline_check(
        unet_run.run_dir, data_root, "--job-exit-code", "1", check=False
    )
    assert crashed.returncode == 1
    assert "PIPELINE CHECK: FAIL" in crashed.stdout
    assert "job exit code 1" in crashed.stdout
    assert "PIPELINE CHECK: PASS" not in crashed.stdout


def test_pipeline_check_names_the_failed_stage(
    private_convnet: support.FinishedRun, data_root: Path
) -> None:
    run_dir = private_convnet.run_dir
    (run_dir / "status" / "calibrate.json").write_text(
        json.dumps({"stage": "calibrate", "status": "failed", "error": "boom"})
    )
    report = support.pipeline_check(run_dir, data_root, check=False)
    assert report.returncode == 1
    assert "failing stage: calibrate" in report.stdout
    assert "boom" in report.stdout


def test_pipeline_check_ignores_a_choice_its_stage_did_not_make(
    private_convnet: support.FinishedRun, data_root: Path
) -> None:
    """Print no chosen layer, nor any table written after it, unless choose_layer's status is ok.

    choose_layer deletes chosen_layer.json when no layer qualifies, but no copy to Drive deletes,
    so the next restore brings the old file back; the report must not print it as the run's choice.
    """
    run_dir = private_convnet.run_dir
    (run_dir / "status" / "choose_layer.json").write_text(
        json.dumps({"stage": "choose_layer", "status": "failed", "error": "no layer qualifies"})
    )
    assert (run_dir / "chosen_layer.json").exists()
    report = support.pipeline_check(run_dir, data_root, check=False)
    sections = {
        part.split("\n", 1)[0]: part.split("\n", 1)[1] for part in report.stdout.split("\n--- ")[1:]
    }
    chosen = sections["layer chosen from the validation role"]
    assert "not chosen: choose_layer is failed" in chosen
    assert private_convnet.layer not in chosen
    # Written after the choice, so they belong to the old one; those before it are still current.
    later = ("monotonicity of the chosen layer", "saturated searches", "PICP and MPIW per method")
    for title in later:
        section = next(text for name, text in sections.items() if name.startswith(title))
        assert section.startswith("not shown: choose_layer is failed"), (title, section)
    for title in ("monotonicity for choosing the layer", "|cos(v1, w)|", "losses, validation"):
        section = next(text for name, text in sections.items() if name.startswith(title))
        assert not section.startswith(("not shown", "not produced")), (title, section)


class AbsentCells(NamedTuple):
    run_dir: Path
    in_test: int
    in_calibration: int
    beyond: int


def stored_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as arrays:
        return {name: arrays[name] for name in arrays.files}


def without_cell(path: Path, value: int) -> dict[str, np.ndarray]:
    arrays = stored_arrays(path)
    return arrays | {"cell_index": arrays["cell_index"][arrays["cell_index"] != value]}


def with_cell_beyond(path: Path, beyond: int) -> dict[str, np.ndarray]:
    arrays = stored_arrays(path)
    return arrays | {"cell_index": np.append(arrays["cell_index"][:-1], beyond)}


@pytest.fixture
def absent_cells(private_convnet: support.FinishedRun, data_root: Path) -> AbsentCells:
    run_dir = private_convnet.run_dir
    cells = garimpo_uncertainty.data.read_predictions(data_root, run_dir)
    return AbsentCells(
        run_dir,
        int(cells[cells.role == "test"].cell_index.iloc[1]),
        int(cells[cells.role == "calibration"].cell_index.iloc[1]),
        int(cells.cell_index.max()) + 1_000,
    )


def calibrate_command(run_dir: Path) -> tuple[str, str, list[str]]:
    args = ["--run-dir", str(run_dir), "--cells", "all", "--score-floor-ha", "0.01", "--force"]
    return "calibrate.py", "calibrate", args


def generate_command(run_dir: Path) -> tuple[str, str, list[str]]:
    args = [*support.checkpoint_args(run_dir), "--layer", support.chosen_layer(run_dir), "--force"]
    return "generate_intervals.py", "generate_intervals", args


@pytest.mark.parametrize(
    ("command", "stored", "cell"),
    [
        (calibrate_command, "scores/scores.npz", "in_calibration"),
        (calibrate_command, "scores/scores.npz", "in_test"),
        (generate_command, "scores/scores.npz", "in_test"),
        (generate_command, "calibration/quantiles.npz", "beyond"),
    ],
    ids=[
        "calibrate-calibration_cell",
        "calibrate-test_cell",
        "generate-test_cell",
        "generate-beyond",
    ],
)
def test_absent_cell_index_is_named(
    absent_cells: AbsentCells,
    data_root: Path,
    command: Callable[[Path], tuple[str, str, list[str]]],
    stored: str,
    cell: str,
) -> None:
    """A lookup of an absent cell must name it, where a bare searchsorted returns a neighbor."""
    run_dir = absent_cells.run_dir
    path = run_dir / stored
    absent = getattr(absent_cells, cell)
    broken = with_cell_beyond(path, absent) if cell == "beyond" else without_cell(path, absent)
    np.savez(path, **broken)
    script, stage, args = command(run_dir)
    result = support.run(script, *args, data_root=data_root, check=False)
    assert result.returncode != 0, (script, absent)
    error = json.loads((run_dir / "status" / f"{stage}.json").read_text())["error"]
    assert "cell_index" in error, error
    assert str(absent) in error, error


def test_metrics_stage_names_an_absent_cell_index(
    absent_cells: AbsentCells, data_root: Path
) -> None:
    run_dir, _, _, beyond = absent_cells
    intervals = run_dir / "intervals" / "intervals.npz"
    np.savez(intervals, **with_cell_beyond(intervals, beyond))
    result = support.compute_metrics(
        run_dir, support.BOUNDARY_TOLERANCE, data_root, force=True, check=False
    )
    assert result.returncode != 0
    error = json.loads((run_dir / "status" / "compute_metrics.json").read_text())["error"]
    assert "cell_index" in error, error
    assert str(beyond) in error, error


def test_asymmetric_quantile_past_the_upper_censoring_gives_the_whole_cell(
    private_convnet: support.FinishedRun, data_root: Path
) -> None:
    """A quantile under beta_range, but past beta_range / side_ratio, where the upper side stopped.

    Validation cells need 8 times more beta above than below, so side_ratio is 8; calibration and
    test cells need half the search range below. The joint quantile, half the range, then sends
    the upper endpoint 4 times past the range that was searched.
    """
    run_dir = private_convnet.run_dir
    path = run_dir / "scores" / "scores.npz"
    arrays = stored_arrays(path)
    cells = garimpo_uncertainty.data.read_predictions(data_root, run_dir)
    validation = np.isin(arrays["cell_index"], cells[cells.role == "validation"].cell_index)
    m = garimpo_uncertainty.intervals.COMPASS_METHODS.index("compass_l")
    beta_range = 1e-3
    arrays["beta_range"][m] = beta_range
    for name, (on_validation, elsewhere) in {
        "lower": (0.1, 0.5),
        "upper": (0.8, 0.5),
        "score": (0.5, 0.5),
    }.items():
        arrays[name][m] = beta_range * np.where(validation, on_validation, elsewhere)
    for name in ("saturated", "lower_saturated", "upper_saturated"):
        arrays[name][m] = False
    np.savez(path, **arrays)
    for command in (calibrate_command, generate_command):
        script, _, args = command(run_dir)
        support.run(script, *args, data_root=data_root)

    rows = [
        k
        for k, (method, _) in enumerate(garimpo_uncertainty.intervals.COMBINATIONS)
        if method == "a_compass_l"
    ]
    with np.load(run_dir / "calibration" / "quantiles.npz") as stored:
        assert np.allclose(stored["side_ratio"][:, rows], 8)
    with np.load(run_dir / "intervals" / "intervals.npz") as stored:
        cell_index, quantile = stored["cell_index"], stored["quantile"][:, rows]
        lower, upper = stored["lower"][:, rows], stored["upper"][:, rows]
    assert np.allclose(quantile, beta_range / 2)
    cell_ha = cells.set_index("cell_index").cell_ha.loc[cell_index].to_numpy()
    assert (lower == 0).all()
    assert np.array_equal(upper, np.broadcast_to(cell_ha, upper.shape))


def test_misaligned_predictions_stop_the_stage(
    private_convnet: support.FinishedRun, data_root: Path
) -> None:
    """Two cells swapped in cells.npz would give each the other's areas, with no other symptom."""
    run_dir = private_convnet.run_dir
    path = run_dir / "predictions" / "cells.npz"
    arrays = stored_arrays(path)
    arrays["cell_index"][[0, 1]] = arrays["cell_index"][[1, 0]]
    np.savez(path, **arrays)
    script, stage, args = calibrate_command(run_dir)
    result = support.run(script, *args, data_root=data_root, check=False)
    assert result.returncode != 0
    error = json.loads((run_dir / "status" / f"{stage}.json").read_text())["error"]
    assert "does not align" in error, error


def test_a_data_root_missing_a_tile_refuses_the_checkpoint(
    private_convnet: support.FinishedRun, private_data_root: Path
) -> None:
    """Another data root, where one test tile failed to download, must not rebuild the run."""
    split = pd.read_csv(support.SPLIT_PATH)
    test_tile = split[split.role == "test"].tile_id.iloc[0]
    sidecar = private_data_root / "tiles" / f"{test_tile}.json"
    sidecar.write_text(json.dumps(json.loads(sidecar.read_text()) | {"status": "missing_item"}))
    run_dir = private_convnet.run_dir
    result = support.run(
        "infer.py",
        *support.checkpoint_args(run_dir),
        "--force",
        data_root=private_data_root,
        check=False,
    )
    assert result.returncode != 0
    error = json.loads((run_dir / "status" / "infer.json").read_text())["error"]
    assert "wrong GARIMPO_DATA_DIR or another subset" in error, error


def test_calibration_requires_a_score_floor(data_root: Path, tmp_path: Path) -> None:
    """Relative and local scores divide by an area, zero on empty cells; no default may hide it."""
    missing = support.run(
        "calibrate.py",
        "--run-dir",
        str(tmp_path / "never_read"),
        "--cells",
        "all",
        data_root=data_root,
        check=False,
    )
    assert missing.returncode == 2
    assert "--score-floor-ha" in missing.stderr


@pytest.mark.parametrize("floor", ["0", "-0.01", "nan", "inf"])
def test_calibration_refuses_a_floor_that_is_not_positive_and_finite(
    floor: str, data_root: Path, tmp_path: Path
) -> None:
    """0 divides an empty cell by zero, a negative floor flips the sign, inf gives NaN margins."""
    run_dir = tmp_path / "never_read"
    refused = support.run(
        "calibrate.py",
        "--run-dir",
        str(run_dir),
        "--cells",
        "all",
        f"--score-floor-ha={floor}",
        data_root=data_root,
        check=False,
    )
    assert refused.returncode == 2
    assert "--score-floor-ha must be positive and finite" in refused.stderr
    assert not run_dir.exists()


def test_local_score_floors_a_confident_model(
    private_convnet: support.FinishedRun, data_root: Path
) -> None:
    """A confident model has no pixel variance on the cells it predicts empty.

    The local score would then divide by zero there; the floor of 0.01 ha must stand in.
    """
    run_dir = private_convnet.run_dir
    path = run_dir / "predictions" / "cells.npz"
    arrays = stored_arrays(path)
    arrays["pixel_var"] = np.where(arrays["vhat"] == 0, 0.0, arrays["pixel_var"])
    np.savez(path, **arrays)
    script, _, args = calibrate_command(run_dir)
    support.run(script, *args, data_root=data_root)
    cells = garimpo_uncertainty.data.read_predictions(data_root, run_dir)
    calibration = cells[cells.role == "calibration"]
    assert (calibration.pixel_var == 0).any()
    floor_ha = 0.01
    scale = np.maximum(calibration.pixel_var * garimpo_uncertainty.data.PIXEL_HA, floor_ha)
    scores = (np.abs(calibration.v - calibration.vhat) / scale).to_numpy()
    groups = calibration.group.to_numpy()
    expected = support.pooled_quantile(scores, groups, 1 - 0.1)
    k = garimpo_uncertainty.intervals.COMBINATIONS.index(("conformal_local", "external"))
    with np.load(run_dir / "calibration" / "quantiles.npz") as stored:
        quantile = stored["quantile"][garimpo_uncertainty.intervals.ALPHAS.index(0.1), k]
    assert np.allclose(quantile, expected)


def fit_directions(
    run: support.FinishedRun, data_root: Path, *extra: str
) -> subprocess.CompletedProcess:
    return support.run(
        "fit_directions.py",
        *support.checkpoint_args(run.run_dir),
        "--layers",
        run.layer,
        "--svm-c",
        "1.0",
        "--svm-class-weight",
        "balanced",
        "--svm-pixels-per-cell",
        "50",
        "--force",
        *extra,
        data_root=data_root,
        check=False,
    )


def test_svm_normal_is_turned_toward_more_area() -> None:
    """Features that encode mining with the sign opposite to what grows the area.

    Mining pixels sit at high z0 while the decoder predicts mining where z0 is low, as a model may
    early in training, so the SVM's raw normal points to less area. On the synthetic runs it never
    does, even with inverted masks, which leaves the orientation unexercised there.
    """
    rng = np.random.default_rng(0)
    labels = (rng.random(2000) < 0.3).astype(int)
    features = rng.normal(0, 1, (2000, 3))
    features[:, 0] += 3 * labels
    z = torch.from_numpy(features).view(-1, 3, 1, 1)

    def decode(z: torch.Tensor) -> torch.Tensor:
        return -z[:, :1] + 0.2 * z[:, 1:2]

    def area(logits: torch.Tensor) -> torch.Tensor:
        return compass2d.metrics.soft_area(logits, garimpo_uncertainty.data.PIXEL_HA, 1.0)

    mean_jacobian = compass2d.directions.jacobian_sum(decode, z, area).mean(0).numpy()
    w = compass2d.directions.fit_svm(features, labels, mean_jacobian, 1.0, "balanced", 1000, 0)
    delta = torch.from_numpy(w).view(1, 3, 1, 1)
    assert (area(decode(z + delta)) - area(decode(z - delta))).sum() > 0


def test_a_model_with_no_gradient_stops_the_direction_fit(
    private_convnet: support.FinishedRun, data_root: Path
) -> None:
    """At this temperature every sigmoid saturates, so every Jacobian is exactly zero.

    No direction can then be oriented toward more area, and none may be written.
    """
    result = fit_directions(private_convnet, data_root, "--direction-temperature", "1e30")
    assert result.returncode != 0
    status = private_convnet.run_dir / "status" / "fit_directions.json"
    error = json.loads(status.read_text())["error"]
    assert "mean Jacobian is orthogonal to w" in error, error


@pytest.mark.parametrize("damage", ["file_absent", "row_absent", "nan_picp", "nan_mpiw"])
def test_pipeline_check_fails_on_an_incomplete_results_table(
    private_convnet: support.FinishedRun, data_root: Path, damage: str
) -> None:
    """Every status says ok here: only the table shows that it, a row or a value is gone."""
    path = private_convnet.run_dir / "metrics" / "results.csv"
    results = pd.read_csv(path)
    expected = {
        "file_absent": "results absent",
        "row_absent": "results rows absent",
        "nan_picp": "NaN in PICP or MPIW",
        "nan_mpiw": "NaN in PICP or MPIW",
    }[damage]
    if damage == "file_absent":
        path.unlink()
    else:
        if damage == "row_absent":
            results = results.drop(index=len(results) // 2)
        else:
            results.loc[3, damage.removeprefix("nan_")] = np.nan
        results.to_csv(path, index=False)
    report = support.pipeline_check(private_convnet.run_dir, data_root, check=False)
    assert report.returncode == 1
    assert expected in report.stdout
    assert "PIPELINE CHECK: PASS" not in report.stdout
