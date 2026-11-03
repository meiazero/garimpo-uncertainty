"""End-to-end checks on synthetic data, on CPU, through the same code the cluster runs.

Covered: select_tiles, the download on local rasters (tile_origin, process_tile, _read_band,
_boa_offset, download_scenes), validation with check_split_counts, and every later stage.
Not covered: real HTTP (STAC, SAS tokens, 429 retries, Zenodo), remote reads, real Sentinel-2
scenes, the shell scripts in slurm/ including FORCE_FROM and the EXIT trap, the download CLI, GPU.
Artifacts stay in tests/artifacts/.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
from itertools import combinations
from pathlib import Path

import compass2d.decomposition
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import rasterio.enums
import rasterio.features
import rasterio.transform
import shapely
import shapely.affinity
import torch

import garimpo_uncertainty.cell_dataset
import garimpo_uncertainty.data
import garimpo_uncertainty.download
import garimpo_uncertainty.evaluation
import garimpo_uncertainty.models
import garimpo_uncertainty.validation

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "tests" / "artifacts" / "e2e"
DOWNLOAD_ARTIFACTS = ROOT / "tests" / "artifacts" / "download"
SCENE_ORIGIN = (600_000.0, 9_100_000.0)
SCENE_PX = 2_200
TILE_PX = 128
CELL_PX = 64
EPSG = 32721
ALPHA = 0.1
# Literal copy of tab:combinacoes plus the SVM direction, kept apart from the pipeline's constant.
EXPECTED_ROWS = {
    ("independent_pixels", "none"),
    ("conformal_simple", "external"),
    ("conformal_simple", "external_weighted"),
    ("conformal_simple", "own_region"),
    ("conformal_relative", "external"),
    ("conformal_relative", "own_region"),
    ("conformal_local", "external"),
    ("conformal_local", "own_region"),
    ("cqr", "external"),
    ("cqr", "own_region"),
    ("compass_l", "external"),
    ("compass_l", "external_weighted"),
    ("compass_l", "own_region"),
    ("compass_j_pca", "external"),
    ("compass_j_pca", "own_region"),
    ("compass_j_svm", "external"),
    ("compass_j_svm", "own_region"),
}
CELLS_PER_TILE = (TILE_PX // CELL_PX) ** 2
RUN_ORDER = {
    "convnet": (["--channels", "8", "16", "32"], "up.1", "up.0"),
    "unet": (["--encoder", "resnet18"], "decoder.blocks.4", "decoder.blocks.3"),
}
TORCH_FREE = (
    "import runpy, sys; sys.argv = sys.argv[1:]; "
    "runpy.run_path(sys.argv[0], run_name='__main__'); "
    "assert 'torch' not in sys.modules, 'metrics stage imported torch'"
)


def group_tiles() -> list[dict]:
    """Make one calibration group much larger, so group weights differ from cell weights."""
    plan = [("Asia/Oceania", "train", g, 2) for g in range(5)]
    plan += [("Africa/Europe", "validation", 5, 2)]
    plan += [("Asia/Oceania", "calibration", 6, 6)]
    plan += [("Asia/Oceania", "calibration", g, 1) for g in range(7, 14)]
    plan += [("Amazonia BR", "test", g, 2) for g in range(14, 22)]
    rows, tile_id = [], 0
    for region, role, group, n_tiles in plan:
        for _ in range(n_tiles):
            rows.append(
                {
                    "tile_id": tile_id,
                    "region": region,
                    "cluster_50km": float(group),
                    "role": role,
                    "scale": "Artisanal",
                }
            )
            tile_id += 1
    industrial = [{"region": None, "cluster_50km": None, "scale": "Industrial"}] * 6
    # The last industrial tile is dropped: in the split, but never expected from the download.
    roles = ["train"] * 5 + ["dropped"]
    rows += [
        row | {"tile_id": tile_id + i, "role": role}
        for i, (row, role) in enumerate(zip(industrial, roles, strict=True))
    ]
    return rows


def write_synthetic_tile(root: Path, tile_id: int, rng: np.random.Generator) -> None:
    x0, y0 = 500_000.0 + 3_000 * tile_id, 9_000_000.0
    transform = rasterio.transform.from_origin(x0, y0 + TILE_PX * 10, 10, 10)
    blobs = [
        shapely.Point(x0 + rng.uniform(0, TILE_PX * 10), y0 + rng.uniform(0, TILE_PX * 10)).buffer(
            rng.uniform(80, 250)
        )
        for _ in range(rng.integers(1, 4))
    ]
    geometry = shapely.union_all(blobs)
    mask = rasterio.features.rasterize(
        [geometry], out_shape=(TILE_PX, TILE_PX), transform=transform, fill=0, default_value=1
    ).astype(bool)
    image = rng.normal(1200, 150, (6, TILE_PX, TILE_PX))
    image[:, mask] += 2500
    scl = np.full((TILE_PX, TILE_PX), 4, dtype=np.uint16)
    cells = garimpo_uncertainty.download.write_cells(
        root,
        tile_id,
        np.clip(image, 1, 65535).astype(np.uint16),
        scl,
        geometry,
        transform,
        EPSG,
        CELL_PX,
    )
    garimpo_uncertainty.download.write_sidecar(
        root,
        tile_id,
        {
            "item": f"synthetic_{tile_id}",
            "status": "ok",
            "epsg": EPSG,
            "boa_offset": 0,
            "tile_offset_m": 0.0,
            "tile_box_excess_m": 0.0,
            "resampling_20m": "bilinear",
            "scene_cloud": 0.0,
            "cells": cells,
        },
    )


def build_dataset(root: Path, split_path: Path) -> pd.DataFrame:
    split = pd.DataFrame(group_tiles())
    split.to_csv(split_path, index=False)
    rng = np.random.default_rng(0)
    selected = garimpo_uncertainty.data.select_tiles(
        split, ["Artisanal", "Industrial"], None, None, 0
    )
    for tile_id in selected.tile_id:
        write_synthetic_tile(root, int(tile_id), rng)
    garimpo_uncertainty.download.write_manifest(root, selected)
    return split


def run(
    script: str, *args: str, data_root: Path, check: bool = True
) -> subprocess.CompletedProcess:
    env = {**os.environ, "GARIMPO_DATA_DIR": str(data_root)}
    command = [sys.executable, f"scripts/{script}", *args]
    return subprocess.run(command, cwd=ROOT, env=env, check=check, capture_output=True, text=True)


def train_args(
    arch: str, run_dir: Path, batch_size: str = "8", lr: str = "1e-3", epochs: str = "3"
) -> list[str]:
    return [
        "--arch",
        arch,
        *RUN_ORDER[arch][0],
        "--run-dir",
        str(run_dir),
        "--epochs",
        epochs,
        "--batch-size",
        batch_size,
        "--lr",
        lr,
        "--weight-decay",
        "0",
        "--loss",
        "dice_bce",
        "--seed",
        "0",
        "--device",
        "cpu",
        "--num-workers",
        "0",
    ]


def validate(
    data_root: Path, split_path: Path, report: Path, check: bool = True
) -> subprocess.CompletedProcess:
    return run(
        "validate_data.py",
        "--split",
        str(split_path),
        "--cell-px",
        str(CELL_PX),
        "--report",
        str(report),
        data_root=data_root,
        check=check,
    )


def compute_metrics(
    run_dir: Path, tolerance: float, data_root: Path, force: bool = False, check: bool = True
) -> subprocess.CompletedProcess:
    """Run the metrics stage while asserting that it never imported torch."""
    args = ["--run-dir", str(run_dir), "--boundary-tolerance-px", str(tolerance)]
    if force:
        args.append("--force")
    env = {**os.environ, "GARIMPO_DATA_DIR": str(data_root)}
    command = [sys.executable, "-c", TORCH_FREE, "scripts/compute_metrics.py", *args]
    return subprocess.run(command, cwd=ROOT, env=env, check=check, capture_output=True, text=True)


def run_stages_after_training(
    run_dir: Path, layer: str, layers: list[str], data_root: Path
) -> None:
    checkpoint = str(run_dir / "best_model.pt")
    common = ["--checkpoint", checkpoint, "--run-dir", str(run_dir), "--batch-size", "8"]
    common += ["--device", "cpu", "--num-workers", "0"]
    run("infer.py", *common, data_root=data_root)
    run(
        "fit_directions.py",
        *common,
        "--layers",
        *layers,
        "--svm-c",
        "1.0",
        "--svm-class-weight",
        "balanced",
        "--svm-pixels-per-cell",
        "50",
        data_root=data_root,
    )
    # beta_max equals the search range below, as slurm/stages.sh passes it.
    for use, swept in (("layer", layers), ("endpoint", [layer])):
        run(
            "verify_monotonicity.py",
            *common,
            "--use",
            use,
            "--layers",
            *swept,
            "--beta-max-logits",
            "100",
            "--beta-max-features",
            "100",
            "--steps",
            "9",
            "--cells-per-role",
            "6",
            data_root=data_root,
        )
    run_scores_and_intervals(run_dir, layer, data_root, beta_range="100", threshold="1e-3")


def run_scores_and_intervals(
    run_dir: Path, layer: str, data_root: Path, beta_range: str, threshold: str
) -> None:
    checkpoint = str(run_dir / "best_model.pt")
    common = ["--checkpoint", checkpoint, "--run-dir", str(run_dir), "--batch-size", "8"]
    common += ["--device", "cpu", "--num-workers", "0"]
    run(
        "compass_scores.py",
        *common,
        "--layer",
        layer,
        "--beta-range-features",
        beta_range,
        "--search-threshold-features",
        threshold,
        data_root=data_root,
    )
    run(
        "calibrate.py",
        "--run-dir",
        str(run_dir),
        "--cells",
        "all",
        "--score-floor-ha",
        "0.01",
        data_root=data_root,
    )
    run("generate_intervals.py", *common, "--layer", layer, data_root=data_root)


def pooled_quantile(scores: np.ndarray, groups: np.ndarray, level: float) -> float:
    """Brute-force Dunn et al. pooling: return the smallest score whose mean ECDF reaches level."""
    for candidate in np.sort(scores):
        ecdfs = [np.mean(scores[groups == group] <= candidate) for group in np.unique(groups)]
        if np.mean(ecdfs) >= level - 1e-12:
            return float(candidate)
    return float(scores.max())


def check_decomposition() -> None:
    """f then g must reproduce the model at every layer, and z must really drive the output."""
    torch.manual_seed(0)
    x = torch.randn(2, 6, CELL_PX, CELL_PX)
    for arch, options in (
        ("convnet", {"channels": [8, 16, 32]}),
        ("unet", {"encoder": "resnet18"}),
    ):
        model = garimpo_uncertainty.models.build(arch, in_channels=6, **options).eval()
        layers = garimpo_uncertainty.models.feature_layers(model)
        assert len(layers) > 1
        assert layers[-1] == compass2d.decomposition.LOGITS
        with torch.no_grad():
            expected = model(x)
            for layer in layers:
                z = compass2d.decomposition.encode(model, layer, x)
                rebuilt = compass2d.decomposition.decode(model, layer, x, z)
                assert torch.allclose(rebuilt, expected, atol=1e-5), f"{arch}/{layer}"
                changed = compass2d.decomposition.decode(model, layer, x, z + 1.0)
                assert not torch.allclose(changed, expected, atol=1e-3), f"{arch}/{layer} ignored z"


def check_validation_catches_defects(root: Path, split_path: Path) -> None:
    report = ARTIFACTS / "validation_clean.json"
    validate(root, split_path, report)
    assert json.loads(report.read_text())["problems"] == []

    broken = ARTIFACTS / "broken_data"
    shutil.copytree(root, broken)
    manifest = pd.read_csv(broken / "manifest.csv")
    # One test tile leaks to calibration, another vanishes with a missing scene, one mask changes.
    test_groups = manifest[manifest.role == "test"].groupby("group").tile_id.unique()
    leaked, vanished = test_groups.iloc[0][0], test_groups.iloc[1][0]
    bad_cell = manifest[(manifest.role == "calibration") & (manifest.mask_ha > 0)].path.iloc[0]
    manifest.loc[manifest.tile_id == leaked, "role"] = "calibration"
    manifest = manifest[manifest.tile_id != vanished]
    # A box wider than the window explains its offset, as in tile 684; beyond that it is a defect.
    columns = ["tile_offset_m", "tile_box_excess_m"]
    explained, misaligned = manifest[manifest.role == "train"].tile_id.unique()[:2]
    manifest.loc[manifest.tile_id == explained, columns] = [15.6, 31.0]
    manifest.loc[manifest.tile_id == misaligned, columns] = [26.6, 31.0]
    manifest.to_csv(broken / "manifest.csv", index=False)
    statuses = pd.read_csv(broken / "download_report.csv")
    statuses.loc[statuses.tile_id == vanished, "status"] = "missing_item"
    statuses.to_csv(broken / "download_report.csv", index=False)
    with rasterio.open(broken / bad_cell, "r+") as dataset:
        dataset.write(np.zeros((CELL_PX, CELL_PX), dtype=np.uint16), 7)

    broken_report = ARTIFACTS / "validation_broken.json"
    result = validate(broken, split_path, broken_report, check=False)
    assert result.returncode != 0
    problems = " ".join(json.loads(broken_report.read_text())["problems"])
    for check in ("group_leakage", "role_mismatch", "missing_scenes", "partial_group", "mask_area"):
        assert check in problems, f"{check} not reported: {problems}"
    assert f"tile_offset: {CELLS_PER_TILE} cases" in problems, problems


def check_training_artifacts(run_dir: Path, arch: str) -> None:
    assert len(pd.read_csv(run_dir / "history.csv")) == 3
    assert (run_dir / "last.pt").exists()
    metadata = garimpo_uncertainty.data.read_run_metadata(run_dir)
    for key in ("arch", "seed", "normalization", "split_hash", "root_commit", "compass_commit"):
        assert key in metadata, key
    assert metadata["arch"] == arch
    assert len(metadata["normalization"]["mean"]) == 6
    assert len(metadata["normalization"]["std"]) == 6


def check_results(run_dir: Path, arch: str, split: pd.DataFrame) -> None:
    cells = pd.read_csv(run_dir / "predictions" / "cells.csv")
    results = pd.read_csv(run_dir / "metrics" / "results.csv")
    keys = set(zip(results.method, results.calibration, strict=True))
    assert keys == EXPECTED_ROWS
    assert len(results) == len(EXPECTED_ROWS)
    assert (results.model == arch).all()

    intervals = pd.read_csv(run_dir / "intervals" / "intervals.csv")
    assert (intervals.lower >= 0).all()
    assert (intervals.lower <= intervals.upper).all()
    assert (intervals.upper <= intervals.cell_ha + 1e-6).all()

    # Metrics in results.csv recomputed from per-cell intervals, as a mean of group means.
    for row in results.itertuples():
        one = intervals[
            (intervals.method == row.method) & (intervals.calibration == row.calibration)
        ]
        width = (one.upper - one.lower).groupby(one.group).mean().mean()
        covered = ((one.v >= one.lower) & (one.v <= one.upper)).groupby(one.group).mean().mean()
        penalty = (2 / ALPHA) * (
            (one.lower - one.v).clip(lower=0) + (one.v - one.upper).clip(lower=0)
        )
        score = (one.upper - one.lower + penalty).groupby(one.group).mean().mean()
        assert np.isclose(row.mpiw, width), (row.method, row.calibration)
        assert np.isclose(row.picp, covered), (row.method, row.calibration)
        assert np.isclose(row.interval_score, score), (row.method, row.calibration)
        assert row.picp_ci_low <= row.picp_ci_high
        assert row.n_groups == intervals.group.nunique()

    # The quantile must weigh groups equally; the big calibration group makes this bite.
    calibration = cells[cells.role == "calibration"]
    test = cells[cells.role == "test"]
    external = intervals[
        (intervals.method == "conformal_simple") & (intervals.calibration == "external")
    ]
    scores = np.abs(calibration.v - calibration.vhat).to_numpy()
    expected = pooled_quantile(scores, calibration.group.to_numpy(), 1 - ALPHA)
    assert np.allclose(external.quantile_value, expected)

    # Own-region calibration must exclude the group being evaluated.
    own = intervals[
        (intervals.method == "conformal_simple") & (intervals.calibration == "own_region")
    ]
    test_scores = np.abs(test.v - test.vhat).to_numpy()
    test_groups = test.group.to_numpy()
    for group in np.unique(test_groups):
        others = test_groups != group
        expected = pooled_quantile(test_scores[others], test_groups[others], 1 - ALPHA)
        assert np.allclose(own[own.group == group].quantile_value, expected), group

    # Exchangeable synthetic regions: conformal must not collapse. Monotone logits hold the point.
    for method in ("conformal_simple", "compass_l"):
        picp = results[(results.method == method) & (results.calibration == "external")].picp.item()
        assert picp >= 0.6, (method, picp)
    logits = intervals[(intervals.method == "compass_l") & (intervals.calibration == "external")]
    assert ((logits.lower <= logits.vhat + 1e-6) & (logits.vhat <= logits.upper + 1e-6)).all()

    # Adding a constant to every logit can only grow the area, whatever the model.
    monotonicity = pd.read_csv(run_dir / "monotonicity_layer.csv")
    uniform = monotonicity[monotonicity.direction == "uniform"]
    assert (uniform.fraction_ascending == 1.0).all()
    assert {"pca", "svm"} <= set(monotonicity.direction)
    # Choosing the layer must never read the test role.
    assert set(monotonicity.role) == {"validation"}
    assert set(pd.read_csv(run_dir / "monotonicity_endpoint.csv").role) == {"calibration", "test"}

    diagnostics = pd.read_csv(run_dir / "directions" / "diagnostics.csv")
    assert diagnostics.abs_cos_v1_w.between(0, 1 + 1e-9).all()
    assert diagnostics.pca_explained_variance.between(0, 1, inclusive="right").all()
    assert (diagnostics.svm_mean_jacobian_projection > 0).all()
    assert (diagnostics.n_train_cells == (split.role == "train").sum() * CELLS_PER_TILE).all()

    # w shares the scale convention of the PCA direction, so one beta_range serves both.
    with np.load(run_dir / "directions" / f"{diagnostics.layer.iloc[0]}.npz") as fitted:
        assert np.isclose(fitted["svm_w"].std(ddof=1), 1.0)

    comparisons = pd.read_csv(run_dir / "metrics" / "width_comparisons.csv")
    assert len(comparisons) == len(list(combinations(EXPECTED_ROWS, 2)))
    rows = results.set_index(["method", "calibration"]).mpiw
    for pair in comparisons.itertuples():
        difference = rows[(pair.method_a, pair.calibration_a)]
        difference -= rows[(pair.method_b, pair.calibration_b)]
        assert np.isclose(pair.mean_difference, difference)
        assert pair.ci_low <= pair.ci_high

    coverage = pd.read_csv(run_dir / "metrics" / "coverage_comparisons.csv")
    assert len(coverage) == len(comparisons)
    picp = results.set_index(["method", "calibration"]).picp
    for pair in coverage.itertuples():
        difference = picp[(pair.method_a, pair.calibration_a)]
        difference -= picp[(pair.method_b, pair.calibration_b)]
        assert np.isclose(pair.mean_difference, difference)

    # Q1 recomputed apart from the pipeline, on test cells with mining.
    spearman = pd.read_csv(run_dir / "metrics" / "q1_spearman.csv")
    assert len(spearman) == 4
    assert (spearman.model == arch).all()
    segmentation = pd.read_csv(run_dir / "metrics" / "cell_metrics.csv")
    per_cell = intervals.drop_duplicates("cell_id").merge(segmentation, on="cell_id")
    per_cell = per_cell[per_cell.v > 0]
    expected = (per_cell.vhat - per_cell.v).abs().corr(per_cell.dice, method="spearman")
    row = spearman[
        (spearman.area_error == "abs_error_ha") & (spearman.segmentation_quality == "dice")
    ].iloc[0]
    assert np.isclose(row.spearman, expected, equal_nan=True)
    assert (results.n_cells_boundary_by_rule == segmentation.boundary_by_rule.sum()).all()


def file_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def metric_files_only(names: set[str]) -> set[str]:
    return {
        name
        for name in names
        if not name.startswith("metrics/") and name != "status/compute_metrics.json"
    }


def check_metrics_never_retrain(run_dir: Path, data_root: Path) -> None:
    """Another tolerance reruns only the metrics stage: nothing upstream may change."""
    first_tolerance, second_tolerance = 1.0, 5.0
    compute_metrics(run_dir, first_tolerance, data_root, force=True)
    first = pd.read_csv(run_dir / "metrics" / "cell_metrics.csv").set_index("cell_id")
    before = file_hashes(run_dir)

    # Without --force a different tolerance must refuse, not silently keep the old table.
    refused = compute_metrics(run_dir, second_tolerance, data_root, check=False)
    assert refused.returncode != 0
    assert not metric_files_only(
        {name for name, value in file_hashes(run_dir).items() if before.get(name) != value}
    )
    assert pd.read_csv(run_dir / "metrics" / "cell_metrics.csv").set_index("cell_id").equals(first)
    # A refusal must never read as a passing run.
    report = run("pipeline_check.py", "--run-dir", str(run_dir), data_root=data_root, check=False)
    assert report.returncode == 1
    assert "failing stage: compute_metrics (refused)" in report.stdout
    assert "PIPELINE CHECK: PASS" not in report.stdout

    compute_metrics(run_dir, second_tolerance, data_root, force=True)
    after = file_hashes(run_dir)
    second = pd.read_csv(run_dir / "metrics" / "cell_metrics.csv").set_index("cell_id")

    assert after["best_model.pt"] == before["best_model.pt"]
    assert after["last.pt"] == before["last.pt"]
    assert set(after) == set(before)
    changed = {name for name in after if after[name] != before[name]}
    assert not metric_files_only(changed), metric_files_only(changed)
    assert "metrics/results.csv" in changed

    results = pd.read_csv(run_dir / "metrics" / "results.csv")
    assert (results.boundary_tolerance_px == second_tolerance).all()
    # A wider tolerance can only match more boundary pixels, and Dice must not move at all.
    pair = first.boundary_f1.dropna().align(second.boundary_f1.dropna())
    assert (pair[1] >= pair[0] - 1e-12).all()
    assert (pair[1] > pair[0]).any()
    assert first.dice.equals(second.dice)


def check_cell_dataset_applies_offset(root: Path) -> None:
    """Baseline 04.00 shifts DN; the dataset must undo it and floor at zero."""
    cells = pd.read_csv(root / "manifest.csv").head(1).assign(boa_offset=-1000)
    image, _, _ = garimpo_uncertainty.cell_dataset.CellDataset(root, cells, None)[0]
    with rasterio.open(root / cells.path.iloc[0]) as source:
        raw = source.read(list(range(1, 7))).astype(np.float32)
    assert (raw < 1000).any()
    expected = np.maximum(raw - 1000, 0) / garimpo_uncertainty.data.REFLECTANCE_SCALE
    assert np.allclose(image.numpy(), expected)


def check_training_refusals(run_dir: Path, arch: str, data_root: Path) -> None:
    status = run_dir / "status" / "train.json"

    # Training's batch size shapes the weights: another value refuses, the original then skips.
    refused = run(
        "train.py", *train_args(arch, run_dir, batch_size="4"), data_root=data_root, check=False
    )
    assert refused.returncode != 0
    assert json.loads(status.read_text())["status"] == "refused"
    rerun = run("train.py", *train_args(arch, run_dir), data_root=data_root)
    assert "skipping" in rerun.stdout
    assert json.loads(status.read_text())["status"] == "ok"

    # An interrupted run has no finished status, so last.pt itself must refuse other arguments.
    healthy = status.read_text()
    weights = file_hashes(run_dir)["last.pt"]
    status.unlink()
    resumed = run(
        "train.py", *train_args(arch, run_dir, lr="1e-2"), data_root=data_root, check=False
    )
    assert resumed.returncode != 0
    record = json.loads(status.read_text())
    assert record["status"] == "failed"
    assert "['lr']" in record["error"]
    assert file_hashes(run_dir)["last.pt"] == weights
    status.write_text(healthy)


def check_missing_input_is_recorded(run_dir: Path, data_root: Path) -> None:
    result = run(
        "compass_scores.py",
        "--checkpoint",
        str(run_dir / "best_model.pt"),
        "--run-dir",
        str(run_dir),
        "--layer",
        "absent.layer",
        "--batch-size",
        "8",
        "--device",
        "cpu",
        "--num-workers",
        "0",
        data_root=data_root,
        check=False,
    )
    assert result.returncode != 0
    record = json.loads((run_dir / "status" / "compass_scores.json").read_text())
    assert record["status"] == "failed"
    assert "missing input" in record["error"]
    assert "absent.layer.npz" in record["error"]


def check_all_nan_dice_fails(data_root: Path, runs: Path) -> None:
    """Give validation cells no mining: NaN every epoch, so no best checkpoint ever exists."""
    no_mining = ARTIFACTS / "data_no_validation_mining"
    shutil.copytree(data_root, no_mining)
    manifest = pd.read_csv(no_mining / "manifest.csv")
    for path in manifest[manifest.role == "validation"].path:
        with rasterio.open(no_mining / path, "r+") as dataset:
            dataset.write(
                np.zeros((CELL_PX, CELL_PX), dtype=np.uint16), garimpo_uncertainty.data.MASK_BAND
            )
    run_dir = runs / "convnet_all_nan"
    result = run(
        "train.py", *train_args("convnet", run_dir, epochs="1"), data_root=no_mining, check=False
    )
    assert result.returncode != 0
    record = json.loads((run_dir / "status" / "train.json").read_text())
    assert record["status"] == "failed"
    assert "every val_dice is NaN" in record["error"]
    assert not (run_dir / "best_model.pt").exists()


def test_boundary_f1_without_a_boundary() -> None:
    whole = np.ones((8, 8), dtype=bool)
    empty = np.zeros((8, 8), dtype=bool)
    partial = empty.copy()
    partial[2:6, 2:6] = True
    assert garimpo_uncertainty.evaluation.boundary_f1(whole, whole, 1) == 1.0
    assert garimpo_uncertainty.evaluation.boundary_f1(empty, whole, 1) == 0.0
    assert garimpo_uncertainty.evaluation.boundary_f1(partial, whole, 1) == 0.0


def test_pipeline_end_to_end() -> None:
    shutil.rmtree(ARTIFACTS, ignore_errors=True)
    ARTIFACTS.mkdir(parents=True)
    data_root = ARTIFACTS / "data"
    split_path = ARTIFACTS / "split.csv"
    split = build_dataset(data_root, split_path)

    check_decomposition()
    check_validation_catches_defects(data_root, split_path)
    check_cell_dataset_applies_offset(data_root)

    runs = ARTIFACTS / "runs"
    for arch, (_, layer, other_layer) in RUN_ORDER.items():
        run_dir = runs / arch
        run("train.py", *train_args(arch, run_dir), data_root=data_root)
        check_training_artifacts(run_dir, arch)
        run_stages_after_training(run_dir, layer, [layer, other_layer], data_root)

        # An undefined boundary tolerance must stop the stage, not be chosen silently.
        missing = run(
            "compute_metrics.py", "--run-dir", str(run_dir), data_root=data_root, check=False
        )
        assert missing.returncode == 2
        compute_metrics(run_dir, 2.0, data_root)
        check_results(run_dir, arch, split)

    check_training_refusals(runs / "convnet", "convnet", data_root)
    check_all_nan_dice_fails(data_root, runs)

    # A finished stage skips its work when rerun, also with another batch size after an OOM.
    unet = runs / "unet"
    before = (unet / "predictions" / "test.npy").stat().st_mtime_ns
    rerun = run(
        "infer.py",
        "--checkpoint",
        str(unet / "best_model.pt"),
        "--run-dir",
        str(unet),
        "--batch-size",
        "4",
        "--device",
        "cpu",
        "--num-workers",
        "0",
        data_root=data_root,
    )
    assert "skipping" in rerun.stdout
    assert (unet / "predictions" / "test.npy").stat().st_mtime_ns == before

    check_metrics_never_retrain(unet, data_root)

    # A beta_range too small to cover anything must be reported as saturation, not hidden.
    saturated = runs / "convnet_saturated"
    shutil.copytree(runs / "convnet", saturated)
    shutil.rmtree(saturated / "scores")
    shutil.rmtree(saturated / "calibration")
    shutil.rmtree(saturated / "intervals")
    shutil.rmtree(saturated / "metrics")
    for stage in ("compass_scores", "calibrate", "generate_intervals", "compute_metrics"):
        (saturated / "status" / f"{stage}.json").unlink()
    run_scores_and_intervals(saturated, "up.1", data_root, beta_range="1e-6", threshold="1e-7")
    compute_metrics(saturated, 2.0, data_root)
    table = pd.read_csv(saturated / "metrics" / "results.csv")
    j_rows = table[table.method.str.startswith("compass_j")]
    assert (j_rows.n_saturated_calibration > 0).all()
    check_missing_input_is_recorded(saturated, data_root)

    # Merge and the debugging report are the last steps the pilot runs.
    merged = ARTIFACTS / "merged"
    run(
        "merge_results.py",
        "--runs",
        str(runs / "convnet"),
        str(runs / "unet"),
        "--out",
        str(merged),
        data_root=data_root,
    )
    results = pd.read_csv(merged / "results.csv")
    assert set(results.model) == {"convnet", "unet"}
    assert len(results) == 2 * len(EXPECTED_ROWS)

    summary = run("pipeline_check.py", "--run-dir", str(unet), data_root=data_root)
    assert "PIPELINE CHECK: PASS" in summary.stdout
    (ARTIFACTS / "pipeline_check.txt").write_text(summary.stdout)

    # Every status is ok here, as after an import or argparse error: the exit code must still fail.
    crashed = run(
        "pipeline_check.py",
        "--run-dir",
        str(unet),
        "--job-exit-code",
        "1",
        data_root=data_root,
        check=False,
    )
    assert crashed.returncode == 1
    assert "PIPELINE CHECK: FAIL" in crashed.stdout
    assert "job exit code 1" in crashed.stdout
    assert "PIPELINE CHECK: PASS" not in crashed.stdout

    # A failed stage must name itself in the report and fail the check.
    status = unet / "status" / "calibrate.json"
    healthy = status.read_text()
    status.write_text(json.dumps({"stage": "calibrate", "status": "failed", "error": "boom"}))
    report = run("pipeline_check.py", "--run-dir", str(unet), data_root=data_root, check=False)
    assert report.returncode == 1
    assert "failing stage: calibrate" in report.stdout
    assert "boom" in report.stdout
    status.write_text(healthy)


def test_select_tiles_keeps_whole_groups_in_a_subset() -> None:
    split = pd.DataFrame(group_tiles())
    scales = ["Artisanal", "Industrial"]
    selected = garimpo_uncertainty.data.select_tiles(split, scales, 1, 2, 0)
    assert selected.equals(garimpo_uncertainty.data.select_tiles(split, scales, 1, 2, 0))

    artisanal = selected[selected.scale == "Artisanal"]
    groups_by_role = artisanal.groupby("role").cluster_50km.nunique().to_dict()
    assert groups_by_role == dict.fromkeys(garimpo_uncertainty.data.ROLES, 1)
    for group, rows in artisanal.groupby("cluster_50km"):
        assert len(rows) == (split.cluster_50km == group).sum(), group
    assert (selected.scale == "Industrial").sum() == 2
    assert "dropped" not in set(selected.role)


def write_local_band(path: Path, band: np.ndarray, pixel_m: float) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=band.shape[0],
        width=band.shape[1],
        count=1,
        dtype="uint16",
        crs=f"EPSG:{EPSG}",
        transform=rasterio.transform.from_origin(*SCENE_ORIGIN, pixel_m, pixel_m),
        compress="zstd",
    ) as target:
        target.write(band.astype(np.uint16), 1)


def write_local_scene(directory: Path) -> tuple[dict, np.ndarray]:
    """Write a Sentinel-2-like scene; 10 m values encode their position, so a shift shows.

    The window of the tile below starts at pixel 50. Its first cell is all cloud in SCL and its last
    cell is nodata in every band.
    """
    directory.mkdir(parents=True)
    rows, cols = np.mgrid[0:SCENE_PX, 0:SCENE_PX]
    position = 1 + cols % 251 + 256 * (rows % 251)
    position[1586:2098, 1586:2098] = 0
    _, coarse_cols = np.mgrid[0 : SCENE_PX // 2, 0 : SCENE_PX // 2]
    gradient = 1000 + 20 * coarse_cols
    gradient[791:1051, 791:1051] = 0  # two pixels wider, so bilinear stays zero across the cell
    scl = np.full(coarse_cols.shape, 4)
    scl[25:281, 25:281] = 9
    assets = {}
    for band in (*garimpo_uncertainty.data.BANDS, "SCL"):
        path = directory / f"{band}.tif"
        if band in garimpo_uncertainty.download.ASSETS_10M:
            write_local_band(path, position, 10.0)
        else:
            write_local_band(path, scl if band == "SCL" else gradient, 20.0)
        assets[band] = {"href": str(path)}
    item = {
        "properties": {"eo:cloud_cover": 1.0, "s2:processing_baseline": "04.00"},
        "assets": assets,
    }
    return item, position


def to_wgs84(geometry: shapely.Geometry) -> shapely.Geometry:
    return gpd.GeoSeries([geometry], crs=EPSG).to_crs(4326).iloc[0]


def test_download_on_local_rasters(monkeypatch: pytest.MonkeyPatch) -> None:
    download = garimpo_uncertainty.download
    shutil.rmtree(DOWNLOAD_ARTIFACTS, ignore_errors=True)
    root = DOWNLOAD_ARTIFACTS / "data"
    item, position = write_local_scene(DOWNLOAD_ARTIFACTS / "scene")

    # Same 20,511 m box against a 20,480 m window as tile 684 (outputs/exemplos_dataset.csv).
    tile_box = shapely.box(600_484.5, 9_079_004.5, 620_995.5, 9_099_515.5)
    grid = rasterio.transform.from_origin(*SCENE_ORIGIN, 10, 10)
    tile_px = garimpo_uncertainty.data.TILE_PX
    assert download.tile_origin(grid, tile_box.bounds, tile_px) == (50, 50, 15.5, 31.0)

    assert download._boa_offset({"properties": {"s2:processing_baseline": "04.00"}}) == -1000
    assert download._boa_offset({"properties": {"s2:processing_baseline": "03.01"}}) == 0

    def fake_get_json(url: str) -> dict:
        if url == download.TOKEN_URL:
            return {"token": "sig=1"}
        if url == download.STAC + "S2_LOCAL":
            return item
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(download, "_get_json", fake_get_json)
    assert download._signed("a.tif", download._Token()) == "a.tif?sig=1"
    monkeypatch.setattr(download, "_signed", lambda href, _token: href)

    # 10 m bands are read without resampling; 20 m bands land on the 10 m grid.
    cell_px = garimpo_uncertainty.data.CELL_PX
    first_cell = (600_500.0, 9_094_380.0, 605_620.0, 9_099_500.0)
    nearest, bilinear = rasterio.enums.Resampling.nearest, rasterio.enums.Resampling.bilinear
    exact = download._read_band(item["assets"]["B02"]["href"], first_cell, cell_px, nearest)
    assert exact.dtype == np.uint16
    assert np.array_equal(exact, position[50:562, 50:562])
    b11 = item["assets"]["B11"]["href"]
    stairs = download._read_band(b11, first_cell, cell_px, nearest)
    ramp = download._read_band(b11, first_cell, cell_px, bilinear)
    expected = 1000.0 + 10.0 * np.arange(50, 562)
    for band in (stairs, ramp):
        assert band.shape == (cell_px, cell_px)
        assert np.abs(band[8:-8, 8:-8] - expected[8:-8]).max() <= 20
    assert (stairs != ramp).any()

    mask = shapely.box(603_000, 9_090_000, 606_000, 9_093_000)
    off_scene = shapely.affinity.translate(tile_box, xoff=100_000)
    items = ["S2_LOCAL", "S2_ABSENT", "S2_ABSENT", "S2_LOCAL"]
    geometries = [tile_box, tile_box, tile_box, off_scene]
    zenodo = root / "zenodo"
    zenodo.mkdir(parents=True)
    gpd.GeoDataFrame(
        {
            "tile_id": range(4),
            "s2_tile_id": items,
            "geometry": [to_wgs84(g) for g in geometries],
        },
        crs=4326,
    ).to_file(download.gpkg_path(root), layer="tiles", driver="GPKG")
    gpd.GeoDataFrame({"tile_id": range(4), "geometry": [to_wgs84(mask)] * 4}, crs=4326).to_file(
        download.gpkg_path(root), layer="preferred_polygons", driver="GPKG"
    )

    split = pd.DataFrame(
        {
            "tile_id": range(4),
            "region": ["Amazonia BR", None, "Asia/Oceania", "Asia/Oceania"],
            "cluster_50km": [0.0, None, 1.0, 2.0],
            "role": ["test", "train", "calibration", "train"],
            "scale": ["Artisanal", "Industrial", "Artisanal", "Artisanal"],
        }
    )
    selected = split[split.tile_id < 2]
    download.download_scenes(root, selected, 2, bilinear)

    sidecar = json.loads((root / "tiles" / "0.json").read_text())
    assert sidecar["status"] == "ok"
    assert sidecar["resampling_20m"] == "bilinear"
    assert sidecar["boa_offset"] == -1000
    assert sidecar["tile_offset_m"] > garimpo_uncertainty.data.PIXEL_M
    assert np.isclose(sidecar["tile_offset_m"], 15.5, atol=0.01)
    assert np.isclose(sidecar["tile_box_excess_m"], 31.0, atol=0.01)
    assert json.loads((root / "tiles" / "1.json").read_text())["status"] == "missing_item"
    with rasterio.open(root / "cells" / "0" / "0_0.tif") as source:
        assert np.array_equal(source.read(1), position[50:562, 50:562])
        assert (source.read(garimpo_uncertainty.data.SCL_BAND) == 9).all()

    # The real offset above must pass, and the missing industrial tile must not fail the run.
    manifest_path = root / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 16
    assert np.isclose(manifest.mask_ha.sum(), mask.area / 1e4, rtol=0.01)
    counts = {"Artisanal/test": 1, "Industrial/train": 1, "Industrial/dropped": 5}
    report, cells = garimpo_uncertainty.validation.validate(
        root, selected, cell_px, {"tiles_by_scale_role": counts}, 2
    )
    assert report["problems"] == []
    assert report["industrial_tiles_excluded"] == 1
    assert report["diagnostics"]["by_role"]["test"]["cells_100_percent_cloud"] == 1
    assert report["diagnostics"]["by_role"]["test"]["cells_100_percent_nodata"] == 1
    assert np.allclose(cells.cloud_fraction, cells.cloud_fraction_file)

    statuses = pd.read_csv(root / "download_report.csv")
    for wrong in ({"Artisanal/test": 2}, {"Industrial/train": 2}):
        problems = garimpo_uncertainty.validation.check_split_counts(
            manifest, statuses, {"tiles_by_scale_role": wrong}
        )
        assert problems and problems[0].startswith("split_counts")

    # An outage is not a missing tile: it must stop the run instead of shrinking the training set.
    healthy_report = (root / "download_report.csv").read_text()
    statuses.loc[statuses.tile_id == 1, "status"] = "http_503"
    statuses.to_csv(root / "download_report.csv", index=False)
    report, _ = garimpo_uncertainty.validation.validate(
        root, selected, cell_px, {"tiles_by_scale_role": counts}, 2
    )
    assert "industrial_download_errors" in " ".join(report["problems"])
    assert "split_counts" in " ".join(report["problems"])
    assert report["industrial_tiles_excluded"] == 0
    (root / "download_report.csv").write_text(healthy_report)

    # An unchanged rerun keeps the manifest's identity, so later stages do not refuse.
    before = manifest_path.stat().st_mtime_ns
    download.download_scenes(root, selected, 2, bilinear)
    assert manifest_path.stat().st_mtime_ns == before

    # Another resampling must refuse, and a forced re-cut must reach later stages.
    with pytest.raises(ValueError, match="resampling"):
        download.download_scenes(root, selected, 2, nearest)
    assert manifest_path.stat().st_mtime_ns == before
    download.download_scenes(root, selected, 2, nearest, force=True)
    assert json.loads((root / "tiles" / "0.json").read_text())["resampling_20m"] == "nearest"
    assert manifest_path.stat().st_mtime_ns != before

    # A missing artisanal tile and a window off the scene both fail the validation.
    download.download_scenes(root, split, 2, nearest)
    assert json.loads((root / "tiles" / "2.json").read_text())["status"] == "missing_item"
    assert json.loads((root / "tiles" / "3.json").read_text())["status"] == "error_ValueError"
    report, _ = garimpo_uncertainty.validation.validate(root, split, cell_px, None, 2)
    problems = " ".join(report["problems"])
    assert "missing_scenes: 2 cases" in problems, problems
    assert "partial_group" in problems, problems
