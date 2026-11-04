"""End-to-end checks on synthetic data, on CPU, through the same code the cluster runs.

Covered: select_tiles on the real split, the cell store, the download on local rasters (tile_origin,
process_tile, _read_band, _boa_offset, download_scenes), validation with check_split_counts, every
later stage, and two training fractions sharing one data root.
Not covered: real HTTP (STAC, SAS tokens, 429 retries, Zenodo), remote reads, real Sentinel-2
scenes, the shell scripts in slurm/ including FORCE_FROM and the EXIT trap, the download CLI, GPU,
the real ImageNet and Sentinel-2 MoCo files (synthetic ones stand in).
Artifacts stay in tests/artifacts/.
"""

import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import urllib.error
from itertools import combinations
from pathlib import Path

import compass2d.compass
import compass2d.directions
import compass2d.metrics
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import rasterio.enums
import rasterio.features
import rasterio.transform
import safetensors.torch
import shapely
import shapely.affinity
import torch

import garimpo_uncertainty.cell_dataset
import garimpo_uncertainty.checkpoints
import garimpo_uncertainty.data
import garimpo_uncertainty.download
import garimpo_uncertainty.evaluation
import garimpo_uncertainty.intervals
import garimpo_uncertainty.models
import garimpo_uncertainty.validation

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "tests" / "artifacts" / "e2e"
STORE_ARTIFACTS = ROOT / "tests" / "artifacts" / "store"
DOWNLOAD_ARTIFACTS = ROOT / "tests" / "artifacts" / "download"
WEIGHTS_ARTIFACTS = ROOT / "tests" / "artifacts" / "weights"
IMAGENET_ARTIFACTS = ROOT / "tests" / "artifacts" / "imagenet"
FRACTION_ARTIFACTS = ROOT / "tests" / "artifacts" / "fractions"
# Literal training tiles per fraction on outputs/split.csv, kept apart from the pipeline.
REAL_SPLIT_TRAIN_TILES = {0.01: (1, 9), 0.05: (3, 47), 0.10: (6, 95), 1.0: (56, 947)}
REAL_SPLIT_OTHER_TILES = 176
# Literal positions of B2, B3, B4, B8, B11, B12 among the 13 bands, kept apart from the pipeline.
SENTINEL2_MOCO_COLUMNS = (1, 2, 3, 7, 11, 12)
SCENE_ORIGIN = (600_000.0, 9_100_000.0)
SCENE_PX = 2_200
# The smallest cell a depth-5 encoder accepts, to offset the cost of ResNet-50 on CPU.
TILE_PX = 64
CELL_PX = 32
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
    "unet": (
        ["--encoder-weights", "none"],
        "decoder.blocks.4",
        "decoder.blocks.3",
    ),
}
MODEL_NAMES = {"convnet": "convnet", "unet": "unet_scratch"}
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


def fill_tiles(
    root: Path, split: pd.DataFrame, tile_ids: list[int], rng: np.random.Generator
) -> None:
    """Write synthetic tiles through the real cell writer, allocating the store on first use."""
    order = garimpo_uncertainty.data.tile_table(split).set_index("tile_id").tile_index
    garimpo_uncertainty.download.create_store(root, len(order), TILE_PX, CELL_PX)
    store = garimpo_uncertainty.data.open_store(root, "r+")
    for tile_id in tile_ids:
        write_synthetic_tile(root, store, tile_id, int(order[tile_id]), rng)


def write_synthetic_tile(
    root: Path,
    store: garimpo_uncertainty.data.CellStore,
    tile_id: int,
    tile_index: int,
    rng: np.random.Generator,
) -> None:
    x0, y0 = 500_000.0 + 3_000 * tile_id, 9_000_000.0
    transform = rasterio.transform.from_origin(x0, y0 + TILE_PX * 10, 10, 10)
    blobs = [
        shapely.Point(x0 + rng.uniform(0, TILE_PX * 10), y0 + rng.uniform(0, TILE_PX * 10)).buffer(
            rng.uniform(40, 125)
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
    bound = garimpo_uncertainty.download.write_cells(
        store,
        tile_index,
        np.clip(image, 1, 65535).astype(np.uint16),
        scl,
        geometry,
        transform,
        CELL_PX,
    )
    garimpo_uncertainty.download.write_sidecar(
        root,
        tile_id,
        bound
        | {
            "item": f"synthetic_{tile_id}",
            "status": "ok",
            "epsg": EPSG,
            "transform": list(transform)[:6],
            "boa_offset": 0,
            "tile_offset_m": 0.0,
            "tile_box_excess_m": 0.0,
            "resampling_20m": "bilinear",
            "scene_cloud": 0.0,
        },
    )


def build_dataset(root: Path, split_path: Path) -> pd.DataFrame:
    split = pd.DataFrame(group_tiles())
    split.to_csv(split_path, index=False)
    selected = garimpo_uncertainty.data.select_tiles(split, 1.0, 0)
    fill_tiles(root, split, selected.tile_id.tolist(), np.random.default_rng(0))
    return split


def run(
    script: str, *args: str, data_root: Path, check: bool = True
) -> subprocess.CompletedProcess:
    env = {**os.environ, "GARIMPO_DATA_DIR": str(data_root)}
    command = [sys.executable, f"scripts/{script}", *args]
    return subprocess.run(command, cwd=ROOT, env=env, check=check, capture_output=True, text=True)


def train_args(
    arch: str,
    run_dir: Path,
    batch_size: str = "8",
    lr: str = "1e-3",
    epochs: str = "3",
    split_path: Path = ARTIFACTS / "split.csv",
    fraction: str = "1.0",
    subset_seed: str = "0",
) -> list[str]:
    return [
        "--arch",
        arch,
        *RUN_ORDER[arch][0],
        "--run-dir",
        str(run_dir),
        "--split",
        str(split_path),
        "--train-fraction",
        fraction,
        "--subset-seed",
        subset_seed,
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
    data_root: Path,
    split_path: Path,
    report: Path,
    check: bool = True,
    fraction: str = "1.0",
    summary: Path | None = None,
) -> subprocess.CompletedProcess:
    return run(
        "validate_data.py",
        "--split",
        str(split_path),
        "--train-fraction",
        fraction,
        "--subset-seed",
        "0",
        "--cell-px",
        str(CELL_PX),
        "--report",
        str(report),
        *(["--split-summary", str(summary)] if summary else []),
        data_root=data_root,
        check=check,
    )


def pipeline_check(
    run_dir: Path, data_root: Path, *extra: str, check: bool = True
) -> subprocess.CompletedProcess:
    return run(
        "pipeline_check.py",
        "--run-dir",
        str(run_dir),
        "--split",
        str(ARTIFACTS / "split.csv"),
        "--train-fraction",
        "1.0",
        "--subset-seed",
        "0",
        *extra,
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
        ("unet", {}),
    ):
        model = garimpo_uncertainty.models.build(arch, in_channels=6, **options).eval()
        layers = garimpo_uncertainty.models.feature_layers(model)
        assert len(layers) > 1
        assert layers[-1] == garimpo_uncertainty.models.LOGITS
        first_layer = model.down[0] if arch == "convnet" else model.encoder.conv1
        encoder_passes = []
        first_layer.register_forward_hook(lambda *_, passes=encoder_passes: passes.append(1))
        with torch.no_grad():
            expected = model(x)
            encoder_passes.clear()
            features = garimpo_uncertainty.models.encode(model, x)
            for layer in layers:
                z, decode = garimpo_uncertainty.models.split_at(model, layer, features)
                assert torch.allclose(decode(z), expected, atol=1e-5), f"{arch}/{layer}"
                changed = decode(z + 1.0)
                assert not torch.allclose(changed, expected, atol=1e-3), f"{arch}/{layer} ignored z"
        assert len(encoder_passes) == 1, f"{arch}: g reran the encoder"


def check_validation_catches_defects(root: Path, split_path: Path) -> None:
    report = ARTIFACTS / "validation_clean.json"
    validate(root, split_path, report)
    assert json.loads(report.read_text())["problems"] == []

    broken = ARTIFACTS / "broken_data"
    shutil.copytree(root, broken)
    split = pd.read_csv(split_path)
    cells = garimpo_uncertainty.data.cell_table(root, split, split.tile_id.tolist())
    # One test tile leaks to calibration, another vanishes with a missing scene, one mask changes.
    test_groups = cells[cells.role == "test"].groupby("group").tile_id.unique()
    leaked, vanished = test_groups.iloc[0][0], test_groups.iloc[1][0]
    split.loc[split.tile_id == leaked, "role"] = "calibration"
    broken_split = ARTIFACTS / "split_broken.csv"
    split.to_csv(broken_split, index=False)

    def edit_sidecar(tile_id: int, **changes: object) -> None:
        path = broken / "tiles" / f"{tile_id}.json"
        path.write_text(json.dumps(json.loads(path.read_text()) | changes))

    edit_sidecar(vanished, status="missing_item")
    # A box wider than the window explains its offset, as in tile 684; beyond that it is a defect.
    explained, misaligned = cells[cells.role == "train"].tile_id.unique()[:2]
    edit_sidecar(explained, tile_offset_m=15.6, tile_box_excess_m=31.0)
    edit_sidecar(misaligned, tile_offset_m=26.6, tile_box_excess_m=31.0)
    store = garimpo_uncertainty.data.open_store(broken, "r+")
    calibration = cells[cells.role == "calibration"].cell_index.to_numpy()
    bad_cell = calibration[store.stats["mask_ha"][calibration] > 0][0]
    store.masks[bad_cell] = 0
    store.masks.flush()

    broken_report = ARTIFACTS / "validation_broken.json"
    result = validate(broken, broken_split, broken_report, check=False)
    assert result.returncode != 0
    problems = " ".join(json.loads(broken_report.read_text())["problems"])
    for check in ("group_leakage", "held_out_leak", "missing_scenes", "partial_group", "mask_area"):
        assert check in problems, f"{check} not reported: {problems}"
    assert "tile_offset: 1 cases" in problems, problems


def check_training_artifacts(run_dir: Path, arch: str) -> None:
    history = pd.read_csv(run_dir / "history.csv")
    assert len(history) == 3
    assert {"train_data_wait_seconds", "seconds"} <= set(history.columns)
    assert (run_dir / "last.pt").exists()
    metadata = garimpo_uncertainty.data.read_run_metadata(run_dir)
    for key in ("arch", "seed", "normalization", "split_hash", "root_commit", "compass_commit"):
        assert key in metadata, key
    assert metadata["train_fraction"] == 1.0
    assert metadata["subset_seed"] == 0
    assert metadata["arch"] == arch
    assert metadata["variant"] == MODEL_NAMES[arch]
    assert len(metadata["normalization"]["mean"]) == 6
    assert len(metadata["normalization"]["std"]) == 6


def brute_force_metrics(
    lower: float,
    upper: float,
    v: float,
    alpha: float,
) -> tuple[float, float, float]:
    """Return covered, width and interval score of one cell, with no array arithmetic."""
    covered = 1.0 if lower <= v <= upper else 0.0
    width = upper - lower
    penalty = (2 / alpha) * (max(lower - v, 0.0) + max(v - upper, 0.0))
    return covered, width, width + penalty


def mean_of_group_means(values: list[float], groups: np.ndarray) -> float:
    """Average per-group means, skipping NaN inside a group and groups left with no value."""
    by_group: dict[int, list[float]] = {}
    for value, group in zip(values, groups.tolist(), strict=True):
        if not np.isnan(value):
            by_group.setdefault(group, []).append(value)
    return sum(sum(kept) / len(kept) for kept in by_group.values()) / len(by_group)


def brute_force_dice(root: Path, run_dir: Path, test_index: np.ndarray) -> list[float]:
    masks = np.load(root / "masks.npy", mmap_mode="r")
    probabilities = np.load(run_dir / "predictions" / "test.npy", mmap_mode="r")
    values = []
    for position, cell_index in enumerate(test_index.tolist()):
        reference = masks[cell_index] > 0
        prediction = probabilities[position] >= 128
        if reference.any():
            overlap = int((reference & prediction).sum())
            values.append(2 * overlap / (int(reference.sum()) + int(prediction.sum())))
        else:
            values.append(float("nan"))
    return values


def check_results(run_dir: Path, arch: str, data_root: Path) -> None:
    cells = garimpo_uncertainty.data.read_predictions(data_root, run_dir)
    results = pd.read_csv(run_dir / "metrics" / "results.csv")
    keys = set(zip(results.method, results.calibration, strict=True))
    assert keys == EXPECTED_ROWS
    assert len(results) == len(EXPECTED_ROWS)
    assert (results.model == MODEL_NAMES[arch]).all()
    assert (results.train_fraction == 1.0).all()

    with np.load(run_dir / "intervals" / "intervals.npz") as stored:
        interval_index, lower, upper = stored["cell_index"], stored["lower"], stored["upper"]
        quantile = stored["quantile"]
    test = cells[cells.role == "test"].reset_index(drop=True)
    assert np.array_equal(interval_index, test.cell_index.to_numpy())
    v, vhat, groups = test.v.to_numpy(), test.vhat.to_numpy(), test.group.to_numpy()
    assert (lower >= 0).all()
    assert (lower <= upper).all()
    assert (upper <= test.cell_ha.to_numpy() + 1e-6).all()

    # Array metrics in results.csv against one Python computation per cell, by group means.
    combinations_order = garimpo_uncertainty.intervals.COMBINATIONS
    assert list(zip(results.method, results.calibration, strict=True)) == list(combinations_order)
    dice_by_cell = brute_force_dice(data_root, run_dir, interval_index)
    for k, row in enumerate(results.itertuples()):
        per_cell = [
            brute_force_metrics(float(lower[k, i]), float(upper[k, i]), float(v[i]), ALPHA)
            for i in range(len(v))
        ]
        covered, width, score = zip(*per_cell, strict=True)
        assert np.isclose(row.picp, mean_of_group_means(list(covered), groups)), k
        assert np.isclose(row.mpiw, mean_of_group_means(list(width), groups)), k
        assert np.isclose(row.interval_score, mean_of_group_means(list(score), groups)), k
        assert np.isclose(row.dice, mean_of_group_means(dice_by_cell, groups)), k
        assert row.picp_ci_low <= row.picp_ci_high
        assert row.n_groups == len(np.unique(groups))

    # The quantile must weigh groups equally; the big calibration group makes this bite.
    calibration = cells[cells.role == "calibration"]
    row_of = {combination: k for k, combination in enumerate(combinations_order)}
    scores = np.abs(calibration.v - calibration.vhat).to_numpy()
    expected = pooled_quantile(scores, calibration.group.to_numpy(), 1 - ALPHA)
    simple = row_of[("conformal_simple", "external")]
    assert np.allclose(quantile[simple], expected)
    # Output-space rows are plain arithmetic on the quantile, recomputed here apart from the stage.
    cell_ha = test.cell_ha.to_numpy()
    assert np.allclose(lower[simple], np.clip(vhat - expected, 0, cell_ha))
    assert np.allclose(upper[simple], np.clip(vhat + expected, 0, cell_ha))

    # Own-region calibration must exclude the group being evaluated.
    own = quantile[row_of[("conformal_simple", "own_region")]]
    test_scores = np.abs(v - vhat)
    for group in np.unique(groups):
        others = groups != group
        expected = pooled_quantile(test_scores[others], groups[others], 1 - ALPHA)
        assert np.allclose(own[groups == group], expected), group

    # Exchangeable synthetic regions: conformal must not collapse. Monotone logits hold the point.
    for method in ("conformal_simple", "compass_l"):
        picp = results[(results.method == method) & (results.calibration == "external")].picp.item()
        assert picp >= 0.6, (method, picp)
    logits_row = row_of[("compass_l", "external")]
    assert ((lower[logits_row] <= vhat + 1e-6) & (vhat <= upper[logits_row] + 1e-6)).all()

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
    assert (diagnostics.n_train_cells == (cells.role == "train").sum()).all()

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
    assert (spearman.model == MODEL_NAMES[arch]).all()
    segmentation = read_cell_metrics(run_dir)
    assert np.array_equal(segmentation["cell_index"], interval_index)
    assert np.array_equal(segmentation["dice"], dice_by_cell, equal_nan=True)
    mining = v > 0
    error_by_cell = pd.Series(np.abs(vhat - v)[mining])
    expected = error_by_cell.corr(pd.Series(segmentation["dice"][mining]), method="spearman")
    row = spearman[
        (spearman.area_error == "abs_error_ha") & (spearman.segmentation_quality == "dice")
    ].iloc[0]
    assert np.isclose(row.spearman, expected, equal_nan=True)
    assert (results.n_cells_boundary_by_rule == segmentation["boundary_by_rule"].sum()).all()


def read_cell_metrics(run_dir: Path) -> dict[str, np.ndarray]:
    with np.load(run_dir / "metrics" / "cell_metrics.npz") as stored:
        return {name: stored[name] for name in stored.files}


def check_one_pass_matches_per_combination(
    run_dir: Path, arch: str, layer: str, data_root: Path
) -> None:
    """Evaluate each COMPASS row alone against the stage's one pass: one loader per row.

    Exact equality, no tolerance: same weights, same batches, same CPU code, so any difference is a
    change in what is computed, and the rewrite promised none.
    """
    intervals = garimpo_uncertainty.intervals
    pixel_ha = garimpo_uncertainty.data.PIXEL_HA
    predictions = garimpo_uncertainty.data.read_predictions(data_root, run_dir)
    model, metadata = garimpo_uncertainty.checkpoints.load_model(
        run_dir / "best_model.pt", predictions, "cpu"
    )
    with np.load(run_dir / "intervals" / "intervals.npz") as stored:
        cell_index, stored_lower = stored["cell_index"], stored["lower"]
        stored_upper, quantile = stored["upper"], stored["quantile"]
    test = predictions.iloc[np.searchsorted(predictions.cell_index.to_numpy(), cell_index)]
    with np.load(run_dir / "scores" / "scores.npz") as stored:
        jacobians = stored["jacobians"][np.searchsorted(stored["cell_index"], cell_index)]
    fitted = compass2d.directions.Directions.load(run_dir / "directions" / f"{layer}.npz")
    normalization = metadata["normalization"]
    compass_rows = [
        (k, method)
        for k, (method, _) in enumerate(intervals.COMBINATIONS)
        if method in intervals.COMPASS_METHODS
    ]
    assert len(compass_rows) == 7

    def soft(logits: torch.Tensor) -> torch.Tensor:
        return compass2d.metrics.soft_area(logits, pixel_ha, compass2d.metrics.UPSTREAM_TEMPERATURE)

    def hard(logits: torch.Tensor) -> torch.Tensor:
        return compass2d.metrics.hard_area(logits, pixel_ha)

    alone_lower = np.empty((len(compass_rows), len(test)))
    alone_upper = np.empty((len(compass_rows), len(test)))
    with torch.no_grad():
        for row, (k, method) in enumerate(compass_rows):
            first = 0
            for images, _, _ in garimpo_uncertainty.cell_dataset.loader(
                data_root, test, normalization, 8, 0
            ):
                last = first + len(images)
                features = garimpo_uncertainty.models.encode(model, images)
                beta = torch.as_tensor(quantile[k, first:last])
                if method == "compass_l":
                    logits, _ = garimpo_uncertainty.models.split_at(
                        model, garimpo_uncertainty.models.LOGITS, features
                    )
                    low, high = compass2d.compass.endpoints(
                        lambda zi: zi, hard, logits, torch.ones_like(logits), beta
                    )
                else:
                    z, decode = garimpo_uncertainty.models.split_at(model, layer, features)
                    direction = method.removeprefix("compass_j_")
                    deltas = compass2d.directions.cell_deltas(
                        direction, jacobians[first:last].astype(np.float64), fitted
                    )
                    delta = compass2d.compass.channel_delta(torch.from_numpy(deltas.copy()).to(z))
                    low, high = compass2d.compass.endpoints(decode, soft, z, delta, beta)
                alone_lower[row, first:last] = low.numpy()
                alone_upper[row, first:last] = high.numpy()
                first = last
    alone_lower, alone_upper = intervals.clip_to_cell(
        alone_lower, alone_upper, test.cell_ha.to_numpy()
    )
    rows = [k for k, _ in compass_rows]
    assert np.array_equal(alone_lower, stored_lower[rows])
    assert np.array_equal(alone_upper, stored_upper[rows])

    # The stage's own function, in this process, so its reads and encoder passes can be counted.
    stage = runpy.run_path(str(ROOT / "scripts" / "generate_intervals.py"))
    first_layer = model.down[0] if arch == "convnet" else model.encoder.conv1
    encoder_passes: list[int] = []
    handle = first_layer.register_forward_hook(lambda *_: encoder_passes.append(1))
    reads: list[int] = []
    read_cell = garimpo_uncertainty.cell_dataset.CellDataset.__getitem__

    def counted_read(self: garimpo_uncertainty.cell_dataset.CellDataset, index: int) -> tuple:
        reads.append(index)
        return read_cell(self, index)

    args = argparse.Namespace(
        temperature=compass2d.metrics.UPSTREAM_TEMPERATURE,
        batch_size=8,
        num_workers=0,
        layer=layer,
    )
    lower, upper = np.full_like(stored_lower, np.nan), np.full_like(stored_upper, np.nan)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(garimpo_uncertainty.cell_dataset.CellDataset, "__getitem__", counted_read)
        stage["compass_endpoints"](
            model, quantile, test, jacobians, fitted, data_root, metadata, args, "cpu", lower, upper
        )
    handle.remove()
    assert sorted(reads) == list(range(len(test)))
    assert len(encoder_passes) == int(np.ceil(len(test) / 8))
    clipped_lower, clipped_upper = intervals.clip_to_cell(
        lower[rows], upper[rows], test.cell_ha.to_numpy()
    )
    assert np.array_equal(clipped_lower, stored_lower[rows])
    assert np.array_equal(clipped_upper, stored_upper[rows])


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
    first = read_cell_metrics(run_dir)
    before = file_hashes(run_dir)

    # Without --force a different tolerance must refuse, not silently keep the old table.
    refused = compute_metrics(run_dir, second_tolerance, data_root, check=False)
    assert refused.returncode != 0
    assert not metric_files_only(
        {name for name, value in file_hashes(run_dir).items() if before.get(name) != value}
    )
    kept = read_cell_metrics(run_dir)
    assert np.array_equal(kept["boundary_f1"], first["boundary_f1"], equal_nan=True)
    # A refusal must never read as a passing run.
    report = pipeline_check(run_dir, data_root, check=False)
    assert report.returncode == 1
    assert "failing stage: compute_metrics (refused)" in report.stdout
    assert "PIPELINE CHECK: PASS" not in report.stdout

    compute_metrics(run_dir, second_tolerance, data_root, force=True)
    after = file_hashes(run_dir)
    second = read_cell_metrics(run_dir)

    assert after["best_model.pt"] == before["best_model.pt"]
    assert after["last.pt"] == before["last.pt"]
    assert set(after) == set(before)
    changed = {name for name in after if after[name] != before[name]}
    assert not metric_files_only(changed), metric_files_only(changed)
    assert "metrics/results.csv" in changed

    results = pd.read_csv(run_dir / "metrics" / "results.csv")
    assert (results.boundary_tolerance_px == second_tolerance).all()
    # A wider tolerance can only match more boundary pixels, and Dice must not move at all.
    defined = ~np.isnan(first["boundary_f1"])
    assert np.array_equal(np.isnan(second["boundary_f1"]), ~defined)
    assert (second["boundary_f1"][defined] >= first["boundary_f1"][defined] - 1e-12).all()
    assert (second["boundary_f1"][defined] > first["boundary_f1"][defined]).any()
    assert np.array_equal(first["dice"], second["dice"], equal_nan=True)


def check_cell_dataset_applies_offset(root: Path, split_path: Path) -> None:
    """Baseline 04.00 shifts DN; the dataset must undo it and floor at zero."""
    split = pd.read_csv(split_path)
    cells = garimpo_uncertainty.data.cell_table(root, split, split.tile_id.tolist())
    cells = cells.head(1).assign(boa_offset=-1000)
    image, mask, cell_index = garimpo_uncertainty.cell_dataset.CellDataset(root, cells, None)[0]
    assert cell_index == cells.cell_index.iloc[0]
    store = garimpo_uncertainty.data.open_store(root)
    raw = store.images[cell_index].astype(np.float32)
    assert (raw < 1000).any()
    expected = np.maximum(raw - 1000, 0) / garimpo_uncertainty.data.REFLECTANCE_SCALE
    assert np.allclose(image.numpy(), expected)
    assert np.array_equal(mask[0].numpy() > 0, store.masks[cell_index] > 0)


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


def check_damaged_store_fails(run_dir: Path, data_root: Path, split_path: Path) -> None:
    """Stages may skip on a damaged store, as fingerprints read sidecars only; nothing else may."""
    report = ARTIFACTS / "validation_clean.json"
    assert "skipping" in validate(data_root, split_path, report).stdout

    def check_fails(message: str) -> None:
        result = pipeline_check(run_dir, data_root, check=False)
        assert result.returncode == 1, result.stderr
        assert message in result.stdout
        assert "PIPELINE CHECK: FAIL" in result.stdout
        assert "PIPELINE CHECK: PASS" not in result.stdout

    def problems_after_validating() -> str:
        result = validate(data_root, split_path, report, check=False)
        assert result.returncode != 0
        return " ".join(json.loads(report.read_text())["problems"])

    # Zeroed masks: the metrics stage skips, so the report and the validation are what must see it.
    masks_path = data_root / "masks.npy"
    healthy = masks_path.read_bytes()
    masks = np.load(masks_path, mmap_mode="r+")
    masks[:] = 0
    masks.flush()
    assert "skipping" in compute_metrics(run_dir, 5.0, data_root).stdout
    check_fails("slot_hash")
    assert "slot_hash" in problems_after_validating()
    masks_path.write_bytes(healthy)
    assert "skipping" not in validate(data_root, split_path, report).stdout

    # The store deleted, its sidecars kept: training skips.
    arrays = [data_root / f"{name}.npy" for name in garimpo_uncertainty.data.CellStore._fields]
    for path in arrays:
        path.rename(path.with_suffix(".away"))
    assert "skipping" in run("train.py", *train_args("unet", run_dir), data_root=data_root).stdout
    check_fails("store_missing")
    assert "store_missing" in problems_after_validating()

    # The sidecars gone too: nothing is left to vouch for the run's tiles.
    tiles_dir = data_root / "tiles"
    tiles_dir.rename(data_root / "tiles_away")
    check_fails("store_missing")
    emptied = validate(data_root, split_path, report, check=False)
    assert emptied.returncode != 0
    assert "missing input" in emptied.stderr
    (data_root / "tiles_away").rename(tiles_dir)
    for path in arrays:
        path.with_suffix(".away").rename(path)

    # Sidecars alone gone, with the arrays still there.
    tiles_dir.rename(data_root / "tiles_away")
    check_fails("sidecars_missing")
    (data_root / "tiles_away").rename(tiles_dir)
    validate(data_root, split_path, report)


def check_split_must_match_store(data_root: Path, split_path: Path, runs: Path) -> None:
    """A split that moved a tile must refuse, not read another tile's slots."""
    split = pd.read_csv(split_path)
    test_tile = split[split.role == "test"].tile_id.iloc[0]
    new_tile = split[split.role != "dropped"].tail(1).assign(tile_id=split.tile_id.max() + 1)
    changes = {
        "reordered": (split.iloc[::-1], "tile_index"),
        "dropped": (
            split.assign(role=split.role.mask(split.tile_id == test_tile, "dropped")),
            "tile_index",
        ),
        "added": (pd.concat([split, new_tile]), "cells per tile"),
    }
    for name, (changed, message) in changes.items():
        path = ARTIFACTS / f"split_{name}.csv"
        changed.to_csv(path, index=False)
        report = ARTIFACTS / f"validation_split_{name}.json"
        assert validate(data_root, path, report, check=False).returncode != 0
        assert message in " ".join(json.loads(report.read_text())["problems"]), name

        run_dir = runs / f"convnet_split_{name}"
        args = train_args("convnet", run_dir, epochs="1", split_path=path)
        assert run("train.py", *args, data_root=data_root, check=False).returncode != 0
        record = json.loads((run_dir / "status" / "train.json").read_text())
        assert record["status"] == "failed"
        assert message in record["error"], name
        assert not (run_dir / "best_model.pt").exists()


def check_absent_cell_index_is_named(runs: Path, data_root: Path) -> None:
    """A lookup of an absent cell must name it, where a bare searchsorted returns a neighbor."""
    run_dir = runs / "convnet_absent_cell"
    shutil.copytree(runs / "convnet", run_dir)
    cells = garimpo_uncertainty.data.read_predictions(data_root, run_dir)
    in_test = int(cells[cells.role == "test"].cell_index.iloc[1])
    in_calibration = int(cells[cells.role == "calibration"].cell_index.iloc[1])
    beyond = int(cells.cell_index.max()) + 1_000
    common = ["--run-dir", str(run_dir)]
    calibrate = (
        "calibrate.py",
        "calibrate",
        [*common, "--cells", "all", "--score-floor-ha", "0.01", "--force"],
    )
    generate = (
        "generate_intervals.py",
        "generate_intervals",
        [
            *common,
            "--checkpoint",
            str(run_dir / "best_model.pt"),
            "--layer",
            "up.1",
            "--batch-size",
            "8",
            "--device",
            "cpu",
            "--num-workers",
            "0",
            "--force",
        ],
    )

    def stored(path: Path) -> dict[str, np.ndarray]:
        with np.load(path) as arrays:
            return {name: arrays[name] for name in arrays.files}

    def without(path: Path, value: int) -> dict[str, np.ndarray]:
        arrays = stored(path)
        return arrays | {"cell_index": arrays["cell_index"][arrays["cell_index"] != value]}

    def with_beyond(path: Path) -> dict[str, np.ndarray]:
        arrays = stored(path)
        return arrays | {"cell_index": np.append(arrays["cell_index"][:-1], beyond)}

    scores = run_dir / "scores" / "scores.npz"
    quantiles = run_dir / "calibration" / "quantiles.npz"
    intervals = run_dir / "intervals" / "intervals.npz"
    cases = (
        (scores, without(scores, in_calibration), calibrate, in_calibration),
        (scores, without(scores, in_test), calibrate, in_test),
        (scores, without(scores, in_test), generate, in_test),
        (quantiles, with_beyond(quantiles), generate, beyond),
    )
    for path, broken, (script, stage, args), absent in cases:
        healthy = path.read_bytes()
        np.savez(path, **broken)
        result = run(script, *args, data_root=data_root, check=False)
        assert result.returncode != 0, (script, absent)
        error = json.loads((run_dir / "status" / f"{stage}.json").read_text())["error"]
        assert "cell_index" in error and str(absent) in error, error
        path.write_bytes(healthy)

    healthy = intervals.read_bytes()
    np.savez(intervals, **with_beyond(intervals))
    assert compute_metrics(run_dir, 2.0, data_root, force=True, check=False).returncode != 0
    error = json.loads((run_dir / "status" / "compute_metrics.json").read_text())["error"]
    assert "cell_index" in error and str(beyond) in error, error
    intervals.write_bytes(healthy)


def check_all_nan_dice_fails(data_root: Path, runs: Path) -> None:
    """Give validation cells no mining: NaN every epoch, so no best checkpoint ever exists."""
    no_mining = ARTIFACTS / "data_no_validation_mining"
    shutil.copytree(data_root, no_mining)
    split = pd.read_csv(ARTIFACTS / "split.csv")
    cells = garimpo_uncertainty.data.cell_table(no_mining, split, split.tile_id.tolist())
    masks = np.load(no_mining / "masks.npy", mmap_mode="r+")
    masks[cells[cells.role == "validation"].cell_index.to_numpy()] = 0
    masks.flush()
    run_dir = runs / "convnet_all_nan"
    result = run(
        "train.py", *train_args("convnet", run_dir, epochs="1"), data_root=no_mining, check=False
    )
    assert result.returncode != 0
    record = json.loads((run_dir / "status" / "train.json").read_text())
    assert record["status"] == "failed"
    assert "every val_dice is NaN" in record["error"]
    assert not (run_dir / "best_model.pt").exists()


def write_synthetic_sentinel2_weights(path: Path, seed: int = 0) -> dict[str, torch.Tensor]:
    """Write a file shaped as the MoCo checkpoint: 13-band ResNet-50 keys plus fc."""
    torch.manual_seed(seed)
    encoder = garimpo_uncertainty.models.build("unet", 13).encoder
    state = {key: value.clone() for key, value in encoder.state_dict().items()}
    for column in range(13):
        state["conv1.weight"][:, column] += column / 100
    state["fc.weight"] = torch.zeros(1000, 2048)
    state["fc.bias"] = torch.zeros(1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return state


def test_imagenet_weights_cycle_and_rescale_rgb() -> None:
    shutil.rmtree(IMAGENET_ARTIFACTS, ignore_errors=True)
    IMAGENET_ARTIFACTS.mkdir(parents=True)
    path = IMAGENET_ARTIFACTS / "synthetic.safetensors"
    encoder = garimpo_uncertainty.models.build("unet", 3).encoder
    state = {key: value.clone() for key, value in encoder.state_dict().items()}
    safetensors.torch.save_file(state, path)

    model = garimpo_uncertainty.models.build(
        "unet", 6, encoder_weights="imagenet", weights_file=path
    )
    loaded = model.encoder.state_dict()
    for band in range(6):
        cycled = state["conv1.weight"][:, band % 3] * 0.5
        assert torch.equal(loaded["conv1.weight"][:, band], cycled)
    assert torch.equal(loaded["layer4.2.conv3.weight"], state["layer4.2.conv3.weight"])

    two_bands = state["conv1.weight"][:, :2].clone()
    safetensors.torch.save_file(state | {"conv1.weight": two_bands}, path)
    with pytest.raises(ValueError, match="3 bands"):
        garimpo_uncertainty.models.build("unet", 6, encoder_weights="imagenet", weights_file=path)


def test_sentinel2_weights_keep_our_bands_in_order() -> None:
    shutil.rmtree(WEIGHTS_ARTIFACTS, ignore_errors=True)
    path = WEIGHTS_ARTIFACTS / "synthetic.pth"
    state = write_synthetic_sentinel2_weights(path)

    def build() -> torch.nn.Module:
        return garimpo_uncertainty.models.build(
            "unet", 6, encoder_weights="sentinel2_moco", weights_file=path
        )

    model = build()
    loaded = model.encoder.state_dict()
    for position, column in enumerate(SENTINEL2_MOCO_COLUMNS):
        assert torch.equal(loaded["conv1.weight"][:, position], state["conv1.weight"][:, column])
    copied = set(state) - {"conv1.weight", "fc.weight", "fc.bias"}
    assert copied == set(loaded) - {"conv1.weight"}
    for key in copied:
        assert torch.equal(loaded[key], state[key]), key

    # The layer names stay valid across starting weights, and RUN_ORDER relies on these two.
    scratch = garimpo_uncertainty.models.build("unet", 6)
    assert garimpo_uncertainty.models.feature_layers(
        model
    ) == garimpo_uncertainty.models.feature_layers(scratch)
    assert {"decoder.blocks.3", "decoder.blocks.4"} <= set(
        garimpo_uncertainty.models.feature_layers(scratch)
    )

    # Bottleneck keys, which only a ResNet-50 layout has at this depth.
    assert "layer4.2.conv3.weight" in copied

    missing = {key: value for key, value in state.items() if key != "layer4.2.bn3.weight"}
    unexpected = state | {"layer5.0.conv1.weight": torch.zeros(1)}
    wrong_bands = state | {"conv1.weight": state["conv1.weight"][:, :6]}
    for broken, message in (
        (missing, "Missing key"),
        (unexpected, "Unexpected key"),
        (wrong_bands, "13 bands"),
    ):
        torch.save(broken, path)
        with pytest.raises((RuntimeError, ValueError), match=message):
            build()


def check_encoder_weights_refusals(runs: Path, data_root: Path) -> None:
    run_dir = runs / "refused"
    no_weights = train_args("unet", run_dir)
    cut = no_weights.index("--encoder-weights")
    del no_weights[cut : cut + 2]
    for args, message in (
        ([*train_args("convnet", run_dir), "--encoder-weights", "none"], "invalid for convnet"),
        (no_weights, "required for unet"),
        # The encoder is fixed, so no run can choose another one.
        ([*train_args("unet", run_dir), "--encoder", "resnet50"], "unrecognized arguments"),
    ):
        result = run("train.py", *args, data_root=data_root, check=False)
        assert result.returncode == 2, message
        assert message in result.stderr


def check_sentinel2_training(runs: Path, data_root: Path) -> None:
    weights = garimpo_uncertainty.data.starting_weights_path(data_root, "sentinel2_moco")
    write_synthetic_sentinel2_weights(weights)
    run_dir = runs / "unet_sentinel2"
    args = [*train_args("unet", run_dir, epochs="1"), "--encoder-weights", "sentinel2_moco"]
    run("train.py", *args, data_root=data_root)

    best = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    metadata = best["meta"]
    assert metadata["normalization"] == {"mean": [0.0] * 6, "std": [1.0] * 6}
    assert metadata["encoder"] == "resnet50"
    assert metadata["encoder_weights"] == "sentinel2_moco"
    assert metadata["variant"] == "unet_sentinel2"
    sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert metadata["encoder_weights_sha256"] == sha256

    # A later stage rebuilds the model from the checkpoint alone, without the weights file.
    weights.rename(weights.with_suffix(".away"))
    run(
        "infer.py",
        "--checkpoint",
        str(run_dir / "best_model.pt"),
        "--run-dir",
        str(run_dir),
        "--batch-size",
        "8",
        "--device",
        "cpu",
        "--num-workers",
        "0",
        data_root=data_root,
    )

    # Without the file, as on a machine with runs/ but no weights/: finished training still skips.
    status = run_dir / "status" / "train.json"
    skipped = run("train.py", *args, data_root=data_root)
    assert "skipping" in skipped.stdout
    assert json.loads(status.read_text())["status"] == "ok"

    # Another variant in the same run directory refuses, with the file absent and present.
    other = [*train_args("unet", run_dir, epochs="1"), "--encoder-weights", "none"]
    for _ in ("absent", "present"):
        changed = run("train.py", *other, data_root=data_root, check=False)
        assert changed.returncode != 0
        assert json.loads(status.read_text())["status"] == "refused"
        run("train.py", *args, data_root=data_root)
        assert json.loads(status.read_text())["status"] == "ok"
        if not weights.exists():
            weights.with_suffix(".away").rename(weights)
    weights.rename(weights.with_suffix(".away"))

    # With the file and run_meta.json both gone the identity is unknown: name what restores it.
    run_meta = run_dir / "run_meta.json"
    run_meta.rename(run_meta.with_suffix(".away"))
    unknown = run("train.py", *args, data_root=data_root, check=False)
    assert unknown.returncode != 0
    record = json.loads(status.read_text())
    assert record["status"] == "failed"
    for name in ("run_meta.json", "download_data.py weights"):
        assert name in record["error"], name
    # Restoring it must skip again rather than retrain, since the completed fingerprint stayed.
    run_meta.with_suffix(".away").rename(run_meta)
    assert "skipping" in run("train.py", *args, data_root=data_root).stdout
    assert json.loads(status.read_text())["status"] == "ok"

    # A training that really has to run needs the file, and names the command that fetches it.
    fresh_dir = runs / "unet_sentinel2_no_file"
    fresh = [*train_args("unet", fresh_dir, epochs="1"), "--encoder-weights", "sentinel2_moco"]
    assert run("train.py", *fresh, data_root=data_root, check=False).returncode != 0
    record = json.loads((fresh_dir / "status" / "train.json").read_text())
    assert record["status"] == "failed"
    assert "download_data.py weights" in record["error"]
    weights.with_suffix(".away").rename(weights)

    # An interrupted run has no finished status, so last.pt itself must refuse other weights.
    healthy = status.read_text()
    status.unlink()
    resumed = run("train.py", *other, data_root=data_root, check=False)
    assert resumed.returncode != 0
    record = json.loads(status.read_text())
    assert record["status"] == "failed"
    assert "encoder_weights" in record["error"]
    status.write_text(healthy)

    # A different file under the same name changes the identity of the run.
    write_synthetic_sentinel2_weights(weights, seed=1)
    refused = run("train.py", *args, data_root=data_root, check=False)
    assert refused.returncode != 0
    assert json.loads(status.read_text())["status"] == "refused"


def check_imagenet_absent_file_names_the_command(runs: Path, data_root: Path) -> None:
    run_dir = runs / "unet_imagenet_no_file"
    args = [*train_args("unet", run_dir, epochs="1"), "--encoder-weights", "imagenet"]
    result = run("train.py", *args, data_root=data_root, check=False)
    assert result.returncode != 0
    record = json.loads((run_dir / "status" / "train.json").read_text())
    assert record["status"] == "failed"
    assert "download_data.py weights" in record["error"]


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
    build_dataset(data_root, split_path)

    check_decomposition()
    check_validation_catches_defects(data_root, split_path)
    check_cell_dataset_applies_offset(data_root, split_path)

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
        check_results(run_dir, arch, data_root)
        check_one_pass_matches_per_combination(run_dir, arch, layer, data_root)

    check_training_refusals(runs / "convnet", "convnet", data_root)
    check_all_nan_dice_fails(data_root, runs)
    check_encoder_weights_refusals(runs, data_root)
    check_sentinel2_training(runs, data_root)
    check_imagenet_absent_file_names_the_command(runs, data_root)

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
    assert set(results.model) == {"convnet", "unet_scratch"}

    # Two runs of one variant would merge into one model in every table.
    same_variant = run(
        "merge_results.py",
        "--runs",
        str(runs / "convnet"),
        str(saturated),
        "--out",
        str(ARTIFACTS / "merged_refused"),
        data_root=data_root,
        check=False,
    )
    assert same_variant.returncode != 0
    assert "same model and train_fraction" in same_variant.stderr
    assert len(results) == 2 * len(EXPECTED_ROWS)

    summary = pipeline_check(unet, data_root)
    assert "PIPELINE CHECK: PASS" in summary.stdout
    assert "train_data_wait_seconds" in summary.stdout
    assert "'data_wait_seconds'" in summary.stdout
    (ARTIFACTS / "pipeline_check.txt").write_text(summary.stdout)

    # Every status is ok here, as after an import or argparse error: the exit code must still fail.
    crashed = pipeline_check(unet, data_root, "--job-exit-code", "1", check=False)
    assert crashed.returncode == 1
    assert "PIPELINE CHECK: FAIL" in crashed.stdout
    assert "job exit code 1" in crashed.stdout
    assert "PIPELINE CHECK: PASS" not in crashed.stdout

    # A failed stage must name itself in the report and fail the check.
    status = unet / "status" / "calibrate.json"
    healthy = status.read_text()
    status.write_text(json.dumps({"stage": "calibrate", "status": "failed", "error": "boom"}))
    report = pipeline_check(unet, data_root, check=False)
    assert report.returncode == 1
    assert "failing stage: calibrate" in report.stdout
    assert "boom" in report.stdout
    status.write_text(healthy)

    check_damaged_store_fails(unet, data_root, split_path)
    check_split_must_match_store(data_root, split_path, runs)
    check_absent_cell_index_is_named(runs, data_root)
    validate(data_root, split_path, ARTIFACTS / "validation_clean.json")
    assert "PIPELINE CHECK: PASS" in pipeline_check(unet, data_root).stdout


def test_select_tiles_on_the_real_split() -> None:
    split = pd.read_csv(ROOT / "outputs" / "split.csv")
    summary = json.loads((ROOT / "outputs" / "split_summary.json").read_text())
    others = set(split[split.role.isin(["validation", "calibration", "test"])].tile_id)
    select = garimpo_uncertainty.data.select_tiles

    nested: set[int] = set()
    for fraction, (artisanal, industrial) in REAL_SPLIT_TRAIN_TILES.items():
        selected = select(split, fraction, 0)
        assert selected.equals(select(split, fraction, 0))
        assert "dropped" not in set(selected.role)
        train = selected[selected.role == "train"]
        assert (train.scale == "Artisanal").sum() == artisanal
        assert (train.scale == "Industrial").sum() == industrial
        assert set(selected[selected.role != "train"].tile_id) == others
        assert len(selected) == artisanal + industrial + REAL_SPLIT_OTHER_TILES
        assert nested < set(train.tile_id)
        nested = set(train.tile_id)

        # The counts validation expects come from the summary, the tiles from the split.
        scaled = garimpo_uncertainty.data.scale_summary(summary, fraction)["tiles_by_scale_role"]
        for (scale, role), rows in selected.groupby(["scale", "role"]):
            assert scaled[f"{scale}/{role}"] == len(rows), (fraction, scale, role)
    assert nested == set(split[split.role == "train"].tile_id)

    assert set(select(split, 0.05, 0).tile_id) != set(select(split, 0.05, 1).tile_id)

    # 10 artisanal and 5 industrial training tiles: the minimum holds where rounding gives zero.
    small = select(pd.DataFrame(group_tiles()), 0.01, 0)
    assert small[small.role == "train"].scale.tolist() == ["Artisanal"]


def without_tile(root: Path, tile_id: int, name: str) -> Path:
    """Copy a data root with a tile's sidecar gone, as if never fetched."""
    damaged = FRACTION_ARTIFACTS / name
    shutil.copytree(root, damaged)
    (damaged / "tiles" / f"{tile_id}.json").unlink()
    return damaged


def test_slots_do_not_depend_on_write_order() -> None:
    shutil.rmtree(STORE_ARTIFACTS, ignore_errors=True)
    split = pd.DataFrame(group_tiles())
    tile_ids = garimpo_uncertainty.data.select_tiles(split, 1.0, 0).tile_id.tolist()
    tiles = garimpo_uncertainty.data.tile_table(split)
    order = tiles.set_index("tile_id").tile_index
    assert len(order) == len(tile_ids)
    shuffled = np.random.default_rng(0).permutation(tile_ids).tolist()
    assert shuffled != tile_ids

    def filled(name: str, written: list[int]) -> garimpo_uncertainty.data.CellStore:
        root = STORE_ARTIFACTS / name
        garimpo_uncertainty.download.create_store(root, len(order), TILE_PX, CELL_PX)
        store = garimpo_uncertainty.data.open_store(root, "r+")
        for tile_id in written:
            rng = np.random.default_rng(tile_id)
            write_synthetic_tile(root, store, tile_id, int(order[tile_id]), rng)
        return garimpo_uncertainty.data.open_store(root)

    ascending, reordered = filled("ascending", tile_ids), filled("shuffled", shuffled)
    for name in garimpo_uncertainty.data.CellStore._fields:
        assert getattr(ascending, name).tobytes() == getattr(reordered, name).tobytes(), name
    assert ascending.images.any()

    # The dataset reads the slot its cell_index names, whichever tile reached the disk first.
    root = STORE_ARTIFACTS / "shuffled"
    cells = garimpo_uncertainty.data.cell_table(root, split, tile_ids)
    dataset = garimpo_uncertainty.cell_dataset.CellDataset(root, cells, None)
    for position in (0, len(cells) // 2, len(cells) - 1):
        image, _, cell_index = dataset[position]
        assert cell_index == cells.cell_index.iloc[position]
        expected = ascending.images[cell_index] / garimpo_uncertainty.data.REFLECTANCE_SCALE
        assert np.allclose(image.numpy(), expected)

    # Measured, not asserted: whether holes exist depends on the filesystem under tests/artifacts.
    one_tile = filled("one_tile", tile_ids[:1])
    measures = {}
    for name in garimpo_uncertainty.data.CellStore._fields:
        status = (STORE_ARTIFACTS / "one_tile" / f"{name}.npy").stat()
        measures[name] = {"apparent_bytes": status.st_size, "allocated": status.st_blocks * 512}
    assert one_tile.images.shape[0] == len(order) * CELLS_PER_TILE
    (STORE_ARTIFACTS / "one_tile_allocation.json").write_text(json.dumps(measures, indent=2))


def check_validation_follows_the_selection(
    root: Path, split: pd.DataFrame, split_path: Path, summary: Path
) -> None:
    small = garimpo_uncertainty.data.select_tiles(split, 0.01, 0)
    train = split[(split.role == "train") & (split.scale == "Artisanal")].tile_id
    outside = int(train[~train.isin(small.tile_id)].iloc[0])
    inside = int(train[train.isin(small.tile_id)].iloc[0])

    # A tile outside the selection is no concern of this fraction, and is of the full one.
    no_outside = without_tile(root, outside, "data_no_outside")
    report = FRACTION_ARTIFACTS / "validation_no_outside.json"
    validate(no_outside, split_path, report, fraction="0.01", summary=summary)
    assert json.loads(report.read_text())["problems"] == []
    report = FRACTION_ARTIFACTS / "validation_no_outside_full.json"
    assert validate(no_outside, split_path, report, False, "1.0", summary).returncode != 0
    assert "not_downloaded" in " ".join(json.loads(report.read_text())["problems"])

    no_inside = without_tile(root, inside, "data_no_inside")
    report = FRACTION_ARTIFACTS / "validation_no_inside.json"
    assert validate(no_inside, split_path, report, False, "0.01", summary).returncode != 0
    problems = " ".join(json.loads(report.read_text())["problems"])
    for check in ("not_downloaded", "split_counts"):
        assert check in problems, problems


def test_training_fractions_on_one_data_root() -> None:
    shutil.rmtree(FRACTION_ARTIFACTS, ignore_errors=True)
    FRACTION_ARTIFACTS.mkdir(parents=True)
    root = FRACTION_ARTIFACTS / "data"
    runs = FRACTION_ARTIFACTS / "runs"
    split_path = FRACTION_ARTIFACTS / "split.csv"
    split = pd.DataFrame(group_tiles())
    split.to_csv(split_path, index=False)
    counts = split.groupby(["scale", "role"]).size()
    summary = FRACTION_ARTIFACTS / "split_summary.json"
    summary.write_text(
        json.dumps({"tiles_by_scale_role": {f"{s}/{r}": int(n) for (s, r), n in counts.items()}})
    )
    small = garimpo_uncertainty.data.select_tiles(split, 0.01, 0)
    full = garimpo_uncertainty.data.select_tiles(split, 1.0, 0)
    rng = np.random.default_rng(0)
    fill_tiles(root, split, small.tile_id.tolist(), rng)

    _, layer, other_layer = RUN_ORDER["convnet"]
    small_dir = runs / "convnet-f0.01"
    status = small_dir / "status" / "train.json"
    report = FRACTION_ARTIFACTS / "validation_small.json"

    # One tile is 4 cells. With few updates no logit nears zero, and at the upstream temperature
    # every Jacobian is then exactly zero, which fit_directions rightly refuses.
    def small_args(fraction: str = "0.01", subset_seed: str = "0") -> list[str]:
        return train_args(
            "convnet",
            small_dir,
            batch_size="2",
            lr="1e-2",
            epochs="40",
            split_path=split_path,
            fraction=fraction,
            subset_seed=subset_seed,
        )

    def finish_run(run_dir: Path) -> None:
        run_stages_after_training(run_dir, layer, [layer, other_layer], root)
        compute_metrics(run_dir, 2.0, root)

    validate(root, split_path, report, fraction="0.01", summary=summary)
    run("train.py", *small_args(), data_root=root)
    finish_run(small_dir)

    # The larger fraction's data arrives: earlier slots stay byte-identical, the run cannot care.
    small_ids = small.tile_id.tolist()
    small_index = garimpo_uncertainty.data.cell_table(root, split, small_ids).cell_index.to_numpy()
    assert len(small_index) == len(small) * CELLS_PER_TILE

    def small_slots() -> dict[str, bytes]:
        store = garimpo_uncertainty.data.open_store(root)
        return {name: getattr(store, name)[small_index].tobytes() for name in store._fields}

    before = small_slots()
    sizes = {path.name: path.stat().st_size for path in root.glob("*.npy")}
    identity = garimpo_uncertainty.data.tiles_identity(root, split, small_ids)
    fill_tiles(root, split, full.tile_id[~full.tile_id.isin(small.tile_id)].tolist(), rng)
    assert small_slots() == before
    assert {path.name: path.stat().st_size for path in root.glob("*.npy")} == sizes
    assert garimpo_uncertainty.data.tiles_identity(root, split, small_ids) == identity
    grown = garimpo_uncertainty.data.cell_table(root, split, full.tile_id.tolist())
    assert len(grown) > len(small_index)
    finished = file_hashes(small_dir)
    assert "skipping" in run("train.py", *small_args(), data_root=root).stdout
    assert json.loads(status.read_text())["status"] == "ok"
    # A refusal would raise here, and a redone stage would change a file.
    finish_run(small_dir)
    assert file_hashes(small_dir) == finished
    rerun = validate(root, split_path, report, fraction="0.01", summary=summary)
    assert "skipping" in rerun.stdout
    # The slot hashes cover this run's tiles only, so the others arriving must keep it passing.
    report_args = ["--run-dir", str(small_dir), "--split", str(split_path)]
    report_args += ["--train-fraction", "0.01", "--subset-seed", "0"]
    assert "PIPELINE CHECK: PASS" in run("pipeline_check.py", *report_args, data_root=root).stdout

    # Another fraction or subset under the same RUN_DIR refuses, and the original then skips.
    for args in (small_args(fraction="1.0"), small_args(subset_seed="1")):
        refused = run("train.py", *args, data_root=root, check=False)
        assert refused.returncode != 0
        assert json.loads(status.read_text())["status"] == "refused"
        assert "skipping" in run("train.py", *small_args(), data_root=root).stdout

    # An interrupted run has no finished status, so last.pt itself must refuse another fraction.
    healthy = status.read_text()
    status.unlink()
    resumed = run("train.py", *small_args(fraction="1.0"), data_root=root, check=False)
    assert resumed.returncode != 0
    assert "['train_fraction']" in json.loads(status.read_text())["error"]
    status.write_text(healthy)

    # The identity follows the run's own rows: a re-cut of one of its tiles must still refuse.
    tile = int(small[small.role == "train"].tile_id.iloc[0])
    sidecar = root / "tiles" / f"{tile}.json"
    original = sidecar.read_text()
    sidecar.write_text(json.dumps(json.loads(original) | {"resampling_20m": "nearest"}))
    assert run("train.py", *small_args(), data_root=root, check=False).returncode != 0
    assert json.loads(status.read_text())["status"] == "refused"
    sidecar.write_text(original)
    assert "skipping" in run("train.py", *small_args(), data_root=root).stdout

    check_validation_follows_the_selection(root, split, split_path, summary)

    full_dir = runs / "convnet-f1.0"
    validate(root, split_path, FRACTION_ARTIFACTS / "validation_full.json", summary=summary)
    full_args = train_args("convnet", full_dir, split_path=split_path, fraction="1.0")
    run("train.py", *full_args, data_root=root)
    finish_run(full_dir)

    # Every fraction is evaluated on the same cells; only the training cells differ.
    small_cells = garimpo_uncertainty.data.read_predictions(root, small_dir)
    full_cells = garimpo_uncertainty.data.read_predictions(root, full_dir)
    small_eval = small_cells[small_cells.role != "train"].cell_index.tolist()
    assert small_eval == full_cells[full_cells.role != "train"].cell_index.tolist()
    small_train = set(small_cells[small_cells.role == "train"].cell_index)
    assert small_train < set(full_cells[full_cells.role == "train"].cell_index)
    assert len(small_train) == CELLS_PER_TILE
    diagnostics = pd.read_csv(small_dir / "directions" / "diagnostics.csv")
    assert (diagnostics.n_train_cells == CELLS_PER_TILE).all()

    # Normalization comes from the run's own training tiles alone.
    small_meta = garimpo_uncertainty.data.read_run_metadata(small_dir)
    full_meta = garimpo_uncertainty.data.read_run_metadata(full_dir)
    own = garimpo_uncertainty.cell_dataset.band_statistics(
        root, small_cells[small_cells.role == "train"], 8, 0
    )
    assert np.allclose(small_meta["normalization"]["mean"], own["mean"])
    assert np.allclose(small_meta["normalization"]["std"], own["std"])
    assert small_meta["normalization"] != full_meta["normalization"]
    assert small_meta["split_hash"] != full_meta["split_hash"]
    assert small_meta["train_fraction"] == 0.01
    assert full_meta["train_fraction"] == 1.0

    # The merge keeps the fractions apart and never compares across them.
    merged = FRACTION_ARTIFACTS / "merged"
    run(
        "merge_results.py",
        "--runs",
        str(small_dir),
        str(full_dir),
        "--out",
        str(merged),
        data_root=root,
    )
    results = pd.read_csv(merged / "results.csv")
    assert results.groupby(["model", "train_fraction"]).size().to_dict() == {
        ("convnet", 0.01): len(EXPECTED_ROWS),
        ("convnet", 1.0): len(EXPECTED_ROWS),
    }
    pairs = len(list(combinations(EXPECTED_ROWS, 2)))
    for table in ("width_comparisons.csv", "coverage_comparisons.csv"):
        sizes = pd.read_csv(merged / table).groupby("train_fraction").size()
        assert sizes.to_dict() == {0.01: pairs, 1.0: pairs}, table
    for table in ("q1_spearman.csv", "layer_diagnostics.csv", "monotonicity_endpoint.csv"):
        assert set(pd.read_csv(merged / table).train_fraction) == {0.01, 1.0}, table

    # Two runs of one model at one fraction would merge into a single row.
    twin = runs / "convnet-f0.01-twin"
    twin.mkdir()
    shutil.copy(small_dir / "run_meta.json", twin / "run_meta.json")
    duplicate = run(
        "merge_results.py",
        "--runs",
        str(small_dir),
        str(twin),
        "--out",
        str(FRACTION_ARTIFACTS / "merged_refused"),
        data_root=root,
        check=False,
    )
    assert duplicate.returncode != 0
    assert "same model and train_fraction" in duplicate.stderr


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
    tile_ids = selected.tile_id.tolist()
    download.download_scenes(root, split, selected, 2, bilinear)

    sidecar_path = root / "tiles" / "0.json"
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["status"] == "ok"
    assert sidecar["resampling_20m"] == "bilinear"
    assert sidecar["boa_offset"] == -1000
    assert sidecar["tile_offset_m"] > garimpo_uncertainty.data.PIXEL_M
    assert np.isclose(sidecar["tile_offset_m"], 15.5, atol=0.01)
    assert np.isclose(sidecar["tile_box_excess_m"], 31.0, atol=0.01)
    # The tile table georeferences a slot: the window starts 50 pixels into the scene.
    assert sidecar["transform"][2] == SCENE_ORIGIN[0] + 500
    assert sidecar["transform"][5] == SCENE_ORIGIN[1] - 500
    assert json.loads((root / "tiles" / "1.json").read_text())["status"] == "missing_item"
    store = garimpo_uncertainty.data.open_store(root)
    assert np.array_equal(store.images[0, 0], position[50:562, 50:562])
    assert (store.scl[0] == 9).all()

    # The real offset above must pass, and the missing industrial tile must not fail the run.
    written = garimpo_uncertainty.data.cell_table(root, split, tile_ids)
    assert written.cell_index.tolist() == list(range(16))
    assert np.isclose(store.stats["mask_ha"][:16].sum(), mask.area / 1e4, rtol=0.01)
    counts = {"Artisanal/test": 1, "Industrial/train": 1, "Industrial/dropped": 5}
    report, cells = garimpo_uncertainty.validation.validate(
        root, split, tile_ids, cell_px, {"tiles_by_scale_role": counts}, 2
    )
    assert report["problems"] == []
    assert report["industrial_tiles_excluded"] == 1
    assert report["diagnostics"]["by_role"]["test"]["cells_100_percent_cloud"] == 1
    assert report["diagnostics"]["by_role"]["test"]["cells_100_percent_nodata"] == 1

    tiles = garimpo_uncertainty.data.tile_records(root, split, tile_ids)
    for wrong in ({"Artisanal/test": 2}, {"Industrial/train": 2}):
        problems = garimpo_uncertainty.validation.check_split_counts(
            tiles, {"tiles_by_scale_role": wrong}
        )
        assert problems and problems[0].startswith("split_counts")

    # An outage is not a missing tile: it must stop the run instead of shrinking the training set.
    industrial = root / "tiles" / "1.json"
    healthy_sidecar = industrial.read_text()
    industrial.write_text(json.dumps(json.loads(healthy_sidecar) | {"status": "http_503"}))
    report, _ = garimpo_uncertainty.validation.validate(
        root, split, tile_ids, cell_px, {"tiles_by_scale_role": counts}, 2
    )
    assert "industrial_download_errors" in " ".join(report["problems"])
    assert "split_counts" in " ".join(report["problems"])
    assert report["industrial_tiles_excluded"] == 0
    industrial.write_text(healthy_sidecar)

    # An unchanged rerun touches neither the sidecar nor the identity later stages check.
    identity = garimpo_uncertainty.data.tiles_identity(root, split, tile_ids)
    sidecar_time = sidecar_path.stat().st_mtime_ns
    download.download_scenes(root, split, selected, 2, bilinear)
    assert sidecar_path.stat().st_mtime_ns == sidecar_time
    assert garimpo_uncertainty.data.tiles_identity(root, split, tile_ids) == identity

    # Another resampling must refuse, and a forced re-cut must reach later stages.
    with pytest.raises(ValueError, match="resampling"):
        download.download_scenes(root, split, selected, 2, nearest)
    assert sidecar_path.stat().st_mtime_ns == sidecar_time
    b02_before, b11_before = np.array(store.images[0, 0]), np.array(store.images[0, 4])

    # A crash after a re-cut's slots and before its sidecar leaves no `ok` vouching for them.
    class Crash(BaseException):
        pass

    write_cells = download.write_cells

    def write_then_crash(*args: object) -> None:
        write_cells(*args)
        raise Crash

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(download, "write_cells", write_then_crash)
        with pytest.raises(Crash):
            download.download_scenes(root, split, selected, 2, nearest, force=True)
    assert (store.images[0, 4] != b11_before).any()
    assert not sidecar_path.exists()
    assert garimpo_uncertainty.data.cell_table(root, split, tile_ids).empty

    # The rerun then finishes that tile, as any pending tile, and the identity moves.
    download.download_scenes(root, split, selected, 2, nearest)
    assert json.loads(sidecar_path.read_text())["resampling_20m"] == "nearest"
    assert garimpo_uncertainty.data.tiles_identity(root, split, tile_ids) != identity
    assert np.array_equal(store.images[0, 0], b02_before)
    assert (store.images[0, 4] != b11_before).any()
    recut = sidecar_path.stat().st_mtime_ns
    download.download_scenes(root, split, selected, 2, nearest, force=True)
    assert sidecar_path.stat().st_mtime_ns != recut

    # A missing artisanal tile and a window off the scene both fail the validation.
    download.download_scenes(root, split, split, 2, nearest)
    assert json.loads((root / "tiles" / "2.json").read_text())["status"] == "missing_item"
    assert json.loads((root / "tiles" / "3.json").read_text())["status"] == "error_ValueError"
    report, _ = garimpo_uncertainty.validation.validate(
        root, split, split.tile_id.tolist(), cell_px, None, 2
    )
    problems = " ".join(report["problems"])
    assert "missing_scenes: 2 cases" in problems, problems
    assert "partial_group" in problems, problems

    # A sidecar binds its tile to its slot and content.
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["tile_index"] == 0
    assert sidecar["cells_per_tile"] == 16
    assert sidecar["slots_sha256"] == garimpo_uncertainty.data.slots_sha256(store, slice(0, 16))

    # A lost array must not come back as zeros under sidecars still saying `ok`, even with --force.
    lost = root / "masks.npy"
    away = lost.with_suffix(".away")
    lost.rename(away)
    for force in (False, True):
        with pytest.raises(ValueError, match=r"masks\.npy.*status ok"):
            download.download_scenes(root, split, selected, 2, nearest, force=force)
        assert not lost.exists()
    away.rename(lost)

    # With no `ok` sidecar nothing vouches for the array, as after a crash during first creation.
    fresh = DOWNLOAD_ARTIFACTS / "fresh"
    download.create_store(fresh, 1, tile_px, cell_px)
    (fresh / "masks.npy").unlink()
    download.create_store(fresh, 1, tile_px, cell_px)
    assert (fresh / "masks.npy").exists()

    # A split that moved or added tiles must not download into another tile's slots.
    moved = split.iloc[::-1]
    with pytest.raises(ValueError, match="tile_index"):
        download.download_scenes(root, moved, moved, 2, nearest)
    added = pd.concat([split, split.tail(1).assign(tile_id=99)])
    with pytest.raises(ValueError, match="holds 64 cells"):
        download.download_scenes(root, added, added, 2, nearest)
