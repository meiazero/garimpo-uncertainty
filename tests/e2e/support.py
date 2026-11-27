"""Synthetic data, stage commands and literal expectations shared by the end-to-end tests."""

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import rasterio.features
import rasterio.transform
import shapely
import torch

import garimpo_uncertainty.artifacts
import garimpo_uncertainty.data
import garimpo_uncertainty.download
import garimpo_uncertainty.models
import garimpo_uncertainty.pipeline.commands

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "tests" / "artifacts" / "e2e"
RUNS = ARTIFACTS / "runs"
SPLIT_PATH = ARTIFACTS / "split.csv"
CLEAN_REPORT = ARTIFACTS / "validation_clean.json"
# The smallest cell a depth-5 encoder accepts, to offset the cost of the encoder on CPU.
TILE_PX = 64
CELL_PX = 32
CELLS_PER_TILE = (TILE_PX // CELL_PX) ** 2
EPSG = 32721
BOUNDARY_TOLERANCE = 2.0
# Cheung et al. (2026), Table 1, kept apart from the pipeline's constant.
ALPHAS = (0.05, 0.1, 0.15)
# 80 samples at batch 8, more than the 60 training cells: an epoch is updates, not a pass.
STEPS_PER_EPOCH = "10"
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
    ("compass_j_pca", "external_weighted"),
    ("compass_j_pca", "own_region"),
    ("compass_j_svm", "external"),
    ("compass_j_svm", "external_weighted"),
    ("compass_j_svm", "own_region"),
}
# Mondrian by predicted mining, one literal row per method and situation.
EXPECTED_ROWS |= {
    (method, calibration)
    for method in (
        "conformal_simple",
        "conformal_local",
        "cqr",
        "compass_l",
        "compass_j_pca",
        "compass_j_svm",
    )
    for calibration in ("external_mondrian", "own_region_mondrian")
}
# Asymmetric COMPASS through the joint score.
EXPECTED_ROWS |= {
    (method, calibration)
    for method in ("a_compass_l", "a_compass_j_pca", "a_compass_j_svm")
    for calibration in ("external", "own_region")
}
# The sensitivity analysis: every pooled external and own-region row again, by repeated subsampling.
EXPECTED_ROWS |= {
    (method, f"{calibration}_subsampled")
    for method, calibration in EXPECTED_ROWS
    if calibration in {"external", "own_region"}
}
# Each architecture's starting weights, then its layers in the order the runs list them.
RUN_ORDER = {
    "convnet": (None, "decoder.relu", "decoder.aspp"),
    "unet": ("none", "decoder.blocks.4", "decoder.blocks.3"),
}
MODEL_NAMES = {"convnet": "convnet", "unet": "unet_scratch"}
# GARIMPO_TEST_DEVICE=cuda runs the suite as production runs on a GPU: no --device, so the stages
# pick it through checkpoints.default_device, and two loader workers. See docs/testing.md.
DEVICE = os.environ.get("GARIMPO_TEST_DEVICE", "cpu")
if DEVICE not in {"cpu", "cuda"}:
    raise ValueError(f"GARIMPO_TEST_DEVICE is cpu or cuda, not {DEVICE!r}")
if DEVICE == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("GARIMPO_TEST_DEVICE=cuda, but torch sees no GPU")
DEVICE_ARGS = ["--device", "cpu"] if DEVICE == "cpu" else []
NUM_WORKERS = "0" if DEVICE == "cpu" else "2"
# Every stage runs torch's deterministic algorithms, but no GPU run has yet shown a resumed or
# repeated training equal to the bit, so on a GPU the norm of the whole difference is compared: a
# lost state or another batch moves every weight. Adam's first steps move a weight by about lr times
# the sign of its gradient, so rounding noise alone would move single weights the other way.
GPU_RELATIVE_DIFFERENCE = 1e-3
# The `garimpo` command itself, then a check that it never imported torch.
TORCH_FREE = (
    "import runpy, sys; sys.argv = ['garimpo', *sys.argv[1:]]; "
    "runpy.run_module('garimpo_uncertainty', run_name='__main__'); "
    "assert 'torch' not in sys.modules, 'metrics stage imported torch'"
)


class FinishedRun(NamedTuple):
    arch: str
    run_dir: Path
    layer: str


@contextmanager
def restored_after(directory: Path) -> Iterator[Path]:
    """Let a scenario damage a shared directory, then bring back its bytes, path and mtimes.

    A copy under another path would not do: stage fingerprints record paths and mtimes, so it could
    never skip as the original does.
    """
    backup = directory.with_name(f"{directory.name}.backup")
    shutil.copytree(directory, backup)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)
        backup.rename(directory)


def build_once(name: str, build: Callable[[], object]) -> None:
    """Run `build` in the first worker to ask; the others wait here, then share what it wrote.

    A failed build is final: retrying over its partial files would hide the error or resume from
    them.
    """
    built = ARTIFACTS / "built"
    built.mkdir(parents=True, exist_ok=True)
    with (built / f"{name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (built / f"{name}.failed").exists():
            raise RuntimeError(f"building {name} failed in another test; see its error")
        if not (built / name).exists():
            try:
                build()
            except Exception:
                (built / f"{name}.failed").touch()
                raise
            (built / name).touch()


def finish_run(arch: str, run_dir: Path, data_root: Path) -> FinishedRun:
    run("train", *train_args(arch, run_dir), data_root=data_root)
    layers = list(RUN_ORDER[arch][1:])
    layer = run_stages_after_training(run_dir, layers, data_root)
    compute_metrics(run_dir, BOUNDARY_TOLERANCE, data_root)
    run_other_layers(run_dir, layers, data_root)
    run("search_profile", *stage_args("search_profile", run_dir), data_root=data_root)
    return FinishedRun(arch, run_dir, layer)


def run_other_layers(run_dir: Path, layers: list[str], data_root: Path) -> None:
    """Run the layer stages on every other layer, into layers/<layer>/, as `garimpo run` does."""
    for layer in layers:
        if layer != chosen_layer(run_dir):
            layer_dir = run_dir / garimpo_uncertainty.artifacts.LAYERS_DIR / layer
            run_scores_and_intervals(
                run_dir, layer, data_root, beta_range="100", threshold="1e-3", layer_dir=layer_dir
            )
            compute_metrics(run_dir, BOUNDARY_TOLERANCE, data_root, layer_dir=layer_dir)


def chosen_layer(run_dir: Path) -> str:
    return json.loads((run_dir / "chosen_layer.json").read_text())["layer"]


def group_tiles() -> list[dict]:
    """Make one calibration group much larger, so group weights differ from cell weights."""
    plan = [("Asia/Oceania", "train", g, 2) for g in range(5)]
    plan += [("Africa/Europe", "validation", 5, 2)]
    plan += [("Asia/Oceania", "calibration", 6, 6)]
    plan += [("Asia/Oceania", "calibration", g, 1) for g in range(7, 14)]
    plan += [("Amazonia BR", "test", g, 2) for g in range(14, 22)]
    # Repeated subsampling needs k > 1/alpha - 1 groups, 19 at alpha 0.05, and own region leaves
    # one out: 20 calibration and 21 test groups.
    plan += [("Asia/Oceania", "calibration", g, 1) for g in range(22, 34)]
    plan += [("Amazonia BR", "test", g, 1) for g in range(34, 47)]
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
        garimpo_uncertainty.download.CutTile(
            np.clip(image, 1, 65535).astype(np.uint16), scl, geometry, transform
        ),
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


def garimpo(command: str, *args: str) -> list[str]:
    """Return `garimpo <command>` as the runner starts it, in a process of its own."""
    return [sys.executable, "-m", "garimpo_uncertainty", command, *args]


def run(
    script: str, *args: str, data_root: Path, check: bool = True
) -> subprocess.CompletedProcess:
    env = {**os.environ, "GARIMPO_DATA_DIR": str(data_root)}
    command = garimpo(script, *args)
    result = subprocess.run(  # noqa: S603 - arguments come from the tests
        command, check=False, cwd=ROOT, env=env, capture_output=True, text=True
    )
    # CalledProcessError's message omits stderr, which is all CI shows of a failing stage.
    if check and result.returncode != 0:
        raise AssertionError(f"{script} exited {result.returncode}:\n{result.stderr[-4000:]}")
    return result


def assert_same_training_state(
    actual: dict[object, torch.Tensor], expected: dict[object, torch.Tensor]
) -> None:
    """Compare named tensors exactly on CPU; on a GPU, integers exactly and floats as one vector."""
    assert actual.keys() == expected.keys()
    if DEVICE == "cpu":
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return
    floats = [name for name, tensor in expected.items() if tensor.is_floating_point()]
    for name in expected.keys() - set(floats):
        assert torch.equal(actual[name], expected[name]), name
    vectors = [
        torch.cat([d[name].double().flatten() for name in floats]) for d in (actual, expected)
    ]
    difference = float(torch.linalg.vector_norm(vectors[0] - vectors[1]) / vectors[1].norm())
    assert difference <= GPU_RELATIVE_DIFFERENCE, f"relative difference {difference:.2e}"


def train_args(
    arch: str,
    run_dir: Path,
    batch_size: str = "8",
    lr: str = "1e-3",
    epochs: str = "3",
    *,
    split_path: Path = SPLIT_PATH,
    fraction: str = "1.0",
    subset_seed: str = "0",
) -> list[str]:
    values = {
        "ARCH": arch,
        "ENCODER_WEIGHTS": RUN_ORDER[arch][0],
        "RUN_DIR": run_dir,
        "TRAIN_FRACTION": fraction,
        "SUBSET_SEED": subset_seed,
        "EVALUATION_TILES": "all",
        "EPOCHS": epochs,
        "STEPS_PER_EPOCH": STEPS_PER_EPOCH,
        "BATCH_SIZE": batch_size,
        "LR": lr,
        "WEIGHT_DECAY": "0",
        "LOSS": "dice_bce",
        "SEED": "0",
        "NUM_WORKERS": NUM_WORKERS,
    }
    arguments = garimpo_uncertainty.pipeline.commands.arguments("train", values)
    return [*arguments, "--split", str(split_path), *DEVICE_ARGS]


# The values the stages take from a run's config, as these tests give them to every run.
STAGE_VALUES = {
    "TRAIN_FRACTION": "1.0",
    "SUBSET_SEED": "0",
    "CELL_PX": CELL_PX,
    "EVAL_BATCH_SIZE": "8",
    "NUM_WORKERS": NUM_WORKERS,
    "SEED": "0",
    "DIRECTION_TEMPERATURE": "10000",
    "SVM_C": "1.0",
    "SVM_CLASS_WEIGHT": "balanced",
    "SVM_PIXELS_PER_CELL": "50",
    # The sweep's beta_max is the search range of the scores, as in a run's config.
    "BETA_RANGE_LOGITS": "100",
    "BETA_RANGE_FEATURES": "100",
    "SWEEP_STEPS": "9",
    "SWEEP_CELLS": "6",
    # Every layer qualifies, as in the pilot: a few epochs on synthetic tiles need not be monotonic.
    "MIN_FRACTION_ASCENDING": "0",
    "SEARCH_THRESHOLD_LOGITS": "0.01",
    "SEARCH_THRESHOLD_FEATURES": "1e-3",
    "CELLS": "all",
    "EVALUATION_TILES": "all",
    "SCORE_FLOOR_HA": "0.01",
    "BOUNDARY_TOLERANCE_PX": BOUNDARY_TOLERANCE,
    "LAYER_DIR": None,
}


def stage_args(stage: str, run_dir: Path, **changes: object) -> list[str]:
    """Return the flags `stage` of `pipeline.commands` takes, as `garimpo run` passes them."""
    values = STAGE_VALUES | {
        "RUN_DIR": run_dir,
        "CHECKPOINT": garimpo_uncertainty.artifacts.RunPaths.of(run_dir).best_model,
    }
    spec = garimpo_uncertainty.pipeline.commands.STAGES[stage]
    device = DEVICE_ARGS if ("--checkpoint", "CHECKPOINT") in spec.flags else []
    return [*garimpo_uncertainty.pipeline.commands.arguments(stage, values | changes), *device]


def checkpoint_args(run_dir: Path, batch_size: str = "8") -> list[str]:
    """Return the flags every stage reading the trained model takes, as the runner passes them."""
    return stage_args("infer", run_dir, EVAL_BATCH_SIZE=batch_size)


def validate(
    data_root: Path,
    split_path: Path,
    report: Path,
    check: bool = True,
    fraction: str = "1.0",
    *,
    summary: Path | None = None,
) -> subprocess.CompletedProcess:
    values = STAGE_VALUES | {
        "TRAIN_FRACTION": fraction,
        "VALIDATION_REPORT": report,
        "SPLIT_SUMMARY": summary,
    }
    arguments = garimpo_uncertainty.pipeline.commands.arguments("validate_data", values)
    return run(
        "validate_data", *arguments, "--split", str(split_path), data_root=data_root, check=check
    )


def pipeline_check(
    run_dir: Path, data_root: Path, *extra: str, check: bool = True
) -> subprocess.CompletedProcess:
    return run(
        "check",
        "--run-dir",
        str(run_dir),
        "--split",
        str(SPLIT_PATH),
        "--train-fraction",
        "1.0",
        "--subset-seed",
        "0",
        "--evaluation-tiles",
        "all",
        *extra,
        data_root=data_root,
        check=check,
    )


def compute_metrics(
    run_dir: Path,
    tolerance: float,
    data_root: Path,
    force: bool = False,
    check: bool = True,
    *,
    layer_dir: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run the metrics stage while asserting that it never imported torch."""
    args = stage_args(
        "compute_metrics", run_dir, BOUNDARY_TOLERANCE_PX=tolerance, LAYER_DIR=layer_dir
    )
    if force:
        args.append("--force")
    env = {**os.environ, "GARIMPO_DATA_DIR": str(data_root)}
    command = [sys.executable, "-c", TORCH_FREE, "compute_metrics", *args]
    return subprocess.run(command, cwd=ROOT, env=env, check=check, capture_output=True, text=True)  # noqa: S603 - arguments come from the tests


def run_stages_after_training(
    run_dir: Path, layers: list[str], data_root: Path, **changes: object
) -> str:
    """Run inference through the intervals as `garimpo run` does; return the chosen layer."""
    run("infer", *checkpoint_args(run_dir), data_root=data_root)
    run(
        "fit_directions",
        *stage_args("fit_directions", run_dir, LAYERS=layers, **changes),
        data_root=data_root,
    )
    sweep(run_dir, "layer", layers, data_root, **changes)
    run("choose_layer", *stage_args("choose_layer", run_dir), data_root=data_root)
    layer = chosen_layer(run_dir)
    sweep(run_dir, "endpoint", [layer], data_root, **changes)
    run_scores_and_intervals(
        run_dir, layer, data_root, beta_range="100", threshold="1e-3", **changes
    )
    return layer


def sweep(run_dir: Path, use: str, layers: list[str], data_root: Path, **changes: object) -> None:
    """Algorithm 2, with beta_max equal to the search range below, as `garimpo run` passes it."""
    stage = f"{use}_monotonicity"
    arguments = stage_args(stage, run_dir, LAYERS=layers, LAYER=layers, **changes)
    run(
        garimpo_uncertainty.pipeline.commands.STAGES[stage].command, *arguments, data_root=data_root
    )


def run_scores_and_intervals(
    run_dir: Path,
    layer: str,
    data_root: Path,
    beta_range: str,
    threshold: str,
    *,
    layer_dir: Path | None = None,
    **changes: object,
) -> None:
    values = {
        "LAYER": layer,
        "LAYER_DIR": layer_dir,
        "BETA_RANGE_FEATURES": beta_range,
        "SEARCH_THRESHOLD_FEATURES": threshold,
    }
    for stage in ("compass_scores", "calibrate", "generate_intervals"):
        run(stage, *stage_args(stage, run_dir, **values, **changes), data_root=data_root)


def file_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def read_cell_metrics(run_dir: Path) -> dict[str, np.ndarray]:
    with np.load(run_dir / "metrics" / "cell_metrics.npz") as stored:
        return {name: stored[name] for name in stored.files}


def write_changed_split(name: str, directory: Path) -> tuple[Path, str]:
    """Write a split that moved a tile; return its path and the message that must refuse it."""
    split = pd.read_csv(SPLIT_PATH)
    test_tile = split[split.role == "test"].tile_id.iloc[0]
    new_tile = split[split.role != "dropped"].tail(1).assign(tile_id=split.tile_id.max() + 1)
    changed, message = {
        "reordered": (split.iloc[::-1], "tile_index"),
        "dropped": (
            split.assign(role=split.role.mask(split.tile_id == test_tile, "dropped")),
            "tile_index",
        ),
        "added": (pd.concat([split, new_tile]), "cells per tile"),
    }[name]
    path = directory / f"split_{name}.csv"
    changed.to_csv(path, index=False)
    return path, message


def write_synthetic_sentinel2_weights(path: Path, seed: int = 0) -> dict[str, torch.Tensor]:
    """Write a file shaped as the MoCo checkpoint: 13-band ResNet-18 keys plus fc."""
    torch.manual_seed(seed)
    encoder = garimpo_uncertainty.models.build("unet", 13).encoder
    state = {key: value.clone() for key, value in encoder.state_dict().items()}
    for column in range(13):
        state["conv1.weight"][:, column] += column / 100
    state["fc.weight"] = torch.zeros(1000, 512)
    state["fc.bias"] = torch.zeros(1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return state


def pooled_quantile(scores: np.ndarray, groups: np.ndarray, level: float) -> float:
    """Brute-force Dunn et al. pooling: return the smallest score whose mean ECDF reaches level."""
    for candidate in np.sort(scores):
        ecdfs = [np.mean(scores[groups == group] <= candidate) for group in np.unique(groups)]
        if np.mean(ecdfs) >= level - 1e-12:
            return float(candidate)
    return float(scores.max())
