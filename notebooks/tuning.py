"""Mechanics shared by the cells of 03_tune_hardware.ipynb: timing, usage, synthetic data.

The notebook holds the experiments and the verdict and imports this module as `t`; nothing here
decides anything.
"""

import argparse
import contextlib
import json
import math
import os
import statistics
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NamedTuple, Self

import compass2d.calibration
import compass2d.compass
import compass2d.directions
import compass2d.metrics
import numpy as np
import pandas as pd
import torch

import garimpo_uncertainty.cell_dataset
import garimpo_uncertainty.compass_j
import garimpo_uncertainty.data
import garimpo_uncertainty.intervals
import garimpo_uncertainty.models.common
import garimpo_uncertainty.stages.compass_scores
import garimpo_uncertainty.stages.train

# Every section measures what `garimpo` runs, which turns these on for every command that loads
# torch. cuBLAS reads CUBLAS_WORKSPACE_CONFIG at its first call, which comes later than this.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(mode=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / "configs" / "base.json")
SPLIT = read_json(ROOT / "outputs" / "split_summary.json")
BANDS = len(garimpo_uncertainty.data.BANDS)
CELL_PX = CONFIG["CELL_PX"]
CELLS_PER_TILE = (garimpo_uncertainty.data.TILE_PX // CELL_PX) ** 2
SEARCH = {
    "beta_range_logits": float(CONFIG["BETA_RANGE_LOGITS"]),
    "beta_range_features": float(CONFIG["BETA_RANGE_FEATURES"]),
    "search_threshold_logits": float(CONFIG["SEARCH_THRESHOLD_LOGITS"]),
    "search_threshold_features": float(CONFIG["SEARCH_THRESHOLD_FEATURES"]),
    "temperature": compass2d.metrics.UPSTREAM_TEMPERATURE,
    "direction_temperature": float(CONFIG["DIRECTION_TEMPERATURE"]),
}
NORMALIZATION = {"mean": [0.1] * BANDS, "std": [0.05] * BANDS}
SCORES = garimpo_uncertainty.stages.compass_scores.BatchScores._fields[:6]
COMPASS_METHODS = garimpo_uncertainty.intervals.COMPASS_METHODS


GPU = "NVIDIA L4"  # the experiment's one GPU: another picks other kernels, so other bits


def require_gpu() -> None:
    """Stop on any GPU but the L4: its bits and timings would size another experiment."""
    if not torch.cuda.is_available():
        raise SystemExit("No GPU: pick the L4 under Runtime > Change runtime type.")
    if (name := torch.cuda.get_device_name(0)) != GPU:
        raise SystemExit(f"This session got the {name}; the experiment runs on the {GPU} only.")


def stack() -> dict[str, str]:
    """Return the software that picks the GPU's kernels; sessions share bits only if it matches."""
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "colab_release": os.environ.get("COLAB_RELEASE_TAG", "unknown"),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "driver": driver,
    }


def ram_gb() -> float:
    meminfo = Path("/proc/meminfo").read_text(encoding="utf-8").split()
    return int(meminfo[meminfo.index("MemTotal:") + 1]) / 2**20


# --- Timing and usage ---------------------------------------------------------------------------


def cpu_ticks() -> tuple[int, int, int]:
    """Busy, I/O-wait and total ticks of every CPU since boot, from /proc/stat."""
    first_line = Path("/proc/stat").read_text(encoding="utf-8").split("\n", 1)[0]
    # guest and guest_nice, the last two, are already counted in user and nice.
    ticks = [int(value) for value in first_line.split()[1:9]]
    idle, iowait = ticks[3], ticks[4]
    return sum(ticks) - idle - iowait, iowait, sum(ticks)


class Usage:
    """How busy the GPU and the CPUs were over a block, and the most GPU memory it reserved.

    nvidia-smi samples the GPU's busy share every 100 ms; /proc/stat counts the CPUs' ticks.
    """

    def __enter__(self) -> Self:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # else the peak starts at what earlier candidates left cached
        torch.cuda.reset_peak_memory_stats()
        self.ticks = cpu_ticks()
        query = ["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-lms", "100"]
        self.sampler = subprocess.Popen(  # noqa: S603
            ["nvidia-smi", *query],  # noqa: S607
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return self

    def __exit__(self, *_: object) -> None:
        torch.cuda.synchronize()
        self.sampler.terminate()
        samples = [float(line) for line in self.sampler.communicate()[0].split() if line.isdigit()]
        now = cpu_ticks()
        busy, iowait, total = (end - start for end, start in zip(now, self.ticks, strict=True))
        self.found = {
            "gpu_busy": statistics.fmean(samples) / 100 if samples else math.nan,
            "gpu_memory_gb": torch.cuda.max_memory_reserved() / 2**30,
            "cpu_busy": busy / max(total, 1),
            "cpu_iowait": iowait / max(total, 1),
        }


class Timed(NamedTuple):
    """One candidate: each timed run's seconds and usage, and the bytes of its untimed run."""

    seconds: list[float]
    usage: list[dict]
    outputs: dict[str, bytes] | None
    stable: bool
    failed: str | None


def bytes_of(arrays: dict[str, np.ndarray]) -> dict[str, bytes]:
    return {name: np.ascontiguousarray(array).tobytes() for name, array in arrays.items()}


def time_candidates(candidates: dict, repeats: int) -> dict:
    """Run each candidate once untimed, then `repeats` times in an order that alternates.

    The untimed run pays CUDA's lazy loading and cuDNN's choice of kernels; the alternating order
    lets clock boost and throttling fall on every candidate alike. A candidate returns its outputs
    as arrays; `stable` says every timed run gave the untimed run's bytes. One that runs out of GPU
    memory is left out of the timing, with the reason.
    """
    found = {}
    for name, run in candidates.items():
        try:
            found[name] = Timed([], [], bytes_of(run()), stable=True, failed=None)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            found[name] = Timed([], [], None, stable=False, failed="out of GPU memory")
    names = [name for name in candidates if found[name].failed is None]
    for repeat in range(repeats):
        for name in names if repeat % 2 == 0 else names[::-1]:
            with Usage() as usage:
                start = time.perf_counter()
                outputs = bytes_of(candidates[name]())
                seconds = time.perf_counter() - start
            timed = found[name]
            timed.seconds.append(seconds)
            timed.usage.append(usage.found)
            found[name] = timed._replace(stable=timed.stable and outputs == timed.outputs)
    return found


def summarized(timed: Timed) -> dict:
    """Return the seconds' median and range, each usage averaged over the runs, and `stable`."""
    if timed.failed:
        return {"failed": timed.failed}
    seconds = {
        "median": statistics.median(timed.seconds),
        "low": min(timed.seconds),
        "high": max(timed.seconds),
    }
    usage = {key: statistics.fmean(run[key] for run in timed.usage) for key in timed.usage[0]}
    return {"seconds": seconds, **usage, "stable": timed.stable}


def same(first: Timed, second: Timed) -> bool:
    return first.outputs is not None and first.outputs == second.outputs


# --- Synthetic data -----------------------------------------------------------------------------


def synthetic_store(root: Path, cells: int, seed: int) -> pd.DataFrame:
    """Write images.npy and masks.npy with the real cell shape and dtypes, unless already there."""
    shape = (cells, BANDS, CELL_PX, CELL_PX)
    table = pd.DataFrame({"cell_index": np.arange(cells), "boa_offset": -1000.0})
    if (root / "masks.npy").exists() and np.load(root / "images.npy", mmap_mode="r").shape == shape:
        return table
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    images = np.lib.format.open_memmap(root / "images.npy", "w+", np.uint16, shape)
    masks = np.lib.format.open_memmap(root / "masks.npy", "w+", np.uint8, (cells, *shape[2:]))
    for cell in range(cells):
        images[cell] = rng.integers(0, 6000, shape[1:], dtype=np.uint16)
        masks[cell] = rng.random(shape[2:]) < rng.uniform(0, 0.3)
    images.flush()
    masks.flush()
    return table


def evict(store: Path) -> None:
    """Drop the store's pages from RAM, so the next read comes from the disk."""
    os.sync()
    for name in ("images.npy", "masks.npy"):
        with (store / name).open("rb") as file:
            os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


class Bench(NamedTuple):
    """One layer's search inputs: a random-weight model, synthetic cells, random directions."""

    model: torch.nn.Module
    images: torch.Tensor
    masks: torch.Tensor
    layer: str
    fitted: compass2d.directions.Directions


def random_directions(channels: int, rng: np.random.Generator) -> compass2d.directions.Directions:
    component = rng.normal(size=(1, channels))
    return compass2d.directions.Directions(
        pca_components=component / np.linalg.norm(component),
        pca_mean=rng.normal(size=channels) * 1e-3,
        pca_explained_variance=np.ones(1),
        svm_w=rng.normal(size=channels),
    )


def benches(arch: str, layers: list[str], cells: int, seed: int) -> Iterator[Bench]:
    """Yield a Bench per decoder layer in `layers`, or per layer of the model when it is empty."""
    require_gpu()
    torch.manual_seed(CONFIG["SEED"])
    model = garimpo_uncertainty.models.common.build(arch, BANDS)
    model = model.eval().requires_grad_(requires_grad=False).to("cuda")
    known = garimpo_uncertainty.models.common.feature_layers(model)[:-1]
    if unknown := sorted(set(layers) - set(known)):
        raise SystemExit(f"{arch} has no layer {unknown}; its layers are {known}")
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn((cells, BANDS, CELL_PX, CELL_PX), generator=generator)
    rate = torch.rand((cells, 1, 1, 1), generator=generator) * 0.3
    masks = (torch.rand((cells, 1, CELL_PX, CELL_PX), generator=generator) < rate).float()
    features = garimpo_uncertainty.models.common.encode(model, images[:1].to("cuda"))
    rng = np.random.default_rng(seed)
    for layer in layers or known:
        z, _ = garimpo_uncertainty.models.common.split_at(model, layer, features)
        yield Bench(model, images, masks, layer, random_directions(z.shape[1], rng))


# --- The pipeline's hot paths -------------------------------------------------------------------


def read_cells(store: Path, cells: pd.DataFrame, workers: int, *, as_train: bool) -> dict:
    """Read every cell from the disk onto the GPU as a stage does; return each batch's checksum.

    As train reads, the cells come shuffled and flipped at BATCH_SIZE. The checksum is integer sums
    on the GPU, the same small cost at every worker count, so no read escapes the clock.
    """

    def checksum(tensor: torch.Tensor) -> torch.Tensor:
        # A plain sum misses a flip; each cell's sums weighted by column and by row catch it.
        bits = tensor.to("cuda").view(torch.int32)
        columns = bits.sum((1, 2), dtype=torch.int64)
        rows = bits.sum((1, 3), dtype=torch.int64)
        width = torch.arange(columns.shape[-1], device="cuda")
        height = torch.arange(rows.shape[-1], device="cuda")
        return torch.cat([(columns * width).sum(1), (rows * height).sum(1)])

    evict(store)
    torch.manual_seed(CONFIG["SEED"])
    loader = garimpo_uncertainty.cell_dataset.loader(
        store,
        cells,
        NORMALIZATION,
        CONFIG["BATCH_SIZE" if as_train else "EVAL_BATCH_SIZE"],
        workers,
        samples=len(cells) if as_train else None,
        augment=as_train,
    )
    sums = [
        torch.cat([checksum(tensor) for tensor in batch[:2]])
        for batch in garimpo_uncertainty.cell_dataset.batches(loader)
    ]
    return {"checksums": torch.cat(sums).cpu().numpy()}


def training_run(
    arch: str, store: Path, cells: pd.DataFrame, workers: int, steps: int, *, loops: list
) -> dict:
    """Run `steps` updates of train's own fit_epoch from a seeded start; return the weights.

    The loop's seconds, data wait and usage go to `loops`, apart from the outputs compared to the
    bit: the model's build and the weights' copy are not the loop's.
    """
    torch.manual_seed(CONFIG["SEED"])
    model = garimpo_uncertainty.models.common.build(arch, BANDS).to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"]
    )
    loss = garimpo_uncertainty.stages.train.build_loss(CONFIG["LOSS"])
    state = garimpo_uncertainty.stages.train.TrainingState(
        model, optimizer, loss, {}, {}, [], math.nan, 0
    )
    batches = garimpo_uncertainty.cell_dataset.loader(
        store,
        cells,
        NORMALIZATION,
        CONFIG["BATCH_SIZE"],
        workers,
        samples=steps * CONFIG["BATCH_SIZE"],
        augment=True,
    )
    evict(store)
    with Usage() as usage:
        started = time.monotonic()
        fitted, _ = garimpo_uncertainty.stages.train.fit_epoch(state, batches, "cuda", started)
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
    loops.append((seconds, fitted["train_data_wait_seconds"], usage.found))
    weights = [parameter.detach().cpu().numpy().ravel() for parameter in model.parameters()]
    return {"weights": np.concatenate(weights)}


class DecodeCounter:
    def __init__(self) -> None:
        self.calls = 0

    def wrap(self, decode: Callable[[torch.Tensor], torch.Tensor]) -> Callable:
        def counted(z: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return decode(z)

        return counted


def separate_search(
    space: garimpo_uncertainty.compass_j.Space,
    beta_range: float,
    threshold: float,
    target: torch.Tensor,
) -> list[torch.Tensor]:
    """Run compass2d's three searches one after another, as compass_scores did before Memo."""
    searched_space = (space.decode, space.metric, space.z, space.delta, target)
    limits = (beta_range, threshold)
    lower, upper = compass2d.compass.asymmetric_binary_search_scores(*searched_space, *limits)
    return [*compass2d.compass.binary_search_scores(*searched_space, *limits), *lower, *upper]


SEARCHERS = {
    "separate": separate_search,
    "memo": garimpo_uncertainty.stages.compass_scores.search_space,
}


class BatchInputs(NamedTuple):
    spaces: list
    target: torch.Tensor
    jacobians: np.ndarray
    feature_sums: np.ndarray


def batch_inputs(bench: Bench, cells: slice) -> BatchInputs:
    """One batch's spaces, target, Jacobians and summed features, from the stage's batch_spaces."""
    stage_args = argparse.Namespace(**SEARCH, layer=bench.layer)
    images = bench.images[cells].to("cuda")
    features = garimpo_uncertainty.models.common.encode(bench.model, images)
    spaces, jacobians, feature_sums = garimpo_uncertainty.stages.compass_scores.batch_spaces(
        bench.model, features, bench.fitted, stage_args
    )
    masks = bench.masks[cells].to("cuda", torch.float64)
    target = masks.flatten(1).sum(1) * garimpo_uncertainty.data.PIXEL_HA
    return BatchInputs(spaces, target, jacobians, feature_sums)


def searched(inputs: BatchInputs, searcher: Callable, counter: DecodeCounter) -> dict:
    """Every method's six score arrays on one batch, counting the decoder's calls.

    COMPASS-L moves the logits themselves, with no decoder, so only COMPASS-J's calls count.
    """
    rows = []
    for method, (space, *limits) in zip(COMPASS_METHODS, inputs.spaces, strict=True):
        counted = (
            space if method == "compass_l" else space._replace(decode=counter.wrap(space.decode))
        )
        rows.append(searcher(counted, *limits, inputs.target))
    return {name: np.stack([row[c].cpu().numpy() for row in rows]) for c, name in enumerate(SCORES)}


def score_pass(bench: Bench, size: int) -> dict[str, np.ndarray]:
    """compass_scores' arrays for every cell at one batch size, with the cells on the last axis."""
    counter, parts = DecodeCounter(), []
    for first in range(0, len(bench.images), size):
        inputs = batch_inputs(bench, slice(first, first + size))
        scores = searched(inputs, SEARCHERS["memo"], counter)
        parts.append(scores | {"jacobians": inputs.jacobians.T, "sums": inputs.feature_sums.T})
    return {name: np.concatenate([part[name] for part in parts], axis=-1) for name in parts[0]}


def current_flags() -> dict[str, bool]:
    return {
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "benchmark": torch.backends.cudnn.benchmark,
    }


@contextlib.contextmanager
def backend_flags(*, cudnn_tf32: bool, matmul_tf32: bool, benchmark: bool) -> Iterator[None]:
    cudnn, matmul = torch.backends.cudnn, torch.backends.cuda.matmul
    saved = cudnn.allow_tf32, matmul.allow_tf32, cudnn.benchmark
    cudnn.allow_tf32, matmul.allow_tf32, cudnn.benchmark = cudnn_tf32, matmul_tf32, benchmark
    try:
        yield
    finally:
        cudnn.allow_tf32, matmul.allow_tf32, cudnn.benchmark = saved


def synthetic_features(rng: np.random.Generator, channels: int) -> tuple[np.ndarray, ...]:
    """Return summed features and groups shaped as the full run's calibration and test cells."""
    tiles, groups = SPLIT["tiles_by_scale_role"], SPLIT["groups_by_role"]
    calibration = tiles["Artisanal/calibration"] * CELLS_PER_TILE
    test = tiles["Artisanal/test"] * CELLS_PER_TILE
    return (
        rng.lognormal(size=(calibration, channels)),
        rng.lognormal(mean=0.2, size=(test, channels)),
        rng.integers(0, groups["calibration"], calibration),
        rng.integers(0, groups["test"], test),
    )


def density_ratio(features: tuple[np.ndarray, ...], jobs: int) -> dict[str, np.ndarray]:
    """Return calibrate's density ratio of test over calibration features at `jobs` threads."""
    calibration, test = compass2d.calibration.density_ratio(
        *features,
        n_folds=garimpo_uncertainty.intervals.DENSITY_RATIO_FOLDS,
        seed=CONFIG["SEED"],
        n_jobs=jobs,
    )
    return {"calibration": calibration, "test": test}
