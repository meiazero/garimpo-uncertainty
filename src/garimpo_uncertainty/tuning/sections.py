"""The tuning's sections, in order: each measures one hot path and returns its rows by name.

A section reads the run's choices from `form`, as `garimpo tune` parses them, and prints one line
per candidate as it goes; the verdict reads the rows.
"""

import json
import os
import statistics
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

import garimpo_uncertainty.tuning.measure
import garimpo_uncertainty.tuning.workloads

measure = garimpo_uncertainty.tuning.measure
work = garimpo_uncertainty.tuning.workloads


def hardware(form: dict, out: Path) -> dict:  # noqa: ARG001 - one signature for every section
    measure.require_gpu()
    gpu = torch.cuda.get_device_properties(0)
    found = {
        "gpu": gpu.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "gpu_memory_gb": round(gpu.total_memory / 2**30, 1),
        "cpus": len(os.sched_getaffinity(0)),
        "ram_gb": round(measure.ram_gb(), 1),
        "stack": measure.stack(),
        "defaults": measure.current_flags()
        | {
            "deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }
    print(json.dumps(found, indent=2))
    return {"hardware": found}


def loader(form: dict, out: Path) -> dict:
    store = out / "store"
    cells = work.synthetic_store(store, form["loader_cells"], form["seed"])
    rows = []
    for as_train in (False, True):
        runs = {
            workers: lambda w=workers, a=as_train: work.read_cells(store, cells, w, as_train=a)
            for workers in form["workers"]
        }
        timed = measure.time_candidates(runs, form["repeats"])
        for workers, found in timed.items():
            row = {"reads_as": "train" if as_train else "stages", "workers": workers}
            row |= measure.summarized(found) | {"bitwise": measure.same(found, timed[0])}
            row["cells_per_second"] = len(cells) / row["seconds"]["median"]
            rows.append(row)
            print(
                f"{row['reads_as']:6} {workers:2} workers: {row['cells_per_second']:5.0f} cells/s, "
                f"CPU busy {row['cpu_busy']:.0%}, I/O wait {row['cpu_iowait']:.0%}, "
                f"same bytes as 0 workers {row['bitwise']}"
            )
    return {"loader": {"rows": rows}}


def training_row(
    workers: int, found: measure.Timed, reference: measure.Timed, steps: int, loops: list
) -> dict:
    """One worker count's row: the loop's own time and usage, apart from the untimed run."""
    row = {"workers": workers} | measure.summarized(found)
    row["weights_match_0_workers"] = measure.same(found, reference)
    if "seconds" not in row:
        return row
    timed_loops = loops[1:]  # the first is the untimed run's
    row["seconds_per_update"] = statistics.median(s for s, *_ in timed_loops) / steps
    row["data_wait_share"] = statistics.median(wait / s for s, wait, _ in timed_loops)
    # The loop's own usage: the build and the weights' copy leave the GPU idle.
    for key in ("cpu_busy", "cpu_iowait"):
        row[key] = statistics.fmean(usage[key] for *_, usage in timed_loops)
    busy = [u["gpu_busy"] for *_, u in timed_loops if not np.isnan(u["gpu_busy"])]
    row["gpu_busy"] = statistics.fmean(busy or [np.nan])
    return row


def train(form: dict, out: Path) -> dict:
    store = out / "store"
    cells = work.synthetic_store(store, form["loader_cells"], form["seed"])
    found = {}
    for arch in form["archs"]:
        loops = {workers: [] for workers in form["workers"]}
        runs = {
            workers: lambda w=workers, a=arch, log=loops[workers]: work.training_run(
                a, store, cells, w, form["steps"], loops=log
            )
            for workers in form["workers"]
        }
        timed = measure.time_candidates(runs, form["repeats"])
        rows = []
        for workers, result in timed.items():
            row = training_row(workers, result, timed[0], form["steps"], loops[workers])
            if "seconds" in row:
                print(
                    f"{arch} {workers:2} workers: {1e3 * row['seconds_per_update']:.0f} ms/update, "
                    f"data wait {row['data_wait_share']:.0%}, GPU busy {row['gpu_busy']:.0%}, "
                    f"CPU busy {row['cpu_busy']:.0%}"
                )
            rows.append(row)
        found[f"train-{arch}"] = {"rows": rows}
    return found


class SearchTimes(NamedTuple):
    seconds: dict[str, list[float]]  # each searcher's summed seconds, per timed pass
    counters: dict[str, work.DecodeCounter]
    usage: list[dict]
    bitwise: bool


def time_searches(bench: work.Bench, form: dict, batches: int, size: int) -> SearchTimes:
    """Time both searchers on every batch, counting their decodes and checking their bits."""
    seconds = {name: [0.0] * form["repeats"] for name in work.SEARCHERS}
    counters = {name: work.DecodeCounter() for name in work.SEARCHERS}
    usage, bitwise = [], True
    for first in range(0, batches * size, size):
        inputs = work.batch_inputs(bench, slice(first, first + size))
        runs = {
            name: lambda s=searcher, c=counters[name], i=inputs: work.searched(i, s, c)
            for name, searcher in work.SEARCHERS.items()
        }
        timed = measure.time_candidates(runs, form["repeats"])
        bitwise &= measure.same(timed["memo"], timed["separate"]) and timed["memo"].stable
        usage += timed["memo"].usage
        for name, found in timed.items():
            runs_so_far = zip(seconds[name], found.seconds, strict=True)
            seconds[name] = [total + run for total, run in runs_so_far]
    return SearchTimes(seconds, counters, usage, bitwise)


def search_row(bench: work.Bench, form: dict) -> dict:
    """One layer's decodes and time per batch, for the memoized and the separate searches."""
    size = work.CONFIG["EVAL_BATCH_SIZE"]
    batches = form["cells"] // size
    times = time_searches(bench, form, batches, size)
    # A pass of every batch is timed `repeats` times after one untimed pass, all decoding alike.
    decodes = {name: c.calls / (form["repeats"] + 1) for name, c in times.counters.items()}
    median = {name: statistics.median(runs) for name, runs in times.seconds.items()}
    return {
        "layer": bench.layer,
        "memo_bitwise": times.bitwise,
        "decodes_per_batch": {name: decodes[name] / batches for name in work.SEARCHERS},
        "ms_per_decode": {name: 1e3 * median[name] / decodes[name] for name in work.SEARCHERS},
        "seconds_per_cell": {name: median[name] / (batches * size) for name in work.SEARCHERS},
        # A batch shorter than nvidia-smi's first sample has no reading.
        "gpu_busy": statistics.fmean(
            [r["gpu_busy"] for r in times.usage if not np.isnan(r["gpu_busy"])] or [np.nan]
        ),
        "gpu_memory_gb": max(run["gpu_memory_gb"] for run in times.usage),
    }


def search(form: dict, out: Path) -> dict:  # noqa: ARG001
    found = {}
    for arch in form["archs"]:
        rows = []
        for bench in work.benches(arch, form["search_layers"][arch], form["cells"], form["seed"]):
            row = search_row(bench, form)
            rows.append(row)
            calls = row["decodes_per_batch"]
            print(
                f"{arch} {bench.layer}: same bits {row['memo_bitwise']}, decodes per batch "
                f"{calls['separate']:.0f} -> {calls['memo']:.0f}, "
                f"{row['ms_per_decode']['memo']:.1f} ms per decode, GPU busy {row['gpu_busy']:.0%}"
            )
        found[f"search-{arch}"] = {"rows": rows}
    return found


def batch(form: dict, out: Path) -> dict:  # noqa: ARG001
    reference = work.CONFIG["EVAL_BATCH_SIZE"]
    sizes = sorted({reference, *form["batch_sizes"]})
    if uneven := [size for size in sizes if form["cells"] % size]:
        raise SystemExit(f"--cells {form['cells']} is no multiple of the batch sizes {uneven}")
    return {
        f"batch-{arch}": {"reference": reference, "rows": batch_rows(arch, sizes, reference, form)}
        for arch in form["archs"]
    }


def batch_rows(arch: str, sizes: list[int], reference: int, form: dict) -> list[dict]:
    rows = []
    for bench in work.benches(arch, form["timed_layers"][arch], form["cells"], form["seed"]):
        runs = {size: lambda s=size, b=bench: work.score_pass(b, s) for size in sizes}
        timed = measure.time_candidates(runs, form["repeats"])
        for size, result in timed.items():
            row = {"layer": bench.layer, "batch_size": size} | measure.summarized(result)
            rows.append(row | {"bitwise": measure.same(result, timed[reference])})
            if "seconds" in row:
                print(
                    f"{arch} {bench.layer} batch {size:3}: {row['seconds']['median']:.1f} s, "
                    f"GPU busy {row['gpu_busy']:.0%}, {row['gpu_memory_gb']:.1f} GB, "
                    f"same bits as {reference} {rows[-1]['bitwise']}"
                )
            else:
                print(f"{arch} {bench.layer} batch {size:3}: {row['failed']}")
    return rows


STRICT = {"cudnn_tf32": False, "matmul_tf32": False, "benchmark": False}


def precision_candidates() -> dict[str, dict[str, bool]]:
    return {
        "strict_fp32": STRICT,
        "strict_fp32_autotuned": STRICT | {"benchmark": True},
        "defaults": measure.current_flags(),
        "tf32": {"cudnn_tf32": True, "matmul_tf32": True, "benchmark": False},
    }


def flagged_pass(bench: work.Bench, flags: dict) -> dict:
    with measure.backend_flags(**flags):
        return work.score_pass(bench, work.CONFIG["EVAL_BATCH_SIZE"])


def precision(form: dict, out: Path) -> dict:  # noqa: ARG001
    candidates = precision_candidates()
    found = {}
    for arch in form["archs"]:
        rows = []
        for bench in work.benches(arch, form["timed_layers"][arch], form["cells"], form["seed"]):
            runs = {name: lambda f=f, b=bench: flagged_pass(b, f) for name, f in candidates.items()}
            timed = measure.time_candidates(runs, form["repeats"])
            for name, result in timed.items():
                row = {"layer": bench.layer, "candidate": name} | candidates[name]
                row |= measure.summarized(result)
                same = measure.same(result, timed["strict_fp32"])
                rows.append(row | {"bitwise_with_strict_fp32": same})
                if "failed" in row:
                    print(f"{arch} {bench.layer} {name:22}: {row['failed']}")
                    continue
                print(
                    f"{arch} {bench.layer} {name:22}: {row['seconds']['median']:.1f} s, "
                    f"same bits as strict float32 {same}"
                )
        found[f"precision-{arch}"] = {"rows": rows}
    return found


def lightgbm(form: dict, out: Path) -> dict:  # noqa: ARG001
    rng = np.random.default_rng(form["seed"])
    rows = []
    for channels in (16, 256):
        features = work.synthetic_features(rng, channels)
        runs = {
            jobs: lambda j=jobs, f=features: work.density_ratio(f, j) for jobs in form["threads"]
        }
        timed = measure.time_candidates(runs, form["repeats"])
        for jobs, found in timed.items():
            row = {"channels": channels, "jobs": jobs} | measure.summarized(found)
            rows.append(row | {"bitwise": measure.same(found, timed[form["threads"][0]])})
            print(
                f"{channels:3} channels, {jobs:2} threads: {row['seconds']['median']:.2f} s, "
                f"CPU busy {row['cpu_busy']:.0%}, same ratios as 1 thread {rows[-1]['bitwise']}"
            )
    return {"lightgbm": {"rows": rows}}


# In the notebook's order; the verdict expects every one, per architecture where it says so.
SECTIONS = {
    "hardware": hardware,
    "loader": loader,
    "train": train,
    "search": search,
    "batch": batch,
    "precision": precision,
    "lightgbm": lightgbm,
}
