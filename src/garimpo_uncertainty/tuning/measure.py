"""How long a candidate takes, how busy it keeps the machine, and whether it repeats its bits."""

import contextlib
import math
import os
import statistics
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple, Self

import numpy as np
import torch

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
