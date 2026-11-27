"""The tuning's verdict: what each section's rows say about the machine, and what failed.

A failure is a lost bit or a section that gave no result; a finding is a measurement to size the
run by. The tuning passes only with no failure.
"""

import math

STARVED = 0.10  # a loop waiting on data longer than this share of its time starves
GPU_BUSY = 0.80  # below this busy share, the GPU waits on the CPU or on kernel launches
MEMORY_IN_USE = 0.80  # a batch leaves this share of GPU memory free; use varies by layer and cell


def limit(gpu_busy: float, data_wait: float = 0.0) -> str:
    if math.isnan(gpu_busy):
        return "GPU use not sampled: nvidia-smi gave no reading"
    if data_wait > STARVED:
        return f"limited by data loading, waiting {data_wait:.0%} of the time"
    if gpu_busy >= GPU_BUSY:
        return f"limited by the GPU, busy {gpu_busy:.0%}: the machine is put to use"
    return f"limited by the CPU or kernel launches: the GPU is busy only {gpu_busy:.0%}"


def overlap(first: dict, second: dict) -> bool:
    return first["low"] <= second["high"] and second["low"] <= first["high"]


def cheapest_tied(rows: list[dict], cost: str) -> dict | None:
    """Return the safe row of least `cost` whose timed range overlaps the fastest safe row's."""
    safe = [row for row in rows if "seconds" in row and row["bitwise"]]
    if not safe:
        return None
    fastest = min(safe, key=lambda row: row["seconds"]["median"])
    tied = [row for row in safe if overlap(row["seconds"], fastest["seconds"])]
    return min(tied, key=lambda row: row[cost])


class Verdict:
    """Findings and failures, judged against the full run: its updates and evaluated cells."""

    def __init__(self, failures: list[str], updates: int, evaluated_cells: int) -> None:
        self.failures, self.findings = list(failures), []
        self.updates, self.evaluated_cells = updates, evaluated_cells

    def loader(self, rows: list[dict]) -> None:
        # train's flips follow its draws, so no worker count may change what either read gives.
        self.failures.extend(
            f"loader: {r['reads_as']} at {r['workers']} workers read other bytes"
            for r in rows
            if not r["bitwise"]
        )
        stages = [row for row in rows if row["reads_as"] == "stages"]
        best = max(stages, key=lambda row: row["cells_per_second"])
        self.findings.append(
            f"loader: the stages read up to {best['cells_per_second']:.0f} cells/s from disk"
        )

    def train(self, arch: str, rows: list[dict]) -> None:
        rows = [row for row in rows if "seconds" in row]
        for row in rows:
            speed = f"{1e3 * row['seconds_per_update']:.0f} ms per update"
            self.findings.append(
                f"train {arch} {row['workers']} workers: {speed}, "
                + limit(row["gpu_busy"], row["data_wait_share"])
            )
        # Weights that differ between two runs at one worker count say nothing about the workers.
        if unstable := [row["workers"] for row in rows if not row["stable"]]:
            self.failures.append(
                f"train {arch}: two runs at {unstable} workers end with other weights under "
                "deterministic algorithms"
            )
        elif moved := [row["workers"] for row in rows if not row["weights_match_0_workers"]]:
            # The draws happen in the main process, so the workers must not move a bit.
            self.failures.append(
                f"train {arch}: {moved} workers end with other weights than 0 workers"
            )
        fed = [row for row in rows if row["data_wait_share"] <= STARVED]
        if not fed:
            self.failures.append(
                f"train {arch}: data loading starves the loop at every worker count"
            )
            return
        pick = min(fed, key=lambda row: row["workers"])
        hours = pick["seconds_per_update"] * self.updates / 3600
        self.findings.append(
            f"train {arch}: {pick['workers']} workers keep it fed; "
            f"{self.updates} updates take ~{hours:.1f} h"
        )

    def search(self, arch: str, rows: list[dict]) -> None:
        self.failures.extend(
            f"search {arch} {r['layer']}: Memo changed a score"
            for r in rows
            if not r["memo_bitwise"]
        )
        for row in rows:
            calls = row["decodes_per_batch"]
            self.findings.append(
                f"search {arch} {row['layer']}: {calls['separate']:.0f} -> {calls['memo']:.0f} "
                f"decodes per batch, {row['ms_per_decode']['memo']:.1f} ms each, "
                + limit(row["gpu_busy"])
            )
        seconds = sum(row["seconds_per_cell"]["memo"] for row in rows) * self.evaluated_cells
        self.findings.append(
            f"search {arch}: these layers take ~{seconds / 3600:.1f} h over the full run's "
            f"{self.evaluated_cells} cells "
            "[estimate: a trained model's searches stop at other steps]"
        )

    def batch(self, arch: str, section: dict, memory: float) -> None:
        reference = section["reference"]
        for layer in dict.fromkeys(row["layer"] for row in section["rows"]):
            rows = [r for r in section["rows"] if r["layer"] == layer and "seconds" in r]
            base = next((r for r in rows if r["batch_size"] == reference), None)
            if base is None or not base["stable"]:
                self.failures.append(
                    f"batch {arch} {layer}: batch {reference} failed or changed its bits"
                )
                continue
            state = (
                f"batch {arch} {layer}: {reference} keeps the GPU {base['gpu_busy']:.0%} busy in "
                f"{base['gpu_memory_gb']:.1f} of {memory} GB"
            )
            roomy = [r for r in rows if r["gpu_memory_gb"] <= MEMORY_IN_USE * memory]
            faster = [r for r in roomy if r["seconds"]["high"] < base["seconds"]["low"]]
            if base["gpu_busy"] >= GPU_BUSY or not faster:
                self.findings.append(f"{state}; no larger batch is worth its memory")
                continue
            pick = faster[0]
            moves = "" if pick["bitwise"] else ", but it moves numbers: a decision, not a setting"
            self.findings.append(f"{state}; batch {pick['batch_size']} is faster{moves}")

    def precision(self, arch: str, rows: list[dict]) -> None:
        for row in rows:
            if row["candidate"] == "strict_fp32" and not row["stable"]:
                self.failures.append(
                    f"precision {arch} {row['layer']}: strict float32 gave other bits twice"
                )
            bits = (
                "the bits of strict float32"
                if row["bitwise_with_strict_fp32"]
                else "other bits than strict float32"
            )
            self.findings.append(
                f"precision {arch} {row['layer']} {row['candidate']}: "
                f"{row['seconds']['median']:.1f} s, {bits}"
            )

    def lightgbm(self, rows: list[dict]) -> None:
        for channels in dict.fromkeys(row["channels"] for row in rows):
            width = [row for row in rows if row["channels"] == channels]
            self.failures.extend(
                f"lightgbm {channels} channels: {r['jobs']} threads gave other ratios"
                for r in width
                if not r["bitwise"]
            )
            pick = cheapest_tied(width, "jobs")
            self.findings.append(
                f"lightgbm {channels} channels: {pick and pick['jobs']} threads are enough"
            )


def expected(archs: list[str]) -> list[str]:
    """Name every result the sections should have given, per architecture where they split."""
    kinds = ("train", "search", "batch", "precision")
    return ["hardware", "loader", "lightgbm", *(f"{k}-{arch}" for k in kinds for arch in archs)]


def judge(results: dict, failed: list[str], archs: list[str], updates: int, cells: int) -> Verdict:
    """Judge every result there is; a missing one is a failure of its own."""
    verdict = Verdict(failed, updates, cells)
    verdict.failures.extend(
        f"{name}: no result, the section crashed or never ran"
        for name in expected(archs)
        if name not in results
    )
    if "hardware" not in results:
        return verdict
    if "loader" in results:
        verdict.loader(results["loader"]["rows"])
    if "lightgbm" in results:
        verdict.lightgbm(results["lightgbm"]["rows"])
    for arch in archs:
        judge_arch(verdict, results, arch)
    return verdict


def judge_arch(verdict: Verdict, results: dict, arch: str) -> None:
    if f"train-{arch}" in results:
        verdict.train(arch, results[f"train-{arch}"]["rows"])
    if f"search-{arch}" in results:
        verdict.search(arch, results[f"search-{arch}"]["rows"])
    if f"batch-{arch}" in results:
        verdict.batch(arch, results[f"batch-{arch}"], results["hardware"]["gpu_memory_gb"])
    if f"precision-{arch}" in results:
        rows = [r for r in results[f"precision-{arch}"]["rows"] if "seconds" in r]
        verdict.precision(arch, rows)
