"""`garimpo tune`: size the run to this GPU on synthetic cells, then judge what was measured.

Every section runs even when one before it failed, so one run reports every finding; the command
exits 1 when a section failed, a run gave other bits when repeated, or the workers changed
what was read or trained; a faster candidate with other bits than strict float32 is a finding to
decide on. outputs/tuning/results.json keeps the rows.
"""

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

import garimpo_uncertainty.files
import garimpo_uncertainty.models
import garimpo_uncertainty.tuning.verdict

# The sections and their workloads read the configs at import, so garimpo tune imports them only
# once its arguments are read: --help works, and a config that cannot be read fails the verdict.

OUT = garimpo_uncertainty.files.ROOT / "outputs" / "tuning"


def worker_counts(chosen: list[int] | None) -> list[int]:
    """Return the chosen counts with 0, the reference, or a spread up to every CPU."""
    if chosen is not None:
        return sorted({0, *chosen})
    cpus = len(os.sched_getaffinity(0))
    return sorted({0, 2, 4, cpus // 2, cpus} & set(range(cpus + 1)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    archs = garimpo_uncertainty.models.MODEL_REGISTRY.names()
    parser.add_argument("--archs", nargs="+", choices=archs, default=archs)
    parser.add_argument("--cells", type=int, default=64, help="cells each search and batch reads")
    parser.add_argument(
        "--search-unet-layers", nargs="*", default=[], help="none: every decoder layer"
    )
    parser.add_argument(
        "--timed-unet-layers", nargs="*", default=["decoder.blocks.0", "decoder.blocks.4"]
    )
    parser.add_argument("--loader-cells", type=int, default=256)
    parser.add_argument("--workers", nargs="+", type=int, help="default: a spread up to every CPU")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--steps", type=int, default=20, help="updates per timed training run")
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def form(args: argparse.Namespace) -> dict:
    """Return the choices as the sections read them, spelled out."""
    workers = worker_counts(args.workers)
    return {
        "seed": 0,
        "archs": args.archs,
        "cells": args.cells,
        # Only the U-Net's layers are chosen; an empty list is every layer the model has.
        "search_layers": {arch: [] for arch in args.archs} | {"unet": args.search_unet_layers},
        "timed_layers": {arch: [] for arch in args.archs} | {"unet": args.timed_unet_layers},
        "loader_cells": args.loader_cells,
        "workers": workers,
        "threads": sorted({1, *(count for count in workers if count > 0)}),
        "batch_sizes": args.batch_sizes,
        "steps": args.steps,
        "repeats": args.repeats,
    }


def run_sections(chosen: dict, out: Path) -> tuple[dict, list[str]]:
    """Run every section in order; a failure is noted and the later sections still run.

    A section yields each architecture's result once it is done, so a later one that fails keeps
    the earlier ones.
    """
    import garimpo_uncertainty.tuning.sections  # noqa: PLC0415 - reads the configs

    results, failed = {}, []
    for name, section in garimpo_uncertainty.tuning.sections.SECTIONS.items():
        print(f"\n===== {name} =====", flush=True)
        try:
            for part in section(chosen, out):
                results |= part
        except (Exception, SystemExit) as error:  # noqa: BLE001 - every section must get its turn
            traceback.print_exc()
            failed.append(f"{name}: {type(error).__name__}: {error}")
    return results, failed


def json_safe(value: object) -> object:
    """Return `value` with None for each NaN, a GPU use never sampled, as JSON has no NaN."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return value


def full_run_size() -> tuple[int, int]:
    """Return the full run's updates and the cells its search stages read."""
    import garimpo_uncertainty.tuning.workloads  # noqa: PLC0415 - reads the configs

    full = garimpo_uncertainty.tuning.workloads.read_json(
        garimpo_uncertainty.files.ROOT / "configs" / "scales" / "full.json"
    )
    tiles = garimpo_uncertainty.tuning.workloads.SPLIT["tiles_by_scale_role"]
    evaluated = sum(
        tiles.get(f"Artisanal/{role}", 0) for role in ("validation", "calibration", "test")
    )
    cells = garimpo_uncertainty.tuning.workloads.CELLS_PER_TILE * evaluated
    return full["EPOCHS"] * full["STEPS_PER_EPOCH"], cells


def show(results: dict, verdict: garimpo_uncertainty.tuning.verdict.Verdict) -> None:
    if hardware := results.get("hardware"):
        print(
            f"{hardware['gpu']}, {hardware['gpu_memory_gb']} GB; {hardware['cpus']} CPUs; "
            f"{hardware['ram_gb']} GB RAM"
        )
        print("Stack:", ", ".join(f"{key} {value}" for key, value in hardware["stack"].items()))
    print("Findings:", *(f"- {line}" for line in verdict.findings), sep="\n")
    print("Failures:", *(f"- {line}" for line in verdict.failures or ["none"]), sep="\n")
    print(
        "TUNING: FAIL"
        if verdict.failures
        else "TUNING: PASS: every section ran, and every reference kept its bits"
    )


def main() -> None:
    chosen = form(parse_args())
    print("Choices as the sections read them:", json.dumps(chosen, indent=2), sep="\n")
    try:  # every section reads the configs, so one that cannot be read leaves nothing to measure
        updates, cells = full_run_size()
    except Exception as error:  # noqa: BLE001 - any misread value; the verdict must still print
        print(f"TUNING: FAIL: the configs cannot be read: {type(error).__name__}: {error}")
        sys.exit(1)
    results, failed = run_sections(chosen, OUT)
    try:  # the verdict is the run's point, so a full disk must not cost it
        OUT.mkdir(parents=True, exist_ok=True)
        garimpo_uncertainty.files.write_json(OUT / "results.json", json_safe(results))
    except OSError as error:
        failed.append(f"results.json: {type(error).__name__}: {error}")
    verdict = garimpo_uncertainty.tuning.verdict.judge(
        results, failed, chosen["archs"], updates, cells
    )
    print("\n===== verdict =====")
    show(results, verdict)
    sys.exit(1 if verdict.failures else 0)
