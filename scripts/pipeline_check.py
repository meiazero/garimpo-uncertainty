"""Debugging report of one run: per-stage status, failing error, shapes, timing, peak memory.

It checks that the pipeline ran, not that a method works; a pilot subset says nothing about quality.
Reads only persisted files and prints what exists, so it also works after a failed stage.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

import garimpo_uncertainty.data
import garimpo_uncertainty.intervals
import garimpo_uncertainty.stages
import garimpo_uncertainty.validation


def read_status(status_dir: Path, stage: str) -> dict:
    path = status_dir / f"{stage}.json"
    return json.loads(path.read_text()) if path.exists() else {"stage": stage, "status": "not_run"}


def print_table(title: str, path: Path, build: Callable[[pd.DataFrame], object]) -> None:
    print(f"\n--- {title}")
    if not path.exists():
        print(f"not produced: {path}")
        return
    print(build(pd.read_csv(path)).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, default="outputs/split.csv")
    # Arguments and not run_meta.json: a job that fails before training has no run_meta.json.
    parser.add_argument("--train-fraction", type=float, required=True)
    parser.add_argument("--subset-seed", type=int, required=True)
    parser.add_argument(
        "--job-exit-code",
        type=int,
        default=0,
        help="exit code of the job; nonzero fails the check even when every status says ok",
    )
    parser.add_argument("--workers", type=int, default=8, help="threads hashing the slots")
    return parser.parse_args()


def print_stage_table(records: list[dict]) -> None:
    print("=" * 78)
    print("PIPELINE CHECK: debugging report of the run, not a result about any method")
    print("=" * 78)
    print(f"{'stage':<22}{'status':<10}{'seconds':>9}{'rss GB':>9}{'gpu GB':>9}  job")
    for record in records:
        print(
            f"{record['stage']:<22}{record['status']:<10}{record.get('seconds', ''):>9}"
            f"{record.get('peak_rss_gb', ''):>9}{record.get('peak_gpu_gb', ''):>9}"
            f"  {record.get('slurm_job_id') or ''}"
        )
    print("\n--- shapes")
    for record in records:
        if "shapes" in record:
            print(f"{record['stage']}: {record['shapes']}")


def report_failed_stages(records: list[dict], job_exit_code: int) -> list[dict]:
    """Print the first failing stage and the job's exit code; return what failed."""
    # download_scenes is shared by every fraction, so another fraction's download running or failing
    # says nothing of this run; validate_data decides whether this run's tiles are there.
    required = [r for r in records if r["stage"] != "download_scenes"]
    failed = [r for r in required if r["status"] != "ok"]
    first_failure = next(
        (r for r in required if r["status"] in ("failed", "refused", "running", "interrupted")),
        None,
    )
    if first_failure is not None:
        print(f"\n--- failing stage: {first_failure['stage']} ({first_failure['status']})")
        print(first_failure.get("error", "no error recorded: the job died inside this stage"))
        print(first_failure.get("traceback", ""))
    if job_exit_code != 0:
        # An import, syntax or argparse error, a refused config or merge_results reach no status.
        failed.append({"stage": f"job exit code {job_exit_code}"})
        print(f"\n--- job exit code {job_exit_code}: see the job's .err log")
    return failed


def report_store(
    root: Path, split: pd.DataFrame, tile_ids: list[int]
) -> tuple[list[str], pd.DataFrame | None]:
    """Print the run's scenes and cells as far as the store allows; return problems and tiles."""
    store_problems = garimpo_uncertainty.validation.unusable_store(root, split, tile_ids)
    tiles = None
    if not store_problems or store_problems[0].startswith("store_missing"):
        tiles = garimpo_uncertainty.data.tile_records(root, split, tile_ids)
        print("\n--- scenes of this run's tiles by scale, role and status")
        print(
            tiles.groupby(["scale", "role", "status"]).size().rename("tiles").to_frame().to_string()
        )
        absent = tiles[tiles.status == garimpo_uncertainty.data.NOT_ATTEMPTED].tile_id.tolist()
        if absent:
            store_problems.append(
                f"sidecars_missing: {len(absent)} tiles, for example {absent[:5]}"
            )
    if not store_problems:
        cells = garimpo_uncertainty.data.cell_table(root, split, tile_ids)
        print("\n--- cells of this run's tiles per role")
        print(
            cells.groupby("role")
            .agg(cells=("cell_index", "size"), groups=("group", "nunique"))
            .to_string()
        )
    return store_problems, tiles


def print_chosen_layer(path: Path) -> None:
    print("\n--- layer chosen from the validation role")
    if not path.exists():
        print(f"not produced: {path}")
        return
    chosen = json.loads(path.read_text(encoding="utf-8"))
    print(f"{chosen['layer']}, ascending on at least {chosen['min_fraction_ascending']}")
    print(pd.DataFrame(chosen["candidates"]).set_index("layer").to_string())


def print_run_tables(run_dir: Path) -> None:
    print_table(
        "losses, validation metrics, gradient norms, timings and GPU peaks per epoch",
        run_dir / "history.csv",
        lambda f: f.set_index("epoch"),
    )
    print_table(
        "monotonicity for choosing the layer (validation role)",
        run_dir / "monotonicity_layer.csv",
        lambda f: f.set_index(["layer", "direction", "role"]),
    )
    print_chosen_layer(run_dir / "chosen_layer.json")
    print_table(
        "monotonicity of the chosen layer, on grid points only",
        run_dir / "monotonicity_endpoint.csv",
        lambda f: f.set_index(["layer", "direction", "role"]),
    )
    print_table(
        "|cos(v1, w)| and PCA explained variance per layer",
        run_dir / "directions" / "diagnostics.csv",
        lambda f: f[
            [
                "layer",
                "feature_shape",
                "n_train_cells",
                "n_zero_jacobians",
                "pca_explained_variance",
                "abs_cos_v1_w",
                "svm_mean_jacobian_projection",
            ]
        ].set_index("layer"),
    )
    print_table(
        "saturated searches",
        run_dir / "metrics" / "results.csv",
        lambda f: f.dropna(subset=["n_saturated_calibration"])[
            [
                "alpha",
                "method",
                "calibration",
                "n_calibration_scores",
                "n_saturated_calibration",
                "quantile_saturated_fraction",
            ]
        ].set_index(["alpha", "method", "calibration"]),
    )
    print_table(
        "PICP and MPIW per method (pipeline check only)",
        run_dir / "metrics" / "results.csv",
        lambda f: f[
            ["alpha", "method", "calibration", "n_cells", "n_groups", "picp", "mpiw"]
        ].set_index(["alpha", "method", "calibration"]),
    )


def results_failures(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []
    results = pd.read_csv(results_path)
    failures = []
    found = set(zip(results.alpha, results.method, results.calibration, strict=True))
    expected = {
        (alpha, method, calibration)
        for alpha in garimpo_uncertainty.intervals.ALPHAS
        for method, calibration in garimpo_uncertainty.intervals.COMBINATIONS
    }
    if absent := expected - found:
        failures.append({"stage": f"results rows absent: {sorted(absent)}"})
    if results[["picp", "mpiw"]].isna().any(axis=None):
        failures.append({"stage": "results contain NaN in PICP or MPIW"})
    return failures


def report_slots(
    root: Path, tiles: pd.DataFrame | None, store_problems: list[str], workers: int
) -> list[str]:
    # Last, and flushed first: this reads every slot of the run, and a job at its limit may not end.
    print("\n--- this run's slots against the hashes in their sidecars", flush=True)
    if tiles is not None and not store_problems:
        store_problems = garimpo_uncertainty.validation.slot_mismatches(root, tiles, workers)
    print("\n".join(store_problems) or "every slot matches")
    return store_problems


def exit_with_verdict(failed: list[dict]) -> None:
    print("\n" + "=" * 78)
    if failed:
        print(f"PIPELINE CHECK: FAIL at {[r['stage'] for r in failed]}")
        print("=" * 78)
        raise SystemExit(1)
    print("PIPELINE CHECK: PASS (every stage ran; this says nothing about method quality)")
    print("=" * 78)


def main() -> None:
    args = parse_args()
    root = garimpo_uncertainty.data.data_root()
    split = pd.read_csv(args.split)
    tile_ids = garimpo_uncertainty.data.select_tiles(
        split, args.train_fraction, args.subset_seed
    ).tile_id.tolist()
    selection = garimpo_uncertainty.data.selection_name(args.train_fraction, args.subset_seed)
    records = [
        read_status(root / "status", "download_scenes"),
        read_status(root / "status" / selection, "validate_data"),
    ]
    records += [
        read_status(args.run_dir / "status", stage)
        for stage in garimpo_uncertainty.stages.STAGE_ORDER
    ]

    print_stage_table(records)
    failed = report_failed_stages(records, args.job_exit_code)
    store_problems, tiles = report_store(root, split, tile_ids)
    print_run_tables(args.run_dir)
    failed += results_failures(args.run_dir / "metrics" / "results.csv")
    store_problems = report_slots(root, tiles, store_problems, args.workers)
    failed += [{"stage": problem} for problem in store_problems]
    exit_with_verdict(failed)


if __name__ == "__main__":
    main()
