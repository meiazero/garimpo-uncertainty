"""Debugging report of one run: per-stage status, failing error, shapes, timing, peak memory.

It checks that the pipeline ran, not that a method works; a pilot subset says nothing about quality.
Reads only persisted files and prints what exists, so it also works after a failed stage.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import garimpo_uncertainty.data
import garimpo_uncertainty.intervals

DATA_STAGES = ("download_scenes", "validate_data")
RUN_STAGES = (
    "train",
    "infer",
    "fit_directions",
    "layer_monotonicity",
    "endpoint_monotonicity",
    "compass_scores",
    "calibrate",
    "generate_intervals",
    "compute_metrics",
)


def read_status(status_dir: Path, stage: str) -> dict:
    path = status_dir / f"{stage}.json"
    return json.loads(path.read_text()) if path.exists() else {"stage": stage, "status": "not_run"}


def print_table(title: str, path: Path, build) -> None:
    print(f"\n--- {title}")
    if not path.exists():
        print(f"not produced: {path}")
        return
    print(build(pd.read_csv(path)).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--through",
        choices=RUN_STAGES,
        default=RUN_STAGES[-1],
        help="last stage expected to have run, for a run stopped after the layer diagnostics",
    )
    parser.add_argument(
        "--job-exit-code",
        type=int,
        default=0,
        help="exit code of the job; nonzero fails the check even when every status says ok",
    )
    args = parser.parse_args()

    root = garimpo_uncertainty.data.data_root()
    run_dir = args.run_dir
    expected = RUN_STAGES[: RUN_STAGES.index(args.through) + 1]
    records = [read_status(root / "status", stage) for stage in DATA_STAGES]
    records += [read_status(run_dir / "status", stage) for stage in expected]

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

    # download_scenes runs on the login node and a rerun may legitimately skip it.
    required = [r for r in records if r["stage"] != "download_scenes" or r["status"] != "not_run"]
    failed = [r for r in required if r["status"] != "ok"]
    first_failure = next(
        (r for r in records if r["status"] in ("failed", "refused", "running")), None
    )
    if first_failure is not None:
        print(f"\n--- failing stage: {first_failure['stage']} ({first_failure['status']})")
        print(first_failure.get("error", "no error recorded: the job died inside this stage"))
        print(first_failure.get("traceback", ""))
    if args.job_exit_code != 0:
        # An import, syntax or argparse error, a bad FORCE_FROM or merge_results reach no status.
        failed.append({"stage": f"job exit code {args.job_exit_code}"})
        print(f"\n--- job exit code {args.job_exit_code}: see the job's .err log")

    print_table(
        "scenes by scale, role and status",
        root / "download_report.csv",
        lambda f: f.groupby(["scale", "role", "status"]).size().rename("tiles").to_frame(),
    )
    print_table(
        "cells per role",
        root / "manifest.csv",
        lambda f: f.groupby("role").agg(cells=("cell_id", "size"), groups=("group", "nunique")),
    )
    print_table(
        "validation Dice per epoch", run_dir / "history.csv", lambda f: f.set_index("epoch")
    )
    print_table(
        "monotonicity for choosing the layer (validation role)",
        run_dir / "monotonicity_layer.csv",
        lambda f: f.set_index(["layer", "direction", "role"]),
    )
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
        run_dir / "calibration" / "summary.csv",
        lambda f: f.dropna(subset=["n_saturated_calibration"])[
            [
                "method",
                "calibration",
                "n_calibration_scores",
                "n_saturated_calibration",
                "quantile_saturated_fraction",
            ]
        ].set_index(["method", "calibration"]),
    )
    print_table(
        "PICP and MPIW per method (pipeline check only)",
        run_dir / "metrics" / "results.csv",
        lambda f: f[["method", "calibration", "n_cells", "n_groups", "picp", "mpiw"]].set_index(
            ["method", "calibration"]
        ),
    )

    results_path = run_dir / "metrics" / "results.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)
        found = set(zip(results.method, results.calibration, strict=True))
        absent = set(garimpo_uncertainty.intervals.COMBINATIONS) - found
        if absent:
            failed.append({"stage": f"results rows absent: {sorted(absent)}"})
        if results[["picp", "mpiw"]].isna().any(axis=None):
            failed.append({"stage": "results contain NaN in PICP or MPIW"})

    print("\n" + "=" * 78)
    if failed:
        print(f"PIPELINE CHECK: FAIL at {[r['stage'] for r in failed]}")
        print("=" * 78)
        raise SystemExit(1)
    print("PIPELINE CHECK: PASS (every stage ran; this says nothing about method quality)")
    print("=" * 78)


if __name__ == "__main__":
    main()
