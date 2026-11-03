"""Stage: joins the small tables of several runs, one row per model x method x calibration."""

import argparse
from pathlib import Path

import pandas as pd

import garimpo_uncertainty.data

TABLES = {
    "results.csv": "metrics/results.csv",
    "width_comparisons.csv": "metrics/width_comparisons.csv",
    "coverage_comparisons.csv": "metrics/coverage_comparisons.csv",
    "q1_spearman.csv": "metrics/q1_spearman.csv",
    "layer_diagnostics.csv": "directions/diagnostics.csv",
    "monotonicity_layer.csv": "monotonicity_layer.csv",
    "monotonicity_endpoint.csv": "monotonicity_endpoint.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    variants = [
        garimpo_uncertainty.data.read_run_metadata(run_dir)["variant"] for run_dir in args.runs
    ]
    if len(set(variants)) != len(variants):
        raise SystemExit(f"two runs of the same variant would merge into one model: {variants}")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, relative in TABLES.items():
        frames = []
        for run_dir, variant in zip(args.runs, variants, strict=True):
            frame = pd.read_csv(run_dir / relative)
            frames.append(frame if "model" in frame else frame.assign(model=variant))
        pd.concat(frames, ignore_index=True).to_csv(args.out / name, index=False)
        print(f"{args.out / name}: {sum(len(f) for f in frames)} rows")


if __name__ == "__main__":
    main()
