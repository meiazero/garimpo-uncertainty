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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, relative in TABLES.items():
        frames = []
        for run_dir in args.runs:
            arch = garimpo_uncertainty.data.read_run_metadata(run_dir)["arch"]
            frame = pd.read_csv(run_dir / relative)
            frames.append(frame if "model" in frame else frame.assign(model=arch))
        pd.concat(frames, ignore_index=True).to_csv(args.out / name, index=False)
        print(f"{args.out / name}: {sum(len(f) for f in frames)} rows")


if __name__ == "__main__":
    main()
