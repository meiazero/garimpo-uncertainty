"""Stage: joins the small tables of several runs; a run is identified by (model, train_fraction)."""

import argparse
import json
from pathlib import Path

import pandas as pd

import garimpo_uncertainty.data
import garimpo_uncertainty.stages

TABLES = {
    "results.csv": "metrics/results.csv",
    "width_comparisons.csv": "metrics/width_comparisons.csv",
    "coverage_comparisons.csv": "metrics/coverage_comparisons.csv",
    "q1_spearman.csv": "metrics/q1_spearman.csv",
    "coverage_by_stratum.csv": "metrics/coverage_by_stratum.csv",
    "score_relation.csv": "metrics/score_relation.csv",
    "layer_diagnostics.csv": "directions/diagnostics.csv",
    "monotonicity_layer.csv": "monotonicity_layer.csv",
    "monotonicity_endpoint.csv": "monotonicity_endpoint.csv",
    "chosen_layers.csv": "chosen_layer.json",
}


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV as is, and chosen_layer.json as one row: its candidates are already tables."""
    if path.suffix == ".csv":
        return pd.read_csv(path)
    chosen = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame([{k: chosen[k] for k in ("layer", "min_fraction_ascending")}])


def stage_statuses(run_dir: Path) -> dict[str, str]:
    statuses = {}
    for stage in garimpo_uncertainty.stages.STAGE_ORDER:
        path = run_dir / "status" / f"{stage}.json"
        statuses[stage] = json.loads(path.read_text())["status"] if path.exists() else "not_run"
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    metadata = [garimpo_uncertainty.data.read_run_metadata(run_dir) for run_dir in args.runs]
    identities = [{"model": m["variant"], "train_fraction": m["train_fraction"]} for m in metadata]
    keys = [tuple(identity.values()) for identity in identities]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"two runs of the same model and train_fraction would merge: {keys}")
    # A failed stage leaves its earlier tables, and a restore from Drive brings deleted ones back.
    not_ok = [
        f"{run_dir}: {stage} is {status}"
        for run_dir in args.runs
        for stage, status in stage_statuses(run_dir).items()
        if status != "ok"
    ]
    if not_ok:
        raise SystemExit("every stage of every run must be ok to merge:\n" + "\n".join(not_ok))

    args.out.mkdir(parents=True, exist_ok=True)
    for name, relative in TABLES.items():
        frames = []
        for run_dir, identity in zip(args.runs, identities, strict=True):
            frame = read_table(run_dir / relative)
            frames.append(frame.assign(**{k: v for k, v in identity.items() if k not in frame}))
        pd.concat(frames, ignore_index=True).to_csv(args.out / name, index=False)
        print(f"{args.out / name}: {sum(len(f) for f in frames)} rows")


if __name__ == "__main__":
    main()
