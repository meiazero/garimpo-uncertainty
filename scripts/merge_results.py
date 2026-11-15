"""Stage: joins the small tables of several runs; a run is identified by (model, train_fraction)."""

import argparse
import json
from pathlib import Path

import pandas as pd

import garimpo_uncertainty.data
import garimpo_uncertainty.intervals
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


def layer_dirs(run_dir: Path) -> dict[str, Path]:
    """Every candidate layer and where its stages wrote, the chosen one first, at the run's root."""
    chosen = json.loads((run_dir / "chosen_layer.json").read_text(encoding="utf-8"))
    others = [c["layer"] for c in chosen["candidates"] if c["layer"] != chosen["layer"]]
    layers_dir = run_dir / garimpo_uncertainty.stages.LAYERS_DIR
    return {chosen["layer"]: run_dir} | {layer: layers_dir / layer for layer in others}


def read_status(status_dir: Path, stage: str) -> str:
    path = status_dir / f"{stage}.json"
    return json.loads(path.read_text())["status"] if path.exists() else "not_run"


def stage_statuses(run_dir: Path) -> dict[str, str]:
    statuses = {
        stage: read_status(run_dir / "status", stage)
        for stage in garimpo_uncertainty.stages.STAGE_ORDER
    }
    # Without a choice there is no list of the other layers to check.
    if statuses["choose_layer"] != "ok":
        return statuses
    others = list(layer_dirs(run_dir).items())[1:]
    return statuses | {
        f"{stage} [{layer}]": read_status(layer_dir / "status", stage)
        for layer, layer_dir in others
        for stage in garimpo_uncertainty.stages.LAYER_STAGES
    }


def layer_results(run_dir: Path) -> pd.DataFrame:
    frames = [
        pd.read_csv(layer_dir / "metrics" / "results.csv")
        for layer_dir in layer_dirs(run_dir).values()
    ]
    table = pd.concat(frames, ignore_index=True)
    # The other methods' rows repeat unchanged in every layer's results.csv.
    return table[table.method.isin(garimpo_uncertainty.intervals.LAYER_METHODS)]


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
        raise SystemExit(
            "every stage of every run, other layers included, must be ok to merge; nothing was "
            f"written, so any tables in {args.out} and in Drive are from an earlier merge:\n"
            + "\n".join(not_ok)
        )

    args.out.mkdir(parents=True, exist_ok=True)
    for name, relative in TABLES.items():
        frames = []
        for run_dir, identity in zip(args.runs, identities, strict=True):
            frame = read_table(run_dir / relative)
            frames.append(frame.assign(**{k: v for k, v in identity.items() if k not in frame}))
        pd.concat(frames, ignore_index=True).to_csv(args.out / name, index=False)
        print(f"{args.out / name}: {sum(len(f) for f in frames)} rows")
    by_layer = pd.concat([layer_results(run_dir) for run_dir in args.runs], ignore_index=True)
    by_layer.to_csv(args.out / "layer_results.csv", index=False)
    print(f"{args.out / 'layer_results.csv'}: {len(by_layer)} rows")


if __name__ == "__main__":
    main()
