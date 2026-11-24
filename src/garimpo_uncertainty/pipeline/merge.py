"""Stage: joins the small tables of several runs; a run is identified by (model, train_fraction)."""

import argparse
from pathlib import Path

import pandas as pd

import garimpo_uncertainty.artifacts
import garimpo_uncertainty.config
import garimpo_uncertainty.data
import garimpo_uncertainty.intervals
import garimpo_uncertainty.pipeline.runner


def tables(run: garimpo_uncertainty.artifacts.RunPaths) -> dict[str, Path]:
    """Return each merged table's name and the file of `run` it takes its rows from."""
    metrics = {
        f"{name}.csv": run.metric_table(name)
        for name in garimpo_uncertainty.artifacts.METRIC_TABLES
    }
    return metrics | {
        "layer_diagnostics.csv": run.direction_diagnostics,
        "monotonicity_layer.csv": run.monotonicity("layer"),
        "monotonicity_endpoint.csv": run.monotonicity("endpoint"),
        "chosen_layers.csv": run.chosen_layer,
    }


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV as is, and chosen_layer.json as one row: its candidates are already tables."""
    if path.suffix == ".csv":
        return pd.read_csv(path)
    chosen = garimpo_uncertainty.artifacts.read_chosen_layer(path)
    return pd.DataFrame([{k: chosen[k] for k in ("layer", "min_fraction_ascending")}])


def layer_paths(run_dir: Path) -> dict[str, garimpo_uncertainty.artifacts.RunPaths]:
    """Every candidate layer and where its stages wrote, the chosen one first, at the run's root."""
    run = garimpo_uncertainty.artifacts.RunPaths.of(run_dir)
    chosen = garimpo_uncertainty.artifacts.read_chosen_layer(run.chosen_layer)
    others = [c["layer"] for c in chosen["candidates"] if c["layer"] != chosen["layer"]]
    return {chosen["layer"]: run} | {layer: run.other_layer(layer) for layer in others}


def status_of(status_dir: Path, stage: str) -> str:
    return garimpo_uncertainty.pipeline.runner.read_status(status_dir, stage)["status"]


def stage_statuses(run_dir: Path) -> dict[str, str]:
    run = garimpo_uncertainty.artifacts.RunPaths.of(run_dir)
    statuses = {
        stage: status_of(run.status, stage)
        for stage in garimpo_uncertainty.pipeline.runner.STAGE_ORDER
    }
    # Without a choice there is no list of the other layers to check.
    if statuses["choose_layer"] != "ok":
        return statuses
    try:
        others = list(layer_paths(run_dir).items())[1:]
    except garimpo_uncertainty.artifacts.UNREADABLE as error:
        return statuses | {run.chosen_layer.name: f"unreadable ({type(error).__name__}: {error})"}
    return statuses | {
        f"{stage} [{name}]": status_of(paths.layer_status, stage)
        for name, paths in others
        for stage in garimpo_uncertainty.pipeline.runner.LAYER_STAGES
    }


def layer_results(run_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(layer.results) for layer in layer_paths(run_dir).values()]
    table = pd.concat(frames, ignore_index=True)
    return table[garimpo_uncertainty.intervals.depends_on_layer(table)]


def merged_table(runs: list[Path], identities: list[dict], name: str) -> pd.DataFrame:
    frames = []
    for run_dir, identity in zip(runs, identities, strict=True):
        frame = read_table(tables(garimpo_uncertainty.artifacts.RunPaths.of(run_dir))[name])
        frames.append(frame.assign(**{k: v for k, v in identity.items() if k not in frame}))
    return pd.concat(frames, ignore_index=True)


def scale_runs(scale: str) -> list[Path]:
    """Return every model's run directory at `scale`; stop on a config that cannot run."""
    configs = Path("configs")
    try:
        return [
            Path(garimpo_uncertainty.config.RunConfig.load(configs, scale, model).run_dir)
            for model in garimpo_uncertainty.config.models(configs)
        ]
    except garimpo_uncertainty.config.ConfigError as error:
        raise SystemExit(str(error)) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    runs = parser.add_mutually_exclusive_group(required=True)
    runs.add_argument("--runs", type=Path, nargs="+")
    runs.add_argument("--scale", help="every model of configs/models/ at this scale")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.scale:
        args.runs = scale_runs(args.scale)

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

    # Read everything first: a table missing under an ok status must not leave a partial merge.
    names = tables(garimpo_uncertainty.artifacts.RunPaths.of(args.runs[0]))
    merged = {name: merged_table(args.runs, identities, name) for name in names}
    merged["layer_results.csv"] = pd.concat(
        [layer_results(run_dir) for run_dir in args.runs], ignore_index=True
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in merged.items():
        table.to_csv(args.out / name, index=False)
        print(f"{args.out / name}: {len(table)} rows")
