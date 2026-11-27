"""`garimpo status <scale>`: every model's run at a scale, its values, checkpoint and last verdict.

`auto` is the full scale once any full run has saved a checkpoint, and the pilot before that, so a
session after the first full epoch continues the full runs. `--next` prints the scale to run next
instead: the pilot until all of its runs passed on the code checked out now, then the full one, so
no full run starts or goes on with code the pilot has not passed. Only files are read, and an
unreadable one is reported, never raised: the notebook calls this between runs.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import garimpo_uncertainty.artifacts
import garimpo_uncertainty.config
import garimpo_uncertainty.files
import garimpo_uncertainty.pipeline.runner

CONFIGS = Path("configs")
LOGS = Path("outputs/logs")
AUTO = "auto"


def load(scale: str, model: str) -> garimpo_uncertainty.config.RunConfig | None:
    try:
        return garimpo_uncertainty.config.RunConfig.load(CONFIGS, scale, model)
    except garimpo_uncertainty.config.ConfigError as error:
        print(f"{scale} {model}: {error}")
        return None


def has_checkpoint(config: garimpo_uncertainty.config.RunConfig) -> bool:
    """Tell whether the run saved last.pt, which it does after its first epoch."""
    run = garimpo_uncertainty.artifacts.RunPaths.of(Path(config.run_dir))
    return run.last_checkpoint.exists()


def checkpointed(scale: str) -> list[str]:
    configs = [load(scale, model) for model in garimpo_uncertainty.config.models(CONFIGS)]
    return [config.model for config in configs if config is not None and has_checkpoint(config)]


def choose_scale(scale: str) -> str:
    if scale != AUTO:
        return scale
    return "full" if checkpointed("full") else "pilot"


def passed_on_this_code(config: garimpo_uncertainty.config.RunConfig) -> bool:
    """Tell whether the run's last job passed on the commits checked out now, with no local edit."""
    path = garimpo_uncertainty.artifacts.RunPaths.of(Path(config.run_dir)).verdict
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False  # its job never ended, or ran before verdicts were kept
    except (OSError, ValueError) as error:
        print(f"{config.log_name}: no verdict on record ({error})")
        return False
    now = garimpo_uncertainty.files.code_versions()
    clean = not now["root_dirty"] and not now["compass_dirty"]
    return (
        clean and recorded.get("verdict") == "PASS" and all(recorded.get(k) == now[k] for k in now)
    )


def next_scale() -> str:
    """Return the pilot until every pilot run passed on this code, then the full scale."""
    pilots = [load("pilot", model) for model in garimpo_uncertainty.config.models(CONFIGS)]
    if all(config is not None and passed_on_this_code(config) for config in pilots):
        return "full"
    return "pilot"


def last_verdict(config: garimpo_uncertainty.config.RunConfig, since: float) -> str:
    """Return the verdict line of the last attempt in the run's log, or why there is none."""
    log = LOGS / f"{config.log_name}.log"
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
        modified = log.stat().st_mtime
    except OSError as error:
        return f"NO LOG: {error}"
    # Restored from Drive with its time: the run died before writing, or never ran.
    if modified < since:
        return "NOT RUN since the session started: read the run cell's output"
    attempt = text.rsplit(garimpo_uncertainty.pipeline.runner.ATTEMPT, 1)[-1]
    verdicts = garimpo_uncertainty.pipeline.runner.VERDICTS
    found = [line for line in attempt.splitlines() if any(v in line for v in verdicts)]
    return found[-1] if found else "NO VERDICT: the job or its check died; see the log"


def print_status(scale: str, since: float) -> None:
    print(f"scale: {scale}")
    print("full runs with a checkpoint, which continue:", ", ".join(checkpointed("full")) or "none")
    print("pilot runs with a checkpoint:", ", ".join(checkpointed("pilot")) or "none")
    loaded = (load(scale, model) for model in garimpo_uncertainty.config.models(CONFIGS))
    configs = [config for config in loaded if config is not None]
    if not configs:
        return
    keys = (
        *garimpo_uncertainty.config.MODEL_KEYS,
        garimpo_uncertainty.config.ENCODER_WEIGHTS,
        *garimpo_uncertainty.config.SCALE_KEYS,
        *garimpo_uncertainty.config.BASE_KEYS,
    )
    # As written in the files: a table of floats would print 1e-15 as 0.0.
    values = {
        config.model: {"run dir": config.run_dir}
        | {key: str(config.values.get(key, "")) for key in keys}
        for config in configs
    }
    print(pd.DataFrame(values).to_string())
    print("\nlast attempt of each run")
    for config in configs:
        print(f"{config.log_name}: {last_verdict(config, since)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("scale", help=f"a file of configs/scales/, or {AUTO}")
    parser.add_argument(
        "--since",
        type=float,
        default=0.0,
        help="a log not written after this Unix time counts as not run; Colab passes the session's "
        "start, since a log restored from Drive keeps its time",
    )
    parser.add_argument(
        "--choose", action="store_true", help="print only the scale, for the notebook to read"
    )
    parser.add_argument(
        "--next",
        action="store_true",
        help="print only the scale to run next: the pilot until all of it passed on this code",
    )
    args = parser.parse_args()
    if args.next:
        print(next_scale())
        return
    scale = choose_scale(args.scale)
    if args.choose:
        print(scale)
        return
    print_status(scale, args.since)
