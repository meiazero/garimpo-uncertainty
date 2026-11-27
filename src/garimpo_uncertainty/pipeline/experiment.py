"""`garimpo experiment`: the pilot until every model passed it on this code, then the full runs.

Each model runs as its own `garimpo run`, and an interrupted one stops the models after it. Each
scale that ran is then reported and merged: the pilot's tables into runs/pilot/tables, the full
ones into outputs. --force-from reaches only the full runs, as the pilot redoes itself until it
passes.
"""

import argparse
import subprocess
from pathlib import Path

import garimpo_uncertainty.config
import garimpo_uncertainty.pipeline.runner
import garimpo_uncertainty.pipeline.status

# garimpo run answers an interrupt with 130; a process killed by a signal ends with its negative.
INTERRUPTED = {130, -2, -9, -15}
TABLES = {"pilot": Path("runs/pilot/tables"), "full": Path("outputs")}


def run_scale(scale: str, options: list[str]) -> bool:
    """Run every model at `scale`; False once one is interrupted, which stops the rest."""
    for model in garimpo_uncertainty.config.models(garimpo_uncertainty.pipeline.status.CONFIGS):
        print(f"\n===== {scale} · {model} =====", flush=True)
        ended = subprocess.run(["garimpo", "run", scale, model, *options], check=False)  # noqa: S603, S607
        if ended.returncode in INTERRUPTED:
            print("Interrupted: the models after this one did not run.")
            return False
    return True


def run_scales(force_from: str | None) -> list[str]:
    """Run the pilot when it has not passed on this code, then the full runs once it has."""
    ran = []
    scale = garimpo_uncertainty.pipeline.status.next_scale()
    if scale == "pilot":
        ran.append("pilot")
        if not run_scale("pilot", []):
            return ran
        scale = garimpo_uncertainty.pipeline.status.next_scale()
        if scale == "pilot":
            print("A pilot did not pass on this commit, so the full runs wait: read its log above.")
    if scale == "full":
        ran.append("full")
        run_scale("full", ["--force-from", force_from] if force_from else [])
    return ran


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--force-from",
        choices=garimpo_uncertainty.pipeline.runner.STAGE_ORDER,
        help="redo this stage of the full runs and those after it, after a code fix",
    )
    parser.add_argument(
        "--since",
        type=float,
        default=0.0,
        help="the session's start: a log not written after it counts as not run",
    )
    args = parser.parse_args()
    for scale in run_scales(args.force_from):
        garimpo_uncertainty.pipeline.status.print_status(scale, args.since)
        merge = ["garimpo", "merge", "--out", str(TABLES[scale]), "--scale", scale]
        subprocess.run(merge, check=False)  # noqa: S603
