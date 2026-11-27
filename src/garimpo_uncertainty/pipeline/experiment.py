"""`garimpo experiment`: the pilot until every model passed it on this code, then the full runs.

Each model runs as its own `garimpo run`, and an interrupted one stops the models after it. Each
scale that ran is then reported, and merged when all its runs passed: the pilot's tables into
runs/pilot/tables, the full ones into outputs. --force-from reaches only the full runs, as the
pilot redoes itself until it passes. Exits 130 after an interrupt, 1 when a job or a merge
failed, else 0.
"""

import argparse
import signal
import subprocess
import sys
from pathlib import Path

import garimpo_uncertainty.config
import garimpo_uncertainty.pipeline.run
import garimpo_uncertainty.pipeline.runner
import garimpo_uncertainty.pipeline.status

# garimpo run answers an interrupt with 130; a process killed by a signal ends with its negative.
STOPPED = 130
INTERRUPTED = {STOPPED, -2, -9, -15}
TABLES = {"pilot": Path("runs/pilot/tables"), "full": Path("outputs")}
# This interpreter, so each job runs the code loaded here whatever PATH holds.
GARIMPO = [sys.executable, "-m", "garimpo_uncertainty"]
# The order main ran them in: a session cut short has the reference U-Net done first.
ORDER = ("unet_scratch", "unet_imagenet", "unet_sentinel2", "convnet")


def run_order(models: list[str]) -> list[str]:
    """Return `models` in ORDER, any other after them by name."""
    return sorted(models, key=lambda m: (ORDER.index(m) if m in ORDER else len(ORDER), m))


def run_model(command: list[str]) -> int:
    """Run one `garimpo run` and return its exit code, or STOPPED once it was interrupted.

    `garimpo run` needs the interrupt to stop its stage and still run the check; the second,
    Colab's SIGTERM, makes it kill the stage at once. The job gets a process group of its own, so
    a terminal's Ctrl+C, sent to the whole group, reaches it once, through here, as Colab's does.
    """
    job: list[subprocess.Popen] = []
    interrupts: list[int] = []

    def forward(signum: int, _frame: object) -> None:
        # A handler that returns, unlike a raise, leaves wait() running, so no second one is lost.
        interrupts.append(signum)
        for process in job:
            process.send_signal(signal.SIGINT if len(interrupts) == 1 else signal.SIGTERM)

    previous = {kind: signal.signal(kind, forward) for kind in (signal.SIGINT, signal.SIGTERM)}
    try:
        job.append(subprocess.Popen(command, process_group=0))  # noqa: S603 - checked words
        code = job[0].wait()
    finally:
        for kind, handler in previous.items():
            signal.signal(kind, handler)
    # A job that ended 0 as the interrupt came must still stop the models after it.
    return STOPPED if interrupts else code


def run_scale(scale: str, options: list[str]) -> int:
    """Run every model at `scale`; return STOPPED once one is interrupted, 1 if one failed."""
    failed = False
    configs = garimpo_uncertainty.pipeline.status.CONFIGS
    for model in run_order(garimpo_uncertainty.config.models(configs)):
        print(f"\n===== {scale} · {model} =====", flush=True)
        code = run_model([*GARIMPO, "run", scale, model, *options])
        if code in INTERRUPTED:
            print("Interrupted: the models after this one did not run.")
            return STOPPED
        failed = failed or code != 0
    return 1 if failed else 0


def run_scales(force_from: str | None) -> dict[str, int]:
    """Run the pilot when it has not passed on this code, then the full runs once it has.

    Return the exit code of each scale that ran, in the order they ran.
    """
    scale = garimpo_uncertainty.pipeline.status.next_scale()
    if scale != "pilot":
        return {"full": run_scale("full", ["--force-from", force_from] if force_from else [])}
    code = run_scale("pilot", [])
    if code == STOPPED:
        return {"pilot": code}
    if garimpo_uncertainty.pipeline.status.next_scale() == "pilot":
        print("A pilot did not pass on this commit, so the full runs wait: read its log above.")
        return {"pilot": code or 1}
    return {
        "pilot": code,
        "full": run_scale("full", ["--force-from", force_from] if force_from else []),
    }


def main() -> None:
    signal.signal(signal.SIGTERM, garimpo_uncertainty.pipeline.run.interrupt)
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
    codes = run_scales(args.force_from)
    code = max(codes.values())
    for scale, ended in codes.items():
        garimpo_uncertainty.pipeline.status.print_status(scale, args.since)
        # A run refused before it wrote a status leaves an earlier attempt's ok, which would merge.
        if ended != 0:
            print(
                f"Not merged: a {scale} run failed or stopped, so {TABLES[scale]} keeps its tables."
            )
            continue
        merge = [*GARIMPO, "merge", "--out", str(TABLES[scale]), "--scale", scale]
        merged = subprocess.run(merge, check=False)  # noqa: S603
        code = code or (1 if merged.returncode else 0)
    sys.exit(code)
