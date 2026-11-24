"""`garimpo <command>`: every stage of the method, and the job that runs them in order.

Only the chosen command's module is imported, so a CPU stage such as compute_metrics never loads
torch, and a stage's own flags and help are those of its module.
"""

import importlib
import sys

STAGES = (
    "download_data",
    "validate_data",
    "train",
    "infer",
    "fit_directions",
    "verify_monotonicity",
    "choose_layer",
    "compass_scores",
    "calibrate",
    "generate_intervals",
    "compute_metrics",
)
COMMANDS = {stage: f"garimpo_uncertainty.stages.{stage}" for stage in STAGES} | {
    "run": "garimpo_uncertainty.pipeline.run",
    "check": "garimpo_uncertainty.pipeline.check",
    "merge": "garimpo_uncertainty.pipeline.merge",
    "status": "garimpo_uncertainty.pipeline.status",
}
USAGE = f"usage: garimpo {{{','.join(COMMANDS)}}} [arguments]; garimpo <command> --help for each"


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:  # noqa: PLR2004 - program and command
        sys.exit(USAGE)
    name = sys.argv[1]
    # The command's parser reads sys.argv, and names itself after sys.argv[0] in its messages.
    sys.argv = [f"garimpo {name}", *sys.argv[2:]]
    importlib.import_module(COMMANDS[name]).main()
