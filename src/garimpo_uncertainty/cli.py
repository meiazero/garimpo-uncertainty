"""`garimpo <command>`: every stage of the method, and the job that runs them in order.

Only the chosen command's module is imported, so a CPU stage such as compute_metrics never loads
torch, and a stage's own flags and help are those of its module.
"""

import importlib
import os
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
    "search_profile",
)
COMMANDS = {stage: f"garimpo_uncertainty.stages.{stage}" for stage in STAGES} | {
    "run": "garimpo_uncertainty.pipeline.run",
    "experiment": "garimpo_uncertainty.pipeline.experiment",
    "check": "garimpo_uncertainty.pipeline.check",
    "merge": "garimpo_uncertainty.pipeline.merge",
    "status": "garimpo_uncertainty.pipeline.status",
    "figures": "garimpo_uncertainty.figures.make",
    "data_figures": "garimpo_uncertainty.figures.data",
    "tune": "garimpo_uncertainty.tuning.tune",
}
USAGE = f"usage: garimpo {{{','.join(COMMANDS)}}} [arguments]; garimpo <command> --help for each"


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:  # noqa: PLR2004 - program and command
        sys.exit(USAGE)
    name = sys.argv[1]
    # The command's parser reads sys.argv, and names itself after sys.argv[0] in its messages.
    sys.argv = [f"garimpo {name}", *sys.argv[2:]]
    # cuBLAS reads this once, at its first call, and torch's deterministic algorithms need it.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    command = importlib.import_module(COMMANDS[name])
    # A command that loaded torch runs only deterministic kernels and fails on an op that has none,
    # so a rerun on the same GPU and software repeats every bit.
    if (torch := sys.modules.get("torch")) is not None:
        torch.use_deterministic_algorithms(mode=True)
    command.main()
