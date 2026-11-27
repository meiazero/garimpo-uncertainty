"""The flags each stage receives from a run, declared once, for the runner and the tests alike.

Each flag names the value it takes: a key of the run's config, or one of the job's own values in
capitals too, such as CHECKPOINT or LAYER. Every name must be given, so a misspelled one fails
instead of leaving the stage at its default; None leaves the flag out, as the ConvNet's
ENCODER_WEIGHTS, a list becomes one word per item, and an object, such as a loss with options,
one word of JSON.
"""

import json
from dataclasses import dataclass

Flags = tuple[tuple[str, str], ...]

# What every stage reading the trained model shares.
CHECKPOINT_FLAGS: Flags = (
    ("--checkpoint", "CHECKPOINT"),
    ("--run-dir", "RUN_DIR"),
    ("--batch-size", "EVAL_BATCH_SIZE"),
    ("--num-workers", "NUM_WORKERS"),
)
# The sweep's beta_max is the search range, so it covers the domain the search walks.
SWEEP_FLAGS: Flags = (
    *CHECKPOINT_FLAGS,
    ("--beta-max-logits", "BETA_RANGE_LOGITS"),
    ("--beta-max-features", "BETA_RANGE_FEATURES"),
    ("--steps", "SWEEP_STEPS"),
    ("--cells-per-role", "SWEEP_CELLS"),
    ("--seed", "SEED"),
    ("--direction-temperature", "DIRECTION_TEMPERATURE"),
)
# Where a layer other than the chosen one writes; None for the chosen layer, at the run's root.
LAYER_DIR_FLAGS: Flags = (("--layer-dir", "LAYER_DIR"),)


@dataclass(frozen=True)
class Stage:
    """A stage's `garimpo` command, the flags it takes from the run, and its fixed words."""

    command: str
    flags: Flags
    fixed: tuple[str, ...] = ()


STAGES = {
    "validate_data": Stage(
        "validate_data",
        (
            ("--train-fraction", "TRAIN_FRACTION"),
            ("--subset-seed", "SUBSET_SEED"),
            ("--evaluation-tiles", "EVALUATION_TILES"),
            ("--report", "VALIDATION_REPORT"),
            ("--split-summary", "SPLIT_SUMMARY"),
            ("--cell-px", "CELL_PX"),
        ),
    ),
    "train": Stage(
        "train",
        (
            ("--arch", "ARCH"),
            # Passed whenever set, so the training alone decides what fits the architecture.
            ("--encoder-weights", "ENCODER_WEIGHTS"),
            ("--run-dir", "RUN_DIR"),
            ("--train-fraction", "TRAIN_FRACTION"),
            ("--subset-seed", "SUBSET_SEED"),
            ("--evaluation-tiles", "EVALUATION_TILES"),
            ("--epochs", "EPOCHS"),
            ("--steps-per-epoch", "STEPS_PER_EPOCH"),
            ("--batch-size", "BATCH_SIZE"),
            ("--num-classes", "NUM_CLASSES"),
            ("--optimizer", "OPTIMIZER"),
            ("--lr", "LR"),
            ("--weight-decay", "WEIGHT_DECAY"),
            ("--loss", "LOSS"),
            ("--postprocess", "POSTPROCESS"),
            ("--metrics", "METRICS"),
            ("--augmentations", "AUGMENTATIONS"),
            ("--callbacks", "CALLBACKS"),
            ("--seed", "SEED"),
            ("--num-workers", "NUM_WORKERS"),
        ),
    ),
    "infer": Stage("infer", CHECKPOINT_FLAGS),
    "fit_directions": Stage(
        "fit_directions",
        (
            *CHECKPOINT_FLAGS,
            ("--layers", "LAYERS"),
            ("--svm-c", "SVM_C"),
            ("--svm-class-weight", "SVM_CLASS_WEIGHT"),
            ("--svm-pixels-per-cell", "SVM_PIXELS_PER_CELL"),
            ("--seed", "SEED"),
            ("--direction-temperature", "DIRECTION_TEMPERATURE"),
        ),
    ),
    "layer_monotonicity": Stage(
        "verify_monotonicity", (*SWEEP_FLAGS, ("--layers", "LAYERS")), ("--use", "layer")
    ),
    "choose_layer": Stage(
        "choose_layer",
        (("--run-dir", "RUN_DIR"), ("--min-fraction-ascending", "MIN_FRACTION_ASCENDING")),
    ),
    "endpoint_monotonicity": Stage(
        "verify_monotonicity", (*SWEEP_FLAGS, ("--layers", "LAYER")), ("--use", "endpoint")
    ),
    "compass_scores": Stage(
        "compass_scores",
        (
            *CHECKPOINT_FLAGS,
            ("--layer", "LAYER"),
            ("--beta-range-logits", "BETA_RANGE_LOGITS"),
            ("--beta-range-features", "BETA_RANGE_FEATURES"),
            ("--search-threshold-logits", "SEARCH_THRESHOLD_LOGITS"),
            ("--search-threshold-features", "SEARCH_THRESHOLD_FEATURES"),
            ("--direction-temperature", "DIRECTION_TEMPERATURE"),
            *LAYER_DIR_FLAGS,
        ),
    ),
    "calibrate": Stage(
        "calibrate",
        (
            ("--run-dir", "RUN_DIR"),
            ("--cells", "CELLS"),
            ("--score-floor-ha", "SCORE_FLOOR_HA"),
            ("--seed", "SEED"),
            ("--lightgbm-jobs", "NUM_WORKERS"),
            *LAYER_DIR_FLAGS,
        ),
    ),
    "generate_intervals": Stage(
        "generate_intervals", (*CHECKPOINT_FLAGS, ("--layer", "LAYER"), *LAYER_DIR_FLAGS)
    ),
    "compute_metrics": Stage(
        "compute_metrics",
        (
            ("--run-dir", "RUN_DIR"),
            ("--boundary-tolerance-px", "BOUNDARY_TOLERANCE_PX"),
            ("--seed", "SEED"),
            *LAYER_DIR_FLAGS,
        ),
    ),
    "search_profile": Stage("search_profile", (*CHECKPOINT_FLAGS, ("--seed", "SEED"))),
    "pipeline_check": Stage(
        "check",
        (
            ("--run-dir", "RUN_DIR"),
            ("--train-fraction", "TRAIN_FRACTION"),
            ("--subset-seed", "SUBSET_SEED"),
            ("--evaluation-tiles", "EVALUATION_TILES"),
            ("--job-exit-code", "JOB_EXIT_CODE"),
            ("--layers", "LAYERS"),
        ),
    ),
}


def arguments(stage: str, values: dict) -> list[str]:
    """Return the words `stage` receives after its name, each flag with the value it names."""
    spec = STAGES[stage]
    words = list(spec.fixed)
    for flag, name in spec.flags:
        value = values[name]
        if value is None:
            continue
        items = value if isinstance(value, list) else [value]
        words += [
            flag,
            *(json.dumps(item) if isinstance(item, dict) else str(item) for item in items),
        ]
    return words
