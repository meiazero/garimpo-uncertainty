"""The keys of a run's JSON config, and why a config cannot run.

A key left out, null, empty or unknown stops scripts/run.py before any stage, not as an empty
argument after training.
"""

import math

REQUIRED = (
    "ARCH",
    "RUN_DIR",
    "TRAIN_FRACTION",
    "SUBSET_SEED",
    "EPOCHS",
    "STEPS_PER_EPOCH",
    "BATCH_SIZE",
    "EVAL_BATCH_SIZE",
    "LR",
    "WEIGHT_DECAY",
    "LOSS",
    "SEED",
    "LAYERS",
    "MIN_FRACTION_ASCENDING",
    "SVM_C",
    "SVM_CLASS_WEIGHT",
    "SVM_PIXELS_PER_CELL",
    "SWEEP_STEPS",
    "SWEEP_CELLS",
    "BETA_RANGE_LOGITS",
    "BETA_RANGE_FEATURES",
    "DIRECTION_TEMPERATURE",
    "SEARCH_THRESHOLD_LOGITS",
    "SEARCH_THRESHOLD_FEATURES",
    "CELLS",
    "SCORE_FLOOR_HA",
    "BOUNDARY_TOLERANCE_PX",
)
# Required for the U-Net only; the ConvNet has no pretrained starting point to choose.
ENCODER_WEIGHTS = "ENCODER_WEIGHTS"


def is_empty(value: object) -> bool:
    return value is None or value in ("", [])


def problems(config: dict) -> list[str]:
    """Return why the config cannot run, one line per key."""
    found = [f"{key}: unknown key" for key in config if key not in {*REQUIRED, ENCODER_WEIGHTS}]
    found += [f"{key}: missing or empty" for key in REQUIRED if is_empty(config.get(key))]
    if config.get("ARCH") == "unet" and is_empty(config.get(ENCODER_WEIGHTS)):
        found.append(f"{ENCODER_WEIGHTS}: missing or empty, and the U-Net needs it")
    if not isinstance(config.get("LAYERS", []), list):
        found.append("LAYERS: must be a list of layer names")
    floor = config.get("SCORE_FLOOR_HA")
    # calibrate.py checks the floor too, but only after training.
    is_number = isinstance(floor, int | float) and not isinstance(floor, bool)
    if not is_empty(floor) and not (is_number and math.isfinite(floor) and floor > 0):
        found.append(f"SCORE_FLOOR_HA must be a positive finite number, got '{floor}'")
    return found
