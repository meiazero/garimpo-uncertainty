"""Print a run's JSON config as bash assignments, refusing it before any stage when it is unusable.

    values=$(uv run --no-sync python scripts/run_config.py configs/full/unet_scratch.json)
    eval "$values"

A key left out, null, empty or unknown stops the job here, not as an empty argument after training.
"""

import argparse
import json
import math
import shlex
import sys
from pathlib import Path

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


def assignment(key: str, value: object) -> str:
    """Return one bash assignment; numbers keep their JSON spelling, lists become arrays."""
    if isinstance(value, list):
        return f"{key}=({' '.join(shlex.quote(str(item)) for item in value)})"
    text = "" if value is None else value if isinstance(value, str) else json.dumps(value)
    return f"{key}={shlex.quote(text)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("config", type=Path)
    path = parser.parse_args().config
    config = json.loads(path.read_text(encoding="utf-8"))
    found = problems(config)
    if found:
        sys.exit("\n".join(f"{path}: {line}" for line in found))
    for key in (*REQUIRED, ENCODER_WEIGHTS):
        print(assignment(key, config.get(key)))


if __name__ == "__main__":
    main()
