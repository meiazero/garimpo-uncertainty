"""A run's values, composed from three files under configs/: base, model and scale.

configs/base.json holds what every run shares, configs/models/<model>.json the architecture and its
candidate layers, and configs/scales/<scale>.json the run's size and length. Each key lives in one
file only, so a value changes in one place. A key left out, null, empty, unknown or in another file
stops the runner before any stage, not as an empty argument after training.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

MODEL_KEYS = ("ARCH", "LAYERS")
# Required for the U-Net only; the ConvNet has no pretrained starting point to choose.
ENCODER_WEIGHTS = "ENCODER_WEIGHTS"
SCALE_KEYS = (
    "RUNS_DIR",
    "TRAIN_FRACTION",
    "EVALUATION_TILES",
    "EPOCHS",
    "STEPS_PER_EPOCH",
    "MIN_FRACTION_ASCENDING",
)
BASE_KEYS = (
    "SUBSET_SEED",
    "BATCH_SIZE",
    "EVAL_BATCH_SIZE",
    "NUM_CLASSES",
    "OPTIMIZER",
    "LR",
    "WEIGHT_DECAY",
    "LOSS",
    "POSTPROCESS",
    "METRICS",
    "AUGMENTATIONS",
    "CALLBACKS",
    "SEED",
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
    "CELL_PX",
)
# A list present in every run, and [] on purpose when a run takes no augmentation. null still
# refuses, as a value left out rather than chosen.
MAY_BE_EMPTY = ("AUGMENTATIONS",)
REQUIRED = (*MODEL_KEYS, *SCALE_KEYS, *BASE_KEYS)


class ConfigError(ValueError):
    """A run config that cannot run, with one line per problem."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = problems


def models(configs: Path) -> list[str]:
    """Return the name of every model in configs/models/."""
    return sorted(path.stem for path in (configs / "models").glob("*.json"))


def is_empty(value: object) -> bool:
    return value is None or value in ("", [])


def layer_files(configs: Path, scale: str, model: str) -> dict[Path, tuple[str, ...]]:
    """Return each file a run reads and the keys it may hold."""
    return {
        configs / "base.json": BASE_KEYS,
        configs / "models" / f"{model}.json": (*MODEL_KEYS, ENCODER_WEIGHTS),
        configs / "scales" / f"{scale}.json": SCALE_KEYS,
    }


def problems(values: dict, misplaced: list[str]) -> list[str]:
    """Return why the composed values cannot run, one line per key."""
    found = list(misplaced)
    found += [
        f"{key}: missing or empty"
        for key in REQUIRED
        if key not in values
        or values[key] is None
        or (key not in MAY_BE_EMPTY and is_empty(values[key]))
    ]
    if values.get("ARCH") == "unet" and is_empty(values.get(ENCODER_WEIGHTS)):
        found.append(f"{ENCODER_WEIGHTS}: missing or empty, and the U-Net needs it")
    # garimpo train takes more classes, but every stage after it reads one logit per pixel.
    if values.get("NUM_CLASSES") not in {None, 1}:
        found.append(f"NUM_CLASSES must be 1 for a run, got {values['NUM_CLASSES']!r}")
    if not isinstance(values.get("LAYERS", []), list):
        found.append("LAYERS: must be a list of layer names")
    floor = values.get("SCORE_FLOOR_HA")
    # calibrate checks the floor too, but only after training.
    is_number = isinstance(floor, int | float) and not isinstance(floor, bool)
    if not is_empty(floor) and not (is_number and math.isfinite(floor) and floor > 0):
        found.append(f"SCORE_FLOOR_HA must be a positive finite number, got '{floor}'")
    return found


@dataclass(frozen=True)
class RunConfig:
    """One model at one scale: every value its stages receive, and where the run writes."""

    scale: str
    model: str
    values: dict[str, Any]

    @classmethod
    def load(cls, configs: Path, scale: str, model: str) -> Self:
        """Compose the three files; raise ConfigError naming every key that cannot run."""
        files = layer_files(configs, scale, model)
        home = {key: path for path, keys in files.items() for key in keys}
        values: dict[str, Any] = {}
        misplaced = []
        for path, allowed in files.items():
            try:
                content = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise ConfigError([f"{path}: could not read the config: {error!r}"]) from error
            for key in set(content) - set(allowed):
                where = f"belongs in {home[key]}" if key in home else "unknown key"
                misplaced.append(f"{path}: {key}: {where}")
            values |= {key: value for key, value in content.items() if key in allowed}
        if found := problems(values, misplaced):
            raise ConfigError(found)
        return cls(scale, model, values)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    @property
    def run_dir(self) -> str:
        """runs/unet_scratch-f1.0 for the full U-Net from scratch, as every recorded run says."""
        return f"{self['RUNS_DIR']}/{self.model}-f{self['TRAIN_FRACTION']}"

    @property
    def log_name(self) -> str:
        return log_name(self.scale, self.model)


def log_name(scale: str, model: str) -> str:
    """Return the name of a run's log, outputs/logs/<scale>-<model>.log, kept across attempts."""
    return f"{scale}-{model}"
