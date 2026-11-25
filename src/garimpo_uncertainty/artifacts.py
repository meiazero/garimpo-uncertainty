"""Where each file of a run lives, named once for the stage that writes it and every reader.

A run has two roots. The run root holds what training, inference, the directions and the layer
choice write; the layer root holds what the stages of one COMPASS-J layer write. The chosen layer's
root is the run root, and every other layer of LAYERS has its own under layers/<layer>/, so the
stages of a layer read the shared predictions from one root and their own scores from the other.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

LAYERS_DIR = "layers"
# What a broken chosen_layer.json raises: unreadable, not JSON, not an object, or a key missing.
UNREADABLE = (OSError, ValueError, KeyError, TypeError)
METRIC_TABLES = (
    "results",
    "width_comparisons",
    "coverage_comparisons",
    "q1_spearman",
    "coverage_by_stratum",
    "score_relation",
)


@dataclass(frozen=True)
class RunPaths:
    root: Path
    layer_root: Path

    @classmethod
    def of(cls, run_dir: Path, layer_dir: Path | None = None) -> Self:
        """Return the paths of a run whose layer stages write to `layer_dir`, or to the run root."""
        return cls(Path(run_dir), Path(layer_dir or run_dir))

    def other_layer(self, layer: str) -> Self:
        """Return the paths of the same run for a layer other than the chosen one."""
        return type(self)(self.root, self.root / LAYERS_DIR / layer)

    @property
    def status(self) -> Path:
        return self.root / "status"

    @property
    def run_meta(self) -> Path:
        return self.root / "run_meta.json"

    @property
    def best_model(self) -> Path:
        return self.root / "best_model.pt"

    @property
    def last_checkpoint(self) -> Path:
        return self.root / "last.pt"

    @property
    def history(self) -> Path:
        return self.root / "history.csv"

    @property
    def predictions(self) -> Path:
        """Every cell's area terms, aligned with the run's cell table."""
        return self.root / "predictions" / "cells.npz"

    def probabilities(self, role: str) -> Path:
        return self.root / "predictions" / f"{role}.npy"

    def directions(self, layer: str) -> Path:
        return self.root / "directions" / f"{layer}.npz"

    @property
    def direction_diagnostics(self) -> Path:
        return self.root / "directions" / "diagnostics.csv"

    def monotonicity(self, use: str) -> Path:
        return self.root / f"monotonicity_{use}.csv"

    def monotonicity_curves(self, use: str) -> Path:
        return self.root / f"monotonicity_{use}_curves.npz"

    @property
    def chosen_layer(self) -> Path:
        return self.root / "chosen_layer.json"

    @property
    def layer_status(self) -> Path:
        return self.layer_root / "status"

    @property
    def scores(self) -> Path:
        return self.layer_root / "scores" / "scores.npz"

    @property
    def quantiles(self) -> Path:
        return self.layer_root / "calibration" / "quantiles.npz"

    @property
    def intervals(self) -> Path:
        return self.layer_root / "intervals" / "intervals.npz"

    def metric_table(self, name: str) -> Path:
        return self.layer_root / "metrics" / f"{name}.csv"

    @property
    def results(self) -> Path:
        return self.metric_table("results")

    @property
    def cell_metrics(self) -> Path:
        return self.layer_root / "metrics" / "cell_metrics.npz"


def read_chosen_layer(path: Path) -> dict:
    """Return chosen_layer.json; raise one of UNREADABLE when it cannot say which layer won."""
    chosen = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(chosen, dict):
        raise TypeError(f"a JSON {type(chosen).__name__}, not an object")
    if absent := {"layer", "min_fraction_ascending", "candidates"} - set(chosen):
        raise KeyError(f"no {sorted(absent)}")
    return chosen
