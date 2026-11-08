"""Stage: choose the COMPASS-J layer from the validation diagnostics; writes chosen_layer.json.

Cheung et al. (2026), §4.1, recommend layers with high explained variance of the Jacobians' first
principal component, verified by the sweep of Algorithm 2, and set no threshold. This repo makes
that a rule: the highest variance among layers ascending on at least the given share of validation
cells, in both directions, since both are calibrated on it. A tie goes to the first of LAYERS.
"""

import argparse
from pathlib import Path

import pandas as pd

import garimpo_uncertainty.stages

DIRECTIONS = ("pca", "svm")


def candidates(diagnostics: pd.DataFrame, monotonicity: pd.DataFrame) -> pd.DataFrame:
    """One row per layer: explained variance and the validation fraction ascending per direction."""
    validation = monotonicity[
        (monotonicity.role == "validation") & monotonicity.direction.isin(DIRECTIONS)
    ]
    ascending = validation.pivot(index="layer", columns="direction", values="fraction_ascending")
    swept = set(ascending.dropna().index)
    if unmatched := swept ^ set(diagnostics.layer):
        raise ValueError(
            f"layers {sorted(unmatched)} are not in both directions/diagnostics.csv and the "
            f"validation rows of monotonicity_layer.csv for {DIRECTIONS}; rerun with one LAYERS"
        )
    table = diagnostics[["layer", "pca_explained_variance"]].set_index("layer")
    return table.join(ascending.add_prefix("fraction_ascending_"))


def choose(table: pd.DataFrame, minimum: float) -> str:
    weakest = table.filter(like="fraction_ascending_").min(axis=1)
    table["qualifies"] = weakest >= minimum
    if not table.qualifies.any():
        raise ValueError(
            f"no layer is ascending on at least {minimum} of the validation cells in both "
            f"directions; the best is {weakest.idxmax()} at {weakest.max()}. Add layers to "
            "LAYERS, or lower MIN_FRACTION_ASCENDING with its reason in docs/decisions.md"
        )
    # idxmax keeps the first of equal values, so a tie follows the diagnostics' row order.
    return str(table[table.qualifies].pca_explained_variance.idxmax())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--min-fraction-ascending", type=float, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    garimpo_uncertainty.stages.log_to_stdout()

    diagnostics_path = args.run_dir / "directions" / "diagnostics.csv"
    monotonicity_path = args.run_dir / "monotonicity_layer.csv"
    output = args.run_dir / "chosen_layer.json"

    def work() -> dict:
        # A refusal must not leave an earlier choice for the pipeline check to print as this one.
        output.unlink(missing_ok=True)
        table = candidates(pd.read_csv(diagnostics_path), pd.read_csv(monotonicity_path))
        layer = choose(table, args.min_fraction_ascending)
        garimpo_uncertainty.stages.write_json(
            output,
            {
                "layer": layer,
                "min_fraction_ascending": args.min_fraction_ascending,
                "candidates": table.reset_index().to_dict(orient="records"),
            },
        )
        return {"layer": layer, "candidates": len(table)}

    garimpo_uncertainty.stages.run_stage(
        args.run_dir / "status",
        "choose_layer",
        [output],
        garimpo_uncertainty.stages.fingerprint(args, [diagnostics_path, monotonicity_path]),
        work,
        force=args.force,
    )


if __name__ == "__main__":
    main()
