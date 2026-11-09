"""Stage: checks the store against the split; fails naming each check, always writes the report."""

import argparse
import json
from pathlib import Path

import pandas as pd

import garimpo_uncertainty.data
import garimpo_uncertainty.stages
import garimpo_uncertainty.validation


def check_slots(
    root: Path, split: pd.DataFrame, tile_ids: list[int], workers: int
) -> tuple[list[str], list[str]]:
    """Return why the store is unreadable, else which slots changed.

    Hashed once, before deciding to skip, and reused by the report: a pass reads the whole store.
    """
    unusable = garimpo_uncertainty.validation.unusable_store(root, split, tile_ids)
    if unusable:
        return unusable, []
    tiles = garimpo_uncertainty.data.tile_records(root, split, tile_ids)
    return [], garimpo_uncertainty.validation.slot_mismatches(root, tiles, workers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--split", type=Path, default="outputs/split.csv")
    parser.add_argument(
        "--train-fraction",
        type=float,
        choices=garimpo_uncertainty.data.TRAIN_FRACTIONS,
        required=True,
    )
    parser.add_argument("--subset-seed", type=int, required=True)
    parser.add_argument("--cell-px", type=int, default=garimpo_uncertainty.data.CELL_PX)
    parser.add_argument("--report", type=Path, required=True, help="one per fraction and seed")
    parser.add_argument(
        "--split-summary", type=Path, help="counts to match, with the train role reduced to fit"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    garimpo_uncertainty.stages.log_to_stdout()

    root = garimpo_uncertainty.data.data_root()
    split = pd.read_csv(args.split)
    tile_ids = garimpo_uncertainty.data.select_tiles(
        split, args.train_fraction, args.subset_seed
    ).tile_id.tolist()
    # Per selection: another fraction checks other tiles, so it must neither refuse nor reuse this.
    selection = garimpo_uncertainty.data.selection_name(args.train_fraction, args.subset_seed)

    # The fingerprint reads sidecars, so only reading the slots shows a store changed underneath.
    unusable, slot_problems = check_slots(root, split, tile_ids, args.workers)

    def work() -> dict:
        summary = None
        if args.split_summary:
            summary = garimpo_uncertainty.data.scale_summary(
                json.loads(args.split_summary.read_text()), args.train_fraction
            )
        # Unreadable when checked: fail on that even if readable by now, since no slot was hashed.
        if unusable:
            report, cells = {"problems": unusable}, pd.DataFrame()
        else:
            report, cells = garimpo_uncertainty.validation.validate(
                root,
                split,
                tile_ids,
                args.cell_px,
                summary,
                workers=args.workers,
                slot_problems=slot_problems,
            )
        garimpo_uncertainty.stages.write_json(args.report, report)
        print(json.dumps(report, indent=2))
        if report["problems"]:
            raise ValueError("; ".join(report["problems"]))
        return {"cells": len(cells)}

    garimpo_uncertainty.stages.run_stage(
        root / "status" / selection,
        "validate_data",
        [args.report],
        garimpo_uncertainty.stages.fingerprint(
            args,
            [],
            hashed=garimpo_uncertainty.data.tiles_identity(root, split, tile_ids),
        ),
        work,
        force=args.force or bool(unusable or slot_problems),
    )


if __name__ == "__main__":
    main()
