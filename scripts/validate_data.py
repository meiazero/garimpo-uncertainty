"""Stage: checks cells against the split; fails naming each check, always writes the report."""

import argparse
import json
from pathlib import Path

import pandas as pd

import garimpo_uncertainty.data
import garimpo_uncertainty.stages
import garimpo_uncertainty.validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--split", type=Path, default="outputs/split.csv")
    parser.add_argument("--cell-px", type=int, default=garimpo_uncertainty.data.CELL_PX)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--split-summary", type=Path, help="counts to match; full dataset only, not a subset"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = garimpo_uncertainty.data.data_root()
    cells_path = root / "validation_cells.csv"

    def work() -> dict:
        summary = json.loads(args.split_summary.read_text()) if args.split_summary else None
        report, cells = garimpo_uncertainty.validation.validate(
            root, pd.read_csv(args.split), args.cell_px, summary, args.workers
        )
        garimpo_uncertainty.stages.write_json(args.report, report)
        with garimpo_uncertainty.stages.atomic(cells_path) as temporary:
            cells.to_csv(temporary, index=False)
        print(json.dumps(report, indent=2))
        if report["problems"]:
            raise ValueError("; ".join(report["problems"]))
        return {"cells": len(cells)}

    garimpo_uncertainty.stages.run_stage(
        root / "status",
        "validate_data",
        [args.report, cells_path],
        args.force,
        garimpo_uncertainty.stages.fingerprint(args, [root / "manifest.csv"]),
        work,
    )


if __name__ == "__main__":
    main()
