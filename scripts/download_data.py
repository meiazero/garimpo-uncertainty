"""Login-node stage: downloads metadata, weights or Sentinel-2 cells; safe to rerun, it resumes."""

import argparse
from pathlib import Path

import pandas as pd
import rasterio.enums

import garimpo_uncertainty.data
import garimpo_uncertainty.download
import garimpo_uncertainty.stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    # Subparsers do not inherit allow_abbrev from the main parser.
    commands.add_parser(
        "metadata", help="Jasansky et al. archive from Zenodo, md5 checked", allow_abbrev=False
    )
    weights = commands.add_parser(
        "weights", help="Sentinel-2 MoCo file and ImageNet cache", allow_abbrev=False
    )
    weights.add_argument(
        "--encoder-weights", nargs="+", choices=["imagenet", "sentinel2_moco"], required=True
    )
    scenes = commands.add_parser("scenes", help="cells from Planetary Computer", allow_abbrev=False)
    scenes.add_argument("--split", type=Path, default="outputs/split.csv")
    scenes.add_argument("--scales", nargs="+", default=["Artisanal", "Industrial"])
    scenes.add_argument("--groups-per-role", type=int, help="whole artisanal groups kept per role")
    scenes.add_argument("--max-industrial-tiles", type=int)
    scenes.add_argument("--seed", type=int, default=0, help="subset selection")
    scenes.add_argument("--workers", type=int, default=4)
    scenes.add_argument(
        "--resampling-20m",
        choices=["nearest", "bilinear"],
        required=True,
        help="how B11 and B12 reach the 10 m grid",
    )
    scenes.add_argument("--force", action="store_true", help="redo every selected tile")
    args = parser.parse_args()

    root = garimpo_uncertainty.data.data_root()
    if args.command == "metadata":
        garimpo_uncertainty.download.download_metadata(root)
        return
    if args.command == "weights":
        garimpo_uncertainty.download.download_weights(root, args.encoder_weights)
        return

    selected = garimpo_uncertainty.data.select_tiles(
        pd.read_csv(args.split),
        args.scales,
        args.groups_per_role,
        args.max_industrial_tiles,
        args.seed,
    )

    def work() -> dict:
        garimpo_uncertainty.download.download_scenes(
            root,
            selected,
            args.workers,
            rasterio.enums.Resampling[args.resampling_20m],
            args.force,
        )
        report = pd.read_csv(root / "download_report.csv")
        return {"selected_tiles": len(selected), "statuses": report.status.value_counts().to_dict()}

    # No outputs listed: a rerun always checks the sidecars and fetches what is still pending.
    garimpo_uncertainty.stages.run_stage(
        root / "status",
        "download_scenes",
        [],
        False,
        garimpo_uncertainty.stages.fingerprint(args, []),
        work,
    )


if __name__ == "__main__":
    main()
