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
        "weights", help="ImageNet and Sentinel-2 MoCo weight files", allow_abbrev=False
    )
    weights.add_argument(
        "--encoder-weights", nargs="+", choices=["imagenet", "sentinel2_moco"], required=True
    )
    scenes = commands.add_parser("scenes", help="cells from Planetary Computer", allow_abbrev=False)
    scenes.add_argument("--split", type=Path, default="outputs/split.csv")
    scenes.add_argument(
        "--train-fraction",
        type=float,
        choices=garimpo_uncertainty.data.TRAIN_FRACTIONS,
        required=True,
        help="share of the training tiles; validation, calibration and test always stay whole",
    )
    scenes.add_argument("--subset-seed", type=int, required=True, help="picks the training tiles")
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

    split = pd.read_csv(args.split)
    selected = garimpo_uncertainty.data.select_tiles(split, args.train_fraction, args.subset_seed)

    def work() -> dict:
        garimpo_uncertainty.download.download_scenes(
            root,
            split,
            selected,
            args.workers,
            rasterio.enums.Resampling[args.resampling_20m],
            args.force,
        )
        statuses = garimpo_uncertainty.data.tile_records(
            root, split, selected.tile_id.tolist()
        ).status.value_counts()
        print(statuses.to_string())
        return {"selected_tiles": len(selected), "statuses": statuses.to_dict()}

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
