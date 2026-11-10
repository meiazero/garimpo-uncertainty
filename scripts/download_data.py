"""Download stage, no GPU: metadata, weights or Sentinel-2 cells; safe to rerun, it resumes.

`lists` writes which tiles each training fraction uses, once per seed; `scenes` downloads one list.
`all` downloads the full store, the metadata and both starting weights, after checking the disk.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import rasterio.enums

import garimpo_uncertainty.data
import garimpo_uncertainty.download
import garimpo_uncertainty.stages

# A child of the package's logger, so its lines reach the terminal and the log file.
logger = logging.getLogger("garimpo_uncertainty.download_data")
# 18,864 slots of 3,670,016 bytes (docs/data-layout.md), plus 2 GB for Zenodo, weights and sidecars.
STORE_BYTES = 18864 * 3670016 + 2 * 1000**3
GB = 1000**3


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tile_lists(split_path: Path, subset_seed: int, directory: Path) -> None:
    """Write one list per training fraction, each complete: training subset plus the other roles."""
    split = pd.read_csv(split_path)
    cells_per_tile = (garimpo_uncertainty.data.TILE_PX // garimpo_uncertainty.data.CELL_PX) ** 2
    for fraction in garimpo_uncertainty.data.TRAIN_FRACTIONS:
        selected = garimpo_uncertainty.data.select_tiles(split, fraction, subset_seed)
        tile_list = {
            "train_fraction": fraction,
            "subset_seed": subset_seed,
            "split_sha256": file_sha256(split_path),
            "tiles": selected[["tile_id", "role"]].sort_values("tile_id").to_dict("records"),
        }
        path = directory / f"tiles_{round(fraction * 100)}pct.json"
        garimpo_uncertainty.stages.write_json(path, tile_list)
        print(f"{path}: {len(selected)} tiles, {len(selected) * cells_per_tile} cells")


def listed_tiles(list_path: Path, split_path: Path, split: pd.DataFrame) -> pd.DataFrame:
    """Return the split's rows a list names; a list from another split would mean other slots."""
    tile_list = json.loads(list_path.read_text(encoding="utf-8"))
    current = file_sha256(split_path)
    if tile_list["split_sha256"] != current:
        raise ValueError(
            f"{list_path} has split_sha256 {tile_list['split_sha256'][:12]}, but {split_path} is "
            f"{current[:12]}: restore that split, or regenerate the lists with "
            "download_data.py lists"
        )
    # Dropped tiles count as absent: they have no slot in the store.
    tiles = garimpo_uncertainty.data.tile_table(split)
    listed = {tile["tile_id"] for tile in tile_list["tiles"]}
    absent = sorted(listed - set(tiles.tile_id))
    if absent:
        raise ValueError(
            f"{list_path} names {len(absent)} tiles that {split_path} does not keep, "
            f"for example {absent[:5]}"
        )
    return tiles[tiles.tile_id.isin(listed)]


def download_scenes_stage(root: Path, args: argparse.Namespace) -> None:
    """Download the cells of one tile list, as the stage that records download_scenes.json."""

    def work() -> dict:
        split = pd.read_csv(args.split)
        selected = listed_tiles(args.tiles, args.split, split)
        garimpo_uncertainty.download.download_scenes(
            root,
            split,
            selected,
            args.workers,
            rasterio.enums.Resampling[args.resampling_20m],
            force=args.force,
        )
        statuses = garimpo_uncertainty.data.tile_records(
            root, split, selected.tile_id.tolist()
        ).status.value_counts()
        logger.info("tiles of this list by status:\n%s", statuses.to_string())
        return {"selected_tiles": len(selected), "statuses": statuses.to_dict()}

    # No outputs listed: a rerun always checks the sidecars and fetches what is still pending.
    garimpo_uncertainty.stages.run_stage(
        root / "status",
        "download_scenes",
        [],
        garimpo_uncertainty.stages.fingerprint(args, []),
        work,
        force=False,
    )


def check_disk(root: Path) -> None:
    """Stop before downloading when the disk cannot hold the store, counting what it holds."""
    held = sum(path.stat().st_blocks * 512 for path in root.rglob("*") if path.is_file())
    available = shutil.disk_usage(root).free + held
    if available < STORE_BYTES:
        sys.exit(
            f"{root}: needs {STORE_BYTES // GB} GB, has {available // GB} GB "
            "(free plus already downloaded)"
        )
    logger.info("%s: %s GB available, %s GB needed", root, available // GB, STORE_BYTES // GB)


def download_all(root: Path) -> None:
    """Download the metadata, the 100% list's cells and both starting weights; rerun to resume."""
    check_disk(root)
    # One core left free, at least 1 and at most 16 workers.
    workers = min(max(len(os.sched_getaffinity(0)) - 1, 1), 16)
    logger.info("download workers: %s", workers)
    garimpo_uncertainty.download.download_metadata(root)
    scenes = argparse.Namespace(
        command="scenes",
        split=Path("outputs/split.csv"),
        tiles=Path("outputs/tiles/tiles_100pct.json"),
        workers=workers,
        resampling_20m="bilinear",
        force=False,
    )
    download_scenes_stage(root, scenes)
    garimpo_uncertainty.download.download_weights(root, ["imagenet", "sentinel2_moco"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    # Subparsers do not inherit allow_abbrev from the main parser.
    commands.add_parser(
        "all",
        help="metadata, the 100%% list's cells and both weights, after a disk check",
        allow_abbrev=False,
    )
    commands.add_parser(
        "metadata", help="Jasansky et al. archive from Zenodo, md5 checked", allow_abbrev=False
    )
    weights = commands.add_parser(
        "weights", help="ImageNet and Sentinel-2 MoCo weight files", allow_abbrev=False
    )
    weights.add_argument(
        "--encoder-weights", nargs="+", choices=["imagenet", "sentinel2_moco"], required=True
    )
    lists = commands.add_parser(
        "lists", help="regenerate the versioned tile list of each fraction", allow_abbrev=False
    )
    lists.add_argument("--split", type=Path, default="outputs/split.csv")
    lists.add_argument("--subset-seed", type=int, required=True, help="picks the training tiles")
    lists.add_argument("--out", type=Path, default="outputs/tiles")
    scenes = commands.add_parser("scenes", help="cells from Planetary Computer", allow_abbrev=False)
    scenes.add_argument("--split", type=Path, default="outputs/split.csv")
    scenes.add_argument(
        "--tiles", type=Path, required=True, help="a list written by the lists command"
    )
    scenes.add_argument("--workers", type=int, default=4)
    scenes.add_argument(
        "--resampling-20m",
        choices=["nearest", "bilinear"],
        required=True,
        help="how B11 and B12 reach the 10 m grid",
    )
    scenes.add_argument("--force", action="store_true", help="redo every listed tile")
    args = parser.parse_args()

    if args.command == "lists":
        garimpo_uncertainty.stages.log_to_stdout()
        write_tile_lists(args.split, args.subset_seed, args.out)
        return
    root = garimpo_uncertainty.data.data_root()
    root.mkdir(parents=True, exist_ok=True)
    # Beside the data it describes, so the log travels with the store's archive.
    garimpo_uncertainty.stages.log_to_stdout(root / "logs" / "download.log")
    if args.command == "metadata":
        garimpo_uncertainty.download.download_metadata(root)
        return
    if args.command == "weights":
        garimpo_uncertainty.download.download_weights(root, args.encoder_weights)
        return
    if args.command == "all":
        download_all(root)
        return
    download_scenes_stage(root, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # A normal exit joins the worker threads, so after a second Ctrl+C it would wait for the
        # tiles in progress; the stage status is already written by now.
        sys.stdout.flush()
        os._exit(130)
