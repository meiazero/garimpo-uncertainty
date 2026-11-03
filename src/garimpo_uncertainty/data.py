"""Cell format, data root, split selection and manifest access; no torch, for CPU-only stages."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
MASK_BAND = 7  # 1-based rasterio band of the reference mask
SCL_BAND = 8
N_CELL_BANDS = 8
RED_INDEX, NIR_INDEX = 2, 3
TILE_PX = 2048
CELL_PX = 512
PIXEL_M = 10.0
PIXEL_HA = 0.01  # 10 m x 10 m
REFLECTANCE_SCALE = 10_000
CLOUD_SCL_CLASSES = (3, 8, 9, 10)  # cloud shadow, medium and high cloud, thin cirrus
HELD_OUT_REGIONS = {"Amazonia BR", "Americas, other"}  # same set as scripts/proposal/make_split.py
ROLES = ("train", "validation", "calibration", "test")


def data_root() -> Path:
    default = "~/Experiments/datasets/garimpo"
    return Path(os.environ.get("GARIMPO_DATA_DIR", default)).expanduser()


def sentinel2_moco_path(root: Path) -> Path:
    return root / "weights" / "resnet50_sentinel2_all_moco-df8b932e.pth"


def huggingface_home(root: Path) -> Path:
    """Return the ImageNet weights cache; slurm/stages.sh and gpu.sbatch export this path too."""
    return root / "weights" / "huggingface"


def select_tiles(
    split: pd.DataFrame,
    scales: list[str],
    groups_per_role: int | None,
    max_industrial_tiles: int | None,
    seed: int,
) -> pd.DataFrame:
    """Return whole artisanal groups per role and at most N industrial tiles, deterministically."""
    selected = split[(split.role != "dropped") & split.scale.isin(scales)]
    rng = np.random.default_rng(seed)
    if groups_per_role is not None:
        kept: list[float] = []
        artisanal = selected[selected.scale == "Artisanal"]
        for role in ROLES:
            groups = np.sort(artisanal[artisanal.role == role].cluster_50km.unique())
            size = min(groups_per_role, len(groups))
            kept.extend(rng.choice(groups, size=size, replace=False))
        selected = selected[(selected.scale != "Artisanal") | selected.cluster_50km.isin(kept)]
    if max_industrial_tiles is not None:
        industrial = np.sort(selected[selected.scale == "Industrial"].tile_id.unique())
        rng.shuffle(industrial)
        selected = selected[
            (selected.scale != "Industrial")
            | selected.tile_id.isin(industrial[:max_industrial_tiles])
        ]
    return selected


def read_manifest(root: Path) -> pd.DataFrame:
    path = root / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run scripts/download_data.py scenes first")
    return pd.read_csv(path)


def split_hash(manifest: pd.DataFrame) -> str:
    """Return the identity of a checkpoint's tiles and roles; detects pilot versus full data."""
    table = manifest[["tile_id", "role", "group"]].drop_duplicates().sort_values("tile_id")
    return hashlib.sha256(table.to_csv(index=False).encode()).hexdigest()


def read_run_metadata(run_dir: Path) -> dict:
    """JSON twin of the checkpoint metadata, so stages with no torch can check the split."""
    return json.loads((run_dir / "run_meta.json").read_text())


def check_split(metadata: dict, manifest: pd.DataFrame) -> None:
    current = split_hash(manifest)
    if metadata["split_hash"] != current:
        raise ValueError(
            f"manifest split {current[:12]} differs from training split "
            f"{metadata['split_hash'][:12]}: wrong GARIMPO_DATA_DIR or pilot versus full data"
        )


def read_reference_mask(root: Path, relative_path: str) -> np.ndarray:
    with rasterio.open(root / relative_path) as source:
        return source.read(MASK_BAND).astype(np.uint8) > 0
