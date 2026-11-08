#!/usr/bin/env bash
# Download the full store, the metadata and both starting weights to GARIMPO_DATA_DIR, on a machine
# outside Colab. Stops before downloading when the disk cannot hold the store. Rerun to resume.
# Log with `2>&1 | tee -i <log>`: a plain tee dies on Ctrl+C and takes the stopping messages along.
set -euo pipefail
cd "$(dirname "$0")/.."

export GARIMPO_DATA_DIR=${GARIMPO_DATA_DIR:-$HOME/Experiments/datasets/garimpo}
# 18,864 slots of 3,670,016 bytes (docs/data-layout.md), plus 2 GB for Zenodo, weights and sidecars.
NEEDED=$((18864 * 3670016 + 2 * 1000 ** 3))
mkdir -p "$GARIMPO_DATA_DIR"
HELD=$(du -s --block-size=1 "$GARIMPO_DATA_DIR" | cut -f1)
FREE=$(df --output=avail --block-size=1 "$GARIMPO_DATA_DIR" | tail -1)
if ((FREE + HELD < NEEDED)); then
    echo "$GARIMPO_DATA_DIR: needs $((NEEDED / 1000 ** 3)) GB, has $(((FREE + HELD) / 1000 ** 3)) GB" \
        "(free plus already downloaded)" >&2
    exit 1
fi
echo "$GARIMPO_DATA_DIR: $(((FREE + HELD) / 1000 ** 3)) GB available, $((NEEDED / 1000 ** 3)) GB needed"
# One core left free, at least 1 and at most 16 workers.
WORKERS=$(($(nproc) - 1))
WORKERS=$((WORKERS < 1 ? 1 : WORKERS > 16 ? 16 : WORKERS))
echo "download workers: $WORKERS"

uv run --no-sync python scripts/download_data.py metadata
uv run --no-sync python scripts/download_data.py scenes --tiles outputs/tiles/tiles_100pct.json \
    --resampling-20m bilinear --workers "$WORKERS"
uv run --no-sync python scripts/download_data.py weights --encoder-weights imagenet sentinel2_moco
