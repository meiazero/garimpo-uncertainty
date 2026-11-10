#!/bin/bash
# Pilot: the real pipeline at the 1% training fraction, to check the code on a GPU before the full
# runs. Validation, calibration and test stay whole: 186 tiles, 2,976 cells. Every value comes from
# the variant's JSON config; they differ from the full run's only where the pilot must stay short
# or reach the last stage (docs/pilot.md).
#
#   bash slurm/pilot.sbatch configs/pilot/unet-none.json            # on the GPU
#   bash slurm/pilot.sbatch configs/pilot/unet-none.json download   # once, resumable; no GPU needed
#
# Each variant is its own run directory. NUM_WORKERS sets the data loader processes.
# The pipeline check prints the seconds and peak memory per stage that size the full run.
set -euo pipefail
cd "$(dirname "$0")/.."

# An assignment, so a config that run_config.py refuses stops the job before any stage.
values=$(uv run --no-sync python scripts/run_config.py "${1:?usage: bash slurm/pilot.sbatch <config.json> [download]}")
eval "$values"
# The versioned list of TRAIN_FRACTION 0.01 and SUBSET_SEED 0; validate_data fails on a tile this did
# not download.
TILES=outputs/tiles/tiles_1pct.json
RESAMPLING_20M=bilinear

if [ "${2:-}" = download ]; then
    uv run --no-sync python scripts/download_data.py metadata
    uv run --no-sync python scripts/download_data.py scenes \
        --tiles "$TILES" --resampling-20m "$RESAMPLING_20M"
    # Last, so a Hugging Face outage never blocks the data, and only what this variant needs.
    if [ "$ENCODER_WEIGHTS" != none ]; then
        uv run --no-sync python scripts/download_data.py weights \
            --encoder-weights "$ENCODER_WEIGHTS"
    fi
    exit 0
fi

VALIDATION_REPORT=runs/pilot/data_validation.json
VALIDATE_ARGS=(--split-summary outputs/split_summary.json)
NUM_WORKERS=${NUM_WORKERS:-8}

# The check runs on any exit, so the failing stage and its error always reach the log. It gets the
# exit code because an error before run_stage leaves every status ok. A failed check fails the
# script, so a broken run never exits 0; a stage's own exit code wins when there is one.
trap 'job_exit=$?; check_exit=0; uv run --no-sync python scripts/pipeline_check.py --run-dir "$RUN_DIR" --train-fraction "$TRAIN_FRACTION" --subset-seed "$SUBSET_SEED" --job-exit-code "$job_exit" || check_exit=$?; exit "$((job_exit ? job_exit : check_exit))"' EXIT

source slurm/stages.sh
run_validate
run_diagnostics
run_calibrated_stages
uv run --no-sync python scripts/merge_results.py --runs "$RUN_DIR" --out "runs/pilot/tables-$ENCODER_WEIGHTS"
