#!/bin/bash
# Pilot: exercises the real pipeline on a small real subset, to adjust code on the cluster.
#
#   bash slurm/pilot.sbatch download        # on the login node, once, resumable
#   ENCODER_WEIGHTS=imagenet bash slurm/pilot.sbatch download   # adds that variant's weights
#   slurm/submit.sh slurm/pilot.sbatch      # everything else, through SLURM
#   ENCODER_WEIGHTS=imagenet slurm/submit.sh slurm/pilot.sbatch   # none, imagenet, sentinel2_moco
#
# Each variant is its own job in its own run directory, with its own time limit below.
#
# Memory and time are estimates, never measured: 8 workers holding 8 cells of 6.3 MB each are
# under 1 GB, and the pilot's SVM matrix is a few MB. The ResNet-50 encoder sets the forward cost
# on one A100 for ~300 calibration and test cells, at no measured rate, and the COMPASS searches
# rerun it. 48 GB and 4 h are guesses; adjust them from the peak memory and seconds per stage that
# the pipeline check prints.
#SBATCH -J garimpo-pilot
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH -o outputs/logs/%x-%j.out
#SBATCH -e outputs/logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

# ---- PILOT-ONLY VALUES: chosen to make the pipeline run, NOT scientific choices ---------------
GROUPS_PER_ROLE=3            # whole artisanal groups per split role
MAX_INDUSTRIAL_TILES=12
SUBSET_SEED=0
RESAMPLING_20M=bilinear

ARCH=unet
ARCH_ARGS=()
ENCODER_WEIGHTS="${ENCODER_WEIGHTS:-none}"
EPOCHS=2
BATCH_SIZE=8
EVAL_BATCH_SIZE=8
LR=1e-4                      # the paper's value, Appendix E
WEIGHT_DECAY=0.01            # torch's AdamW default
LOSS=dice_bce
SEED=0

LAYER=decoder.blocks.4
LAYERS=(decoder.blocks.3 decoder.blocks.4)
SVM_C=1.0
SVM_CLASS_WEIGHT=balanced
SVM_PIXELS_PER_CELL=200
SWEEP_STEPS=21
SWEEP_CELLS=20
BETA_RANGE_LOGITS=100        # upstream's value; also the sweep's beta_max
BETA_RANGE_FEATURES=1000     # upstream's value; also the sweep's beta_max
SEARCH_THRESHOLD_LOGITS=1e-2 # upstream's value
SEARCH_THRESHOLD_FEATURES=1e-3
CELLS=all
SCORE_FLOOR_HA=0.01          # one pixel
BOUNDARY_TOLERANCE_PX=2
# -----------------------------------------------------------------------------------------------

if [ "${1:-}" = download ]; then
    uv run --no-sync python scripts/download_data.py metadata
    uv run --no-sync python scripts/download_data.py scenes \
        --groups-per-role "$GROUPS_PER_ROLE" --max-industrial-tiles "$MAX_INDUSTRIAL_TILES" \
        --seed "$SUBSET_SEED" --resampling-20m "$RESAMPLING_20M"
    # Last, so a hub outage never blocks the data, and only what this variant needs.
    if [ "$ENCODER_WEIGHTS" != none ]; then
        uv run --no-sync python scripts/download_data.py weights \
            --encoder-weights "$ENCODER_WEIGHTS"
    fi
    exit 0
fi

RUN_DIR=runs/pilot/unet-$ENCODER_WEIGHTS
VALIDATION_REPORT=runs/pilot/data_validation.json
VALIDATE_ARGS=()             # no --split-summary: the counts of the full split do not apply
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

# The check runs on any exit, so the failing stage and its error always reach the log. It gets the
# exit code because an error before run_stage leaves every status ok.
trap 'job_exit=$?; uv run --no-sync python scripts/pipeline_check.py --run-dir "$RUN_DIR" --job-exit-code "$job_exit" || true; exit "$job_exit"' EXIT

source slurm/stages.sh
run_validate
run_diagnostics
run_calibrated_stages
uv run --no-sync python scripts/merge_results.py --runs "$RUN_DIR" --out "runs/pilot/tables-$ENCODER_WEIGHTS"
