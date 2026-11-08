#!/bin/bash
# Pilot: the real pipeline at the 1% training fraction, to adjust code on the cluster. Validation,
# calibration and test stay whole: 186 tiles, 2,976 cells, 10.9 GB written where the filesystem
# keeps holes, 69.2 GB allocated at the first download where it does not (docs/data-layout.md).
#
#   bash slurm/pilot.sbatch download        # on the login node, once, resumable
#   ENCODER_WEIGHTS=imagenet bash slurm/pilot.sbatch download   # adds that variant's weights
#   slurm/submit.sh slurm/pilot.sbatch      # everything else, through SLURM
#   ENCODER_WEIGHTS=imagenet slurm/submit.sh slurm/pilot.sbatch   # none, imagenet, sentinel2_moco
#
# Each variant is its own job in its own run directory, with its own time limit below.
#
# Memory and time are estimates, never measured, and rest on the current code:
# - Memory: 8 workers, 2 prefetched batches each, 8 cells of 6.3 MB stay under 1 GB; the SVM matrix
#   is 160 training cells x 200 pixels, under 10 MB at 32 channels.
# - Downstream cells are 2,528 calibration and test (158 tiles), about 8x the ~300 of the earlier
#   pilot subset, so 4 h is less safe than it was.
# - Cells read and passed through the encoder: train 896, infer 2,976, fit_directions 320,
#   monotonicity 60, compass_scores 2,528, generate_intervals 1,424, 8,204 in total. So 4 h needs
#   about 0.57 cells/s, reading from NFS included. Unmeasured. validate_data and the pipeline check
#   each also hash every slot of the run, 10.9 GB read per call, which this count leaves out.
# - compass_scores adds, per cell, 80 g(z) evaluations (20 steps x 2 endpoints x 2 directions) and
#   one Jacobian, on the layers after LAYER; at decoder.blocks.4 that is the head alone.
#   generate_intervals adds 48 g(z) per test cell (8 COMPASS-J rows x 2 endpoints x 3 alphas);
#   compass_l needs no decoder.
# - EPOCHS=2 is 40 updates on 160 cells. A model still predicting nothing has all-zero Jacobians at
#   temperature 10,000, and fit_directions stops; whether 40 updates avoid that is unknown.
# 48 GB and 4 h are guesses; adjust them from the peak memory and seconds per stage that the
# pipeline check prints. A job killed at the limit keeps its finished stages, which then skip.
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
TRAIN_FRACTION=0.01
SUBSET_SEED=0
# The versioned list of the two values above; validate_data fails on a tile this did not download.
TILES=outputs/tiles/tiles_1pct.json
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
        --tiles "$TILES" --resampling-20m "$RESAMPLING_20M"
    # Last, so a Hugging Face outage never blocks the data, and only what this variant needs.
    if [ "$ENCODER_WEIGHTS" != none ]; then
        uv run --no-sync python scripts/download_data.py weights \
            --encoder-weights "$ENCODER_WEIGHTS"
    fi
    exit 0
fi

RUN_DIR=runs/pilot/unet-$ENCODER_WEIGHTS
VALIDATION_REPORT=runs/pilot/data_validation.json
VALIDATE_ARGS=(--split-summary outputs/split_summary.json)
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

# The check runs on any exit, so the failing stage and its error always reach the log. It gets the
# exit code because an error before run_stage leaves every status ok. A failed check fails the job,
# so SLURM cannot show a broken run as completed; a stage's own exit code wins when there is one.
trap 'job_exit=$?; check_exit=0; uv run --no-sync python scripts/pipeline_check.py --run-dir "$RUN_DIR" --train-fraction "$TRAIN_FRACTION" --subset-seed "$SUBSET_SEED" --job-exit-code "$job_exit" || check_exit=$?; exit "$((job_exit ? job_exit : check_exit))"' EXIT

source slurm/stages.sh
run_validate
run_diagnostics
run_calibrated_stages
uv run --no-sync python scripts/merge_results.py --runs "$RUN_DIR" --out "runs/pilot/tables-$ENCODER_WEIGHTS"
