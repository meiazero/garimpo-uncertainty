#!/bin/bash
# Pilot: the real pipeline at the 1% training fraction, to adjust the code on a GPU. Validation,
# calibration and test stay whole: 186 tiles, 2,976 cells, 10.9 GB written where the filesystem
# keeps holes, 69.2 GB allocated at the first download where it does not (docs/data-layout.md).
#
#   bash slurm/pilot.sbatch download        # once, resumable; no GPU needed
#   ENCODER_WEIGHTS=imagenet bash slurm/pilot.sbatch download   # adds that variant's weights
#   bash slurm/pilot.sbatch                 # everything else, on the GPU
#   ENCODER_WEIGHTS=imagenet bash slurm/pilot.sbatch   # none, imagenet, sentinel2_moco
#
# Each variant is its own run directory. NUM_WORKERS sets the data loader processes.
#
# What the pilot reads, from the current code; nothing here was measured:
# - Memory: 8 workers, 2 prefetched batches each, 16 cells of 6.3 MB stay under 2 GB; the SVM matrix
#   is 160 training cells x 200 pixels, under 10 MB at 32 channels.
# - Cells read and passed through the encoder: train 1,216 (2 x 20 x 16 plus 2 x 288
#   validation), infer 2,976, fit_directions 320, monotonicity 60, compass_scores 2,528,
#   generate_intervals 1,424, 8,524 in total. validate_data and the pipeline check each also hash
#   every slot of the run, 10.9 GB read per call, which this count leaves out.
# - compass_scores adds, per cell, 160 g(z) evaluations (20 steps x 2 directions x 4: the two
#   symmetric endpoints plus Algorithm 6's lower and upper sides) and
#   one Jacobian, on the layers after the chosen one: the head alone at decoder.blocks.4, one
#   more block at decoder.blocks.3.
#   generate_intervals adds 48 g(z) per test cell (8 COMPASS-J rows x 2 endpoints x 3 alphas);
#   compass_l needs no decoder.
# - EPOCHS=2 x STEPS_PER_EPOCH=20 is 40 updates on 160 cells. A model still predicting nothing has
#   all-zero Jacobians at temperature 10,000, and fit_directions stops; whether 40 updates avoid
#   that is unknown.
# The pipeline check prints the seconds and peak memory per stage that size the full run.
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- PILOT VALUES: the decided ones of docs/decisions.md where it has one, so the pilot measures
# them; the rest (fraction, EPOCHS, STEPS_PER_EPOCH, EVAL_BATCH_SIZE, layers, SVM, sweep,
# SEARCH_THRESHOLD_FEATURES, SCORE_FLOOR_HA) only make the pipeline run, NOT scientific choices --
TRAIN_FRACTION=0.01
SUBSET_SEED=0
# The versioned list of the two values above; validate_data fails on a tile this did not download.
TILES=outputs/tiles/tiles_1pct.json
RESAMPLING_20M=bilinear

ARCH=unet
ENCODER_WEIGHTS="${ENCODER_WEIGHTS:-none}"
EPOCHS=2
STEPS_PER_EPOCH=20
BATCH_SIZE=16
EVAL_BATCH_SIZE=8
LR=1e-4                      # the paper's value, Appendix E
WEIGHT_DECAY=0.01            # torch's AdamW default
LOSS=dice_bce
SEED=0

LAYERS=(decoder.blocks.3 decoder.blocks.4)
# Every layer qualifies, so the pilot reaches the last stage; the check prints each fraction.
MIN_FRACTION_ASCENDING=0
SVM_C=1.0
SVM_CLASS_WEIGHT=balanced
SVM_PIXELS_PER_CELL=200
SWEEP_STEPS=21
SWEEP_CELLS=20
BETA_RANGE_LOGITS=100        # upstream's value; also the sweep's beta_max
BETA_RANGE_FEATURES=1000     # upstream's value; also the sweep's beta_max
DIRECTION_TEMPERATURE=10000  # upstream's temperature, the metric's; see docs/decisions.md
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
