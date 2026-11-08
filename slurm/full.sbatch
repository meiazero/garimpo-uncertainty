#!/bin/bash
# Full run of one model, every value required from a file written by you:
#   cp slurm/full.env.example slurm/unet_scratch-f1.0.env    # fill in; see docs/decisions.md
#   CUDA_VISIBLE_DEVICES=0 bash slurm/full.sbatch slurm/unet_scratch-f1.0.env
#
# One run per GPU: on a pod with several, start one per CUDA_VISIBLE_DEVICES and give each
# NUM_WORKERS of about the pod's vCPUs divided by the runs.
#
# The COMPASS-J layer is chosen inside the run, from the validation role (scripts/choose_layer.py).
set -euo pipefail
cd "$(dirname "$0")/.."
source "${1:?usage: bash slurm/full.sbatch <values.env>}"

: "${ARCH:?}" "${RUN_DIR:?}" "${TRAIN_FRACTION:?}" "${SUBSET_SEED:?}"
: "${EPOCHS:?}" "${STEPS_PER_EPOCH:?}" "${BATCH_SIZE:?}" "${LR:?}" "${WEIGHT_DECAY:?}"
: "${EVAL_BATCH_SIZE:?}" "${LOSS:?}" "${SEED:?}" "${SVM_C:?}" "${SVM_CLASS_WEIGHT:?}" "${SVM_PIXELS_PER_CELL:?}"
: "${SWEEP_STEPS:?}" "${SWEEP_CELLS:?}" "${BETA_RANGE_LOGITS:?}" "${BETA_RANGE_FEATURES:?}"
: "${DIRECTION_TEMPERATURE:?}"
[ "$ARCH" != unet ] || : "${ENCODER_WEIGHTS:?}"
[ "$ARCH" != convnet ] || [ "${#ARCH_ARGS[@]}" -gt 0 ] || { echo "ARCH_ARGS is empty" >&2; exit 1; }
: "${MIN_FRACTION_ASCENDING:?}" "${SEARCH_THRESHOLD_LOGITS:?}" "${SEARCH_THRESHOLD_FEATURES:?}"
: "${CELLS:?}" "${SCORE_FLOOR_HA:?}" "${BOUNDARY_TOLERANCE_PX:?}"
[ "${#LAYERS[@]}" -gt 0 ] || { echo "LAYERS is empty" >&2; exit 1; }

VALIDATION_REPORT=outputs/data_validation-fraction-$TRAIN_FRACTION-seed-$SUBSET_SEED.json
VALIDATE_ARGS=(--split-summary outputs/split_summary.json)
NUM_WORKERS=${NUM_WORKERS:-16}

# The exit code goes to the check because an error before run_stage leaves every status ok. A failed
# check fails the script, so a broken run never exits 0; a stage's exit code wins.
trap 'job_exit=$?; check_exit=0; uv run --no-sync python scripts/pipeline_check.py --run-dir "$RUN_DIR" --train-fraction "$TRAIN_FRACTION" --subset-seed "$SUBSET_SEED" --job-exit-code "$job_exit" || check_exit=$?; exit "$((job_exit ? job_exit : check_exit))"' EXIT

source slurm/stages.sh
run_validate
run_diagnostics
run_calibrated_stages
