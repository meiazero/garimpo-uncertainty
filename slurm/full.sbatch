#!/bin/bash
# Full run of one model, every value required from a file written by you:
#   cp slurm/full.env.example slurm/unet_scratch-f1.0.env    # fill in; see docs/decisions.md
#   slurm/submit.sh slurm/full.sbatch slurm/unet_scratch-f1.0.env
#
# Choosing the layer: set STOP_AFTER_DIAGNOSTICS=1 and leave LAYER empty, read
# directions/diagnostics.csv and monotonicity_layer.csv, then set LAYER, clear the flag and submit
# again. Finished stages skip themselves, so this does not retrain.
#
# ponytail: memory and time follow the repo's other gpuq jobs, not a measurement; set them from
# the peak and the seconds per stage that the pilot's pipeline check prints.
#SBATCH -J garimpo-full
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH -o outputs/logs/%x-%j.out
#SBATCH -e outputs/logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source "${1:?usage: slurm/submit.sh slurm/full.sbatch <values.env>}"

: "${ARCH:?}" "${RUN_DIR:?}" "${TRAIN_FRACTION:?}" "${SUBSET_SEED:?}"
: "${EPOCHS:?}" "${STEPS_PER_EPOCH:?}" "${BATCH_SIZE:?}" "${LR:?}" "${WEIGHT_DECAY:?}"
: "${EVAL_BATCH_SIZE:?}" "${LOSS:?}" "${SEED:?}" "${SVM_C:?}" "${SVM_CLASS_WEIGHT:?}" "${SVM_PIXELS_PER_CELL:?}"
: "${SWEEP_STEPS:?}" "${SWEEP_CELLS:?}" "${BETA_RANGE_LOGITS:?}" "${BETA_RANGE_FEATURES:?}"
[ "$ARCH" != unet ] || : "${ENCODER_WEIGHTS:?}"
[ "$ARCH" != convnet ] || [ "${#ARCH_ARGS[@]}" -gt 0 ] || { echo "ARCH_ARGS is empty" >&2; exit 1; }
[ "${#LAYERS[@]}" -gt 0 ] || { echo "LAYERS is empty" >&2; exit 1; }

if [ -z "${STOP_AFTER_DIAGNOSTICS:-}" ]; then
    : "${LAYER:?}" "${SEARCH_THRESHOLD_LOGITS:?}" "${SEARCH_THRESHOLD_FEATURES:?}" "${CELLS:?}"
    : "${SCORE_FLOOR_HA:?}" "${BOUNDARY_TOLERANCE_PX:?}"
    if [[ " ${LAYERS[*]} " != *" $LAYER "* ]]; then
        echo "LAYER=$LAYER is not in LAYERS=(${LAYERS[*]}): no direction was fitted for it" >&2
        exit 1
    fi
fi

VALIDATION_REPORT=outputs/data_validation-fraction-$TRAIN_FRACTION-seed-$SUBSET_SEED.json
VALIDATE_ARGS=(--split-summary outputs/split_summary.json)
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-16}
THROUGH=compute_metrics
[ -z "${STOP_AFTER_DIAGNOSTICS:-}" ] || THROUGH=layer_monotonicity

# The exit code goes to the check because an error before run_stage leaves every status ok. A failed
# check fails the job, so SLURM cannot show a broken run as completed; a stage's exit code wins.
trap 'job_exit=$?; check_exit=0; uv run --no-sync python scripts/pipeline_check.py --run-dir "$RUN_DIR" --train-fraction "$TRAIN_FRACTION" --subset-seed "$SUBSET_SEED" --through "$THROUGH" --job-exit-code "$job_exit" || check_exit=$?; exit "$((job_exit ? job_exit : check_exit))"' EXIT

source slurm/stages.sh
run_validate
run_diagnostics
if [ -z "${STOP_AFTER_DIAGNOSTICS:-}" ]; then
    run_calibrated_stages
fi
