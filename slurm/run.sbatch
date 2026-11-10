#!/bin/bash
# One run of one model, every value from its JSON config (docs/decisions.md gives their sources).
# configs/full/ trains on 100% of the training tiles; configs/pilot/ holds the same models on 1%,
# with a short training, to check the code first:
#   CUDA_VISIBLE_DEVICES=0 bash slurm/run.sbatch configs/full/unet_scratch.json
#
# One run per GPU. NUM_WORKERS sets the data loader processes, such as $(nproc) on Colab.
#
# The COMPASS-J layer is chosen inside the run, from the validation role (scripts/choose_layer.py).
set -euo pipefail
cd "$(dirname "$0")/.."
# An assignment, so a config that run_config.py refuses stops the job before any stage.
values=$(uv run --no-sync python scripts/run_config.py "${1:?usage: bash slurm/run.sbatch <config.json>}")
eval "$values"

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
