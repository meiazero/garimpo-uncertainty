#!/bin/bash
# One stage on the GPU node, e.g.:
#   slurm/submit.sh --time=1-00:00:00 slurm/gpu.sbatch scripts/compass_scores.py --run-dir ... --force
#SBATCH -J garimpo-gpu
#SBATCH -p gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH -o outputs/logs/%x-%j.out
#SBATCH -e outputs/logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
uv run --no-sync python "$@"
