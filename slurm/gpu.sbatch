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
# Compute nodes are offline: ImageNet weights come from the cache `download_data.py weights` fills.
export HF_HOME="${GARIMPO_DATA_DIR:-$HOME/Experiments/datasets/garimpo}/weights/huggingface"
export HF_HUB_OFFLINE=1
uv run --no-sync python "$@"
