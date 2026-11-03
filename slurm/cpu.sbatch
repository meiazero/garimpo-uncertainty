#!/bin/bash
# One stage with no GPU (validate, calibrate, metrics), e.g.:
#   slurm/submit.sh slurm/cpu.sbatch scripts/compute_metrics.py --run-dir ... --boundary-tolerance-px 2 --force
#SBATCH -J garimpo-cpu
#SBATCH -p cpuq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -o outputs/logs/%x-%j.out
#SBATCH -e outputs/logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
uv run --no-sync python "$@"
