#!/usr/bin/env bash
# SLURM does not create the directory of -o; every sbatch here logs to outputs/logs.
set -euo pipefail
mkdir -p outputs/logs
sbatch "$@"
