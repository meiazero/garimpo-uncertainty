# Full run, from zero

Read when the pilot passes and the values in [decisions.md](decisions.md) are fixed.

| Step | Where | Command |
|---|---|---|
| 1 | login | `git clone --recurse-submodules <ROOT_REPO_URL> garimpo-uncertainty && cd garimpo-uncertainty && uv sync` |
| 2 | login | `uv run --no-sync python scripts/download_data.py metadata` |
| 3 | login | `uv run --no-sync python scripts/download_data.py scenes --tiles outputs/tiles/tiles_100pct.json --resampling-20m bilinear` (from the versioned list, so `lists` need not run first; resumable; contains every smaller fraction of that list's seed; sizes in [runs.md](runs.md#training-fractions); another resampling refuses; `--force` redoes every selected tile, then rerun with `FORCE_VALIDATE=1 FORCE_FROM=train`, since a re-cut that changes a tile's slots changes the hash in its sidecar, so later stages refuse until redone; a re-cut with identical content changes nothing) |
| 4 | login | `uv run --no-sync python scripts/download_data.py weights --encoder-weights imagenet sentinel2_moco` |
| 5 | login | `cp slurm/full.env.example slurm/unet_scratch-f1.0.env`, then fill in every value (see [decisions.md](decisions.md)) |
| 6 | `sbatch` | `slurm/submit.sh slurm/full.sbatch slurm/unet_scratch-f1.0.env` (the job chooses the layer from the validation role and continues; `chosen_layer.json` records why) |
| 7 | both | Repeat steps 5 and 6 for the other three runs in [runs.md](runs.md#model-variants) |
| 8 | login | `uv run --no-sync python scripts/merge_results.py --runs runs/unet_*-f* runs/convnet-f1.0 --out outputs` |

Compute nodes have no internet, so `uv sync` and steps 2 to 4 only run on the login node; every job
uses `uv run --no-sync`. The data root is `$GARIMPO_DATA_DIR`, default
`~/Experiments/datasets/garimpo`.

## Download a larger fraction

Downloads are incremental on the one store, so a larger list fetches only what is missing:

```bash
uv run --no-sync python scripts/download_data.py scenes --tiles outputs/tiles/tiles_1pct.json --resampling-20m bilinear
uv run --no-sync python scripts/download_data.py scenes --tiles outputs/tiles/tiles_5pct.json --resampling-20m bilinear
```

## Regenerate the tile lists

The versioned lists use subset seed 0, the decided `SUBSET_SEED`. `lists` regenerates all four, and
is needed only after `outputs/split.csv` changes or to use another seed:

```bash
uv run --no-sync python scripts/download_data.py lists --subset-seed <SUBSET_SEED>
```

The E2E fails if regenerating with the recorded seed does not reproduce the versioned files byte for
byte, so commit the regenerated files with whatever changed them.

## Run outside the cluster

On Colab
or RunPod, which have internet, run the same commands directly:

```bash
export GARIMPO_DATA_DIR=<DATA_DIR>
uv run python scripts/download_data.py weights --encoder-weights sentinel2_moco
# train.py alone does not check that the selected tiles were downloaded; this does.
uv run python scripts/validate_data.py --train-fraction 1.0 --subset-seed <SUBSET_SEED> --report <REPORT>
uv run python scripts/train.py --arch unet --encoder-weights sentinel2_moco --run-dir runs/unet_sentinel2-f1.0 --train-fraction 1.0 --subset-seed <SUBSET_SEED> ...
```

## Rerun one stage

Rerunning one stage needs no earlier stage: `slurm/submit.sh slurm/gpu.sbatch scripts/<stage>.py ...`
for stages that need a GPU, `slurm/cpu.sbatch` for the rest. A new metric or another boundary F1
tolerance, with the run's `SEED` because the seed is part of the fingerprint:

```bash
slurm/submit.sh slurm/cpu.sbatch scripts/compute_metrics.py --run-dir runs/unet_scratch-f1.0 --boundary-tolerance-px <PX> --seed <SEED> --force
```
