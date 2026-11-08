# Full run, from zero

Read when the pilot passes and the values in [decisions.md](decisions.md) are fixed.

| Step | Where | Command |
|---|---|---|
| 1 | any pod | The [install](../README.md#install), with `GARIMPO_DATA_DIR` on the network volume |
| 2 | any pod | `uv run --no-sync python scripts/download_data.py metadata` |
| 3 | any pod | `uv run --no-sync python scripts/download_data.py scenes --tiles outputs/tiles/tiles_100pct.json --resampling-20m bilinear` (from the versioned list, so `lists` need not run first; resumable; contains every smaller fraction of that list's seed; sizes in [runs.md](runs.md#training-fractions); another resampling refuses; `--force` redoes every selected tile, then rerun with `FORCE_VALIDATE=1 FORCE_FROM=train`, since a re-cut that changes a tile's slots changes the hash in its sidecar, so later stages refuse until redone; a re-cut with identical content changes nothing) |
| 4 | any pod | `uv run --no-sync python scripts/download_data.py weights --encoder-weights imagenet sentinel2_moco` |
| 5 | any pod | `cp slurm/full.env.example slurm/unet_scratch-f1.0.env`, then fill in every value (see [decisions.md](decisions.md)) |
| 6 | GPU pod | `CUDA_VISIBLE_DEVICES=0 nohup bash slurm/full.sbatch slurm/unet_scratch-f1.0.env > outputs/logs/unet_scratch-f1.0.log 2>&1 &` (the run chooses the layer from the validation role and continues; `chosen_layer.json` records why) |
| 7 | GPU pod | Repeat steps 5 and 6 for each other run of the learning curve in [runs.md](runs.md#model-variants), one per GPU, as [below](#several-runs-on-one-pod) |
| 8 | any pod | `uv run --no-sync python scripts/merge_results.py --runs runs/unet_*-f* runs/convnet-f1.0 --out outputs` |

Steps 2 to 4 need no GPU, so a CPU pod on the same network volume can do them while no GPU is
billed. Every script uses `uv run --no-sync`, so a run never changes the environment under the
others. Create `outputs/logs` once before step 6.

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

## Several runs on one pod

Each run uses one GPU, chosen with `CUDA_VISIBLE_DEVICES`. On a pod with four, start four runs, and
give each `NUM_WORKERS` of about the pod's vCPUs divided by four (default 16):

```bash
for gpu_env in 0:unet_scratch-f1.0 1:unet_imagenet-f1.0 2:unet_sentinel2-f1.0 3:convnet-f1.0; do
    gpu=${gpu_env%%:*}; env=${gpu_env#*:}
    CUDA_VISIBLE_DEVICES=$gpu NUM_WORKERS=<N> nohup bash slurm/full.sbatch slurm/$env.env \
        > outputs/logs/$env.log 2>&1 &
done
```

Training and the later stages write only to their own `RUN_DIR`. Runs of one fraction share the
validation report and its status, written atomically. Each run hashes every slot of its selection
when it starts, so four 100% runs started together read 4 x 69.2 GB at once from the volume. No test
covers runs started side by side.

## Rerun one stage

Rerunning one stage needs no earlier stage: `uv run --no-sync python scripts/<stage>.py ...`, on the
GPU pod for the stages [stages.md](stages.md) marks GPU. A new metric or another boundary F1
tolerance, with the run's `SEED` because the seed is part of the fingerprint:

```bash
uv run --no-sync python scripts/compute_metrics.py --run-dir runs/unet_scratch-f1.0 --boundary-tolerance-px <PX> --seed <SEED> --force
```
