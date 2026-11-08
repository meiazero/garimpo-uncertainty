# Full run, from zero

Read when the pilot passes and the values in [decisions.md](decisions.md) are fixed.

| Step | Where | Command |
|---|---|---|
| 1 | every session | The session cell of the [install](../README.md#install) |
| 2 | CPU runtime, once | The data cell of the [quick start](../README.md#quick-start): metadata, the 100% list, both weights, then `rsync` to Drive (from the versioned list, so `lists` need not run first; resumable within a session; contains every smaller fraction of that list's seed; sizes in [runs.md](runs.md#training-fractions); another resampling refuses; `--force` redoes every selected tile, then rerun with `FORCE_VALIDATE=1 FORCE_FROM=train`, since a re-cut that changes a tile's slots changes the hash in its sidecar) |
| 3 | any | `cp slurm/full.env.example slurm/unet_scratch-f1.0.env`, fill in every value (see [decisions.md](decisions.md)), and commit it, since each session starts from a fresh clone |
| 4 | GPU runtime | The run cell below, one run per session (the run chooses the layer from the validation role and continues; `chosen_layer.json` records why) |
| 5 | GPU runtime | Repeat steps 3 and 4 for each other run in [runs.md](runs.md#model-variants) |
| 6 | any | `uv run --no-sync python scripts/merge_results.py --runs runs/unet_*-f* runs/convnet-f1.0 --out outputs` |

The run cell, after the session cell. A background `rsync` copies `runs/` to Drive every 30 minutes,
so a session that ends mid-run keeps the finished stages and the last checkpoint. Resuming training
from that checkpoint is not covered by the tests. `NUM_WORKERS` follows the runtime's CPUs:

```python
ENV = "unet_scratch-f1.0"
!(while sleep 1800; do rsync -a --exclude '*.tmp' runs/ {DRIVE}/runs/; done) > /dev/null 2>&1 &
!NUM_WORKERS={os.cpu_count()} bash slurm/full.sbatch slurm/{ENV}.env 2>&1 | tee outputs/logs/{ENV}.log
!rsync -a runs/ {DRIVE}/runs/ && rsync -a outputs/logs/ {DRIVE}/logs/
from google.colab import runtime; runtime.unassign()
```

Every script uses `uv run --no-sync`, so a run never changes the environment under itself.

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

## Rerun one stage

Rerunning one stage needs no earlier stage: `uv run --no-sync python scripts/<stage>.py ...`, on a GPU
runtime for the stages [stages.md](stages.md) marks GPU. A new metric or another boundary F1
tolerance, with the run's `SEED` because the seed is part of the fingerprint:

```bash
uv run --no-sync python scripts/compute_metrics.py --run-dir runs/unet_scratch-f1.0 --boundary-tolerance-px <PX> --seed <SEED> --force
```
