# Run the full experiment

Read when the pilot passes and the values in [decisions.md](decisions.md) are fixed.

## Steps

1. Start every Colab session with the session setup of `colab.ipynb`.
2. Download the store once, either on your own machine or on a Colab CPU runtime
   ([below](#download-the-store)).
3. Copy the values file with `cp slurm/full.env.example slurm/unet_scratch-f1.0.env` and fill in
   every value, using [decisions.md](decisions.md), with `RUN_DIR=runs/unet_scratch-f1.0`: the
   merge reads the runs from those paths. Commit and push it from your machine: each session
   starts from a fresh clone, which cannot push.
4. On a GPU runtime, run the session setup and then the notebook's Full run step
   ([below](#the-run-cells)), one run per session. The run chooses the layer from the validation
   role and continues; `chosen_layer.json` records why.
5. Repeat the two items above for the other three runs in [runs.md](runs.md#model-variants).
6. Merge the four runs' tables into `outputs/` with the notebook's Merge step, on a CPU runtime.
   It runs the command below; on your own machine, copy `runs/` from Drive first.

   ```bash
   uv run --no-sync python scripts/merge_results.py --out outputs --runs runs/unet_scratch-f1.0 runs/unet_imagenet-f1.0 runs/unet_sentinel2-f1.0 runs/convnet-f1.0
   ```

7. On Colab, the Merge step copies `outputs/` to `MyDrive/garimpo/outputs/`. Download it and commit
   it from your own machine: these are the tables the dissertation reports.

## Download the store

On your own machine:

```bash
GARIMPO_DATA_DIR=<DIR> bash slurm/download.sh
tar --format=posix -I 'zstd -19 -T0' -cf <DIR>.tar.zst -C <DIR> .
```

Then upload the archive to Drive as `garimpo/data.tar.zst`. `slurm/download.sh` downloads the
metadata, the 100% list and both starting weights. It stops before downloading when the disk cannot
hold the store, counting what is already downloaded as available, and resumes when rerun. It uses
one worker per CPU minus one, between 1 and 16. Each tile is printed as it ends, with the rate
and the time left, and the download's own lines are also appended, timestamped, to
`<DIR>/logs/download.log`; a failed stage's traceback goes to the terminal and to
`<DIR>/status/download_scenes.json` instead. Without `GARIMPO_DATA_DIR` it writes to
`~/Experiments/datasets/garimpo`. The `tar` line needs free disk for the archive as well.

On Colab, run step 2 of the notebook on a CPU runtime: metadata, the 1% and 100% lists, both
weights, then the archive to Drive.

Either way:

- The download reads the versioned list, so `lists` does not need to run first.
- It resumes within a session.
- The 100% list contains every smaller fraction of the same seed. Sizes are in
  [runs.md](runs.md#training-fractions).
- Downloading with another `--resampling-20m` than the tiles already cut refuses.
- `--force` redoes every listed tile. Rerun the stages afterwards with
  `FORCE_VALIDATE=1 FORCE_FROM=train`, since a re-cut that changes a tile's slots changes the hash
  in its sidecar.

## The run cells

The notebook's Full run step runs one model, in this order:

1. Start the copy of `runs/` and the logs to Drive every 30 minutes.
2. Run the model: choose `ENV` in the cell's option, and leave `FORCE_FROM` and `FORCE_VALIDATE`
   empty on a first run. `NUM_WORKERS` follows the runtime's CPUs. An option stays set in the open
   tab, so empty `FORCE_FROM` once a forced rerun passes. Reopening the notebook from its link
   restores the values on `main`.
3. Stop the copy, then save the runs and the logs in Drive.
4. Print the verdict, then release the GPU.

With the copy every 30 minutes, a session that ends mid-run keeps the finished stages and the
last checkpoint in Drive. `tests/e2e/test_resume.py` kills training after
its first epoch and requires the rerun to end where an uninterrupted run ends, on CPU; on a GPU the
kernels are not deterministic, so only the resume itself is the same. Rerun without `FORCE_FROM=train`
to resume: with it, `train.py` refuses to discard an unfinished training and says so. Every script
runs with `uv run --no-sync`, so a run never changes the environment under itself.

Outside Colab, one run is one job:

```bash
mkdir -p outputs/logs
NUM_WORKERS=$(nproc) bash slurm/full.sbatch slurm/unet_scratch-f1.0.env 2>&1 | tee -a outputs/logs/unet_scratch-f1.0.log
```

## Regenerate the tile lists

The versioned lists use subset seed 0, the decided `SUBSET_SEED`. `lists` regenerates all four. Run
it only after `outputs/split.csv` changes, or to use another seed:

```bash
uv run --no-sync python scripts/download_data.py lists --subset-seed <SUBSET_SEED>
```

The E2E suite fails if regenerating with the recorded seed does not reproduce the versioned files
byte for byte, so commit the regenerated files together with whatever changed them.

## Rerun one stage

A stage reruns on its own, without the earlier stages:
`uv run --no-sync python scripts/<stage>.py ...`. Use a GPU runtime for the stages that
[stages.md](stages.md#stage-table) marks GPU. For example, a new metric or another boundary F1
tolerance, passing the run's `SEED` because the seed is part of the fingerprint:

```bash
uv run --no-sync python scripts/compute_metrics.py --run-dir runs/unet_scratch-f1.0 --boundary-tolerance-px <PX> --seed <SEED> --force
```
