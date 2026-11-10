# Run the full experiment

Read when the pilot passes and the values in [decisions.md](decisions.md) are fixed.

## Steps

1. Download the store once on your own machine and upload it to Drive
   ([below](#download-the-store)). The notebook downloads nothing.
2. Each run reads every value from its config in `configs/full/` ([below](#the-run-configs)). To
   change one, edit the file, then commit and push it from your machine: each session starts from a
   fresh clone, which cannot push.
3. On a GPU runtime, run Runtime > Run all in `colab.ipynb`. Section 6 trains the four models one
   after another ([below](#the-run-cells)). Each run chooses the layer from the validation role
   and continues; `chosen_layer.json` records why.
4. If the session ends first, run Runtime > Run all again in a new session: finished stages skip
   and training resumes from its last checkpoint.
5. Section 7 merges the four runs' tables into `outputs/` with the command below; on your own
   machine, copy `runs/` from Drive first.

   ```bash
   uv run --no-sync python scripts/merge_results.py --out outputs --runs runs/unet_scratch-f1.0 runs/unet_imagenet-f1.0 runs/unet_sentinel2-f1.0 runs/convnet-f1.0
   ```

6. On Colab, section 7 copies `outputs/` to `MyDrive/garimpo/outputs/`. Download it and commit it
   from your own machine: these are the final result tables.

## The run configs

| Config | Model |
|---|---|
| `configs/full/unet_scratch-f1.0.json` | U-Net from scratch |
| `configs/full/unet_imagenet-f1.0.json` | U-Net from ImageNet weights |
| `configs/full/unet_sentinel2-f1.0.json` | U-Net from Sentinel-2 MoCo weights |
| `configs/full/convnet-f1.0.json` | DeepLabV3 from segmentation-models-pytorch |

Each value has its source in [decisions.md](decisions.md). `scripts/run_config.py` reads a config
for the job and stops it before any stage when a key is missing, null, empty or unknown, or when
`SCORE_FLOOR_HA` is not a positive finite number. `RUN_DIR` is `runs/<config name>`: the merge
reads the runs from those paths.

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

About the download:

- The download reads the versioned list, so `lists` does not need to run first.
- It resumes when rerun.
- The 100% list contains every smaller fraction of the same seed. Sizes are in
  [runs.md](runs.md#training-fractions).
- Downloading with another `--resampling-20m` than the tiles already cut refuses.
- `--force` redoes every listed tile. Rerun the stages afterwards with
  `FORCE_VALIDATE=1 FORCE_FROM=train`, since a re-cut that changes a tile's slots changes the hash
  in its sidecar.

## The run cells

Section 6 of the notebook has one cell per model, run in order:

1. "Train <model>" runs `bash slurm/full.sbatch configs/full/<model>.json`. `NUM_WORKERS` follows
   the runtime's CPUs.
2. "Rerun <model> - (in a reexecution)" does nothing unless its `RUN` box is checked; then it runs
   the same job with its `FORCE_FROM` and `FORCE_VALIDATE` options. An option stays set in the
   open tab, so clear `RUN` once the rerun passes. Reopening the notebook from its link restores
   the values on `main`.
3. "Print the verdicts of the training" prints one line per model.

The copy of `runs/` and the logs to Drive every 30 minutes starts before the pilot and stops in
section 8, which saves the runs and logs once more and releases the GPU.

With the copy every 30 minutes, a session that ends mid-run keeps the finished stages and the
last checkpoint in Drive. `tests/e2e/test_resume.py` kills training after
its first epoch and requires the rerun to end where an uninterrupted run ends, on CPU; on a GPU the
kernels are not deterministic, so only the resume itself is the same. Rerun without `FORCE_FROM=train`
to resume: with it, `train.py` refuses to discard an unfinished training and says so. Every script
runs with `uv run --no-sync`, so a run never changes the environment under itself.

Outside Colab, one run is one job:

```bash
mkdir -p outputs/logs
NUM_WORKERS=$(nproc) bash slurm/full.sbatch configs/full/unet_scratch-f1.0.json 2>&1 | tee -a outputs/logs/unet_scratch-f1.0.log
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
