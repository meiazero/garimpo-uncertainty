# Run the full experiment

Read when the pilot passes and the values in [decisions.md](decisions.md) are fixed.

## Steps

1. Download the store once on your own machine and upload it to Drive
   ([below](#download-the-store)). The notebook downloads nothing.
2. Each run reads every value from `configs/` ([below](#the-run-configs)). To
   change one, edit its file, then commit and push it from your machine: each session starts from a
   fresh clone, which cannot push.
3. On a GPU runtime, run Runtime > Run all in `notebooks/01_run_experiment.ipynb`. Once the four
   pilot runs passed on the commit checked out, section 5 runs the four full runs, one after
   another
   ([below](#the-run-cell)). Each run chooses the layer from the validation role and continues;
   `chosen_layer.json` records why.
4. If the session ends first, run Runtime > Run all again in a new session. The full runs'
   `last.pt`, restored from Drive, lets them continue: finished stages skip and training resumes
   from its last checkpoint. A new commit runs the tests and the pilot first.
5. Section 5 then merges the four runs' tables into `outputs/` with the command below; on your own
   machine, copy `runs/` from Drive first.

   ```bash
   garimpo merge --out outputs --scale full
   ```

6. On Colab, section 5 copies `outputs/` to `MyDrive/garimpo/outputs/`. Download it and commit it
   from your own machine: these are the final result tables.

## The run configs

| Model config | Model |
|---|---|
| `configs/models/unet_scratch.json` | U-Net from scratch |
| `configs/models/unet_imagenet.json` | U-Net from ImageNet weights |
| `configs/models/unet_sentinel2.json` | U-Net from Sentinel-2 MoCo weights |
| `configs/models/convnet.json` | DeepLabV3 from segmentation-models-pytorch |

A run is one of these models at the `full` scale, `configs/scales/full.json`, with the values of
`configs/base.json` that every run shares. Each value has its source in
[decisions.md](decisions.md). `garimpo run` composes the three files and stops the run before
any stage when a key is missing, null, empty (except `AUGMENTATIONS`, which may be `[]`), unknown or in another file, or when `SCORE_FLOOR_HA`
is not a positive finite number. The run writes to `runs/full-<D-M-Y>/<model>-f100pct-<D-M-Y>`,
dated the day it started: the merge reads each model's latest run. The `pilot` scale runs the same four models at 1% ([pilot.md](pilot.md)).

## Download the store

On your own machine:

```bash
GARIMPO_DATA_DIR=<DIR> garimpo download_data all
tar --format=posix -I 'zstd -19 -T0' -cf <DIR>.tar.zst -C <DIR> .
```

Then upload the archive to Drive as `garimpo/data.tar.zst`. `garimpo download_data all` downloads the
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
  `--force-validate --force-from train`, since a re-cut that changes a tile's slots changes the hash
  in its sidecar.

## The run cell

Section 5 of the notebook is one cell, "Run the pilot, then the full runs, then merge each scale
that ran", which calls `garimpo experiment`:

1. For each of the four models, in order, it runs `garimpo run pilot <model>` until all four
   pilots passed on the commit checked out, then `garimpo run full <model>`. `--num-workers` defaults to the runtime's CPUs. The
   cell shows one line per stage with its duration and key numbers, such as
   `[ 3/28] infer  ok  39.7 s · test area error 0.042 ha/cell, bias -0.004 ha`; while a stage
   runs, its epochs, and a progress bar once a minute in the long loops. A failed stage adds its
   last 30 lines. `outputs/logs/<scale>-<model>.log` holds all that and every other line, each
   stage's command included, with the UTC time.
2. Section 1's `FORCE_FROM` reaches every full job as `--force-from`. It stays set in the open
   tab, so clear it once the rerun passes; reopening the notebook from its link empties it.
3. For each scale that ran, it prints `garimpo status <scale> --since <session start>`, one line
   per model with the verdict of its last attempt, then runs `garimpo merge` for that scale when
   all its runs passed; a scale with a failed or stopped run keeps its earlier tables.

The copy of `runs/` and the logs to Drive every 30 minutes starts with the run cell and stops
in section 6, which saves the runs and logs once more; section 7 releases the GPU.

With the copy every 30 minutes, a session that ends mid-run keeps the finished stages and the
last checkpoint in Drive. `tests/e2e/test_resume.py` kills training after
its first epoch and requires the rerun to end where an uninterrupted run ends: exactly on CPU, and
on a GPU within a tolerance, the generators still exactly. Rerun without
`--force-from train` to resume: with it, `garimpo train` refuses to discard an unfinished training and says so. No `garimpo`
command installs anything, so a run never changes the environment under itself.

Outside Colab, one run is one job, and `garimpo experiment` runs them as the notebook does:

```bash
garimpo run full unet_scratch
garimpo experiment
```

## Regenerate the tile lists

The versioned lists use subset seed 0, the decided `SUBSET_SEED`. `lists` regenerates all four. Run
it only after `outputs/split.csv` changes, or to use another seed:

```bash
garimpo download_data lists --subset-seed <SUBSET_SEED>
```

The E2E suite fails if regenerating with the recorded seed does not reproduce the versioned files
byte for byte, so commit the regenerated files together with whatever changed them.

## Rerun one stage

A stage reruns on its own, without the earlier stages:
`garimpo <stage> ...`. Use a GPU runtime for the stages that
[stages.md](stages.md#stage-table) marks GPU. For example, a new metric or another boundary F1
tolerance, passing the run's `SEED` because the seed is part of the fingerprint:

```bash
garimpo compute_metrics --run-dir runs/full-28-11-2026/unet_scratch-f100pct-28-11-2026 --boundary-tolerance-px <PX> --seed <SEED> --force
```
