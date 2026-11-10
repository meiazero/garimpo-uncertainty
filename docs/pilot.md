# Run the pilot

Read before the first run on a GPU.

The pilot is the 1% training fraction, run through the same scripts as the full run: 186 tiles and
2,976 cells, since validation, calibration and test stay whole. It trains on the versioned list
`outputs/tiles/tiles_1pct.json`. Its purpose is to debug the implementation, and its numbers say
nothing about any method. Each variant reads its config in `configs/pilot/`, which keeps the full
run's values ([decisions.md](decisions.md)) except where the pilot must stay short or reach the last
stage: `TRAIN_FRACTION` 0.01, `EPOCHS` 2, `STEPS_PER_EPOCH` 20, `MIN_FRACTION_ASCENDING` 0, so
every layer qualifies, and `SEARCH_THRESHOLD_FEATURES` 1e-3, a third of the search steps. It
diagnoses the same `LAYERS` as the full run, so it exercises every layer the full run can choose.
Results go to `runs/pilot/`, never to `outputs/`.

## Steps

1. Put the data archive in Drive as `MyDrive/garimpo/data.tar.zst`
   ([full-run.md](full-run.md#download-the-store)). This is needed once.
2. On a GPU runtime, run Runtime > Run all in `colab.ipynb`. Section 5 is the pilot.
3. Read the verdicts the section prints, then the pipeline check at the end of each variant's log,
   `outputs/logs/pilot-<config>.log`, also copied to `logs/` in Drive.

Each variant (`unet-none`, `unet-imagenet`, `unet-sentinel2_moco`) has its own cell, and they run
one after another on the one GPU. Each writes to `runs/pilot/<config>/` and merges its tables into
`runs/pilot/tables-<weights>/`. The runs and logs reach Drive and the notebook's last cell releases
the GPU even when a variant fails.

Outside Colab, `bash slurm/pilot.sbatch configs/pilot/<config>.json download` downloads the pilot's
data and that variant's weights once, and `bash slurm/pilot.sbatch configs/pilot/<config>.json`
runs one variant.

## The check report

Every run ends with `scripts/pipeline_check.py`, which prints:

- status, seconds, peak memory and shapes per stage;
- the failing stage with its error;
- scenes missing and cells per role;
- the training history, with Dice, timings, data wait and GPU peaks per epoch;
- percent monotonic per direction, and the chosen layer with every candidate's fractions;
- `|cos(v1, w)|`, saturated searches, and PICP and MPIW per method;
- the run's slots against the hashes in their sidecars.

The first failing stage stops the job, so no later stage runs on its missing or stale outputs.
It receives the script's exit code, so an error outside every stage also reads FAIL. Which statuses
fail the check: [stages.md](stages.md#statuses).

## Rerun after a code fix

1. Commit and push the fix from your machine. The session's clone cannot push, and the next
   session's clone picks the fix up.
2. In a new session, check the `RUN` box of each "Rerun the pilot of ... - (in a reexecution)"
   cell and choose the first stage the fix affects in its `FORCE_FROM` option, then run
   Runtime > Run all. The options apply to their own cell only. They stay set in the open tab, so
   clear `RUN` once the rerun passes; reopening the notebook from its link also clears it.

Commits are recorded in the metadata but kept out of the fingerprint, so a code fix alone
invalidates nothing; that is why the stage has to be named. `FORCE_FROM` takes a name from
`STAGE_ORDER` in `slurm/stages.sh` and redoes that stage and every stage after it. Earlier stages
skip, and `train` runs again only with `FORCE_FROM=train`, which refuses while `last.pt` holds an
unfinished training: drop it to resume, or delete `last.pt` to start over. The data validation is outside that
order: redo it with `FORCE_VALIDATE=1`. Both variables work the same way with `slurm/full.sbatch`.

A stage whose inputs or arguments changed without being forced stops with status `refused`, which
fails the check. A run stopped midway keeps its finished stages, and they skip on the next start
once `runs/` is restored from Drive with its modification times.
