# Run the pilot

Read before the first run on a GPU.

The pilot is the four models of the full run on the 1% training fraction, run through the same
script: 186 tiles and 2,976 cells, since validation, calibration and test stay whole. It trains on
the versioned list `outputs/tiles/tiles_1pct.json`. Its purpose is to check on the GPU the code the
full runs will run, and its numbers say nothing about any method.

Each model's pilot config in `configs/pilot/` equals its full config in `configs/full/`
([decisions.md](decisions.md)) except in its scale: `TRAIN_FRACTION` 0.01, `EPOCHS` 2 and
`STEPS_PER_EPOCH` 20, so it stays short, `MIN_FRACTION_ASCENDING` 0, so a model trained for 40
updates still reaches the last stage, and `RUN_DIR` `runs/pilot/<model>-f0.01`. `test_shell.py`
fails on any other difference. Results go to `runs/pilot/`, never to `outputs/`.

## Steps

1. Put the data archive in Drive as `MyDrive/garimpo/data.tar.zst`
   ([full-run.md](full-run.md#download-the-store)). This is needed once.
2. On a GPU runtime, run Runtime > Run all in `colab.ipynb`. Section 3's `SCALE`, at `auto`, runs
   the pilot while no full run has a checkpoint; choose `pilot` to run it after that.
3. Read the verdicts section 5 prints, then the pipeline check at the end of each model's log,
   `outputs/logs/pilot-<model>.log`, also copied to `logs/` in Drive.

Each model has its own cell, and they run one after another on the one GPU. Each writes to
`runs/pilot/<model>-f0.01/`, and section 6 merges their tables into `runs/pilot/tables/`. The runs
and logs reach Drive and the notebook's last cell releases the GPU even when a model fails.

Outside Colab, `bash slurm/run.sbatch configs/pilot/<model>.json` runs one model's pilot.

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
2. In a new session, check the `RUN` box of each "Rerun ... - (in a reexecution)" cell and choose
   the first stage the fix affects in its `FORCE_FROM` option, then run Runtime > Run all. The options apply to their own cell only. They stay set in the open tab, so
   clear `RUN` once the rerun passes; reopening the notebook from its link also clears it.

Commits are recorded in the metadata but kept out of the fingerprint, so a code fix alone
invalidates nothing; that is why the stage has to be named. `FORCE_FROM` takes a name from
`STAGE_ORDER` in `slurm/stages.sh` and redoes that stage and every stage after it. Earlier stages
skip, and `train` runs again only with `FORCE_FROM=train`, which refuses while `last.pt` holds an
unfinished training: drop it to resume, or delete `last.pt` to start over. The data validation is
outside that order: redo it with `FORCE_VALIDATE=1`. Both variables work the same way for the full
runs.

A stage whose inputs or arguments changed without being forced stops with status `refused`, which
fails the check. A run stopped midway keeps its finished stages, and they skip on the next start
once `runs/` is restored from Drive with its modification times.
