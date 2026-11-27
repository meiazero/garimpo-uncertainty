# Cookbook

Recipes for the situations the experiment runs into, each a few steps long. The why behind each
step lives in the document it links to.

## Run the experiment from the start

1. On your own machine, build the data archive and put it in Drive as
   `MyDrive/garimpo/data.tar.zst` ([full-run.md](full-run.md#download-the-store)). This is needed
   once.
2. Open `notebooks/01_run_experiment.ipynb` from its GitHub link, on an L4 runtime at the latest
   runtime version (Runtime > Change runtime type).
3. Run Runtime > Run all and keep the tab open. Nothing needs to be set.
4. Read the verdicts at the end of section 5: one `PIPELINE CHECK: PASS` or `FAIL` per model.

The session runs the tests on the GPU once per commit, then the pilot until all four models pass on
the commit checked out, then the four full runs ([colab.md](colab.md#the-gates)).

## Continue after the session ended

1. Open the notebook from its GitHub link again, on an L4 runtime.
2. Run Runtime > Run all.

Finished stages skip, and a training resumes from its last checkpoint, copied to Drive every 30
minutes. The tests and the pilot skip on a commit once they passed on it.

## Stop a session early

1. Interrupt section 5 once. The current stage stops, its check still prints, and no later model
   runs.
2. Run sections 6 and 7 by hand: they copy the runs and logs to Drive and release the GPU.

The next Run all resumes where the stage stopped.

## Change the code

1. Change it on your own machine, run `uv run ruff format . && uv run ruff check .`, then commit and
   push to `main`. A session's clone cannot push.
2. In a new session, run Runtime > Run all. The new commit runs the tests, then redoes the pilot
   from `train`, and the full runs wait until all four pilots pass on it.

A change to a stage the full runs already finished does not redo it on its own: see the next recipe.

## Redo a stage of the full runs after a fix

1. Push the fix, as above.
2. In section 1, set `FORCE_FROM` to the first stage the fix touches, then run Runtime > Run all.
   The four full runs redo that stage and every one after it. `validate_data` cannot be chosen,
   and a choice after `choose_layer` does not redo `search_profile`, which reads none of them.
3. Clear `FORCE_FROM` once they pass; reopening the notebook from its link also clears it.

`FORCE_FROM` refuses `train` while a full run's `last.pt` holds an unfinished training
([pilot.md](pilot.md)).

## A test failed

No run starts on a commit whose tests failed. Section 4 prints the failing test: fix the code, push,
and run Runtime > Run all again ([testing.md](testing.md)).

## A pilot failed

The full runs wait. Open the pilot's log, `MyDrive/garimpo/logs/pilot-<model>.log`, and find the
line `<stage> FAILED after ...`: the stage's last 30 lines follow it, and `PIPELINE CHECK: FAIL at
['<stage>']` closes the log. Fix, push, and run again ([pilot.md](pilot.md)).

## A full run failed

The other models still run. Read `MyDrive/garimpo/logs/full-<model>.log` the same way, then fix and
push. Run again without `FORCE_FROM`: a failed stage runs again on its own, and an unfinished
training resumes. Set `FORCE_FROM` only when the fix changed an earlier stage that already
finished.

## Get the final tables

Once all four full runs pass, section 5 merges their tables into `outputs/` and copies them to
`MyDrive/garimpo/outputs/`. Download that folder and commit it from your own machine
([outputs.md](outputs.md)).

## Draw the figures

- The dissertation's figures: see [figures.md](figures.md).
- The qualitative comparison of the models: unpack the store into
  `MyDrive/garimpo/garimpo-dataset/` once, then open `notebooks/02_compare_segmentation.ipynb` from
  its GitHub link, on a CPU runtime, and run Runtime > Run all. The figure goes to
  `MyDrive/garimpo/figures/<scale>/`.

## Size the run to the GPU

Open `notebooks/03_tune_hardware.ipynb` from its GitHub link on an L4 runtime and run Runtime > Run
all. It measures data loading, training, the COMPASS search, the batch size, the float32 precision
and LightGBM's threads on synthetic cells, and ends with `TUNING: PASS` or `TUNING: FAIL`. Copy the
verdict before the last cell releases the runtime.

## Move to a newer Colab runtime

1. If Colab moved to another CUDA, change the `cu130` index in `pyproject.toml` and in
   `.github/workflows/ci.yml` to match its torch build.
2. Lock against the pip freeze of the image Colab serves: the commit of
   github.com/googlecolab/backend-info whose freeze the runtime matches, not the newest, in place
   of `6e5e98c`. Then export:

   ```bash
   uv add --upgrade -c https://raw.githubusercontent.com/googlecolab/backend-info/6e5e98c/pip-freeze.gpu.txt --group dev --bounds exact pytest pytest-xdist ruff
   uv export --format requirements-txt --no-hashes --group dev -o requirements.txt
   ```

3. Commit `pyproject.toml`, `uv.lock` and `requirements.txt`, and push. The next session's tests and
   pilot run on the new commit before any full run.
4. Another torch or CUDA build changes the GPU's bits, but a stage's fingerprint does not record it:
   full runs that already finished stages keep them. Set `FORCE_FROM` to `train` to redo them.

## Run outside Colab

```bash
git clone --recurse-submodules https://github.com/meiazero/garimpo-uncertainty.git
cd garimpo-uncertainty
uv sync
uv run garimpo run pilot unet_scratch
```

The lock covers Linux on x86_64 only. A run reads the unpacked store from `GARIMPO_DATA_DIR`,
`~/Experiments/datasets/garimpo` by default; `uv run garimpo download_data all` builds it there.
Results on another GPU or another software stack differ in their last bits from the L4's
([design.md](design.md)).

## The time it takes

On the L4, with every stage under deterministic algorithms. Measured where a source is named; the
rest is an estimate.

| Part | Per run | Runs | Source |
|---|---|---|---|
| Training, U-Net | ~3.2 h: 25,000 updates at ~0.42 s with 2 workers, plus ~7 s of evaluation per epoch | 3 | `garimpo tune`'s train section on the L4 (2026-11-27, [`outputs/tuning-l4/tune.log`](../outputs/tuning-l4/tune.log)), under deterministic algorithms; the pilot's `history.csv` |
| Training, ConvNet | ~3.9 h: 25,000 updates at ~0.54 s with 2 workers, plus the evaluation | 1 | The same run of `garimpo tune` |
| COMPASS scores, U-Net | ~2.6 h on the 5 layers | 3 | The projection of the same run of `garimpo tune`; the pilot's 5.6 h, measured before each beta was decoded once (PR #58), divided by the 2.2 times fewer decodes that run measured gives ~2.5 h |
| COMPASS scores, ConvNet | ~0.1 h | 1 | The search section of the same run of `garimpo tune` |
| Intervals and the other stages | up to ~1 h | 4 | The pilot's statuses: 44 min of `generate_intervals`, the rest under 10 min |

That is about 25 h for the four full runs. The pilot scores 864 cells, 31% of the full run's 2,816,
so its stages after training take about 1.1 h per U-Net and 4 h for the four models. The tests on
the GPU and `search_profile` are not measured yet.
