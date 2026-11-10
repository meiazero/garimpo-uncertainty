# Testing

Read before changing the pipeline or adding a test.

```bash
uv run pytest -q
uv run ruff format --check . && uv run ruff check .
```

## How the suite runs

- The tests live in `tests/e2e/` and leave their artifacts in `tests/artifacts/`, which is reset
  when a session starts.
- They run on 4 workers by default; `pytest -n <N>` changes that and `pytest -n 0` runs serially.
  Each module has one subject and runs whole on one worker (`--dist loadfile`).
- `tests/e2e/conftest.py` builds the shared dataset and the finished runs once, under a file lock,
  and every worker reads them.
- A scenario that damages data works on a private copy under `tmp_path`, or, when it must still
  skip, on data of its own in a module fixture, since fingerprints record paths.
- Each process runs with `OMP_NUM_THREADS=1`, because the thread count changes the trained weights.
  `tests/conftest.py` records that the one-tile run of `test_fractions.py` failed
  `fit_directions.py` at 2 threads; that run now passes `--direction-temperature 1`.
- `test_run.py` sets `GARIMPO_TEST_CELL_PX`, which `scripts/run.py` passes to
  `validate_data.py` as `--cell-px`, because the synthetic cells are 32 pixels, not 512.
- oneDNN picks its CPU kernels from the core a process starts on. On a hybrid CPU (performance and
  efficiency cores) two trainings with one seed therefore differ in the last bits, so a test that
  compares weights across processes runs them inside `support.one_core()`, which pins them to one
  CPU.
- CI runs the same `uv run --no-sync pytest -q` after `uv sync --locked`, with a 45-minute limit.

## Modules

| Module | Subject |
|---|---|
| `test_models.py` | f/g decomposition of each architecture; ImageNet and Sentinel-2 starting weights; renamed decoder modules refused |
| `test_tile_lists.py` | Counts and nesting on the real `outputs/split.csv`; the versioned lists regenerated and compared by bytes, tiles and `select_tiles` |
| `test_store.py` | The store's tiles written in shuffled order and compared by bytes; the dataset reading slots; a sidecar's radiometric offset reaching only its own tile's cells |
| `test_download.py` | The download code on local rasters, including `lists` and `scenes --tiles` from a small list to a larger one, the refusals of a list, Ctrl+C finishing the tile in progress with status `interrupted`, a second Ctrl+C quitting without waiting, the log file kept across runs, and the SAS token renewal and HTTP 429 wait on a fake clock and HTTP |
| `test_validation.py` | Defects in the store and the split against the validation, including a store of another cell size or `stats` length |
| `test_damaged_store.py` | A store damaged under a finished run: stages may skip, the validation and the report may not |
| `test_training.py` | What training writes and every refusal |
| `test_sentinel2_training.py` | The identity of the Sentinel-2 weights file, present, absent and replaced |
| `test_results.py` | PICP, MPIW, interval score and Dice recomputed one cell at a time against `results.csv`; every COMPASS row with its own loader against the one pass in `generate_intervals.py`, with no tolerance; the population of each subsampled row; every alpha level; coverage by stratum and the score relation; the relative scores' floor; the weighted group ESS; the bootstrap resampling groups, with paired rows; width and coverage comparisons per level; the cells with no boundary to match, counted from the masks, and a mask filling its cell |
| `test_choose_layer.py` | The layer choice on hand-written diagnostics: validation role only, both directions, threshold, a missing direction, no qualifying layer, ties, a changed threshold, a refused rerun |
| `test_calibration_rules.py` | Repeated subsampling reduced to split conformal at one cell per group, its draws within and across groups, its seed, its refusal, and simulated coverage at k = 25 in `tests/artifacts/calibration_rules/coverage_k25.json`; the density ratio's cross-fit by group, on the real split's groups and with a fold lacking a population; weights refused by the weighted quantile |
| `test_stages.py` | Reruns, refusals, saturation, an asymmetric quantile past its upper censoring, absent and misaligned cells, another data root, the score floors, the SVM orientation and a direction fit with no gradient, merge, peak memory records and `pipeline_check.py`, including an incomplete results table |
| `test_fractions.py` | Two fractions on one data root, the larger fraction's data arriving after the smaller run finished |
| `test_run.py` | `scripts/run.py` run as Colab runs it, from a root linking the real `scripts/`: a full job to PASS, its log holding every line it printed and every stage's end, each flag that takes a value receiving, in the command the runner logs, the config value its name gives (or a short table of exceptions gives), every config value distinct from the others and each one reaching a stage, the chosen layer (not the first of `LAYERS`) reaching the calibrated stages, `--force-from` and `--force-validate` with a second attempt appended to the log, an unknown `--force-from`, a failed stage stopping the job, Ctrl+C stopping the training inside its stage, its traceback still reaching the log, and the check still running, every key of `configs/full/unet_scratch.json` left null, a misspelled key, also in the log, a `SCORE_FLOOR_HA` that is not a positive finite number, every versioned config accepted with only layers of its model, and each pilot config differing from its full one only in `RUN_DIR`, `TRAIN_FRACTION`, `EPOCHS`, `STEPS_PER_EPOCH` and `MIN_FRACTION_ASCENDING` |
| `test_resume.py` | Training killed (SIGKILL, as a Colab session ending, and Ctrl+C) once epoch 0's `last.pt` exists, then rerun: it resumes after epoch 0 and ends with the weights, optimizer state, generator, best checkpoint and history of an uninterrupted run, with no tolerance; a resume keeps the best Dice so far; `--force` refuses to discard an unfinished training and never resumes a finished one's end |
| `test_workers.py` | Two DataLoader workers on CPU: two trainings with one seed give one model, band statistics and every prediction equal those read without workers |

## On a GPU

Run the suite on a Colab GPU runtime before the first paid run, with section 4 of `colab.ipynb`.
Its cell runs:

```python
!GARIMPO_TEST_DEVICE=cuda uv run --no-sync pytest -q
```

The stages then get no `--device`, so they choose the GPU through `checkpoints.default_device` as a
real run does, and every loader runs with 2 workers and pinned memory. `test_training_artifacts`
then requires a positive GPU peak in `history.csv`, the `infer` status and `status/train.json`.
Skipped there: `test_workers.py`, which every stage then covers; the two `test_results.py` tests
that recompute a stage in the test process on CPU; the `test_training.py` test that recomputes the
validation metrics from `best_model.pt` on CPU; and the comparison with the uninterrupted run in
`test_resume.py`, since GPU kernels are not deterministic. No GPU run of this command is recorded
yet.

## Not covered

- `--cells nonempty`
- real HTTP (STAC, SAS tokens, 429 retries, Zenodo; token renewal and the 429 wait run against a fake
  clock and HTTP), remote reads, real Sentinel-2 scenes
- `download_data.py all`
- a second interrupt, which kills the stage: the training stops on the first one
- the pilot configs' own run: `test_run.py` runs the job on its own values
- `download_data.py metadata` and `weights`
- cleanup of the temporary file in `infer.py`
- GPU, unless run as in [On a GPU](#on-a-gpu); CI has none
- the real ImageNet weights (a synthetic file covers the load)
- the real Sentinel-2 MoCo file: a synthetic 13-band file stands in, with the variant trained for
  one epoch and only inference after it

The boundary F1 assertion in `test_wider_tolerance_can_only_match_more_boundary` depends on the
short synthetic training not being degenerate; it held on the suite's fixed seeds.
