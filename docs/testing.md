# Testing

Read before changing the pipeline or adding a test.

```bash
pytest -q
ruff format --check . && ruff check .
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
  `garimpo fit_directions` at 2 threads; that run now passes `--direction-temperature 1` to every stage that takes it, as a job would.
- `test_run.py` writes configs of its own, whose `CELL_PX` is 32, the synthetic cells' size, where
  the real store's is 512.
- oneDNN picks its CPU kernels by the kind of core a process runs on: on a hybrid CPU the ConvNet's
  dilated convolution gives other last bits on a performance core than on an efficiency core. So
  `tests/conftest.py` keeps every test process, workers and stages included, on the efficiency
  cores, and any comparison across processes is exact.
- CI runs the same `pytest -q` on Python 3.13, Colab's, after `pip install -r requirements.txt`,
  with a 45-minute limit.

## Modules

| Module | Subject |
|---|---|
| `test_models.py` | f/g decomposition of each architecture; ImageNet and Sentinel-2 starting weights; renamed decoder modules refused |
| `test_tile_lists.py` | Counts and nesting on the real `outputs/split.csv`; the versioned lists regenerated and compared by bytes, tiles and `select_tiles` |
| `test_store.py` | The store's tiles written in shuffled order and compared by bytes; the dataset reading slots; a sidecar's radiometric offset reaching only its own tile's cells |
| `test_download.py` | The download code on local rasters, including `lists` and `scenes --tiles` from a small list to a larger one, the refusals of a list, Ctrl+C finishing the tile in progress with status `interrupted`, a second Ctrl+C quitting without waiting, the log file kept across runs, and the SAS token renewal and HTTP 429 wait on a fake clock and HTTP |
| `test_validation.py` | Defects in the store and the split against the validation, including a store of another cell size or `stats` length |
| `test_damaged_store.py` | A store damaged under a finished run: stages may skip, the validation and the report may not |
| `test_training.py` | What training writes, the training subset's eval-mode scores per epoch among it, and every refusal |
| `test_sentinel2_training.py` | The identity of the Sentinel-2 weights file, present, absent and replaced |
| `test_results.py` | PICP, MPIW, interval score and Dice recomputed one cell at a time against `results.csv`; every COMPASS row with its own loader against the one pass in `garimpo generate_intervals`, with no tolerance; the population of each subsampled row; every alpha level; coverage by stratum and the score relation; the monotonicity curves read back to the table's ascending share; the relative scores' floor; the weighted group ESS; the bootstrap resampling groups, with paired rows; width and coverage comparisons per level; the cells with no boundary to match, counted from the masks, and a mask filling its cell |
| `test_figures.py` | `garimpo figures` drawing every figure, PDF and SVG, from the finished U-Net and ConvNet runs, whose search profiles the fixtures' last stage wrote; `garimpo data_figures` from the store alone |
| `test_choose_layer.py` | The layer choice on hand-written diagnostics: validation role only, both directions, threshold, a missing direction, no qualifying layer, ties, a changed threshold, a refused rerun |
| `test_calibration_rules.py` | Repeated subsampling reduced to split conformal at one cell per group, its draws within and across groups, its seed, its refusal, and simulated coverage at k = 25, written to `tests/artifacts/calibration_rules/coverage_k25.json`; the density ratio's cross-fit by group, on the real split's groups and with a fold lacking a population; weights refused by the weighted quantile |
| `test_stages.py` | Reruns, refusals, saturation, an asymmetric quantile past its upper censoring, absent and misaligned cells, another data root, the score floors, the SVM orientation and a direction fit with no gradient, merge, peak memory records, the commit each stage records, and `garimpo check`, including an incomplete results table |
| `test_fractions.py` | Two fractions on one data root, the larger fraction's data arriving after the smaller run finished |
| `test_run.py` | `garimpo run` run as Colab runs it, from a root holding configs of its own: a full job to PASS, its log holding every line it printed and every stage's end, each flag that takes a value receiving, in the command the runner logs, the config value its name gives (or a short table of exceptions gives), every config value distinct from the others and each one reaching a stage, the chosen layer (not the first of `LAYERS`) reaching the calibrated stages, `--force-from` and `--force-validate` with a second attempt appended to the log, an unknown `--force-from`, a failed stage stopping the job, the other layer of `LAYERS` run through the layer's stages into `layers/<layer>/`, its output-space rows equal to the chosen layer's to the bit and its COMPASS-J rows not, the check naming an other layer that never ran, failing on a NaN in an other layer's `results.csv` and naming that layer, ending in a verdict, naming the cause, without `chosen_layer.json` or with it not an object, `--force-from` redoing the other layer's stages too, a job lost after `choose_layer` leaving the later stages, the other layer's included, `stale`, refused by the merge, and its rerun skipping them back to `ok`, Ctrl+C stopping the training inside its stage, its traceback still reaching the log, and the check still running, a refused config stopping the job before any stage, with its reasons in the log; in the process, every key left null, a misspelled key, a key in another file, a `SCORE_FLOOR_HA` that is not a positive finite number, every versioned model at every scale accepted with only layers of its model, and each one's run directory and log name, those of the recorded runs |
| `test_resume.py` | Training killed (SIGKILL, as a Colab session ending, and Ctrl+C) once epoch 0's `last.pt` exists, then rerun: it resumes after epoch 0 and ends with the weights, optimizer state, generator, best checkpoint and history of an uninterrupted run, with no tolerance on CPU; a resume keeps the best Dice so far; `--force` refuses to discard an unfinished training and never resumes a finished one's end |
| `test_workers.py` | Two DataLoader workers: two trainings with one seed give one model and the band statistics of the shared run; every prediction equals those read without workers |

## On a GPU

Run the suite on a Colab GPU runtime before the first paid run: check `RUN_TESTS` in section 3
of `notebooks/01_run_experiment.ipynb`, and section 4 runs:

```python
!GARIMPO_TEST_DEVICE=cuda pytest -q
```

The stages then get no `--device`, so they choose the GPU through `checkpoints.default_device` as a
real run does, and every loader runs with 2 workers and pinned memory. `test_training_artifacts`
then requires a positive GPU peak in `history.csv`, the `infer` status and `status/train.json`.
Nothing is skipped there. The tests that recompute a stage in the test process do it on the GPU,
as the stage did, and still require exact equality. `test_workers.py` compares 0 and 2 workers
explicitly for inference. GPU kernels sum in no fixed order, so the trainings repeated in
`test_workers.py` and `test_resume.py` compare their float weights and each optimizer moment as one
vector, whose difference must stay under `GPU_RELATIVE_DIFFERENCE` of `tests/e2e/support.py` in
norm, and the history within 1% (`HISTORY_TOLERANCE`), with no peak GPU memory; integers and the
generators, CUDA's included, must still match exactly. On CPU every comparison stays exact.
No GPU run of this version is recorded yet.

## Not covered

- `--cells nonempty`
- real HTTP (STAC, SAS tokens, 429 retries, Zenodo; token renewal and the 429 wait run against a fake
  clock and HTTP), remote reads, real Sentinel-2 scenes
- `garimpo download_data all`
- a second interrupt, which kills the stage: the training stops on the first one
- the pilot configs' own run: `test_run.py` runs the job on its own values
- `garimpo download_data metadata` and `weights`
- cleanup of the temporary file in `garimpo infer`
- GPU, unless run as in [On a GPU](#on-a-gpu); CI has none
- the real ImageNet weights (a synthetic file covers the load)
- the real Sentinel-2 MoCo file: a synthetic 13-band file stands in, with the variant trained for
  one epoch and only inference after it

The boundary F1 assertion in `test_wider_tolerance_can_only_match_more_boundary` depends on the
short synthetic training not being degenerate; it held on the suite's fixed seeds.
