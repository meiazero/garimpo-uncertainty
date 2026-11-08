# Testing

Read before changing the pipeline or adding a test.

```bash
slurm/submit.sh slurm/cpu.sbatch -m pytest -q
uv run ruff format --check . && uv run ruff check .
```

The tests live in `tests/e2e/` and leave their artifacts in `tests/artifacts/`, reset when a session
starts. They run on 4 workers by default; `pytest -n <N>` changes that and `pytest -n 0` runs serially.
Each module has one subject and runs whole on one worker (`--dist loadfile`).
`tests/e2e/conftest.py` builds the shared dataset and the finished runs once, under a file lock, and
every worker reads them. A scenario that damages data works on a private copy under `tmp_path`, or on
data of its own in a module fixture when it must still skip, since fingerprints record paths. Each
process runs with `OMP_NUM_THREADS=1`: the thread count changes the trained weights, and the one-tile
run of `test_fractions.py` fails `fit_directions.py` at 2 threads.

| Module | Subject |
|---|---|
| `test_models.py` | f/g decomposition of each architecture; ImageNet and Sentinel-2 starting weights |
| `test_tile_lists.py` | Counts and nesting on the real `outputs/split.csv`; the versioned lists regenerated and compared by bytes, tiles and `select_tiles` |
| `test_store.py` | The store's tiles written in shuffled order and compared by bytes; the dataset reading slots |
| `test_download.py` | The download code on local rasters, including `lists` and `scenes --tiles` from a small list to a larger one and the refusals of a list |
| `test_validation.py` | Defects in the store and the split against the validation |
| `test_damaged_store.py` | A store damaged under a finished run: stages may skip, the validation and the report may not |
| `test_training.py` | What training writes and every refusal |
| `test_sentinel2_training.py` | The identity of the Sentinel-2 weights file, present, absent and replaced |
| `test_results.py` | PICP, MPIW, interval score and Dice recomputed one cell at a time against `results.csv`; every COMPASS row with its own loader against the one pass in `generate_intervals.py`, with no tolerance; the population of each subsampled row; every alpha level; coverage by stratum and the score relation |
| `test_choose_layer.py` | The layer choice on hand-written diagnostics: validation role only, both directions, threshold, a missing direction, no qualifying layer, ties, a changed threshold, a refused rerun |
| `test_calibration_rules.py` | Repeated subsampling reduced to split conformal at one cell per group, its draws within and across groups, its seed, its refusal, and simulated coverage at k = 25 in `tests/artifacts/calibration_rules/coverage_k25.json` |
| `test_stages.py` | Reruns, refusals, saturation, absent cells, merge and `pipeline_check.py` |
| `test_fractions.py` | Two fractions on one data root, the larger fraction's data arriving after the smaller run finished |

Not covered: `--cells nonempty`, real HTTP (STAC, SAS tokens, 429 retries, Zenodo), remote reads, real
Sentinel-2 scenes, the shell scripts in `slurm/` including `FORCE_FROM` and the EXIT trap,
`download_data.py metadata` and `weights`, a resume that actually continues training, cleanup
of the temporary file in `infer.py`, GPU, the real ImageNet weights (a synthetic file covers the load), and the real
Sentinel-2 MoCo file: a synthetic 13-band file stands in, with the variant trained for one epoch and
only inference after it. The boundary F1 assertion in `test_wider_tolerance_can_only_match_more_boundary` depends on the
short synthetic training not being degenerate; it held on the suite's fixed seeds.
