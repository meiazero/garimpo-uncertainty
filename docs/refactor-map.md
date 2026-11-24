# Refactor map: move the pipeline core into an installable package with one CLI; notebooks only drive or analyze it

Status: approved · Base: `ee8ac88` · Scope: whole codebase except `compass/` (own repository) · Additions: none

## Target layout

```
configs/
  base.json                    the 20 keys identical in every run
  models/<model>.json          ARCH, ENCODER_WEIGHTS, LAYERS
  scales/{pilot,full}.json     RUNS_DIR, EPOCHS, STEPS_PER_EPOCH, TRAIN_FRACTION, MIN_FRACTION_ASCENDING
scripts/split/                 one-shot build of the frozen outputs/split.csv; not part of the CLI
src/garimpo_uncertainty/
  __main__.py, cli.py          `garimpo <stage|run|check|merge|status>`, each subcommand imported lazily
  config.py                    RunConfig: compose, validate, derive RUN_DIR and log name
  artifacts.py                 RunPaths: run root and layer root
  files.py                     atomic writes, JSON/NPZ, sha256, code versions
  checkpoints.py, intervals.py, evaluation.py
  data/{store,cell_dataset,download,validation}.py
  models/                      unchanged
  stages/<stage>.py            moved from scripts/, same flags as today
  pipeline/{runner,run,check,merge,status}.py
notebooks/01_run_experiment.ipynb   Colab: Drive restore/copy/release, then `garimpo run|status|merge`
```

Constraints every fix keeps:

- Stage flags, run directories and log names stay byte-identical, so recorded fingerprints and `last.pt` arguments still match. A changed fingerprint does not redo a stage: it refuses (`src/garimpo_uncertainty/stages.py:213`).
- Supported install is the editable checkout (`uv sync`), as on Colab; `compass` is a local path source (`pyproject.toml:34`) and provenance reads git.
- No new dependency: argparse, dataclasses and JSON cover it.

## Fixes

- [x] 1. Move the file helpers out of the stage runner into `files.py`, so library modules stop importing pipeline infrastructure. (7 files)
  - `src/garimpo_uncertainty/checkpoints.py:24` and `:33` import `stages` for `code_versions` and `atomic`.
  - `src/garimpo_uncertainty/download.py:68`, `:185`, `:262` import `stages` for `atomic` and `write_json`.
  - SHA-256 is written three times: `checkpoints.py:17`, `download.py:96`, and `scripts/download_data.py:30`, the last reading the whole file into memory.
- [x] 3. Name every artifact once in `RunPaths`, with a run root and a layer root, so writer and readers cannot drift. (~12 files)
  - `chosen_layer.json` is spelled 8 times (`scripts/choose_layer.py:58`, `pipeline_check.py:189`, `:247`, `merge_results.py:37`, `run.py:293`, …); `scores.npz` 7, `quantiles.npz` 6, `results.csv` 6, `cells.npz` 5, across 9 scripts and `src/garimpo_uncertainty/data.py`.
  - A layer stage reads shared predictions from the run root and its own scores from the layer root (`scripts/calibrate.py:432`); one root cannot express that.
  - The evaluated roles (validation, calibration, test) are restated in `scripts/infer.py:26`, `calibrate.py:194`, `compass_scores.py:201`. `verify_monotonicity.py:24` selects roles by purpose and stays explicit.
- [x] 4. Replace the 8 flat configs and `run_config.py` with a typed `RunConfig` composed from base + model + scale, deriving today's `RUN_DIR` and log name. (~12 files)
  - 20 of the 28 keys are identical in all 8 files; changing `SVM_C` touches 8 files.
  - Pilot and full differ only in `EPOCHS`, `STEPS_PER_EPOCH`, `TRAIN_FRACTION`, `MIN_FRACTION_ASCENDING`, `RUN_DIR`.
  - `src/garimpo_uncertainty/run_config.py:9` keeps the key list by hand; `:54` re-checks `SCORE_FLOOR_HA` because `scripts/calibrate.py` checks it only after training; `ENCODER_WEIGHTS` is a special case outside `REQUIRED` (`:39`).
  - The log name comes from the JSON's folder (`scripts/run.py:326`); `tests/e2e/test_run.py:536` globs `configs/*/*.json`.
- [x] 5. Generate each stage's flags from `RunConfig` through one declared table, keeping every stage's CLI as it is. (~6 files, after 4)
  - `scripts/run.py:138-272` (`Stages`) maps config keys to flags by hand; `tests/e2e/test_run.py:231-259` checks that mapping key by key.
  - `tests/e2e/support.py:310` (`train_args`), `:354` (`checkpoint_args`) and its other builders are a third copy.
  - `--num-workers` defaults to 4 in 6 stages (`scripts/infer.py:120`, `train.py:546`, …) while `run.py:45` defaults it to the CPU count.
  - `scripts/run.py:133` reads `GARIMPO_TEST_CELL_PX`, a test-only variable, in production code.
- [x] 6. Move every stage from `scripts/` into `garimpo_uncertainty.stages`, behind the `garimpo` CLI with lazy imports; `scripts/` keeps only `split/`. (~30 files, after 1, 3)
  - Stage logic lives in non-importable scripts: `scripts/train.py` (584 lines), `calibrate.py` (448), `pipeline_check.py` (359).
  - Call sites by path: `tests/e2e/support.py:296`, `test_resume.py:41`, `:161`, `test_run.py:119`, `:160` (parses `scripts/` out of the log), `:175` (symlinks `scripts/` into a fake cwd), `:313`, `:417`, `:429`; by `runpy`: `support.py:105` (`TORCH_FREE`), `test_results.py:853`, `test_download.py:611`, `test_tile_lists.py:112`.
  - Messages and docs: `scripts/train.py:53`, `:479`, `README.md:14`, `docs/outputs.md` (18), `docs/full-run.md` (8), the notebook (12), 7 other docs.
  - `pyproject.toml:165` allows `print` only under `scripts/*`; the moved CLI modules need the same exemption, or CI and the pre-push hook fail.
  - The CLI imports only the selected subcommand, so `compute_metrics` stays free of torch (`checkpoints.py:8` imports it).
  - `HELD_OUT_REGIONS` is duplicated at `scripts/split/make_split.py:12` and `src/garimpo_uncertainty/data.py:25`; the split script imports it.
- [x] 7. Group the status runner, orchestrator, check and merge in `pipeline/`, with one status reader and no Slurm leftovers. (6 files, after 6)
  - `read_status` is defined twice: `scripts/merge_results.py:43` returns a string, `scripts/pipeline_check.py:25` a dict.
  - `src/garimpo_uncertainty/stages.py:188` records `SLURM_JOB_ID`, printed at `scripts/pipeline_check.py:90` and documented at `docs/data-layout.md:110`; `slurm/` no longer exists.
- [x] 8. Move the notebook's run logic into `garimpo status`; the notebook keeps only the Colab adapters. (3 files, after 4, 6, 7)
  - The attempt marker is duplicated in the notebook (`ATTEMPT`, `print_verdicts`) and `scripts/run.py:30`.
  - `MODELS` and `checkpointed()` in the notebook re-derive the configs and checkpoints `RunConfig` knows.
  - Drive restore, the copy every 30 minutes and `release_runtime` stay in the notebook: they exist only on Colab.

## Second sweep (base `a6ab152`, approved)

- [x] 9. Delete the stages' fallback to the run root, which `RunPaths.of` already does. (4 files)
  - `args.layer_dir = args.layer_dir or args.run_dir` at `stages/calibrate.py:428`, `compass_scores.py:194`, `generate_intervals.py:234`, `compute_metrics.py:189`; `artifacts.py:32` falls back the same way, and `layer_dir` is outside the fingerprint (`pipeline/runner.py:34`).
- [x] 10. Read `chosen_layer.json` through one function, so the job, the check and the merge refuse the same broken files. (4 files)
  - `pipeline/run.py:164` catches `(OSError, ValueError, KeyError)`; `pipeline/check.py:26` (`UNREADABLE`) also catches `TypeError`, so a JSON list there is a FAIL in the check but a traceback in the job, before the check runs.
  - `pipeline/merge.py:41` is a third reader.
- [x] 11. Make the config the only source of each value the job passes: those flags have no default in their stage, and the test builders take their flags from `pipeline/commands.py`. (~10 files)
  - `stages/compass_scores.py:178` defaults `--search-threshold-features` to `1e-8`; `configs/base.json:18` holds `1e-15`. A stage run by hand searches with another threshold and says nothing.
  - 13 other defaults copy a config value: `--seed` (4 stages), `--direction-temperature` (3), the beta ranges and the logits threshold (`compass_scores.py:175-177`), `--cell-px`.
  - `tests/e2e/support.py:365`, `:412`, `:428`, `:460`, `:490` write 35 flags by hand beside the table in `pipeline/commands.py`; `direction_args` (`support.py:423`) has no caller.
- [x] 12. Put the COMPASS-J recipe in one library module: the Jacobian of the soft area, the per-direction channel deltas, the search spaces and `DIRECTIONS`. (6 files, after 9, 11)
  - The Jacobian metric is written at `stages/fit_directions.py:44`, `verify_monotonicity.py:66`, `compass_scores.py:80`.
  - `for direction in ("pca", "svm")` at `verify_monotonicity.py:75`, `compass_scores.py:88`, `generate_intervals.py:74`; `choose_layer.py:18` names it `DIRECTIONS`.
  - `compass_scores.py:69-99` (`batch_spaces`) and `generate_intervals.py:57-82` (`search_spaces`) build the same three spaces.
- [x] 13. Let `data.py` name the store's layout once: the split path, a tile's cell slots and the starting-weights files. (6 files)
  - `"outputs/split.csv"` is the default at `stages/validate_data.py:31`, `train.py:525`, `download_data.py:162`, `:166`, `pipeline/check.py:59`, and is copied into a Namespace at `download_data.py:121`.
  - `tile_index * cells_per_tile` at `validation.py:163`, `:185`, `download.py:236`, `data.py:188`; `validation.py:183` infers the count from `len(cells) // len(ok_tiles)` instead of the sidecar's `cells_per_tile`.
  - `resnet18_sentinel2_all_moco-59bfdff9.pth` and its hash are in `data.py:74` and `download.py:46`; the ImageNet file the same way (`data.py:73`, `download.py:49`).

Later, low value:

- `configs` and `outputs/logs` are spelled in `pipeline/run.py:208`, `:217`, `status.py:18-19`, `merge.py:86`.
- `intervals.py:75` finds the weighted rows with `str.contains("weighted")` on the labels of `COMBINATIONS`; a renamed label changes `layer_results.csv` silently.

Not worth it:

- Removing `--temperature`, `--svm-max-iter` or `--n-components`, which no command passes: a flag removed from a stage changes every recorded fingerprint, which then refuses.
- `CELL_PX` as both `data.CELL_PX` and a config key: a mismatch fails validation with `store_layout`, never silently.
- Deriving `cli.STAGES` and `runner.STAGE_ORDER` from `commands.STAGES`: command names and stage ids differ (`verify_monotonicity` runs as two stages), and a new stage is rare.

## Third sweep (base `3b44c6c`, approved)

- [x] 14. Let `data.py` name the tile sidecars and their download statuses, and `download_data` its tile lists. (4 files)
  - `tiles/<tile_id>.json` is built by hand at `data.py:181`, `:241`, `download.py:149`, `:243`, `:356`, `:413`.
  - `"missing_item"` and `"no_mask_polygon"` are written at `download.py:389`, `:597` and matched at `validation.py:20`.
  - `stages/download_data.py:42` builds `tiles_<pct>pct.json` and `:122` spells `tiles_100pct.json`.
- [x] 15. Keep the stage names and log strings the pipeline shares in `pipeline/runner.py`, so the check and the status stop reading other modules' text. (5 files)
  - `pipeline/check.py:124` tells a missing store by `startswith("store_missing")`, the text `validation.py:145` writes; `data.missing_arrays` answers it directly.
  - `"download_scenes"` at `check.py:96`, `:328`, written by `stages/download_data.py:92`.
  - `pipeline/status.py:15-16` imports `pipeline.check` (and with it sklearn) and `pipeline.run` only for `VERDICTS` and `ATTEMPT`.
- [ ] 16. Derive two counts from what they count. (2 files)
  - `evaluation.py:273` labels the area strata `np.arange(6)`, which is `len(AREA_QUANTILES) + 2`.
  - `stages/compass_scores.py:126-128` writes `beta_range` as `[logits, features, features]` by position, beside the limits `batch_spaces` already pairs with each method.
- [ ] 17. Drop the parameter no caller passes: `steps_per_epoch` of `tests/e2e/support.py:321` (`train_args`). (1 file)

Not worth it (third sweep):

- Parsing the calibration labels by suffix in `stages/calibrate.py`: the labels are columns of every recorded `results.csv`, and one module reads them.
- Literal `.npz` keys between writer and reader: a renamed key fails loudly with `KeyError`.
- `FORCE_FROM` in the notebook's cells 39-42: a Colab form needs its options written in the cell.
- Passing `--temperature`, a `hard_area` helper, one `censored` rule, `ENCODER_WEIGHTS: None` in `job_values`, the literals of `support.py`: one place each to change, or a value that never varies.
- `scripts/split/` (`* 16`, the GeoPackage path from the working directory): the split is frozen and versioned.
- `files.py:49` returning `"unknown"` without git: every supported install is a git checkout.

## Cut after review

- Fix 2, paths from the project root: fingerprints and `last.pt` record cwd-relative paths (`outputs/split.csv`, `runs/...`), and `tests/e2e/test_run.py` runs the runner from a workspace of its own. The working directory stays the workspace, as on Colab (`%cd {CODE}`).

- A shared stage preamble: `checkpoints.default_device()` is already one function; repeating its call is no duplication.
- `garimpo split`: `outputs/split.csv` is frozen and versioned; its build stays a one-shot script.
- `models/checkpoints.py` and a `data/split/` subpackage: no change lands in fewer places for them.

## Special cases kept

- Each stage runs in its own process (`scripts/run.py:78`): CPU stages must stay free of torch (`tests/e2e/support.py:105`), `ru_maxrss` measures one stage's peak only per process (`stages.py:126`), and the GPU memory is freed between stages.
- `scripts/run.py:324` turns SIGTERM into Ctrl+C: Colab's second interrupt is SIGTERM, then SIGKILL 0.5 s later.
- `segmentation-models-pytorch==0.6.0` stays pinned (`pyproject.toml:18`): `models.common.split_at` mirrors that release's decoders.
