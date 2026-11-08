# Stages

Read when a stage fails, skips or refuses.

## Stage table

Paths without a prefix are inside the run directory.

| Stage | Node | Input | Output |
|---|---|---|---|
| `download_data.py metadata` | any | Zenodo 14195737 | `$DATA/zenodo/mining_area_data.gpkg` |
| `download_data.py lists` | any, only to regenerate | `outputs/split.csv`, subset seed | `outputs/tiles/tiles_{1,5,10,100}pct.json` |
| `download_data.py scenes` | any | `outputs/tiles/tiles_<P>pct.json`, `outputs/split.csv`, gpkg, Planetary Computer | `$DATA/{images,masks,scl,stats}.npy`, `$DATA/tiles/*.json` |
| `download_data.py weights` | any | Hugging Face | `$DATA/weights/resnet18_sentinel2_all_moco-59bfdff9.pth`, `$DATA/weights/resnet18_imagenet-b1d1d36f.safetensors` |
| `validate_data.py` | CPU | store, sidecars, split, fraction | `outputs/data_validation-fraction-<F>-seed-<S>.json` |
| `train.py` | GPU | the run's slots and sidecars, split, starting weights of the variant | `best_model.pt`, `last.pt`, `run_meta.json`, `history.csv` |
| `infer.py` | GPU | checkpoint, store | `predictions/cells.npz`, `predictions/{validation,calibration,test}.npy` |
| `fit_directions.py` | GPU | checkpoint, training slots | `directions/<layer>.npz`, `directions/diagnostics.csv` |
| `verify_monotonicity.py --use layer` | GPU | checkpoint, directions, validation slots | `monotonicity_layer.csv` |
| `choose_layer.py` | CPU | `directions/diagnostics.csv`, validation rows of `monotonicity_layer.csv` | `chosen_layer.json` (the layer, the threshold, each candidate's explained variance and fractions) |
| `verify_monotonicity.py --use endpoint` | GPU | checkpoint, direction of the layer, calibration and test slots | `monotonicity_endpoint.csv` |
| `compass_scores.py` | GPU | checkpoint, direction of the layer, validation, calibration and test slots | `scores/scores.npz` |
| `calibrate.py` | CPU | `predictions/cells.npz`, `scores/scores.npz` | `calibration/quantiles.npz` (`quantile`, the `beta_hat` of each row and cell) |
| `generate_intervals.py` | GPU | checkpoint, `calibration/quantiles.npz`, directions, `scores/scores.npz`, test slots; one pass, one encoder pass per batch | `intervals/intervals.npz` |
| `compute_metrics.py` | CPU | `intervals/intervals.npz`, `predictions/`, `masks.npy`, `calibration/quantiles.npz`, `scores/scores.npz` | `metrics/cell_metrics.npz`, `metrics/results.csv`, `metrics/width_comparisons.csv`, `metrics/coverage_comparisons.csv`, `metrics/q1_spearman.csv`, `metrics/coverage_by_stratum.csv`, `metrics/score_relation.csv` |
| `merge_results.py` | any | small tables of each run | `outputs/results.csv`, `outputs/width_comparisons.csv`, `outputs/coverage_comparisons.csv`, `outputs/q1_spearman.csv`, `outputs/coverage_by_stratum.csv`, `outputs/score_relation.csv`, `outputs/layer_diagnostics.csv`, `outputs/monotonicity_layer.csv`, `outputs/monotonicity_endpoint.csv`, `outputs/chosen_layers.csv` |
| `pipeline_check.py` | any | status files and tables | prints the debugging report |

`slurm/stages.sh` runs the stages from `train` to `compute_metrics` in its `STAGE_ORDER`: `train`,
`infer`, `fit_directions`, `layer_monotonicity`, `choose_layer`, `endpoint_monotonicity`,
`compass_scores`, `calibrate`, `generate_intervals`, `compute_metrics`. These are also the names of
their status files.

## Skip, refuse, force

Every stage from `validate_data.py` on, and `download_data.py scenes`, records its status in a
status file:

| Stage | Status file |
|---|---|
| `download_data.py scenes` | `$DATA/status/download_scenes.json` |
| `validate_data.py` | `$DATA/status/fraction-<F>-seed-<S>/validate_data.json` |
| every stage in `STAGE_ORDER` | `status/<stage>.json` in the run directory |

A stage skips its work when its outputs exist from the same fingerprint, stops with status
`refused` when they exist from another one, and redoes the work with `--force`. The fingerprint
holds the arguments that change a result (not `--force`, `--device`, `--num-workers`,
`--batch-size` after training, `--workers`, `--lightgbm-jobs` or `--report`) and each input file's
path, size and modification time. A refusal keeps the last completed fingerprint, so it never turns
into a rerun on the next attempt.

Two stages differ. `download_data.py scenes` lists no outputs, so every run checks the sidecars and
fetches what is still pending. `validate_data.py` recomputes the slot hashes on every call and
validates again, whatever its status, when they disagree.

A missing input fails the stage before any work, keeping the last completed fingerprint, so once
the input is back a finished stage skips again.

## Statuses

| Status | Written when | Pipeline check |
|---|---|---|
| `running` | the stage starts; it stays if the process dies inside the stage | fails: "no error recorded: the job died inside this stage" |
| `ok` | the work finished, or the stage skipped on the same fingerprint | passes |
| `failed` | the work raised, or an input is missing; `error` and `traceback` record why | fails, printing both |
| `refused` | outputs exist from another fingerprint | fails, printing the `FORCE_FROM` or `FORCE_VALIDATE` that redoes it |
| `interrupted` | Ctrl+C reached the stage's work (`KeyboardInterrupt`); `error` is "Ctrl+C; rerun to resume" and `seconds` the time until then | fails, like `failed` |
| `not_run` | no status file exists; printed by the check, never written | fails |

`pipeline_check.py` fails on any status other than `ok` in `validate_data` or a stage of
`STAGE_ORDER`, and prints the first of them that is `failed`, `refused`, `running` or `interrupted`.
`download_scenes` is left out, since the download is shared by every fraction and
`validate_data` decides whether this run's tiles are there.

An `interrupted` status keeps no completed fingerprint, so a rerun runs the stage's work again
instead of skipping it. `train.py` continues from `last.pt`, written after every epoch. The
download starts no queued tile after Ctrl+C, names the tiles in progress and logs each as it
writes its sidecar, and on rerun fetches only the tiles without an `ok` sidecar. A second Ctrl+C
quits without waiting; the tiles still in progress then have no `ok` sidecar and are fetched again.
Other stages start their work over.

## Run identity

Only `train.py` ever updates weights. `best_model.pt` stores the model configuration, variant,
starting weights, seed, normalization, fraction, subset seed, the run's tile ids, split hash and the
commits of root and submodule. Later stages take the normalization from it and need no weights file.
They refuse a checkpoint from another split, and `train.py` refuses to resume `last.pt` under other
arguments, weights included.

Code is not part of any fingerprint. A run calibrated before rows were added to
`intervals.COMBINATIONS` still skips `calibrate`, and the pipeline check fails on the absent rows:
resubmit with `FORCE_FROM=calibrate`.

Only a fresh start of `train.py` reads starting weights. Without the file it fails, naming the
`scripts/download_data.py weights` command of step 2 in [full-run.md](full-run.md). A finished or
resumed run needs no weights file and takes the hash it started from out of `run_meta.json`; with
both absent, `train.py` fails naming both.

The store is known by the hash of the run's index set and sidecars, so restoring it with new
modification times refuses nothing, but the data root must stay at the same path. Restoring
`runs/` without `weights/` reaches the later stages only if the paths and modification times of
their other inputs are preserved too: a `best_model.pt` with a new modification time makes the
stages after training refuse. Both weights are plain files under `$GARIMPO_DATA_DIR/weights/`, so
training never reaches a network or a library cache.

## Data validation

`validate_data.py` separates what can fail from what cannot, on the tiles of the run's selection
only. `problems` fail the stage:

- a missing artisanal tile, a selected tile never downloaded (`not_downloaded`) and group leakage
  are always problems;
- an industrial tile with `missing_item` or `no_mask_polygon` is listed in `tiles_not_ok` and the
  run proceeds without it; any other industrial status, including a window off the scene, is a
  problem.

`diagnostics` have no threshold and cannot fail the stage: cloud and nodata fractions recomputed
from the arrays, counts of 100% nodata and 100% cloud cells, raster versus vector area, box excess
and NDVI contrast.

The `tile_offset` problem discounts the polygon's box excess over the window. The centered window
then bounds it at half a pixel, so it trips only on a defect in the sidecar or in `tile_origin`.
