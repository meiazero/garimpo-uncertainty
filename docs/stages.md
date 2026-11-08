# Stages

Read when a stage fails, skips or refuses.

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
| `generate_intervals.py` | GPU | checkpoint, `calibration/quantiles.npz`, directions, `scores/scores.npz`, test slots; one pass, one encoder per batch | `intervals/intervals.npz` |
| `compute_metrics.py` | CPU | `intervals/intervals.npz`, `predictions/`, `masks.npy`, `calibration/quantiles.npz`, `scores/scores.npz` | `metrics/cell_metrics.npz`, `metrics/results.csv`, `metrics/width_comparisons.csv`, `metrics/coverage_comparisons.csv`, `metrics/q1_spearman.csv`, `metrics/coverage_by_stratum.csv`, `metrics/score_relation.csv` |
| `merge_results.py` | any | small tables of each run | `outputs/results.csv`, `outputs/width_comparisons.csv`, `outputs/coverage_comparisons.csv`, `outputs/q1_spearman.csv`, `outputs/coverage_by_stratum.csv`, `outputs/score_relation.csv`, `outputs/layer_diagnostics.csv`, `outputs/monotonicity_layer.csv`, `outputs/monotonicity_endpoint.csv`, `outputs/chosen_layers.csv` |
| `pipeline_check.py` | any | status files and tables | prints the debugging report |

Paths without a prefix are inside the run directory. Every stage skips its work when its output
exists from the same inputs, stops with status `refused` when it exists from other inputs, and
redoes it with `--force`; each writes `status/<stage>.json`, and a missing input is recorded there
as a failure. Only `train.py` ever updates weights. `best_model.pt` stores the model configuration,
variant, starting weights, seed, normalization, fraction, subset seed, the run's tile ids, split hash
and the commits of root and submodule.
Later stages take the normalization from it and need no weights file. They refuse a checkpoint from
another split, and `train.py` refuses to resume `last.pt` under other arguments, weights included.

`validate_data.py` separates what can fail from what cannot, on the tiles of the run's selection only.
`problems` fail the stage. A missing artisanal tile, a selected tile never downloaded
(`not_downloaded`) and group leakage are always problems. An industrial tile with `missing_item` or
`no_mask_polygon` is listed in `tiles_not_ok` and the run proceeds without it; any other industrial
status, including a window off the scene, is a problem. `diagnostics` have no threshold and cannot fail the
stage: cloud and nodata fractions recomputed from the arrays, counts of 100% nodata and 100% cloud
cells, raster versus vector area, box excess, NDVI contrast. The `tile_offset` problem discounts the
polygon's box excess over the window; the centered window then bounds it at half a pixel, so it
trips only on a defect in the sidecar or in `tile_origin`.
