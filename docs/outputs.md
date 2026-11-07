# Numbers, scripts and outputs

Read to trace a number in the dissertation back to its script.

Run from the repository root. The split scripts read `data/zenodo`: after
`download_data.py metadata`, link it with `mkdir -p data && ln -s "$GARIMPO_DATA_DIR/zenodo" data/zenodo`.

```bash
scripts/split/get_biomes.sh                 # IBGE biomes into data/biomes, sha256 checked
uv run python -I scripts/split/frame.py
uv run python -I scripts/split/make_split.py   # writes outputs/split.csv; a changed tile order or set then refuses the store
```

| Number | Script | Output |
|---|---|---|
| Tiles by scale and region, chip areas, image years | `scripts/split/frame.py` | `outputs/frame.json`, `outputs/tiles_artisanal.csv` |
| Geographic partition and training chips | `scripts/split/make_split.py` | `outputs/split.csv`, `outputs/split_summary.json` |
| Tiles and cells per training fraction | `scripts/download_data.py lists` | `outputs/tiles/tiles_{1,5,10,100}pct.json` |
| Data checks, scenes missing, clouds and nodata per role, per fraction | `scripts/validate_data.py` | `outputs/data_validation-fraction-<F>-seed-<S>.json` |
| PICP, MPIW, interval score, MAE, bias, relative error, Dice, boundary F1, with 95% group bootstrap, one row per model x fraction x method x calibration | `scripts/compute_metrics.py`, `merge_results.py` | `outputs/results.csv` |
| Paired width differences between methods | `scripts/compute_metrics.py`, `merge_results.py` | `outputs/width_comparisons.csv` |
| Paired coverage differences between methods and calibrations (Q3) | `scripts/compute_metrics.py`, `merge_results.py` | `outputs/coverage_comparisons.csv` |
| Spearman of area error with Dice and boundary F1, per model (Q1) | `scripts/compute_metrics.py`, `merge_results.py` | `outputs/q1_spearman.csv` |
| PCA explained variance and `\|cos(v1, w)\|` per layer | `scripts/fit_directions.py`, `merge_results.py` | `outputs/layer_diagnostics.csv` |
| Monotonic fraction per layer and direction, validation role, for choosing the layer | `scripts/verify_monotonicity.py`, `merge_results.py` | `outputs/monotonicity_layer.csv` |
| Monotonic fraction of the chosen layer, calibration and test roles | `scripts/verify_monotonicity.py`, `merge_results.py` | `outputs/monotonicity_endpoint.csv` |

Groups join tiles closer than 50 km, measured edge to edge, in EPSG:6933 (12 Amazon groups).
