# Numbers, scripts and outputs

Read to trace a reported number back to its script.

## Regenerate the split

Run from the repository root. The split scripts read `data/zenodo`, so after
`garimpo download_data metadata` link it with
`mkdir -p data && ln -s "$GARIMPO_DATA_DIR/zenodo" data/zenodo`.

```bash
scripts/split/get_biomes.sh                 # IBGE biomes into data/biomes, sha256 checked
uv run python -I scripts/split/frame.py
uv run python -I scripts/split/make_split.py   # writes outputs/split.csv; a changed tile order or set then refuses the store
```

Groups join tiles closer than 50 km, measured edge to edge, in EPSG:6933 (12 Amazon groups).

## Number to script

| Number | Script | Output |
|---|---|---|
| Tiles by scale and region, chip areas, image years | `scripts/split/frame.py` | `outputs/frame.json`, `outputs/tiles_artisanal.csv` |
| Geographic partition and training chips | `scripts/split/make_split.py` | `outputs/split.csv`, `outputs/split_summary.json` |
| Tiles and cells per training fraction | `garimpo download_data lists` | `outputs/tiles/tiles_{1,5,10,100}pct.json` |
| Data checks, scenes missing, clouds and nodata per role, per fraction | `garimpo validate_data` | `outputs/data_validation-fraction-<F>-seed-<S>.json` |
| PICP, MPIW, interval score, MAE, bias, relative error, Dice, boundary F1, with 95% group bootstrap, one row per model x fraction x alpha (0.05, 0.10, 0.15, as in Cheung et al. 2026, Table 1) x method x calibration | `garimpo compute_metrics`, `garimpo merge` | `outputs/results.csv` |
| PICP and MPIW of the rows that change with the layer, COMPASS-J and every weighted calibration, whose density ratio reads the layer's features, on every layer of `LAYERS`, the chosen one included, to compare widths across layers (Cheung et al. 2026, Table 4) | `garimpo compute_metrics` with `--layer-dir`, `garimpo merge` | `outputs/layer_results.csv` |
| Paired width differences between methods | `garimpo compute_metrics`, `garimpo merge` | `outputs/width_comparisons.csv` |
| Paired coverage differences between methods and calibrations (Q3) | `garimpo compute_metrics`, `garimpo merge` | `outputs/coverage_comparisons.csv` |
| Spearman of area error with Dice and boundary F1, per model (Q1) | `garimpo compute_metrics`, `garimpo merge` | `outputs/q1_spearman.csv` |
| PICP and MPIW per stratum, never recalibrated: by reference area, cells without mining then quintiles of those with it (adapted from Cheung et al. 2026, Figure 7, which does not define its bins); by predicted mining, `vhat = 0` or `> 0`, the stratum known when an interval is used; by held-out region, named in `label` | `garimpo compute_metrics`, `garimpo merge` | `outputs/coverage_by_stratum.csv` |
| Log-log slope of COMPASS scores on split-conformal scores, calibration cells, zero and saturated scores left out (Cheung et al. 2026, Figure 4) | `garimpo compute_metrics`, `garimpo merge` | `outputs/score_relation.csv` |
| PCA explained variance and `\|cos(v1, w)\|` per layer | `garimpo fit_directions`, `garimpo merge` | `outputs/layer_diagnostics.csv` |
| Monotonic fraction per layer and direction, validation role, for choosing the layer | `garimpo verify_monotonicity`, `garimpo merge` | `outputs/monotonicity_layer.csv` |
| Layer chosen per run and the threshold it met | `garimpo choose_layer`, `garimpo merge` | `outputs/chosen_layers.csv` |
| Monotonic fraction of the chosen layer, calibration and test roles | `garimpo verify_monotonicity`, `garimpo merge` | `outputs/monotonicity_endpoint.csv` |

## Metric rules

Boundary F1 has a rule for a mask with no boundary, such as one filling the cell. With no boundary
in either mask, F1 is 1 for identical masks and 0 otherwise; with a boundary in only one, F1 is 0.
`n_cells_boundary_by_rule` in `results.csv` counts these cells. Cells with no reference mining stay
NaN and out of the means.
