# Ways the pipeline can fail

Read to see what guards a given failure, or before adding a stage.

Written before the pipeline, and the origin of the test.

| # | Failure | Caught by |
|---|---|---|
| 1 | A Sentinel-2 scene is missing and the tile disappears silently | `validate_data.py` (artisanal: `missing_scenes`; industrial with `missing_item` or `no_mask_polygon`: `tiles_not_ok`, excluded from the counts), E2E |
| 2 | Only part of a group's selected tiles is downloaded, so a group's coverage is computed on a biased subset | `validate_data.py` (`partial_group`), E2E |
| 3 | A group spans two split roles (leakage) | `validate_data.py` (`group_leakage`), E2E |
| 4 | An interrupted download leaves half-written slots that a resume trusts | slots flushed before the sidecar says `ok`; a forced re-cut removes the sidecar first, E2E |
| 5 | The SAS token expires during a long download | token refresh in `download.py` (real downloads only) |
| 6 | Arithmetic on `uint16` DN overflows or goes negative | cast to float before the offset in `CellDataset` |
| 7 | Processing baseline 04.00 adds a radiometric offset, so scenes across 2022 differ by construction | `boa_offset` in the sidecar of each tile, applied by `CellDataset`, E2E |
| 8 | The mask is rasterized on a grid other than the image's | Not checked automatically: raster versus vector area and NDVI contrast are diagnostics, with no threshold, to read in the report |
| 9 | Cell size is not what the area metric assumes | `validate_data.py` (`store_layout`, `pixel_size`) |
| 10 | `g(f(x))` differs from `model(x)`: wrong layer, skip connection lost, batch norm in train mode | E2E |
| 11 | A library upgrade renames the U-Net decoder modules | `models.common.feature_layers` raises, E2E |
| 12 | Empty cells have zero Jacobians and normalization gives NaN | `directions._scale` raises; mean term in the PCA direction, E2E |
| 13 | The SVM normal points to less area | orientation against the mean Jacobian, E2E |
| 14 | SVM or PCA see calibration or test cells | training-role pass only, `n_train_cells` checked in E2E |
| 15 | Non-monotonic response makes endpoint intervals non-nested | Algorithm 2 sweep over the search range, `monotonicity_endpoint.csv`; sampled cells and grid points only, so a turn between two points passes |
| 16 | The search stops at `beta_range` and a width comparison is meaningless | `saturated` flag reported per row, E2E forces it |
| 17 | A group with many cells dominates the quantile | group-weighted quantile checked by brute force in E2E |
| 18 | Own-region calibration uses the held-out group | leave-one-group-out quantile checked by brute force in E2E |
| 19 | Relative or local score divides by zero on empty cells | required `--score-floor-ha` |
| 20 | Bootstrap resamples cells instead of groups, or method pairs use different resamples | shared group index matrix in `evaluation.summarize`; the E2E checks the paired point differences, not the resamples |
| 21 | A method x calibration row is missing from the results | exact row set checked in E2E |
| 22 | A stage changes the environment under a run in progress | `uv run --no-sync` in every script |
| 23 | The Colab session ends and its disk takes the data or the runs with it | store kept in Drive as `data.tar.zst` and `runs/` in Drive, `runs/` copied every 30 minutes during a run ([full-run.md](full-run.md)); not checked automatically |
| 24 | A metric change forces a retrain | stages read persisted files; E2E hashes everything around a metrics rerun |
| 25 | A changed argument silently reuses an old output | stage fingerprint refuses and fails the pipeline check, E2E; a code change is redone by name with `FORCE_FROM` |
| 26 | A later stage rebuilds the model on another split | split hash in the checkpoint |
| 27 | An import, syntax or `argparse` error, a bad `FORCE_FROM` or a failed `merge_results` leaves every status `ok` and the check prints PASS | `--job-exit-code` from the EXIT trap; E2E covers the script, not the trap |
| 28 | An outage removes hundreds of industrial training tiles | any other industrial status is `industrial_download_errors`, E2E |
| 29 | An OOM after training forces a lower batch size and a retrain | `EVAL_BATCH_SIZE` apart from `BATCH_SIZE` |
| 30 | A mask filling the cell has no boundary, so its boundary F1 leaves the means | F1 is 1 for identical masks and 0 otherwise, `n_cells_boundary_by_rule` in `results.csv`, E2E |
| 31 | Band columns of the 13-band `conv1` land out of order | literal column positions checked in E2E |
| 32 | A weights file with missing or extra keys loads partially | strict load after dropping `fc.*`, E2E |
| 33 | Runs merge under one model and fraction | `variant` and `train_fraction` in the metadata and the tables; `merge_results.py` refuses a repeated pair, E2E |
| 34 | Training tries to fetch weights over the network | Nothing in training can: `models.common.build` reads the two weight files from the data root only |
| 35 | Pretrained weights resume or skip under another identity | `encoder_weights` and the file hash in the fingerprint and `last.pt`, E2E |
| 36 | A larger fraction's download refuses every finished smaller run, or moves its slots | hash of the run's index set and sidecars in the fingerprint; slots fixed by `cell_index`; E2E checks earlier slots stay byte-identical |
| 37 | Tiles written in another order land in other slots | `cell_index` comes from the split alone, E2E writes in a shuffled order and compares bytes |
| 38 | A run trains or evaluates on tiles outside its selection, or a fraction changes under one `RUN_DIR` | `tile_ids` in `run_meta.json`, `train_fraction` and `subset_seed` in the fingerprint and `last.pt`, E2E |
| 39 | A fraction is run with tiles still missing, so it silently trains on fewer | `validate_data.py` (`not_downloaded`, `split_counts`), E2E; it runs before `train` in `slurm/stages.sh`, while `train.py` alone does not check this |
| 40 | A model that predicts nothing has logits far from zero, so at temperature 10,000 every Jacobian is exactly zero | `fit_svm` raises `mean Jacobian is orthogonal to w`, which stops `fit_directions`; seen in the E2E with 4 training cells and 30 updates. `n_zero_jacobians` in `directions/diagnostics.csv` counts the cells |
| 41 | The one-pass interval stage computes something other than each combination alone | E2E recomputes every COMPASS row with its own loader and encoder pass, exact equality, and counts one read per cell and one encoder pass per batch |
| 42 | An array is lost and comes back as zeros under `ok` sidecars | `create_store` refuses, E2E |
| 43 | The store changes under finished stages, which skip on sidecars alone | `slots_sha256` recomputed by `validate_data.py` and `pipeline_check.py`, E2E zeroes and deletes on the real stages |
| 44 | A changed split reads another tile's slots | `tile_index` in the sidecar and `N == n_tiles * cells_per_tile`, E2E reorders, drops and adds tiles |
| 45 | A lookup of an absent cell returns a neighbor | `data.positions_of` raises naming the `cell_index`, E2E on calibrate, generate_intervals and compute_metrics |
| 46 | Repeated subsampling draws per cell, so a large group dominates as under no group weighting | one 30-cell group among 24 single-cell groups, E2E |
| 47 | Too few groups give an unbounded subsampled quantile that reads as the largest score | `subsampled_quantile` refuses at k ≤ 1/α - 1, E2E at k = 9 and 10 |
| 48 | The subsampled rule loses the finite-sample +1 or reads the global RNG | one cell per group must equal split conformal's ⌈(k+1)(1-α)⌉-th score; the seed alone fixes the draws, E2E |
| 49 | An interval built at one alpha is scored or reported at another | every level's metrics recomputed per cell with its own alpha; quantiles must grow as alpha shrinks, E2E |
| 50 | Comparisons pair rows of different levels | `alpha` in every comparison row, pair counts per level, E2E |
| 51 | Cells without mining hide a shortfall in the cells with it | `coverage_by_stratum.csv`, by reference area with cells without mining apart and by predicted mining, recomputed per cell in E2E |
| 52 | Zero or saturated scores enter the log-log fit | both left out, `n_saturated` reported, fit recomputed in E2E |
| 53 | The layer is chosen on calibration or test cells | `choose_layer.py` reads the validation rows only, E2E with test rows that would change the choice |
| 54 | A layer whose sweep is not ascending is calibrated, or one direction passes for both | a layer qualifies only when the `pca` and `svm` rows both reach `MIN_FRACTION_ASCENDING`; one missing refuses, E2E |
| 55 | No layer qualifies and one is taken anyway, or an earlier choice survives a refusal | `choose_layer` fails naming the best layer and its fraction, so the job stops; it deletes any earlier `chosen_layer.json` first, E2E |
| 56 | A layer is chosen with no sweep behind it, or a new threshold keeps the old choice | layers must match between `directions/diagnostics.csv` and the sweep; the threshold is in the fingerprint, so a change refuses until `FORCE_FROM=choose_layer`, E2E |
| 57 | The weighted quantile leaves out the test point's mass at +inf, or scores a cell with a classifier fitted on it | `group_weighted_quantiles` adds the test cell's own ratio at +inf (Tibshirani et al. 2019, Corollary 1, groups as units); ratios cross-fitted by group; quantile recomputed per test cell in E2E |
| 58 | A quantile censored at `beta_range`, or inf, still yields a narrow COMPASS interval | `widen_saturated` turns it into `[0, cell_ha]`, E2E on the saturated run |
| 59 | Coverage under the shift is never reported where the shift is | `coverage_by_stratum.csv` has a `region` partition, recomputed per cell in E2E; `infinite_quantile_fraction` and the weighted group ESS are reported |
| 60 | One pooled quantile undercovers cells predicted with mining, a rare and noisy stratum | `*_mondrian` rows calibrate `vhat = 0` and `vhat > 0` apart; a stratum with no other score gives inf; brute-forced per cell in E2E |
| 61 | The asymmetric rows split alpha in halves, which repeated subsampling refuses below 2/alpha groups, or take their side ratio from the calibration cells | one joint score `max(lower, upper / ratio)` calibrated at 1 - alpha; the ratio comes from validation cells; brute-forced in E2E |
| 62 | An asymmetric row's upper side is censored below `beta_range` and still yields a narrow interval | `censor_at = min(beta_range, beta_range / ratio)` per row, persisted and applied by `widen_saturated`, E2E on the saturated run |
