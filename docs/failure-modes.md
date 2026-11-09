# Ways the pipeline can fail

Read to see what guards a given failure, or before adding a stage.

The list was written before the pipeline, and the E2E tests were written from it. "E2E" in the
last column names a guard that a test in `tests/e2e/` exercises.

| # | Failure | Caught by |
|---|---|---|
| 1 | A Sentinel-2 scene is missing and the tile disappears silently | `validate_data.py` (artisanal: `missing_scenes`; industrial with `missing_item` or `no_mask_polygon`: `tiles_not_ok`, excluded from the counts), E2E for `missing_scenes`, `missing_item` and `no_mask_polygon` |
| 2 | Only part of a group's selected tiles is downloaded, so a group's coverage is computed on a biased subset | `validate_data.py` (`partial_group`), E2E |
| 3 | A group spans two split roles (leakage) | `validate_data.py` (`group_leakage`), E2E |
| 4 | An interrupted download leaves half-written slots that a resume trusts | slots flushed before the sidecar says `ok` (not checked: only a host crash loses unflushed pages); a forced re-cut removes the sidecar first, E2E |
| 5 | The SAS token expires during a long download | token refresh in `download.py`, E2E with a fake clock and HTTP, as is the wait on HTTP 429; real downloads only otherwise |
| 6 | Arithmetic on `uint16` DN overflows or goes negative | cast to float before the offset in `CellDataset` |
| 7 | Processing baseline 04.00 adds a radiometric offset, so scenes across 2022 differ by construction | `boa_offset` in the sidecar of each tile, carried by `cell_table` to its own tile's cells and applied by `CellDataset`, E2E |
| 8 | The mask is rasterized on a grid other than the image's | Not checked automatically: raster versus vector area and NDVI contrast are diagnostics, with no threshold, to read in the report |
| 9 | Cell size is not what the area metric assumes | `validate_data.py` (`store_layout` for another cell size or `stats` length, `pixel_size` for another grid), E2E |
| 10 | `g(f(x))` differs from `model(x)`: wrong layer, skip connection lost, batch norm in train mode | E2E for the layer and the skip connection; eval mode is set by `checkpoints.load_model`, E2E on a trained checkpoint: no module left in train mode, and each cell's output is the same alone or in a batch |
| 11 | A library upgrade renames the U-Net decoder modules | `models.common.feature_layers` raises, E2E with the decoder modules of both models renamed |
| 12 | Empty cells have zero Jacobians and normalization gives NaN | `directions._scale` raises; mean term in the PCA direction; E2E on the fitted directions of a run, with a zero Jacobian and with no mean |
| 13 | The SVM normal points to less area | orientation against the mean Jacobian, a test of `fit_svm` (not of the stage) on features that encode mining against the area's sign; on the synthetic runs the raw normal never needs turning |
| 14 | SVM or PCA see calibration or test cells | training-role pass only, `n_train_cells` checked in E2E |
| 15 | Non-monotonic response makes endpoint intervals non-nested | Algorithm 2 sweep over the search range, `monotonicity_endpoint.csv`; sampled cells and grid points only, so a turn between two points passes |
| 16 | The search stops at `beta_range` and a width comparison is meaningless | `saturated` flag reported per row, E2E forces it |
| 17 | A group with many cells dominates the quantile | group-weighted quantile checked by brute force in E2E |
| 18 | Own-region calibration uses the held-out group | leave-one-group-out quantile checked by brute force in E2E |
| 19 | Relative or local score divides by zero on empty cells | required `--score-floor-ha`, refused unless positive and finite (0, negative, NaN, inf) by `calibrate.py` and, before training, by `slurm/full.sbatch`, E2E; quantiles brute-forced with the floor, the relative one on the shared runs, the local one with zero pixel variance on cells predicted empty |
| 20 | Bootstrap resamples cells instead of groups, or method pairs use different resamples | shared group index matrix in `evaluation.summarize`, E2E on the run's test groups: coverage constant within groups and a constant paired width difference must give zero-width intervals |
| 21 | A method x calibration row is missing from the results | exact row set checked in E2E; `pipeline_check.py` fails on an absent row or NaN in PICP or MPIW, E2E |
| 22 | A stage changes the environment under a run in progress | `uv run --no-sync` in every script |
| 23 | The Colab session ends and its disk takes the data or the runs with it | store kept in Drive as `data.tar.zst` and `runs/` in Drive, `runs/` copied every 30 minutes during a run ([full-run.md](full-run.md)); not checked automatically |
| 24 | A metric change forces a retrain | stages read persisted files; E2E hashes everything around a metrics rerun |
| 25 | A changed argument silently reuses an old output | stage fingerprint refuses and fails the pipeline check, E2E; a code change is redone by name with `FORCE_FROM` |
| 26 | A later stage rebuilds the model on another split | split hash in the checkpoint, E2E with a data root missing a test tile; `read_predictions` refuses a `cells.npz` out of the cell table's order, E2E |
| 27 | An import, syntax or `argparse` error, a bad `FORCE_FROM` or a failed `merge_results` leaves every status `ok` and the check prints PASS | `--job-exit-code` from the EXIT trap, E2E through `slurm/full.sbatch` with an unknown `FORCE_FROM` |
| 28 | An outage removes hundreds of industrial training tiles | any other industrial status is `industrial_download_errors`, E2E |
| 29 | An OOM after training forces a lower batch size and a retrain | `EVAL_BATCH_SIZE` apart from `BATCH_SIZE`; E2E reads each stage's `--batch-size` in the job's `bash -x` trace |
| 30 | A mask filling the cell has no boundary, so its boundary F1 leaves the means | F1 is 1 for identical masks and 0 otherwise, `n_cells_boundary_by_rule` in `results.csv`, E2E against a count from the masks and probabilities, and a test of `segmentation_metrics` where a mask filling the cell has no boundary while a square inside it has one |
| 31 | Band columns of the 13-band `conv1` land out of order | literal column positions checked in E2E |
| 32 | A weights file with missing or extra keys loads partially | strict load after dropping `fc.*`, E2E |
| 33 | Runs merge under one model and fraction | `variant` and `train_fraction` in the metadata and the tables; `merge_results.py` refuses a repeated pair, E2E |
| 34 | Training tries to fetch weights over the network | Nothing in training can: `data.starting_weights_path` resolves both weight files under the data root, and smp is built with `encoder_weights=None` |
| 35 | Pretrained weights resume or skip under another identity | `encoder_weights` and the file hash in the fingerprint and `last.pt`, E2E |
| 36 | A larger fraction's download refuses every finished smaller run, or moves its slots | hash of the run's index set and sidecars in the fingerprint; slots fixed by `cell_index`; E2E checks earlier slots stay byte-identical |
| 37 | Tiles written in another order land in other slots | `cell_index` comes from the split alone, E2E writes in a shuffled order and compares bytes; the download's own `tile_index` is checked by the list tests of `test_download.py` |
| 38 | A run trains or evaluates on tiles outside its selection, or a fraction changes under one `RUN_DIR` | `tile_ids` in `run_meta.json`, `train_fraction` and `subset_seed` in the fingerprint and `last.pt`, E2E |
| 39 | A fraction is run with tiles still missing, so it silently trains on fewer | `validate_data.py` (`not_downloaded`, `split_counts`), E2E; it runs before `train` in `slurm/stages.sh`, while `train.py` alone does not check this |
| 40 | A model that predicts nothing has logits far from zero, so at temperature 10,000 every Jacobian is exactly zero | `fit_svm` raises `mean Jacobian is orthogonal to w`, which stops `fit_directions`, E2E at a temperature that saturates every sigmoid. `test_fractions.py` passes `--direction-temperature 1` since its 4-cell, 80-update run hit this on the CI runner. `n_zero_jacobians` in `directions/diagnostics.csv` counts the cells |
| 41 | The one-pass interval stage computes something other than each combination alone | E2E recomputes every COMPASS row with its own loader and encoder pass, exact equality, and counts one read per cell and one encoder pass per batch |
| 42 | An array is lost and comes back as zeros under `ok` sidecars | `create_store` refuses, E2E |
| 43 | The store changes under finished stages, which skip on sidecars alone | `slots_sha256` recomputed by `validate_data.py` and `pipeline_check.py`, E2E zeroes and deletes on the real stages |
| 44 | A changed split reads another tile's slots | `tile_index` in the sidecar and `N == n_tiles * cells_per_tile`, E2E reorders, drops and adds tiles |
| 45 | A lookup of an absent cell returns a neighbor | `data.positions_of` raises naming the `cell_index`, E2E on calibrate, generate_intervals and compute_metrics |
| 46 | Repeated subsampling draws per cell, so a large group dominates as under no group weighting | one 30-cell group among 24 single-cell groups, E2E |
| 47 | Too few groups give an unbounded subsampled quantile that reads as the largest score | `subsampled_quantile` refuses at k ≤ 1/α - 1, E2E at k = 9 and 10 |
| 48 | The subsampled rule loses the finite-sample +1 or reads the global RNG | one cell per group must equal split conformal's ⌈(k+1)(1-α)⌉-th score; the seed alone fixes the draws, E2E |
| 49 | An interval built at one alpha is scored or reported at another | every level's metrics recomputed per cell with its own alpha; quantiles must grow as alpha shrinks, E2E |
| 50 | Comparisons pair rows of different levels | `alpha` in every comparison row, pair counts per level, differences recomputed at every level, E2E |
| 51 | Cells without mining hide a shortfall in the cells with it | `coverage_by_stratum.csv`, by reference area with cells without mining apart and by predicted mining, recomputed per cell in E2E |
| 52 | Zero or saturated scores enter the log-log fit | both left out, `n_saturated` reported, fit recomputed in E2E |
| 53 | The layer is chosen on calibration or test cells | `choose_layer.py` reads the validation rows only, E2E with test rows that would change the choice |
| 54 | A layer whose sweep is not ascending is calibrated, or one direction passes for both | a layer qualifies only when the `pca` and `svm` rows both reach `MIN_FRACTION_ASCENDING`; one missing refuses, E2E |
| 55 | No layer qualifies and one is taken anyway, or an earlier choice survives a refusal | `choose_layer` fails naming the best layer and its fraction, so the job stops; it deletes any earlier `chosen_layer.json` first, E2E |
| 56 | A layer is chosen with no sweep behind it, or a new threshold keeps the old choice | layers must match between `directions/diagnostics.csv` and the sweep; the threshold is in the fingerprint, so a change refuses until `FORCE_FROM=choose_layer`, E2E |
| 57 | The weighted quantile leaves out the test point's mass at +inf, or scores a cell with a classifier fitted on it | `group_weighted_quantiles` adds the test cell's own ratio at +inf (Tibshirani et al. 2019, Corollary 1, groups as units); ratios cross-fitted by group, E2E where only a classifier that saw a group can tell its population, with the cells shuffled so a cell-level split cannot pass by keeping groups contiguous, on the real split's groups, and with a fold lacking a population; invalid weights refused, E2E; quantile recomputed per test cell in E2E |
| 58 | A quantile censored at `beta_range`, or inf, still yields a narrow COMPASS interval | `widen_saturated` turns it into `[0, cell_ha]`, E2E on the saturated run |
| 59 | Coverage under the shift is never reported where the shift is | `coverage_by_stratum.csv` has a `region` partition, recomputed per cell in E2E; `infinite_quantile_fraction` is reported, and the weighted group ESS in `status/calibrate.json`, recomputed in E2E |
| 60 | One pooled quantile undercovers cells predicted with mining, a rare and noisy stratum | `*_mondrian` rows calibrate `vhat = 0` and `vhat > 0` apart; a stratum with no other score gives inf; brute-forced per cell in E2E |
| 61 | The asymmetric rows split alpha in halves, which repeated subsampling refuses below 2/alpha groups, or take their side ratio from the calibration cells | one joint score `max(lower, upper / ratio)` calibrated at 1 - alpha; the ratio comes from validation cells; brute-forced in E2E |
| 62 | An asymmetric row's upper side is censored below `beta_range` and still yields a narrow interval | `censor_at = min(beta_range, beta_range / ratio)` per row, persisted and applied by `widen_saturated`, E2E with a side ratio of 8 and a quantile between `beta_range / 8` and `beta_range` |
| 63 | A stage fails and the job goes on, so later stages run on missing or stale outputs, such as an old `best_model.pt`, and the last command's exit code hides the failure | `set -euo pipefail` in `slurm/full.sbatch` and `slurm/pilot.sbatch`, E2E on both with a `train` that fails |
| 64 | A value left empty in the env file reaches a stage as an empty argument, after training | `:?` checks and the `LAYERS` check in `slurm/full.sbatch`, before any stage; E2E for every value of `full.env.example` |
| 65 | `slurm/stages.sh` passes a value to the wrong flag, or the chosen layer is not the one the calibrated stages receive | E2E reads the arguments of each stage in the job's `bash -x` trace |
| 66 | A session ends mid-training and the rerun starts over, or continues without the optimizer's moments, the generators, the best Dice or the history | `last.pt` after every epoch holds all of them, the CUDA generators included; E2E kills training after epoch 0, by SIGKILL and by Ctrl+C, and requires the rerun to equal an uninterrupted run exactly, on one CPU core only; E2E raises last.pt's best Dice and requires epoch 1 not to replace `best_model.pt`; the CUDA generators are not checked, CI has no GPU |
| 67 | A loader worker cannot pickle the dataset or open the store, takes a seed that is not the run's, or returns batches out of order | E2E trains twice and infers with 2 workers on CPU: one model per seed, band statistics and predictions equal to those without workers |
| 68 | `FORCE_FROM=train` left in a Colab cell restarts, after a lost session, a training that `last.pt` could resume, or a forced redo cut short resumes from the end of the old run | `train.py --force` refuses over an unfinished run's `last.pt` (no `completed_fingerprint` in its status) and marks a finished run's discarded before training, a mark the Colab rsync copies to Drive where a deletion would not reach, E2E for both, the second through a Drive round trip; not checked: the 30-minute background rsync copying the old `last.pt` in the second the mark replaces it, then the session ending before its next pass |
