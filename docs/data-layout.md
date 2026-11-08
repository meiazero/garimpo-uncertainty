# Data layout

Read when inspecting the store or a run's artifacts by hand.

One authoritative store, arrays indexed by integer, nothing stored twice. Under `$GARIMPO_DATA_DIR`:

| Path | Shape, type | Holds | Apparent size |
|---|---|---|---|
| `images.npy` | `(18864, 6, 512, 512)`, `uint16` | Raw DN of B02, B03, B04, B08, B11, B12 | 59.3 GB |
| `masks.npy` | `(18864, 512, 512)`, `uint8` | Reference mask | 4.9 GB |
| `scl.npy` | `(18864, 512, 512)`, `uint8` | Scene classification; only the validation reads it | 4.9 GB |
| `stats.npy` | `(18864,)`, 2 `float64` fields | `mask_ha`, the writer's claim the validation checks against `masks.npy`, and `vector_ha`, which needs the polygon | 0.3 MB |
| `tiles/<tile_id>.json` | one per tile | `status`, `item`, `detail`, `epsg`, `transform`, `boa_offset`, `tile_offset_m`, `tile_box_excess_m`, `resampling_20m`, `scene_cloud`, `tile_index`, `cells_per_tile`, `slots_sha256` | small |

`cell_index = tile_index * 16 + position`. `tile_index` is the tile's row among the non-dropped tiles
of `outputs/split.csv`, and `position` runs row by row over the 4 x 4 cells. So a fraction is an index
set, a larger fraction moves nothing, and a run's identity is its index set plus the sidecars that
vouch for those slots. `data.tile_records` and `data.cell_table` build the tile and cell tables in
memory from the split and the sidecars; neither is persisted. Each `ok` sidecar records its
`tile_index`, `cells_per_tile` and `slots_sha256`. A split that reorders rows, or moves a tile into or
out of `dropped`, refuses when `tile_index` disagrees or the array length differs from
`n_tiles * cells_per_tile`, in the stages and in the download; restore the split, or empty the store
and `tiles/` and download again.

Files are uncompressed by construction. A slot is `(6 * 2 + 1 + 1) * 262,144 = 3,670,016` bytes, so a
tile is 58.7 MB. The download writes a tile's 16 slots, flushes, and only then writes its sidecar with
`ok`, so a crash leaves no `ok`; a forced re-cut removes the sidecar before overwriting the slots.

`slots_sha256` hashes a tile's images, masks and SCL when written. `validate_data.py` recomputes it for
the selection on every call, skip included, and on any difference validates again and fails with
`slot_hash`. `pipeline_check.py` also recomputes it, last, and fails on `slot_hash`, `store_missing`
or `sidecars_missing`. Each therefore reads every slot of the selection, up to 69.2 GB per call on the
full run, in time not measured on NFS. Other stages' fingerprints read sidecars only: they still skip
on a damaged store, and these two are what fails. Hashes cover a selection's tiles only, so tiles
arriving elsewhere leave finished runs valid. If an array is lost while an `ok` sidecar exists, the
download refuses rather than recreate it as zeros.

The arrays are created whole, for all 18,864 cells, before the first tile. `np.lib.format.open_memmap`
extends each file by seeking past its end and writing one byte (`numpy/_core/memmap.py`), so unwritten
slots stay holes where the filesystem supports them, and only written slots take disk. Where it does
not, the full 69.2 GB is allocated at creation, even for the pilot. This was not measured on the
cluster's NFS `/home`: after the first `download`, compare `du -h $GARIMPO_DATA_DIR/images.npy` with
`ls -lh`. A copy that does not preserve holes also takes the full size, so use `rsync --sparse`. A
full disk under a write to a hole can kill the process with SIGBUS rather than raise; no `ok` is then
written, so rerun after freeing space. Untested. In all, images 59.3 GB plus masks and SCL 9.9 GB.

A cell's georeferencing comes from its tile's sidecar: with `row, col = divmod(position, 4)`, the
cell's transform is `Affine(*transform) @ Affine.translation(col * 512, row * 512)`, in `epsg`.

Artifacts inside a run align by position with its cell table or carry `cell_index`, always ascending:

| Artifact | Arrays |
|---|---|
| `predictions/cells.npz` | `cell_index`, `v`, `vhat`, `p_sum`, `pixel_var`, `cell_ha`, one row per cell of the run's table |
| `predictions/{validation,calibration,test}.npy` | `uint8` probabilities, `[N, 512, 512]`, by role |
| `scores/scores.npz` | `cell_index` of validation, calibration and test cells; `score[M, N]`, `saturated[M, N]`, the one-sided `lower`, `lower_saturated`, `upper`, `upper_saturated` (Algorithm 6) and `beta_range[M]`, `M` indexing `intervals.COMPASS_METHODS`; `jacobians`, `feature_sums` |
| `calibration/quantiles.npz` | `cell_index`, and `quantile`, `lower_term`, `upper_term`, `scale` as `[A, K, N]`, `A` indexing `intervals.ALPHAS`, `K` `intervals.COMBINATIONS`, `N` the test cells; `alpha`; per `[A, K]`, score counts, saturation and `infinite_quantile_fraction`; `density_ratio_calibration`, `density_ratio_test` (cross-fitted odds); per `[A, K]`, `side_ratio` of the `a_` rows and `censor_at`, the quantile from which a COMPASS row is the whole cell |
| `intervals/intervals.npz` | `cell_index`, and `lower`, `upper`, `quantile` as `[A, K, N]`, plus `alpha` and `layer` |
| `metrics/cell_metrics.npz` | `cell_index`, `dice`, `boundary_f1`, `boundary_by_rule`, one row per test cell |

CSV is left to human tables: `history.csv`, `directions/diagnostics.csv`, the two monotonicity tables,
and the metrics tables. No stage keeps a derived copy of the store, and
nothing is memoized. `history.csv` records `train_data_wait_seconds` and `seconds` per epoch, and
`infer.py` records `data_wait_seconds` per role in its status; `pipeline_check.py` prints both.

Resource measurements stay raw, with no summary table, so how to present them can be decided later.
A GPU peak counts the bytes of tensors on the current device, not the memory reserved by torch or
what `nvidia-smi` shows. Each part's peak starts from what is already allocated, so
`validation_peak_gpu_bytes` includes the weights and the last step's gradients.

| Where | Field | Meaning |
|---|---|---|
| `history.csv`, per epoch | `train_seconds`, `validation_seconds` | Unrounded time of each part of the epoch |
| `history.csv`, per epoch | `updates` | Optimizer steps the epoch made, which is `STEPS_PER_EPOCH` |
| `history.csv`, per epoch | `train_peak_gpu_bytes`, `validation_peak_gpu_bytes` | GPU peak of each part, empty if no GPU is in use |
| `status/infer.json`, `shapes` | `seconds_per_role` | Unrounded time per role, including writing its probabilities |
| `status/infer.json`, `shapes` | `peak_gpu_bytes_per_role` | GPU peak per role, `null` if no GPU is in use |
| `status/<stage>.json` | `seconds`, `peak_rss_gb`, `peak_gpu_gb`, `host`, `slurm_job_id` | The stage's last process, so a resumed `train` reports only its last stretch; `history.csv` keeps every epoch. `peak_rss_gb` is the largest single process, not a sum |

Boundary F1 has a rule for a mask with no boundary, such as one filling the cell: with no boundary in
either mask, F1 is 1 for identical masks and 0 otherwise; with a boundary in only one, F1 is 0.
`n_cells_boundary_by_rule` in `results.csv` counts these cells. Cells with no reference mining stay
NaN and out of the means.

Predictions are uint8 probabilities at 262,144 bytes per cell: 2,816 validation, calibration and
test cells make about 0.74 GB per model. Training cells keep their row in `predictions/cells.npz`.
