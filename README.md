# garimpo-uncertainty

Replication package of a master's dissertation: calibrated prediction intervals for artisanal mining
area from Sentinel-2, evaluated under geographic domain shift. `compass/` (submodule) holds the
adapted COMPASS method; `src/garimpo_uncertainty/` the satellite code; `scripts/` the stages;
`slurm/` the cluster jobs; `outputs/` the versioned tables, including `split.csv`, the split used,
and `tiles/`, the tiles each training fraction downloads.

No model result exists yet. The code has run only under pytest, on synthetic data; the pipeline has
never run on real scenes or on the cluster.

## Pilot: adjust the code on the cluster

The pilot is the 1% training fraction, run through the same scripts as the full run: 186 tiles and
2,976 cells, since validation, calibration and test stay whole. It downloads the versioned list
`outputs/tiles/tiles_1pct.json`. It exists to debug the
implementation, and its numbers say nothing about any method. Its values are in `slurm/pilot.sbatch`,
labeled pilot-only. Results go to `runs/pilot/`, never to `outputs/`.

On the login node:

```bash
git clone --recurse-submodules <ROOT_REPO_URL> garimpo-uncertainty
cd garimpo-uncertainty
uv sync
git config core.hooksPath .githooks
bash slurm/pilot.sbatch download
ENCODER_WEIGHTS=imagenet bash slurm/pilot.sbatch download           # data, then that variant's weights
ENCODER_WEIGHTS=sentinel2_moco bash slurm/pilot.sbatch download
```

Then through SLURM, on the login node too:

```bash
slurm/submit.sh slurm/pilot.sbatch                                  # U-Net from scratch
ENCODER_WEIGHTS=imagenet slurm/submit.sh slurm/pilot.sbatch
ENCODER_WEIGHTS=sentinel2_moco slurm/submit.sh slurm/pilot.sbatch
tail -f outputs/logs/garimpo-pilot-<JOBID>.out
```

Each command is its own job with its own 4 h limit, and writes to `runs/pilot/unet-<weights>/`. That
limit is a guess, with its arithmetic in `slurm/pilot.sbatch`; a job killed at the limit keeps its
finished stages.

The job ends with `scripts/pipeline_check.py`: status, seconds, peak memory and shapes per stage, slot hashes,
the failing stage with its error, scenes missing, cells per role, Dice, percent monotonic per
direction, `|cos(v1, w)|`, saturated searches, PICP and MPIW per method. It receives the job's exit
code, so an error outside every stage also reads FAIL.

After fixing the code, `git pull && uv sync` on the login node and resubmit. Finished stages skip
themselves. A stage whose inputs or arguments changed stops with status `refused`, which fails the
check. Commits are recorded in the metadata but stay out of the fingerprint, so a code fix alone
invalidates nothing: name the first stage it affects.

```bash
FORCE_FROM=calibrate slurm/submit.sh slurm/pilot.sbatch
```

`FORCE_FROM` takes a name from `STAGE_ORDER` in `slurm/stages.sh` and redoes that stage and those
after it. Earlier stages skip, and `train` runs again only with `FORCE_FROM=train`. The validation
sits outside that order: redo it with `FORCE_VALIDATE=1`. The same variables work with
`slurm/full.sbatch`.

## Full run, from zero

| Step | Where | Command |
|---|---|---|
| 1 | login | `git clone --recurse-submodules <ROOT_REPO_URL> garimpo-uncertainty && cd garimpo-uncertainty && uv sync` |
| 2 | login | `uv run --no-sync python scripts/download_data.py metadata` |
| 3 | login | `uv run --no-sync python scripts/download_data.py scenes --tiles outputs/tiles/tiles_100pct.json --resampling-20m <nearest\|bilinear>` (from the versioned list, so `lists` need not run first; resumable; contains every smaller fraction of that list's seed; sizes in the table below; another resampling refuses; `--force` redoes every selected tile, then rerun with `FORCE_VALIDATE=1 FORCE_FROM=train`, since a re-cut that changes a tile's slots changes the hash in its sidecar, so later stages refuse until redone; a re-cut with identical content changes nothing) |
| 4 | login | `uv run --no-sync python scripts/download_data.py weights --encoder-weights imagenet sentinel2_moco` |
| 5 | login | `cp slurm/full.env.example slurm/unet_scratch-f1.0.env`, then fill in every value (see decisions below) except `LAYER`, with `STOP_AFTER_DIAGNOSTICS=1` |
| 6 | `sbatch` | `slurm/submit.sh slurm/full.sbatch slurm/unet_scratch-f1.0.env` (stops after the layer diagnostics) |
| 7 | login | Read `runs/unet_scratch-f1.0/directions/diagnostics.csv` and `runs/unet_scratch-f1.0/monotonicity_layer.csv` (validation role only); set `LAYER`, clear `STOP_AFTER_DIAGNOSTICS` |
| 8 | `sbatch` | `slurm/submit.sh slurm/full.sbatch slurm/unet_scratch-f1.0.env` (finished stages skip, so nothing retrains) |
| 9 | both | Repeat steps 5 to 8 for each other run of the learning curve below |
| 10 | login | `uv run --no-sync python scripts/merge_results.py --runs runs/unet_*-f* runs/convnet-f1.0 --out outputs` |

Compute nodes have no internet, so `uv sync` and steps 2 to 4 only run on the login node; every job
uses `uv run --no-sync`. The data root is `$GARIMPO_DATA_DIR`, default
`~/Experiments/datasets/garimpo`.

### Training fractions

Only the `train` role is reduced, so every fraction is evaluated on the same cells. Tiles are drawn
whole, per scale, from one seeded shuffle, so under one `SUBSET_SEED` each subset contains the
smaller ones. `SUBSET_SEED` is apart from the training `SEED`.

| `TRAIN_FRACTION` | Artisanal training tiles | Industrial training tiles | Tiles, with the 176 of the other roles | Cells | GB written (58.7 MB per tile) |
|---|---|---|---|---|---|
| `0.01` (the pilot) | 1 | 9 | 186 | 2,976 | 10.9 |
| `0.05` | 3 | 47 | 226 | 3,616 | 13.3 |
| `0.10` | 6 | 95 | 277 | 4,432 | 16.3 |
| `1.0` | 56 | 947 | 1,179 | 18,864 | 69.2 |

Limits of the curve:

- There is one seed per fraction, so differences between fractions are descriptive only.
- At 1% a single artisanal tile is drawn, so that point depends heavily on which tile it is.
- Epochs are an explicit value of each run. Whether the fractions share a number of epochs or a
  number of updates is an open decision, and with fixed epochs a smaller fraction gets fewer updates.

A run reads only its own slots of the shared store, its tiles recorded as `tile_ids` in
`run_meta.json`, and each stage's fingerprint holds a hash of its index set and its tiles' sidecars.
So a larger fraction's download leaves finished smaller runs skipping and their slots untouched.
The written sizes hold where the filesystem keeps holes; see the data layout below.
`validate_data.py` runs per fraction and seed, with its status in
`$DATA/status/fraction-<F>-seed-<S>/`. Another `--train-fraction` or `--subset-seed` under the same
`RUN_DIR` refuses.

### Tile lists

`outputs/tiles/tiles_{1,5,10,100}pct.json` are versioned and are the source of every download, so a
fresh clone downloads without generating anything. Each is complete for its fraction: the reduced
training tiles plus all validation, calibration and test tiles, sorted by `tile_id`:

```json
{
  "train_fraction": 0.01,
  "subset_seed": 0,
  "split_sha256": "<SHA-256 of outputs/split.csv>",
  "tiles": [
    {
      "tile_id": 80,
      "role": "train"
    }
  ]
}
```

Downloads are incremental on the one store, so a larger list fetches only what is missing:

```bash
uv run --no-sync python scripts/download_data.py scenes --tiles outputs/tiles/tiles_1pct.json --resampling-20m bilinear
uv run --no-sync python scripts/download_data.py scenes --tiles outputs/tiles/tiles_5pct.json --resampling-20m bilinear
```

`scenes` refuses a list whose `split_sha256` differs from the current split, or that names a tile the
split does not keep. It downloads exactly what the list names, and training still selects by
`--train-fraction` and `--subset-seed`. `validate_data.py` fails with `not_downloaded` if those two
ask for a tile the download did not fetch, and the E2E asserts that each versioned list equals
`select_tiles` for its fraction and seed.

The versioned lists use subset seed 0, the value `slurm/pilot.sbatch` labels pilot-only. `lists`
regenerates all four, and is needed only after `outputs/split.csv` changes or to use another seed:

```bash
uv run --no-sync python scripts/download_data.py lists --subset-seed <SUBSET_SEED>
```

The E2E fails if regenerating with the recorded seed does not reproduce the versioned files byte for
byte, so commit the regenerated files with whatever changed them.

### Model variants

| Model in the results | `ARCH` | `ENCODER_WEIGHTS` | `RUN_DIR` | Input |
|---|---|---|---|---|
| `unet_scratch` | `unet` | `none` | `runs/unet_scratch-f<F>` | z-score of the run's training cells |
| `unet_imagenet` | `unet` | `imagenet` | `runs/unet_imagenet-f<F>` | z-score of the run's training cells |
| `unet_sentinel2` | `unet` | `sentinel2_moco` | `runs/unet_sentinel2-f<F>` | reflectance / 10000, stored as mean 0 and std 1 |
| `convnet` | `convnet` | empty | `runs/convnet-f1.0` | z-score of the run's training cells |

The learning curve is 3 U-Net variants x 4 fractions plus the ConvNet at 100%: 13 runs, one env file
each, and 13 x 17 = 221 method and calibration rows. The code does not refuse a ConvNet at a lower
fraction; it is simply not among these runs. Each run is one training, so differences between
variants have no replication behind them and the comparison is descriptive only.
`merge_results.py` identifies a run by `(model, train_fraction)` and refuses a repeated pair; each
run's paired comparisons stay inside that run, so no comparison crosses fractions.

| Weights | Source | License |
|---|---|---|
| `sentinel2_moco` | `torchgeo/resnet50_sentinel2_all_moco` on Hugging Face, revision `da4f3c9`, the file `ResNet50_Weights.SENTINEL2_ALL_MOCO` of torchgeo, trained on SSL4EO-S12; URL and hash from torchgeo's `models/resnet.py`; sha256 prefix `df8b932e` checked on download, full hash kept in the checkpoint | `cc-by-4.0` (Hugging Face API metadata) |
| `imagenet` | `smp-hub/resnet50.imagenet` on Hugging Face, from `encoders/resnet.py` of segmentation-models-pytorch 0.6.0, revision `00cb74e`, file `model.safetensors`, sha256 prefix `df1aad85` from the Hugging Face API, checked on download | `other` (Hugging Face API metadata): read the model card before citing it |

Limits of these weights:

- `sentinel2_moco` was trained on 13 TOA (L1C) bands, and these cells are L2A surface reflectance.
  The filters meet a different radiometry than the one they learned.
- Its original transform also resizes to 256 and crops to 224. Cells here stay at 512 pixels.
- Only `conv1` columns of B2, B3, B4, B8, B11 and B12 are kept, without rescaling: the factor 13/6
  assumes every band contributes alike, and `bn1` follows `conv1`, so training re-estimates the
  activation scale.
- `imagenet` starts `conv1` from RGB filters that smp cycles over the 6 bands, so its first layer
  has no relation to these bands' wavelengths.

Only a fresh start of `train.py` reads starting weights. Without them it fails naming the command in
step 4. A finished or resumed run needs no weights file and takes the hash it started from out of
`run_meta.json`; with both absent, `train.py` fails naming both. The store is known by the hash of the
run's index set and sidecars, so restoring it with new mtimes refuses nothing, though the data root
must stay at the same path. Restoring `runs/` without `weights/` reaches the later stages only if
the paths and modification times of their other inputs are preserved too, because the fingerprint
compares each such file's path, size and mtime: a `best_model.pt` with a new mtime makes the stages
after training refuse. Both weights are plain files
under `$GARIMPO_DATA_DIR/weights/`, so training never reaches a network or a library cache. On Colab
or RunPod, which have internet, run the same commands directly:

```bash
export GARIMPO_DATA_DIR=<DATA_DIR>
uv run python scripts/download_data.py weights --encoder-weights sentinel2_moco
# train.py alone does not check that the selected tiles were downloaded; this does.
uv run python scripts/validate_data.py --train-fraction 1.0 --subset-seed <SUBSET_SEED> --report <REPORT>
uv run python scripts/train.py --arch unet --encoder-weights sentinel2_moco --run-dir runs/unet_sentinel2-f1.0 --train-fraction 1.0 --subset-seed <SUBSET_SEED> ...
```

Every U-Net uses a ResNet-50 encoder, fixed in `models.UNET_ENCODER`, so `LAYERS` carries over between
them and no run can mix encoders. `LAYER` is chosen per variant. The names `decoder.blocks.N` come
from smp's decoder (`decoders/unet/decoder.py` of the pinned 0.6.0), which keeps one block per
`decoder_channels` entry and takes its input widths from the encoder. `fit_directions.py` checks each name against
the built model and stops on an unknown one, and the E2E asserts the two names the pilot uses.

Rerunning one stage needs no earlier stage: `slurm/submit.sh slurm/gpu.sbatch scripts/<stage>.py ...`
for stages that need a GPU, `slurm/cpu.sbatch` for the rest. A new metric or another boundary F1
tolerance, with the run's `SEED` because the seed is part of the fingerprint:

```bash
slurm/submit.sh slurm/cpu.sbatch scripts/compute_metrics.py --run-dir runs/unet_scratch-f1.0 --boundary-tolerance-px <PX> --seed <SEED> --force
```

### Stages

| Stage | Node | Input | Output |
|---|---|---|---|
| `download_data.py metadata` | login | Zenodo 14195737 | `$DATA/zenodo/mining_area_data.gpkg` |
| `download_data.py lists` | login, only to regenerate | `outputs/split.csv`, subset seed | `outputs/tiles/tiles_{1,5,10,100}pct.json` |
| `download_data.py scenes` | login | `outputs/tiles/tiles_<P>pct.json`, `outputs/split.csv`, gpkg, Planetary Computer | `$DATA/{images,masks,scl,stats}.npy`, `$DATA/tiles/*.json` |
| `download_data.py weights` | login | Hugging Face | `$DATA/weights/resnet50_sentinel2_all_moco-df8b932e.pth`, `$DATA/weights/resnet50_imagenet-df1aad85.safetensors` |
| `validate_data.py` | CPU | store, sidecars, split, fraction | `outputs/data_validation-fraction-<F>-seed-<S>.json` |
| `train.py` | GPU | the run's slots and sidecars, split, starting weights of the variant | `best_model.pt`, `last.pt`, `run_meta.json`, `history.csv` |
| `infer.py` | GPU | checkpoint, store | `predictions/cells.npz`, `predictions/{validation,calibration,test}.npy` |
| `fit_directions.py` | GPU | checkpoint, training slots | `directions/<layer>.npz`, `directions/diagnostics.csv` |
| `verify_monotonicity.py --use layer` | GPU | checkpoint, directions, validation slots | `monotonicity_layer.csv` |
| `verify_monotonicity.py --use endpoint` | GPU | checkpoint, direction of the layer, calibration and test slots | `monotonicity_endpoint.csv` |
| `compass_scores.py` | GPU | checkpoint, direction of the layer, calibration and test slots | `scores/scores.npz` |
| `calibrate.py` | CPU | `predictions/cells.npz`, `scores/scores.npz` | `calibration/quantiles.npz` (`quantile`, the `beta_hat` of each row and cell) |
| `generate_intervals.py` | GPU | checkpoint, `calibration/quantiles.npz`, directions, `scores/scores.npz`, test slots; one pass, one encoder per batch | `intervals/intervals.npz` |
| `compute_metrics.py` | CPU | `intervals/intervals.npz`, `predictions/`, `masks.npy`, `calibration/quantiles.npz` | `metrics/cell_metrics.npz`, `metrics/results.csv`, `metrics/width_comparisons.csv`, `metrics/coverage_comparisons.csv`, `metrics/q1_spearman.csv` |
| `merge_results.py` | login | small tables of each run | `outputs/results.csv`, `outputs/width_comparisons.csv`, `outputs/coverage_comparisons.csv`, `outputs/q1_spearman.csv`, `outputs/layer_diagnostics.csv`, `outputs/monotonicity_layer.csv`, `outputs/monotonicity_endpoint.csv` |
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

### Data layout

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
cell's transform is `Affine(*transform) * Affine.translation(col * 512, row * 512)`, in `epsg`.

Artifacts inside a run align by position with its cell table or carry `cell_index`, always ascending:

| Artifact | Arrays |
|---|---|
| `predictions/cells.npz` | `cell_index`, `v`, `vhat`, `p_sum`, `pixel_var`, `cell_ha`, one row per cell of the run's table |
| `predictions/{validation,calibration,test}.npy` | `uint8` probabilities, `[N, 512, 512]`, by role |
| `scores/scores.npz` | `cell_index`; `score[M, N]`, `saturated[M, N]` and `beta_range[M]`, `M` indexing `intervals.COMPASS_METHODS`; `jacobians`, `feature_sums` |
| `calibration/quantiles.npz` | `cell_index`, and `quantile`, `lower_term`, `upper_term`, `scale` as `[K, N]`, `K` indexing `intervals.COMBINATIONS`, `N` the test cells |
| `intervals/intervals.npz` | `cell_index`, and `lower`, `upper`, `quantile` as `[K, N]`, plus `alpha` and `layer` |
| `metrics/cell_metrics.npz` | `cell_index`, `dice`, `boundary_f1`, `boundary_by_rule`, one row per test cell |

CSV is left to human tables: `history.csv`, `directions/diagnostics.csv`, the two monotonicity tables,
and the metrics tables. No stage keeps a derived copy of the store, and
nothing is memoized. `history.csv` records `train_data_wait_seconds` and `seconds` per epoch, and
`infer.py` records `data_wait_seconds` per role in its status; `pipeline_check.py` prints both.

Boundary F1 has a rule for a mask with no boundary, such as one filling the cell: with no boundary in
either mask, F1 is 1 for identical masks and 0 otherwise; with a boundary in only one, F1 is 0.
`n_cells_boundary_by_rule` in `results.csv` counts these cells. Cells with no reference mining stay
NaN and out of the means.

Predictions are uint8 probabilities at 262,144 bytes per cell: 2,816 validation, calibration and
test cells make about 0.74 GB per model. Training cells keep their row in `predictions/cells.npz`.

### Values to decide before the full run

Nothing fixes these yet, so `slurm/full.sbatch` stops when one is missing. Pilot values
are not suggestions.

| Value | Where |
|---|---|
| Channel widths (ConvNet), epochs, batch size, learning rate, weight decay, loss, seed | `train.py` |
| Fixed epochs or fixed updates across fractions; `EPOCHS` is set per run either way | `slurm/full.env.example` |
| Subset seed; one value keeps the fractions nested; another than the versioned lists' 0 means regenerating them | `download_data.py lists`, `train.py` |
| Starting weights and fraction: one env file per run | `train.py`, `slurm/full.env.example` |
| `EVAL_BATCH_SIZE`, for every stage after training; changing it after an OOM refuses nothing | `slurm/full.env.example` |
| Layer to calibrate; layers to diagnose | `fit_directions.py`, `compass_scores.py` |
| SVM `C`, `class_weight`, pixels sampled per cell | `fit_directions.py` |
| Population: every cell, or cells with mining (`--cells`) | `calibrate.py` |
| Score floor of the relative and local methods | `calibrate.py` |
| Boundary F1 tolerance in pixels | `compute_metrics.py` |
| Steps and cells of the monotonicity sweep; its `beta_max` is the search range | `verify_monotonicity.py` |
| Search range and threshold (defaults are upstream's) | `compass_scores.py` |
| Resampling of B11 and B12 to 10 m | `download_data.py scenes` |

## Numbers, scripts and outputs

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

## Test

```bash
slurm/submit.sh slurm/cpu.sbatch -m pytest -q
uvx ruff format --check . && uvx ruff check .
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
| `test_results.py` | PICP, MPIW, interval score and Dice recomputed one cell at a time against `results.csv`; every COMPASS row with its own loader against the one pass in `generate_intervals.py`, with no tolerance |
| `test_stages.py` | Reruns, refusals, saturation, absent cells, merge and `pipeline_check.py` |
| `test_fractions.py` | Two fractions on one data root, the larger fraction's data arriving after the smaller run finished |

Not covered: `--cells nonempty`, real HTTP (STAC, SAS tokens, 429 retries, Zenodo), remote reads, real
Sentinel-2 scenes, the shell scripts in `slurm/` including `FORCE_FROM` and the EXIT trap,
`download_data.py metadata` and `weights`, a resume that actually continues training, cleanup
of the temporary file in `infer.py`, GPU, the real ImageNet weights (a synthetic file covers the load), and the real
Sentinel-2 MoCo file: a synthetic 13-band file stands in, with the variant trained for one epoch and
only inference after it. The boundary F1 assertion in `test_wider_tolerance_can_only_match_more_boundary` depends on the
short synthetic training not being degenerate; it held on the suite's fixed seeds.

## Ways the pipeline can fail

Written before the pipeline, and the origin of the test.

| # | Failure | Caught by |
|---|---|---|
| 1 | A Sentinel-2 scene is missing and the tile disappears silently | `validate_data.py` (artisanal: `missing_scenes`; industrial with `missing_item` or `no_mask_polygon`: `tiles_not_ok`, excluded from the counts), E2E |
| 2 | Only part of a group's selected tiles is downloaded, so a group's coverage is computed on a biased subset | `validate_data.py` (`partial_group`), E2E |
| 3 | A group spans two split roles (leakage) | `validate_data.py` (`group_leakage`), E2E |
| 4 | An interrupted download leaves half-written slots that a resume trusts | slots flushed before the sidecar says `ok`; a forced re-cut removes the sidecar first, E2E |
| 5 | The SAS token expires during a long download | token refresh in `download.py` (cluster only) |
| 6 | Arithmetic on `uint16` DN overflows or goes negative | cast to float before the offset in `CellDataset` |
| 7 | Processing baseline 04.00 adds a radiometric offset, so scenes across 2022 differ by construction | `boa_offset` in the sidecar of each tile, applied by `CellDataset`, E2E |
| 8 | The mask is rasterized on a grid other than the image's | Not checked automatically: raster versus vector area and NDVI contrast are diagnostics, with no threshold, to read in the report |
| 9 | Cell size is not what the area metric assumes | `validate_data.py` (`store_layout`, `pixel_size`) |
| 10 | `g(f(x))` differs from `model(x)`: wrong layer, skip connection lost, batch norm in train mode | E2E |
| 11 | A library upgrade renames the U-Net decoder modules | `models.feature_layers` raises, E2E |
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
| 22 | Compute node tries to reach the network | `uv run --no-sync` in every job, download on the login node |
| 23 | SLURM fails because the log directory does not exist | `slurm/submit.sh` |
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
| 34 | A compute node tries to fetch weights | Nothing in training can: `models.build` reads the two weight files from the data root only |
| 35 | Pretrained weights resume or skip under another identity | `encoder_weights` and the file hash in the fingerprint and `last.pt`, E2E |
| 36 | A larger fraction's download refuses every finished smaller run, or moves its slots | hash of the run's index set and sidecars in the fingerprint; slots fixed by `cell_index`; E2E checks earlier slots stay byte-identical |
| 37 | Tiles written in another order land in other slots | `cell_index` comes from the split alone, E2E writes in a shuffled order and compares bytes |
| 38 | A run trains or evaluates on tiles outside its selection, or a fraction changes under one `RUN_DIR` | `tile_ids` in `run_meta.json`, `train_fraction` and `subset_seed` in the fingerprint and `last.pt`, E2E |
| 39 | A fraction is run with tiles still missing, so it silently trains on fewer | `validate_data.py` (`not_downloaded`, `split_counts`), E2E; it runs before `train` in `slurm/stages.sh`, while `train.py` alone does not check this |
| 40 | A model that predicts nothing has logits far from zero, so at temperature 10,000 every Jacobian is exactly zero | `fit_svm` raises `mean Jacobian is orthogonal to w`, which stops `fit_directions`; seen in the E2E with 4 training cells and 3 epochs. `n_zero_jacobians` in `directions/diagnostics.csv` counts the cells |
| 41 | The one-pass interval stage computes something other than each combination alone | E2E recomputes every COMPASS row with its own loader and encoder pass, exact equality, and counts one read per cell and one encoder pass per batch |
| 42 | An array is lost and comes back as zeros under `ok` sidecars | `create_store` refuses, E2E |
| 43 | The store changes under finished stages, which skip on sidecars alone | `slots_sha256` recomputed by `validate_data.py` and `pipeline_check.py`, E2E zeroes and deletes on the real stages |
| 44 | A changed split reads another tile's slots | `tile_index` in the sidecar and `N == n_tiles * cells_per_tile`, E2E reorders, drops and adds tiles |
| 45 | A lookup of an absent cell returns a neighbor | `data.positions_of` raises naming the `cell_index`, E2E on calibrate, generate_intervals and compute_metrics |

## Attribution and licenses

- **COMPASS** (MIT, `compass/LICENSE`): Cheung, Veeraraghavan and Balakrishnan, COMPASS: Robust
  Feature Conformal Prediction for Medical Segmentation Metrics, ICLR 2026, arXiv:2509.22240. Code
  adapted from <https://github.com/matthewyccheung/compass>.
- **Dataset** (CC BY-SA 4.0): Jasansky et al., Global ML-ready dataset for mining areas in satellite
  images, Zenodo, 2024, DOI 10.5281/zenodo.14195737.
- **Imagery**: modified Copernicus Sentinel data, through Microsoft Planetary Computer.
