# garimpo-uncertainty

Replication package of a master's dissertation: calibrated prediction intervals for artisanal mining
area from Sentinel-2, evaluated under geographic domain shift. `compass/` (submodule) holds the
adapted COMPASS method; `src/garimpo_uncertainty/` the satellite code; `scripts/` the stages;
`slurm/` the cluster jobs; `outputs/` the versioned tables, including `split.csv`, the split used.

No model result exists yet. None of this code has been executed so far.

## Pilot: adjust the code on the cluster

The pilot runs the real pipeline on a small real subset through the same scripts as the full run. It
exists to debug the implementation, and its numbers say nothing about any method. Its values are in
`slurm/pilot.sbatch`, labeled pilot-only. Results go to `runs/pilot/`, never to `outputs/`.

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

Each command is its own job with its own 4 h limit, and writes to `runs/pilot/unet-<weights>/`.

The job ends with `scripts/pipeline_check.py`: status, seconds, peak memory and shapes per stage,
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
| 3 | login | `uv run --no-sync python scripts/download_data.py scenes --resampling-20m <nearest\|bilinear>` (resumable; about 59 GB uncompressed, per `proposta/pendencias.md`; another resampling refuses; `--force` redoes every selected tile, then rerun with `FORCE_VALIDATE=1 FORCE_FROM=train`, since a re-cut that leaves the manifest unchanged reaches no later stage by itself) |
| 4 | login | `uv run --no-sync python scripts/download_data.py weights --encoder-weights imagenet sentinel2_moco` |
| 5 | login | `cp slurm/full.env.example slurm/unet_scratch.env`, then fill in every value (see decisions below) except `LAYER`, with `STOP_AFTER_DIAGNOSTICS=1` |
| 6 | `sbatch` | `slurm/submit.sh slurm/full.sbatch slurm/unet_scratch.env` (stops after the layer diagnostics) |
| 7 | login | Read `runs/unet_scratch/directions/diagnostics.csv` and `runs/unet_scratch/monotonicity_layer.csv` (validation role only); set `LAYER`, clear `STOP_AFTER_DIAGNOSTICS` |
| 8 | `sbatch` | `slurm/submit.sh slurm/full.sbatch slurm/unet_scratch.env` (finished stages skip, so nothing retrains) |
| 9 | both | Repeat steps 5 to 8 for each other variant below |
| 10 | login | `uv run --no-sync python scripts/merge_results.py --runs runs/unet_scratch runs/unet_imagenet runs/unet_sentinel2 runs/convnet --out outputs` |

Compute nodes have no internet, so `uv sync` and steps 2 to 4 only run on the login node; every job
uses `uv run --no-sync`. The data root is `$GARIMPO_DATA_DIR`, default
`~/Experiments/datasets/garimpo`.

### Model variants

| Model in the results | `ARCH` | `ENCODER_WEIGHTS` | `RUN_DIR` | Input |
|---|---|---|---|---|
| `unet_scratch` | `unet` | `none` | `runs/unet_scratch` | z-score of the training cells |
| `unet_imagenet` | `unet` | `imagenet` | `runs/unet_imagenet` | z-score of the training cells |
| `unet_sentinel2` | `unet` | `sentinel2_moco` | `runs/unet_sentinel2` | reflectance / 10000, stored as mean 0 and std 1 |
| `convnet` | `convnet` | empty | `runs/convnet` | z-score of the training cells |

Each of the 4 variants gives 17 method and calibration rows: 68 in total.
Each variant is one training, so differences between variants have no replication behind them and
the comparison is descriptive only. `merge_results.py` refuses two runs of the same variant.

| Weights | Source | License |
|---|---|---|
| `sentinel2_moco` | `torchgeo/resnet50_sentinel2_all_moco` on Hugging Face, revision `da4f3c9`, the file `ResNet50_Weights.SENTINEL2_ALL_MOCO` of torchgeo, trained on SSL4EO-S12; URL and hash from torchgeo's `models/resnet.py`; sha256 prefix `df8b932e` checked on download, full hash kept in the checkpoint | `cc-by-4.0` (Hugging Face API metadata) |
| `imagenet` | `smp-hub/resnet50.imagenet` on Hugging Face, from `encoders/resnet.py` of segmentation-models-pytorch 0.6.0, whose `get_encoder` fetches `config.json` and `model.safetensors` with no URL fallback | `other` (Hugging Face API metadata): read the model card before citing it |

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
`run_meta.json`; with both absent, `train.py` fails naming both. Restoring `runs/` without `weights/`
reaches the later stages only if paths and modification times are preserved too, because the
fingerprint compares each input's path, size and mtime: a `manifest.csv` restored with a new mtime
makes `train` refuse, and the same holds for any stage's other inputs. ImageNet weights resolve from
`$GARIMPO_DATA_DIR/weights/huggingface`, and the jobs set `HF_HOME` to that path and
`HF_HUB_OFFLINE=1`. On Colab or RunPod, which have internet, run the same commands directly:

```bash
export GARIMPO_DATA_DIR=<DATA_DIR>
export HF_HOME="$GARIMPO_DATA_DIR/weights/huggingface"
uv run python scripts/download_data.py weights --encoder-weights sentinel2_moco
uv run python scripts/train.py --arch unet --encoder-weights sentinel2_moco --run-dir runs/unet_sentinel2 ...
```

Every U-Net uses a ResNet-50 encoder, fixed in `models.UNET_ENCODER`, so `LAYERS` carries over between
them and no run can mix encoders. `LAYER` is chosen per variant. The names `decoder.blocks.N` come
from smp's decoder, which keeps one block per encoder stage and takes its input widths from the
encoder (from memory of smp, `[NÃO VERIFICADO]` here). `fit_directions.py` checks each name against
the built model and stops on an unknown one, and the E2E asserts the two names the pilot uses.

Rerunning one stage needs no earlier stage: `slurm/submit.sh slurm/gpu.sbatch scripts/<stage>.py ...`
for stages that need a GPU, `slurm/cpu.sbatch` for the rest. A new metric or another boundary F1
tolerance, with the run's `SEED` because the seed is part of the fingerprint:

```bash
slurm/submit.sh slurm/cpu.sbatch scripts/compute_metrics.py --run-dir runs/unet_scratch --boundary-tolerance-px <PX> --seed <SEED> --force
```

### Stages

| Stage | Node | Input | Output |
|---|---|---|---|
| `download_data.py metadata` | login | Zenodo 14195737 | `$DATA/zenodo/mining_area_data.gpkg` |
| `download_data.py scenes` | login | `outputs/split.csv`, gpkg, Planetary Computer | `$DATA/cells/`, `$DATA/tiles/*.json`, `$DATA/manifest.csv`, `$DATA/download_report.csv` |
| `download_data.py weights` | login | Hugging Face | `$DATA/weights/resnet50_sentinel2_all_moco-df8b932e.pth`, `$DATA/weights/huggingface/` |
| `validate_data.py` | CPU | manifest, report, cells, split | `outputs/data_validation.json`, `$DATA/validation_cells.csv` |
| `train.py` | GPU | manifest, cells, starting weights of the variant | `best_model.pt`, `last.pt`, `run_meta.json`, `history.csv` |
| `infer.py` | GPU | checkpoint, cells | `predictions/cells.csv`, `predictions/{validation,calibration,test}.npy` |
| `fit_directions.py` | GPU | checkpoint, training cells | `directions/<layer>.npz`, `directions/diagnostics.csv` |
| `verify_monotonicity.py --use layer` | GPU | checkpoint, directions, validation cells | `monotonicity_layer.csv` |
| `verify_monotonicity.py --use endpoint` | GPU | checkpoint, direction of the layer, calibration and test cells | `monotonicity_endpoint.csv` |
| `compass_scores.py` | GPU | checkpoint, direction of the layer, cells | `scores/compass_scores.csv`, `scores/eval_features.npz` |
| `calibrate.py` | CPU | `predictions/cells.csv`, scores, features | `calibration/quantiles.csv` (`beta_hat` per cell), `calibration/summary.csv` |
| `generate_intervals.py` | GPU | checkpoint, quantiles, directions, features | `intervals/intervals.csv` |
| `compute_metrics.py` | CPU | intervals, `predictions/`, dataset masks, `calibration/summary.csv` | `metrics/cell_metrics.csv`, `metrics/results.csv`, `metrics/width_comparisons.csv`, `metrics/coverage_comparisons.csv`, `metrics/q1_spearman.csv` |
| `merge_results.py` | login | small tables of each run | `outputs/results.csv`, `outputs/width_comparisons.csv`, `outputs/coverage_comparisons.csv`, `outputs/q1_spearman.csv`, `outputs/layer_diagnostics.csv`, `outputs/monotonicity_layer.csv`, `outputs/monotonicity_endpoint.csv` |
| `pipeline_check.py` | any | status files and tables | prints the debugging report |

Paths without a prefix are inside the run directory. Every stage skips its work when its output
exists from the same inputs, stops with status `refused` when it exists from other inputs, and
redoes it with `--force`; each writes `status/<stage>.json`, and a missing input is recorded there
as a failure. Only `train.py` ever updates weights. `best_model.pt` stores the model configuration,
variant, starting weights, seed, normalization, split hash and the commits of root and submodule.
Later stages take the normalization from it and need no weights file. They refuse a checkpoint from
another split, and `train.py` refuses to resume `last.pt` under other arguments, weights included.

`validate_data.py` separates what can fail from what cannot. `problems` fail the stage. A missing
artisanal tile and group leakage are always problems. An industrial tile with `missing_item` or
`no_mask_polygon` is listed in `tiles_not_ok` and the run proceeds without it; any other industrial
status, including a window off the scene, is a problem. `diagnostics` have no threshold and cannot fail the
stage: cloud and nodata fractions recomputed from each file, counts of 100% nodata and 100% cloud
cells, raster versus vector area, box excess, NDVI contrast. The `tile_offset` problem discounts the
polygon's box excess over the window; the centered window then bounds it at half a pixel, so it
trips only on a defect in the manifest or in `tile_origin`.

Boundary F1 has a rule for a mask with no boundary, such as one filling the cell: with no boundary in
either mask, F1 is 1 for identical masks and 0 otherwise; with a boundary in only one, F1 is 0.
`n_cells_boundary_by_rule` in `results.csv` counts these cells. Cells with no reference mining stay
NaN and out of the means.

Predictions are uint8 probabilities at 262,144 bytes per cell: 2,816 validation, calibration and
test cells make about 0.74 GB per model. Training cells keep their row in `predictions/cells.csv`.

### Values to decide before the full run

The proposal fixes none of these, so `slurm/full.sbatch` stops when one is missing. Pilot values
are not suggestions.

| Value | Where |
|---|---|
| Channel widths (ConvNet), epochs, batch size, learning rate, weight decay, loss, seed | `train.py` |
| Starting weights: one env file per variant | `train.py`, `slurm/full.env.example` |
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

Run from the repository root. The proposal scripts need `data/zenodo`: link it with
`mkdir -p data && ln -s "$GARIMPO_DATA_DIR/zenodo" data/zenodo`, or run
`scripts/proposal/get_data.sh`.

```bash
scripts/proposal/get_biomes.sh                        # IBGE biomes into data/biomes, sha256 checked
uv run python -I scripts/proposal/frame.py
uv run python -I scripts/proposal/check_groups.py
uv run python -I scripts/proposal/check_imagery.py    # one STAC request per artisanal tile
uv run python -I scripts/proposal/coverage_precision.py
uv run python -I scripts/proposal/make_split.py
uv run python -I scripts/proposal/check_labels.py     # agreement between the dataset's own masks
uv run python -I scripts/proposal/make_examples.py    # reads 4 Sentinel-2 chips
python3 -I scripts/proposal/get_papers.py             # open-access references into papers/
cd proposta && "$(mise -C ../../dissertacao-v4 which latexmk)" -lualatex proposta_orientador.tex
```

| Number | Script | Output |
|---|---|---|
| Tiles by scale and region, chip areas, image years | `scripts/proposal/frame.py` | `outputs/frame.json`, `outputs/tiles_artisanal.csv` |
| Groups at 50 km, cross-check in EPSG:5880 | `scripts/proposal/frame.py`, `check_groups.py` | `outputs/groups.json` |
| Sentinel-2 items found (232 of 232 artisanal) | `scripts/proposal/check_imagery.py` | `outputs/imagery.csv` |
| PICP confidence intervals, power, required M, realized coverage | `scripts/proposal/coverage_precision.py` | `outputs/coverage_precision.json` |
| Geographic partition and training chips | `scripts/proposal/make_split.py` | `outputs/split.csv`, `outputs/split_summary.json` |
| Agreement between dataset masks (not cited in the text yet) | `scripts/proposal/check_labels.py` | `outputs/label_agreement.json` |
| Figure of dataset examples (`fig:exemplos`) | `scripts/proposal/make_examples.py` | `proposta/exemplos_dataset.pdf`, `outputs/exemplos_dataset.png`, `outputs/exemplos_dataset.csv` |
| Data checks, scenes missing, clouds and nodata per role | `scripts/validate_data.py` | `outputs/data_validation.json` |
| PICP, MPIW, interval score, MAE, bias, relative error, Dice, boundary F1, with 95% group bootstrap, one row per model x method x calibration | `scripts/compute_metrics.py`, `merge_results.py` | `outputs/results.csv` |
| Paired width differences between methods | `scripts/compute_metrics.py`, `merge_results.py` | `outputs/width_comparisons.csv` |
| Paired coverage differences between methods and calibrations (Q3) | `scripts/compute_metrics.py`, `merge_results.py` | `outputs/coverage_comparisons.csv` |
| Spearman of area error with Dice and boundary F1, per model (Q1) | `scripts/compute_metrics.py`, `merge_results.py` | `outputs/q1_spearman.csv` |
| PCA explained variance and `\|cos(v1, w)\|` per layer | `scripts/fit_directions.py`, `merge_results.py` | `outputs/layer_diagnostics.csv` |
| Monotonic fraction per layer and direction, validation role, for choosing the layer | `scripts/verify_monotonicity.py`, `merge_results.py` | `outputs/monotonicity_layer.csv` |
| Monotonic fraction of the chosen layer, calibration and test roles | `scripts/verify_monotonicity.py`, `merge_results.py` | `outputs/monotonicity_endpoint.csv` |

Claims outside these scripts map to their sources in `proposta/evidencias.csv`. Groups join tiles
closer than 50 km, measured edge to edge. EPSG:6933 is the frame used (12 Amazon groups); EPSG:5880
gives 14, within the distortion of a global projection.

## Test

```bash
slurm/submit.sh slurm/cpu.sbatch -m pytest -q
uvx ruff format --check . && uvx ruff check .
```

The tests leave their artifacts in `tests/artifacts/`. One builds a synthetic dataset through the
real cell writer and runs every stage after the download on CPU. Another runs the download code on
local rasters. Not covered: real HTTP (STAC, SAS tokens, 429 retries, Zenodo), remote reads, real
Sentinel-2 scenes, the shell scripts in `slurm/` including `FORCE_FROM` and the EXIT trap, the
download CLI including `download_data.py weights`, a resume that actually continues training, cleanup
of the temporary file in `infer.py`, GPU, loading ImageNet weights (they need the hub; only the failure on an empty offline cache runs), and the real
Sentinel-2 MoCo file: a synthetic 13-band file stands in, with the variant trained for one epoch and
only inference after it. Also undecided until a run: whether the short synthetic training gives
predictions degenerate enough to fail the boundary F1 assertion in `check_metrics_never_retrain`.

## Ways the pipeline can fail

Written before the pipeline, and the origin of the test.

| # | Failure | Caught by |
|---|---|---|
| 1 | A Sentinel-2 scene is missing and the tile disappears silently | `validate_data.py` (artisanal: `missing_scenes`; industrial with `missing_item` or `no_mask_polygon`: `tiles_not_ok`, excluded from the counts), E2E |
| 2 | Only part of a group is downloaded, so a group's coverage is computed on a biased subset | `validate_data.py` (`partial_group`), E2E |
| 3 | A group spans two split roles (leakage) | `validate_data.py` (`group_leakage`), E2E |
| 4 | An interrupted download leaves a half-written cell that a resume trusts | atomic write plus sidecar written last |
| 5 | The SAS token expires during a long download | token refresh in `download.py` (cluster only) |
| 6 | Rasters arrive in a compacted dtype and arithmetic overflows | cast after read in `CellDataset` |
| 7 | Processing baseline 04.00 adds a radiometric offset, so scenes across 2022 differ by construction | `boa_offset` per tile in the manifest |
| 8 | The mask is rasterized on a grid other than the image's | Not checked automatically: raster versus vector area and NDVI contrast are diagnostics, with no threshold, to read in the report |
| 9 | Cell size is not what the area metric assumes | `validate_data.py` (`cell_shape`, `pixel_size`) |
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
| 20 | Bootstrap resamples cells instead of groups, or method pairs use different resamples | shared group index matrix, recomputed in E2E |
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
| 33 | Variants merge under one model name | `variant` in the metadata and `model` column; `merge_results.py` refuses a repeated variant, E2E |
| 34 | A compute node tries to fetch weights | `HF_HUB_OFFLINE=1` in `slurm/stages.sh` and `gpu.sbatch`; the Sentinel-2 file is read from the data root only |
| 35 | Pretrained weights resume or skip under another identity | `encoder_weights` and the file hash in the fingerprint and `last.pt`, E2E |

## Attribution and licenses

- **COMPASS** (MIT, `compass/LICENSE`): Cheung, Veeraraghavan and Balakrishnan, COMPASS: Robust
  Feature Conformal Prediction for Medical Segmentation Metrics, ICLR 2026, arXiv:2509.22240. Code
  adapted from <https://github.com/matthewyccheung/compass>.
- **Dataset** (CC BY-SA 4.0): Jasansky et al., Global ML-ready dataset for mining areas in satellite
  images, Zenodo, 2024, DOI 10.5281/zenodo.14195737.
- **Imagery**: modified Copernicus Sentinel data, through Microsoft Planetary Computer.
