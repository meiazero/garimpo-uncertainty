# Full run, from zero

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

## Training fractions

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
The written sizes hold where the filesystem keeps holes; see [data-layout.md](data-layout.md).
`validate_data.py` runs per fraction and seed, with its status in
`$DATA/status/fraction-<F>-seed-<S>/`. Another `--train-fraction` or `--subset-seed` under the same
`RUN_DIR` refuses.

## Tile lists

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

## Model variants

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

## Values to decide before the full run

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
