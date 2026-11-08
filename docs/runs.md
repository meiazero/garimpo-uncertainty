# Runs

Read to know what a run selects, downloads and starts from.

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
- Every run makes the same updates: an epoch is `STEPS_PER_EPOCH` batches drawn as back-to-back
  shuffles of the run's training cells, not one pass over them. So the 1% run sees each cell about
  100 times as often as the 100% run, and its best checkpoint, chosen on validation Dice, guards
  against the overfitting this invites.

A run reads only its own slots of the shared store, its tiles recorded as `tile_ids` in
`run_meta.json`, and each stage's fingerprint holds a hash of its index set and its tiles' sidecars.
So a larger fraction's download leaves finished smaller runs skipping and their slots untouched.
The written sizes hold where the filesystem keeps holes; see [data-layout.md](data-layout.md).
`validate_data.py` runs per fraction and seed, with its status in
`$DATA/status/fraction-<F>-seed-<S>/`. Another `--train-fraction` or `--subset-seed` under the same
`RUN_DIR` refuses.

## Quantile rule and its sensitivity analysis

The main rows take the group quantile by CDF pooling (Dunn et al. 2022, §3.2), which weighs every
group equally but is valid only as k grows and for a continuous response. Their unsupervised
simulation undercovers at k ≤ 35; their residual simulation, closer to this one, stays near 1 - α
(§4.4). Calibration has 25 groups and own region 23, and the areas are far from continuous: 372 of the
864 Amazon cells have no mining (`chips_amazon_nonempty` in `outputs/frame.json`). So every `external` and `own_region` row is repeated
as `external_subsampled` and `own_region_subsampled`, by repeated subsampling (§4.3, B = 100): one
score per group per subsample, coverage at least 1 - 2α for a new group at any k (Theorem 10), and
a refusal at k ≤ 1/α - 1, where no interval is bounded. `external_weighted` has no subsampled row,
since subsampling has no definition with density-ratio weights.

A run calibrated before these rows existed skips `calibrate` with 17 rows, and the pipeline check
fails on the absent ones: resubmit with `FORCE_FROM=calibrate`.

Both rules see the same scores, so `coverage_comparisons.csv` and `width_comparisons.csv` already
pair them. The reading is fixed before the data:

1. Robust: each method's Q2 verdict and the width order of the methods that pass are the same under
   both rules.
2. Pooling's own undercoverage: own-region coverage below 0.90 by pooling and not by subsampling.
   Then the geographic effect is read as external minus own region within one rule.
3. Geographic shift: external coverage falls under both rules and own region does not.

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

`scenes` refuses a list whose `split_sha256` differs from the current split, or that names a tile the
split does not keep. It downloads exactly what the list names, and training still selects by
`--train-fraction` and `--subset-seed`. `validate_data.py` fails with `not_downloaded` if those two
ask for a tile the download did not fetch, and the E2E asserts that each versioned list equals
`select_tiles` for its fraction and seed.

## Model variants

| Model in the results | `ARCH` | `ENCODER_WEIGHTS` | `RUN_DIR` | Input |
|---|---|---|---|---|
| `unet_scratch` | `unet` | `none` | `runs/unet_scratch-f<F>` | z-score of the run's training cells |
| `unet_imagenet` | `unet` | `imagenet` | `runs/unet_imagenet-f<F>` | z-score of the run's training cells |
| `unet_sentinel2` | `unet` | `sentinel2_moco` | `runs/unet_sentinel2-f<F>` | reflectance / 10000, stored as mean 0 and std 1 |
| `convnet` | `convnet` | empty | `runs/convnet-f1.0` | z-score of the run's training cells |

The learning curve is 3 U-Net variants x 4 fractions plus the ConvNet at 100%: 13 runs, one env file
each, and 13 x 31 = 403 method and calibration rows: 17 main rows and 14 of the sensitivity
analysis above. The code does not refuse a ConvNet at a lower
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
step 4 of [full-run.md](full-run.md). A finished or resumed run needs no weights file and takes the hash it started from out of
`run_meta.json`; with both absent, `train.py` fails naming both. The store is known by the hash of the
run's index set and sidecars, so restoring it with new mtimes refuses nothing, though the data root
must stay at the same path. Restoring `runs/` without `weights/` reaches the later stages only if
the paths and modification times of their other inputs are preserved too, because the fingerprint
compares each such file's path, size and mtime: a `best_model.pt` with a new mtime makes the stages
after training refuse. Both weights are plain files
under `$GARIMPO_DATA_DIR/weights/`, so training never reaches a network or a library cache.

Every U-Net uses a ResNet-50 encoder, fixed in `models.UNET_ENCODER`, so `LAYERS` carries over between
them and no run can mix encoders. `LAYER` is chosen per variant. The names `decoder.blocks.N` come
from smp's decoder (`decoders/unet/decoder.py` of the pinned 0.6.0), which keeps one block per
`decoder_channels` entry and takes its input widths from the encoder. `fit_directions.py` checks each name against
the built model and stops on an unknown one, and the E2E asserts the two names the pilot uses.
