# 🏃 Runs

Read to know what a run selects, downloads and starts from.

## 📊 Dataset size

The 1,207 tiles of Jasansky et al. (2024) that carry the dataset's split field enter, 975 industrial
and 232 artisanal (`outputs/frame.json`). A tile is 2,048 x 2,048 pixels at 10 m, cut into 16 cells
of 512 pixels. Industrial tiles within 50 km of a held-out tile are dropped, 28 of them, which
leaves 1,179 tiles and 18,864 cells (`outputs/split_summary.json`):

| Role | Artisanal tiles | Industrial tiles | Cells | Groups |
|---|---|---|---|---|
| `train` | 56 | 947 | 16,048 | 21 |
| `validation` | 18 | 0 | 288 | 5 |
| `calibration` | 69 | 0 | 1,104 | 25 |
| `test` | 89 | 0 | 1,424 | 24 |
| Total | 232 | 947 | 18,864 | 75 |

The store takes 69.2 GB on disk at the 100% fraction and 10.9 GB for the pilot
([training fractions](#-training-fractions)). Each array's size is in
[data-layout.md](data-layout.md).

## ➗ Training fractions

| `TRAIN_FRACTION` | Artisanal training tiles | Industrial training tiles | Tiles, with the 176 of the other roles | Cells | GB written (58.7 MB per tile) |
|---|---|---|---|---|---|
| `0.01` (the pilot) | 1 | 9 | 186 | 2,976 | 10.9 |
| `0.05` | 3 | 47 | 226 | 3,616 | 13.3 |
| `0.10` | 6 | 95 | 277 | 4,432 | 16.3 |
| `1.0` | 56 | 947 | 1,179 | 18,864 | 69.2 |

- The experiment trains at `1.0` only. `0.01` is the pilot's, which debugs the code on 186 tiles.
  `0.05` and `0.10` keep their versioned lists, but no run uses them.
- Only the `train` role is reduced, so every fraction is evaluated on the same cells.
- Training tiles are drawn whole, per scale, from one seeded shuffle, so under one `SUBSET_SEED`
  each subset contains the smaller ones. `SUBSET_SEED` is separate from the training `SEED`.
- An epoch is `STEPS_PER_EPOCH` batches drawn as back-to-back shuffles of the run's training cells,
  not one pass over them, so a run's training time depends on `EPOCHS` and not on its fraction.
- A run reads only its own slots of the shared store, with its tiles recorded as `tile_ids` in
  `run_meta.json`, and each stage's fingerprint holds a hash of its index set and its tiles'
  sidecars. A larger fraction's download therefore leaves finished smaller runs skipping and their
  slots untouched.
- The written sizes hold where the filesystem keeps holes ([data-layout.md](data-layout.md)).
- `validate_data.py` runs per fraction and seed, with its status in
  `$DATA/status/fraction-<F>-seed-<S>/`. Another `--train-fraction` or `--subset-seed` under the
  same `RUN_DIR` refuses.

## 🌍 Tile lists

`outputs/tiles/tiles_{1,5,10,100}pct.json` are versioned and are the source of every download, so a
fresh clone downloads without generating anything. Each list is complete for its fraction: the
reduced training tiles plus all validation, calibration and test tiles, sorted by `tile_id`:

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

`scenes` refuses a list whose `split_sha256` differs from the current split, or that names a tile
the split does not keep. It downloads exactly what the list names; training still selects by
`--train-fraction` and `--subset-seed`. `validate_data.py` fails with `not_downloaded` if those two
ask for a tile the download did not fetch, and the E2E suite asserts that each versioned list equals
`select_tiles` for its fraction and seed.

## 🧬 Model variants

| Model in the results | `ARCH` | `ENCODER_WEIGHTS` | `RUN_DIR` | Input |
|---|---|---|---|---|
| `unet_scratch` | `unet` | `none` | `runs/unet_scratch-f1.0` | z-score of the run's training cells |
| `unet_imagenet` | `unet` | `imagenet` | `runs/unet_imagenet-f1.0` | z-score of the run's training cells |
| `unet_sentinel2` | `unet` | `sentinel2_moco` | `runs/unet_sentinel2-f1.0` | reflectance / 10000, stored as mean 0 and std 1 |
| `convnet` (smp's DeepLabV3) | `convnet` | empty | `runs/convnet-f1.0` | z-score of the run's training cells |

The experiment is these four runs, all at 100%, one config in `configs/full/` each. Each run has 55 method and
calibration rows (`intervals.COMBINATIONS`) at each of the three levels alpha = 0.05, 0.10 and 0.15:

| Rows | Per run and alpha |
|---|---|
| Base rows: `tab:combinacoes` plus the SVM direction | 17 |
| Mondrian rows, `*_mondrian` | 12 |
| Asymmetric COMPASS rows, `a_*` | 6 |
| Repeated subsampling rows, `*_subsampled` ([design.md](design.md#-quantile-rule-and-its-sensitivity-analysis)) | 20 |
| Total | 55 |

Over 4 runs and 3 levels that is 660 rows. `merge_results.py` identifies a run by
`(model, train_fraction)` and refuses a repeated pair; each run's paired comparisons stay inside
that run. What the comparison between variants can show is in
[design.md](design.md#-what-the-model-comparison-varies).

## 🌱 Starting weights

| Weights | Source | License |
|---|---|---|
| `sentinel2_moco` | `torchgeo/resnet18_sentinel2_all_moco` on Hugging Face, revision `5b8cddc`, the file `ResNet18_Weights.SENTINEL2_ALL_MOCO` of torchgeo, trained on SSL4EO-S12; URL and hash from torchgeo's `models/resnet.py`; sha256 prefix `59bfdff9` checked on download, full hash kept in the checkpoint | `cc-by-4.0` (Hugging Face API metadata) |
| `imagenet` | `smp-hub/resnet18.imagenet` on Hugging Face, the repository that `encoders/resnet.py` of segmentation-models-pytorch 0.6.0 names; file `model.safetensors` at revision `2342da2`, pinned in `download.py` (smp 0.6.0 itself pins `3f2325f`); sha256 prefix `b1d1d36f` from the Hugging Face API, checked on download | `other` (Hugging Face API metadata): read the model card before citing it |

Both files land under `$GARIMPO_DATA_DIR/weights/`
([stages.md](stages.md#-stage-table)). Their limits:
[design.md](design.md#-limits-of-the-starting-weights).

## 🔤 Encoder and layer names

Every model uses a ResNet-18 encoder, fixed in `models.ENCODER`, so `LAYERS` carries over between
the U-Nets and no run can mix encoders. `choose_layer.py` chooses the layer per run.

| Model | Layer names | Source |
|---|---|---|
| U-Net | `decoder.blocks.N` | smp's decoder (`decoders/unet/decoder.py` of the pinned 0.6.0), which keeps one block per `decoder_channels` entry and takes its input widths from the encoder |
| ConvNet | `decoder.aspp`, `decoder.relu` | `decoders/deeplabv3/decoder.py` |

`fit_directions.py` checks each name against the built model and stops on an unknown one. Every
config diagnoses all of them: `decoder.blocks.0` to `decoder.blocks.4` for the U-Nets,
`decoder.aspp` and `decoder.relu` for the ConvNet. The E2E suite checks that every versioned config
names only layers of its model.

## 🌈 Inclusion

🏳️‍🌈 🏳️‍⚧️ LGBTQIA+ people are welcome in this project. What that means for issues, pull requests
and new text: [Inclusion](../README.md#-inclusion) in the README.
