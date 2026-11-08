# Values of the full run

Read before step 3 of [full-run.md](full-run.md), where the values file is filled in.

`slurm/full.env.example` holds the decided values, so a copy of it starts with them filled in.
`slurm/full.sbatch` stops on the first open value left empty. The pilot's values outside the
Decided table are not suggestions. `ARCH`, `ENCODER_WEIGHTS`, `RUN_DIR` and `TRAIN_FRACTION` name
the run, one env file each, as listed in [runs.md](runs.md#model-variants).

## Decided

| Value | Set to | Source |
|---|---|---|
| Training budget across fractions | Same `EPOCHS` and `STEPS_PER_EPOCH` in every run, so every fraction gets the same updates and is validated at the same steps | Bornschein et al. 2020, arXiv:2009.12583, §3.3: an epoch is "always 1.28M/256 = 5000 gradient steps, regardless of the size of the actual training set used"; with fixed epochs the 1% run would get 100× fewer updates than the 100% run |
| `EPOCHS` | 100, every run | User decision, 2026-11-08: 25,000 updates of 16 cells. The time per update is what the pilot measures |
| `STEPS_PER_EPOCH` | 250 | nnU-Net, Isensee et al. 2021, arXiv:1904.08128: "each epoch is defined as 250 training iterations" |
| `BATCH_SIZE` | 16 | Jasansky et al., same dataset and 512-pixel chips: `configs/cnn/cnn_segment_config_gpu_L4.yaml` of github.com/SimonJasansky/mine-segmentation. COMPASS used 32 on 128-pixel images (Appendix E) |
| `LR` | 1e-4, AdamW | Cheung et al. 2026, COMPASS, Appendix E; `2D/datasets/kvasir/run_nn.py:58` of github.com/matthewyccheung/compass |
| `WEIGHT_DECAY` | 0.01 | The same upstream line calls AdamW without `weight_decay`, so it trains at PyTorch's default, 1e-2 |
| `LOSS` | `dice_bce` | Ma et al. 2021, Medical Image Analysis, PubMed 33813286: compound losses "are the most robust losses"; nnU-Net combines Dice and cross-entropy. Upstream COMPASS trains on Dice alone |
| `SEED`, `SUBSET_SEED` | 0, 0 | `SUBSET_SEED` is the versioned lists' seed, so nothing is regenerated |
| Resampling of B11 and B12 to 10 m | `bilinear` | The `sentinel2_moco` weights were pretrained on SSL4EO-S12, upsampled with `cv2.INTER_LINEAR_EXACT` (`ssl4eo_dataset.py:102` of github.com/zhu-xlab/SSL4EO-S12); torchgeo resamples float rasters bilinearly. No study comparing nearest and bilinear for this was found. Jasansky et al.'s code downloads through `stackstac`, whose default is nearest |
| `BETA_RANGE_LOGITS`, `BETA_RANGE_FEATURES`, `SEARCH_THRESHOLD_LOGITS` | 100, 1000, 1e-2 | Upstream's active runner, `2D/run_cp.py:488-492`. The paper gives none of them |
| `CELLS` | `all` | Calibration must come from the test cells' distribution (Mossina et al. 2024, Algorithm 1). `nonempty` selects by the reference, unknown when an interval is used |
| `BOUNDARY_TOLERANCE_PX` | 5.43 | 0.75% of the image diagonal, Csurka et al. 2013, BMVC; also MATLAB `bfscore`'s default. A 512-pixel cell's diagonal is 724 pixels |
| ConvNet (`--arch convnet`) | smp's `DeepLabV3`: encoder `models.ENCODER` (ResNet-18, the U-Net's), `encoder_weights=None`, `classes=1`, every other argument at smp's default: `encoder_output_stride` 8, `decoder_channels` 256, `decoder_atrous_rates` (12, 24, 36), ASPP not separable, `decoder_aspp_dropout` 0.5, final upsampling equal to the output stride | User decision, 2026-11-08. In the installed `segmentation_models_pytorch` 0.6.0: defaults at `decoders/deeplabv3/model.py:61-73`, upsampling at `model.py:93-99`; `DeepLabV3Decoder.forward` at `decoders/deeplabv3/decoder.py:64-70` starts from `features[-1]` alone, so no skip connections. Against `unet_scratch` it changes the decoder and, with it, the encoder's output stride: the same ResNet-18 layers, dilated to stride 8 (`model.py:84-91` passes `output_stride` to the encoder) |
| `alpha` | 0.05, 0.1 and 0.15, every run | Cheung et al. 2026, COMPASS, Table 1; the constant `intervals.ALPHAS`, no flag |

## Open

| Value | Where | What settles it |
|---|---|---|
| `EVAL_BATCH_SIZE` | `slurm/full.env.example` | The pilot's GPU peaks; changing it after an OOM refuses nothing |
| Layers to diagnose (`LAYERS`) | `fit_directions.py` | COMPASS Appendix E tried every layer of the head; `choose_layer` then calibrates one of them |
| Direction temperature (`--direction-temperature`) | `fit_directions.py`, `verify_monotonicity.py`, `compass_scores.py` | The sigmoid temperature of the area whose Jacobian sets the COMPASS-J direction, separate from the metric's 1e4. At 1e4 the gradient vanishes beyond \|logit\| ≈ 5e-3, so cells predicted empty get the PCA mean as direction. Choose it by leave-one-group-out on calibration; the same value in all three stages |
| `MIN_FRACTION_ASCENDING`: share of validation cells whose sweep must be ascending, in both directions, for a layer to be chosen | `choose_layer.py` | Cheung et al. (2026) set none: §4.1 only says to verify monotonicity, and §2.3 reports it held in all their experiments. Read the pilot's `monotonicity_layer.csv` before fixing it. If it is too high, `choose_layer` stops the job naming the best layer and its fraction; resubmitting with another value redoes only `choose_layer` and what follows, since the stages before it skip |
| SVM `C`, `class_weight`, pixels sampled per cell | `fit_directions.py` | No source: COMPASS takes its direction from a PCA of the Jacobians, and the SVM is this work's own |
| Steps and cells of the monotonicity sweep | `verify_monotonicity.py` | The sweep's time in the pilot; COMPASS gives no values (Appendix C) |
| `SEARCH_THRESHOLD_FEATURES` | `compass_scores.py` | Upstream's runner uses 1e-15, 60 bisection steps from 1000; the pilot's 1e-3 takes 20, a third of the cost |
| Score floor of the relative and local methods | `calibrate.py` | Romano et al. 2019 add γ to the denominator, set to 1 after dividing the response by its mean absolute value; COMPASS has no relative score |
