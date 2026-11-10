# Values of the full run

Read before changing a value in `configs/full/`.

Every value of the four runs is decided here, before any run, with its source; where no paper sets
one, the row says what the choice is for. None is tuned on the pilot, which only checks the code.
`configs/full/` holds the four runs' configs and `configs/pilot/` the pilot's.
`scripts/run_config.py` stops a job on a config with a value missing, null, empty or unknown.
`ARCH`, `ENCODER_WEIGHTS`, `RUN_DIR` and `TRAIN_FRACTION` name the run, one config each, as listed
in [runs.md](runs.md#model-variants).

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
| `EVAL_BATCH_SIZE` | 16, `BATCH_SIZE` | The stages after training load the model in eval mode with `requires_grad` off (`checkpoints.py:69`), and their one backward pass, the Jacobian, differentiates only the layers after `z` (`directions.jacobian_sum`). At the same batch they hold less than training, which also keeps the whole network's activations, the weight gradients and AdamW's two moments. Results do not depend on it, since in eval mode each sample is independent; changing it after an OOM refuses nothing |
| Layers to diagnose (`LAYERS`) | U-Net: `decoder.blocks.0` to `decoder.blocks.4`. ConvNet: `decoder.aspp`, `decoder.relu` | COMPASS Appendix E tried every layer of the head; `choose_layer` then calibrates one of them |
| `DIRECTION_TEMPERATURE` | 10000, in `fit_directions`, `verify_monotonicity` and `compass_scores` | Upstream's temperature, the metric's. Its known cost: the gradient vanishes beyond \|logit\| ≈ 5e-3, so cells predicted empty get the PCA mean as direction. A temperature chosen by leave-one-group-out on calibration was not pursued, since the aim is COMPASS's own setting, not a tuned one |
| `MIN_FRACTION_ASCENDING`: share of validation cells whose sweep must be ascending, in both directions, for a layer to be chosen | 0.95 | Cheung et al. (2026) set none: §4.1 only says to verify monotonicity, and §2.3 reports it held in all their experiments. Chosen as 1 minus the smallest alpha, 0.05, so the share of non-monotonic cells stays below the smallest miscoverage the intervals allow. With `SWEEP_CELLS` 20 it needs 19 of 20 cells. If no layer reaches it, `choose_layer` stops the job naming the best layer and its fraction; rerunning with another value redoes only `choose_layer` and what follows |
| SVM `C`, `class_weight`, pixels sampled per cell | 1.0, `balanced`, 200 | No paper sets them: COMPASS takes its direction from a PCA of the Jacobians, and the SVM is this work's own. `C` is `LinearSVC`'s default (scikit-learn 1.9.1); `balanced` weights mining and background pixels by their inverse frequency, so the boundary is not set by the larger class; 200 pixels per cell bound the fit's size. Not tuned |
| Steps and cells of the monotonicity sweep | 21 steps, 20 cells per role | COMPASS gives no values (Appendix C). The sweep is `linspace(-beta_max, beta_max, steps)` (`compass2d/compass.py:151`), so 21 steps keep the origin and 10 points per side; 20 cells bound its cost. Not tuned |
| `SEARCH_THRESHOLD_FEATURES` | 1e-15 | Upstream's runner: 60 bisection steps from 1000. The pilot keeps 1e-3, 20 steps, to stay short |
| Score floor of the relative and local methods (`SCORE_FLOOR_HA`) | 0.01 ha | One 10 m pixel, the smallest area a cell can show. Romano et al. 2019 add γ to the denominator, set to 1 after dividing the response by its mean absolute value; here the floor stays in hectares instead, so it has a physical meaning. Not tuned |
