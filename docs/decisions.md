# Values of the full run

Read before step 5 of [full-run.md](full-run.md).

`slurm/full.env.example` holds the decided values, so a copy of it starts with them filled in.
`slurm/full.sbatch` stops on the first open value left empty. Pilot values outside the Decided
table are not suggestions. `ARCH`, `ARCH_ARGS`, `ENCODER_WEIGHTS`, `RUN_DIR` and `TRAIN_FRACTION`
name the run, one env file each, as listed in [runs.md](runs.md#model-variants).

## Decided

| Value | Set to | Source |
|---|---|---|
| Training budget across fractions | Same `EPOCHS` and `STEPS_PER_EPOCH` in every run, so every fraction gets the same updates and is validated at the same steps | Bornschein et al. 2020, arXiv:2009.12583, §3.3: an epoch is "always 1.28M/256 = 5000 gradient steps, regardless of the size of the actual training set used"; with fixed epochs the 1% run would get 100× fewer updates than the 100% run |
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
| `alpha` | 0.05, 0.1 and 0.15, every run | Cheung et al. 2026, COMPASS, Table 1; the constant `intervals.ALPHAS`, no flag |

## Open

| Value | Where | What settles it |
|---|---|---|
| `EPOCHS`, one value for every run | `slurm/full.env.example` | The pilot's seconds per update, against the 7-day `gpuq` limit and 13 runs |
| `EVAL_BATCH_SIZE` | `slurm/full.env.example` | The pilot's GPU peaks; changing it after an OOM refuses nothing |
| ConvNet channel widths | `train.py` | No source; the proposal only asks for a network without skip connections |
| Layers to diagnose, and the layer to calibrate | `fit_directions.py`, `compass_scores.py` | COMPASS §4.1: the explained variance of the Jacobians' first component, then a monotonicity sweep. Appendix E tried every layer of the head |
| SVM `C`, `class_weight`, pixels sampled per cell | `fit_directions.py` | No source: COMPASS takes its direction from a PCA of the Jacobians, and the SVM is this work's own |
| Steps and cells of the monotonicity sweep | `verify_monotonicity.py` | The sweep's time in the pilot; COMPASS gives no values (Appendix C) |
| `SEARCH_THRESHOLD_FEATURES` | `compass_scores.py` | Upstream's runner uses 1e-15, 60 bisection steps from 1000; the pilot's 1e-3 takes 20, a third of the cost |
| Score floor of the relative and local methods | `calibrate.py` | Romano et al. 2019 add γ to the denominator, set to 1 after dividing the response by its mean absolute value; COMPASS has no relative score |
