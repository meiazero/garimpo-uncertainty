# Refactor map: the training pipeline and compass2d follow the project's architecture guidelines, every variation is chosen in config, and the notebooks keep only the Colab session

Status: approved · Base: `2388cc5` (compass `3b46e76`) · Scope: training pipeline (`models/`, `stages/train.py`, `cell_dataset.py`, post-processing), `compass/compass2d`, `notebooks/` · Additions: Comments and Tests, as review rules (below)

Guidelines (given by the user on 2026-11-27): Registry + Factory, Strategy, Pipeline/Builder, Template Method + Callbacks, Adapter. Every guideline is implemented; a variation the protocol does not use ships off in `configs/base.json`.

Hard constraints on every fix:
- The default config trains bitwise the same weights as `2388cc5`, at any worker count and across a resume (`tests/e2e/test_workers.py`, `test_resume.py`, `test_training.py`).
- Checkpoints and `directions.npz` already on Drive keep loading: same state_dict keys, same layer names, same npz fields.
- compass: work on a `dev` branch of the submodule repo; its `main` is never touched. garimpo's submodule pointer moves to the `dev` commit.

## Review rules (Comments, Tests)

After each fix, before its push, one agent reviews the diff against two rules and reports; fixes stay in the main session.
- Comments: remove comments that restate the code; keep those about something outside the code.
- Tests: remove tests that would not catch a real bug; fix tests that fail without one.
- Reviewer: `gemini-4-hydrogen` (Antigravity) and `gpt-6.2-sol` (Codex), both listed by the orchestrator on 2026-11-27. When neither is available, a blind Sonnet at high effort.

## Fixes: training pipeline

- [x] 1. Give each architecture an adapter registered by name (`@register_model("unet")`) that builds it from config, lists its COMPASS layers and splits it into f and g, so no caller branches on the architecture. (6 files)
  - `models/common.py:42` `build` branches `if arch == "convnet" / "unet"`.
  - `models/common.py:84` `split_at` branches on `isinstance(model, DeepLabV3)`.
  - `models/common.py:17` `FEATURE_LAYER` is one regex for both models, which must "never match the `decoder` both models have".
  - `models/common.py:19` and `:30` (`check_encoder_weights`, `variant_name`) are two more `if arch ==` chains.
  - `stages/train.py:512` `--arch choices=["convnet", "unet"]` repeats the list by hand.
  - `NUM_CLASSES` comes from config (default 1); the COMPASS stages refuse anything but 1 with a clear error.
- [x] 2. Put losses, optimizers and training metrics in `LOSS_REGISTRY`, `OPTIMIZER_REGISTRY` and `METRIC_REGISTRY`, each loss an `nn.Module`, so `LOSS`, `OPTIMIZER` and the logged metrics come from config. (4 files)
  - `stages/train.py:60` `build_loss` is an `if name ==` chain, and `dice_bce` is a lambda.
  - `stages/train.py:548` `--loss choices=[...]` repeats the names by hand.
  - `stages/train.py:236` builds `AdamW` directly.
  - `stages/train.py:74` `batch_scores` computes Dice, precision and recall inline in `validate`.
- [x] 3. Put post-processing behind one `POSTPROCESS_REGISTRY` of pure `(logits: Tensor) -> Tensor` functions (`sigmoid_threshold` for 1 class, `softmax_argmax` for more), so the 0.5 decision is written once. (5 files)
  - `stages/train.py:76` decides `logits >= 0`, `evaluation.py:20` decides `u8 >= 128`, `stages/infer.py:108` calls `torch.sigmoid`, and `compass2d/metrics.py:11` decides `logits >= 0` again.
- [ ] 4. Make `CellDataset` only read `(image, mask)` as reflectance and take a `transform` built from config (`TRANSFORMS: [normalize, flips]`), with an optional `albumentations` entry. (4 files)
  - `cell_dataset.py:69-88` reads, converts DN to reflectance, z-scores and flips in one `__getitem__`.
  - The `augment` flag (`cell_dataset.py:52`) and `normalization=None` (`:50`) select steps inside the reader.
  - `flips` keeps drawing in `FlippedDraws`, in the main process (`cell_dataset.py:17`, commit `5fcfc42`). `albumentations` draws inside the workers, so its weights change with the worker count; a run that picks it records that in `run_meta.json`.
- [ ] 5. Make training a fixed loop (`train_step`, `val_step`, `on_epoch_end`) that calls callbacks built from config: `CheckpointCallback` (best by `val_dice` plus `last.pt`), `ConsoleLogCallback` (history.csv and the epoch line), `EarlyStoppingCallback` (`EARLY_STOPPING: null` by default). (4 files, after 2)
  - `stages/train.py:258` `fit_epoch` mixes the update with the data-wait clock and the gradient norms.
  - `stages/train.py:298` `save_epoch` writes history, `best_model.pt` and `last.pt` inside the epoch loop.
  - `stages/train.py:382` `run_epochs` builds the log line inside the loop.
  - Off by default because the protocol trains a fixed number of updates at every fraction (`--steps-per-epoch`).

## Fixes: compass2d (on `dev`)

- [ ] 6. Make each direction a strategy in a `DIRECTION_REGISTRY` (`pca`, `svm`) with `fit` and `deltas`, so a new direction is one class. (3 files + 2 in garimpo)
  - `compass2d/directions.py:108-113` `cell_deltas` branches `if direction == "pca" / "svm"`.
  - `compass2d/directions.py:15` `Directions` holds the fields of both strategies in one dataclass; `save`/`load` keep writing the same npz fields.
- [ ] 7. Make the searches strategies in a `SEARCH_REGISTRY` (`symmetric`, `asymmetric`) with one signature, so garimpo picks one by method name. (2 files + 2 in garimpo)
  - `compass2d/compass.py:60` and `:79` are two search functions with different return shapes.
  - `stages/calibrate.py:331-338` branches `if method == "independent_pixels" / in COMPASS_METHODS / in ASYMMETRIC_METHODS`.
  - `compass_j.py:107` keeps its own memoized search "equal to the bit" to compass2d's: a copy that nothing links. It becomes a strategy in the same registry.
- [ ] 8. Put the quantile rules and area metrics in `QUANTILE_RULE_REGISTRY` and `METRIC_REGISTRY` of compass2d, so garimpo names them in config. (3 files + 1 in garimpo)
  - `stages/calibrate.py:65-73` picks the rule by `if situation == "external" / "own_region"` and `calibration == "external_weighted"`.
  - `compass2d/metrics.py:9` and `:14` (`hard_area`, `soft_area`) are chosen by caller code.

## Fixes: notebooks keep the Colab session, the package keeps the logic

- [ ] 9. Make `garimpo tune` a command backed by a `garimpo_uncertainty/tuning/` package, so `03_tune_hardware.ipynb` is a form and one `!garimpo tune` line. (8 files, after 1-5)
  - `notebooks/tuning.py` is imported through `sys.path.insert(0, CODE / "notebooks")` (03, cell 4); no test covers it.
  - The measurement loops (03, cells 9-19) and the verdict with its thresholds (03, cell 21) live in cells; 20 commits touched `notebooks/` from 2026-11-24 to 2026-11-26, 5 of them `fix(notebook)`.
  - `%%section`, `FAILED`, `RESULTS`, `exec(cell, user_ns)`, `loaded_and_replaced()` and `site.addsitedir` (03, cell 4) exist only because the logic runs in the kernel.
  - `EVALUATED_CELLS = 16 * ...` (03, cell 21) hardcodes what `notebooks/tuning.py:51` derives as `CELLS_PER_TILE`.
- [ ] 10. Make `garimpo experiment` run pilot then full, the merge and the status (01, cells 10-12), and let `garimpo figures` choose the scale and the runs with intervals (02, cell 6). (5 files)
  - `next_scale()` (01, cell 10) and 02 cell 6 parse the last stdout line of `garimpo status`.
  - `MODELS` is typed by hand in 01 cell 6 and 02 cell 6, a copy of `configs/models/`.
  - 02 cell 6 rebuilds `RunConfig.run_dir` by hand.
- [ ] 11. Leave in the notebooks only the Colab session: mount, clone at `BRANCH`, install, unpack, rsync to and from Drive, the 30-minute copy, the test gate's Drive marker, release. One identical bootstrap cell and one identical release cell in all three. (3 files, after 9-10)
  - 01 and 02 never update an existing clone; 03 fetches and checks out `BRANCH` (03, cell 4).
  - `release_runtime()` is copied in 01 and 02; 03 cell 23 releases without flushing Drive.
- [ ] 12. Point `docs/` and `README.md` at the new registries, config keys and commands, and delete `notebooks/tuning.py`. (6 files, after 1-11)

## Special cases kept

- `batches()` (`cell_dataset.py:114`) keeps the Ctrl+C that lands while the workers fork.
- An interrupt stops the models after the current one (01, cell 10, `INTERRUPTED`): moves into `garimpo experiment` as is.
- 01 unpacks the data to `/content/garimpo-data`, 02 reads it in place in Drive.
