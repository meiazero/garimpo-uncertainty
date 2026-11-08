# Values to decide before the full run

Read before step 5 of [full-run.md](full-run.md).

Nothing fixes these yet, so `slurm/full.sbatch` stops when one is missing. Pilot values
are not suggestions.

| Value | Where |
|---|---|
| Channel widths (ConvNet), epochs, batch size, learning rate, weight decay, loss, seed | `train.py` |
| Fixed epochs or fixed updates across fractions; `EPOCHS` is set per run either way | `slurm/full.env.example` |
| Subset seed; one value keeps the fractions nested; another than the versioned lists' 0 means regenerating them | `download_data.py lists`, `train.py` |
| Starting weights and fraction: one env file per run | `train.py`, `slurm/full.env.example` |
| `EVAL_BATCH_SIZE`, for every stage after training; changing it after an OOM refuses nothing | `slurm/full.env.example` |
| Layers to diagnose (`LAYERS`) | `fit_directions.py` |
| Direction temperature (`--direction-temperature`): the sigmoid temperature of the area whose Jacobian sets the COMPASS-J direction, apart from the metric's 1e4. At 1e4 the gradient vanishes beyond \|logit\| ≈ 5e-3, so cells predicted empty get the PCA mean as direction. Choose it by leave-one-group-out on calibration; the same value in all three stages | `fit_directions.py`, `verify_monotonicity.py`, `compass_scores.py` |
| Share of validation cells whose sweep must be ascending, in both directions, for a layer to be chosen (`MIN_FRACTION_ASCENDING`). Cheung et al. (2026) set none: §4.1 only says to verify monotonicity, and §2.3 reports it held in all their experiments. Read the pilot's `monotonicity_layer.csv` before fixing it. Too high, and `choose_layer` stops the job naming the closest layer; resubmitting with another value redoes only `choose_layer`, since the stages before it skip | `choose_layer.py` |
| SVM `C`, `class_weight`, pixels sampled per cell | `fit_directions.py` |
| Population: every cell, or cells with mining (`--cells`) | `calibrate.py` |
| Score floor of the relative and local methods | `calibrate.py` |
| Boundary F1 tolerance in pixels | `compute_metrics.py` |
| Steps and cells of the monotonicity sweep; its `beta_max` is the search range | `verify_monotonicity.py` |
| Search range and threshold (defaults are upstream's) | `compass_scores.py` |
| Resampling of B11 and B12 to 10 m | `download_data.py scenes` |
