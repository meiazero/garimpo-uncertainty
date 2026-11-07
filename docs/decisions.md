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
| Layer to calibrate; layers to diagnose | `fit_directions.py`, `compass_scores.py` |
| SVM `C`, `class_weight`, pixels sampled per cell | `fit_directions.py` |
| Population: every cell, or cells with mining (`--cells`) | `calibrate.py` |
| Score floor of the relative and local methods | `calibrate.py` |
| Boundary F1 tolerance in pixels | `compute_metrics.py` |
| Steps and cells of the monotonicity sweep; its `beta_max` is the search range | `verify_monotonicity.py` |
| Search range and threshold (defaults are upstream's) | `compass_scores.py` |
| Resampling of B11 and B12 to 10 m | `download_data.py scenes` |
