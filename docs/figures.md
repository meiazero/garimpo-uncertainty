# Figures

`garimpo figures` draws the dissertation's figures from finished runs, one PDF and one SVG each.
Every number comes from the runs' files, so a figure is redrawn, never edited.

```bash
garimpo figures runs/*-f1.0 --out outputs/figures
garimpo figures runs/*-f1.0 --out outputs/figures --only q2_coverage_and_width
garimpo figures --scale auto --models unet_scratch convnet --out outputs/figures
```

With `--scale`, a model's run is drawn once every stage of it is ok, or, with `--only
qualitative_cells`, once it has intervals; the others are named and left out. Run it from the
repository root, with `GARIMPO_DATA_DIR` set as for the stages: the qualitative
figure and every figure that reads the cell table open the data root. Each figure is built at the
dissertation's text width, 16 cm, so it goes in with `\includegraphics[width=\linewidth]` and is
not rescaled. The audit line printed after each figure must read `clean`.

On Colab, `notebooks/02_compare_segmentation.ipynb` draws `qualitative_cells` from the runs in
Drive, on the real cells of the unpacked store in `MyDrive/garimpo/garimpo-dataset/`, read in
place, shows it, and copies it to `MyDrive/garimpo/figures/`.

Every file a figure reads is written by a stage of `garimpo run`, so the figures draw on a CPU,
from a copy of the runs, without the checkpoint. `search_convergence` and `threshold_tradeoff`
read `search_profile.npz`, which the last stage measures on the run's GPU, since the times are that
GPU's. A run finished before that stage existed gets it by running `garimpo run` again; until then
it is left out of those two figures, with a warning. `monotonicity_curves` likewise leaves out a
run whose `layer_monotonicity` predates its curves: a rerun skips that stage, and only
`--force-from layer_monotonicity`, which redoes every later stage, writes them.

`garimpo data_figures` draws the figures of the data itself, from the cell store and the split
alone, before any run: `split_composition` (cells per role and region), `area_distribution` (the
share of empty cells per role and the spread of mining areas), `example_cells` (per region, the
evaluation cell at the median mining area, with its reference outlined), `tile_map` (each tile's
center from its sidecar, the artisanal ones by role over the industrial training tiles, on Natural
Earth's 1:110m land, `geojson/ne_110m_land.geojson` at tag v5.1.2 of github.com/nvkelso/natural-earth-vector, public domain by naturalearthdata.com's terms of use, vendored as `figures/ne_110m_land.geojson`) and
`tile_cells` (the evaluation tile whose cells with mining are nearest half of its 16, with its
reference, the cell grid and each cell's area).

```bash
garimpo data_figures --out outputs/figures
```

The dissertation's diagrams, the protocol figure among them, are not drawn here: they live in
their own repository, `dissertation-figures`, beside this one, so a figure change never changes
the commit the experiment runs on.

## Each figure and why it exists

A figure earns its place by showing something the text's table cannot. The section is the
dissertation's; the source is the run file it reads.

| Figure | What it shows that a table cannot | Section | Source |
|---|---|---|---|
| `q1_area_error_and_boundary` | The whole spread behind Q1's single Spearman correlation, including cells with a near-perfect boundary and a large area error | Results, Q1 (`graf:q1`) | `metrics/cell_metrics.npz`, `predictions/cells.npz`, `q1_spearman.csv` |
| `q2_coverage_and_width` | Coverage against width as a trade-off: which methods reach the target and, among those, which is narrowest | Results, Q2 (`graf:q2`) | `results.csv` |
| `q3_calibration_situations` | Per method, coverage calibrated in the test's own region, in the others, and in the others weighted: the cost of the shift and what weighting recovers | Results, Q3 (`graf:q3`) | `results.csv` |
| `coverage_by_area_band` | Whether the marginal coverage holds for empty cells and for the largest areas alike | Results, COMPASS diagnostics (`graf:faixas`) | `coverage_by_stratum.csv` |
| `coverage_by_region_and_prediction` | Coverage per test region and per predicted mining, the only stratum known when an interval is used | Results, COMPASS diagnostics | `coverage_by_stratum.csv` |
| `quantile_rule` | Whether coverage, width or ranking change when repeated subsampling replaces pooled scores | Results, quantile rule sensitivity | `results.csv` |
| `paired_differences` | The interval of each method's difference from the conformal simple, from the same resamples: overlapping intervals do not mean two methods are alike | Results, Q2 | `coverage_comparisons.csv`, `width_comparisons.csv` |
| `coverage_by_level` | Whether each method holds its coverage at every nominal level, not only at 90% | Results, Q2 | `results.csv` |
| `intervals_by_cell` | Which test cells pay for the average coverage and width: every interval, in order of reference area, misses marked | Results, COMPASS diagnostics | `intervals/intervals.npz`, `predictions/cells.npz` |
| `score_relation` | Why COMPASS intervals are narrower: their scores grow slower than the area error (Cheung et al., Figure 4, on this data) | Results, COMPASS diagnostics | `scores/scores.npz`, `score_relation.csv` |
| `layer_comparison` | Why the chosen layer was chosen, and what each other layer would have given | Results, COMPASS diagnostics | `directions/diagnostics.csv`, `monotonicity_layer.csv`, each layer's `results.csv` |
| `monotonicity_curves` | What Algorithm 2 sees: the area of each swept cell as beta grows, where it saturates and how many cells never move | Results, COMPASS diagnostics | `monotonicity_layer_curves.npz` |
| `density_ratio` | How far the weighted calibration leans on a few calibration cells, with the effective number of groups | Results, Q3 | `calibration/quantiles.npz`, `status/calibrate.json` |
| `training_curves` | That each checkpoint is the epoch with the best validation Dice, and whether training stopped while still improving | Experiments, training | `history.csv` |
| `qualitative_cells` | What a cell, its reference and each model's probability look like, with the intervals of two methods under each | Results, data and segmentation | `images.npy`, `masks.npy`, `predictions/test.npy`, `intervals/intervals.npz` |
| `coverage_by_latent_distance` | Whether coverage falls on the test cells the calibration represents least, per quintile of the density ratio, and whether weighting recovers it | Results, Q3 | `calibration/quantiles.npz`, `intervals/intervals.npz` |
| `latent_shift_map` | The geographic shift in the chosen layer's features, and which calibration cells the weighting leans on | Results, Q3 | `scores/scores.npz` (`feature_sums`), `calibration/quantiles.npz` |
| `jacobian_geometry` | That the Jacobians lie close to one line, the premise of one COMPASS-J direction, and the angle between the PCA and SVM directions | Methods or COMPASS diagnostics | `scores/scores.npz` (`jacobians`), `directions/<layer>.npz` |
| `sensitivity_and_saturation` | Whether the cells whose search saturates are those where the area barely responds to the direction | COMPASS diagnostics | each layer's `scores/scores.npz`, `directions/<layer>.npz` |
| `saturation_by_layer` | That the search saturates at the layers the rule rejects, not at the chosen one | COMPASS diagnostics | each layer's `status/compass_scores.json` |
| `cost_by_layer` | What evaluating each layer costs, the beta search above all | Experiments, runs | each layer's `status/compass_scores.json`, `status/generate_intervals.json` |
| `search_convergence` | After how many bisection steps the interval stops changing | Appendix | `search_profile.npz` |
| `threshold_tradeoff` | What a coarser search threshold saves in time and costs in interval precision | Appendix | `search_profile.npz` |

The qualitative figure picks its cells from the reference alone, before any model is read: the
first empty test cell in cell order, then the mining cells at the quartiles and the maximum of
reference area. The true-color stretch is the 2nd to 98th percentile of the cells shown, the same
for every row.

## What each caption must say

Every model is a single training, and every interval is the 95% group bootstrap of
`compute_metrics`. A caption leads with the finding, then states both facts, the calibration
shown, and the level, which is alpha = 0.1 unless the figure is `coverage_by_level`.

## A new method

A method that `garimpo_uncertainty.intervals.COMBINATIONS` gains appears in every figure without
a change here. It takes its family's hue from its prefix (`conformal_`, `cqr`, `compass_l`,
`compass_j_`, any other `compass`) and a shade of its own; an `a_` prefix draws it hollow, as the
asymmetric version. Give it a Portuguese name in `METHOD_NAMES` of
`src/garimpo_uncertainty/figures/common.py`, or it keeps its code name. A method with another
prefix is drawn in cyan.
