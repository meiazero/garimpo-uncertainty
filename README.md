# ⛏️ garimpo-uncertainty

Replication package for calibrated prediction intervals for the area of artisanal mining in
Sentinel-2 imagery, evaluated under geographic domain shift. It trains segmentation models, builds
COMPASS and conformal intervals for the mined area of each cell, and reports their coverage and
width.

- 1,179 Sentinel-2 tiles cut into 18,864 cells of 512 x 512 pixels: 69.2 GB on disk at the 100%
  training fraction, 10.9 GB for the pilot. Counts per role are in
  [docs/runs.md](docs/runs.md#-dataset-size).
- Four runs, all at 100% and all on a ResNet-18 encoder: a U-Net from scratch, from ImageNet
  weights and from Sentinel-2 MoCo weights, and smp's DeepLabV3 as the ConvNet.
- `compass/` (submodule) holds the adapted COMPASS method, `src/garimpo_uncertainty/` the satellite
  code, `scripts/` the stages, and `scripts/run.py` the runner that chains them. `outputs/` holds
  the versioned tables, including `split.csv`, the split used, and `tiles/`, the tiles each
  training fraction downloads.
- No model result exists yet. The code has run only under pytest, on synthetic data; the pipeline
  has never run on real scenes or on a GPU.

## 📦 Install

The experiment runs on Google Colab through `colab.ipynb`, which explains each of its cells.

1. Open the notebook from GitHub, in every session, at
   <https://colab.research.google.com/github/meiazero/garimpo-uncertainty/blob/main/colab.ipynb>.
   A copy uploaded or saved to Drive keeps the cells of the day it was made.
2. Choose a GPU runtime, then run Runtime > Run all. The setup mounts Drive, clones the code, runs
   `uv sync`, and restores the data and earlier runs from Drive.

Why the cells are written as they are: [docs/colab.md](docs/colab.md).

## 🚀 Quick start

Runtime > Run all runs the notebook's sections in order, on one GPU runtime. Its `SCALE` option
runs the pilot, the same four models on 1%, until a full run has a checkpoint; choose `full` once
the pilot passes, and later sessions continue the full runs on their own:

| Section | What it does |
|---|---|
| 1. Setup | Mounts Drive, clones the code, builds the environment |
| 2. Data | Unpacks the data and restores the earlier runs and logs |
| 3. Run configs | Chooses the scale with `SCALE` and shows the values of every run |
| 4. Tests | Runs the suite on the GPU (optional) |
| 5. Training | The four models: on 1% of the training tiles for the pilot, on 100% for the full runs |
| 6. Evaluation | Merges the four runs' tables, into `outputs/` for the full runs |
| 7. Release | Writes the pending files to Drive and releases the GPU |

The notebook downloads nothing: download the data on your own machine and put the archive in Drive
first ([docs/full-run.md](docs/full-run.md#-download-the-store)). If the session ends early, run
Runtime > Run all again in a new one: finished stages are skipped and training resumes. Pilot
runs and result tables go to `runs/pilot/`, never to `outputs/`.
Reading the pilot's report and rerunning after a fix: [docs/pilot.md](docs/pilot.md). The full run:
[docs/full-run.md](docs/full-run.md).

## 📚 Documentation

| Document | Covers |
|---|---|
| [docs/colab.md](docs/colab.md) | Why the cells of `colab.ipynb` are written as they are: the Drive archive, one command per cell, modification times, releasing the runtime, the job verdicts, the copy every 30 minutes |
| [docs/pilot.md](docs/pilot.md) | Running the pilot, its check report, rerunning with `--force-from` |
| [docs/full-run.md](docs/full-run.md) | The full run step by step, the run configs, downloading on your own machine, the run cells, regenerating tile lists, rerunning one stage |
| [docs/decisions.md](docs/decisions.md) | The values to fix before the full run, their sources, and where each is set |
| [docs/runs.md](docs/runs.md) | Dataset size by role, training fractions, tile lists, model variants, starting weights, layer names |
| [docs/design.md](docs/design.md) | What the model comparison can and cannot show, limits of the starting weights, the quantile rule and its sensitivity analysis |
| [docs/stages.md](docs/stages.md) | Every stage's node, inputs and outputs; statuses, skips, refusals, run identity, the data validation |
| [docs/data-layout.md](docs/data-layout.md) | The store, slot hashes, sparse files, run artifacts, resource measurements |
| [docs/outputs.md](docs/outputs.md) | Which script produces each number and table in `outputs/`, regenerating the split, metric rules |
| [docs/testing.md](docs/testing.md) | Running the E2E suite, its modules, what it does not cover |
| [docs/failure-modes.md](docs/failure-modes.md) | The 68 ways the pipeline can fail and what catches each |

## 🔧 Development

```bash
uv run pytest -q                                       # E2E tests, artifacts in tests/artifacts/
uv run ruff format --check . && uv run ruff check .    # the pre-push checks
git config core.hooksPath .githooks                    # once per clone, enables the pre-push hook
```

- CI (`.github/workflows/ci.yml`) runs ruff and the E2E suite on every pull request and on every
  push to `main`.
- The pre-push hook checks the working tree, not the pushed commits, so push from a clean tree.

## 🌈 Inclusion

🏳️‍🌈 🏳️‍⚧️ This project supports LGBTQIA+ people. Everyone is welcome to use it, open issues and
contribute, whatever their gender identity, gender expression or sexual orientation.

- Address people by the names and pronouns they use.
- Write gender-neutral text, as the documentation already does: "you" for the reader, "they" for a
  person whose pronouns you do not know.
- Harassment and discrimination for gender identity, gender expression or sexual orientation are
  not accepted in issues, pull requests or reviews.

## 📜 Attribution and licenses

- COMPASS (MIT, `compass/LICENSE`): Cheung, Veeraraghavan and Balakrishnan, COMPASS: Robust
  Feature Conformal Prediction for Medical Segmentation Metrics, ICLR 2026, arXiv:2509.22240. Code
  adapted from <https://github.com/matthewyccheung/compass>.
- Dataset (CC BY-SA 4.0): Jasansky et al., Global ML-ready dataset for mining areas in satellite
  images, Zenodo, 2024, DOI 10.5281/zenodo.14195737.
- Imagery: modified Copernicus Sentinel data, through Microsoft Planetary Computer.
- Starting weights: `sentinel2_moco` is `cc-by-4.0` and `imagenet` is `other` on Hugging Face; see
  [docs/runs.md](docs/runs.md#-starting-weights).
