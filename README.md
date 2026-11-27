# garimpo-uncertainty

Replication package for calibrated prediction intervals for the area of artisanal mining in
Sentinel-2 imagery, evaluated under geographic domain shift. It trains segmentation models, builds
COMPASS and conformal intervals for the mined area of each cell, and reports their coverage and
width.

- 1,179 Sentinel-2 tiles cut into 18,864 cells of 512 x 512 pixels: 69.2 GB on disk at the 100%
  training fraction, 10.9 GB for the pilot. Counts per role are in
  [docs/runs.md](docs/runs.md#dataset-size).
- Four runs, all at 100% and all on a ResNet-18 encoder: a U-Net from scratch, from ImageNet
  weights and from Sentinel-2 MoCo weights, and smp's DeepLabV3 as the ConvNet.
- `compass/` (submodule) holds the adapted COMPASS method, and `src/garimpo_uncertainty/` the
  satellite code: `stages/` the stages, each one a `garimpo <stage>` command, and `pipeline/` the
  job, `garimpo run`, that chains them. `outputs/` holds
  the versioned tables, including `split.csv`, the split used, and `tiles/`, the tiles each
  training fraction downloads.
- No model result exists yet. The code has run only under pytest, on synthetic data; the pipeline
  has never run on real scenes or on a GPU.

## Install

The experiment runs on Google Colab through `notebooks/01_run_experiment.ipynb`, which explains
each of its cells.

1. Open the notebook from GitHub, in every session, at
   <https://colab.research.google.com/github/meiazero/garimpo-uncertainty/blob/main/notebooks/01_run_experiment.ipynb>.
   A copy uploaded or saved to Drive keeps the cells of the day it was made.
2. Choose a GPU runtime, then run Runtime > Run all. The setup mounts Drive, clones the code,
   installs `requirements.txt`, and restores the data and earlier runs from Drive.

`uv.lock` holds Colab's version of every package Colab ships, so the install there replaces none of
them, except a few that a runtime a day or two behind that list still has older; `requirements.txt`
is its export, which the notebooks install with pip. Elsewhere:

```bash
git clone --recurse-submodules https://github.com/meiazero/garimpo-uncertainty.git
cd garimpo-uncertainty
uv sync
```

To move to a newer Colab runtime, lock against its published pip freeze, with the latest commit of
github.com/googlecolab/backend-info in place of `0cadcd7`, then export:

```bash
uv add --upgrade -c https://raw.githubusercontent.com/googlecolab/backend-info/0cadcd7/pip-freeze.gpu.txt --group dev --bounds exact pytest pytest-xdist ruff
uv export --format requirements-txt --no-hashes --group dev -o requirements.txt
```

Why the cells are written as they are: [docs/colab.md](docs/colab.md).

## Quick start

Runtime > Run all runs the notebook's sections in order, on one GPU runtime, with nothing to set. It
runs the tests once per commit, then the pilot, the same four models on 1%, until all four pass on
the commit checked out, then the four full runs, which later sessions continue on their own:

| Section | What it does |
|---|---|
| 1. Setup | Mounts Drive, clones the code, builds the environment |
| 2. Data | Unpacks the data once and restores the earlier runs and logs |
| 3. Redo after a fix | The one option: a stage the full runs redo after a code fix |
| 4. Tests | Runs the suite on the GPU, once per commit |
| 5. Runs | The four models through every stage: the pilot on 1% of the training tiles until all four pass, then the full runs on 100% |
| 6. Merge | Merges the four runs' tables of each scale that ran, into `outputs/` for the full runs |
| 7. Release | Writes the pending files to Drive and releases the GPU |

The notebook downloads nothing: download the data on your own machine and put the archive in Drive
first ([docs/full-run.md](docs/full-run.md#download-the-store)). If the session ends early, run
Runtime > Run all again in a new one: finished stages are skipped and training resumes. Pilot
runs and result tables go to `runs/pilot/`, never to `outputs/`.
Reading the pilot's report and rerunning after a fix: [docs/pilot.md](docs/pilot.md). The full run:
[docs/full-run.md](docs/full-run.md).

## Sizing the runtime

`notebooks/03_tune_hardware.ipynb` measures, on synthetic cells and without Drive, what limits each
hot path of the pipeline on the session's GPU, CPUs and disk. It times data loading, training, the
COMPASS searches, the eval batch size, the float32 precision and LightGBM's threads. Its verdict
names the smallest setting that is enough and the full run's hours on that machine. A `Memo` score
or a strict float32 result that loses a bit fails it; another candidate that moves a bit is
reported for a decision. Open it at
<https://colab.research.google.com/github/meiazero/garimpo-uncertainty/blob/main/notebooks/03_tune_hardware.ipynb>
on an L4 runtime at the latest runtime version, then run Runtime > Run all.

## Documentation

| Document | Covers |
|---|---|
| [docs/colab.md](docs/colab.md) | Why the cells of the run notebook are written as they are: the Drive archive, cells that run again safely, the gates, modification times, releasing the runtime, the job verdicts, the copy every 30 minutes |
| [docs/pilot.md](docs/pilot.md) | Running the pilot, its check report, rerunning with `--force-from` |
| [docs/full-run.md](docs/full-run.md) | The full run step by step, the run configs, downloading on your own machine, the run cells, regenerating tile lists, rerunning one stage |
| [docs/decisions.md](docs/decisions.md) | The values to fix before the full run, their sources, and where each is set |
| [docs/runs.md](docs/runs.md) | Dataset size by role, training fractions, tile lists, model variants, starting weights, layer names |
| [docs/design.md](docs/design.md) | What the model comparison can and cannot show, limits of the starting weights, the quantile rule and its sensitivity analysis |
| [docs/stages.md](docs/stages.md) | Every stage's node, inputs and outputs; statuses, skips, refusals, run identity, the data validation |
| [docs/data-layout.md](docs/data-layout.md) | The store, slot hashes, sparse files, run artifacts, resource measurements |
| [docs/outputs.md](docs/outputs.md) | Which script produces each number and table in `outputs/`, regenerating the split, metric rules |
| [docs/figures.md](docs/figures.md) | The dissertation's figures: drawing them from finished runs, why each exists, what its caption must say |
| [docs/testing.md](docs/testing.md) | Running the E2E suite, its modules, what it does not cover |
| [docs/failure-modes.md](docs/failure-modes.md) | The 70 ways the pipeline can fail and what catches each |

## Development

```bash
pytest -q                                # E2E tests, artifacts in tests/artifacts/
ruff format --check . && ruff check .    # the pre-push checks
git config core.hooksPath .githooks                    # once per clone, enables the pre-push hook
```

- CI (`.github/workflows/ci.yml`) runs ruff and the E2E suite on every pull request and on every
  push to `main`.
- The pre-push hook checks the working tree, not the pushed commits, so push from a clean tree.

## Attribution and licenses

- COMPASS (MIT, `compass/LICENSE`): Cheung, Veeraraghavan and Balakrishnan, COMPASS: Robust
  Feature Conformal Prediction for Medical Segmentation Metrics, ICLR 2026, arXiv:2509.22240. Code
  adapted from <https://github.com/matthewyccheung/compass>.
- Dataset (CC BY-SA 4.0): Jasansky et al., Global ML-ready dataset for mining areas in satellite
  images, Zenodo, 2024, DOI 10.5281/zenodo.14195737.
- Imagery: modified Copernicus Sentinel data, through Microsoft Planetary Computer.
- Starting weights: `sentinel2_moco` is `cc-by-4.0` and `imagenet` is `other` on Hugging Face; see
  [docs/runs.md](docs/runs.md#starting-weights).
