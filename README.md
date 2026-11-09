# garimpo-uncertainty

Replication package of a master's dissertation on calibrated prediction intervals for the area of
artisanal mining in Sentinel-2 imagery, evaluated under geographic domain shift. It trains
segmentation models, builds COMPASS and conformal intervals for the mined area of each cell, and
reports their coverage and width.

- 1,179 Sentinel-2 tiles cut into 18,864 cells of 512 x 512 pixels: 69.2 GB on disk at the 100%
  training fraction, 10.9 GB for the pilot. Counts per role are in
  [docs/runs.md](docs/runs.md#dataset-size).
- Four runs, all at 100% and all on a ResNet-18 encoder: a U-Net from scratch, from ImageNet
  weights and from Sentinel-2 MoCo weights, and smp's DeepLabV3 as the ConvNet.
- `compass/` (submodule) holds the adapted COMPASS method, `src/garimpo_uncertainty/` the satellite
  code, `scripts/` the stages, and `slurm/` the bash scripts that chain them. `outputs/` holds the
  versioned tables, including `split.csv`, the split used, and `tiles/`, the tiles each training
  fraction downloads.
- No model result exists yet. The code has run only under pytest, on synthetic data; the pipeline
  has never run on real scenes or on a GPU.

## Install

The experiment runs on Google Colab through `colab.ipynb`, which explains each of its cells.

1. On GitHub, create a fine-grained token for this repository with Contents: Read-only, since the
   repository is private.
2. In Colab, add it as the secret `GH_TOKEN`, with Notebook access on.
3. Open the notebook with File > Upload notebook.
4. In every session, run the cells of its "Session setup" in order. They mount Drive, clone the
   code, run `uv sync`, and restore the data and earlier runs from Drive. Skip the unpack cell in
   steps 1 and 5, which do not read the data.

Why the cells are written as they are: [docs/colab.md](docs/colab.md).

## Quick start

After the session setup, run the cells of one step of the notebook, in order:

| Step | Runtime | When |
|---|---|---|
| 1. Tests | GPU | Once, before the first paid run |
| 2. Data | CPU | Once; writes `data.tar.zst` to Drive |
| 3. Pilot | L4 GPU | Until all three variants pass |
| 4. Full run | GPU | Once per model, one model per session |
| 5. Merge | CPU | Once, after the four full runs pass |

Each step ends by releasing the runtime. Pilot results go to `runs/pilot/`, never to `outputs/`.
Reading the pilot's report and rerunning after a fix: [docs/pilot.md](docs/pilot.md). The full run:
[docs/full-run.md](docs/full-run.md).

## Documentation

| Document | Covers |
|---|---|
| [docs/colab.md](docs/colab.md) | Why the cells of `colab.ipynb` are written as they are: the Drive archive, one command per cell, modification times, the data step, releasing the runtime, the job verdicts, the copy every 30 minutes |
| [docs/pilot.md](docs/pilot.md) | Running the pilot, its check report, rerunning with `FORCE_FROM` |
| [docs/full-run.md](docs/full-run.md) | The full run step by step, downloading on your own machine, the run cells, regenerating tile lists, rerunning one stage |
| [docs/decisions.md](docs/decisions.md) | The values to fix before the full run, their sources, and where each is set |
| [docs/runs.md](docs/runs.md) | Dataset size by role, training fractions, tile lists, model variants, starting weights, layer names |
| [docs/design.md](docs/design.md) | What the model comparison can and cannot show, limits of the starting weights, the quantile rule and its sensitivity analysis |
| [docs/stages.md](docs/stages.md) | Every stage's node, inputs and outputs; statuses, skips, refusals, run identity, the data validation |
| [docs/data-layout.md](docs/data-layout.md) | The store, slot hashes, sparse files, run artifacts, resource measurements |
| [docs/outputs.md](docs/outputs.md) | Which script produces each number and table in `outputs/`, regenerating the split, metric rules |
| [docs/testing.md](docs/testing.md) | Running the E2E suite, its modules, what it does not cover |
| [docs/failure-modes.md](docs/failure-modes.md) | The 68 ways the pipeline can fail and what catches each |

## Development

```bash
uv run pytest -q                                       # E2E tests, artifacts in tests/artifacts/
uv run ruff format --check . && uv run ruff check .    # the pre-push checks
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
