# garimpo-uncertainty

Replication package of a master's dissertation: calibrated prediction intervals for artisanal mining
area from Sentinel-2, evaluated under geographic domain shift.

- `compass/` (submodule) holds the adapted COMPASS method, `src/garimpo_uncertainty/` the satellite
  code, `scripts/` the stages, and `slurm/` the cluster jobs.
- `outputs/` holds the versioned tables, including `split.csv`, the split used, and `tiles/`, the
  tiles each training fraction downloads.
- No model result exists yet. The code has run only under pytest, on synthetic data; the pipeline has
  never run on real scenes or on the cluster.

Attribution and licenses: [below](#attribution-and-licenses).

## Install

On the cluster's login node, since compute nodes have no internet:

```bash
git clone --recurse-submodules <ROOT_REPO_URL> garimpo-uncertainty
cd garimpo-uncertainty
uv sync
git config core.hooksPath .githooks
```

The data root is `$GARIMPO_DATA_DIR`, default `~/Experiments/datasets/garimpo`.

## Quick start

The pilot, the 1% training fraction, on the login node:

```bash
bash slurm/pilot.sbatch download
ENCODER_WEIGHTS=imagenet bash slurm/pilot.sbatch download           # data, then that variant's weights
ENCODER_WEIGHTS=sentinel2_moco bash slurm/pilot.sbatch download
slurm/submit.sh slurm/pilot.sbatch                                  # U-Net from scratch
ENCODER_WEIGHTS=imagenet slurm/submit.sh slurm/pilot.sbatch
ENCODER_WEIGHTS=sentinel2_moco slurm/submit.sh slurm/pilot.sbatch
tail -f outputs/logs/garimpo-pilot-<JOBID>.out
FORCE_FROM=calibrate slurm/submit.sh slurm/pilot.sbatch             # after a code fix: redo from a stage
```

Results go to `runs/pilot/`, never to `outputs/`. How a run goes and how to resubmit:
[docs/pilot.md](docs/pilot.md).

## Documentation

| Document | Covers |
|---|---|
| [docs/pilot.md](docs/pilot.md) | The pilot run, its check report, resubmitting with `FORCE_FROM` |
| [docs/full-run.md](docs/full-run.md) | The full run step by step, downloading a larger fraction, regenerating tile lists, running outside the cluster, rerunning one stage |
| [docs/runs.md](docs/runs.md) | Training fractions, tile lists, model variants, starting weights and their limits |
| [docs/decisions.md](docs/decisions.md) | The values to fix before the full run, and where each is set |
| [docs/stages.md](docs/stages.md) | Every stage's node, inputs and outputs; skips, refusals and the data validation |
| [docs/data-layout.md](docs/data-layout.md) | The store, slot hashes, sparse files, run artifacts, resource measurements |
| [docs/outputs.md](docs/outputs.md) | Which script produces each number and table in `outputs/` |
| [docs/testing.md](docs/testing.md) | Running the E2E suite, its modules, what it does not cover |
| [docs/failure-modes.md](docs/failure-modes.md) | The 45 ways the pipeline can fail and what catches each |

## Development

```bash
slurm/submit.sh slurm/cpu.sbatch -m pytest -q          # E2E tests, artifacts in tests/artifacts/
uv run ruff format --check . && uv run ruff check .    # the pre-push checks
```

## Attribution and licenses

- **COMPASS** (MIT, `compass/LICENSE`): Cheung, Veeraraghavan and Balakrishnan, COMPASS: Robust
  Feature Conformal Prediction for Medical Segmentation Metrics, ICLR 2026, arXiv:2509.22240. Code
  adapted from <https://github.com/matthewyccheung/compass>.
- **Dataset** (CC BY-SA 4.0): Jasansky et al., Global ML-ready dataset for mining areas in satellite
  images, Zenodo, 2024, DOI 10.5281/zenodo.14195737.
- **Imagery**: modified Copernicus Sentinel data, through Microsoft Planetary Computer.
