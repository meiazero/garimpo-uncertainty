# garimpo-uncertainty

Replication package of a master's dissertation: calibrated prediction intervals for artisanal mining
area from Sentinel-2, evaluated under geographic domain shift.

- `compass/` (submodule) holds the adapted COMPASS method, `src/garimpo_uncertainty/` the satellite
  code, `scripts/` the stages, and `slurm/` the bash scripts that chain them.
- `outputs/` holds the versioned tables, including `split.csv`, the split used, and `tiles/`, the
  tiles each training fraction downloads.
- The data are 1,179 Sentinel-2 tiles cut into 18,864 cells of 512 x 512 pixels, 69.2 GB on disk
  at the 100% training fraction and 10.9 GB for the pilot; per role in
  [docs/runs.md](docs/runs.md#dataset-size).
- No model result exists yet. The code has run only under pytest, on synthetic data; the pipeline has
  never run on real scenes or on a GPU.

Attribution and licenses: [below](#attribution-and-licenses).

## Install

On a RunPod pod with a network volume at `/workspace`. Keep the clone and the data there: the
container disk is erased when the pod stops. The repository is private, so clone with a GitHub token.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                     # again on every new pod
cd /workspace
git clone --recurse-submodules <ROOT_REPO_URL> garimpo-uncertainty
cd garimpo-uncertainty
uv sync
git config core.hooksPath .githooks
export GARIMPO_DATA_DIR=/workspace/garimpo-data                     # in every new shell
```

The data root is `$GARIMPO_DATA_DIR`; its default, `~/Experiments/datasets/garimpo`, is on the
container disk of a pod.

## Quick start

The pilot, the 1% training fraction. The download needs no GPU: where the volume's data center
offers CPU pods with network volumes, one can do it while no GPU is billed.

```bash
bash slurm/pilot.sbatch download
ENCODER_WEIGHTS=imagenet bash slurm/pilot.sbatch download           # data, then that variant's weights
ENCODER_WEIGHTS=sentinel2_moco bash slurm/pilot.sbatch download
```

Then on the GPU pod, under `nohup` so a dropped SSH session does not stop it:

```bash
mkdir -p outputs/logs
nohup bash slurm/pilot.sbatch > outputs/logs/pilot-none.log 2>&1 &  # U-Net from scratch
tail -f outputs/logs/pilot-none.log
ENCODER_WEIGHTS=imagenet nohup bash slurm/pilot.sbatch > outputs/logs/pilot-imagenet.log 2>&1 &
ENCODER_WEIGHTS=sentinel2_moco nohup bash slurm/pilot.sbatch > outputs/logs/pilot-sentinel2_moco.log 2>&1 &
FORCE_FROM=calibrate nohup bash slurm/pilot.sbatch > outputs/logs/pilot-none.log 2>&1 &  # after a code fix
```

Results go to `runs/pilot/`, never to `outputs/`. How a run goes and how to restart it:
[docs/pilot.md](docs/pilot.md).

## Documentation

| Document | Covers |
|---|---|
| [docs/pilot.md](docs/pilot.md) | The pilot run, its check report, resubmitting with `FORCE_FROM` |
| [docs/full-run.md](docs/full-run.md) | The full run step by step, downloading a larger fraction, regenerating tile lists, running several runs on one pod, rerunning one stage |
| [docs/runs.md](docs/runs.md) | Dataset size by role, training fractions, tile lists, model variants, starting weights and their limits |
| [docs/decisions.md](docs/decisions.md) | The values to fix before the full run, and where each is set |
| [docs/stages.md](docs/stages.md) | Every stage's node, inputs and outputs; skips, refusals and the data validation |
| [docs/data-layout.md](docs/data-layout.md) | The store, slot hashes, sparse files, run artifacts, resource measurements |
| [docs/outputs.md](docs/outputs.md) | Which script produces each number and table in `outputs/` |
| [docs/testing.md](docs/testing.md) | Running the E2E suite, its modules, what it does not cover |
| [docs/failure-modes.md](docs/failure-modes.md) | The 56 ways the pipeline can fail and what catches each |

## Development

```bash
uv run pytest -q                                       # E2E tests, artifacts in tests/artifacts/
uv run ruff format --check . && uv run ruff check .    # the pre-push checks
```

## Attribution and licenses

- **COMPASS** (MIT, `compass/LICENSE`): Cheung, Veeraraghavan and Balakrishnan, COMPASS: Robust
  Feature Conformal Prediction for Medical Segmentation Metrics, ICLR 2026, arXiv:2509.22240. Code
  adapted from <https://github.com/matthewyccheung/compass>.
- **Dataset** (CC BY-SA 4.0): Jasansky et al., Global ML-ready dataset for mining areas in satellite
  images, Zenodo, 2024, DOI 10.5281/zenodo.14195737.
- **Imagery**: modified Copernicus Sentinel data, through Microsoft Planetary Computer.
