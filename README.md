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

Run this cell at the start of every Colab session. The repository is private, so save a read-only
GitHub token as the Colab secret `GH_TOKEN` first.

```python
import os
from google.colab import drive, userdata
drive.mount("/content/drive")
REPO = "github.com/<OWNER>/garimpo-uncertainty.git"         # the repository's path on GitHub
DRIVE = "/content/drive/MyDrive/garimpo"                    # store and runs between sessions
DATA = "/content/garimpo-data"                              # always this path: runs record it
TOKEN = userdata.get("GH_TOKEN")
os.environ["GARIMPO_DATA_DIR"] = DATA
os.environ["PATH"] = "/root/.local/bin:" + os.environ["PATH"]
!apt-get -qq install -y rsync zstd > /dev/null
!curl -LsSf https://astral.sh/uv/install.sh | sh
!git clone --recurse-submodules https://{TOKEN}@{REPO} /content/garimpo-uncertainty
%cd /content/garimpo-uncertainty
!uv sync
!mkdir -p {DRIVE}/runs {DATA} runs outputs/logs
!tar -I zstd -xf {DRIVE}/data.tar.zst -C {DATA}
!rsync -a {DRIVE}/runs/ runs/
```

Until the archive exists, the `tar` line prints an error and the cell carries on. Check that
`df -h /content` has room for the 69.2 GB store, `.venv` (6.9 GB) and the runs. Why the cell is
written this way: [docs/colab.md](docs/colab.md).

## Quick start

Download the data once, on a CPU runtime, after the session cell:

```python
!date; uv run --no-sync python scripts/download_data.py metadata; date
!date; uv run --no-sync python scripts/download_data.py scenes --tiles outputs/tiles/tiles_1pct.json --resampling-20m bilinear; date
!date; uv run --no-sync python scripts/download_data.py scenes --tiles outputs/tiles/tiles_100pct.json --resampling-20m bilinear; date
!uv run --no-sync python scripts/download_data.py weights --encoder-weights imagenet sentinel2_moco
!tar --format=posix -I 'zstd -3 -T0' -cf {DRIVE}/data.tar.zst -C {DATA} .
from google.colab import runtime; runtime.unassign()
```

Then run the pilot, the 1% training fraction, on an L4 runtime, after the session cell:

```python
!for w in none imagenet sentinel2_moco; do ENCODER_WEIGHTS=$$w bash slurm/pilot.sbatch 2>&1 | tee outputs/logs/pilot-$$w.log; done
!rsync -a runs/ {DRIVE}/runs/ && rsync -a outputs/logs/ {DRIVE}/logs/
from pathlib import Path
for log in sorted(Path("outputs/logs").glob("pilot-*.log")):  # the cell succeeds whatever the jobs did
    verdict = [line for line in log.read_text().splitlines() if line.startswith("PIPELINE CHECK: ")]
    print(log.name, verdict[-1] if verdict else "NO PIPELINE CHECK: the job died before it")
from google.colab import runtime; runtime.unassign()
```

Results go to `runs/pilot/`, never to `outputs/`. Reading the report and rerunning after a fix:
[docs/pilot.md](docs/pilot.md).

## Documentation

| Document | Covers |
|---|---|
| [docs/colab.md](docs/colab.md) | Why the Colab cells are written as they are: the Drive archive, `$$`, modification times, zstd level, releasing the GPU |
| [docs/pilot.md](docs/pilot.md) | Running the pilot, its check report, rerunning with `FORCE_FROM` |
| [docs/full-run.md](docs/full-run.md) | The full run step by step, downloading on your own machine, the run cell, regenerating tile lists, rerunning one stage |
| [docs/decisions.md](docs/decisions.md) | The values to fix before the full run, their sources, and where each is set |
| [docs/runs.md](docs/runs.md) | Dataset size by role, training fractions, tile lists, model variants, starting weights, layer names |
| [docs/design.md](docs/design.md) | What the model comparison can and cannot show, limits of the starting weights, the quantile rule and its sensitivity analysis |
| [docs/stages.md](docs/stages.md) | Every stage's node, inputs and outputs; statuses, skips, refusals, run identity, the data validation |
| [docs/data-layout.md](docs/data-layout.md) | The store, slot hashes, sparse files, run artifacts, resource measurements |
| [docs/outputs.md](docs/outputs.md) | Which script produces each number and table in `outputs/`, regenerating the split, metric rules |
| [docs/testing.md](docs/testing.md) | Running the E2E suite, its modules, what it does not cover |
| [docs/failure-modes.md](docs/failure-modes.md) | The 62 ways the pipeline can fail and what catches each |

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
