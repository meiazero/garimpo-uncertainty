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

Every Colab session starts the same way. In a Python cell, with a read-only GitHub token saved as
the Colab secret `GH_TOKEN`, since the repository is private:

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

The runtime's disk is erased when the session ends, so the store lives in Drive as one archive,
`data.tar.zst`, extracted to the local disk at every session, and `runs/` is copied from Drive: reading
the store's memmaps through Drive would make every cell a remote read. Before the archive exists,
the `tar` line prints an error and the cell goes on. A `!` line mixing a Python `{name}` with a shell `$name` is passed on unexpanded, so the
cells use Python names only and write `$$` for the shell's `$`. `tar` and `rsync -a` keep modification times, which the stages' fingerprints compare. Check that
`df -h /content` leaves room for the 69.2 GB store, `.venv` (6.9 GB) and the runs.

## Quick start

The data, once, on a CPU runtime, after the session cell. The 1% list comes first: the time between its two `date` lines
says whether the 100% list fits the session, since a session that ends mid-download loses what the
local disk held. The archive uses zstd level 3 here, as higher levels compress far slower on the
runtime's 2 CPUs; any level extracts with the same command. `--format=posix` keeps the modification
times to the nanosecond, where tar's default format keeps whole seconds:

```python
!date; uv run --no-sync python scripts/download_data.py metadata; date
!date; uv run --no-sync python scripts/download_data.py scenes --tiles outputs/tiles/tiles_1pct.json --resampling-20m bilinear; date
!date; uv run --no-sync python scripts/download_data.py scenes --tiles outputs/tiles/tiles_100pct.json --resampling-20m bilinear; date
!uv run --no-sync python scripts/download_data.py weights --encoder-weights imagenet sentinel2_moco
!tar --format=posix -I 'zstd -3 -T0' -cf {DRIVE}/data.tar.zst -C {DATA} .
from google.colab import runtime; runtime.unassign()
```

The pilot, the 1% training fraction, on an L4 runtime, after the session cell. A cell goes on after
a failed `!` line, so the runs reach Drive and `runtime.unassign()` releases the GPU, which spends
compute units until it is released, whatever the pilot's outcome:

```python
!for w in none imagenet sentinel2_moco; do ENCODER_WEIGHTS=$$w bash slurm/pilot.sbatch 2>&1 | tee outputs/logs/pilot-$$w.log; done
!rsync -a runs/ {DRIVE}/runs/ && rsync -a outputs/logs/ {DRIVE}/logs/
from google.colab import runtime; runtime.unassign()
```

After a code fix, `FORCE_FROM=calibrate` before `bash slurm/pilot.sbatch` redoes each variant from
that stage.

Results go to `runs/pilot/`, never to `outputs/`. How a run goes and how to restart it:
[docs/pilot.md](docs/pilot.md).

## Documentation

| Document | Covers |
|---|---|
| [docs/pilot.md](docs/pilot.md) | The pilot run, its check report, resubmitting with `FORCE_FROM` |
| [docs/full-run.md](docs/full-run.md) | The full run step by step, downloading on your own machine, regenerating tile lists, the run cell for Colab, rerunning one stage |
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
