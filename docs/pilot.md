# Pilot: adjust the code on the cluster

Read before the first job on the cluster.

The pilot is the 1% training fraction, run through the same scripts as the full run: 186 tiles and
2,976 cells, since validation, calibration and test stay whole. It downloads the versioned list
`outputs/tiles/tiles_1pct.json`. It exists to debug the
implementation, and its numbers say nothing about any method. Its values are in `slurm/pilot.sbatch`,
labeled pilot-only. Results go to `runs/pilot/`, never to `outputs/`.

On the login node:

```bash
git clone --recurse-submodules <ROOT_REPO_URL> garimpo-uncertainty
cd garimpo-uncertainty
uv sync
git config core.hooksPath .githooks
bash slurm/pilot.sbatch download
ENCODER_WEIGHTS=imagenet bash slurm/pilot.sbatch download           # data, then that variant's weights
ENCODER_WEIGHTS=sentinel2_moco bash slurm/pilot.sbatch download
```

Then through SLURM, on the login node too:

```bash
slurm/submit.sh slurm/pilot.sbatch                                  # U-Net from scratch
ENCODER_WEIGHTS=imagenet slurm/submit.sh slurm/pilot.sbatch
ENCODER_WEIGHTS=sentinel2_moco slurm/submit.sh slurm/pilot.sbatch
tail -f outputs/logs/garimpo-pilot-<JOBID>.out
```

Each command is its own job with its own 4 h limit, and writes to `runs/pilot/unet-<weights>/`. That
limit is a guess, with its arithmetic in `slurm/pilot.sbatch`; a job killed at the limit keeps its
finished stages.

The job ends with `scripts/pipeline_check.py`: status, seconds, peak memory and shapes per stage, slot hashes,
the failing stage with its error, scenes missing, cells per role, Dice, percent monotonic per
direction, the chosen layer with every candidate's fractions, `|cos(v1, w)|`, saturated searches, PICP and MPIW per method. It receives the job's exit
code, so an error outside every stage also reads FAIL.

After fixing the code, `git pull && uv sync` on the login node and resubmit. Finished stages skip
themselves. A stage whose inputs or arguments changed stops with status `refused`, which fails the
check. Commits are recorded in the metadata but stay out of the fingerprint, so a code fix alone
invalidates nothing: name the first stage it affects.

```bash
FORCE_FROM=calibrate slurm/submit.sh slurm/pilot.sbatch
```

`FORCE_FROM` takes a name from `STAGE_ORDER` in `slurm/stages.sh` and redoes that stage and those
after it. Earlier stages skip, and `train` runs again only with `FORCE_FROM=train`. The validation
sits outside that order: redo it with `FORCE_VALIDATE=1`. The same variables work with
`slurm/full.sbatch`.
