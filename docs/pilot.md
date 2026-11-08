# Pilot: adjust the code on a GPU pod

Read before the first run on a GPU.

The pilot is the 1% training fraction, run through the same scripts as the full run: 186 tiles and
2,976 cells, since validation, calibration and test stay whole. It downloads the versioned list
`outputs/tiles/tiles_1pct.json`. It exists to debug the
implementation, and its numbers say nothing about any method. Its values are in `slurm/pilot.sbatch`,
the decided ones from [decisions.md](decisions.md) and the rest labeled pilot-only. Results go to `runs/pilot/`, never to `outputs/`.

After the [install](../README.md#install), with `GARIMPO_DATA_DIR` on the network volume, on any pod:

```bash
bash slurm/pilot.sbatch download
ENCODER_WEIGHTS=imagenet bash slurm/pilot.sbatch download           # data, then that variant's weights
ENCODER_WEIGHTS=sentinel2_moco bash slurm/pilot.sbatch download
```

Then on the GPU pod, inside `tmux` so a dropped SSH session does not stop it. The pod bills by the hour until it stops, so the chain below stops it when the last variant ends,
failed or not; the data and runs stay on the volume. The variants run one after another, since they
would share one GPU. First check that `command -v runpodctl && echo "$RUNPOD_POD_ID"` prints both
(RunPod's docs spell the command `runpodctl pod stop`; older versions take `runpodctl stop pod`):

```bash
tmux new -s pilot                                  # apt-get install -y tmux if missing
mkdir -p outputs/logs
for w in none imagenet sentinel2_moco; do
    ENCODER_WEIGHTS=$w bash slurm/pilot.sbatch 2>&1 | tee outputs/logs/pilot-$w.log
done; runpodctl pod stop "$RUNPOD_POD_ID"
```

Detach with `Ctrl-b d` and reattach with `tmux attach -t pilot`.

Each command writes to `runs/pilot/unet-<weights>/`. A run stopped midway, by a crash or a stopped
pod, keeps its finished stages, which skip on the next start.

The run ends with `scripts/pipeline_check.py`: status, seconds, peak memory and shapes per stage, slot hashes,
the failing stage with its error, scenes missing, cells per role, Dice, percent monotonic per
direction, the chosen layer with every candidate's fractions, `|cos(v1, w)|`, saturated searches, PICP and MPIW per method. It receives the script's exit
code, so an error outside every stage also reads FAIL.

After fixing the code, `git pull && uv sync` on the pod and start the same command again. Finished stages skip
themselves. A stage whose inputs or arguments changed stops with status `refused`, which fails the
check. Commits are recorded in the metadata but stay out of the fingerprint, so a code fix alone
invalidates nothing: name the first stage it affects.

```bash
for w in none imagenet sentinel2_moco; do
    FORCE_FROM=calibrate ENCODER_WEIGHTS=$w bash slurm/pilot.sbatch 2>&1 | tee outputs/logs/pilot-$w.log
done; runpodctl pod stop "$RUNPOD_POD_ID"
```

`FORCE_FROM` takes a name from `STAGE_ORDER` in `slurm/stages.sh` and redoes that stage and those
after it. Earlier stages skip, and `train` runs again only with `FORCE_FROM=train`. The validation
sits outside that order: redo it with `FORCE_VALIDATE=1`. The same variables work with
`slurm/full.sbatch`.
