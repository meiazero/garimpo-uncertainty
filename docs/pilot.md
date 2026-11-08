# Pilot: adjust the code on a Colab GPU

Read before the first run on a GPU.

The pilot is the 1% training fraction, run through the same scripts as the full run: 186 tiles and
2,976 cells, since validation, calibration and test stay whole. It downloads the versioned list
`outputs/tiles/tiles_1pct.json`. It exists to debug the
implementation, and its numbers say nothing about any method. Its values are in `slurm/pilot.sbatch`,
the decided ones from [decisions.md](decisions.md) and the rest labeled pilot-only. Results go to `runs/pilot/`, never to `outputs/`.

Run the data cell of the [quick start](../README.md#quick-start) once, then the pilot cell on an
L4 runtime, each after the session cell of the [install](../README.md#install). The variants run
one after another on the one GPU, the runs and logs reach Drive, and `runtime.unassign()` releases
the GPU even when a variant fails.

Each variant writes to `runs/pilot/unet-<weights>/`. A run stopped midway keeps its finished stages,
which skip on the next start once `runs/` is restored from Drive with its modification times.

The run ends with `scripts/pipeline_check.py`: status, seconds, peak memory and shapes per stage, slot hashes,
the failing stage with its error, scenes missing, cells per role, Dice, percent monotonic per
direction, the chosen layer with every candidate's fractions, `|cos(v1, w)|`, saturated searches, PICP and MPIW per method. It receives the script's exit
code, so an error outside every stage also reads FAIL.

After fixing the code, push it; the next session's clone picks it up. Finished stages skip
themselves. A stage whose inputs or arguments changed stops with status `refused`, which fails the
check. Commits are recorded in the metadata but stay out of the fingerprint, so a code fix alone
invalidates nothing: name the first stage it affects.

```python
!for w in none imagenet sentinel2_moco; do FORCE_FROM=calibrate ENCODER_WEIGHTS=$$w bash slurm/pilot.sbatch 2>&1 | tee outputs/logs/pilot-$$w.log; done
```

`FORCE_FROM` takes a name from `STAGE_ORDER` in `slurm/stages.sh` and redoes that stage and those
after it. Earlier stages skip, and `train` runs again only with `FORCE_FROM=train`. The validation
sits outside that order: redo it with `FORCE_VALIDATE=1`. The same variables work with
`slurm/full.sbatch`.
