# Colab sessions

Read before changing one of the Colab cells in the [README](../README.md), or when a cell behaves
unexpectedly.

## The store is an archive in Drive

The runtime's disk is erased when the session ends. The store therefore lives in Drive as one
archive, `data.tar.zst`, which the session cell extracts to the local disk every time, and `runs/`
is copied from Drive the same way. Reading the store's memmaps through Drive would make every cell
a remote read.

`DATA` is always `/content/garimpo-data` because runs record the data root's path, and a run
restored under another path refuses ([stages.md](stages.md#run-identity)).

Each session clones the repository again with a read-only token, so code changes are committed and
pushed from your own machine, never from the session.

## Python names in `!` lines

A `!` line that mixes a Python `{name}` with a shell `$name` is passed on unexpanded. The cells
therefore use Python names only, and write `$$` where the shell needs `$`, as in the pilot cell's
`$$w`.

## Modification times

The stages' fingerprints compare each input file's size and modification time, so every copy has
to keep them. `tar` and `rsync -a` do. The data cell writes the archive with `--format=posix`,
which keeps modification times to the nanosecond; tar's default format keeps whole seconds.

## The data cell

The 1% list is downloaded before the 100% list. The time between its two `date` lines tells you
whether the 100% list fits in one session, which matters because a session that ends mid-download
loses everything on the local disk.

The archive uses zstd level 3 because higher levels compress far slower on the runtime's 2 CPUs.
Any level extracts with the same command. On your own machine,
[full-run.md](full-run.md#download-the-store) uses level 19.

## Releasing the GPU

A cell carries on after a failed `!` line. Whatever the pilot's outcome, the runs and logs still
reach Drive and `runtime.unassign()` releases the GPU, which spends compute units until it is
released.

So the cell itself always succeeds, also when a job failed: `tee` hides the job's exit code, and
the pilot's loop goes on to the next variant, which is wanted, since the variants are independent.
The job's verdict is the `PIPELINE CHECK: PASS` or `PIPELINE CHECK: FAIL` line `pipeline_check.py`
prints at the end of every job, failed or not. The cell prints that line of each log before
releasing the GPU, so read those lines, not the cell's status; `NO VERDICT` is a job that died
before its check.
