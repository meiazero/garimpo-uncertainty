# Colab sessions

Read before changing a cell of `colab.ipynb`, or when a cell behaves unexpectedly.

## The store is an archive in Drive

The runtime's disk is erased when the session ends. The store therefore lives in Drive as one
archive, `data.tar.zst`, which section 2 of the notebook unpacks to the local disk every time, and `runs/`
and the logs are copied back from Drive the same way. Reading the store's memmaps through Drive
would make every cell a remote read.

`DATA` is always `/content/garimpo-data` because runs record the data root's path, and a run
restored under another path refuses ([stages.md](stages.md#run-identity)).

Each session clones the repository again without credentials, so code changes are committed and
pushed from your own machine, never from the session.

## One command per cell

Each cell holds at most one shell command, so a cell that failed can run again on its own,
without repeating the commands before it. Shell lines use Python names and expressions only
(`{DRIVE}`, `{os.cpu_count()}`), never shell variables, loops or `;`, `&&` and `||` chains:

- IPython expands every `{...}` and every `$name` of a `!` line as Python, and `$$` is the escape
  for a shell `$`.
- If one of them fails, such as a `$name` with no Python variable of that name or a brace group
  `{ ...; }`, IPython runs the line with nothing expanded.

## Modification times

The stages' fingerprints compare each input file's size and modification time, so every copy has
to keep them. `tar` and `rsync -a` do. The archive is written with `--format=posix`
([full-run.md](full-run.md#download-the-store)), which keeps modification times to the
nanosecond; tar's default format keeps whole seconds.

## Releasing the runtime

The notebook ends with a cell that calls `release_runtime()`, so Runtime > Run all runs every
section on one runtime and releases it only at the end. It unmounts Drive, which writes the files
still pending, and then releases the runtime, which spends compute units until it is released.

When a Python cell raises, Colab cancels the cells queued after it, the release included. So no
cell before the release raises:

- A failed `!` line only prints its error, and the cell carries on.
- The Python cells that can fail catch the error and print it, such as `print_verdicts`.

If you interrupt a run cell once, `scripts/run.py` stops the stage, keeps the lines it prints as it
stops, runs the pipeline check and exits 130. Colab turns a second interrupt into SIGTERM and then
SIGKILL: the runner kills the stage at once, then still runs the check, whose verdict reaches the
cell and the log. Then run the cells of
section 7 by hand.

## Optional and reexecution cells

A cell titled "- (optional)" runs only while its `RUN` box is checked, which it is by default. A
cell titled "- (in a reexecution)" runs only once its `RUN` box is checked, so Runtime > Run all
never forces a stage by accident. Both write their command in full, so the cell shows what runs.

## Job verdicts

Since every job cell succeeds, read the job's verdict instead of the cell's status. Every pilot and
full-run job ends with the `PIPELINE CHECK: PASS` or `PIPELINE CHECK: FAIL` line of
`pipeline_check.py`, failed or not. `scripts/run.py` appends every line to the model's log with the
UTC time, so the log keeps earlier attempts, including one cut short by a session that ended. It
writes an `=== attempt started` line before each attempt, with the config, the commit and the
options, and `print_verdicts` prints the verdict of the last attempt only:

- `NO VERDICT`: the job or its check died before the verdict; the log says where.
- `NO LOG`: the log does not exist yet or could not be read.
- `NOT RUN`: this session wrote nothing to the log, which still has the time it had in Drive. The
  run died before its first line (an `argparse` or import error, or no environment), as the cell's
  output shows, or its cell did not run.

## The copy every 30 minutes

During a full run, a Python thread copies `runs/` (without `*.tmp` files) and the logs to Drive
every 30 minutes, so a session that ends mid-run keeps the finished stages and the last
checkpoint. The stop cell waits for a copy in progress to finish, so it never writes to Drive at
the same time as the final copy.
