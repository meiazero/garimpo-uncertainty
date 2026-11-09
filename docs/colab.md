# Colab sessions

Read before changing a cell of `colab.ipynb`, or when a cell behaves unexpectedly.

## The store is an archive in Drive

The runtime's disk is erased when the session ends. The store therefore lives in Drive as one
archive, `data.tar.zst`, which the session setup unpacks to the local disk every time, and `runs/`
and the logs are copied back from Drive the same way. Reading the store's memmaps through Drive
would make every cell a remote read.

`DATA` is always `/content/garimpo-data` because runs record the data root's path, and a run
restored under another path refuses ([stages.md](stages.md#run-identity)).

Each session clones the repository again with a read-only token, so code changes are committed and
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
to keep them. `tar` and `rsync -a` do. The archive cell writes with `--format=posix`, which keeps
modification times to the nanosecond; tar's default format keeps whole seconds.

## The data step

The 1% list is downloaded before the 100% list. The 1% cell's run time, shown when you hover over
the icon to its left, tells you whether the 100% list fits in one session, which matters because a
session that ends mid-download loses everything not yet archived.

The archive cell archives nothing unless `status/download_scenes.json` says `ok`. `tar` writes
`data.tar.zst.tmp`, which replaces the archive only when `tar` succeeds, so a failed download or
archive leaves the old archive as it was. Two limits:

- `ok` means the download stage finished, not that every tile did. A tile that failed keeps its
  status and reason in its sidecar, `tiles/<tile_id>.json`, and is archived that way; rerunning
  the scenes cell retries it.
- When an archive already exists, unpack it before downloading, so the downloads resume from it
  and the new archive keeps everything the old one had. If the unpack failed partway, do not run
  the archive cell: the status it reads comes from the old archive and says `ok` even for a store
  cut short.

The archive uses zstd level 3 because higher levels compress far slower on the runtime's 2 CPUs.
Any level extracts with the same command. On your own machine,
[full-run.md](full-run.md#download-the-store) uses level 19.

## Releasing the runtime

Each step ends with a cell that calls `release_runtime()`. It unmounts Drive, which writes the
files still pending, and then releases the runtime, which spends compute units until it is
released.

When a Python cell raises, Colab cancels the cells queued after it, the release included. So no
cell of a step raises before its release:

- A failed `!` line only prints its error, and the cell carries on. `tee` also hides the job's exit
  code.
- The Python cells that can fail catch the error and print it: the archive cell, `start_log` and
  `print_verdicts`.

If you interrupt a cell, check that the release cell ran, and run it by hand if it did not.

## Job verdicts

Since every job cell succeeds, read the job's verdict instead of the cell's status. Every pilot and
full-run job ends with the `PIPELINE CHECK: PASS` or `PIPELINE CHECK: FAIL` line of
`pipeline_check.py`, failed or not. Each job appends to its log with `tee -a`, so the log keeps
earlier attempts, including one cut short by a session that ended. `start_log` writes an
`=== attempt started` line before each attempt, and `print_verdicts` prints the verdict of the last
attempt only:

- `NO VERDICT`: the job or its check died before the verdict; the log says where.
- `NO LOG`: the log does not exist yet or could not be read.

## The copy every 30 minutes

During a full run, a Python thread copies `runs/` (without `*.tmp` files) and the logs to Drive
every 30 minutes, so a session that ends mid-run keeps the finished stages and the last
checkpoint. The stop cell waits for a copy in progress to finish, so it never writes to Drive at
the same time as the final copy.
