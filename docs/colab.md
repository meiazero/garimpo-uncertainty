# Colab sessions

Read before changing a cell of `notebooks/01_run_experiment.ipynb`, or when a cell behaves
unexpectedly. The last section covers what `notebooks/03_tune_hardware.ipynb` does differently.

## The store is an archive in Drive

The runtime's disk is erased when the session ends. The store therefore lives in Drive as one
archive, `data.tar.zst`, which section 3 of the notebook unpacks to the local disk once per
session, skipping it once `.unpacked` is there, a marker written only after tar succeeds, so a cut
unpack is redone: reading the store's memmaps through Drive would make every cell a remote read.

`runs/` and the logs are copied back from Drive with `rsync -a --ignore-existing`, which copies
only the files the local disk lacks. A new session gets everything; Run all again in the same
session keeps the local files, which are never older than Drive's copy, since Drive only receives
copies of them. A file deleted locally in the session comes back, as no copy uses `--delete`:
`--delete` toward Drive would wipe the saved runs after a restore that failed. The one file a stage
deletes, `chosen_layer.json` when no layer qualifies, can therefore come back: the pipeline check
shows a stage's table only while it and every stage before it are `ok`, and `garimpo merge`
refuses a run with any stage not `ok`.

`DATA` is always `/content/garimpo-data` because runs record the data root's path, and a run
restored under another path refuses ([stages.md](stages.md#run-identity)).

Each session clones the repository again without credentials, so code changes are committed and
pushed from your own machine, never from the session. The setup cell is the same in the three
notebooks: it clones when the clone is missing, else moves it to the last commit of `BRANCH`, which
is `main` in the run notebook.

Every notebook installs `requirements.txt` with `scripts/colab_install.py`. It stops, naming the
package, when pip would replace Colab's torch, torchvision, triton or an `nvidia-*` or `cuda-*`
library, which pick the GPU's kernels and so its bits: a runtime version with another torch, such
as 2026.10, stops there. A runtime a day or two behind its published versions differs in a few
other packages, which pip aligns to the pins.

## Every cell runs again safely

One cell per section, and each can run again, alone or through Run all, in the same session or a
new one: the setup clones only when the clone is missing, the data unpacks only when it is not
there yet, and finished stages skip. Shell lines use Python names and expressions only
(`{DRIVE}`, `{model}`), never shell variables or `;`, `&&` and `||` chains; loops are Python
loops around `!` lines:

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

The notebook ends with a release cell, the same in the three notebooks, so Runtime > Run all runs
every section on one runtime and releases it only at the end. It unmounts Drive, which writes the files
still pending, and then releases the runtime, which spends compute units until it is released.

When a Python cell raises, Colab cancels the cells queued after it, the release included. So no
cell before the release raises:

- A failed `!` line only prints its error, and the cell carries on.
- The Python cells that can fail catch the error and print it, such as the record of the tests'
  pass in Drive.

If you interrupt a run cell once, `garimpo run` stops the stage, keeps the lines it prints as it
stops, runs the pipeline check and exits 130. Colab turns a second interrupt into SIGTERM and then
SIGKILL: the runner kills the stage at once, then still runs the check, whose verdict reaches the
cell and the log, and `garimpo experiment` starts no later model. Then run the cells of sections 6
and 7 by hand.

## The gates

A session needs no choice. Section 4 runs the tests on the GPU unless `MyDrive/garimpo/tested/`
records a pass for the commit checked out, and no run starts after a failure. Section 5 then runs
`garimpo experiment`, which picks the scale as `garimpo status auto --next` does: the pilot until each pilot run's `verdict.json` holds a
`PASS` from the commits checked out, with no local edit, then the full scale, so no full run starts
or goes on with code the pilot has not passed. A pilot trained on other commits redoes itself from
`train` (`redo_from` in `pipeline/run.py`); the full runs are never redone on their own.

Section 1's one option, `FORCE_FROM`, is for a code fix to a stage the full runs already finished:
the four full runs redo that stage and every one after it. It stays set in the open tab; reopening
the notebook from its link empties it.

## Job verdicts

Since every job cell succeeds, read the job's verdict instead of the cell's status. Every pilot and
full-run job ends with the `PIPELINE CHECK: PASS` or `PIPELINE CHECK: FAIL` line of
`garimpo check`, failed or not. `garimpo run` appends every line to the model's log with the
UTC time, so the log keeps earlier attempts, including one cut short by a session that ended. It
writes an `=== attempt started` line before each attempt, with the config, the commit and the
options, and `garimpo experiment --since <session start>` ends by printing, for each scale that ran,
the verdict of the last attempt only:

- `NO VERDICT`: the job or its check died before the verdict; the log says where.
- `NO LOG`: the log does not exist yet or could not be read.
- `NOT RUN`: this session wrote nothing to the log, which still has the time it had in Drive. The
  run died before its first line (an `argparse` or import error, or no environment), as the cell's
  output shows, or a gate of section 4 or 5 kept it from starting.

## The copy every 30 minutes

During a full run, a Python thread copies `runs/` (without `*.tmp` files, as the final copy and the
restore do) and the logs to Drive every 30 minutes, so a session that ends mid-run keeps the
finished stages and the last checkpoint. Section 6 waits for a copy in progress to finish, so it
never writes to Drive at the same time as the final copy.

## The sizing notebook

`notebooks/03_tune_hardware.ipynb` follows the rules above with two differences, because it
measures the machine instead of running the pipeline:

- It touches neither Drive nor the real store. `garimpo tune` writes synthetic cells to
  `outputs/tuning/store/` on the local disk and its rows to `outputs/tuning/results.json`, which
  the release at the end, when `RELEASE` is checked, erases: copy the verdict first.
- A section that raises is reported and the next one runs (`tuning/tune.py`, `run_sections`), so
  one run reports every finding, and the command exits 1 when any section failed or lost a bit.

Its form's `BRANCH` names the code it measures; set it to a branch that is not merged yet to size
that branch's code.
