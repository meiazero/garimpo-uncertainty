# Colab sessions

Read before changing a cell of `notebooks/01_run_experiment.ipynb`, or when a cell behaves
unexpectedly. The last section covers what `notebooks/03_tune_hardware.ipynb` does differently.

## The store is an archive in Drive

The runtime's disk is erased when the session ends. The store therefore lives in Drive as one
archive, `data.tar.zst`, which section 2 of the notebook unpacks to the local disk once per
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
pushed from your own machine, never from the session.

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

The notebook ends with a cell that calls `release_runtime()`, so Runtime > Run all runs every
section on one runtime and releases it only at the end. It unmounts Drive, which writes the files
still pending, and then releases the runtime, which spends compute units until it is released.

When a Python cell raises, Colab cancels the cells queued after it, the release included. So no
cell before the release raises:

- A failed `!` line only prints its error, and the cell carries on.
- The Python cells that can fail catch the error and print it, such as the choice of the scale.

If you interrupt a run cell once, `garimpo run` stops the stage, keeps the lines it prints as it
stops, runs the pipeline check and exits 130. Colab turns a second interrupt into SIGTERM and then
SIGKILL: the runner kills the stage at once, then still runs the check, whose verdict reaches the
cell and the log, and the run cell starts no later model. Then run the cell of section 7 by
hand.

## The form

Section 3 holds every choice a session makes: the scale, the models to run, `FORCE_FROM`,
`FORCE_VALIDATE` and `RUN_TESTS`. `FORCE_FROM` applies to every model checked, so uncheck the
others to redo one. Its defaults force nothing and skip the tests, so Runtime > Run all never
redoes a stage by accident. The values stay set in the open tab; reopening the notebook from its
link restores the defaults.

## Job verdicts

Since every job cell succeeds, read the job's verdict instead of the cell's status. Every pilot and
full-run job ends with the `PIPELINE CHECK: PASS` or `PIPELINE CHECK: FAIL` line of
`garimpo check`, failed or not. `garimpo run` appends every line to the model's log with the
UTC time, so the log keeps earlier attempts, including one cut short by a session that ended. It
writes an `=== attempt started` line before each attempt, with the config, the commit and the
options, and `garimpo status <scale> --since <session start>`, which ends the run cell, prints the
verdict of the last attempt only:

- `NO VERDICT`: the job or its check died before the verdict; the log says where.
- `NO LOG`: the log does not exist yet or could not be read.
- `NOT RUN`: this session wrote nothing to the log, which still has the time it had in Drive. The
  run died before its first line (an `argparse` or import error, or no environment), as the cell's
  output shows, or the model was not checked in the form.

## The copy every 30 minutes

During a full run, a Python thread copies `runs/` (without `*.tmp` files, as the final copy and the
restore do) and the logs to Drive every 30 minutes, so a session that ends mid-run keeps the
finished stages and the last checkpoint. Section 7 waits for a copy in progress to finish, so it
never writes to Drive at the same time as the final copy.

## The sizing notebook

`notebooks/03_tune_hardware.ipynb` follows the rules above with three differences, because it
measures the machine instead of running the pipeline:

- It touches neither Drive nor the real store. Its cells are synthetic, written to
  `outputs/tuning/store/` on the local disk, and its results stay in the kernel's memory, so the
  release at the end, when `RELEASE` is checked, erases them: copy the verdict first.
- Its measurements run in the kernel's own Python, the one the pipeline's commands run on. The
  setup installs `requirements.txt` there, after checking that it replaces none of Colab's packages:
  on another runtime version than the latest one, whose versions it pins, it stops. Then it reads
  the `.pth` files of the editable installs, which Python only reads at startup, so `import
  garimpo_uncertainty` works without restarting the session.
- Each measuring cell starts with `%%section <name>`, a cell magic defined in the setup. It runs the
  cell, and when the cell raises it prints the traceback and records the failure for the verdict
  instead of raising, so Colab cancels no later cell and the release still runs. A failed setup
  skips every section the same way.

Its form's `BRANCH` names the code it measures; set it to a branch that is not merged yet to size
that branch's code.
