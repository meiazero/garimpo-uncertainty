"""Run the experiment in a Colab CLI session, as notebooks/01_run_experiment.ipynb does in Colab.

    colab new -s garimpo --gpu L4
    colab drivemount -s garimpo
    colab exec -s garimpo --timeout 60 -f scripts/colab_session.py

drive.mount() hangs in the CLI's kernel, so `colab drivemount` mounts Drive first. This file runs in
the kernel, not over ssh, because Colab ends a session whose kernel sits idle: the kernel stays busy
until the runs end, and `colab exec` stops waiting after its timeout without interrupting it.
`--env GARIMPO_FORCE_FROM=<stage>` redoes that stage and those after it. Everything goes to
/content/session.log, which reaches MyDrive/garimpo/session-logs/ every 5 minutes. At the end the
session releases the VM, as the notebook does; `colab stop -s garimpo` if it did not.
"""

import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

REPO = "https://github.com/meiazero/garimpo-uncertainty.git"
CODE = Path("/content/garimpo-uncertainty")  # the clone; erased with the session
DRIVE = Path("/content/drive/MyDrive/garimpo")  # kept between sessions
DATA = Path("/content/garimpo-data")  # runs record this path, so it never changes
SESSION_LOG = Path("/content/session.log")
# The logs are small and say where a lost session stopped; runs/ holds the checkpoints.
LOGS_EVERY, RUNS_EVERY = 300, 1800
# drive.flush_and_unmount() may hang in the CLI's kernel, as drive.mount() does.
FLUSH_SECONDS = 600


class Session:
    """The steps of the run notebook, each one's output appended to SESSION_LOG."""

    def __init__(self) -> None:
        self.started = time.time()
        self.log = SESSION_LOG.open("a", encoding="utf-8", buffering=1)
        self.stopped = threading.Event()
        self.copiers: list[threading.Thread] = []
        # Only a session that runs may write logs/ and runs/ back: the restore kept local files.
        self.running = False

    def say(self, text: str) -> None:
        self.log.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {text}\n")

    def ran(self, *command: str, quiet: bool = False, **env: str) -> bool:
        """Run one command inside the clone once it exists; True when it exits 0."""
        if not quiet:
            self.say(f"$ {shlex.join(command)}")
        try:
            done = subprocess.run(  # noqa: S603 - the commands below
                command,
                cwd=CODE if CODE.is_dir() else None,
                env=os.environ | env,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        except OSError as error:
            self.say(f"could not start {command[0]}: {error}")
            return False
        if done.returncode != 0 and quiet:
            self.say(f"{shlex.join(command)} ended with code {done.returncode}")
        return done.returncode == 0

    def every(self, seconds: int, action: Callable[[], None]) -> None:
        """Call `action` every `seconds` in a thread until the session stops copying."""

        def loop() -> None:
            while not self.stopped.wait(seconds):
                action()

        self.copiers.append(threading.Thread(target=loop, daemon=True))
        self.copiers[-1].start()

    def keep_log(self) -> None:
        """Copy this session's log to Drive, outside logs/, which the restore copies back."""
        name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(self.started))
        kept = DRIVE / "session-logs" / f"{name}.log"
        try:
            kept.parent.mkdir(parents=True, exist_ok=True)
            # A copy cut by the session's end must not replace the last whole one.
            shutil.copyfile(SESSION_LOG, kept.with_suffix(".tmp"))
            kept.with_suffix(".tmp").replace(kept)
        except OSError as error:
            self.say(f"could not copy {SESSION_LOG} to Drive: {error}")

    def save_logs(self) -> None:
        if self.running:
            self.ran("rsync", "-a", "outputs/logs/", f"{DRIVE}/logs/", quiet=True)
        self.keep_log()

    def save_runs(self) -> None:
        self.ran("rsync", "-a", "--exclude", "*.tmp", "runs/", f"{DRIVE}/runs/", quiet=True)

    def set_up(self) -> bool:
        """Clone and install main, unpack the data and restore the runs from Drive."""
        steps = [] if (CODE / ".git").is_dir() else [["git", "clone", "-q", REPO, str(CODE)]]
        steps += [
            ["git", "fetch", "-q", "origin", "main"],
            ["git", "checkout", "-q", "-B", "main", "origin/main"],
            ["git", "submodule", "update", "-q", "--init", "--recursive"],
            [sys.executable, "scripts/colab_install.py"],
            ["git", "log", "-1", "--format=Running commit %h %cs %s"],
            ["apt-get", "-qq", "install", "-y", "rsync", "zstd"],
        ]
        if not all(self.ran(*step, DEBIAN_FRONTEND="noninteractive") for step in steps):
            return False
        for folder in (DATA, CODE / "runs", CODE / "outputs/logs", DRIVE / "runs", DRIVE / "logs"):
            folder.mkdir(parents=True, exist_ok=True)
        # The marker is written only after tar succeeds, so a cut unpack is redone, never trusted.
        if not (DATA / ".unpacked").exists():
            if not self.ran(
                "tar", "-I", "zstd", "-xf", str(DRIVE / "data.tar.zst"), "-C", str(DATA)
            ):
                return False
            (DATA / ".unpacked").touch()
        # A run started without its last.pt would train from scratch and overwrite the one in Drive.
        return self.ran(
            "rsync", "-a", "--ignore-existing", "--exclude", "*.tmp", f"{DRIVE}/runs/", "runs/"
        ) and self.ran("rsync", "-a", "--ignore-existing", f"{DRIVE}/logs/", "outputs/logs/")

    def tested(self) -> bool:
        """Run the tests on the GPU unless this commit passed them in an earlier session."""
        found = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=CODE,
            capture_output=True,
            text=True,
            check=False,
        )
        if not (commit := found.stdout.strip()):
            self.say("git named no commit, so a pass could not be recorded")
            return False
        marker = DRIVE / "tested" / commit
        if marker.exists():
            self.say(f"Tests skipped: commit {commit[:7]} passed them in an earlier session.")
            return True
        if not self.ran("pytest", "-q", GARIMPO_TEST_DEVICE="cuda"):
            return False
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
        except OSError as error:
            self.say(
                f"Could not record the pass in Drive, so the next session tests again: {error}"
            )
        return True

    def run(self) -> int:
        """Run the pilot, then the full runs, then merge each scale that ran; return its code."""
        merged = CODE / "outputs" / "layer_results.csv"
        before = merged.stat().st_mtime_ns if merged.exists() else None
        command = ["garimpo", "experiment", "--since", str(self.started)]
        if force_from := os.environ.get("GARIMPO_FORCE_FROM", ""):
            command += ["--force-from", force_from]
        self.running = True
        self.every(RUNS_EVERY, self.save_runs)
        self.say(f"$ {shlex.join(command)}")
        with subprocess.Popen(command, cwd=CODE, stdout=self.log, stderr=subprocess.STDOUT) as job:  # noqa: S603
            try:
                code = job.wait()
            except KeyboardInterrupt:
                # garimpo experiment stops its stage and records the attempt; killing it would not.
                job.send_signal(signal.SIGINT)
                code = job.wait()
        after = merged.stat().st_mtime_ns if merged.exists() else None
        # A merge killed mid-write leaves its partial *.tmp beside the table.
        if after is not None and after != before:
            self.ran("rsync", "-a", "--exclude", "*.tmp", "outputs/", f"{DRIVE}/outputs/")
        return code

    def end(self) -> None:
        """Wait for a copy in progress, save once more and release the VM."""
        self.stopped.set()
        for copier in self.copiers:
            copier.join()  # two copies at once could leave Drive with the older file
        if self.running:
            self.save_runs()
        self.save_logs()
        self.release()

    def release(self) -> None:
        """Flush Drive's writes and give the VM back, as the notebook's last cell does."""
        try:
            from google.colab import drive, runtime  # noqa: PLC0415 - only in Colab
        except ImportError as error:
            self.say(f"Could not release the VM, so run `colab stop`: {error!r}")
            return
        self.say(flushed(drive.flush_and_unmount))
        try:
            runtime.unassign()
        except Exception as error:  # noqa: BLE001 - whatever failed, the VM is released by hand
            self.say(f"Could not release the VM, so run `colab stop`: {error!r}")


def flushed(flush_and_unmount: Callable[[], None]) -> str:
    """Wait up to FLUSH_SECONDS for Drive's last writes; say whether they were confirmed."""
    failed: list[Exception] = []

    def flush() -> None:
        try:
            flush_and_unmount()
        except Exception as error:  # noqa: BLE001 - reported, then the VM goes anyway
            failed.append(error)

    thread = threading.Thread(target=flush, daemon=True)
    thread.start()
    thread.join(FLUSH_SECONDS)
    if thread.is_alive() or failed:
        return f"Drive did not confirm its writes; check the files in Drive: {failed}"
    return "Drive flushed."


def blocker(session: Session) -> str | None:
    """Name what keeps the runs from starting, or None once main is set up and tested."""
    if not (DRIVE / "data.tar.zst").exists():
        return f"{DRIVE / 'data.tar.zst'} is missing"
    if not session.set_up():
        return "the setup failed; read the lines above"
    if not session.tested():
        return "the tests failed on the GPU; fix the code and push"
    return None


def main() -> None:
    os.environ["GARIMPO_DATA_DIR"] = str(DATA)  # where the stages read the unpacked store
    # Children inherit an ignored SIGINT, and the tests' Ctrl+C would not reach them.
    if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    session = Session()
    if not os.path.ismount("/content/drive"):
        session.say("STOPPED: Drive is not mounted; run `colab drivemount` first.")
        return
    session.every(LOGS_EVERY, session.save_logs)
    try:
        if reason := blocker(session):
            session.say(f"STOPPED: {reason}.")
        else:
            session.say(f"FINISHED: garimpo experiment exited {session.run()}")
    except Exception:  # noqa: BLE001 - the cause must reach the log before the VM goes
        session.say(f"STOPPED: {traceback.format_exc()}")
    finally:
        session.end()


if __name__ == "__main__":
    main()
