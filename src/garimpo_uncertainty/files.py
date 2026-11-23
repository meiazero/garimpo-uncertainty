"""Writing run files whole or not at all, hashing them, and recording the code that wrote them."""

import hashlib
import json
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


@contextmanager
def atomic(path: Path) -> Iterator[Path]:
    """Yield a temporary path renamed to `path` on success, so a crash leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        yield temporary
        Path(temporary).replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, content: dict) -> None:
    with atomic(path) as temporary:
        temporary.write_text(json.dumps(content, indent=2, default=str) + "\n")


def write_arrays(path: Path, **arrays: np.ndarray) -> None:
    with atomic(path) as temporary, temporary.open("wb") as file:
        np.savez(file, **arrays)


# The checkout this package runs from, so the records do not depend on the working directory.
ROOT = Path(__file__).resolve().parents[2]


def _git(directory: Path, *args: str) -> str:
    try:
        command = ["git", "-C", str(directory), *args]
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()  # noqa: S603 - literal git arguments only
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def code_versions() -> dict[str, str | bool]:
    """Return the commits of the package and the submodule, as the job actually saw them."""
    return {
        "root_commit": _git(ROOT, "rev-parse", "HEAD"),
        "root_dirty": bool(
            _git(ROOT, "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=dirty")
        ),
        "compass_commit": _git(ROOT / "compass", "rev-parse", "HEAD"),
        "compass_dirty": bool(_git(ROOT / "compass", "status", "--porcelain")),
    }
