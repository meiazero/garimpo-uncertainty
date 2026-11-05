"""The synthetic dataset and the finished runs, built once and read by every worker.

Nothing may write to these. A scenario that damages data works on a private copy, or holds its own
data in a module fixture: fingerprints record paths, so only the second can still skip. A module
runs whole on one worker (`--dist loadfile`), so its scenarios take turns restoring that data.
"""

import shutil
from pathlib import Path

import pytest

from tests.e2e import support


@pytest.fixture(scope="session")
def data_root() -> Path:
    root = support.ARTIFACTS / "data"

    def build() -> None:
        support.build_dataset(root, support.SPLIT_PATH)
        support.validate(root, support.SPLIT_PATH, support.CLEAN_REPORT)

    support.build_once("data_root", build)
    return root


def shared_run(arch: str, data_root: Path) -> support.FinishedRun:
    run_dir = support.RUNS / arch
    support.build_once(f"{arch}_run", lambda: support.finish_run(arch, run_dir, data_root))
    return support.FinishedRun(arch, run_dir, support.RUN_ORDER[arch][1])


@pytest.fixture(scope="session")
def convnet_run(data_root: Path) -> support.FinishedRun:
    return shared_run("convnet", data_root)


@pytest.fixture(scope="session")
def unet_run(data_root: Path) -> support.FinishedRun:
    return shared_run("unet", data_root)


@pytest.fixture(scope="session", params=["convnet", "unet"])
def finished_run(request: pytest.FixtureRequest) -> support.FinishedRun:
    return request.getfixturevalue(f"{request.param}_run")


@pytest.fixture
def private_data_root(data_root: Path, tmp_path: Path) -> Path:
    """A copy to damage. Stages recorded the original's path, so here they redo, never skip."""
    return Path(shutil.copytree(data_root, tmp_path / "data"))


@pytest.fixture
def private_convnet(convnet_run: support.FinishedRun, tmp_path: Path) -> support.FinishedRun:
    """A copy to damage. Stages recorded the original's path, so here they need --force."""
    run_dir = Path(shutil.copytree(convnet_run.run_dir, tmp_path / "convnet"))
    return convnet_run._replace(run_dir=run_dir)
