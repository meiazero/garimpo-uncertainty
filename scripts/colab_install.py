"""Install requirements.txt into Colab's Python, unless pip would replace its CUDA build.

requirements.txt pins Colab's published versions, but the live runtime can trail them by a day or
two. pip then aligns those packages to the pins; another torch or CUDA library, though, picks other
GPU kernels and so other bits, so the install stops there. Exits 1 then, else with pip's code.
"""

import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"


def is_cuda_build(name: str) -> bool:
    """Tell a wheel of the CUDA build, which picks the GPU's kernels and so its bits."""
    return name in {"torch", "torchvision", "triton"} or name.startswith(("nvidia-", "cuda-"))


def replaced_by_install(requirements: Path = REQUIREMENTS) -> list[tuple[str, str, str]]:
    """Return name, installed version and pin of each installed package the pins do not admit."""
    replaced = []
    pins = re.findall(
        r"^([\w.-]+)==([^\s;]+)", requirements.read_text(encoding="utf-8"), re.MULTILINE
    )
    for name, pin in pins:
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue  # pip installs it, replacing nothing
        if not SpecifierSet(f"=={pin}").contains(installed, prereleases=True):
            replaced.append((canonicalize_name(name), installed, pin))
    return replaced


def main() -> int:
    """Install, or name the CUDA build pip would replace and return 1."""
    replaced = {
        name: f"{name} {installed} (pinned {pin})" for name, installed, pin in replaced_by_install()
    }
    if blocking := [text for name, text in replaced.items() if is_cuda_build(name)]:
        print(
            f"pip would replace Colab's CUDA build, {', '.join(blocking)}: "
            "pick the latest runtime version"
        )
        return 1
    if replaced:
        print("pip aligns to the pins:", ", ".join(replaced.values()))
    install = [sys.executable, "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS)]
    return subprocess.run(install, check=False).returncode  # noqa: S603


if __name__ == "__main__":
    sys.exit(main())
