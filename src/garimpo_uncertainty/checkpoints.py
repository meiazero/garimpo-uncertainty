"""Checkpoint metadata: enough to rebuild the exact model and to refuse mismatched data."""

import hashlib
import logging
from pathlib import Path

import pandas as pd
import torch

import garimpo_uncertainty.data
import garimpo_uncertainty.models.common
import garimpo_uncertainty.stages

logger = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def warn_on_code_change(metadata: dict) -> None:
    """Warn, not fail: adjusting code between stages is expected while debugging the pilot."""
    current = garimpo_uncertainty.stages.code_versions()
    for key in ("root_commit", "compass_commit"):
        if metadata[key] != current[key]:
            logger.warning(
                "warning: %s is %s, the checkpoint came from %s", key, current[key], metadata[key]
            )


def save(path: Path, content: dict) -> None:
    with garimpo_uncertainty.stages.atomic(path) as temporary:
        torch.save(content, temporary)


def load_model(checkpoint: Path, cells: pd.DataFrame, device: str) -> tuple[torch.nn.Module, dict]:
    """Model in eval mode with frozen weights; stages after training never update it."""
    content = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = content["meta"]
    garimpo_uncertainty.data.check_split(metadata, cells)
    warn_on_code_change(metadata)
    model = garimpo_uncertainty.models.common.build(
        metadata["arch"],
        in_channels=metadata["in_channels"],
    )
    model.load_state_dict(content["model"])
    return model.eval().requires_grad_(requires_grad=False).to(device), metadata


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
