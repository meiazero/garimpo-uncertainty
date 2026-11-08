"""What both models share: building one by name and splitting it into f and g for COMPASS."""

import re
from collections.abc import Callable
from pathlib import Path

import segmentation_models_pytorch
import torch

import garimpo_uncertainty.models.convnet
import garimpo_uncertainty.models.unet

FEATURE_LAYER = re.compile(r"(up|decoder\.blocks)\.\d+")
LOGITS = "logits"


def _build_unet(
    in_channels: int, encoder_weights: str, weights_file: Path | None
) -> segmentation_models_pytorch.Unet:
    model = segmentation_models_pytorch.Unet(
        encoder_name=garimpo_uncertainty.models.unet.UNET_ENCODER,
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
    )
    if encoder_weights == "imagenet":
        garimpo_uncertainty.models.unet.load_imagenet(model, weights_file)
    if encoder_weights == "sentinel2_moco":
        garimpo_uncertainty.models.unet.load_sentinel2_moco(model, weights_file)
    return model


def build(
    arch: str,
    in_channels: int,
    channels: list[int] | None = None,
    encoder_weights: str = "none",
    weights_file: Path | None = None,
) -> torch.nn.Module:
    """Build a model; leave `encoder_weights` as none to rebuild one that a checkpoint will fill.

    Only the start of a training reads pretrained weights, so later stages never need them.
    """
    if arch == "convnet":
        if not channels:
            raise ValueError("convnet needs channels")
        return garimpo_uncertainty.models.convnet.ConvNet(in_channels, channels)
    if arch == "unet":
        return _build_unet(in_channels, encoder_weights, weights_file)
    raise ValueError(f"unknown architecture: {arch!r}")


def feature_layers(model: torch.nn.Module) -> list[str]:
    """Return decoder layers usable by COMPASS-J, shallow to deep, then the logits."""
    names = [name for name, _ in model.named_modules() if FEATURE_LAYER.fullmatch(name)]
    if not names:
        children = [name for name, _ in model.named_children()]
        raise ValueError(f"no decoder layer matches {FEATURE_LAYER.pattern}; children: {children}")
    return [*names, LOGITS]


@torch.no_grad()
def encode(model: torch.nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    """Return the encoder's output for a batch; every g(z) on it reads from this one pass."""
    return (
        [model.encode(x)]
        if isinstance(model, garimpo_uncertainty.models.convnet.ConvNet)
        else model.encoder(x)
    )


def _run_blocks(
    model: torch.nn.Module, features: list[torch.Tensor], z: torch.Tensor, blocks: range
) -> torch.Tensor:
    if isinstance(model, garimpo_uncertainty.models.convnet.ConvNet):
        for index in blocks:
            z = model.up[index](z)
        return z
    # Mirrors UnetDecoder.forward of smp 0.6.0, which pyproject.toml pins for this reason.
    shapes = [feature.shape[2:] for feature in features][::-1]
    skips = features[1:-1][::-1]
    for index in blocks:
        height, width = shapes[index + 1]
        skip = skips[index] if index < len(skips) else None
        z = model.decoder.blocks[index](z, height, width, skip_connection=skip)
    return z


def split_at(
    model: torch.nn.Module, layer: str, features: list[torch.Tensor]
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Return z at `layer` and g, which maps z to logits running only the layers after `layer`.

    COMPASS evaluates g dozens of times per batch, so g must not depend on the encoder's cost.
    """
    is_convnet = isinstance(model, garimpo_uncertainty.models.convnet.ConvNet)
    total = len(model.up if is_convnet else model.decoder.blocks)
    head = model.head if is_convnet else model.segmentation_head
    stage = total - 1 if layer == LOGITS else int(layer.rsplit(".", 1)[1])
    with torch.no_grad():
        first = features[0] if is_convnet else model.decoder.center(features[-1])
        z = _run_blocks(model, features, first, range(stage + 1))
        if layer == LOGITS:
            return head(z), lambda logits: logits

    def decode(zi: torch.Tensor) -> torch.Tensor:
        return head(_run_blocks(model, features, zi, range(stage + 1, total)))

    return z, decode
