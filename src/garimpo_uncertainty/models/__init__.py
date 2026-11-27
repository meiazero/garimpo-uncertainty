"""The segmentation models, each registered by architecture name, and the contract they meet.

A model takes (B, C, H, W) and returns raw logits (B, classes, H, W), with no sigmoid or softmax.
COMPASS also needs to split it into f, up to a layer, and g, from there to the logits.
"""

import re
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Self

import torch

import garimpo_uncertainty.registry

# Shared by both models; they still differ in decoder and output stride, 32 against DeepLabV3's 8.
ENCODER = "resnet18"
LOGITS = "logits"
MODEL_REGISTRY = garimpo_uncertainty.registry.Registry("architecture")

Decode = Callable[[torch.Tensor], torch.Tensor]


class SegmentationAdapter(torch.nn.Module):
    """Mixed into a library's model class, so its state_dict keys and layer names stay its own."""

    # The decoder layers COMPASS may split at; never a module that only holds them.
    feature_layer: ClassVar[re.Pattern]
    # Starting weights the encoder may load, by the model name each gives in the results.
    variants: ClassVar[dict[str, str]]

    @classmethod
    def create(
        cls,
        in_channels: int,
        num_classes: int,
        encoder_weights: str | None,
        weights_file: Path | None,
    ) -> Self:
        raise NotImplementedError

    def feature_layers(self) -> list[str]:
        """Return the layers COMPASS-J may split at, shallow to deep, then the logits."""
        names = [name for name, _ in self.named_modules() if self.feature_layer.fullmatch(name)]
        if not names:
            children = [name for name, _ in self.named_children()]
            raise ValueError(
                f"no decoder layer matches {self.feature_layer.pattern}; children: {children}"
            )
        return [*names, LOGITS]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return the encoder's output for a batch; every g(z) on it reads from this one pass."""
        return self.encoder(x)

    def split_at(self, layer: str, features: list[torch.Tensor]) -> tuple[torch.Tensor, Decode]:
        """Return z at `layer` and g, which maps z to logits running only the layers after it.

        COMPASS evaluates g dozens of times per batch, so g must not depend on the encoder's cost.
        """
        raise NotImplementedError


def check_encoder_weights(arch: str, encoder_weights: str | None) -> None:
    """Raise when a choice of starting weights does not fit the architecture."""
    variants = MODEL_REGISTRY[arch].variants
    if encoder_weights is None and variants:
        raise ValueError(f"--encoder-weights is required for {arch}: {', '.join(variants)}")
    if encoder_weights is not None and encoder_weights not in variants:
        takes = f"takes {', '.join(variants)}" if variants else "always starts from random weights"
        raise ValueError(f"--encoder-weights is invalid for {arch}, which {takes}")


def variant_name(arch: str, encoder_weights: str | None) -> str:
    """Return the model's name in the results, which arch alone would merge across weights."""
    return MODEL_REGISTRY[arch].variants.get(encoder_weights, arch)


def build(
    arch: str,
    in_channels: int,
    num_classes: int = 1,
    encoder_weights: str | None = None,
    weights_file: Path | None = None,
) -> torch.nn.Module:
    """Build a model; leave `encoder_weights` out to rebuild one that a checkpoint will fill.

    Only the start of a training reads pretrained weights, so later stages never need them.
    """
    return MODEL_REGISTRY[arch].create(in_channels, num_classes, encoder_weights, weights_file)


# Each architecture registers itself on import; they come last, as they subclass the contract above.
import garimpo_uncertainty.models.convnet  # noqa: E402
import garimpo_uncertainty.models.unet  # noqa: E402
