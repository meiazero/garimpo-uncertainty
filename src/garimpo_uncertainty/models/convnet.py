"""The ConvNet is smp's DeepLabV3: its decoder reads only the deepest feature, no skip connections.

Every argument not passed here keeps its smp 0.6.0 default (docs/decisions.md).
"""

import re
from pathlib import Path
from typing import Self

import segmentation_models_pytorch
import torch

# A from-import, as garimpo_uncertainty.models imports this module while it initializes.
from garimpo_uncertainty.models import ENCODER, LOGITS, MODEL_REGISTRY, SegmentationAdapter


@MODEL_REGISTRY.register("convnet")
class ConvNet(SegmentationAdapter, segmentation_models_pytorch.DeepLabV3):
    # The ASPP and the decoder's output; never `decoder.conv` or `decoder.bn`, mid-chain.
    feature_layer = re.compile(r"decoder\.(aspp|relu)")
    variants: dict[str, str] = {}  # noqa: RUF012 - read only, like the U-Net's

    @classmethod
    def create(
        cls,
        in_channels: int,
        num_classes: int,
        encoder_weights: str | None,  # noqa: ARG003 - always random, checked upstream
        weights_file: Path | None,  # noqa: ARG003
    ) -> Self:
        """Build the ConvNet from random weights, on the U-Net's encoder."""
        return cls(
            encoder_name=ENCODER,
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
        )

    def split_at(
        self, layer: str, features: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.nn.Sequential]:
        # Mirrors DeepLabV3Decoder.forward of smp 0.6.0: one chain on the deepest feature alone.
        chain = {
            "decoder.aspp": self.decoder.aspp,
            "decoder.conv": self.decoder.conv,
            "decoder.bn": self.decoder.bn,
            "decoder.relu": self.decoder.relu,
            LOGITS: self.segmentation_head,
        }
        stop = list(chain).index(layer) + 1
        modules = list(chain.values())
        with torch.no_grad():
            z = torch.nn.Sequential(*modules[:stop])(features[-1])
        return z, torch.nn.Sequential(*modules[stop:])
