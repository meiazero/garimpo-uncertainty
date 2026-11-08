"""The ConvNet is smp's DeepLabV3: its decoder reads only the deepest feature, no skip connections.

Every argument not passed here keeps its smp 0.6.0 default (docs/decisions.md).
"""

import segmentation_models_pytorch

import garimpo_uncertainty.models


def build(in_channels: int) -> segmentation_models_pytorch.DeepLabV3:
    """Build the ConvNet from random weights, on the U-Net's encoder."""
    return segmentation_models_pytorch.DeepLabV3(
        encoder_name=garimpo_uncertainty.models.ENCODER,
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
    )
