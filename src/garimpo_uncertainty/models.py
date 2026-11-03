"""The two segmentation models of the proposal: a plain ConvNet and an smp U-Net."""

import re

import compass2d.decomposition
import segmentation_models_pytorch
import torch

FEATURE_LAYER = re.compile(r"(up|decoder\.blocks)\.\d+")


def _conv_block(in_channels: int, out_channels: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True),
    )


class ConvNet(torch.nn.Module):
    """Down and up convolutions with no skip connections, the proposal's simple model."""

    def __init__(self, in_channels: int, channels: list[int]) -> None:
        super().__init__()
        widths = [in_channels, *channels]
        self.down = torch.nn.ModuleList(
            _conv_block(a, b) for a, b in zip(widths[:-1], widths[1:], strict=True)
        )
        self.pool = torch.nn.MaxPool2d(2)
        reverse = channels[::-1]
        self.up = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Upsample(scale_factor=2), _conv_block(a, b))
            for a, b in zip(reverse[:-1], reverse[1:], strict=True)
        )
        self.head = torch.nn.Conv2d(channels[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, block in enumerate(self.down):
            x = block(x)
            if index < len(self.down) - 1:
                x = self.pool(x)
        for block in self.up:
            x = block(x)
        return self.head(x)


def build(
    arch: str, in_channels: int, channels: list[int] | None = None, encoder: str | None = None
) -> torch.nn.Module:
    if arch == "convnet":
        if not channels:
            raise ValueError("convnet needs channels")
        return ConvNet(in_channels, channels)
    if arch == "unet":
        if not encoder:
            raise ValueError("unet needs an encoder name")
        # encoder_weights=None: ImageNet weights do not fit 6 bands, and compute nodes are offline.
        return segmentation_models_pytorch.Unet(
            encoder_name=encoder, encoder_weights=None, in_channels=in_channels, classes=1
        )
    raise ValueError(f"unknown architecture: {arch!r}")


def feature_layers(model: torch.nn.Module) -> list[str]:
    """Return decoder layers usable by COMPASS-J, shallow to deep, then the logits."""
    names = [name for name, _ in model.named_modules() if FEATURE_LAYER.fullmatch(name)]
    if not names:
        children = [name for name, _ in model.named_children()]
        raise ValueError(f"no decoder layer matches {FEATURE_LAYER.pattern}; children: {children}")
    return [*names, compass2d.decomposition.LOGITS]
