"""The segmentation models: a plain ConvNet and a ResNet-50 smp U-Net that may start pretrained."""

import re
from collections.abc import Callable
from pathlib import Path

import safetensors.torch
import segmentation_models_pytorch
import torch

import garimpo_uncertainty.data

FEATURE_LAYER = re.compile(r"(up|decoder\.blocks)\.\d+")
LOGITS = "logits"
# Each choice names one model in every results table.
UNET_VARIANTS = {
    "none": "unet_scratch",
    "imagenet": "unet_imagenet",
    "sentinel2_moco": "unet_sentinel2",
}
UNET_ENCODER = "resnet50"
SENTINEL2_MOCO_BANDS = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
)


def _conv_block(in_channels: int, out_channels: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True),
    )


class ConvNet(torch.nn.Module):
    """Down and up convolutions with no skip connections, the simple model compared."""

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for index, block in enumerate(self.down):
            x = block(x)
            if index < len(self.down) - 1:
                x = self.pool(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        for block in self.up:
            x = block(x)
        return self.head(x)


def check_encoder_weights(arch: str, encoder_weights: str | None) -> None:
    """Raise when a choice of starting weights does not fit the model."""
    if arch == "convnet" and encoder_weights is not None:
        raise ValueError("--encoder-weights is invalid for convnet, which has no encoder")
    if arch == "unet" and encoder_weights is None:
        raise ValueError(f"--encoder-weights is required for unet: {', '.join(UNET_VARIANTS)}")


def variant_name(arch: str, encoder_weights: str | None) -> str:
    """Return the model's name in the results, which arch alone would merge across weights."""
    return UNET_VARIANTS[encoder_weights] if arch == "unet" else arch


def load_imagenet(model: torch.nn.Module, path: Path) -> None:
    """Load ImageNet's ResNet-50 into a U-Net encoder, fitting its RGB filters to our bands.

    Cycles and rescales conv1 as smp's patch_first_conv does, without going through its hub lookup.
    """
    file_state = safetensors.torch.load_file(path)
    state = {key: value for key, value in file_state.items() if not key.startswith("fc.")}
    conv1 = state.get("conv1.weight")
    if conv1 is None or conv1.shape[1] != 3:
        shape = None if conv1 is None else tuple(conv1.shape)
        raise ValueError(f"{path}: conv1.weight must have 3 bands, found shape {shape}")
    n_bands = model.encoder.conv1.in_channels
    state["conv1.weight"] = conv1[:, [band % 3 for band in range(n_bands)]] * (3 / n_bands)
    model.encoder.load_state_dict(state)


def load_sentinel2_moco(model: torch.nn.Module, path: Path) -> None:
    """Load a 13-band MoCo ResNet-50 into a U-Net encoder, keeping the columns of our bands.

    The kept filters are not rescaled by 13/6: that factor assumes every band contributes alike, and
    bn1 follows conv1, so training re-estimates the activation scale. smp's 3/in_channels differs,
    since it compensates for filters it repeats, while nothing is repeated here.
    """
    file_state = torch.load(path, map_location="cpu", weights_only=True)
    state = {key: value for key, value in file_state.items() if not key.startswith("fc.")}
    conv1 = state.get("conv1.weight")
    if conv1 is None or conv1.shape[1] != len(SENTINEL2_MOCO_BANDS):
        shape = None if conv1 is None else tuple(conv1.shape)
        raise ValueError(f"{path}: conv1.weight must have 13 bands, found shape {shape}")
    columns = [SENTINEL2_MOCO_BANDS.index(band) for band in garimpo_uncertainty.data.BANDS]
    state["conv1.weight"] = conv1[:, columns]
    model.encoder.load_state_dict(state)


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
        return ConvNet(in_channels, channels)
    if arch == "unet":
        model = segmentation_models_pytorch.Unet(
            encoder_name=UNET_ENCODER, encoder_weights=None, in_channels=in_channels, classes=1
        )
        if encoder_weights == "imagenet":
            load_imagenet(model, weights_file)
        if encoder_weights == "sentinel2_moco":
            load_sentinel2_moco(model, weights_file)
        return model
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
    return [model.encode(x)] if isinstance(model, ConvNet) else model.encoder(x)


def _run_blocks(
    model: torch.nn.Module, features: list[torch.Tensor], z: torch.Tensor, blocks: range
) -> torch.Tensor:
    if isinstance(model, ConvNet):
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
    is_convnet = isinstance(model, ConvNet)
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
