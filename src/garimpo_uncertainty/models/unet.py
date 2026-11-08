"""The smp U-Net's fixed ResNet-18 encoder and the starting weights it may load."""

from pathlib import Path

import safetensors.torch
import segmentation_models_pytorch
import torch

import garimpo_uncertainty.data
import garimpo_uncertainty.models

# Each choice names one model in every results table.
UNET_VARIANTS = {
    "none": "unet_scratch",
    "imagenet": "unet_imagenet",
    "sentinel2_moco": "unet_sentinel2",
}
IMAGENET_N_BANDS = 3
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


def build(
    in_channels: int, encoder_weights: str, weights_file: Path | None
) -> segmentation_models_pytorch.Unet:
    """Build the U-Net, loading the chosen starting weights into its encoder."""
    model = segmentation_models_pytorch.Unet(
        encoder_name=garimpo_uncertainty.models.ENCODER,
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
    )
    if encoder_weights == "imagenet":
        load_imagenet(model, weights_file)
    if encoder_weights == "sentinel2_moco":
        load_sentinel2_moco(model, weights_file)
    return model


def load_imagenet(model: torch.nn.Module, path: Path) -> None:
    """Load ImageNet's ResNet-18 into a U-Net encoder, fitting its RGB filters to our bands.

    Cycles and rescales conv1 as smp's patch_first_conv does, without going through its hub lookup.
    """
    file_state = safetensors.torch.load_file(path)
    state = {key: value for key, value in file_state.items() if not key.startswith("fc.")}
    conv1 = state.get("conv1.weight")
    if conv1 is None or conv1.shape[1] != IMAGENET_N_BANDS:
        shape = None if conv1 is None else tuple(conv1.shape)
        raise ValueError(
            f"{path}: conv1.weight must have {IMAGENET_N_BANDS} bands, found shape {shape}"
        )
    n_bands = model.encoder.conv1.in_channels
    columns = [band % IMAGENET_N_BANDS for band in range(n_bands)]
    state["conv1.weight"] = conv1[:, columns] * (IMAGENET_N_BANDS / n_bands)
    model.encoder.load_state_dict(state)


def load_sentinel2_moco(model: torch.nn.Module, path: Path) -> None:
    """Load a 13-band MoCo ResNet-18 into a U-Net encoder, keeping the columns of our bands.

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
