"""The f/g decomposition of each architecture and the loading of starting weights."""

from pathlib import Path

import pytest
import safetensors.torch
import torch

import garimpo_uncertainty.models
from tests.e2e import support

IMAGENET_ARTIFACTS = support.ROOT / "tests" / "artifacts" / "imagenet"
WEIGHTS_ARTIFACTS = support.ROOT / "tests" / "artifacts" / "weights"
# Literal positions of B2, B3, B4, B8, B11, B12 among the 13 bands, kept apart from the pipeline.
SENTINEL2_MOCO_COLUMNS = (1, 2, 3, 7, 11, 12)


@pytest.mark.parametrize(
    ("arch", "options"),
    [("convnet", {"channels": [8, 16, 32]}), ("unet", {})],
    ids=["convnet", "unet"],
)
def test_decomposition_reproduces_the_model(arch: str, options: dict) -> None:
    """F then g must reproduce the model at every layer, and z must really drive the output."""
    torch.manual_seed(0)
    x = torch.randn(2, 6, support.CELL_PX, support.CELL_PX)
    model = garimpo_uncertainty.models.build(arch, in_channels=6, **options).eval()
    layers = garimpo_uncertainty.models.feature_layers(model)
    assert len(layers) > 1
    assert layers[-1] == garimpo_uncertainty.models.LOGITS
    first_layer = model.down[0] if arch == "convnet" else model.encoder.conv1
    encoder_passes = []
    first_layer.register_forward_hook(lambda *_, passes=encoder_passes: passes.append(1))
    with torch.no_grad():
        expected = model(x)
        encoder_passes.clear()
        features = garimpo_uncertainty.models.encode(model, x)
        for layer in layers:
            z, decode = garimpo_uncertainty.models.split_at(model, layer, features)
            assert torch.allclose(decode(z), expected, atol=1e-5), f"{arch}/{layer}"
            changed = decode(z + 1.0)
            assert not torch.allclose(changed, expected, atol=1e-3), f"{arch}/{layer} ignored z"
    assert len(encoder_passes) == 1, f"{arch}: g reran the encoder"


@pytest.fixture(scope="module")
def imagenet_file() -> tuple[Path, dict[str, torch.Tensor]]:
    IMAGENET_ARTIFACTS.mkdir(parents=True)
    path = IMAGENET_ARTIFACTS / "synthetic.safetensors"
    encoder = garimpo_uncertainty.models.build("unet", 3).encoder
    state = {key: value.clone() for key, value in encoder.state_dict().items()}
    safetensors.torch.save_file(state, path)
    return path, state


def test_imagenet_weights_cycle_and_rescale_rgb(
    imagenet_file: tuple[Path, dict[str, torch.Tensor]],
) -> None:
    path, state = imagenet_file
    model = garimpo_uncertainty.models.build(
        "unet", 6, encoder_weights="imagenet", weights_file=path
    )
    loaded = model.encoder.state_dict()
    for band in range(6):
        cycled = state["conv1.weight"][:, band % 3] * 0.5
        assert torch.equal(loaded["conv1.weight"][:, band], cycled)
    assert torch.equal(loaded["layer4.1.conv2.weight"], state["layer4.1.conv2.weight"])


def test_imagenet_weights_refuse_another_band_count(
    imagenet_file: tuple[Path, dict[str, torch.Tensor]],
) -> None:
    _, state = imagenet_file
    path = IMAGENET_ARTIFACTS / "two_bands.safetensors"
    two_bands = state["conv1.weight"][:, :2].clone()
    safetensors.torch.save_file(state | {"conv1.weight": two_bands}, path)
    with pytest.raises(ValueError, match="3 bands"):
        garimpo_uncertainty.models.build("unet", 6, encoder_weights="imagenet", weights_file=path)


@pytest.fixture(scope="module")
def sentinel2_file() -> tuple[Path, dict[str, torch.Tensor]]:
    path = WEIGHTS_ARTIFACTS / "synthetic.pth"
    return path, support.write_synthetic_sentinel2_weights(path)


def build_from_sentinel2(path: Path) -> torch.nn.Module:
    return garimpo_uncertainty.models.build(
        "unet", 6, encoder_weights="sentinel2_moco", weights_file=path
    )


def test_sentinel2_weights_keep_our_bands_in_order(
    sentinel2_file: tuple[Path, dict[str, torch.Tensor]],
) -> None:
    path, state = sentinel2_file
    loaded = build_from_sentinel2(path).encoder.state_dict()
    for position, column in enumerate(SENTINEL2_MOCO_COLUMNS):
        assert torch.equal(loaded["conv1.weight"][:, position], state["conv1.weight"][:, column])
    copied = set(state) - {"conv1.weight", "fc.weight", "fc.bias"}
    assert copied == set(loaded) - {"conv1.weight"}
    for key in copied:
        assert torch.equal(loaded[key], state[key]), key

    # Basic blocks, two per stage: a ResNet-18 layout, which a deeper ResNet's keys would not match.
    assert "layer4.1.conv2.weight" in copied
    assert not any(".conv3." in key for key in copied)


def test_layer_names_stay_valid_across_starting_weights(
    sentinel2_file: tuple[Path, dict[str, torch.Tensor]],
) -> None:
    """RUN_ORDER relies on these two layers."""
    path, _ = sentinel2_file
    scratch = garimpo_uncertainty.models.build("unet", 6)
    assert garimpo_uncertainty.models.feature_layers(
        build_from_sentinel2(path)
    ) == garimpo_uncertainty.models.feature_layers(scratch)
    assert {"decoder.blocks.3", "decoder.blocks.4"} <= set(
        garimpo_uncertainty.models.feature_layers(scratch)
    )


@pytest.mark.parametrize(
    ("defect", "message"),
    [("missing", "Missing key"), ("unexpected", "Unexpected key"), ("wrong_bands", "13 bands")],
)
def test_sentinel2_weights_refuse_a_broken_file(
    sentinel2_file: tuple[Path, dict[str, torch.Tensor]], defect: str, message: str
) -> None:
    _, state = sentinel2_file
    broken = {
        "missing": {key: value for key, value in state.items() if key != "layer4.1.bn2.weight"},
        "unexpected": state | {"layer5.0.conv1.weight": torch.zeros(1)},
        "wrong_bands": state | {"conv1.weight": state["conv1.weight"][:, :6]},
    }[defect]
    path = WEIGHTS_ARTIFACTS / f"{defect}.pth"
    torch.save(broken, path)
    with pytest.raises((RuntimeError, ValueError), match=message):
        build_from_sentinel2(path)
