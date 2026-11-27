"""The parts a config swaps in training: registries, early stopping, two classes, Albumentations."""

import albumentations
import albumentations.pytorch
import compass2d.directions
import compass2d.metrics
import numpy as np
import pytest
import torch

import garimpo_uncertainty.config
import garimpo_uncertainty.intervals
import garimpo_uncertainty.models
import garimpo_uncertainty.postprocess
import garimpo_uncertainty.registry
import garimpo_uncertainty.training.augmentations
import garimpo_uncertainty.training.callbacks
import garimpo_uncertainty.training.loop
import garimpo_uncertainty.training.losses
import garimpo_uncertainty.training.optimizers


def test_registry_names_what_it_knows_and_refuses_a_second_entry() -> None:
    registry = garimpo_uncertainty.registry.Registry("part")
    registry.register("one")(dict)
    with pytest.raises(ValueError, match="registered twice"):
        registry.register("one")(list)
    with pytest.raises(ValueError, match="unknown part 'two'; known: one"):
        registry.build("two")
    assert registry.build({"name": "one", "size": 3}) == {"size": 3}
    with pytest.raises(ValueError, match="needs a name"):
        garimpo_uncertainty.registry.parse_spec('{"size": 3}')


def test_a_spec_reads_the_same_whatever_order_its_keys_were_typed_in() -> None:
    """A resume compares a spec's text with last.pt's, so the order must not refuse one."""
    parse = garimpo_uncertainty.registry.parse_spec
    typed = parse('{"name": "early_stopping", "patience": 2, "monitor": "val_loss"}')
    assert str(typed) == str(
        parse('{"monitor": "val_loss", "patience": 2, "name": "early_stopping"}')
    )


def test_dice_bce_is_the_sum_of_its_two_losses_to_the_bit() -> None:
    torch.manual_seed(0)
    logits, mask = torch.randn(2, 1, 8, 8), (torch.rand(2, 1, 8, 8) > 0.5).float()
    losses = garimpo_uncertainty.training.losses
    combined = losses.LOSS_REGISTRY.build("dice_bce")(logits, mask)
    assert torch.equal(combined, losses.DiceLoss()(logits, mask) + losses.BCELoss()(logits, mask))


def stopper(
    patience: int, **options: object
) -> garimpo_uncertainty.training.callbacks.EarlyStopping:
    spec = {"name": "early_stopping", "patience": patience, **options}
    return garimpo_uncertainty.training.callbacks.CALLBACK_REGISTRY.build(spec)


def state() -> garimpo_uncertainty.training.loop.TrainingState:
    return garimpo_uncertainty.training.loop.TrainingState(
        *(None,) * 5,
        metadata={},
        arguments={},
        history=[],
        best_dice=-1.0,
        first_epoch=0,
        run_dir=None,
    )


@pytest.mark.parametrize(
    ("dice", "options", "stops_after"),
    [
        ([0.5, 0.6, 0.6, 0.6], {}, 3),
        # A NaN never counts as an improvement.
        ([0.5, float("nan"), float("nan")], {}, 2),
        # Within min_delta of the best, an epoch does not improve.
        ([0.5, 0.55, 0.58], {"min_delta": 0.1}, 2),
        ([0.5, 0.6, 0.7, 0.8], {}, None),
        ([0.9, 0.8, 0.7], {"monitor": "val_loss", "mode": "min"}, None),
    ],
)
def test_early_stopping_waits_patience_epochs_without_improvement(
    dice: list[float], options: dict, stops_after: int | None
) -> None:
    callback, training = stopper(2, **options), state()
    monitor = options.get("monitor", "val_dice")
    stopped_at = None
    for epoch, value in enumerate(dice):
        callback.on_epoch_end(training, {"epoch": epoch, monitor: value})
        if training.stopped and stopped_at is None:
            stopped_at = epoch
    assert stopped_at == (None if stops_after is None else stops_after)


def test_early_stopping_resumes_from_what_last_pt_kept() -> None:
    first, training = stopper(2), state()
    for epoch, value in enumerate([0.5, 0.4]):
        first.on_epoch_end(training, {"epoch": epoch, "val_dice": value})
    # What CheckpointCallback writes and a resume reads back, then a fresh callback.
    kept = {name: dict(values) for name, values in training.callbacks.items()}
    resumed = state()
    resumed.callbacks = kept
    stopper(2).on_epoch_end(resumed, {"epoch": 2, "val_dice": 0.45})
    assert resumed.stopped


@pytest.mark.parametrize(
    "options",
    [
        {"patience": 0},
        {"patience": True},
        {"patience": 2, "mode": "best"},
        {"patience": 2, "monitor": None},
    ],
)
def test_early_stopping_refuses_options_that_never_stop(options: dict) -> None:
    with pytest.raises(ValueError, match="early_stopping"):
        stopper(**options)


def test_two_classes_train_and_score_through_the_loop() -> None:
    """A two-logit U-Net with cross-entropy and argmax runs the fixed loop end to end."""
    torch.manual_seed(0)
    model = garimpo_uncertainty.models.build("unet", in_channels=3, num_classes=2)
    images = torch.randn(4, 3, 64, 64)
    masks = (torch.rand(4, 1, 64, 64) > 0.5).float()
    data = torch.utils.data.TensorDataset(images, masks, torch.arange(4))
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    training = garimpo_uncertainty.training.loop.TrainingState(
        model=model,
        optimizer=garimpo_uncertainty.training.optimizers.OPTIMIZER_REGISTRY.build(
            {"name": "sgd", "momentum": 0.9}, params=model.parameters(), lr=0.01, weight_decay=0
        ),
        loss_fn=garimpo_uncertainty.training.losses.LOSS_REGISTRY.build("cross_entropy"),
        postprocess=garimpo_uncertainty.postprocess.POSTPROCESS_REGISTRY.build("softmax_argmax"),
        metric_names=["dice", "precision"],
        metadata={},
        arguments={},
        history=[],
        best_dice=-1.0,
        first_epoch=0,
        run_dir=None,
    )
    loaders = garimpo_uncertainty.training.loop.Loaders(loader, loader, loader)
    trainer = garimpo_uncertainty.training.loop.Trainer(training, "cpu", 1, [])
    assert trainer.fit(loaders) == [2, 3, 64, 64]
    (record,) = training.history
    assert record["updates"] == 2
    assert 0 <= record["val_dice"] <= 1
    assert np.isfinite(record["val_loss"])


@pytest.mark.parametrize(
    "transform",
    [
        {},
        {"__class_fullname__": "Nope"},
        None,
        "flip",
        {"__class_fullname__": "Compose", "transforms": ["x"]},
    ],
)
def test_albumentations_refuses_what_is_not_a_serialized_pipeline(transform: object) -> None:
    with pytest.raises(ValueError, match="not a serialized pipeline"):
        garimpo_uncertainty.training.augmentations.AUGMENTATION_REGISTRY.build(
            {"name": "albumentations", "transform": transform}
        )


@pytest.mark.parametrize(
    "step",
    [
        albumentations.FromFloat(dtype="uint8"),
        albumentations.FromFloat(dtype="uint8", p=0.5),
        albumentations.CenterCrop(510, 510, p=0.5),
    ],
    ids=["dtype", "dtype on some draws", "shape on some draws"],
)
def test_albumentations_refuses_a_pipeline_that_changes_the_cell(step: object) -> None:
    pipeline = albumentations.Compose([step])
    with pytest.raises(ValueError, match="must keep a float32 cell"):
        garimpo_uncertainty.training.augmentations.AUGMENTATION_REGISTRY.build(
            {"name": "albumentations", "transform": albumentations.to_dict(pipeline)["transform"]}
        )


def test_the_architectures_that_need_starting_weights_are_the_registry_s() -> None:
    registry = garimpo_uncertainty.models.MODEL_REGISTRY
    weighted = {arch for arch in registry.names() if registry[arch].variants}
    assert weighted == set(garimpo_uncertainty.config.WEIGHTED_ARCHS)


def test_albumentations_refuses_a_pipeline_that_returns_tensors() -> None:
    pipeline = albumentations.Compose(
        [albumentations.HorizontalFlip(), albumentations.pytorch.ToTensorV2()]
    )
    with pytest.raises(ValueError, match="leave out ToTensorV2"):
        garimpo_uncertainty.training.augmentations.AUGMENTATION_REGISTRY.build(
            {"name": "albumentations", "transform": albumentations.to_dict(pipeline)["transform"]}
        )


def test_albumentations_repeats_the_draw_it_is_given() -> None:
    """The seed comes from the sampler, so a worker applies what the main process drew."""
    pipeline = albumentations.Compose([albumentations.HorizontalFlip(p=0.5)])
    augmentation = garimpo_uncertainty.training.augmentations.AUGMENTATION_REGISTRY.build(
        {"name": "albumentations", "transform": albumentations.to_dict(pipeline)["transform"]}
    )
    image = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    mask = (image[:1] > 10).astype(np.float32)
    generator = torch.Generator().manual_seed(0)
    flipped = set()
    for _ in range(8):
        drawn = augmentation.draw(generator)
        first = augmentation(image, mask, drawn)
        again = augmentation(image, mask, drawn)
        assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
        assert first[0].shape == image.shape
        assert first[1].shape == mask.shape
        # The mask follows the image, as a flip must move both.
        is_flipped = np.array_equal(first[0], image[..., ::-1])
        assert np.array_equal(first[1], mask[..., ::-1] if is_flipped else mask)
        flipped.add(is_flipped)
    assert flipped == {True, False}


def test_both_post_processings_mark_a_pixel_mined_from_a_probability_of_one_half() -> None:
    probabilities = torch.tensor([0.2, 0.5, 0.9]).reshape(1, 1, 1, 3)
    logits = torch.logit(probabilities)
    build = garimpo_uncertainty.postprocess.POSTPROCESS_REGISTRY.build
    expected = probabilities >= garimpo_uncertainty.postprocess.THRESHOLD
    assert torch.equal(build("sigmoid_threshold")(logits), expected)
    # Two logits whose softmax gives class 1 the same probabilities; a tie goes to class 0.
    two = torch.cat([torch.zeros_like(logits), logits], dim=1)
    assert build("softmax_argmax")(two).tolist() == [[[[False, False, True]]]]


def test_the_interval_methods_name_every_direction_compass2d_registers() -> None:
    intervals = garimpo_uncertainty.intervals
    assert tuple(compass2d.directions.DIRECTIONS) == intervals.DIRECTIONS
    assert tuple(f"a_{m}" for m in intervals.COMPASS_METHODS) == intervals.ASYMMETRIC_METHODS
    combined = {method for method, _ in intervals.COMBINATIONS}
    assert {*intervals.COMPASS_METHODS, *intervals.ASYMMETRIC_METHODS} <= combined


def test_compass2d_hard_area_counts_the_pixels_the_threshold_marks_mined() -> None:
    logits = torch.tensor([-1.0, -0.0, 0.0, 2.0]).reshape(1, 1, 2, 2)
    marked = garimpo_uncertainty.postprocess.POSTPROCESS_REGISTRY.build("sigmoid_threshold")(logits)
    area = compass2d.metrics.hard_area(logits, pixel_area=1.0)
    assert area.tolist() == [float(marked.sum())] == [3.0]
