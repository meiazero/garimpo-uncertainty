"""Stage: trains one segmentation model. Writes artifacts only; every metric runs elsewhere.

No hyperparameter has a default, so a run states every value it uses.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

import garimpo_uncertainty.artifacts
import garimpo_uncertainty.cell_dataset
import garimpo_uncertainty.checkpoints
import garimpo_uncertainty.data
import garimpo_uncertainty.files
import garimpo_uncertainty.models
import garimpo_uncertainty.pipeline.runner
import garimpo_uncertainty.postprocess
import garimpo_uncertainty.registry
import garimpo_uncertainty.training.augmentations
import garimpo_uncertainty.training.callbacks
import garimpo_uncertainty.training.loop
import garimpo_uncertainty.training.losses
import garimpo_uncertainty.training.metrics
import garimpo_uncertainty.training.optimizers


def result_arguments(args: argparse.Namespace) -> dict[str, str]:
    """Arguments that shape the trained weights; batch size does, unlike in the later stages."""
    ignored = garimpo_uncertainty.pipeline.runner.IGNORED_ARGUMENTS - {"batch_size"}
    return {k: str(v) for k, v in vars(args).items() if k not in ignored}


def starting_weights_sha256(args: argparse.Namespace, root: Path) -> str | None:
    """Return the hash of the weights file or, without the file, the one this run started from.

    Only a fresh start reads the file, so its absence must not change a run's identity: that would
    refuse `train` and, with it, every later stage on a machine with `runs/` but no `weights/`.
    """
    weights_file = garimpo_uncertainty.data.starting_weights_path(root, args.encoder_weights)
    if weights_file is None:
        return None
    if weights_file.exists():
        return garimpo_uncertainty.files.file_sha256(weights_file)
    run = garimpo_uncertainty.artifacts.RunPaths.of(args.run_dir)
    if not run.run_meta.exists():
        garimpo_uncertainty.pipeline.runner.fail_stage(
            run.status,
            "train",
            f"train: cannot tell which weights this run used: {weights_file} and "
            f"{run.run_meta} are both absent. On a node with internet run "
            f"garimpo download_data weights --encoder-weights {args.encoder_weights}, or "
            "restore run_meta.json with the run.",
        )
    return garimpo_uncertainty.data.read_run_metadata(args.run_dir).get("encoder_weights_sha256")


# What a forced redo leaves in place of a finished run's last.pt.
DISCARDED = {"discarded": True}


def read_resume(args: argparse.Namespace) -> dict | None:
    """Return last.pt when this run continues it; refuse one written with other arguments."""
    last_path = garimpo_uncertainty.artifacts.RunPaths.of(args.run_dir).last_checkpoint
    if not last_path.exists() or args.force:
        return None
    resume = torch.load(last_path, map_location="cpu", weights_only=True)
    if resume == DISCARDED:
        return None
    arguments = result_arguments(args)
    if resume["arguments"] != arguments:
        different = sorted(k for k in arguments if resume["arguments"].get(k) != arguments[k])
        raise ValueError(
            f"last.pt was written with other values of {different}: rerun with those values, or "
            "with the tiles it trained on for tiles_sha256, to resume it, or delete last.pt to "
            "start over with these"
        )
    print(f"resuming after epoch {resume['epoch']}")
    return resume


def keep_unfinished_training(run_dir: Path, *, force: bool, discard_unfinished: bool) -> None:
    """Refuse --force over an unfinished run's last.pt, which training from scratch overwrites.

    --force-from train left in a Colab cell would otherwise restart after every lost session. A
    finished run's last.pt is marked discarded instead, so a forced redo cut short never resumes
    from its end: a deleted file would come back with the next restore from Drive, whose rsync
    has no --delete, where the mark replaces it.
    """
    run = garimpo_uncertainty.artifacts.RunPaths.of(run_dir)
    last = run.last_checkpoint
    if not force or not last.exists():
        return
    if torch.load(last, map_location="cpu", weights_only=True) == DISCARDED:
        return
    status = garimpo_uncertainty.pipeline.runner.read_status(run.status, "train")
    # Starting the work drops completed_fingerprint, so only a finished training still has it.
    if "completed_fingerprint" not in status and not discard_unfinished:
        garimpo_uncertainty.pipeline.runner.fail_stage(
            run.status,
            "train",
            f"{last} holds an unfinished training (status {status['status']}): "
            f"run without --force (--force-from train) to resume it, or delete {last} to start "
            "over",
        )
    garimpo_uncertainty.checkpoints.save(last, DISCARDED)


def new_metadata(
    args: argparse.Namespace,
    root: Path,
    cells: pd.DataFrame,
    train_cells: pd.DataFrame,
    tile_ids: list[int],
) -> dict:
    return {
        "arch": args.arch,
        "variant": garimpo_uncertainty.models.variant_name(args.arch, args.encoder_weights),
        "in_channels": len(garimpo_uncertainty.data.BANDS),
        "num_classes": args.num_classes,
        "encoder": garimpo_uncertainty.models.ENCODER,
        "encoder_weights": args.encoder_weights,
        "encoder_weights_sha256": args.encoder_weights_sha256,
        "seed": args.seed,
        # MoCo's filters saw reflectance / 10000; ImageNet's statistics do not fit our bands.
        "normalization": (
            garimpo_uncertainty.cell_dataset.identity_normalization()
            if args.encoder_weights == "sentinel2_moco"
            else garimpo_uncertainty.cell_dataset.band_statistics(
                root, train_cells, args.batch_size, args.num_workers
            )
        ),
        "train_fraction": args.train_fraction,
        "subset_seed": args.subset_seed,
        # Recorded, so later stages read this run's slots however much else the store holds.
        "split": str(args.split),
        "tile_ids": tile_ids,
        "split_hash": garimpo_uncertainty.data.split_hash(
            cells, args.train_fraction, args.subset_seed
        ),
        "arguments": {k: str(v) for k, v in vars(args).items()},
        **garimpo_uncertainty.files.code_versions(),
    }


def start_training(
    args: argparse.Namespace,
    device: str,
    metadata: dict,
    weights_file: Path | None,
    resume: dict | None,
) -> garimpo_uncertainty.training.loop.TrainingState:
    """Build the model and its optimizer, then restore both and the RNG from last.pt if resuming."""
    # A resume fills the model from last.pt, so only a fresh start reads pretrained weights.
    model = garimpo_uncertainty.models.build(
        args.arch,
        metadata["in_channels"],
        args.num_classes,
        args.encoder_weights if resume is None else None,
        weights_file,
    ).to(device)
    state = garimpo_uncertainty.training.loop.TrainingState(
        model=model,
        optimizer=garimpo_uncertainty.training.optimizers.OPTIMIZER_REGISTRY.build(
            args.optimizer, params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        ),
        loss_fn=garimpo_uncertainty.training.losses.LOSS_REGISTRY.build(args.loss),
        postprocess=garimpo_uncertainty.postprocess.POSTPROCESS_REGISTRY.build(args.postprocess),
        metric_names=args.metrics,
        metadata=metadata,
        arguments=result_arguments(args),
        history=[],
        best_dice=garimpo_uncertainty.training.loop.NO_BEST_DICE,
        first_epoch=0,
    )
    if resume is None:
        return state
    model.load_state_dict(resume["model"])
    state.optimizer.load_state_dict(resume["optimizer"])
    torch.set_rng_state(resume["rng"])
    if resume["cuda_rng"] is not None:
        torch.cuda.set_rng_state(resume["cuda_rng"])
    state.history, state.best_dice = resume["history"], resume["best_dice"]
    state.stopped, state.callbacks = resume["stopped"], resume["callbacks"]
    state.first_epoch = resume["epoch"] + 1
    return state


def start_line(
    args: argparse.Namespace,
    state: garimpo_uncertainty.training.loop.TrainingState,
    device: str,
    loaders: garimpo_uncertainty.training.loop.Loaders,
) -> str:
    """Return the line before the first epoch: where it trains, on what, and the model's size."""
    parameters = sum(p.numel() for p in state.model.parameters())
    return (
        f"training on {device} · {len(loaders.train.dataset)} train cells · "
        f"{len(loaders.validation.dataset)} validation cells · {parameters / 1e6:.1f}M params · "
        f"epochs {state.first_epoch + 1} to {args.epochs}"
    )


def build_loaders(
    args: argparse.Namespace,
    root: Path,
    normalization: dict[str, list[float]],
    train_cells: pd.DataFrame,
    validation_cells: pd.DataFrame,
) -> garimpo_uncertainty.training.loop.Loaders:
    """Return the augmented training loader and the two eval-mode ones."""
    batch_size, workers = args.batch_size, args.num_workers
    # ponytail: as many training cells as validation ones, so the pass costs what validation does;
    # a TRAIN_EVAL_CELLS config value is the upgrade if a run needs another size. pandas draws them,
    # so torch's random stream is left alone.
    train_eval_cells = train_cells.sample(
        n=min(len(validation_cells), len(train_cells)), random_state=args.subset_seed
    ).sort_index()
    return garimpo_uncertainty.training.loop.Loaders(
        train=garimpo_uncertainty.cell_dataset.loader(
            root,
            train_cells,
            normalization,
            batch_size,
            workers,
            samples=args.steps_per_epoch * batch_size,
            augmentations=tuple(
                garimpo_uncertainty.training.augmentations.AUGMENTATION_REGISTRY.build(spec)
                for spec in args.augmentations
            ),
        ),
        validation=garimpo_uncertainty.cell_dataset.loader(
            root, validation_cells, normalization, batch_size, workers
        ),
        train_eval=garimpo_uncertainty.cell_dataset.loader(
            root, train_eval_cells, normalization, batch_size, workers
        ),
    )


def fit(
    args: argparse.Namespace,
    state: garimpo_uncertainty.training.loop.TrainingState,
    loaders: garimpo_uncertainty.training.loop.Loaders,
    device: str,
) -> list[int] | None:
    """Run the epochs left, with the configured callbacks before the checkpoint and the log."""
    callbacks = [
        *(
            garimpo_uncertainty.training.callbacks.CALLBACK_REGISTRY.build(spec)
            for spec in args.callbacks
        ),
        garimpo_uncertainty.training.callbacks.CheckpointCallback(args.run_dir),
        garimpo_uncertainty.training.callbacks.ConsoleLogCallback(args.epochs, state.best_dice),
    ]
    trainer = garimpo_uncertainty.training.loop.Trainer(state, device, args.epochs, callbacks)
    batch_shape = trainer.fit(loaders)
    if state.stopped:
        last = state.history[-1]["epoch"] + 1 if state.history else 0
        garimpo_uncertainty.pipeline.runner.result(f"stopped early after epoch {last}")
    return batch_shape


def report_best(
    state: garimpo_uncertainty.training.loop.TrainingState, validation_cells: int
) -> None:
    """Show the best epoch; raise when no epoch had a Dice, so no best_model.pt exists."""
    if state.best_dice == garimpo_uncertainty.training.loop.NO_BEST_DICE:
        raise ValueError(
            "every val_dice is NaN, so no best_model.pt was written: "
            f"the {validation_cells} validation cells have no reference mining"
        )
    # A resumed best Dice may come from no epoch of the history, as when last.pt was edited.
    best = next((r["epoch"] for r in state.history if r["val_dice"] == state.best_dice), None)
    if best is not None:
        garimpo_uncertainty.pipeline.runner.result(
            f"best val_dice {state.best_dice:.4f} at epoch {best + 1} (* marks a new best)"
        )


def train(
    args: argparse.Namespace, root: Path, device: str, split: pd.DataFrame, tile_ids: list[int]
) -> dict:
    cells = garimpo_uncertainty.data.cell_table(root, split, tile_ids)
    train_cells = cells[cells.role == "train"]
    validation_cells = cells[cells.role == "validation"]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    weights_file = garimpo_uncertainty.data.starting_weights_path(root, args.encoder_weights)

    # Seeded before band_statistics, whose loader draws from this generator.
    torch.manual_seed(args.seed)
    resume = read_resume(args)
    if resume is None and weights_file is not None and not weights_file.exists():
        raise FileNotFoundError(
            f"{weights_file} is absent: on a node with internet run "
            f"garimpo download_data weights --encoder-weights {args.encoder_weights}"
        )

    if resume is None:
        metadata = new_metadata(args, root, cells, train_cells, tile_ids)
    else:
        metadata = resume["meta"]
        garimpo_uncertainty.data.check_split(metadata, cells)
    garimpo_uncertainty.files.write_json(
        garimpo_uncertainty.artifacts.RunPaths.of(args.run_dir).run_meta, metadata
    )

    state = start_training(args, device, metadata, weights_file, resume)
    loaders = build_loaders(args, root, metadata["normalization"], train_cells, validation_cells)
    garimpo_uncertainty.pipeline.runner.show(start_line(args, state, device, loaders))
    batch_shape = fit(args, state, loaders, device)
    report_best(state, len(validation_cells))
    return {
        "train_cells": len(train_cells),
        "validation_cells": len(validation_cells),
        "batch": batch_shape,
        "best_val_dice": state.best_dice,
    }


def add_part_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the flags that pick a registered part: a name, or a JSON object with a name."""
    optimizers = garimpo_uncertainty.training.optimizers.OPTIMIZER_REGISTRY
    losses = garimpo_uncertainty.training.losses.LOSS_REGISTRY
    augmentations = garimpo_uncertainty.training.augmentations.AUGMENTATION_REGISTRY
    callbacks = garimpo_uncertainty.training.callbacks.CALLBACK_REGISTRY
    parse = garimpo_uncertainty.registry.parse_spec
    parser.add_argument("--num-classes", type=int, required=True, help="1: one logit per pixel")
    parser.add_argument(
        "--optimizer", type=parse, required=True, help=", ".join(optimizers.names())
    )
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--loss", type=parse, required=True, help=", ".join(losses.names()))
    parser.add_argument(
        "--postprocess",
        choices=garimpo_uncertainty.postprocess.POSTPROCESS_REGISTRY.names(),
        required=True,
    )
    parser.add_argument(
        "--augmentations",
        nargs="*",
        type=parse,
        required=True,
        help=f"in order, after normalization: {', '.join(augmentations.names())}",
    )
    parser.add_argument(
        "--callbacks",
        nargs="*",
        type=parse,
        required=True,
        help=f"run before the checkpoint: {', '.join(callbacks.names())}",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=garimpo_uncertainty.training.metrics.METRIC_REGISTRY.names(),
        required=True,
    )


def check_choices(args: argparse.Namespace) -> None:
    """Raise when a chosen part is unknown, takes other options, or misfits the classes."""
    garimpo_uncertainty.training.optimizers.OPTIMIZER_REGISTRY.entry(args.optimizer)
    for spec in args.augmentations:
        garimpo_uncertainty.training.augmentations.AUGMENTATION_REGISTRY.build(spec)
    check_classes(args)
    check_callbacks(args)


def check_classes(args: argparse.Namespace) -> None:
    parts = {
        "--loss": garimpo_uncertainty.training.losses.LOSS_REGISTRY.build(args.loss),
        "--postprocess": garimpo_uncertainty.postprocess.POSTPROCESS_REGISTRY.build(
            args.postprocess
        ),
    }
    for flag, part in parts.items():
        if part.multiclass != (args.num_classes > 1):
            wants = "two or more classes" if part.multiclass else "one logit per pixel"
            raise ValueError(f"{flag} wants {wants}, and --num-classes is {args.num_classes}")


def check_callbacks(args: argparse.Namespace) -> None:
    if "dice" not in args.metrics:
        raise ValueError("--metrics must include dice: val_dice picks the best checkpoint")
    columns = {"val_loss", *(f"val_{name}" for name in args.metrics)}
    for spec in args.callbacks:
        callback = garimpo_uncertainty.training.callbacks.CALLBACK_REGISTRY.build(spec)
        monitor = getattr(callback, "monitor", None)
        if monitor is not None and monitor not in columns:
            raise ValueError(f"--callbacks monitors {monitor}, not one of {sorted(columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--arch",
        choices=garimpo_uncertainty.models.MODEL_REGISTRY.names(),
        required=True,
        help="convnet is smp's DeepLabV3, with no skip connections; both use one ResNet-18 encoder",
    )
    parser.add_argument(
        "--encoder-weights",
        choices=sorted(
            variant
            for model in garimpo_uncertainty.models.MODEL_REGISTRY.entries.values()
            for variant in model.variants
        ),
        help="starting weights of the encoder; required for unet, invalid for convnet",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=garimpo_uncertainty.data.SPLIT_PATH)
    parser.add_argument(
        "--train-fraction",
        type=float,
        choices=garimpo_uncertainty.data.TRAIN_FRACTIONS,
        required=True,
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        required=True,
        help="picks the training tiles; apart from --seed, so subsets stay nested across seeds",
    )
    parser.add_argument(
        "--evaluation-tiles", choices=garimpo_uncertainty.data.EVALUATION_TILES, required=True
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        required=True,
        help="updates between validations, the same at every fraction (Bornschein et al., 2020)",
    )
    parser.add_argument("--batch-size", type=int, required=True)
    add_part_arguments(parser)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device")
    parser.add_argument(
        "--num-workers", type=int, default=garimpo_uncertainty.pipeline.runner.available_cpus()
    )
    parser.add_argument("--force", action="store_true", help="restart, ignoring last.pt")
    parser.add_argument(
        "--discard-unfinished",
        action="store_true",
        help="with --force, restart even over an unfinished last.pt; the pilot's job passes it",
    )
    args = parser.parse_args()
    garimpo_uncertainty.pipeline.runner.log_to_stdout()
    try:
        garimpo_uncertainty.models.check_encoder_weights(args.arch, args.encoder_weights)
        check_choices(args)
    # A spec's options reach a constructor, which refuses an unknown one with a TypeError.
    except (ValueError, TypeError, ImportError) as error:
        parser.error(str(error))

    root = garimpo_uncertainty.data.data_root()
    device = args.device or garimpo_uncertainty.checkpoints.default_device()
    # In args, so both the stage fingerprint and the resume check see the file's identity. Hash and
    # not mtime: downloading the same file again must not invalidate a finished training.
    args.encoder_weights_sha256 = starting_weights_sha256(args, root)
    split = pd.read_csv(args.split)
    tile_ids = garimpo_uncertainty.data.select_tiles(
        split, args.train_fraction, args.subset_seed, args.evaluation_tiles
    ).tile_id.tolist()
    identity = garimpo_uncertainty.data.tiles_identity(root, split, tile_ids)
    # In args, so a resume refuses tiles re-cut since last.pt, as it refuses other weights.
    args.tiles_sha256 = next(iter(identity.values()))
    keep_unfinished_training(
        args.run_dir, force=args.force, discard_unfinished=args.discard_unfinished
    )
    run = garimpo_uncertainty.artifacts.RunPaths.of(args.run_dir)
    garimpo_uncertainty.pipeline.runner.run_stage(
        run.status,
        "train",
        [run.best_model, run.history],
        garimpo_uncertainty.pipeline.runner.fingerprint(
            args,
            [],
            keep=["batch_size"],
            hashed=identity,
        ),
        lambda: train(args, root, device, split, tile_ids),
        force=args.force,
    )
