"""Stage: trains one segmentation model. Writes artifacts only; every metric runs elsewhere.

No hyperparameter has a default, so a run states every value it uses.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch
import torch

import garimpo_uncertainty.cell_dataset
import garimpo_uncertainty.checkpoints
import garimpo_uncertainty.data
import garimpo_uncertainty.models
import garimpo_uncertainty.models.common
import garimpo_uncertainty.models.unet
import garimpo_uncertainty.stages

NO_BEST_DICE = -1.0


def result_arguments(args: argparse.Namespace) -> dict[str, str]:
    """Arguments that shape the trained weights; batch size does, unlike in the later stages."""
    ignored = garimpo_uncertainty.stages.IGNORED_ARGUMENTS - {"batch_size"}
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
        return garimpo_uncertainty.checkpoints.file_sha256(weights_file)
    if not (args.run_dir / "run_meta.json").exists():
        garimpo_uncertainty.stages.fail_stage(
            args.run_dir / "status",
            "train",
            f"train: cannot tell which weights this run used: {weights_file} and "
            f"{args.run_dir / 'run_meta.json'} are both absent. On a node with internet run "
            f"scripts/download_data.py weights --encoder-weights {args.encoder_weights}, or "
            "restore run_meta.json with the run.",
        )
    return garimpo_uncertainty.data.read_run_metadata(args.run_dir).get("encoder_weights_sha256")


def build_loss(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    dice = segmentation_models_pytorch.losses.DiceLoss(mode="binary", from_logits=True)
    bce = torch.nn.BCEWithLogitsLoss()
    if name == "dice":
        return dice
    if name == "bce":
        return bce
    return lambda logits, mask: dice(logits, mask) + bce(logits, mask)


def batches(loader: torch.utils.data.DataLoader) -> Iterator:
    """Start iterating the loader without losing a Ctrl+C that lands while it forks its workers.

    Python raises such a Ctrl+C inside an at-fork callback, where it only prints it as unraisable
    and training carries on; it is kept here and raised again once the workers exist.
    """
    swallowed: list[KeyboardInterrupt] = []
    previous = sys.unraisablehook

    def keep(unraisable: object) -> None:  # sys.UnraisableHookArgs exists only in typeshed
        if isinstance(unraisable.exc_value, KeyboardInterrupt):
            swallowed.append(unraisable.exc_value)
        else:
            previous(unraisable)

    sys.unraisablehook = keep
    try:
        iterator = iter(loader)
    finally:
        sys.unraisablehook = previous
    if swallowed:
        raise swallowed[0]
    return iterator


@torch.no_grad()
def validation_dice(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str
) -> float:
    """Mean Dice over validation cells with mining; the criterion that picks the best checkpoint."""
    model.eval()
    scores = []
    for images, masks, _ in batches(loader):
        predictions = model(images.to(device)) >= 0
        references = masks.to(device) > 0
        intersection = (predictions & references).flatten(1).sum(1)
        totals = predictions.flatten(1).sum(1) + references.flatten(1).sum(1)
        has_mining = references.flatten(1).any(1)
        scores += (2 * intersection / totals.clamp(min=1))[has_mining].tolist()
    return float(np.mean(scores)) if scores else float("nan")


@dataclass
class TrainingState:
    """What the epochs share, and what last.pt restores after an interruption."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    metadata: dict
    arguments: dict[str, str]
    history: list[dict]
    best_dice: float
    first_epoch: int


# What a forced redo leaves in place of a finished run's last.pt.
DISCARDED = {"discarded": True}


def read_resume(args: argparse.Namespace) -> dict | None:
    """Return last.pt when this run continues it; refuse one written with other arguments."""
    last_path = args.run_dir / "last.pt"
    if not last_path.exists() or args.force:
        return None
    resume = torch.load(last_path, map_location="cpu", weights_only=True)
    if resume == DISCARDED:
        return None
    arguments = result_arguments(args)
    if resume["arguments"] != arguments:
        different = sorted(k for k in arguments if resume["arguments"].get(k) != arguments[k])
        raise ValueError(
            f"last.pt was written with other values of {different}: rerun with those values to "
            "resume it, or delete last.pt to start over with these"
        )
    print(f"resuming after epoch {resume['epoch']}")
    return resume


def keep_unfinished_training(run_dir: Path, *, force: bool) -> None:
    """Refuse --force over an unfinished run's last.pt, which training from scratch overwrites.

    FORCE_FROM=train left in a Colab cell would otherwise restart after every lost session. A
    finished run's last.pt is marked discarded instead, so a forced redo cut short never resumes
    from its end: a deleted file would come back with the next restore from Drive, whose rsync
    has no --delete, where the mark replaces it.
    """
    last = run_dir / "last.pt"
    if not force or not last.exists():
        return
    if torch.load(last, map_location="cpu", weights_only=True) == DISCARDED:
        return
    status_path = run_dir / "status" / "train.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    # Starting the work drops completed_fingerprint, so only a finished training still has it.
    if "completed_fingerprint" not in status:
        garimpo_uncertainty.stages.fail_stage(
            run_dir / "status",
            "train",
            f"{last} holds an unfinished training (status {status.get('status', 'absent')}): "
            f"run without --force (FORCE_FROM=train) to resume it, or delete {last} to start over",
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
        "variant": garimpo_uncertainty.models.common.variant_name(args.arch, args.encoder_weights),
        "in_channels": len(garimpo_uncertainty.data.BANDS),
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
        **garimpo_uncertainty.checkpoints.code_versions(),
    }


def start_training(
    args: argparse.Namespace,
    device: str,
    metadata: dict,
    weights_file: Path | None,
    resume: dict | None,
) -> TrainingState:
    """Build the model and its optimizer, then restore both and the RNG from last.pt if resuming."""
    # A resume fills the model from last.pt, so only a fresh start reads pretrained weights.
    starting_weights = args.encoder_weights if resume is None else "none"
    model = garimpo_uncertainty.models.common.build(
        args.arch,
        metadata["in_channels"],
        starting_weights,
        weights_file,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = build_loss(args.loss)
    arguments = result_arguments(args)
    if resume is None:
        return TrainingState(model, optimizer, loss_fn, metadata, arguments, [], NO_BEST_DICE, 0)
    model.load_state_dict(resume["model"])
    optimizer.load_state_dict(resume["optimizer"])
    torch.set_rng_state(resume["rng"])
    if resume["cuda_rng"] is not None:
        torch.cuda.set_rng_state(resume["cuda_rng"])
    return TrainingState(
        model,
        optimizer,
        loss_fn,
        metadata,
        arguments,
        resume["history"],
        resume["best_dice"],
        resume["epoch"] + 1,
    )


def fit_epoch(
    state: TrainingState,
    loader: torch.utils.data.DataLoader,
    device: str,
    started: float,
) -> tuple[dict, list[int] | None]:
    """Return the epoch's loss, updates and data wait, and the last batch's shape."""
    losses = []
    data_wait = 0.0
    batch_shape = None
    # The wait starts with the epoch's clock, so the two timings share one origin.
    ready = started
    for images, masks, _ in batches(loader):
        data_wait += time.monotonic() - ready
        state.optimizer.zero_grad()
        loss = state.loss_fn(state.model(images.to(device)), masks.to(device))
        loss.backward()
        state.optimizer.step()
        losses.append(loss.item())
        batch_shape = list(images.shape)
        ready = time.monotonic()
    fitted = {
        "train_loss": float(np.mean(losses)),
        "updates": len(losses),
        "train_data_wait_seconds": round(data_wait, 3),
    }
    return fitted, batch_shape


def save_epoch(args: argparse.Namespace, state: TrainingState, epoch: int, dice: float) -> None:
    """Write the epoch's history and checkpoints; best_model.pt only when its Dice improved."""
    # Every epoch, so a job killed at its limit still reports what it measured.
    with garimpo_uncertainty.stages.atomic(args.run_dir / "history.csv") as temporary:
        pd.DataFrame(state.history).to_csv(temporary, index=False)
    score = NO_BEST_DICE if np.isnan(dice) else dice
    weights = {k: v.cpu() for k, v in state.model.state_dict().items()}
    if score > state.best_dice:
        state.best_dice = score
        garimpo_uncertainty.checkpoints.save(
            args.run_dir / "best_model.pt",
            {"model": weights, "meta": state.metadata, "epoch": epoch, "val_dice": dice},
        )
    garimpo_uncertainty.checkpoints.save(
        args.run_dir / "last.pt",
        {
            "model": weights,
            "optimizer": state.optimizer.state_dict(),
            "meta": state.metadata,
            "arguments": state.arguments,
            "epoch": epoch,
            "best_dice": state.best_dice,
            "history": state.history,
            "rng": torch.get_rng_state(),
            # The DeepLabV3 head's dropout draws on the GPU, so a resume must restore that too;
            # only the device in use, so a run from a 2-GPU host resumes on a 1-GPU runtime.
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
        },
    )


def run_epoch(
    state: TrainingState,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    device: str,
    epoch: int,
) -> tuple[dict, list[int] | None]:
    """Train and validate once; return the epoch's raw record and the last batch's shape."""
    garimpo_uncertainty.stages.take_gpu_peak_bytes()
    state.model.train()
    epoch_start = time.monotonic()
    fitted, batch_shape = fit_epoch(state, train_loader, device, epoch_start)
    train_end = time.monotonic()
    train_peak = garimpo_uncertainty.stages.take_gpu_peak_bytes()
    validation_start = time.monotonic()
    dice = validation_dice(state.model, validation_loader, device)
    validation_end = time.monotonic()
    record = {
        "epoch": epoch,
        **fitted,
        "val_dice": dice,
        "seconds": round(validation_end - epoch_start, 3),
        "train_seconds": train_end - epoch_start,
        "validation_seconds": validation_end - validation_start,
        "train_peak_gpu_bytes": train_peak,
        "validation_peak_gpu_bytes": garimpo_uncertainty.stages.take_gpu_peak_bytes(),
    }
    return record, batch_shape


def run_epochs(
    args: argparse.Namespace,
    state: TrainingState,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    device: str,
) -> list[int] | None:
    """Train up to args.epochs; return the shape of the last batch, None if no epoch ran."""
    batch_shape = None
    for epoch in range(state.first_epoch, args.epochs):
        record, batch_shape = run_epoch(state, train_loader, validation_loader, device, epoch)
        state.history.append(record)
        print(record, flush=True)
        save_epoch(args, state, epoch, record["val_dice"])
    return batch_shape


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
            f"scripts/download_data.py weights --encoder-weights {args.encoder_weights}"
        )

    if resume is None:
        metadata = new_metadata(args, root, cells, train_cells, tile_ids)
    else:
        metadata = resume["meta"]
        garimpo_uncertainty.data.check_split(metadata, cells)
    garimpo_uncertainty.stages.write_json(args.run_dir / "run_meta.json", metadata)

    state = start_training(args, device, metadata, weights_file, resume)
    train_loader = garimpo_uncertainty.cell_dataset.loader(
        root,
        train_cells,
        metadata["normalization"],
        args.batch_size,
        args.num_workers,
        samples=args.steps_per_epoch * args.batch_size,
        augment=True,
    )
    validation_loader = garimpo_uncertainty.cell_dataset.loader(
        root, validation_cells, metadata["normalization"], args.batch_size, args.num_workers
    )
    batch_shape = run_epochs(args, state, train_loader, validation_loader, device)
    if state.best_dice == NO_BEST_DICE:
        raise ValueError(
            "every val_dice is NaN, so no best_model.pt was written: "
            f"the {len(validation_cells)} validation cells have no reference mining"
        )
    return {
        "train_cells": len(train_cells),
        "validation_cells": len(validation_cells),
        "batch": batch_shape,
        "best_val_dice": state.best_dice,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--arch",
        choices=["convnet", "unet"],
        required=True,
        help="convnet is smp's DeepLabV3, with no skip connections; both use one ResNet-18 encoder",
    )
    parser.add_argument(
        "--encoder-weights",
        choices=list(garimpo_uncertainty.models.unet.UNET_VARIANTS),
        help="starting weights of the encoder; required for unet, invalid for convnet",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, default="outputs/split.csv")
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
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        required=True,
        help="updates between validations, the same at every fraction (Bornschein et al., 2020)",
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--loss", choices=["dice", "bce", "dice_bce"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="restart, ignoring last.pt")
    args = parser.parse_args()
    garimpo_uncertainty.stages.log_to_stdout()
    try:
        garimpo_uncertainty.models.common.check_encoder_weights(args.arch, args.encoder_weights)
    except ValueError as error:
        parser.error(str(error))

    root = garimpo_uncertainty.data.data_root()
    device = args.device or garimpo_uncertainty.checkpoints.default_device()
    # In args, so both the stage fingerprint and the resume check see the file's identity. Hash and
    # not mtime: downloading the same file again must not invalidate a finished training.
    args.encoder_weights_sha256 = starting_weights_sha256(args, root)
    split = pd.read_csv(args.split)
    tile_ids = garimpo_uncertainty.data.select_tiles(
        split, args.train_fraction, args.subset_seed
    ).tile_id.tolist()
    keep_unfinished_training(args.run_dir, force=args.force)
    garimpo_uncertainty.stages.run_stage(
        args.run_dir / "status",
        "train",
        [args.run_dir / "best_model.pt", args.run_dir / "history.csv"],
        garimpo_uncertainty.stages.fingerprint(
            args,
            [],
            keep=["batch_size"],
            hashed=garimpo_uncertainty.data.tiles_identity(root, split, tile_ids),
        ),
        lambda: train(args, root, device, split, tile_ids),
        force=args.force,
    )


if __name__ == "__main__":
    main()
