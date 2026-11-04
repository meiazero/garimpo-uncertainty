"""Stage: trains one segmentation model. Writes artifacts only; every metric runs elsewhere.

No hyperparameter has a default: the proposal fixes none of them.
"""

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch
import torch

import garimpo_uncertainty.cell_dataset
import garimpo_uncertainty.checkpoints
import garimpo_uncertainty.data
import garimpo_uncertainty.models
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


@torch.no_grad()
def validation_dice(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str
) -> float:
    """Mean Dice over validation cells with mining; the proposal's first-training criterion."""
    model.eval()
    scores = []
    for images, masks, _ in loader:
        predictions = model(images.to(device)) >= 0
        references = masks.to(device) > 0
        intersection = (predictions & references).flatten(1).sum(1)
        totals = predictions.flatten(1).sum(1) + references.flatten(1).sum(1)
        has_mining = references.flatten(1).any(1)
        scores += (2 * intersection / totals.clamp(min=1))[has_mining].tolist()
    return float(np.mean(scores)) if scores else float("nan")


def train(
    args: argparse.Namespace, root: Path, device: str, split: pd.DataFrame, tile_ids: list[int]
) -> dict:
    cells = garimpo_uncertainty.data.cell_table(root, split, tile_ids)
    train_cells = cells[cells.role == "train"]
    validation_cells = cells[cells.role == "validation"]
    last_path = args.run_dir / "last.pt"
    best_path = args.run_dir / "best_model.pt"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    weights_file = garimpo_uncertainty.data.starting_weights_path(root, args.encoder_weights)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    arguments = result_arguments(args)
    resume = None
    if last_path.exists() and not args.force:
        resume = torch.load(last_path, map_location="cpu", weights_only=True)
        if resume["arguments"] != arguments:
            different = sorted(k for k in arguments if resume["arguments"].get(k) != arguments[k])
            raise ValueError(f"last.pt was written with other values of {different}; use --force")
        print(f"resuming after epoch {resume['epoch']}")

    fetch = (
        "on a node with internet run scripts/download_data.py weights "
        f"--encoder-weights {args.encoder_weights}"
    )
    if resume is None and weights_file is not None and not weights_file.exists():
        raise FileNotFoundError(f"{weights_file} is absent: {fetch}")

    if resume is not None:
        metadata = resume["meta"]
        garimpo_uncertainty.data.check_split(metadata, cells)
    else:
        metadata = {
            "arch": args.arch,
            "variant": garimpo_uncertainty.models.variant_name(args.arch, args.encoder_weights),
            "in_channels": len(garimpo_uncertainty.data.BANDS),
            "channels": args.channels,
            "encoder": garimpo_uncertainty.models.UNET_ENCODER if args.arch == "unet" else None,
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
    garimpo_uncertainty.stages.write_json(args.run_dir / "run_meta.json", metadata)

    # A resume fills the model from last.pt, so only a fresh start reads pretrained weights.
    starting_weights = args.encoder_weights if resume is None else "none"
    model = garimpo_uncertainty.models.build(
        args.arch,
        metadata["in_channels"],
        args.channels,
        starting_weights,
        weights_file,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = build_loss(args.loss)
    history, best_dice, first_epoch = [], NO_BEST_DICE, 0
    if resume is not None:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        torch.set_rng_state(resume["rng"])
        history, best_dice, first_epoch = (
            resume["history"],
            resume["best_dice"],
            resume["epoch"] + 1,
        )

    normalization = metadata["normalization"]
    train_loader = garimpo_uncertainty.cell_dataset.loader(
        root, train_cells, normalization, args.batch_size, args.num_workers, True, True
    )
    validation_loader = garimpo_uncertainty.cell_dataset.loader(
        root, validation_cells, normalization, args.batch_size, args.num_workers
    )
    last_batch_shape = None
    for epoch in range(first_epoch, args.epochs):
        model.train()
        losses = []
        data_wait = 0.0
        epoch_start = ready = time.monotonic()
        for images, masks, _ in train_loader:
            data_wait += time.monotonic() - ready
            optimizer.zero_grad()
            loss = loss_fn(model(images.to(device)), masks.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            last_batch_shape = list(images.shape)
            ready = time.monotonic()
        dice = validation_dice(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_dice": dice,
                "train_data_wait_seconds": round(data_wait, 3),
                "seconds": round(time.monotonic() - epoch_start, 3),
            }
        )
        print(history[-1], flush=True)
        # Every epoch, so a job killed at its limit still reports what it measured.
        with garimpo_uncertainty.stages.atomic(args.run_dir / "history.csv") as temporary:
            pd.DataFrame(history).to_csv(temporary, index=False)
        score = NO_BEST_DICE if np.isnan(dice) else dice
        state = {k: v.cpu() for k, v in model.state_dict().items()}
        if score > best_dice:
            best_dice = score
            garimpo_uncertainty.checkpoints.save(
                best_path, {"model": state, "meta": metadata, "epoch": epoch, "val_dice": dice}
            )
        garimpo_uncertainty.checkpoints.save(
            last_path,
            {
                "model": state,
                "optimizer": optimizer.state_dict(),
                "meta": metadata,
                "arguments": arguments,
                "epoch": epoch,
                "best_dice": best_dice,
                "history": history,
                "rng": torch.get_rng_state(),
            },
        )
    if best_dice == NO_BEST_DICE:
        raise ValueError(
            "every val_dice is NaN, so no best_model.pt was written: "
            f"the {len(validation_cells)} validation cells have no reference mining"
        )
    return {
        "train_cells": len(train_cells),
        "validation_cells": len(validation_cells),
        "batch": last_batch_shape,
        "best_val_dice": best_dice,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--arch", choices=["convnet", "unet"], required=True)
    parser.add_argument("--channels", type=int, nargs="+", help="widths; required for convnet")
    parser.add_argument(
        "--encoder-weights",
        choices=list(garimpo_uncertainty.models.UNET_VARIANTS),
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
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--loss", choices=["dice", "bce", "dice_bce"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="restart, ignoring last.pt")
    args = parser.parse_args()
    try:
        garimpo_uncertainty.models.check_encoder_weights(args.arch, args.encoder_weights)
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
    garimpo_uncertainty.stages.run_stage(
        args.run_dir / "status",
        "train",
        [args.run_dir / "best_model.pt", args.run_dir / "history.csv"],
        args.force,
        garimpo_uncertainty.stages.fingerprint(
            args,
            [],
            keep=["batch_size"],
            hashed=garimpo_uncertainty.data.tiles_identity(root, split, tile_ids),
        ),
        lambda: train(args, root, device, split, tile_ids),
    )


if __name__ == "__main__":
    main()
