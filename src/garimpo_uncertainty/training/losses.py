"""Losses by name, each an nn.Module on raw logits and the cell's float mask (B, 1, H, W)."""

import segmentation_models_pytorch
import torch

import garimpo_uncertainty.registry

LOSS_REGISTRY = garimpo_uncertainty.registry.Registry("loss")


@LOSS_REGISTRY.register("dice")
class DiceLoss(segmentation_models_pytorch.losses.DiceLoss):
    multiclass = False

    def __init__(self) -> None:
        super().__init__(mode="binary", from_logits=True)


@LOSS_REGISTRY.register("bce")
class BCELoss(torch.nn.BCEWithLogitsLoss):
    multiclass = False


@LOSS_REGISTRY.register("dice_bce")
class DiceBCELoss(torch.nn.Module):
    multiclass = False

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0) -> None:
        super().__init__()
        self.dice, self.bce = DiceLoss(), BCELoss()
        self.dice_weight, self.bce_weight = dice_weight, bce_weight

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, mask) + self.bce_weight * self.bce(logits, mask)


@LOSS_REGISTRY.register("cross_entropy")
class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """One logit per class; the mask's 1 is class 1, the mined one."""

    multiclass = True

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return super().forward(logits, mask[:, 0].long())
