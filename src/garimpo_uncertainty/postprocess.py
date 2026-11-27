"""Logits to the mined mask, by name: pure functions (logits: Tensor) -> Tensor.

The 0.5 decision is written here once; stored probabilities in u8 compare with its scaled value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import garimpo_uncertainty.registry

# compute_metrics reads THRESHOLD and must not load torch, which only the annotations name.
if TYPE_CHECKING:
    import torch

POSTPROCESS_REGISTRY = garimpo_uncertainty.registry.Registry("post-processing")
THRESHOLD = 0.5  # a pixel is mined from this probability on


@POSTPROCESS_REGISTRY.register("sigmoid_threshold")
class SigmoidThreshold:
    """One logit per pixel: sigmoid(logit) >= 0.5 is logit >= 0, with no sigmoid to round."""

    multiclass = False

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return logits >= 0


@POSTPROCESS_REGISTRY.register("softmax_argmax")
class SoftmaxArgmax:
    """One logit per class: a pixel is mined when class 1 has the highest logit."""

    multiclass = True

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(1, keepdim=True) == 1
