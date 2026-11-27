"""Optimizers by name; a config's extra options, such as SGD's momentum, reach the constructor."""

import torch

import garimpo_uncertainty.registry

OPTIMIZER_REGISTRY = garimpo_uncertainty.registry.Registry("optimizer")
OPTIMIZER_REGISTRY.register("adamw")(torch.optim.AdamW)
OPTIMIZER_REGISTRY.register("adam")(torch.optim.Adam)
OPTIMIZER_REGISTRY.register("sgd")(torch.optim.SGD)
