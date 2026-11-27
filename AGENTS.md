# Agent rules for garimpo-uncertainty

These rules bind every agent that writes, refactors or reviews code here, in this repository and in
the `compass/` submodule.

## Architecture

1. **Registry + Factory.** Models, losses, optimizers, metrics, transforms, post-processing,
   callbacks, COMPASS directions, searches and quantile rules are chosen by name from a registry
   (`@register_model("unet")`, `LOSS_REGISTRY`, ...) and built by a factory from a config dict.
   Never pick one with `if/elif` on a name, and never repeat a registry's names in `argparse`
   `choices`: read them from the registry.
2. **Strategy.** Each loss is an `nn.Module`; a combined loss such as Dice + BCE is its own
   module. Post-processing steps are pure functions `(logits: Tensor) -> Tensor`.
3. **Pipeline / Builder.** `CellDataset` only reads `(image, mask)` from disk; normalization and
   augmentation come from an injected `transform` built from config.
4. **Template Method + Callbacks.** The training loop is fixed (`train_step`, `val_step`,
   `on_epoch_end`). Checkpoints, logs and early stopping are callbacks; none of them is written
   inside the loop.
5. **Adapter.** Every model, whatever library it comes from, takes `(B, C, H, W)` and returns raw
   logits `(B, NUM_CLASSES, H, W)`, with no sigmoid or softmax inside. Its adapter also names the
   layers COMPASS may split at and splits the model there.

A variation the protocol does not use is still implemented, and ships off in `configs/base.json`.

## Constraints that override the architecture

- The default config trains bitwise the same weights at any worker count and across a resume
  (`tests/e2e/test_workers.py`, `test_resume.py`, `test_training.py`). Random draws stay in the
  main process; a step that draws inside the workers says so in `run_meta.json`.
- Files already written by runs keep loading: checkpoint state_dict keys, layer names such as
  `decoder.blocks.0`, and the fields of `directions.npz`.
- Notebooks hold only the Colab session: mount Drive, clone, install, copy to and from Drive,
  release the runtime. Everything else is a `garimpo` command they call.
- `compass/` changes go to its `dev` branch; its `main` is never pushed to.

## Review

Before a push, one agent reviews the diff, given the requirements and the diff, not the
implementer's conclusions. Reviewers: `gemini-4-hydrogen` and `gpt-6.2-sol`, when available;
otherwise a blind Sonnet at high effort. Reviewers report, the implementer fixes. Besides bugs,
they check two rules:

- **Comments:** a comment that restates the code is removed. A comment about something outside the
  code (an upstream bug, a paper's choice, a platform limit) stays.
- **Tests:** a test that would not fail on a real regression is removed; a test that fails without
  a bug is fixed.
