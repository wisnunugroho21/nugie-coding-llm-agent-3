"""Orbax checkpointing of a full training state.

A checkpoint at `step` bundles four items so a run resumes bit-for-bit:

    model      nnx.state(model)      — weights + MoE router biases (StandardSave)
    optimizer  nnx.state(optimizer)  — AdamW moments + step        (StandardSave)
    data       grain iterator state  — position in the shuffled stream (JsonSave)
    meta       {step, ...}           — bookkeeping                 (JsonSave)

Restore is target-based: the caller has already built `model` and `optimizer`
(with the right shapes), and we restore INTO their current states, so no abstract
tree needs to be reconstructed by hand.
"""

from __future__ import annotations

from pathlib import Path

import flax.nnx as nnx
import orbax.checkpoint as ocp


class CheckpointManager:
    def __init__(self, directory: str, max_to_keep: int = 3):
        self.dir = Path(directory).absolute() / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep, create=True, enable_async_checkpointing=False
        )
        self._mgr = ocp.CheckpointManager(self.dir, options=options)

    # ------------------------------------------------------------------ #
    def save(self, step: int, model, optimizer, data_state: dict, meta: dict) -> None:
        self._mgr.save(
            step,
            args=ocp.args.Composite(
                model=ocp.args.StandardSave(nnx.state(model)),
                optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
                data=ocp.args.JsonSave(data_state),
                meta=ocp.args.JsonSave(meta),
            ),
        )
        self._mgr.wait_until_finished()

    # ------------------------------------------------------------------ #
    def latest_step(self) -> int | None:
        return self._mgr.latest_step()

    def restore(self, model, optimizer, step: int | None = None) -> tuple[dict, dict]:
        """Restore model+optimizer IN PLACE. Returns (data_state, meta)."""
        step = self._mgr.latest_step() if step is None else step
        if step is None:
            raise FileNotFoundError(f"no checkpoint found under {self.dir}")

        model_target = nnx.state(model)
        opt_target = nnx.state(optimizer)
        restored = self._mgr.restore(
            step,
            args=ocp.args.Composite(
                model=ocp.args.StandardRestore(model_target),
                optimizer=ocp.args.StandardRestore(opt_target),
                data=ocp.args.JsonRestore(),
                meta=ocp.args.JsonRestore(),
            ),
        )
        nnx.update(model, restored["model"])
        nnx.update(optimizer, restored["optimizer"])
        return restored["data"], restored["meta"]

    def restore_model(self, model, step: int | None = None) -> int:
        """Restore ONLY the model weights (not optimizer/data), in place.

        Used to warm-start a new run from another run's checkpoint — e.g. loading
        pretrained weights into an SFT run (`--init-from`). The optimizer, data
        iterator, and step counter are left fresh. Returns the restored step.

        The two models must share the same architecture and vocab size, or Orbax
        raises a structure/shape mismatch (surfaced with a clear message by the
        caller).
        """
        step = self._mgr.latest_step() if step is None else step
        if step is None:
            raise FileNotFoundError(f"no checkpoint found under {self.dir}")
        restored = self._mgr.restore(
            step,
            args=ocp.args.Composite(
                model=ocp.args.StandardRestore(nnx.state(model)),
            ),
        )
        nnx.update(model, restored["model"])
        return step

    def close(self) -> None:
        self._mgr.close()
