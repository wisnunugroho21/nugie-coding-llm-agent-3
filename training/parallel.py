"""Single-node multi-GPU data parallelism (e.g. Kaggle 2x NVIDIA T4).

PURE DATA PARALLELISM — the smallest change that uses every visible device:

    * the model + optimizer state are REPLICATED on every device;
    * each training batch is SPLIT along its leading (batch) axis across devices.

That is all that is needed. The train step is already `nnx.jit`'d, and JAX's GSPMD
partitioner traces it on the GLOBAL batch shape, so every reduction inside the step
(the token-summed cross-entropy, the MoE `group_sizes`, the router probabilities)
is computed over the whole global batch and the parameter gradients are all-reduced
across devices automatically. No change to the model or the loss is required, and
the math is identical to a single-device run of the same global batch.

This works only while the model FITS on a single device (the ~140M T4 config does).
Larger models need parameter sharding (FSDP/ZeRO-3) — see docs/multi_device_plan.md.

Degrades gracefully: with one visible device the mesh has size 1, replication and
the 1-way split are no-ops, and behavior matches the original single-device trainer.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def build_mesh() -> Mesh:
    """A 1-D device mesh over ALL visible devices, with axis name ``"data"``."""
    return Mesh(np.asarray(jax.devices()), ("data",))


def n_devices(mesh: Mesh) -> int:
    return int(np.asarray(mesh.devices).size)


def replicate(mesh: Mesh, *modules) -> None:
    """Replicate each module's nnx state onto every device, in place.

    Call AFTER the model/optimizer are built (and after any checkpoint restore),
    so the training-loop state is committed to a replicated sharding before the
    first step. Idempotent if a state is already replicated.
    """
    repl = NamedSharding(mesh, P())  # no partitioned axes -> full replica per device
    for m in modules:
        nnx.update(m, jax.device_put(nnx.state(m), repl))


def shard_batch(mesh: Mesh, batch) -> jax.Array:
    """Place a host batch ``[B, ...]`` on the devices, split along axis 0 (``"data"``).

    ``B`` must be divisible by the number of devices (validated by the caller).
    """
    return jax.device_put(np.asarray(batch), NamedSharding(mesh, P("data")))
