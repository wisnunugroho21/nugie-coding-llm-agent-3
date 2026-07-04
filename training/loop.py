"""Loss, Optax optimizer, and jitted train / eval steps.

The training objective is the standard next-token cross-entropy PLUS the MoE
load-balancing aux loss that every layer's router contributes (returned by the model
as `aux["aux_loss"]`). On top of the gradient step we also nudge each MoE layer's
router *selection bias* toward uniform load — DeepSeek-V3's aux-loss-free balancing
(`update_router_bias`), applied outside the gradient inside the same jitted step.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from multi_latent_attention.moe import update_router_bias

from .config import OptimConfig
from .data import IGNORE


# --------------------------------------------------------------------------- #
#  Optimizer
# --------------------------------------------------------------------------- #
def build_optimizer(
    cfg: OptimConfig, total_steps: int
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """AdamW with a linear warmup + cosine decay schedule and global-norm clipping.

    Weight decay is applied to matrices only (ndim > 1); RMSNorm scales, biases, and
    other 1-D vectors are excluded — the usual convention, since decaying norm/bias
    terms hurts more than it helps.
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.peak_lr,
        warmup_steps=cfg.warmup_steps,
        decay_steps=max(total_steps, cfg.warmup_steps + 1),
        end_value=cfg.peak_lr * cfg.min_lr_ratio,
    )

    def decay_mask(params):
        return jax.tree.map(lambda p: p.ndim > 1, params)

    chain = []
    if cfg.grad_clip and cfg.grad_clip > 0:
        chain.append(optax.clip_by_global_norm(cfg.grad_clip))
    chain.append(
        optax.adamw(
            learning_rate=schedule,
            b1=cfg.b1,
            b2=cfg.b2,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            mask=decay_mask,
        )
    )
    return optax.chain(*chain), schedule


# --------------------------------------------------------------------------- #
#  Loss
# --------------------------------------------------------------------------- #
def masked_ce(logits: jax.Array, labels: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Sum of cross-entropy over non-ignored positions, and the token count.

    logits: fp32 [B, L, V]; labels: int [B, L] with IGNORE (-100) meaning "skip".
    Returns (summed_loss, n_tokens) so callers can aggregate across batches exactly.
    """
    mask = labels != IGNORE
    safe = jnp.where(mask, labels, 0)  # keep indices in range for the gather
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, safe)
    ce = ce * mask
    return ce.sum(), mask.sum()


def _loss_fn(model, input_ids, labels):
    logits, aux = model(input_ids)
    ce_sum, ntok = masked_ce(logits, labels)
    ce = ce_sum / jnp.maximum(ntok, 1.0)
    loss = ce + aux["aux_loss"]
    return loss, (ce, aux)


# --------------------------------------------------------------------------- #
#  Steps
# --------------------------------------------------------------------------- #
def make_train_step(router_bias_lr: float):
    """Build the jitted training step. `router_bias_lr` is captured as a constant."""

    @nnx.jit
    def train_step(model, optimizer, input_ids, labels):
        (loss, (ce, aux)), grads = nnx.value_and_grad(_loss_fn, has_aux=True)(
            model, input_ids, labels
        )
        optimizer.update(model, grads)

        # Aux-loss-free load balancing: bump each MoE layer's selection bias toward
        # uniform load using this step's realized per-expert token counts. Outside the
        # gradient (update_router_bias is a plain array update); jit tracks the mutation.
        group_sizes = aux["group_sizes"]  # [n_layers, E]
        for i, layer in enumerate(model.layers):
            moe = layer.channel_mixer
            moe.router_bias.value = update_router_bias(
                moe.router_bias.value, group_sizes[i], router_bias_lr
            )
        return loss, ce

    return train_step


@nnx.jit
def eval_step(model, input_ids, labels):
    """Forward-only masked-CE for perplexity. Returns (summed_loss, n_tokens)."""
    logits, _ = model(input_ids)
    return masked_ce(logits, labels)
