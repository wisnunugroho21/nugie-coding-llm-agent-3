"""Training entrypoint.

    python -m training.train --config configs/tiny_pretrain.yaml
    python -m training.train --config configs/tiny_sft.yaml --resume

Pipeline:
    1. tokenizer   — load an existing one, else train a ByteLevel-BPE on the corpus.
    2. data        — OpenCoder -> tokenized [2, seq_len] rows in .npy shards on
                     disk (written once, mmap-streamed) -> Grain iterators.
    3. model       — KimiLinear (GDN-2) sized to the tokenizer's vocab.
    4. optimizer   — Optax AdamW + warmup-cosine (see loop.build_optimizer).
    5. loop        — step-based training with periodic perplexity eval and Orbax
                     checkpoints (resumable, including the exact data position).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from kimi_linear_gdn2 import KimiLinear, count_params

from . import data as datamod
from .checkpoint import CheckpointManager
from .config import ExperimentConfig, load_config
from .loop import build_optimizer, eval_step, make_train_step
from .tokenizer import CodeTokenizer, sample_corpus
from .utils import device_summary, get_logger, human, set_xla_flags_for_cpu

log = get_logger()


# --------------------------------------------------------------------------- #
def ensure_tokenizer(cfg: ExperimentConfig) -> CodeTokenizer:
    """Load the tokenizer at cfg.train.tokenizer_path, or train one if absent.

    The requested vocab size is taken from `model.vocab_size` in the YAML; the trained
    tokenizer may end up slightly smaller on a tiny corpus. The ACTUAL size is written
    back onto the model config after loading, so the two never disagree.
    """
    import os

    path = cfg.train.tokenizer_path
    if os.path.exists(path):
        log.info("loading tokenizer from %s", path)
        tok = CodeTokenizer.load(path)
    else:
        log.info("training tokenizer (target vocab=%d) ...", cfg.model.vocab_size)
        corpus = sample_corpus(datamod.corpus_for_tokenizer(cfg.data), max_chars=20_000_000)
        tok = CodeTokenizer.train(corpus, cfg.model.vocab_size, path)
    cfg.model.vocab_size = tok.vocab_size
    log.info("tokenizer ready: vocab=%d", tok.vocab_size)
    return tok


def build_data(cfg: ExperimentConfig, tok: CodeTokenizer):
    """Tokenize OpenCoder into on-disk shards (once) and open them for streaming.

    Returns (train_source, val_rows): a mmap-backed random-access source over the
    train shards (fed to `make_train_iterator`) and the small in-memory val split.
    """
    shards_dir = cfg.data.shards_dir or str(Path(cfg.train.out_dir) / "shards")
    log.info("preparing %s shards at %s ...", cfg.data.task, shards_dir)
    train_source, val_rows = datamod.ensure_shards(cfg.data, tok, shards_dir)
    log.info(
        "dataset: %s train rows (streamed from disk), %s val rows (seq_len=%d)",
        human(len(train_source)), human(len(val_rows)), cfg.data.seq_len,
    )
    return train_source, val_rows


def evaluate(model, val_rows, batch_size, max_batches) -> float:
    """Token-weighted perplexity over the validation rows."""
    if len(val_rows) < batch_size:
        return float("nan")
    tot_loss, tot_tok = 0.0, 0
    for batch in datamod.iterate_eval_batches(val_rows, batch_size, max_batches):
        inp, lab = datamod.split_batch(batch)
        ls, nt = eval_step(model, jnp.asarray(inp), jnp.asarray(lab))
        tot_loss += float(ls)
        tot_tok += int(nt)
    if tot_tok == 0:
        return float("nan")
    return math.exp(tot_loss / tot_tok)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="restore the latest checkpoint under train.out_dir and continue")
    ap.add_argument("--init-from", default=None, metavar="RUN_DIR",
                    help="warm-start model WEIGHTS from another run's checkpoint dir "
                         "(e.g. a pretrained run) before an SFT run. Fresh optimizer, "
                         "data, and step counter. Ignored if --resume restores a "
                         "checkpoint from this run's own out_dir.")
    ap.add_argument("--init-from-step", type=int, default=None,
                    help="specific step to load with --init-from (default: latest).")
    args = ap.parse_args()

    set_xla_flags_for_cpu()
    cfg = load_config(args.config)
    log.info("devices: %s", device_summary())

    np.random.seed(cfg.train.seed)

    # 1. tokenizer -> 2. data
    tok = ensure_tokenizer(cfg)
    train_source, val_rows = build_data(cfg, tok)

    # 3. model
    cfg.validate()  # re-check now that vocab_size is finalized
    model = KimiLinear(cfg.model, rngs=nnx.Rngs(cfg.train.seed))
    log.info("model: %s params (%d layers, d_model=%d)",
             human(count_params(model)), cfg.model.n_layers, cfg.model.d_model)

    # 4. optimizer
    tx, schedule = build_optimizer(cfg.optim, cfg.train.total_steps)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    train_step = make_train_step(cfg.optim.router_bias_lr)

    # 5. data iterator + checkpoints
    train_it = datamod.make_train_iterator(
        train_source, cfg.train.batch_size, cfg.data.shuffle_seed
    )
    ckpt = CheckpointManager(cfg.train.out_dir, cfg.train.keep_checkpoints)

    start_step = 0
    if args.resume and ckpt.latest_step() is not None:
        # Continue THIS run: full state (weights + optimizer + data position + step).
        data_state, meta = ckpt.restore(model, optimizer)
        train_it.set_state(data_state)
        start_step = int(meta["step"]) + 1
        log.info("resumed from step %d", start_step - 1)
    elif args.init_from:
        # Warm-start a NEW run (e.g. SFT) from another run's WEIGHTS only. The
        # optimizer and data iterator stay fresh and training begins at step 0.
        src = CheckpointManager(args.init_from, cfg.train.keep_checkpoints)
        try:
            loaded = src.restore_model(model, args.init_from_step)
        except Exception as e:  # shape/structure mismatch -> actionable message
            raise SystemExit(
                f"--init-from failed to load weights from {args.init_from!r}: {e}\n"
                "The init-from checkpoint and this config must share the same model "
                "architecture and vocab (reuse the same tokenizer_path)."
            )
        src.close()
        log.info("initialized weights from %s (step %d); optimizer & data are fresh",
                 args.init_from, loaded)

    # 6. training loop
    log.info("training %d -> %d steps", start_step, cfg.train.total_steps)
    t0 = time.time()
    running = 0.0
    for step in range(start_step, cfg.train.total_steps):
        batch = np.asarray(next(train_it))
        inp, lab = datamod.split_batch(batch)
        loss, ce = train_step(
            model, optimizer, jnp.asarray(inp), jnp.asarray(lab)
        )
        running += float(ce)

        if (step + 1) % cfg.train.log_every == 0:
            avg = running / cfg.train.log_every
            running = 0.0
            dt = time.time() - t0
            toks = cfg.train.log_every * cfg.train.batch_size * cfg.data.seq_len
            log.info(
                "step %6d | ce %.4f | ppl %.2f | lr %.2e | %.0f tok/s",
                step + 1, avg, math.exp(min(avg, 20)),
                float(schedule(step)), toks / dt,
            )
            t0 = time.time()

        if (step + 1) % cfg.train.eval_every == 0:
            ppl = evaluate(model, val_rows, cfg.train.batch_size, cfg.train.eval_batches)
            log.info("step %6d | VAL ppl %.3f", step + 1, ppl)
            t0 = time.time()

        if (step + 1) % cfg.train.save_every == 0:
            ckpt.save(step, model, optimizer,
                      data_state=train_it.get_state(), meta={"step": step})
            log.info("saved checkpoint at step %d", step + 1)

    # final checkpoint + eval
    final = cfg.train.total_steps - 1
    ckpt.save(final, model, optimizer,
              data_state=train_it.get_state(), meta={"step": final})
    ppl = evaluate(model, val_rows, cfg.train.batch_size, cfg.train.eval_batches)
    log.info("done. final VAL ppl %.3f | checkpoint at step %d", ppl, cfg.train.total_steps)
    ckpt.close()


if __name__ == "__main__":
    main()
