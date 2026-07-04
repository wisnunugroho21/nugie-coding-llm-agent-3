"""End-to-end smoke test — runs the WHOLE pipeline offline (no HuggingFace download).

It feeds synthetic code-like documents through the real tokenizer, data-packing,
model, Optax optimizer, jitted train/eval steps, and Orbax checkpoint save/restore.
Run it to verify the plumbing before launching a real OpenCoder run:

    python -m tests.smoke_test        # or:  python tests/smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flax.nnx as nnx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from kimi_linear_gdn2 import KimiLinear, KimiLinearConfig, count_params  # noqa: E402
from training import data as datamod  # noqa: E402
from training.checkpoint import CheckpointManager  # noqa: E402
from training.config import OptimConfig  # noqa: E402
from training.loop import build_optimizer, eval_step, make_train_step  # noqa: E402
from training.tokenizer import CodeTokenizer  # noqa: E402


def synth_docs(n: int) -> list[str]:
    """Repetitive pseudo-code so a tiny model can actually drive the loss down."""
    docs = []
    for i in range(n):
        docs.append(
            f"def func_{i % 7}(x):\n"
            f"    y = x + {i % 5}\n"
            f"    for k in range(y):\n"
            f"        y = y * 2 + k\n"
            f"    return y\n\n"
            f"result = func_{i % 7}({i})\nprint(result)\n"
        )
    return docs


def main():
    tmp = Path(tempfile.mkdtemp(prefix="kimi_smoke_"))
    print(f"[smoke] workdir: {tmp}")

    # 1. tokenizer -------------------------------------------------------
    tok_path = tmp / "tokenizer.json"
    tok = CodeTokenizer.train(synth_docs(400), vocab_size=512, save_path=tok_path)
    assert tok.vocab_size <= 512
    reloaded = CodeTokenizer.load(tok_path)
    assert reloaded.encode("def f(x): return x") == tok.encode("def f(x): return x")
    print(f"[smoke] tokenizer OK (vocab={tok.vocab_size})")

    seq_len = 64  # multiple of gdn_chunk_size below

    # 2. data pipeline (pretrain + sft) ---------------------------------
    pre_rows = datamod.build_pretrain_rows(iter(synth_docs(400)), tok, seq_len)
    assert pre_rows.ndim == 3 and pre_rows.shape[1:] == (2, seq_len)
    train_rows, val_rows = datamod.train_val_split(pre_rows, 0.1, 32, seed=0)
    assert len(train_rows) > 0 and len(val_rows) > 0

    sft_pairs = [(f"Write function {i}", f"def g_{i}():\n    return {i}") for i in range(200)]
    sft_rows = datamod.build_sft_rows(iter(sft_pairs), tok, seq_len)
    # prompt is masked -> some labels are IGNORE, some are supervised
    assert (sft_rows[:, 1] == datamod.IGNORE).any() and (sft_rows[:, 1] != datamod.IGNORE).any()
    print(f"[smoke] data OK (pretrain {len(pre_rows)} rows, sft {len(sft_rows)} rows)")

    # 3. model -----------------------------------------------------------
    cfg = KimiLinearConfig(
        vocab_size=tok.vocab_size, d_model=64, n_layers=2, full_attn_period=2,
        gdn_num_heads=2, gdn_head_k_dim=16, gdn_head_v_dim=16, gdn_chunk_size=32,
        gdn_conv_size=4, mla_num_q_heads=4, mla_num_kv_heads=1, mla_head_dim=16,
        max_seq_len=128, moe_d_ff=64, moe_n_routed=4, moe_n_shared=1, moe_top_k=2,
        compute_dtype="float32",
    )
    model = KimiLinear(cfg, rngs=nnx.Rngs(0))
    print(f"[smoke] model OK ({count_params(model)} params)")

    # 4. optimizer + train steps ----------------------------------------
    tx, schedule = build_optimizer(OptimConfig(peak_lr=1e-3, warmup_steps=5), total_steps=60)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    train_step = make_train_step(router_bias_lr=1e-3)
    it = datamod.make_train_iterator(train_rows, batch_size=4, seed=0)

    first_loss = None
    for step in range(40):
        batch = np.asarray(next(it))
        inp, lab = datamod.split_batch(batch)
        loss, ce = train_step(model, optimizer, jnp.asarray(inp), jnp.asarray(lab))
        loss = float(loss)
        assert np.isfinite(loss), f"non-finite loss at step {step}"
        if first_loss is None:
            first_loss = loss
    print(f"[smoke] trained 40 steps: loss {first_loss:.3f} -> {loss:.3f}")
    assert loss < first_loss, "loss did not decrease on the synthetic task"

    # 5. eval ------------------------------------------------------------
    ls, nt = eval_step(model, jnp.asarray(val_rows[:4, 0]), jnp.asarray(val_rows[:4, 1]))
    ppl = float(jnp.exp(ls / jnp.maximum(nt, 1)))
    assert np.isfinite(ppl)
    print(f"[smoke] eval OK (val ppl {ppl:.2f})")

    # 6. checkpoint save/restore + data-iterator resume ------------------
    ckpt = CheckpointManager(str(tmp), max_to_keep=2)
    data_state = it.get_state()
    ckpt.save(39, model, optimizer, data_state=data_state, meta={"step": 39})
    logits_before, _ = model(jnp.asarray(val_rows[:2, 0]))

    # perturb the model, then restore and confirm weights + data position come back
    optimizer.update(model, nnx.state(model, nnx.Param))  # scribble on the weights
    dstate, meta = ckpt.restore(model, optimizer)
    it.set_state(dstate)
    logits_after, _ = model(jnp.asarray(val_rows[:2, 0]))
    assert meta["step"] == 39
    assert jnp.allclose(logits_before, logits_after, atol=1e-4), "restore did not recover weights"
    ckpt.close()
    print("[smoke] checkpoint save/restore OK")

    # 7. generation ------------------------------------------------------
    prompt = tok.encode("def func_1(x):", add_special=True)[:-1]
    out = model.generate(jnp.asarray([prompt], dtype=jnp.int32), max_new_tokens=16)
    text = tok.decode(np.asarray(out[0]).tolist())
    assert isinstance(text, str)
    print(f"[smoke] generation OK -> {text!r}")

    # 8. one SFT step to exercise the masked-loss path -------------------
    sft_it = datamod.make_train_iterator(sft_rows, batch_size=4, seed=1)
    b = np.asarray(next(sft_it))
    inp, lab = datamod.split_batch(b)
    loss, _ = train_step(model, optimizer, jnp.asarray(inp), jnp.asarray(lab))
    assert np.isfinite(float(loss))
    print("[smoke] SFT step OK")

    print("\n[smoke] ALL CHECKS PASSED ✓")


if __name__ == "__main__":
    main()
