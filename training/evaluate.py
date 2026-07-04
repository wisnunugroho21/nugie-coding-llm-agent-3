"""Evaluation entrypoint: perplexity, sample generation, and (optional) HumanEval.

    # perplexity on the config's validation split + a few generated samples
    python -m training.evaluate --config configs/tiny_pretrain.yaml

    # add functional HumanEval pass@1 (executes generated code — see the warning)
    python -m training.evaluate --config configs/tiny_sft.yaml --humaneval --humaneval-limit 20

What it reports
---------------
* perplexity  — token-weighted exp(mean CE) over the held-out OpenCoder split.
* samples     — greedy continuations for a few prompts, decoded to text, so you can
                eyeball whether the model produces plausible code.
* HumanEval   — optional functional correctness (pass@1): the model completes each
                prompt, the completion is run against the unit tests in a subprocess.

WARNING: HumanEval runs model-generated code on your machine. It is gated behind
`--humaneval` and each program runs in a separate process with a wall-clock timeout,
but you should still run it in a disposable/sandboxed environment.
"""

from __future__ import annotations

import argparse
import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from kimi_linear_gdn2 import KimiLinear

from . import data as datamod
from .checkpoint import CheckpointManager
from .config import ExperimentConfig, load_config
from .loop import build_optimizer
from .tokenizer import CodeTokenizer
from .train import build_data, evaluate
from .utils import get_logger

log = get_logger()


# --------------------------------------------------------------------------- #
def load_for_eval(cfg: ExperimentConfig, step: int | None = None):
    """Rebuild the model and restore weights from the latest (or given) checkpoint."""
    tok = CodeTokenizer.load(cfg.train.tokenizer_path)
    cfg.model.vocab_size = tok.vocab_size
    cfg.validate()

    model = KimiLinear(cfg.model, rngs=nnx.Rngs(cfg.train.seed))
    tx, _ = build_optimizer(cfg.optim, cfg.train.total_steps)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    ckpt = CheckpointManager(cfg.train.out_dir, cfg.train.keep_checkpoints)
    _, meta = ckpt.restore(model, optimizer, step)
    ckpt.close()
    log.info("restored checkpoint at step %s", meta.get("step"))
    return model, tok


# --------------------------------------------------------------------------- #
def generate_text(
    model: KimiLinear, tok: CodeTokenizer, prompt_ids: list[int], max_new_tokens: int
) -> str:
    """Greedy decode a single prompt and return the decoded continuation (cut at <eos>)."""
    ids = jnp.asarray([prompt_ids], dtype=jnp.int32)  # [1, P]
    out = model.generate(ids, max_new_tokens)  # [1, max_new_tokens]
    gen = np.asarray(out[0]).tolist()
    if tok.eos_id in gen:
        gen = gen[: gen.index(tok.eos_id)]
    return tok.decode(gen)


def sample_prompts(cfg: ExperimentConfig, tok: CodeTokenizer) -> list[tuple[str, list[int]]]:
    """A handful of task-appropriate prompts as (display, token_ids)."""
    if cfg.data.task == "sft":
        instrs = [
            "Write a Python function that returns the nth Fibonacci number.",
            "Implement binary search over a sorted list in Python.",
        ]
        return [
            (i, [tok.bos_id, tok.user_id] + tok.encode_raw(i) + [tok.assistant_id])
            for i in instrs
        ]
    seeds = ["def quicksort(arr):", "import numpy as np\n\ndef softmax(x):"]
    return [(s, tok.encode(s, add_special=True)[:-1]) for s in seeds]  # drop trailing <eos>


# --------------------------------------------------------------------------- #
#  HumanEval pass@1 (optional; executes generated code)
# --------------------------------------------------------------------------- #
def _run_humaneval_program(program: str, timeout: float) -> bool:
    """Execute one HumanEval program in a subprocess. True iff it exits cleanly."""
    import subprocess
    import sys
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as f:
        f.write(program)
        f.flush()
        try:
            r = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, timeout=timeout, text=True,
            )
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False


def humaneval_pass_at_1(
    model: KimiLinear, tok: CodeTokenizer, limit: int, max_new_tokens: int,
    timeout: float, sft: bool,
) -> float:
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")
    n = min(limit, len(ds))
    passed = 0
    for i in range(n):
        ex = ds[i]
        prompt = ex["prompt"]
        # An SFT model expects the instruction wrapped as a user turn; a base
        # (pretrained) model just completes the raw prompt.
        if sft:
            pid = [tok.bos_id, tok.user_id] + tok.encode_raw(prompt) + [tok.assistant_id]
        else:
            pid = tok.encode(prompt, add_special=True)[:-1]
        completion = generate_text(model, tok, pid, max_new_tokens)
        program = (
            prompt + completion + "\n" + ex["test"] + f"\ncheck({ex['entry_point']})\n"
        )
        ok = _run_humaneval_program(program, timeout)
        passed += int(ok)
        log.info("  [%2d/%2d] %s %s", i + 1, n, ex["task_id"], "PASS" if ok else "fail")
    return passed / n if n else float("nan")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--no-ppl", action="store_true", help="skip the perplexity pass")
    ap.add_argument("--humaneval", action="store_true", help="run HumanEval pass@1 (executes code)")
    ap.add_argument("--humaneval-limit", type=int, default=20)
    ap.add_argument("--humaneval-timeout", type=float, default=10.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, tok = load_for_eval(cfg, args.step)

    # 1. perplexity
    if not args.no_ppl:
        _, val_rows = build_data(cfg, tok)
        ppl = evaluate(model, val_rows, cfg.train.batch_size, max_batches=10_000)
        log.info("VAL perplexity: %.4f", ppl)

    # 2. sample generations
    log.info("--- sample generations ---")
    for display, pid in sample_prompts(cfg, tok):
        text = generate_text(model, tok, pid, args.max_new_tokens)
        log.info("PROMPT: %s", display)
        log.info("OUTPUT: %s\n", text)

    # 3. optional HumanEval
    if args.humaneval:
        log.info("--- HumanEval pass@1 (limit=%d) ---", args.humaneval_limit)
        p1 = humaneval_pass_at_1(
            model, tok, args.humaneval_limit, args.max_new_tokens,
            args.humaneval_timeout, sft=(cfg.data.task == "sft"),
        )
        log.info("HumanEval pass@1: %.3f", p1)


if __name__ == "__main__":
    main()
