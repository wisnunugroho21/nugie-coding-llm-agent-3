"""OpenCoder -> tokenized examples -> Grain input pipeline.

Two tasks, one shared representation
------------------------------------
Every training example is a fixed-shape int32 array of shape [2, seq_len]:

    row[0] = input_ids   (fed to the model)
    row[1] = labels      (next-token targets; -100 == ignored in the loss)

* pretrain: documents from OpenCoder's `opc-annealing-corpus` are tokenized
  (each wrapped in <bos> ... <eos>), concatenated into one long stream, then
  sliced into windows of length seq_len+1. input = win[:-1], labels = win[1:].
  Every position is supervised.

* sft: (instruction, output) pairs from `opc-sft-stage1` / `opc-sft-stage2` are
  formatted as  <bos> <|user|> instruction <|assistant|> output <eos>  and the
  loss is masked (-100) over the prompt, so only the answer is supervised.

The uniform [2, seq_len] shape lets Grain treat both tasks identically: a
`grain.MapDataset` over the rows, shuffled, repeated (endless, step-based training),
and batched to [B, 2, seq_len]. `train_step` slices out inputs/labels. The Grain
iterator exposes get_state/set_state so training resumes exactly (see checkpoint.py).
"""

from __future__ import annotations

from typing import Iterator, Sequence

import grain
import numpy as np

from .config import DataConfig, SourceSpec
from .tokenizer import CodeTokenizer

IGNORE = -100  # label value excluded from the cross-entropy loss


# --------------------------------------------------------------------------- #
#  HuggingFace streaming
# --------------------------------------------------------------------------- #
def _load_hf(spec: SourceSpec, streaming: bool, cache_dir: str | None):
    from datasets import load_dataset

    return load_dataset(
        spec.repo,
        spec.name,
        split=spec.split,
        streaming=streaming,
        cache_dir=cache_dir,
    )


def iter_pretrain_docs(cfg: DataConfig) -> Iterator[str]:
    """Yield raw document strings from every configured corpus source."""
    for spec in cfg.sources:
        ds = _load_hf(spec, cfg.streaming, cfg.hf_cache_dir)
        for i, rec in enumerate(ds):
            if spec.max_docs is not None and i >= spec.max_docs:
                break
            text = rec.get(cfg.text_field)
            if text:
                yield text


def iter_sft_pairs(cfg: DataConfig) -> Iterator[tuple[str, str]]:
    """Yield (instruction, output) pairs from every configured SFT source."""
    for spec in cfg.sources:
        ds = _load_hf(spec, cfg.streaming, cfg.hf_cache_dir)
        for i, rec in enumerate(ds):
            if spec.max_docs is not None and i >= spec.max_docs:
                break
            instr = rec.get(cfg.instruction_field)
            resp = rec.get(cfg.response_field)
            if instr and resp:
                yield instr, resp


def corpus_for_tokenizer(cfg: DataConfig) -> Iterator[str]:
    """Flat text stream used to TRAIN the tokenizer (task-agnostic)."""
    if cfg.task == "pretrain":
        yield from iter_pretrain_docs(cfg)
    else:
        for instr, resp in iter_sft_pairs(cfg):
            yield instr + "\n" + resp


# --------------------------------------------------------------------------- #
#  Tokenize + pack into [2, seq_len] rows
# --------------------------------------------------------------------------- #
def build_pretrain_rows(
    docs: Iterator[str], tok: CodeTokenizer, seq_len: int, max_rows: int | None = None
) -> np.ndarray:
    """Concatenate tokenized docs and slice into contiguous [2, seq_len] windows."""
    window = seq_len + 1
    buf: list[int] = []
    rows: list[np.ndarray] = []
    for doc in docs:
        buf.extend(tok.encode(doc))  # <bos> ... <eos> via the tokenizer template
        # Emit as many full windows as the buffer allows, then keep the remainder.
        while len(buf) >= window:
            chunk = np.asarray(buf[:window], dtype=np.int32)
            rows.append(np.stack([chunk[:-1], chunk[1:]]))  # [2, seq_len]
            del buf[:seq_len]  # slide by seq_len (1-token overlap for the shift)
            if max_rows is not None and len(rows) >= max_rows:
                return np.stack(rows)
    if not rows:
        raise ValueError("no full sequences produced — corpus too small for seq_len.")
    return np.stack(rows)


def build_sft_rows(
    pairs: Iterator[tuple[str, str]],
    tok: CodeTokenizer,
    seq_len: int,
    max_rows: int | None = None,
) -> np.ndarray:
    """Format each (instruction, output) pair into a prompt-masked [2, seq_len] row."""
    rows: list[np.ndarray] = []
    for instr, resp in pairs:
        # <bos> <|user|> instr <|assistant|> resp <eos>   (built without the auto
        # template so we control the special-token placement exactly).
        prompt = (
            [tok.bos_id, tok.user_id]
            + tok.encode_raw(instr)
            + [tok.assistant_id]
        )
        answer = tok.encode_raw(resp) + [tok.eos_id]
        ids = prompt + answer
        if len(ids) < 2:
            continue
        ids = ids[: seq_len + 1]

        inp = np.full(seq_len, tok.pad_id, dtype=np.int32)
        lab = np.full(seq_len, IGNORE, dtype=np.int32)

        seq = np.asarray(ids, dtype=np.int32)
        n = seq.shape[0] - 1  # number of (input, target) positions
        inp[:n] = seq[:-1]
        lab[:n] = seq[1:]
        # Mask the prompt: target position j predicts ids[j+1]; supervise only when
        # ids[j+1] is part of the answer (index >= len(prompt)).
        mask_until = max(0, len(prompt) - 1)
        lab[:mask_until] = IGNORE

        rows.append(np.stack([inp, lab]))  # [2, seq_len]
        if max_rows is not None and len(rows) >= max_rows:
            break
    if not rows:
        raise ValueError("no SFT rows produced — check the source fields.")
    return np.stack(rows)


def build_rows(cfg: DataConfig, tok: CodeTokenizer, max_rows: int | None = None) -> np.ndarray:
    if cfg.task == "pretrain":
        return build_pretrain_rows(iter_pretrain_docs(cfg), tok, cfg.seq_len, max_rows)
    return build_sft_rows(iter_sft_pairs(cfg), tok, cfg.seq_len, max_rows)


# --------------------------------------------------------------------------- #
#  Train / val split + Grain iterators
# --------------------------------------------------------------------------- #
def train_val_split(
    rows: np.ndarray, val_fraction: float, max_val: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    n = rows.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    n_val = min(max_val, int(round(n * val_fraction)))
    n_val = max(0, min(n_val, n - 1))  # always keep at least one train row
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return rows[train_idx], rows[val_idx]


def make_train_iterator(
    rows: np.ndarray, batch_size: int, seed: int
) -> grain.DatasetIterator:
    """Endless, shuffled, batched iterator for step-based training.

    Batches are [B, 2, seq_len] int32. The iterator supports get_state/set_state so
    a resumed run continues from the exact same position in the shuffled stream.
    """
    ds = (
        grain.MapDataset.source(rows)
        .shuffle(seed=seed)
        .repeat()  # infinite: training is bounded by total_steps, not epochs
        .batch(batch_size, drop_remainder=True)
        .to_iter_dataset()
    )
    return iter(ds)


def iterate_eval_batches(
    rows: np.ndarray, batch_size: int, max_batches: int
) -> Iterator[np.ndarray]:
    """Finite, in-order batches over `rows` (for perplexity eval). Drops the tail."""
    n = rows.shape[0]
    nb = min(max_batches, n // batch_size)
    for b in range(nb):
        yield rows[b * batch_size : (b + 1) * batch_size]


def split_batch(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[B, 2, seq_len] -> (input_ids [B, seq_len], labels [B, seq_len])."""
    return batch[:, 0], batch[:, 1]
