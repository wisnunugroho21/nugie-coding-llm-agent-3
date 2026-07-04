"""OpenCoder -> tokenized .npy shards on disk -> Grain input pipeline.

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

Sharded on-disk pipeline (scales beyond RAM)
--------------------------------------------
Tokenization+packing happens ONCE, streaming from HuggingFace: `write_shards`
routes every `1/val_fraction`-th row to a small in-memory val split (capped at
`max_val_docs`) and appends the rest to fixed-size `train-NNNNN.npy` shard files
under `shards_dir`, plus a `manifest.json` recording shard sizes and a fingerprint
of the data config so stale shards are detected instead of silently reused.

Training then never loads the corpus into RAM: `ShardedRowSource` is a
random-access Grain source over the shards backed by `np.load(mmap_mode="r")` —
the OS pages in only the rows actually read. `make_train_iterator` builds the
Grain dataset over it (globally shuffled, repeated, batched) and yields a
`grain.DatasetIterator` whose get_state/set_state checkpoint the exact position
in the shuffled stream (see checkpoint.py), same as the old in-memory pipeline.

The in-memory `build_*_rows` builders are kept for tests and small corpora; the
trainer itself goes through `ensure_shards`.
"""

from __future__ import annotations

import dataclasses
import json
from itertools import islice
from pathlib import Path
from typing import Iterator, Sequence

import grain
import numpy as np

from .config import DataConfig, SourceSpec
from .tokenizer import CodeTokenizer
from .utils import get_logger

log = get_logger()

IGNORE = -100  # label value excluded from the cross-entropy loss

_MANIFEST = "manifest.json"
_VAL_FILE = "val.npy"


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
#  Tokenize + pack into [2, seq_len] rows (streaming generators)
# --------------------------------------------------------------------------- #
def iter_pretrain_rows(
    docs: Iterator[str], tok: CodeTokenizer, seq_len: int
) -> Iterator[np.ndarray]:
    """Concatenate tokenized docs and yield contiguous [2, seq_len] windows."""
    window = seq_len + 1
    buf: list[int] = []
    for doc in docs:
        buf.extend(tok.encode(doc))  # <bos> ... <eos> via the tokenizer template
        # Emit as many full windows as the buffer allows, then keep the remainder.
        while len(buf) >= window:
            chunk = np.asarray(buf[:window], dtype=np.int32)
            yield np.stack([chunk[:-1], chunk[1:]])  # [2, seq_len]
            del buf[:seq_len]  # slide by seq_len (1-token overlap for the shift)


def iter_sft_rows(
    pairs: Iterator[tuple[str, str]], tok: CodeTokenizer, seq_len: int
) -> Iterator[np.ndarray]:
    """Format each (instruction, output) pair into a prompt-masked [2, seq_len] row."""
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

        yield np.stack([inp, lab])  # [2, seq_len]


def iter_rows(cfg: DataConfig, tok: CodeTokenizer) -> Iterator[np.ndarray]:
    """Streaming [2, seq_len] rows for the configured task, straight from the Hub."""
    if cfg.task == "pretrain":
        return iter_pretrain_rows(iter_pretrain_docs(cfg), tok, cfg.seq_len)
    return iter_sft_rows(iter_sft_pairs(cfg), tok, cfg.seq_len)


# ---- In-memory builders (tests / small corpora) ---------------------------- #
def build_pretrain_rows(
    docs: Iterator[str], tok: CodeTokenizer, seq_len: int, max_rows: int | None = None
) -> np.ndarray:
    """Materialize pretrain rows in memory (the trainer streams via shards instead)."""
    rows = list(islice(iter_pretrain_rows(docs, tok, seq_len), max_rows))
    if not rows:
        raise ValueError("no full sequences produced — corpus too small for seq_len.")
    return np.stack(rows)


def build_sft_rows(
    pairs: Iterator[tuple[str, str]],
    tok: CodeTokenizer,
    seq_len: int,
    max_rows: int | None = None,
) -> np.ndarray:
    """Materialize SFT rows in memory (the trainer streams via shards instead)."""
    rows = list(islice(iter_sft_rows(pairs, tok, seq_len), max_rows))
    if not rows:
        raise ValueError("no SFT rows produced — check the source fields.")
    return np.stack(rows)


# --------------------------------------------------------------------------- #
#  Sharded on-disk dataset: tokenize once, then stream via mmap
# --------------------------------------------------------------------------- #
def _fingerprint(cfg: DataConfig, vocab_size: int) -> dict:
    """Everything that changes the CONTENT of the packed rows. Stored in the
    manifest; a mismatch on load means the shards were built for a different
    config/tokenizer and must not be silently reused. (rows_per_shard is layout,
    not content, so it is deliberately excluded.)"""
    return {
        "task": cfg.task,
        "seq_len": cfg.seq_len,
        "vocab_size": vocab_size,
        "text_field": cfg.text_field,
        "instruction_field": cfg.instruction_field,
        "response_field": cfg.response_field,
        "val_fraction": cfg.val_fraction,
        "max_val_docs": cfg.max_val_docs,
        "sources": [dataclasses.asdict(s) for s in cfg.sources],
    }


def write_shards(
    cfg: DataConfig,
    tok: CodeTokenizer,
    shards_dir: str | Path,
    rows: Iterator[np.ndarray] | None = None,
) -> dict:
    """Tokenize + pack the corpus ONCE, writing it as .npy shards + manifest.

    Streams rows (never holding more than one shard in memory): every
    `1/val_fraction`-th row goes to the val split (until `max_val_docs` rows),
    the rest are appended to `train-NNNNN.npy` files of `cfg.rows_per_shard`
    rows each ([n, 2, seq_len] int32). Returns the manifest dict.

    `rows` overrides the source stream (used by tests); by default rows come
    from the configured HuggingFace sources via `iter_rows`.
    """
    out = Path(shards_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = iter_rows(cfg, tok)

    # Deterministic streaming val split: every k-th row until the cap. (The old
    # in-memory pipeline shuffled before splitting; a streaming split instead
    # spreads val rows evenly over the first cap*k rows of the corpus.)
    val_every = round(1.0 / cfg.val_fraction) if cfg.val_fraction > 0 else 0

    shards: list[dict] = []
    buf: list[np.ndarray] = []
    val: list[np.ndarray] = []
    n_train = 0

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        name = f"train-{len(shards):05d}.npy"
        np.save(out / name, np.stack(buf))
        shards.append({"file": name, "rows": len(buf)})
        log.info("wrote %s (%d rows, %d shards so far)", name, len(buf), len(shards))
        buf = []

    for i, row in enumerate(rows):
        if val_every and i % val_every == 0 and len(val) < cfg.max_val_docs:
            val.append(row)
            continue
        buf.append(row)
        n_train += 1
        if len(buf) >= cfg.rows_per_shard:
            flush()
    flush()

    if n_train == 0:
        raise ValueError("no full sequences produced — corpus too small for seq_len.")

    val_arr = (
        np.stack(val) if val else np.zeros((0, 2, cfg.seq_len), dtype=np.int32)
    )
    np.save(out / _VAL_FILE, val_arr)

    manifest = {
        "format": 1,
        "rows_per_shard": cfg.rows_per_shard,
        "n_train_rows": n_train,
        "n_val_rows": len(val),
        "shards": shards,
        "fingerprint": _fingerprint(cfg, tok.vocab_size),
    }
    (out / _MANIFEST).write_text(json.dumps(manifest, indent=2))
    log.info(
        "shards complete: %d train rows in %d shards + %d val rows -> %s",
        n_train, len(shards), len(val), out,
    )
    return manifest


class ShardedRowSource:
    """Random-access Grain source over the tokenized train shards, mmap-backed.

    Satisfies `grain.sources.RandomAccessDataSource` (__len__/__getitem__), so it
    drops into `grain.MapDataset.source` exactly where the in-memory ndarray used
    to go — global shuffle and exact checkpoint resume keep working — but shards
    are opened with np.load(mmap_mode="r"), so only the rows actually read are
    paged in. Row lookups binary-search the cumulative shard sizes.
    """

    def __init__(self, shard_paths: Sequence[str | Path], shard_rows: Sequence[int]):
        if len(shard_paths) != len(shard_rows):
            raise ValueError("shard_paths and shard_rows must have equal length")
        self._paths = [Path(p) for p in shard_paths]
        self._mmaps: list[np.ndarray | None] = [None] * len(self._paths)
        self._starts = np.concatenate(
            [[0], np.cumsum(np.asarray(shard_rows, dtype=np.int64))]
        )

    def __len__(self) -> int:
        return int(self._starts[-1])

    def __getitem__(self, idx: int) -> np.ndarray:
        s = int(np.searchsorted(self._starts, idx, side="right")) - 1
        if self._mmaps[s] is None:
            self._mmaps[s] = np.load(self._paths[s], mmap_mode="r")
        # np.array copies the row OUT of the mmap, so batches never hold live
        # references into the file mapping.
        return np.array(self._mmaps[s][int(idx - self._starts[s])])

    def __getstate__(self) -> dict:
        # mmap handles are not picklable (e.g. Grain worker processes); drop them
        # and let each process reopen its own lazily.
        state = self.__dict__.copy()
        state["_mmaps"] = [None] * len(self._paths)
        return state


def load_shards(
    cfg: DataConfig, shards_dir: str | Path, vocab_size: int
) -> tuple[ShardedRowSource, np.ndarray]:
    """Open existing shards. Returns (train source, val rows [n, 2, seq_len]).

    Raises if the manifest's fingerprint does not match the current config, so a
    changed corpus/seq_len/tokenizer cannot silently train on stale shards.
    """
    out = Path(shards_dir)
    manifest = json.loads((out / _MANIFEST).read_text())
    want = _fingerprint(cfg, vocab_size)
    if manifest["fingerprint"] != want:
        raise ValueError(
            f"shards at {out} were built for a different data config/tokenizer:\n"
            f"  on disk: {manifest['fingerprint']}\n"
            f"  wanted:  {want}\n"
            "Delete the directory (or point data.shards_dir elsewhere) to rebuild."
        )
    source = ShardedRowSource(
        [out / s["file"] for s in manifest["shards"]],
        [s["rows"] for s in manifest["shards"]],
    )
    val_rows = np.load(out / _VAL_FILE)
    return source, val_rows


def ensure_shards(
    cfg: DataConfig, tok: CodeTokenizer, shards_dir: str | Path
) -> tuple[ShardedRowSource, np.ndarray]:
    """Reuse the shards at `shards_dir` if present (and fingerprint-compatible),
    else tokenize the corpus and write them first. The trainer's entry point."""
    if not (Path(shards_dir) / _MANIFEST).exists():
        log.info("no shards at %s — tokenizing the corpus (one-time) ...", shards_dir)
        write_shards(cfg, tok, shards_dir)
    else:
        log.info("reusing shards at %s", shards_dir)
    return load_shards(cfg, shards_dir, tok.vocab_size)


# --------------------------------------------------------------------------- #
#  Train / val split + Grain iterators
# --------------------------------------------------------------------------- #
def train_val_split(
    rows: np.ndarray, val_fraction: float, max_val: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """In-memory split (tests / small corpora; the trainer splits at shard-write time)."""
    n = rows.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    n_val = min(max_val, int(round(n * val_fraction)))
    n_val = max(0, min(n_val, n - 1))  # always keep at least one train row
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return rows[train_idx], rows[val_idx]


def make_train_iterator(
    rows: np.ndarray | ShardedRowSource, batch_size: int, seed: int
) -> grain.DatasetIterator:
    """Endless, shuffled, batched iterator for step-based training.

    `rows` is any random-access source of [2, seq_len] rows — the mmap-backed
    `ShardedRowSource` for real runs, or a plain ndarray in tests. Batches are
    [B, 2, seq_len] int32. The iterator supports get_state/set_state so a resumed
    run continues from the exact same position in the shuffled stream.
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
