"""Experiment configuration: nested dataclasses + a YAML loader.

A single YAML file fully describes a run. It has four sections that map onto the
dataclasses below:

    model:   -> KimiLinearConfig      (the network; see kimi_linear_gdn2.py)
    data:    -> DataConfig            (which OpenCoder splits, task, seq_len, ...)
    optim:   -> OptimConfig           (Optax schedule + AdamW hyper-parameters)
    train:   -> TrainConfig           (batch size, steps, checkpoint/eval cadence)

`load_config(path)` parses the YAML and returns an `ExperimentConfig`. Unknown keys
raise, so typos fail loudly instead of being silently ignored. `vocab_size` is left
at a placeholder in the YAML and overwritten at runtime from the trained tokenizer,
so the two can never drift apart.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from kimi_linear_gdn2 import KimiLinearConfig


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class SourceSpec:
    """One HuggingFace dataset split to pull from.

    `repo` is the dataset id (e.g. "OpenCoder-LLM/opc-annealing-corpus"), `name`
    the config/subset (e.g. "algorithmic_corpus"), `split` usually "train".
    `weight` lets you over/under-sample a source relative to the others when
    interleaving (used only as a soft cap via `max_docs`).
    """

    repo: str
    name: str | None = None
    split: str = "train"
    max_docs: int | None = None  # hard cap on docs pulled from this source


@dataclasses.dataclass
class DataConfig:
    # "pretrain": next-token LM on a raw-text corpus (opc-annealing-corpus).
    # "sft":      instruction tuning on (instruction, output) pairs (opc-sft-*).
    task: str = "pretrain"

    sources: list[SourceSpec] = dataclasses.field(default_factory=list)

    # Field names inside the HF records.
    text_field: str = "text"  # pretrain corpus text column
    instruction_field: str = "instruction"  # sft prompt column
    response_field: str = "output"  # sft answer column

    seq_len: int = 256  # tokens per training sequence; MUST be a multiple of the
    #                     GDN-2 chunk size (model.gdn_chunk_size).
    val_fraction: float = 0.01  # held-out fraction for perplexity eval
    max_val_docs: int = 512  # cap the val set so eval stays quick

    shuffle_seed: int = 0
    streaming: bool = True  # stream from the Hub instead of a full local download
    hf_cache_dir: str | None = None

    # On-disk shard cache of packed [2, seq_len] rows (see data.write_shards).
    # The corpus is tokenized ONCE into .npy shards there and training streams
    # them via mmap — RAM no longer bounds corpus size. None -> defaults to
    # <train.out_dir>/shards at runtime; reused across runs if the fingerprint
    # (task/sources/seq_len/tokenizer) matches.
    shards_dir: str | None = None
    rows_per_shard: int = 4096  # rows per shard file (~128 MiB at seq_len 4096)

    def __post_init__(self):
        self.sources = [
            s if isinstance(s, SourceSpec) else SourceSpec(**s) for s in self.sources
        ]


# --------------------------------------------------------------------------- #
#  Optimizer
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class OptimConfig:
    peak_lr: float = 3e-4
    min_lr_ratio: float = 0.1  # end LR = peak_lr * min_lr_ratio (cosine floor)
    warmup_steps: int = 200
    weight_decay: float = 0.1  # applied to matrices only (not norms/biases)
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0  # global-norm clip; <=0 disables
    router_bias_lr: float = 1e-3  # aux-loss-free MoE load-balancing step


# --------------------------------------------------------------------------- #
#  Trainer
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class TrainConfig:
    batch_size: int = 8
    total_steps: int = 2000
    seed: int = 0

    log_every: int = 20
    eval_every: int = 200
    eval_batches: int = 20  # val batches averaged per perplexity eval
    save_every: int = 500
    keep_checkpoints: int = 3

    out_dir: str = "runs/tiny"
    tokenizer_path: str = "runs/tiny/tokenizer.json"


# --------------------------------------------------------------------------- #
#  Top-level
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class ExperimentConfig:
    model: KimiLinearConfig
    data: DataConfig
    optim: OptimConfig
    train: TrainConfig

    def validate(self) -> None:
        if self.data.seq_len % self.model.gdn_chunk_size != 0:
            raise ValueError(
                f"data.seq_len ({self.data.seq_len}) must be a multiple of "
                f"model.gdn_chunk_size ({self.model.gdn_chunk_size}): the GDN-2 "
                "chunkwise core reshapes L into L/C chunks."
            )
        if self.data.seq_len > self.model.max_seq_len:
            raise ValueError(
                f"data.seq_len ({self.data.seq_len}) exceeds model.max_seq_len "
                f"({self.model.max_seq_len})."
            )
        if not self.data.sources:
            raise ValueError("data.sources is empty — nothing to train on.")
        if self.data.task not in ("pretrain", "sft"):
            raise ValueError(f"unknown data.task {self.data.task!r}")


def _build(cls, section: dict[str, Any]):
    """Construct a dataclass from a dict, rejecting unknown keys with a clear error."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(section) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**section)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - {"model", "data", "optim", "train"}
    if unknown:
        raise ValueError(f"unknown top-level config sections: {sorted(unknown)}")

    cfg = ExperimentConfig(
        model=_build(KimiLinearConfig, raw.get("model", {})),
        data=_build(DataConfig, raw.get("data", {})),
        optim=_build(OptimConfig, raw.get("optim", {})),
        train=_build(TrainConfig, raw.get("train", {})),
    )
    cfg.validate()
    return cfg
