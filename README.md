# Kimi-Linear (GDN-2) — code LM training & evaluation pipeline

A decoder-only language model for **programming code generation**, built from
scratch in JAX / Flax NNX. The architecture follows **Kimi Linear** (hybrid 3:1
linear/full attention, NoPE MLA full-attention layers, MoE FFN) with one deliberate
substitution: the linear token mixer is **Gated DeltaNet-2** instead of Kimi Delta
Attention. See [`kimi_linear_gdn2.py`](kimi_linear_gdn2.py) for the model.

This repo adds a complete pipeline around that model:

| Stage | Library | Module |
|-------|---------|--------|
| Data loading | **Grain** | [`training/data.py`](training/data.py) |
| Optimization | **Optax** | [`training/loop.py`](training/loop.py) |
| Checkpointing | **Orbax** | [`training/checkpoint.py`](training/checkpoint.py) |
| Datasets | **OpenCoder** (HuggingFace) | [`training/data.py`](training/data.py) |

## Data — OpenCoder

Rather than building a corpus from scratch, the pipeline streams
[OpenCoder](https://huggingface.co/OpenCoder-LLM) datasets from the Hub:

- **Pretraining** — [`opc-annealing-corpus`](https://huggingface.co/datasets/OpenCoder-LLM/opc-annealing-corpus)
  (`algorithmic_corpus`, `synthetic_code_snippet`, `synthetic_qa`): raw code text,
  used for next-token language modeling.
- **Instruction tuning (SFT)** — [`opc-sft-stage1`](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage1)
  and [`opc-sft-stage2`](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2):
  `(instruction, output)` pairs, formatted with a chat template and a **prompt-masked
  loss** (only the answer is supervised).

Which splits are used is entirely config-driven (`data.sources` in the YAML).

## Install

```bash
pip install -r requirements.txt
# GPU: install the matching CUDA wheel, e.g.  pip install -U "jax[cuda12]"
```

## Quickstart

Verify the whole pipeline offline first (no download):

```bash
python -m tests.smoke_test
```

Then a real run (the tiny configs fit on a laptop CPU; scale up for GPU):

```bash
# 1. Pretrain on the OpenCoder annealing corpus
python -m training.train --config configs/tiny_pretrain.yaml

# 2. Resume if interrupted (restores weights, optimizer, and exact data position)
python -m training.train --config configs/tiny_pretrain.yaml --resume

# 3. Instruction-tune on the OpenCoder SFT pairs, warm-started from the
#    pretrained checkpoint (weights only; fresh optimizer + data).
#    Reuse the pretraining tokenizer so the vocab matches (set train.tokenizer_path
#    in the SFT config to the pretrain run's tokenizer.json).
python -m training.train --config configs/tiny_sft.yaml --init-from runs/tiny_pretrain

# 4. Evaluate: perplexity + sample generations
python -m training.evaluate --config configs/tiny_pretrain.yaml

# 5. Functional HumanEval pass@1 (executes generated code — sandbox recommended)
python -m training.evaluate --config configs/tiny_sft.yaml --humaneval --humaneval-limit 20
```

## How it works

1. **Tokenizer** ([`training/tokenizer.py`](training/tokenizer.py)) — a ByteLevel-BPE
   tokenizer is trained on a sample of the corpus the first time you run (saved to
   `train.tokenizer_path`), with `<pad>/<bos>/<eos>/<|user|>/<|assistant|>` specials.
   The model's `vocab_size` is set from the tokenizer, so the two never drift.
2. **Data** ([`training/data.py`](training/data.py)) — documents/pairs are tokenized
   and packed into fixed `[2, seq_len]` rows (`input_ids`, `labels`). A
   `grain.MapDataset` shuffles, repeats, and batches them; its iterator state is
   checkpointed so a resumed run continues from the same position.
3. **Optimizer** ([`training/loop.py`](training/loop.py)) — Optax AdamW with a linear
   warmup + cosine-decay schedule, global-norm clipping, and weight decay on matrices
   only. The loss is next-token cross-entropy **plus** the MoE aux loss; each MoE
   layer's router bias is additionally nudged toward balanced load (DeepSeek-V3
   aux-loss-free balancing) inside the jitted step.
4. **Checkpointing** ([`training/checkpoint.py`](training/checkpoint.py)) — Orbax
   `CheckpointManager` saves model + optimizer + data-iterator + metadata per step,
   keeping the last `keep_checkpoints`. `--resume` continues the same run (full
   state); `--init-from <run_dir>` warm-starts a **new** run from another run's
   weights only (fresh optimizer/data, step 0) — this is how pretrain → SFT chains.
   The two runs must share the architecture and vocab (reuse the same
   `tokenizer_path`), or the restore raises a clear mismatch error.
5. **Evaluation** ([`training/evaluate.py`](training/evaluate.py)) — token-weighted
   perplexity on a held-out OpenCoder split, greedy generations from a few prompts,
   and optional functional **HumanEval pass@1** (each completion is executed against
   its unit tests in a subprocess with a timeout).

## Configuration

One YAML fully describes a run, in four sections mapping to dataclasses in
[`training/config.py`](training/config.py): `model` → `KimiLinearConfig`, `data`,
`optim`, `train`. Unknown keys raise, so typos fail loudly. Two constraints are
checked: `seq_len` must be a multiple of `model.gdn_chunk_size` (the GDN-2 chunkwise
core requirement) and `seq_len <= model.max_seq_len`.

To scale up on a GPU: raise `d_model`, `n_layers`, `seq_len`, `moe_n_routed`, the
head counts, and `total_steps`, and set `compute_dtype: bfloat16`.

### Config presets

| Config | Target hardware | Size | Precision | Notes |
|--------|-----------------|------|-----------|-------|
| [`configs/tiny_pretrain.yaml`](configs/tiny_pretrain.yaml) / [`tiny_sft.yaml`](configs/tiny_sft.yaml) | Laptop CPU | ~5M | float32 | quick end-to-end sanity |
| Colab cells in [the notebook](Kimi_Linear_GDN2_Colab.ipynb) | Colab **T4** (16 GB) | ~140M | float32 | fits a free-tier GPU |
| [`configs/h200_pretrain.yaml`](configs/h200_pretrain.yaml) / [`h200_sft.yaml`](configs/h200_sft.yaml) | Single **H200** (141 GB) | **4.2B** | bfloat16 | from-scratch training |

The H200 preset is a ~4.2B-parameter model (d_model 2048, 24 layers, 16-expert MoE,
4k context) in bfloat16 — master weights + AdamW moments stay fp32 (~50 GB), leaving
~90 GB of HBM for activations. Train it from scratch, then chain into SFT:

```bash
python -m training.train --config configs/h200_pretrain.yaml
python -m training.train --config configs/h200_sft.yaml --init-from runs/h200_pretrain
```

This pipeline is **single-device** (no `jax.sharding`/`pmap` and no gradient
accumulation, so effective batch == `train.batch_size`). Tune `batch_size` /
`seq_len` down if you hit `ResourceExhaustedError`; multi-GPU would require adding
sharding to the train step.

### Scaling the data beyond memory
The trainer never holds the corpus in RAM. On first launch it streams the
configured HuggingFace sources, tokenizes and packs them into `[2, seq_len]` rows,
and writes fixed-size `.npy` shards (plus a `manifest.json` with a config
fingerprint) under `train.out_dir/shards` — or `data.shards_dir` if set. Training
then streams those shards through a Grain `IterDataset` backed by a memory-mapped
random-access source ([`training/data.py`](training/data.py):
`write_shards` / `ShardedRowSource` / `ensure_shards`), keeping global shuffling
and exact checkpoint resume (the Grain iterator state is saved with each Orbax
checkpoint). Relaunches reuse the shards if the data config and tokenizer
fingerprint match; a mismatch fails loudly instead of training on stale data.
Corpus size is bounded by disk (`~seq_len × 8` bytes per row), not RAM; `max_docs`
caps the one-time tokenization cost per source. The small val split is written at
shard time (`val.npy`) and kept in memory for perplexity eval.

> **Note on HumanEval:** `--humaneval` runs model-generated code on your machine.
> Each program runs in a separate process with a wall-clock timeout, but you should
> still run it in a disposable/sandboxed environment.
