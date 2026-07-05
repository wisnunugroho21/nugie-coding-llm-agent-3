# Multi-GPU / TPU Scaling Plan

Plan to take this repo from **single-device** (README: *"multi-GPU would require adding
`jax.sharding` to the train step"*) to **multi-GPU and multi-host TPU/GPU**, using JAX
**GSPMD** (`jax.sharding` + `jit`), phased from single-node → multi-host, targeting a
full **2D/3D parallelism** design: **FSDP + Expert Parallel (EP) + Tensor Parallel (TP)**.

The model is a ~4.2B Kimi-Linear (GDN-2) MoE (`kimi_linear_gdn2.py`): embedding, a stack
of `DecoderLayer`s (GDN-2 or MLA token mixer + `GroupedGemmMoE` channel mixer), final
norm, LM head. Training is `nnx.jit`'d in [`training/loop.py`](../training/loop.py) and
driven by the loop in [`training/train.py`](../training/train.py).

---

## 0. Design decisions (fixed for this plan)

| Decision | Choice | Rationale |
|---|---|---|
| Sharding API | **GSPMD** (`jax.jit` + `NamedSharding`), *not* `pmap` | Handles FSDP/EP/TP uniformly; multi-host native; already how `nnx.jit` works. |
| Mesh | 3D: **`("data", "expert", "model")`** | `data`=FSDP+DP, `expert`=MoE EP, `model`=TP. Any axis size 1 ⇒ that parallelism off. |
| Rollout | **Phased**: single-node FSDP → EP → TP → multi-host | Each phase independently testable; earlier phases cover most needs. |
| Correctness bar | **Loss-parity** vs single-device on CPU with 8 simulated devices, every phase | Sharding must not change math (only grad-accum / dtype may, intentionally). |

**Mesh semantics.** Global batch `B` is split over `data`. Every parameter is
**fully sharded** (ZeRO-3) over `data` on one axis; MoE experts additionally shard over
`expert`; hidden/head dims additionally shard over `model`. Effective config:
`n_devices = dp * ep * tp`.

**TPU note.** The default GDN-2 core ([`gated_deltanet_2/core.py`](../gated_deltanet_2/core.py))
is pure `jax.lax` → runs on TPU unchanged. The new
[`gated_deltanet_2/triton_chunkwise.py`](../gated_deltanet_2/triton_chunkwise.py) is
**GPU-only**; the layer must dispatch to `core.py` when `jax.default_backend() != "gpu"`.
Confirm/guard this before any TPU run (Phase 4).

---

## Phase 0 — Foundations (device-agnostic plumbing)

No parallelism yet; just the scaffolding every later phase needs.

1. **Config surface** — add a `ParallelConfig` to [`training/config.py`](../training/config.py)
   and a top-level `parallel:` YAML section:
   ```yaml
   parallel:
     dp: 8        # data / FSDP axis
     ep: 1        # expert-parallel axis
     tp: 1        # tensor-parallel axis
     grad_accum: 1
     fsdp: true   # shard params over dp (ZeRO-3) vs pure replication (DDP)
   ```
   Validate divisibility in `ExperimentConfig.validate()`:
   `dp*ep*tp == len(jax.devices())`, `batch_size % dp == 0`,
   `moe_n_routed % ep == 0`, `d_model % tp == 0`, all head counts `% tp == 0`,
   `moe_d_ff % tp == 0`, `vocab_size % tp == 0`.

2. **Mesh module** — new `training/parallel.py`:
   `build_mesh(cfg) -> jax.sharding.Mesh` via
   `jax.make_mesh((dp, ep, tp), ("data","expert","model"))`. Helpers:
   `data_sharding(mesh)` = `NamedSharding(mesh, P("data", None))` for `[B, ...]` batches,
   and `replicated(mesh)`.

3. **Fix CPU multi-device simulation** — [`training/utils.py`](../training/utils.py)
   `set_xla_flags_for_cpu` currently hard-codes device count `1`
   (`--xla_force_host_platform_device_count={1}`). Change it to honor an env/config
   `CPU_DEVICE_COUNT` so tests can request 8 fake devices. This is the backbone of the
   correctness harness — do it first.

4. **Logical axis names** — define constants for parameter partition specs in one place
   (e.g. `EMBED_VOCAB = ("data", "model")`) so sharding rules live in one file, not
   scattered across modules.

**Deliverable:** config parses, mesh builds, `python -c "import jax; print(jax.devices())"`
shows 8 CPU devices under the test flag. No behavior change.

---

## Phase 1 — FSDP on a single node (ZeRO-3 data-parallel)

Goal: shard params + optimizer state + grads over the `data` axis; replicate compute;
shard the batch. Smallest change that unlocks multi-GPU and fits bigger models. `ep=tp=1`.

1. **Annotate parameter shardings.** Attach sharding metadata so `nnx.state(model)` carries
   a `PartitionSpec` per leaf. Two mechanisms:
   - Flax NNX `nnx.Linear`/`nnx.Embed`: pass `kernel_init=nnx.with_partitioning(init, spec)`
     (wrap `_XAVIER`) in `kimi_linear_gdn2.py`, `multi_latent_attention/*`, `gated_deltanet_2/layer.py`.
   - Raw `nnx.Param` (MoE `w_in`/`w_out`/shared, `router_bias`): wrap the initializer value
     with `nnx.with_partitioning(...)` / set `.sharding` metadata.

   **FSDP rule (Phase 1):** shard exactly one (largest) axis of each param over `"data"`,
   rest `None`. E.g. `lm_head` kernel `[d_model, vocab] → P(None, "data")`; embedding
   `[vocab, d_model] → P("data", None)`; MoE `w_in [E, d, 2ff] → P(None, "data", None)`.

2. **Sharded initialization without single-device OOM.** Do **not** build the 4B model on one
   device then copy. Instead:
   ```python
   abstract = nnx.eval_shape(lambda: KimiLinear(cfg.model, rngs=nnx.Rngs(seed)))
   specs = nnx.get_partition_spec(nnx.state(abstract))
   shardings = jax.tree.map(lambda s: NamedSharding(mesh, s), specs)
   state = jax.jit(init_fn, out_shardings=shardings)()   # allocates already-sharded
   ```
   Wrap in `nnx.split`/`nnx.merge` to reattach to the NNX module. Do the same for the
   optimizer state (`nnx.Optimizer`) so AdamW moments are born sharded.

3. **Sharded train step.** `make_train_step` in [`training/loop.py`](../training/loop.py) stays
   `nnx.jit`, but:
   - `in_shardings`: `input_ids`/`labels` → `P("data", None)`; model/opt state → their annotated shardings.
   - Keep `donate_argnums=(0,1)` — donation is compatible with sharding.
   - GSPMD inserts the ZeRO-3 all-gather (params) / reduce-scatter (grads) automatically.
   - The Python `for layer in model.layers` router-bias update still works under `jit`;
     ensure `group_sizes [n_layers, E]` is replicated (`with_sharding_constraint(..., P(None, None))`).

4. **Data feeding (single process).** In [`training/train.py`](../training/train.py), replace
   `jnp.asarray(next(train_it))` with `jax.device_put(batch, data_sharding(mesh))`. Global
   batch is one host array sharded across local devices — no pipeline change yet.

5. **Checkpointing.** Orbax `StandardSave/Restore` already persists sharded arrays and restores
   **into** the target state's shardings (our restore is target-based — see
   [`training/checkpoint.py`](../training/checkpoint.py)). Because Phase 2 init produces a
   sharded target, restore "just works," but **add a test**: save on 8 devices, restore, compare.

6. **Grad accumulation (optional here).** Add a `jax.lax.scan` microbatch loop inside the step
   (`grad_accum` from config) so effective batch decouples from device memory. Divide the CE by
   token count *after* accumulation to keep the masked-CE math exact.

**Validate:** loss curve on 8 simulated CPU devices matches single-device to ~1e-4 for 20 steps.
Then run on real 8×GPU: expect ~linear throughput scaling and lower per-device memory.

---

## Phase 2 — Expert parallelism (`expert` axis)

Goal: place different MoE experts on different devices so per-device expert-weight memory
drops ~`ep×`. This is the natural sharding axis for `GroupedGemmMoE`
([`multi_latent_attention/moe.py`](../multi_latent_attention/moe.py)).

1. **Expert-weight shardings.** `w_in [E, d, 2ff] → P("expert", "data", None)`,
   `w_out [E, ff, d] → P("expert", None, "data")`, `router_bias [E] → P("expert")`. Router
   Linear and shared expert stay FSDP-only (replicated over `expert`).

2. **Dispatch under EP — the crux.** The current forward sorts all `T*k` assignments and calls
   one `jax.lax.ragged_dot` over all `E` experts locally. Under EP, each device holds only
   `E/ep` experts, so tokens must be routed to the device owning their expert. Two options:
   - **(a) GSPMD-first (recommended MVP).** Keep the code, add
     `jax.lax.with_sharding_constraint` on `x_sorted`/`group_sizes` so XLA shards `ragged_dot`
     along the expert axis and inserts the needed all-to-all. Simplest; measure first.
   - **(b) Explicit all-to-all.** Use `jax.lax.ragged_all_to_all` (recent JAX) to send each
     token to its expert's device, run the local grouped GEMM over `E/ep` experts, all-to-all
     back before combine. Faster/more predictable; more code. Adopt only if (a) profiles poorly.

   Add **device-limited (group-limited) routing** (note at bottom of `moe.py`) to bound
   all-to-all traffic: mask `sel` to top expert-groups per token before `top_k`, groups aligned
   to the `expert` axis.

3. **Capacity.** Current impl is drop-free (dynamic `group_sizes`). Ragged all-to-all needs
   ragged send counts; if a static-shape all-to-all is required, introduce an expert-capacity
   factor and pad/drop — keep drop-free (a) as default.

**Validate:** with `ep>1`, MoE output equals the `ep=1` result (dispatch is a permutation —
math is sharding-invariant). Reuse `dense_forward` as the oracle.

---

## Phase 3 — Tensor parallelism (`model` axis)

Goal: shard within-layer hidden/head dims for the largest configs. Most invasive; likely
optional at 4B, but designed in for headroom.

1. **Attention / GDN-2 / FFN.** Shard projection outputs over `model`:
   - MLA/GDN q/k/v/out projections: shard the **head** dimension → `P(None, "model")` on
     `[d_model, n_heads*head_dim]` kernels; the out-projection input matches. Head counts must
     be `% tp == 0` (validated in Phase 0).
   - MoE inner `d_ff`: already sharded over `model` in Phase 2 specs (`w_in` last dim,
     `w_out` middle dim). Shared-expert SwiGLU: `ws_gate/ws_up [d, ish] → P(None,"model")`,
     `ws_down [ish, d] → P("model", None)`.
   - Embedding / LM head: vocab-parallel over `model` (`P("model", None)` / `P(None,"model")`),
     with a `with_sharding_constraint` to gather logits before the fp32 cross-entropy, or a
     sharded-softmax loss to avoid the full-vocab all-gather.

2. **Activation constraints.** Insert `jax.lax.with_sharding_constraint` at block boundaries
   (residual stream `[B, L, d]` replicated over `model`, sharded over `data`) so XLA doesn't
   pick surprising layouts. This is the main tuning surface — do it empirically with the profiler.

3. **GDN-2 chunkwise core.** The chunk recurrence in `core.py` runs per (head, chunk); with
   head-sharding it parallelizes cleanly. Verify no cross-head reduction assumes all heads local.

**Validate:** loss-parity at `tp=2` on simulated devices; the GDN-2 chunkwise test
([`tests/test_triton_chunkwise.py`](../tests/test_triton_chunkwise.py)) still passes under sharding.

---

## Phase 4 — Multi-host (GPU multi-node / TPU pod)

Goal: scale beyond one host. GSPMD code from Phases 1–3 is unchanged; the *environment*,
*data pipeline*, and *checkpoint I/O* change.

1. **Process init.** Call `jax.distributed.initialize()` at the very top of
   [`training/train.py`](../training/train.py) `main()` (before any JAX op; auto-detects
   TPU/Slurm, or takes coordinator addr + process id/count for GPU). Now `jax.devices()` spans
   all hosts and the same mesh spans the pod.

2. **Per-process data pipeline — required.** Each process must feed only its **local** shard of
   the global batch. Update `make_train_iterator` in [`training/data.py`](../training/data.py)
   to take `shard_index=jax.process_index()`, `shard_count=jax.process_count()` (Grain
   `ShardOptions` / index sharding), then assemble the global array with
   `jax.make_array_from_process_local_data(data_sharding, local_batch)`. `ShardedRowSource`
   already pickles cleanly for Grain workers (`__getstate__` drops mmaps) — good.

3. **Checkpointing.** Point `train.out_dir` at a **shared filesystem** (GCS/NFS/Lustre). Orbax
   does distributed sharded read/write natively; ensure `CheckpointManager` is constructed on
   all processes and `wait_until_finished()` is collective. Consider
   `enable_async_checkpointing=True` at pod scale to hide save latency.

4. **Host hygiene.** Log/print and tokenizer-training only on `jax.process_index() == 0`
   (guard `log.info` throughput lines and the `write_shards` tokenization — or pre-build shards
   as a separate one-host job the pod then reads read-only). Barrier
   (`jax.experimental.multihost_utils.sync_global_devices`) after data prep so all hosts start
   together.

5. **TPU specifics.** Guard the Triton path (Phase 0 TPU note); set `compute_dtype: bfloat16`;
   choose mesh from the pod topology (e.g. a v5e-256 slice → `dp*ep*tp = 256`). GDN-2/MLA are
   pure-JAX so no kernel port is needed.

6. **Launch scripts.** Add `scripts/launch_gpu_multinode.sh` (Slurm/`torchrun`-style env:
   `--coordinator_address`, per-node process ids) and `scripts/launch_tpu_pod.sh`
   (`gcloud ... ssh --worker=all` running the same module). Document in README.

**Validate:** a 2-process run on one machine (2 CPU processes, 4 devices each) reproduces the
single-process 8-device loss; then a real 2-node / TPU-v5e-8 smoke run.

---

## Phase 5 — Throughput, correctness, and ergonomics

- **Profiling** — capture `jax.profiler` traces per phase; look for unexpected all-gathers
  (bad TP constraints) and all-to-all imbalance (MoE). Tune `with_sharding_constraint`
  placement and remat (`_rematted_layer`) accordingly; remat + FSDP interact on activation
  memory — re-tune `batch_size`/`seq_len` per config.
- **Grad accumulation** — finalize the scan-based accumulator (Phase 1) and expose
  `parallel.grad_accum`; document effective-batch = `batch_size * grad_accum`.
- **MoE all-to-all** — if GSPMD-first (2a) underperforms, switch to explicit
  `ragged_all_to_all` (2b) + device-limited routing.
- **Config presets** — add `configs/{8xh100,tpu_v5e}_pretrain.yaml` with populated `parallel:`
  blocks; update the README scaling table and delete the "single-device only" caveats.
- **Tests** — extend [`tests/smoke_test.py`](../tests/smoke_test.py) with an 8-simulated-device
  loss-parity test and a sharded save/restore round-trip; run in CI under
  `XLA_FLAGS=--xla_force_host_platform_device_count=8`.

---

## Files touched (summary)

| File | Change |
|---|---|
| `training/config.py` | `ParallelConfig` + divisibility validation |
| `training/parallel.py` *(new)* | mesh + sharding helpers, partition-spec rules |
| `training/utils.py` | fix CPU device-count flag; process-0-only logging |
| `training/train.py` | `jax.distributed.initialize`, sharded init, `device_put`/global-array feed |
| `training/loop.py` | sharded `nnx.jit` in/out shardings; optional grad-accum scan |
| `training/data.py` | per-process Grain sharding; `make_array_from_process_local_data` |
| `training/checkpoint.py` | multi-host/shared-FS + async options; sharded restore test |
| `kimi_linear_gdn2.py`, `multi_latent_attention/*`, `gated_deltanet_2/layer.py` | `with_partitioning` sharding annotations; TP head/ffn/expert/vocab specs |
| `gated_deltanet_2/layer.py` | guard Triton path to CPU/TPU fallback |
| `configs/*`, `README.md`, `scripts/*` | presets, launch scripts, docs |

## Key risks / watch-items

1. **MoE dispatch × sharding** — the `argsort`/`ragged_dot`/scatter-add path is the hardest to
   shard correctly and efficiently (Phase 2). Start with GSPMD + sharding constraints; keep
   `dense_forward` as the correctness oracle. Highest-uncertainty item.
2. **Sharded 4B init OOM** — must use `eval_shape` + `jit(out_shardings=...)`; never realize the
   full model on one device.
3. **Vocab all-gather for the fp32 loss** under TP — either gather logits or implement a
   sharded cross-entropy; the naive path adds a `[B,L,vocab]` all-gather.
4. **Triton kernel portability** — GPU-only; must fall back to `core.py` on TPU/CPU (Phase 0/4).
5. **Donation + sharding** — verify `donate_argnums` still buys the memory win under sharded
   state (it should); watch for donation-alias warnings.

## Suggested order & effort (rough)

Phase 0 (S) → **Phase 1 FSDP (M, highest value)** → Phase 4 multi-host wiring (M, can precede TP)
→ Phase 2 EP (L) → Phase 3 TP (L) → Phase 5 (ongoing). Phases 1 and 4 alone give multi-GPU and
multi-host FSDP — enough for most training runs; EP/TP are for pushing model size further.
