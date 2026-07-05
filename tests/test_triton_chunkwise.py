"""Parity test for the Triton chunkwise Gated DeltaNet-2 forward.

Checks gated_deltanet_2.triton_chunkwise against a token-by-token recurrent
reference (the same ground truth core.py is tested against in smoke_test.py),
plus the extreme-decay stability regression.

Needs torch + triton + a CUDA GPU (e.g. the Colab runtime); exits cleanly with
a skip message otherwise:

    python -m tests.test_triton_chunkwise
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def recurrent_reference(q, k, v, g, b, w, S0):
    """Token-by-token Eq. 9 recurrence in fp32 torch (mirrors core._recurrent_single)."""
    import torch

    q, k, v, g, b, w, S0 = (t.float() for t in (q, k, v, g, b, w, S0))
    B, H, L, _ = q.shape
    alpha = g.exp()  # Eq. 12/30
    e = b * k  # Eq. 8
    z = w * v  # Eq. 8
    S = S0.clone()
    o = torch.empty(B, H, L, v.shape[-1], dtype=torch.float32, device=q.device)
    for t in range(L):
        S = alpha[:, :, t, :, None] * S  # S_bar = Diag(alpha) S
        r = torch.einsum("bhkv,bhk->bhv", S, e[:, :, t])  # r = S_bar^T e
        S = S + k[:, :, t, :, None] * (z[:, :, t] - r)[:, :, None, :]
        o[:, :, t] = torch.einsum("bhkv,bhk->bhv", S, q[:, :, t])  # o = S^T q
    return o, S


def make_inputs(torch, gen, B, H, L, dk, dv, decay_scale, s0_scale, dtype):
    def randn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)

    def unit(x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-6)

    q = unit(randn(B, H, L, dk))
    k = unit(randn(B, H, L, dk))
    v = randn(B, H, L, dv)
    g = -decay_scale * torch.nn.functional.softplus(randn(B, H, L, dk))
    b = torch.sigmoid(randn(B, H, L, dk))
    w = torch.sigmoid(randn(B, H, L, dv))
    S0 = s0_scale * randn(B, H, dk, dv)
    return tuple(t.to(dtype) for t in (q, k, v, g, b, w)) + (S0.to(dtype),)


def main():
    try:
        import torch
        import triton  # noqa: F401
    except ImportError as exc:
        print(f"[triton-test] SKIP: {exc}")
        return
    if not torch.cuda.is_available():
        print("[triton-test] SKIP: no CUDA device")
        return

    from gated_deltanet_2.triton_chunkwise import chunkwise_gated_delta_rule_2

    gen = torch.Generator(device="cuda").manual_seed(7)
    B, H, L = 2, 3, 256

    # 1. parity vs the recurrent ground truth, mild decay (floor inert),
    #    across chunk sizes, square/rect/non-pow2 head dims, and S0 != 0.
    for dk, dv, C in [(16, 16, 64), (64, 128, 64), (32, 48, 32), (128, 64, 16)]:
        inp = make_inputs(torch, gen, B, H, L, dk, dv, 0.05, 0.5, torch.float32)
        o_t, s_t = chunkwise_gated_delta_rule_2(*inp, chunk_size=C)
        o_r, s_r = recurrent_reference(*inp)
        d_o = (o_t - o_r).abs().max().item()
        d_s = (s_t - s_r).abs().max().item()
        assert d_o < 1e-4 and d_s < 1e-4, f"dk={dk} dv={dv} C={C}: O {d_o} S {d_s}"
        print(f"[triton-test] fp32 dk={dk:3d} dv={dv:3d} C={C:2d}: O diff {d_o:.1e}, S diff {d_s:.1e}")

    # 2. bf16 inputs (math still fp32 inside the kernel) — looser tolerance.
    inp = make_inputs(torch, gen, B, H, L, 16, 16, 0.05, 0.5, torch.bfloat16)
    o_t, s_t = chunkwise_gated_delta_rule_2(*inp, chunk_size=64)
    o_r, s_r = recurrent_reference(*inp)
    d_o = (o_t - o_r).abs().max().item()
    assert d_o < 1e-4, f"bf16 inputs: O diff {d_o}"
    print(f"[triton-test] bf16 inputs OK (O diff {d_o:.1e})")

    # 3. extreme decay: the log-decay floor must keep everything finite
    #    (same regression as smoke_test.py step 9).
    inp = make_inputs(torch, gen, B, H, L, 16, 16, 4.0, 0.5, torch.float32)
    o_t, s_t = chunkwise_gated_delta_rule_2(*inp, chunk_size=64)
    assert o_t.isfinite().all() and s_t.isfinite().all(), "non-finite under extreme decay"
    print("[triton-test] extreme-decay stability OK")

    print("\n[triton-test] ALL CHECKS PASSED ✓")


if __name__ == "__main__":
    main()
