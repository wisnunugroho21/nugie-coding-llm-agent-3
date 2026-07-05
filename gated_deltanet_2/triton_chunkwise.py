"""
Gated DeltaNet-2 — chunkwise parallel forward as fused Triton kernels (PyTorch).

GPU twin of `gated_deltanet_2.core._chunkwise_single` (JAX): same equations of
Hatamizadeh, Choi, Kautz, "Gated DeltaNet-2: Decoupling Erase and Write in
Linear Attention" (arXiv:2605.22791), same fp32 math, same log-decay floor —
the two paths are numerically interchangeable (parity is checked in
tests/test_triton_chunkwise.py against a token-by-token recurrent reference).

Two-kernel decomposition of the WY-form chunkwise algorithm (Eqs. 18-25 / 30-44):

  Kernel 1 `_gdn2_prepare_kernel` — embarrassingly parallel over
  (chunk, batch*head). Per chunk: G = floored cumsum(g) (Eq. 18/30),
  K_bar / E_bar / Z (Eqs. 19-20 / 32-33), T = tril(E_bar K_bar^T, -1) and
  A = (I+T)^{-1} by in-register forward substitution (Eq. 21/34), then the WY
  auxiliaries Y = A E_bar, U = A Z (Eq. 22/34). G, Y, U are staged to HBM for
  the second pass.

  Kernel 2 `_gdn2_scan_kernel` — parallel over (dv-blocks, batch*head),
  sequential over chunks (the cross-chunk recurrence). Keeps the running state
  block S[dk, BV] in registers across the whole sequence:
      R = U - Y S_0                          (Eq. 35)
      O = Q_gamma S_0 + A_qk R               (Eq. 24/44)
      S <- Diag(gamma_C) S_0 + K_tail^T R    (Eq. 23/40)
  The dv axis is blocked (state columns are independent); the dk axis is NOT
  blocked, because Y S_0 contracts over all of dk.

All tl.dot calls disable TF32 so fp32 results track the reference bit-for-bit
up to summation order.

Forward only. For training keep using the JAX path (core.py), whose backward
comes from jax.grad; the hand-derived gate-aware backward (paper Appendix B,
Eqs. 64-82) is the natural follow-up for these kernels.

Requires torch + triton on a CUDA GPU. Neither is in requirements.txt — the
training pipeline is JAX; install separately (`pip install torch triton`).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Must stay equal to core._LOG_DECAY_FLOOR (duplicated so this module does not
# import JAX). See core.py for the full derivation: G is floored so that
# exp(-G) in K_bar cannot overflow fp32 under strong trained decay.
_LOG_DECAY_FLOOR = -30.0


# --------------------------------------------------------------------------- #
#  Kernel 1: per-chunk WY preparation (parallel over chunks)
# --------------------------------------------------------------------------- #
@triton.jit
def _gdn2_prepare_kernel(
    k_ptr, v_ptr, g_ptr, b_ptr, w_ptr,  # inputs, [B*H, L, dk|dv]
    G_ptr, Y_ptr, U_ptr,  # staged outputs (fp32)
    L, dk, dv,
    C: tl.constexpr,  # chunk size
    BK: tl.constexpr,  # padded key dim (pow2 >= dk)
    BV: tl.constexpr,  # padded value dim (pow2 >= dv)
    LOG_DECAY_FLOOR: tl.constexpr,
):
    i_n = tl.program_id(0)  # chunk index
    i_bh = tl.program_id(1).to(tl.int64)  # flattened (batch, head)

    o_c = tl.arange(0, C)
    o_t = (i_n * C + o_c).to(tl.int64)  # token rows of this chunk
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_k = (o_k < dk)[None, :]
    m_v = (o_v < dv)[None, :]

    idx_k = i_bh * L * dk + o_t[:, None] * dk + o_k[None, :]
    idx_v = i_bh * L * dv + o_t[:, None] * dv + o_v[None, :]

    b_k = tl.load(k_ptr + idx_k, mask=m_k, other=0.0).to(tl.float32)
    b_g = tl.load(g_ptr + idx_k, mask=m_k, other=0.0).to(tl.float32)
    b_b = tl.load(b_ptr + idx_k, mask=m_k, other=0.0).to(tl.float32)
    b_v = tl.load(v_ptr + idx_v, mask=m_v, other=0.0).to(tl.float32)
    b_w = tl.load(w_ptr + idx_v, mask=m_v, other=0.0).to(tl.float32)

    # Eq. 18/30: G_r = cumsum(g) (inclusive, resets each chunk) + overflow floor
    b_G = tl.maximum(tl.cumsum(b_g, axis=0), LOG_DECAY_FLOOR)
    tl.store(G_ptr + idx_k, b_G, mask=m_k)

    b_kbar = b_k * tl.exp(-b_G)  # Eq. 19/32: K_bar = gamma^{-1} . K
    b_ebar = tl.exp(b_G) * (b_b * b_k)  # Eq. 20/33: E_bar = gamma . (B . K)
    b_z = b_w * b_v  # Eq. 20/33: Z = W . V

    # Eq. 21/34: T = tril(E_bar K_bar^T, -1)
    b_T = tl.dot(b_ebar, tl.trans(b_kbar), allow_tf32=False)

    # A = (I+T)^{-1}. With M = -T strictly lower, A = I + N where the strictly
    # lower N solves N = M + M N, computed row by row (forward substitution):
    # rows j < i already hold N_j when row i is formed.
    b_A = tl.where(o_c[:, None] > o_c[None, :], -b_T, 0.0)
    for i in range(1, C):
        m_i = o_c == i
        b_a = tl.sum(tl.where(m_i[:, None], b_A, 0.0), 0)  # M_i (row i, unmodified)
        b_a = b_a + tl.where(o_c < i, tl.sum(b_a[:, None] * b_A, 0), 0.0)  # + M_i N
        b_A = tl.where(m_i[:, None], b_a, b_A)
    b_A = b_A + tl.where(o_c[:, None] == o_c[None, :], 1.0, 0.0)

    # Eq. 22/34: WY auxiliaries, same inverse, two right-hand sides
    tl.store(Y_ptr + idx_k, tl.dot(b_A, b_ebar, allow_tf32=False), mask=m_k)
    tl.store(U_ptr + idx_v, tl.dot(b_A, b_z, allow_tf32=False), mask=m_v)


# --------------------------------------------------------------------------- #
#  Kernel 2: cross-chunk scan — state recurrence + outputs (sequential in n)
# --------------------------------------------------------------------------- #
@triton.jit
def _gdn2_scan_kernel(
    q_ptr, k_ptr,  # inputs, [B*H, L, dk]
    G_ptr, Y_ptr, U_ptr,  # staged fp32 tensors from kernel 1
    S0_ptr,  # initial state, [B*H, dk, dv]
    O_ptr, SF_ptr,  # outputs: O [B*H, L, dv], final state [B*H, dk, dv] (fp32)
    L, dk, dv, N,  # N = number of chunks
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,  # dv block width (dv may span several programs)
):
    i_v = tl.program_id(0)  # dv block
    i_bh = tl.program_id(1).to(tl.int64)

    o_c = tl.arange(0, C)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = (o_k < dk)[None, :]
    m_v = (o_v < dv)[None, :]

    # Running state block S[dk, BV], held in registers across all chunks.
    idx_S = i_bh * dk * dv + o_k[:, None].to(tl.int64) * dv + o_v[None, :]
    m_S = (o_k < dk)[:, None] & (o_v < dv)[None, :]
    b_S = tl.load(S0_ptr + idx_S, mask=m_S, other=0.0).to(tl.float32)

    for i_n in range(0, N):
        o_t = (i_n * C + o_c).to(tl.int64)
        idx_k = i_bh * L * dk + o_t[:, None] * dk + o_k[None, :]
        idx_v = i_bh * L * dv + o_t[:, None] * dv + o_v[None, :]

        b_q = tl.load(q_ptr + idx_k, mask=m_k, other=0.0).to(tl.float32)
        b_k = tl.load(k_ptr + idx_k, mask=m_k, other=0.0).to(tl.float32)
        b_G = tl.load(G_ptr + idx_k, mask=m_k, other=0.0)
        b_Y = tl.load(Y_ptr + idx_k, mask=m_k, other=0.0)
        b_U = tl.load(U_ptr + idx_v, mask=m_v, other=0.0)

        # Eq. 35: R = U - Y S_0 (S_0 = raw state at chunk entry)
        b_R = b_U - tl.dot(b_Y, b_S, allow_tf32=False)

        # Eq. 24/43-44: O = Q_gamma S_0 + tril(Q_gamma K_bar^T) R
        b_qg = b_q * tl.exp(b_G)
        b_kbar = b_k * tl.exp(-b_G)
        b_Aqk = tl.dot(b_qg, tl.trans(b_kbar), allow_tf32=False)
        b_Aqk = tl.where(o_c[:, None] >= o_c[None, :], b_Aqk, 0.0)
        b_o = tl.dot(b_qg, b_S, allow_tf32=False) + tl.dot(
            b_Aqk, b_R, allow_tf32=False
        )
        tl.store(O_ptr + idx_v, b_o, mask=m_v)

        # Eq. 23/40-41: S <- Diag(gamma_C) S_0 + K_tail^T R,
        # gamma_C = exp(G[C-1]), K_tail = (gamma_C / gamma) . K = exp(G_C - G) . K
        b_gC = tl.sum(tl.where((o_c == C - 1)[:, None], b_G, 0.0), 0)  # [BK]
        b_ktail = b_k * tl.exp(b_gC[None, :] - b_G)
        b_S = tl.exp(b_gC)[:, None] * b_S + tl.dot(
            tl.trans(b_ktail), b_R, allow_tf32=False
        )

    tl.store(SF_ptr + idx_S, b_S, mask=m_S)


# --------------------------------------------------------------------------- #
#  Public entry point — same signature/semantics as core.chunkwise_gated_delta_rule_2
# --------------------------------------------------------------------------- #
def chunkwise_gated_delta_rule_2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    S0: torch.Tensor,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Triton chunkwise forward (inference / no-grad evaluation).

    q, k, g, b : [B, H, L, dk]      v, w : [B, H, L, dv]      S0 : [B, H, dk, dv]
    returns (O : [B, H, L, dv], S_final : [B, H, dk, dv]), both fp32
    (the math runs in fp32 regardless of input dtype, as in core.py).
    """
    B, H, L, dk = q.shape
    dv = v.shape[-1]
    C = chunk_size
    if L % C != 0:
        raise ValueError(f"L={L} must be a multiple of chunk_size={C}")
    if C not in (16, 32, 64):
        raise ValueError(f"chunk_size must be 16, 32 or 64, got {C}")
    if dk > 256 or dv > 256:
        raise ValueError(f"head dims up to 256 supported, got dk={dk}, dv={dv}")
    if not q.is_cuda:
        raise RuntimeError("Triton path needs CUDA tensors; use core.py on CPU/TPU")
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (q, k, v, g, b, w, S0)
    ):
        raise RuntimeError(
            "forward-only kernel: no backward is registered — train via the JAX "
            "path (gated_deltanet_2.core) or call under torch.no_grad()"
        )

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    g, b, w = g.contiguous(), b.contiguous(), w.contiguous()
    S0 = S0.contiguous()

    N = L // C
    BK = max(16, triton.next_power_of_2(dk))
    BV = max(16, triton.next_power_of_2(dv))
    f32 = dict(device=q.device, dtype=torch.float32)

    G = torch.empty(B, H, L, dk, **f32)
    Y = torch.empty(B, H, L, dk, **f32)
    U = torch.empty(B, H, L, dv, **f32)
    _gdn2_prepare_kernel[(N, B * H)](
        k, v, g, b, w, G, Y, U,
        L, dk, dv,
        C=C, BK=BK, BV=BV, LOG_DECAY_FLOOR=_LOG_DECAY_FLOOR,
        num_warps=4,
    )

    O = torch.empty(B, H, L, dv, **f32)
    SF = torch.empty(B, H, dk, dv, **f32)
    # Cap the state block held in registers at [dk, 64] per program.
    BVs = min(BV, 64)
    _gdn2_scan_kernel[(triton.cdiv(dv, BVs), B * H)](
        q, k, G, Y, U, S0, O, SF,
        L, dk, dv, N,
        C=C, BK=BK, BV=BVs,
        num_warps=4,
    )
    return O, SF
