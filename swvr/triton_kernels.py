"""Fused Triton kernels for the elementwise work between the chunked GEMMs.

The large ``h @ W_c^T`` products stay in cuBLAS. These kernels replace the elementwise
torch ops on the fp32 ``[N, C]`` score chunk, and they also absorb the small ``D``-sized
products, so the probabilities ``P`` are never written to memory:

- ``fwd_chunk``: for each row, scale and bias the scores, run the online-softmax update of
  ``(m, l)``, and accumulate ``acc += P @ E_c`` with ``tl.dot``. It reads ``S`` once and
  writes nothing ``[N, C]``-sized.
- ``bwd_chunk``: one program per column tile, looping over all rows. It computes
  ``P = exp(s*S + b - LSE)`` and ``dP = gO @ E_c^T`` in registers, writes
  ``gS = s * P * (dP - delta)`` in the compute dtype for the two cuBLAS gradient GEMMs, and
  accumulates ``gE_c = P^T @ gO`` and ``gb_c = colsum(P * (dP - delta))`` in registers, which
  keeps them deterministic (no atomics).

Used only on CUDA with bf16/fp16 compute and ``D <= 256``; everything else takes the torch path.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

MAX_D = 256


def _dpad(D):
    return max(16, triton.next_power_of_2(D))


if HAVE_TRITON:

    @triton.jit
    def _fwd_chunk_kernel(S, stride_s, Bias, E, stride_e, M, L, Acc, stride_acc,
                          N, C, D, scale,
                          HAS_BIAS: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
                          DP: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < N
        rows64 = rows.to(tl.int64)
        dcols = tl.arange(0, DP)
        dmask = dcols < D
        m = tl.load(M + rows, mask=rmask, other=float("-inf"))
        l = tl.load(L + rows, mask=rmask, other=0.0)
        acc_ptrs = Acc + rows64[:, None] * stride_acc + dcols[None, :]
        amask = rmask[:, None] & dmask[None, :]
        acc = tl.load(acc_ptrs, mask=amask, other=0.0)
        for c0 in range(0, C, BC):
            cols = c0 + tl.arange(0, BC)
            cmask = cols < C
            mask = rmask[:, None] & cmask[None, :]
            x = tl.load(S + rows64[:, None] * stride_s + cols[None, :], mask=mask, other=0.0)
            x = x * scale.to(tl.float32)     # under inductor, scale arrives as fp64
            if HAS_BIAS:
                x += tl.load(Bias + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
            x = tl.where(mask, x, float("-inf"))
            m_new = tl.maximum(m, tl.max(x, axis=1))
            # rows whose scores so far are all -inf keep m = -inf; shift those by 0
            shift = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m - shift)
            p = tl.exp(x - shift[:, None])
            l = l * alpha + tl.sum(p, axis=1)
            e = tl.load(E + cols[:, None].to(tl.int64) * stride_e + dcols[None, :],
                        mask=cmask[:, None] & dmask[None, :], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(e.dtype), e)
            m = m_new
        tl.store(M + rows, m, mask=rmask)
        tl.store(L + rows, l, mask=rmask)
        tl.store(acc_ptrs, acc, mask=amask)

    @triton.jit
    def _bwd_chunk_kernel(S, stride_s, Bias, E, stride_e, LSE, Delta, GO, stride_go,
                          GS, stride_gs, GE, stride_ge, GB, N, C, D, scale,
                          HAS_BIAS: tl.constexpr, NEED_GE: tl.constexpr, NEED_GB: tl.constexpr,
                          BM: tl.constexpr, BC: tl.constexpr, DP: tl.constexpr):
        pid = tl.program_id(0)
        cols = pid * BC + tl.arange(0, BC)
        cmask = cols < C
        cols64 = cols.to(tl.int64)
        dcols = tl.arange(0, DP)
        dmask = dcols < D
        e = tl.load(E + cols64[:, None] * stride_e + dcols[None, :],
                    mask=cmask[:, None] & dmask[None, :], other=0.0)          # [BC, DP]
        if HAS_BIAS:
            b = tl.load(Bias + cols, mask=cmask, other=0.0).to(tl.float32)
        ge = tl.zeros((BC, DP), dtype=tl.float32)
        gb = tl.zeros((BC,), dtype=tl.float32)
        for r0 in range(0, N, BM):
            rows = r0 + tl.arange(0, BM)
            rmask = rows < N
            rows64 = rows.to(tl.int64)
            mask = rmask[:, None] & cmask[None, :]
            x = tl.load(S + rows64[:, None] * stride_s + cols[None, :], mask=mask, other=0.0)
            x = x * scale.to(tl.float32)
            if HAS_BIAS:
                x += b[None, :]
            lse = tl.load(LSE + rows, mask=rmask, other=0.0)
            p = tl.where(mask, tl.exp(x - lse[:, None]), 0.0)
            go = tl.load(GO + rows64[:, None] * stride_go + dcols[None, :],
                         mask=rmask[:, None] & dmask[None, :], other=0.0)     # [BM, DP]
            dp = tl.dot(go, tl.trans(e))                                       # [BM, BC]
            delta = tl.load(Delta + rows, mask=rmask, other=0.0)
            gs = p * (dp - delta[:, None])
            tl.store(GS + rows64[:, None] * stride_gs + cols[None, :],
                     (gs * scale.to(tl.float32)).to(GS.dtype.element_ty), mask=mask)
            if NEED_GB:
                gb += tl.sum(gs, axis=0)
            if NEED_GE:
                ge += tl.dot(tl.trans(p.to(go.dtype)), go)
        if NEED_GE:
            tl.store(GE + cols64[:, None] * stride_ge + dcols[None, :], ge,
                     mask=cmask[:, None] & dmask[None, :])
        if NEED_GB:
            tl.store(GB + cols, gb, mask=cmask)


def usable(h, E, cdt):
    return (HAVE_TRITON and h.is_cuda and cdt in (torch.bfloat16, torch.float16)
            and E.shape[1] <= MAX_D and h.shape[0] > 0)


def fwd_chunk(S, bias_c, scale, E_c, m, l, acc):
    """In place: fold the score chunk ``S`` [N, C] (fp32) into ``(m, l, acc)``."""
    N, C = S.shape
    D = E_c.shape[1]
    DP = _dpad(D)
    BM, BC = (64, 64) if DP <= 64 else (32, 64)
    grid = (triton.cdiv(N, BM),)
    _fwd_chunk_kernel[grid](
        S, S.stride(0), bias_c if bias_c is not None else S, E_c, E_c.stride(0),
        m, l, acc, acc.stride(0), N, C, D, float(scale),
        HAS_BIAS=bias_c is not None, BM=BM, BC=BC, DP=DP, num_warps=4)


def bwd_chunk(S, bias_c, scale, E_c, lse, delta, gOc, need_gE, need_gb):
    """Return ``(gS [N, C] in gOc.dtype, gE_c [C, D] fp32 or None, gb_c [C] fp32 or None)``.

    ``gS`` already includes ``scale``, so it is ``dL/d(h W_c^T)``. ``gb_c`` does not.
    """
    N, C = S.shape
    D = E_c.shape[1]
    DP = _dpad(D)
    gS = torch.empty(N, C, dtype=gOc.dtype, device=S.device)
    gE = torch.empty(C, D, dtype=torch.float32, device=S.device) if need_gE else None
    gb = torch.empty(C, dtype=torch.float32, device=S.device) if need_gb else None
    BM, BC = 64, 32
    grid = (triton.cdiv(C, BC),)
    _bwd_chunk_kernel[grid](
        S, S.stride(0), bias_c if bias_c is not None else S, E_c, E_c.stride(0),
        lse, delta, gOc, gOc.stride(0), gS, gS.stride(0),
        gE if need_gE else gS, D if need_gE else 0, gb if need_gb else gS,
        N, C, D, float(scale),
        HAS_BIAS=bias_c is not None, NEED_GE=need_gE, NEED_GB=need_gb,
        BM=BM, BC=BC, DP=DP, num_warps=4)
    return gS, gE, gb
