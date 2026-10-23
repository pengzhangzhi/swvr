"""Vocabulary-chunked online softmax with a recomputing backward.

The vocabulary is processed ``chunk`` rows at a time, so the largest live intermediates are
``[N, chunk]``. The backward saves only ``O`` and ``LSE`` and recomputes each score chunk.
See ``docs/DESIGN.md`` for the math.

Precision: matrix products take operands in the compute dtype ``cdt`` (the autocast dtype, or
the promoted input dtype) and accumulate *and output* in ``adt`` (fp32, or fp64 for fp64
inputs), so scores are never rounded to bf16. The softmax state, ``O`` and ``LSE`` are ``adt``.

On CUDA with bf16/fp16 compute, the elementwise work between the GEMMs runs in the fused
kernels of ``triton_kernels``; elsewhere (including CPU) plain torch ops are used.
``SWVR_DISABLE_TRITON=1`` forces the torch path.
"""
import os
from typing import List, Optional, Tuple

import torch

from . import triton_kernels

DEFAULT_CHUNK = 8192
_has_mm_out_dtype = None


def acc_dtype(cdt):
    return torch.float64 if cdt == torch.float64 else torch.float32


def mm_acc(a, b):
    """``a @ b`` with a high-precision result: never rounded to bf16/fp16.

    Uses ``torch.mm(..., out_dtype=torch.float32)`` on CUDA when torch has it. Otherwise it
    upcasts to fp32 and runs a TF32 GEMM, which is still exact on bf16 operands (TF32 holds
    them exactly) but about 2x slower.
    """
    global _has_mm_out_dtype
    if a.dtype not in (torch.bfloat16, torch.float16):
        return torch.mm(a, b)
    if a.is_cuda:
        if _has_mm_out_dtype is None:
            try:
                torch.mm(a[:1, :1], b[:1, :1], out_dtype=torch.float32)
                _has_mm_out_dtype = True
            except (TypeError, RuntimeError):
                _has_mm_out_dtype = False
        if _has_mm_out_dtype:
            return torch.mm(a, b, out_dtype=torch.float32)
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            return torch.mm(a.float(), b.float())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
    return torch.mm(a.float(), b.float())

def default_chunk_size():
    raw = os.environ.get("SWVR_CHUNK")
    return int(raw) if raw else DEFAULT_CHUNK


def _use_triton(h, E, cdt):
    return os.environ.get("SWVR_DISABLE_TRITON", "0") in ("", "0") and triton_kernels.usable(h, E, cdt)


def _rows(x, v0, v1, dtype):
    """``x[v0:v1]`` cast to ``dtype``, with unit stride in the last dim (Triton needs it)."""
    x = x[v0:v1].to(dtype)
    return x if x.stride(-1) == 1 else x.contiguous()


def _scores(hc, W, bias, v0, v1, s, cdt, adt):
    """``s * hc @ W[v0:v1].T + bias[v0:v1]`` as an ``adt`` tensor ``[N, v1 - v0]``."""
    S = mm_acc(hc, W[v0:v1].to(cdt).T)
    if s != 1.0:
        S.mul_(s)
    if bias is not None:
        S.add_(bias[v0:v1].to(adt))
    return S


def _forward(h, W, E, bias, logit_scale, chunk, cdt, use_triton):
    adt = acc_dtype(cdt)
    N, V, D = h.shape[0], W.shape[0], E.shape[1]
    hc = h.to(cdt)
    m = torch.full((N,), float("-inf"), dtype=adt, device=h.device)
    l = torch.zeros(N, dtype=adt, device=h.device)
    acc = torch.zeros(N, D, dtype=adt, device=h.device)
    for v0 in range(0, V, chunk):
        v1 = min(v0 + chunk, V)
        if use_triton:
            S = mm_acc(hc, W[v0:v1].to(cdt).T)
            b = _rows(bias, v0, v1, bias.dtype) if bias is not None else None
            triton_kernels.fwd_chunk(S, b, logit_scale, _rows(E, v0, v1, cdt), m, l, acc)
            continue
        S = _scores(hc, W, bias, v0, v1, logit_scale, cdt, adt)
        m_new = torch.maximum(m, S.amax(dim=1))
        # A row whose scores so far are all -inf keeps m = -inf; shift it by 0 instead.
        shift = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
        alpha = torch.exp(m - shift)
        P = S.sub_(shift[:, None]).exp_()                          # in place: S is dead now
        l.mul_(alpha).add_(P.sum(dim=1))
        acc.mul_(alpha[:, None]).add_(mm_acc(P.to(cdt), E[v0:v1].to(cdt)))
        m = m_new
    return acc.div_(l[:, None]), m + torch.log(l)


def _backward(gO, glse, h, W, E, bias, O, lse, s, chunk, cdt, use_triton,
              need_h, need_W, need_E, need_b):
    adt = O.dtype
    N, V = h.shape[0], W.shape[0]
    hc = h.to(cdt)
    delta = (O * gO.to(adt)).sum(dim=1)                            # [N]
    delta.sub_(glse.to(adt))                                       # fold dLSE into the shift
    gOc = gO.to(cdt).contiguous()
    gH = torch.zeros(N, h.shape[1], dtype=adt, device=h.device) if need_h else None
    gW = torch.empty_like(W) if need_W else None
    gE = torch.empty_like(E) if need_E else None
    gb = torch.empty_like(bias) if need_b else None
    for v0 in range(0, V, chunk):
        v1 = min(v0 + chunk, V)
        if use_triton:
            S = mm_acc(hc, W[v0:v1].to(cdt).T)
            b = _rows(bias, v0, v1, bias.dtype) if bias is not None else None
            gS, gEc, gbc = triton_kernels.bwd_chunk(S, b, s, _rows(E, v0, v1, cdt), lse,
                                                 delta, gOc, need_E, need_b)
            del S
            if need_E:
                gE[v0:v1] = gEc
            if need_b:
                gb[v0:v1] = gbc
        else:
            P = _scores(hc, W, bias, v0, v1, s, cdt, adt).sub_(lse[:, None]).exp_()
            # dL/dS = P * (gO @ E^T - delta + dLSE)
            gS = mm_acc(gOc, E[v0:v1].to(cdt).T).sub_(delta[:, None]).mul_(P)
            if need_b:
                gb[v0:v1] = gS.sum(dim=0)
            if need_E:
                gE[v0:v1] = mm_acc(P.to(cdt).T, gOc)
            del P
            if not (need_h or need_W):
                continue
            if s != 1.0:
                gS.mul_(s)                                         # dL/d(h W^T)
            gS = gS.to(cdt)
        Wc = W[v0:v1].to(cdt)
        if need_h:
            gH.add_(mm_acc(gS, Wc))
        if need_W:
            gW[v0:v1] = mm_acc(gS.T, hc)
    return gH.to(h.dtype) if need_h else None, gW, gE, gb


# The forward and backward are registered as opaque custom ops so torch.compile cannot see
# into, and so cannot reorder, the chunk loop: that loop's order is what bounds peak memory.

@torch.library.custom_op("swvr::chunked_fwd", mutates_args=())
def _fwd_op(h: torch.Tensor, W: torch.Tensor, E: torch.Tensor, bias: Optional[torch.Tensor],
            logit_scale: float, chunk: int, cdt: torch.dtype,
            use_triton: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    return _forward(h, W, E, bias, logit_scale, chunk, cdt, use_triton)


@_fwd_op.register_fake
def _(h, W, E, bias, logit_scale, chunk, cdt, use_triton):
    adt = acc_dtype(cdt)
    return h.new_empty(h.shape[0], E.shape[1], dtype=adt), h.new_empty(h.shape[0], dtype=adt)


@torch.library.custom_op("swvr::chunked_bwd", mutates_args=())
def _bwd_op(gO: torch.Tensor, glse: torch.Tensor, h: torch.Tensor, W: torch.Tensor,
            E: torch.Tensor, bias: Optional[torch.Tensor], O: torch.Tensor, lse: torch.Tensor,
            logit_scale: float, chunk: int, cdt: torch.dtype, use_triton: bool,
            need_h: bool, need_W: bool, need_E: bool, need_b: bool) -> List[torch.Tensor]:
    grads = _backward(gO, glse, h, W, E, bias, O, lse, logit_scale, chunk, cdt, use_triton,
                      need_h, need_W, need_E, need_b)
    # custom ops cannot return None; an empty tensor stands for "no gradient"
    return [g if g is not None else h.new_empty(0) for g in grads]


@_bwd_op.register_fake
def _(gO, glse, h, W, E, bias, O, lse, logit_scale, chunk, cdt, use_triton,
      need_h, need_W, need_E, need_b):
    like = [(need_h, h), (need_W, W), (need_E, E), (need_b, bias)]
    return [torch.empty_like(x) if need else h.new_empty(0) for need, x in like]


def _setup_context(ctx, inputs, output):
    h, W, E, bias, logit_scale, chunk, cdt, use_triton = inputs
    O, lse = output
    ctx.save_for_backward(h, W, E, bias, O, lse)
    ctx.args = (logit_scale, chunk, cdt, use_triton)


def _autograd_backward(ctx, gO, glse):
    h, W, E, bias, O, lse = ctx.saved_tensors
    needs = [bool(n) for n in ctx.needs_input_grad[:4]]
    needs[3] = needs[3] and bias is not None
    gO = gO if gO is not None else torch.zeros_like(O)
    glse = glse if glse is not None else torch.zeros_like(lse)
    grads = _bwd_op(gO, glse, h, W, E, bias, O, lse, *ctx.args, *needs)
    return (*[g if need else None for g, need in zip(grads, needs)], None, None, None, None)


_fwd_op.register_autograd(_autograd_backward, setup_context=_setup_context)


def softmax_readout_chunked(h2, W, E, bias, logit_scale, chunk, cdt):
    """``h2`` is ``[N, K]``; returns ``(O [N, D], LSE [N])`` in the accumulation dtype."""
    with torch.autocast(h2.device.type, enabled=False):
        return _fwd_op(h2, W, E, bias, float(logit_scale), int(chunk), cdt,
                       _use_triton(h2, E, cdt))
