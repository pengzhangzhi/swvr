"""Eager reference: materializes the ``[N, V]`` scores. A test oracle for small shapes."""
import torch


def softmax_readout_reference(h, W, E, bias=None, logit_scale=1.0):
    """Same contract as ``swvr.softmax_readout``, computed eagerly in at least fp32."""
    dtype = torch.float32
    for t in (h, W, E, bias):
        if t is not None:
            dtype = torch.promote_types(dtype, t.dtype)
    S = logit_scale * (h.to(dtype) @ W.to(dtype).T)
    if bias is not None:
        S = S + bias.to(dtype)
    return torch.softmax(S, dim=-1) @ E.to(dtype), torch.logsumexp(S, dim=-1)
