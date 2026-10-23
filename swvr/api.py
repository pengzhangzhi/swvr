"""Public entry points: ``softmax_readout``, ``softmax_readout_mse`` and ``SoftmaxReadoutMSE``."""
import torch
from torch import nn

from .chunked import default_chunk_size, softmax_readout_chunked


def _compute_dtype(h, W):
    if torch.is_autocast_enabled(h.device.type):
        return torch.get_autocast_dtype(h.device.type)
    return torch.promote_types(h.dtype, W.dtype)


def _check(h, W, E, bias):
    if W.dim() != 2 or E.dim() != 2:
        raise ValueError(f"W and E must be 2-D, got {tuple(W.shape)} and {tuple(E.shape)}")
    if h.shape[-1] != W.shape[1]:
        raise ValueError(f"h has hidden size {h.shape[-1]} but W is {tuple(W.shape)}")
    if E.shape[0] != W.shape[0]:
        raise ValueError(f"W has {W.shape[0]} vocabulary rows but E has {E.shape[0]}")
    if bias is not None and bias.shape != (W.shape[0],):
        raise ValueError(f"bias must have shape ({W.shape[0]},), got {tuple(bias.shape)}")


def softmax_readout(h, W, E, bias=None, *, logit_scale=1.0, chunk_size=None):
    """Return ``(O, LSE)`` for ``S = logit_scale * h @ W.T + bias``.

    ``O = softmax(S) @ E`` has shape ``[..., D]`` and ``LSE = logsumexp(S)`` has shape ``[...]``,
    both in fp32 (fp64 for fp64 inputs). ``h`` is ``[..., K]``, ``W`` is ``[V, K]``, ``E`` is
    ``[V, D]`` and ``bias`` is ``[V]``. Gradients flow to every input that requires them.
    The ``[..., V]`` scores are never materialized; ``chunk_size`` (default 8192, or env
    ``SWVR_CHUNK``) vocabulary rows are processed at a time.
    """
    _check(h, W, E, bias)
    lead = h.shape[:-1]
    h2 = h.reshape(-1, h.shape[-1])
    chunk = int(chunk_size) if chunk_size else default_chunk_size()
    if chunk <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk}")
    O, lse = softmax_readout_chunked(h2, W, E, bias, logit_scale, chunk, _compute_dtype(h, W))
    return O.view(*lead, E.shape[1]), lse.view(lead)


def softmax_readout_mse(h, W, E, target, bias=None, *, weight=None, reduction="mean",
                        z_loss_weight=0.0, logit_scale=1.0, chunk_size=None):
    """MSE between ``softmax(S) @ E`` and ``target`` (``[..., D]``), plus an optional z-loss.

    The per-token loss is ``mean_d (O - target)^2 + z_loss_weight * LSE^2``. ``weight`` (shape
    ``[...]``) scales each token's loss, and 0 masks it out. ``reduction="mean"`` divides by
    ``weight.sum()`` when a weight is given, else by the token count. ``"sum"`` and ``"none"``
    are also accepted.
    """
    O, lse = softmax_readout(h, W, E, bias, logit_scale=logit_scale, chunk_size=chunk_size)
    per_tok = (O - target.to(O.dtype)).pow(2).mean(dim=-1)
    if z_loss_weight:
        per_tok = per_tok + z_loss_weight * lse.pow(2)
    if weight is not None:
        per_tok = per_tok * weight.to(per_tok.dtype)
    if reduction == "none":
        return per_tok
    if reduction == "sum":
        return per_tok.sum()
    if reduction == "mean":
        denom = weight.to(per_tok.dtype).sum() if weight is not None else per_tok.numel()
        return per_tok.sum() / denom
    raise ValueError(f"unknown reduction {reduction!r}")


def _weight_of(x):
    return x.weight if isinstance(x, nn.Module) else x


class SoftmaxReadoutMSE(nn.Module):
    """Loss module around an existing LM head and embedding table. It does not copy them.

    ``head`` is an ``nn.Linear(K, V)`` (its bias is used if present) or a ``[V, K]`` tensor;
    ``embedding`` is an ``nn.Embedding(V, D)`` or a ``[V, D]`` tensor. Passing the model's own
    modules keeps weight tying and optimizer state intact.
    """

    def __init__(self, head, embedding, *, z_loss_weight=0.0, logit_scale=1.0,
                 reduction="mean", chunk_size=None):
        super().__init__()
        self.head, self.embedding = head, embedding
        self.z_loss_weight, self.logit_scale = z_loss_weight, logit_scale
        self.reduction, self.chunk_size = reduction, chunk_size

    def forward(self, h, target, weight=None):
        bias = getattr(self.head, "bias", None) if isinstance(self.head, nn.Module) else None
        return softmax_readout_mse(
            h, _weight_of(self.head), _weight_of(self.embedding), target, bias,
            weight=weight, reduction=self.reduction, z_loss_weight=self.z_loss_weight,
            logit_scale=self.logit_scale, chunk_size=self.chunk_size)
