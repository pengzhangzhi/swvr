import pytest
import torch
from torch import nn

import swvr

f64 = torch.float64


def make(N=6, K=12, V=41, D=5, dtype=f64, bias=True, lead=None, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    shape = (*(lead or (N,)), K)
    t = lambda *s: torch.randn(*s, generator=g, dtype=f64).to(device=device, dtype=dtype)
    h, W, E = t(*shape), t(V, K), t(V, D)
    b = t(V) if bias else None
    for x in (h, W, E, b):
        if x is not None:
            x.requires_grad_(True)
    return h, W, E, b


def grads(fn, inputs, seed=1):
    """Gradients of a fixed random linear functional of fn's outputs w.r.t. inputs."""
    outs = fn(*inputs)
    g = torch.Generator().manual_seed(seed)
    loss = sum((o * torch.randn(o.shape, generator=g, dtype=f64).to(o)).sum() for o in outs)
    xs = [x for x in inputs if x is not None and x.requires_grad]
    return outs, torch.autograd.grad(loss, xs, allow_unused=True)


def assert_matches_reference(h, W, E, b, scale=1.0, chunk=8, **tol):
    fast = lambda *a: swvr.softmax_readout(*a, logit_scale=scale, chunk_size=chunk)
    ref = lambda *a: swvr.softmax_readout_reference(*a, logit_scale=scale)
    (o1, g1), (o2, g2) = grads(fast, (h, W, E, b)), grads(ref, (h, W, E, b))
    for a, r in zip(o1, o2):
        torch.testing.assert_close(a, r.to(a.dtype), **tol)
    for a, r in zip(g1, g2):
        torch.testing.assert_close(a, r, **tol)


@pytest.mark.parametrize("chunk", [1, 3, 8, 40, 41, 42, 1000])
def test_chunk_sizes_match_reference(chunk):
    assert_matches_reference(*make(), chunk=chunk)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("scale", [1.0, 0.3, 2.5])
def test_gradcheck(bias, scale):
    h, W, E, b = make(N=4, K=6, V=13, D=3, bias=bias)
    fn = lambda *a: swvr.softmax_readout(*a, logit_scale=scale, chunk_size=5)
    assert torch.autograd.gradcheck(fn, (h, W, E, b))


def test_leading_dims():
    h, W, E, b = make(lead=(2, 3, 4))
    O, lse = swvr.softmax_readout(h, W, E, b, chunk_size=7)
    assert O.shape == (2, 3, 4, 5) and lse.shape == (2, 3, 4)
    assert_matches_reference(h, W, E, b)


def test_non_contiguous_h():
    h, W, E, b = make(N=8)
    h_t = h.detach().t().contiguous().t()[::2].requires_grad_(True)  # strided view
    assert not h_t.is_contiguous()
    assert_matches_reference(h_t, W, E, b)


def test_empty_batch():
    h, W, E, b = make(N=0)
    O, lse = swvr.softmax_readout(h, W, E, b, chunk_size=8)
    assert O.shape == (0, 5) and lse.shape == (0,)
    (O.sum() + lse.sum()).backward()
    assert W.grad.abs().sum() == 0


def test_single_token_and_tiny_vocab():
    assert_matches_reference(*make(N=1, V=1), chunk=8)
    assert_matches_reference(*make(N=1, V=2), chunk=1)


def test_large_logits_no_overflow():
    h, W, E, b = make()
    with torch.no_grad():
        b[3] = 1e4      # one dominant token
        b[7] = -1e4
    assert_matches_reference(h, W, E, b)
    O, _ = swvr.softmax_readout(h, W, E, b, chunk_size=8)
    torch.testing.assert_close(O, E[3].expand_as(O).detach())


def test_masked_vocab_with_neg_inf_bias():
    h, W, E, b = make(V=40)
    with torch.no_grad():
        b[:16] = float("-inf")     # two full chunks of masked vocabulary come first
        b[30:] = float("-inf")
    O, lse = swvr.softmax_readout(h, W, E, b, chunk_size=8)
    O_r, lse_r = swvr.softmax_readout_reference(h, W, E, b)
    torch.testing.assert_close(O, O_r)
    torch.testing.assert_close(lse, lse_r)
    (o, gs), (_, gs_r) = grads(lambda *a: swvr.softmax_readout(*a, chunk_size=8), (h, W, E, b)), \
        grads(lambda *a: swvr.softmax_readout_reference(*a), (h, W, E, b))
    for a, r in zip(gs[:3], gs_r[:3]):
        torch.testing.assert_close(a, r)
    # bias gradient at -inf entries: the reference produces 0 there too
    torch.testing.assert_close(gs[3], gs_r[3])


@pytest.mark.parametrize("frozen", ["h", "W", "E", "b", "WE"])
def test_frozen_inputs(frozen):
    h, W, E, b = make()
    names = dict(h=h, W=W, E=E, b=b)
    for k in frozen:
        names[k].requires_grad_(False)
    assert_matches_reference(h, W, E, b)


def test_tied_W_and_E():
    h, W, _, b = make(K=7, D=7)
    assert_matches_reference(h, W, W, b)


def test_only_lse_or_only_O_used():
    h, W, E, b = make()
    for pick in (0, 1):
        out = swvr.softmax_readout(h, W, E, b, chunk_size=8)[pick].sum()
        ref = swvr.softmax_readout_reference(h, W, E, b)[pick].sum()
        g = torch.autograd.grad(out, (h, W, E, b), allow_unused=True)
        gr = torch.autograd.grad(ref, (h, W, E, b), allow_unused=True)
        for a, r in zip(g, gr):
            if r is None:
                assert a is None or a.abs().max() == 0
            else:
                torch.testing.assert_close(a, r)


def test_fp32_inputs():
    h, W, E, b = make(dtype=torch.float32)
    assert_matches_reference(h, W, E, b, rtol=1e-4, atol=1e-5)


def test_bf16_inputs_cpu():
    h, W, E, b = make(dtype=torch.bfloat16, K=32, V=300)
    O, lse = swvr.softmax_readout(h, W, E, b, chunk_size=64)
    assert O.dtype == torch.float32 and lse.dtype == torch.float32
    O_r, lse_r = swvr.softmax_readout_reference(*(x.detach().double() for x in (h, W, E, b)))
    torch.testing.assert_close(lse.double(), lse_r, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(O.double(), O_r, rtol=2e-2, atol=2e-2)


def mse_reference(h, W, E, target, b, weight=None, reduction="mean", z_loss_weight=0.0,
                  logit_scale=1.0):
    O, lse = swvr.softmax_readout_reference(h, W, E, b, logit_scale)
    per_tok = (O - target).pow(2).mean(-1) + z_loss_weight * lse.pow(2)
    if weight is not None:
        per_tok = per_tok * weight
    if reduction == "none":
        return per_tok
    return per_tok.sum() / (1 if reduction == "sum" else
                            weight.sum() if weight is not None else per_tok.numel())


def test_reference_matches_naive_formula():
    h, W, E, b = make(lead=(2, 5))
    O, lse = swvr.softmax_readout_reference(h, W, E, b, logit_scale=0.5)
    S = 0.5 * torch.einsum("btk,vk->btv", h, W) + b
    torch.testing.assert_close(O, (S.exp() / S.exp().sum(-1, keepdim=True)) @ E)
    torch.testing.assert_close(lse, S.exp().sum(-1).log())


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_mse_matches_reference(reduction):
    h, W, E, b = make(lead=(2, 3))
    target = torch.randn(2, 3, 5, dtype=f64)
    weight = torch.tensor([[1., 0., 2.], [1., 1., 0.]], dtype=f64)
    kw = dict(weight=weight, reduction=reduction, z_loss_weight=0.1, logit_scale=0.5)
    loss = swvr.softmax_readout_mse(h, W, E, target, b, chunk_size=8, **kw)
    ref = mse_reference(h, W, E, target, b, **kw)
    torch.testing.assert_close(loss, ref)
    g = torch.autograd.grad(loss.sum(), (h, W, E, b))
    gr = torch.autograd.grad(ref.sum(), (h, W, E, b))
    for a, r in zip(g, gr):
        torch.testing.assert_close(a, r)


def test_module_wraps_linear_and_embedding():
    torch.manual_seed(0)
    head, emb = nn.Linear(12, 41).double(), nn.Embedding(41, 5).double()
    h, target = torch.randn(7, 12, dtype=f64), torch.randn(7, 5, dtype=f64)
    loss_fn = swvr.SoftmaxReadoutMSE(head, emb, z_loss_weight=1e-2, chunk_size=8)
    loss = loss_fn(h, target)
    ref = mse_reference(h, head.weight, emb.weight, target, head.bias, z_loss_weight=1e-2)
    torch.testing.assert_close(loss, ref)
    loss.backward()
    assert head.weight.grad is not None and head.bias.grad is not None
    assert emb.weight.grad is not None


def test_activation_checkpointing():
    from torch.utils.checkpoint import checkpoint
    h, W, E, b = make()
    fn = lambda h: swvr.softmax_readout(h, W, E, b, chunk_size=8)[0].pow(2).sum()
    g1 = torch.autograd.grad(checkpoint(fn, h, use_reentrant=False), h)[0]
    g2 = torch.autograd.grad(fn(h), h)[0]
    torch.testing.assert_close(g1, g2)


def test_input_validation():
    h, W, E, b = make()
    with pytest.raises(ValueError):
        swvr.softmax_readout(h, W[:, :-1], E)
    with pytest.raises(ValueError):
        swvr.softmax_readout(h, W, E[:-1])
    with pytest.raises(ValueError):
        swvr.softmax_readout(h, W, E, b[:-1])
    with pytest.raises(ValueError):
        swvr.softmax_readout(h, W, E, chunk_size=-1)
