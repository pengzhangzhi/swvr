"""GPU tests. Skipped without CUDA; run them on the target GPUs (A100, H200)."""
import pytest
import torch

import swvr

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(params=["triton", "torch"], autouse=True)
def glue(request, monkeypatch):
    """Run every test on both the fused Triton path and the plain torch path."""
    monkeypatch.setenv("SWVR_DISABLE_TRITON", "1" if request.param == "torch" else "0")
    return request.param


def make(N, K, V, D, dtype, bias=True, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    t = lambda *s, scale=1.0: (scale * torch.randn(*s, generator=g, device="cuda")).to(dtype)
    # Realistic scales: unit hidden states, head init ~ 1/sqrt(K) * a few, so logits spread over ~±10.
    h, W, E = t(N, K), t(V, K, scale=3 / K ** 0.5), t(V, D)
    b = t(V, scale=0.5) if bias else None
    return [x.requires_grad_(True) if x is not None else None for x in (h, W, E, b)]


def run(fn, inputs, gO, glse):
    O, lse = fn(*inputs)
    xs = [x for x in inputs if x is not None and x.requires_grad]
    gs = torch.autograd.grad((O * gO).sum() + (lse * glse).sum(), xs)
    return [O, lse, *gs]


def rel_err(a, r):
    return ((a.double() - r.double()).norm() / r.double().norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("chunk", [1024, 3000])
@pytest.mark.parametrize("D", [5, 16, 128])
def test_low_precision_matches_fp64(dtype, chunk, D, glue):
    N, K, V = 517, 512, 10007
    inputs = make(N, K, V, D, dtype)
    gO = torch.randn(N, D, device="cuda")
    glse = 0.1 * torch.randn(N, device="cuda")
    out = run(lambda *a: swvr.softmax_readout(*a, chunk_size=chunk, logit_scale=0.8), inputs, gO, glse)
    ref_in = [x.detach().double().requires_grad_(True) for x in inputs]
    ref = run(lambda *a: swvr.softmax_readout_reference(*a, logit_scale=0.8), ref_in,
              gO.double(), glse.double())
    names = ["O", "lse", "gh", "gW", "gE", "gb"]
    errs = {n: rel_err(a, r) for n, a, r in zip(names, out, ref)}
    print(glue, dtype, chunk, D, {k: f"{v:.1e}" for k, v in errs.items()})
    # Matrix-product operands are rounded to bf16/fp16, so relative errors around 2^-8 are expected.
    for n, e in errs.items():
        assert e < 2e-2, (n, e)


def test_fp32_no_tf32_is_exact():
    torch.backends.cuda.matmul.allow_tf32 = False
    N, K, V, D = 256, 128, 5000, 8
    inputs = make(N, K, V, D, torch.float32)
    gO, glse = torch.randn(N, D, device="cuda"), torch.randn(N, device="cuda")
    out = run(lambda *a: swvr.softmax_readout(*a, chunk_size=777), inputs, gO, glse)
    ref = run(swvr.softmax_readout_reference, inputs, gO, glse)
    for a, r in zip(out, ref):
        assert rel_err(a, r) < 1e-5


def test_autocast_with_fp32_params():
    N, K, V, D = 256, 256, 4000, 16
    h, W, E, b = make(N, K, V, D, torch.float32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        O, lse = swvr.softmax_readout(h, W, E, b, chunk_size=1024)
    assert O.dtype == torch.float32
    (O.sum() + lse.sum()).backward()
    assert W.grad.dtype == torch.float32 and h.grad.dtype == torch.float32
    O_r, _ = swvr.softmax_readout_reference(h.detach(), W.detach(), E.detach(), b.detach())
    assert rel_err(O, O_r) < 2e-2


def test_bf16_chunk_invariance():
    inputs = make(300, 256, 9000, 16, torch.bfloat16)
    gO, glse = torch.randn(300, 16, device="cuda"), torch.randn(300, device="cuda")
    a = run(lambda *x: swvr.softmax_readout(*x, chunk_size=512), inputs, gO, glse)
    b = run(lambda *x: swvr.softmax_readout(*x, chunk_size=9000), inputs, gO, glse)
    for x, y in zip(a, b):
        assert rel_err(x, y) < 1e-2


def test_peak_memory_is_chunk_bounded():
    N, K, V, D, C = 4096, 1024, 65536, 16, 4096
    inputs = make(N, K, V, D, torch.bfloat16)
    target = torch.randn(N, D, device="cuda")
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    loss = swvr.softmax_readout_mse(*inputs[:3], target, inputs[3], chunk_size=C)
    loss.backward()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base
    # Grad buffers (W, h, E, b) + gH fp32 accumulator + a few [N, C] fp32 tiles.
    grads = sum(x.numel() * x.element_size() for x in inputs) + N * K * 4
    tiles = 6 * N * C * 4
    one_logit_copy = N * V * 2
    print(f"extra {extra / 2**20:.0f} MiB, budget {(grads + tiles) / 2**20:.0f} MiB, "
          f"one bf16 logits copy {one_logit_copy / 2**20:.0f} MiB")
    assert extra < grads + tiles


def test_masked_vocab_and_frozen_codebook():
    N, K, V, D = 300, 256, 5000, 16
    h, W, E, b = make(N, K, V, D, torch.bfloat16)
    E.requires_grad_(False)
    with torch.no_grad():
        b[:2048] = float("-inf")
    gO, glse = torch.randn(N, D, device="cuda"), torch.randn(N, device="cuda")
    out = run(lambda *a: swvr.softmax_readout(*a, chunk_size=1024), (h, W, E, b), gO, glse)
    ref_in = [x.detach().double().requires_grad_(x.requires_grad) for x in (h, W, E, b)]
    ref = run(swvr.softmax_readout_reference, ref_in, gO.double(), glse.double())
    for a, r in zip(out, ref):
        assert torch.isfinite(a).all()
        assert rel_err(a, r) < 2e-2


def test_torch_compile_matches_eager_and_stays_chunked(glue):
    N, K, V, D, C = 2048, 512, 32768, 16, 2048
    h, W, E, b = make(N, K, V, D, torch.bfloat16)
    target = torch.randn(N, D, device="cuda")
    fn = lambda h: swvr.softmax_readout_mse(h, W, E, target, b, chunk_size=C)
    torch._dynamo.reset()
    cfn = torch.compile(fn, fullgraph=True)
    for _ in range(2):                                  # compile, then measure a warm call
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        loss_c = cfn(h)
        g_c = torch.autograd.grad(loss_c, (h, W))
        torch.cuda.synchronize()
        extra = torch.cuda.max_memory_allocated() - base
    loss_e = fn(h)
    g_e = torch.autograd.grad(loss_e, (h, W))
    assert rel_err(loss_c, loss_e) < 1e-3
    for a, r in zip(g_c, g_e):
        assert rel_err(a, r) < 1e-2
    one_logit_copy = N * V * 2
    print(f"compiled extra {extra / 2**20:.0f} MiB vs one bf16 logits copy {one_logit_copy / 2**20:.0f} MiB")
    assert extra < one_logit_copy
