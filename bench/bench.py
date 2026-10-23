"""Peak memory and fwd+bwd time of the swvr MSE loss against an eager baseline.

    python bench/bench.py
    python bench/bench.py --shapes 0.6B --chunks 4096,8192

Peak memory is measured above the inputs and includes the gradient buffers.
"""
import argparse

import torch

import swvr

SHAPES = {                         # N (tokens per step), K, V, D
    "0.6B": (16384, 1024, 151936, 16),
    "4B": (16384, 2560, 151936, 64),
    "8B": (8192, 4096, 128256, 128),
}


def eager(h, W, E, b, target, chunk=None):
    O = torch.softmax(torch.addmm(b, h, W.T).float(), dim=-1) @ E.float()
    return (O - target).pow(2).mean()


def chunked(h, W, E, b, target, chunk=None):
    return swvr.softmax_readout_mse(h, W, E, target, b, chunk_size=chunk)


def measure(fn, inputs, chunk, iters=5):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(2):
            fn(*inputs, chunk=chunk).backward()
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - base) / 2**30
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(*inputs, chunk=chunk).backward()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters, peak
    except torch.OutOfMemoryError:
        return None, None
    finally:
        for x in inputs[:4]:
            x.grad = None
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--chunks", default="8192")
    args = ap.parse_args()
    print(f"# {torch.cuda.get_device_name()}, torch {torch.__version__}")
    for name in args.shapes.split(","):
        N, K, V, D = SHAPES[name]
        g = torch.Generator(device="cuda").manual_seed(0)
        r = lambda *s, sc=1.0: (sc * torch.randn(*s, device="cuda", generator=g)).to(torch.bfloat16)
        inputs = [r(N, K), r(V, K, sc=3 / K ** 0.5), r(V, D), r(V, sc=0.5)]
        inputs = [x.requires_grad_(True) for x in inputs] + [torch.randn(N, D, device="cuda")]
        runs = [("eager", eager, None)] + [(f"swvr C={c}", chunked, int(c))
                                            for c in args.chunks.split(",")]
        for label, fn, chunk in runs:
            ms, gib = measure(fn, inputs, chunk)
            result = "OOM" if ms is None else f"{ms:7.1f} ms   peak {gib:6.2f} GiB"
            print(f"{name:5s} (N={N}, K={K}, V={V}, D={D})  {label:12s} {result}", flush=True)
        del inputs


if __name__ == "__main__":
    main()
