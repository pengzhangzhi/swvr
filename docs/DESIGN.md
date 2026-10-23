# Design

## Problem

`O = softmax(h @ W.T + b) @ E` needs the `[N, V]` logits. At N = 16k tokens and V = 152k,
one bf16 copy is 4.9 GB, and eager autograd keeps several copies.

## Algorithm

Process the vocabulary in chunks of `C` rows (default 8192).

**Forward** (an online softmax, as in FlashAttention):

```
for each chunk c:
    S    = h @ W_c.T + b_c
    m'   = max(m, rowmax(S))
    l    = l * exp(m - m') + rowsum(exp(S - m'))
    acc  = acc * exp(m - m') + exp(S - m') @ E_c
    m    = m'
O = acc / l,   LSE = m + log(l)
```

Only `O` `[N, D]` and `LSE` `[N]` are saved.

**Backward** (scores are recomputed):

```
delta = rowsum(O * dO) - dLSE
for each chunk c:
    P    = exp(h @ W_c.T + b_c - LSE)
    dS   = P * (dO @ E_c.T - delta)
    dh  += dS @ W_c
    dW_c = dS.T @ h
    db_c = colsum(dS)
    dE_c = P.T @ dO
```

Peak extra memory is O(N · C). The cost is one extra GEMM, the recompute.

## Implementation

- **GEMMs** run in cuBLAS with fp32 output, so the scores are never rounded to bf16.
- **Everything else** runs in one fused Triton kernel per chunk, including the small `P @ E`
  products, so `P` never touches memory. Gradients are deterministic (no atomics).
- **Frozen inputs** skip their gradient GEMMs.
- **`torch.compile`:** the forward and backward are opaque custom ops, so the compiler
  can't reorder the chunk loop and lose the memory bound.
- **CPU and fp64** use the same algorithm with plain torch ops, and that path is
  gradchecked.

Because the GEMMs take about 80% of the runtime, a fully fused kernel would gain little.
The original PlaidQ CUTLASS kernel measured about 1.5× slower than this design.

## Accuracy

Relative error against an fp64 reference is about 3e-3 for bf16 inputs and about 4e-4 for
fp16. The result doesn't depend on the chunk size.
