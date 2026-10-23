# swvr

Memory-efficient **softmax-weighted embedding readout** for PyTorch:

```
O    = softmax(h @ W.T + b) @ E      # expected embedding under the model's token distribution
loss = mse(O, target)
```

It never builds the `[N, V]` logits, so it uses **~20× less memory** than the eager version
and runs **1.2–1.5× faster** (A100, H200). It's useful for continuous diffusion LMs,
soft-embedding distillation, and any loss on `softmax(logits) @ E`.

## Install

```bash
pip install git+https://github.com/pengzhangzhi/swvr
```

Requires PyTorch with Triton, on any CUDA GPU (also runs on CPU).

## Usage

```python
import swvr

loss = swvr.softmax_readout_mse(h, lm_head.weight, embed.weight, target, bias=lm_head.bias)
```

| argument | shape | |
|---|---|---|
| `h` | `[..., K]` | hidden states (bf16 / fp16 / fp32) |
| `W` | `[V, K]` | output projection |
| `E` | `[V, D]` | embedding table |
| `target` | `[..., D]` | regression target |
| `bias` | `[V]` | optional |
| `weight` | `[...]` | optional per-token weight (0 = ignore) |
| `z_loss_weight` | | optional z-loss on logsumexp |

Other entry points:

```python
loss_fn = swvr.SoftmaxReadoutMSE(model.lm_head, model.embed_tokens)  # module, shares params
O, lse = swvr.softmax_readout(h, W, E, bias)                         # raw op, any loss on top
```

Gradients flow to every input. The op works with autocast, checkpointing, DDP and
`torch.compile`.

## Benchmarks

Forward + backward, bf16, `python bench/bench.py`:

| N, K, V, D | A100 eager | A100 swvr | H200 eager | H200 swvr |
|---|---|---|---|---|
| 16k, 1024, 152k, 16 | 171 ms, 37 GiB | **115 ms, 1.7 GiB** | 72 ms, 37 GiB | **48 ms, 1.7 GiB** |
| 16k, 2560, 152k, 64 | 294 ms, 38 GiB | **240 ms, 2.7 GiB** | 121 ms, 38 GiB | **98 ms, 2.7 GiB** |
| 8k, 4096, 128k, 128 | 184 ms, 17 GiB | **160 ms, 2.7 GiB** | 73 ms, 17 GiB | **62 ms, 2.7 GiB** |

## How it works

The vocabulary is processed in chunks with an online softmax, and the backward recomputes
scores instead of storing them. See [docs/DESIGN.md](docs/DESIGN.md).

## Tests

```bash
pip install -e ".[test]" && pytest tests/
```

Extracted from [PlaidQ](https://github.com/pengzhangzhi/plaidq). BSD-3-Clause.
