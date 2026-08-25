# Multi-Head Self-Attention

## Overview

A decoder-only transformer needs one function above all others: each token position must pull information from earlier positions in the same sequence. This module builds that function as a causal multi-head self-attention block with explicit tensor operations:

- A single `D -> 3D` linear projection produces Q, K, and V together.
- The result is split into `H` parallel heads of width `d_head = D // H`.
- Scaled dot-product attention runs as batched tensor math over `(B, H, T, T)` scores.
- A registered causal mask blocks every future key with `-inf` before the softmax.
- A final `D -> D` projection mixes the heads.
- `forward(..., return_weights=True)` exposes per-head attention weights for inspection.
- A small repeat-task training demo confirms gradients reach every parameter and prints a per-head heatmap of the trained block.

The shape contract is `(B, T, D) -> (B, T, D)`, with `D % H == 0`. The block is independently verifiable, imports no sibling module, and runs offline on CPU. The only external dependency is PyTorch.

The training demo follows the spec's repeat task, which is solvable through each position's own token/value path. Its measured head maps are reported below as observed; this toy task is not evidence of strong head specialization or language-model quality.

## Architecture

```text
(B, T, D)
   │
   ├── qkv_projection: Linear(D, 3D) ── Q, K, V each (B, T, D)
   │        │
   │        └── reshape (B, T, H, d_head) -> transpose -> (B, H, T, d_head)
   │
   ├── scores = Q @ K^T / sqrt(d_head)        (B, H, T, T)
   ├── scores.masked_fill(causal_mask[:T, :T], -inf)
   ├── weights = softmax(scores, dim=-1)       (B, H, T, T)
   ├── context = weights @ V                   (B, H, T, d_head)
   ├── transpose/reshape -> (B, T, D)
   └── output_projection: Linear(D, D) -> (B, T, D)
```

The block has exactly two linear layers:

| parameter | shape | count |
| --- | --- | ---: |
| `qkv_projection.weight` + bias | `(3D, D)`, `(3D,)` | `3D² + 3D` |
| `output_projection.weight` + bias | `(D, D)`, `(D,)` | `D² + D` |

At `D=32` that is `4·32² + 4·32 = 4224` parameters. The causal mask, softmax, matmuls, and reshapes add no parameters.

### Fused QKV split

Three separate `D -> D` projections are mathematically the same as one `D -> 3D` projection whose weight is the three matrices stacked. The fused version launches one matmul, stores the three sub-matrices in one parameter tensor, and lets the implementation slice the result:

```python
qkv = self.qkv_projection(x)        # (B, T, 3D)
q, k, v = qkv.split(self.d_model, dim=-1)
```

The order is Q first, then K, then V. Tests pin this against three independent `nn.Linear` layers and against an explicit per-head reference.

### Head reshape

Each of Q, K, V is reshaped from `(B, T, D)` to `(B, T, H, d_head)` and transposed to `(B, H, T, d_head)`. The head dimension sits next to the batch dimension, so PyTorch treats the per-head attention as `B * H` independent batched problems. `d_head` stays last so `Q @ K.transpose(-2, -1)` contracts it and produces `(B, H, T, T)` scores.

### Scaling

Scores are divided by `sqrt(d_head)` before the softmax:

```text
scores = Q K^T / sqrt(d_head)
```

Without scaling, dot-product magnitude grows with `d_head`, pushing the softmax toward one-hot distributions with tiny gradients. The scaling keeps the score variance roughly constant across head sizes. Tests recompute the scores from the block's own parameters and compare the returned weights at `atol=1e-10` in float64.

### Causal mask

The mask is a non-persistent boolean buffer of shape `(L, L)`, where `L` is the maximum context length. `True` means blocked, matching the convention of `nn.MultiheadAttention`'s boolean mask (and the inverse of `scaled_dot_product_attention`'s boolean mask). At forward time the top-left `(T, T)` corner is sliced:

```python
scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
```

`torch.softmax` therefore assigns exactly zero weight to future keys. Because the buffer is non-persistent it moves with `.to(device)` and is absent from `state_dict()`. It stays boolean under dtype conversion. It is derived state, not checkpoint state.

### Output projection and inspection

Per-head context vectors `(B, H, T, d_head)` are transposed back to `(B, T, H, d_head)`, reshaped to `(B, T, D)`, and passed through a final `D -> D` projection. That projection is the only place where the heads mix inside the block. `return_weights=True` returns the softmax weights with shape `(B, H, T, T)`; each row sums to one, is nonnegative, and is exactly zero above the diagonal.

### Initialization and dtypes

Both linear layers use `N(0, 0.02)` weights and zero biases, with a local `torch.Generator` when `seed` is provided. The mask is boolean regardless of the input dtype. Float32 is the measured configuration; float64 is tested end-to-end. Integer inputs are rejected. Float16/bfloat16 execution is not claimed or measured.

## Quickstart

Requires `uv` and Python 3.11 or newer. From the repository root:

```bash
cd 33-multihead-self-attention
./verify.sh
```

`verify.sh` syncs the frozen lockfile, runs the tests, runs strict mypy, and exercises the default demo plus the `--heads 8` variant. It exits 0 and is safe to run repeatedly. The first run downloads the pinned CPU wheels into the ignored `.venv/`; later runs work offline.

Run the demo directly:

```bash
uv run python main.py demo
uv run python main.py demo --heads 8
uv run python main.py demo --dim 32 --heads 8 --context-length 12 --vocab-size 64 \
    --epochs 3 --batch-size 64 --train-size 2048 --learning-rate 0.01 --seed 1234
```

| flag | default | meaning |
| --- | ---: | --- |
| `--dim` | `32` | model dimension `D` |
| `--heads` | `4` | attention heads `H`; `D % H` must be 0 |
| `--context-length` | `12` | tokens per sequence `T`; must be at least 2 |
| `--vocab-size` | `64` | synthetic vocabulary size `V` |
| `--epochs` | `3` | training epochs |
| `--batch-size` | `64` | sequences per batch |
| `--train-size` | `2048` | synthetic training sequences |
| `--learning-rate` | `0.01` | AdamW learning rate |
| `--heatmap-head` | `0` | head index rendered as a character heatmap |
| `--seed` | `1234` | base seed for model initialization and data |

Each training row is a single random id replicated across all `T` positions. The loss predicts the next token, so the target is the input shifted by one. Validation uses 256 fresh rows, and the inspection input is a separate fixed random sequence. A non-positive size, non-integer seed, invalid head count, invalid heatmap index, or divisibility violation exits with status 2 and one `error:` line.

Torch may emit a `Failed to initialize NumPy` user warning because NumPy is intentionally not a dependency. The module does not use NumPy; the warning is harmless.

## Offline boundary and optional GPU mode

Everything in this module runs offline on CPU. The tests and demo make no network calls and need no provider, paid API, dataset, tokenizer, or remote service. Token ids are obviously synthetic integers, and all randomness comes from local generators seeded by `--seed`. There is no real-service adapter to simulate; the entire boundary is local tensor math.

The block is device-agnostic: the projection weights, embedding tables, and derived mask buffer all move with `.to(device)`. GPU execution is possible but was not run, because this environment has no CUDA device:

```python
import torch
from main import MultiHeadSelfAttention

block = MultiHeadSelfAttention(32, 4, 12, seed=0).to("cuda")
x = torch.randn(2, 12, 32, device="cuda")
output, weights = block(x, return_weights=True)  # both on cuda
```

No GPU timing or numerical result is claimed here.

## Measured results

Measured locally with Python 3.14.10, `torch==2.15.0+cpu`, `uv 0.13.9`, Linux 7.3.5-2-cachyos (x86_64), and no CUDA device. These are single-run local observations, not benchmark claims. Values are deterministic for the pinned CPU environment and the noted seed.

### Verification

```text
$ uv run --frozen pytest
23 passed, 1 warning in 2.09s

$ uv run --frozen mypy main.py tests
Success: no issues found in 2 source files

$ ./verify.sh
verify: ok
```

A warm `./verify.sh` run completed in about 8 seconds on this machine. The one pytest warning is the optional NumPy probe described above.

The 23 tests cover the shape contract, parameter count, mask buffer, input/configuration validation, fused-QKV equivalence, an explicit per-head reference, scaling/mask/softmax arithmetic, a cross-check against PyTorch's `scaled_dot_product_attention`, causality, softmax properties, gradient flow, float64 execution, seeded initialization, entropy, and the demo's smoke, eight-head, determinism, learning, and rejection paths.

### Default demo (`D=32`, `H=4`, `T=12`, `V=64`)

```text
training (repeat task):
initial_val_loss=4.1589
  epoch=1 train_loss=3.4468 val_loss=1.9966
  epoch=2 train_loss=0.9836 val_loss=0.3221
  epoch=3 train_loss=0.1178 val_loss=0.0143
final_val_loss=0.0143
final_val_accuracy=1.0000
grad_flow=ok
attention_config: d_model=32 n_heads=4 d_head=8 max_context_length=12
inspection_input: shape=(1, 12, 32) dtype=torch.float32
attention_output: shape=(1, 12, 32)
attention_weights: shape=(1, 4, 12, 12)
causal_violation_max=0.0
row_sum_max_error=1.79e-07
parameter_counts: attention=4224 model=8704 attention_share=48.5%
inspection_ids=[57, 33, 37, 56, 63, 11, 14, 51, 61, 60, 37, 12]
```

The attention block alone is `4224` parameters. The full toy model is `8704`: `64·32` token embedding `= 2048`, `12·32` position embedding `= 384`, `32·64` LM head `= 2048`, plus `4224` attention parameters.

Head statistics are averages over the batch and query positions `1..T-1`; the first row is trivial (it can only attend to key 0) and is excluded. `self` is the diagonal weight, `previous` is the key one position back, and `first` is key 0.

| head | entropy | self | previous | first |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.704 | 0.237 | 0.138 | 0.169 |
| 1 | 1.093 | 0.404 | 0.204 | 0.083 |
| 2 | 1.358 | 0.289 | 0.195 | 0.136 |
| 3 | 1.324 | 0.374 | 0.197 | 0.098 |

Measured heatmap for head 0 (`q00` is the first query; characters `" .:-=+*#%@"` scale weights from zero to one):

```text
q00 | @
q01 | - *
q02 | : . +
q03 | : = . .
q04 | : : . . :
q05 | . :     : :
q06 | . . . . . . .
q07 | . :     . :
q08 |     . .     :   :
q09 |   . . . .       .
q10 |       .     .   .
q11 |       .     .   .
```

The held-out loss falls from random `log(64) ≈ 4.1589` to `0.0143`, and next-token accuracy on held-out repeat sequences reaches `1.0000`. The repeat task is easy enough that the model can solve it through each position's own token/value path; the measured heads differ in entropy and self/previous focus, but this is not a strong specialization demonstration.

### Eight heads (`D=32`, `H=8`, `d_head=4`)

```text
initial_val_loss=4.1589
  epoch=1 train_loss=3.4468 val_loss=1.9965
  epoch=2 train_loss=0.9836 val_loss=0.3222
  epoch=3 train_loss=0.1179 val_loss=0.0143
final_val_loss=0.0143
final_val_accuracy=1.0000
attention_weights: shape=(1, 8, 12, 12)
```

The parameter count stays `4224` for the attention block and `8704` for the toy model: splitting the same `D` into more heads does not change the parameter inventory.

| head | entropy | self | previous | first |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.180 | 0.279 | 0.055 | 0.088 |
| 1 | 1.508 | 0.241 | 0.126 | 0.132 |
| 2 | 0.989 | 0.382 | 0.178 | 0.090 |
| 3 | 1.465 | 0.297 | 0.185 | 0.120 |
| 4 | 1.561 | 0.248 | 0.230 | 0.166 |
| 5 | 1.465 | 0.275 | 0.156 | 0.157 |
| 6 | 1.579 | 0.248 | 0.203 | 0.176 |
| 7 | 1.245 | 0.394 | 0.192 | 0.069 |

The measured head-0 heatmap changes from the four-head pattern above to a more self-focused map:

```text
q00 | @
q01 |   @
q02 | -   *
q03 |   @
q04 | . -   . :
q05 |   -     . =
q06 | .   . .     =
q07 |   -     . =
q08 |             #
q09 | .   .       : .
q10 |             %
q11 |             .         .
```

The comparison is a visualization of how the same block's per-head weights respond to changing `H`; it is not a claim that eight heads are better on this toy task. The loss and accuracy are effectively unchanged.

## Design decisions

- **Manual matmul/softmax, not a fused kernel.** The lesson is the mechanism, so the block writes out projection, reshape, score, mask, softmax, weighted sum, and output projection. Tests cross-check the result against PyTorch's optimized `scaled_dot_product_attention`; production models should use the fused path (`nn.MultiheadAttention`/SDPA) unless they need the explicit weights.
- **Boolean mask with `-inf`, non-persistent.** A bool buffer is dtype-independent and gives exact zeros after softmax. Masking before the softmax (rather than after) is what makes future weights zero, not merely small. The buffer is derived from `(L, L)` and is therefore not checkpoint state.
- **One fused QKV projection.** It matches the efficient implementation pattern and is mathematically equivalent to three separate projections; a test copies the fused rows into three `nn.Linear` layers and compares. Fused initialization also keeps the three views under one parameter tensor.
- **`N(0, 0.02)` weights, zero biases, local seeded generator.** This is the GPT-2/nanoGPT initialization convention. Using a local generator makes the demo reproducible without changing global RNG state.
- **No dropout, residual, layer norm, or feed-forward block.** Those belong to the transformer-layer lesson; adding them here would obscure the attention contract. The toy model is deliberately just embeddings, attention, and an LM head.
- **Head statistics exclude query 0.** Query 0 can attend only to key 0, so its row is always one-hot and would bias every average. The reported stats use queries `1..T-1`.
- **Repeat task reported honestly.** Each row repeats one id, so the next-token target is visible at the current position; the task does not force cross-token attention. The demo still exercises the full block, gradients, masking, and per-head inspection. It does not claim that heads must specialize.
- **Synthetic ids only.** No tokenizer, corpus, or provider is needed. This keeps the module deterministic and offline; it also means the vocabulary and loss are toy quantities, not language-model benchmarks.

## Limitations

- Context length is bounded by `max_context_length`; the mask table is `O(L²)` and the weights are `O(B·H·T²)`.
- The implementation is the readable path, not a fused kernel. No FlashAttention, memory-efficient attention, or throughput benchmark is included.
- Float32 CPU is the measured configuration. Float64 is tested, but mixed precision, float16/bfloat16 behavior, and GPU numerics are not claimed.
- No KV cache. Autoregressive inference would need cached keys and values and a different K/V shape contract.
- No rotary or ALiBi positional encoding. Both would transform Q/K before the score matmul; the block is compatible but does not implement them.
- No feed-forward block, residual connections, layer normalization, dropout, padding mask, cross-attention, or multi-layer stack. The next lesson adds the transformer layer.
- The repeat task is solvable locally and does not require the model to attend across positions. The measured head differences are weak specialization signals, not evidence of useful language modeling.
- Torch's optional NumPy probe emits a warning in this environment; NumPy is intentionally not a dependency.
- Deterministic values are guaranteed within the pinned CPU environment. PyTorch does not guarantee reproducibility across releases, platforms, or CPU/GPU execution.

## References

1. AI Engineering From Scratch, [Multi-Head Self-Attention capstone](https://aiengineeringfromscratch.com/learn/capstone-projects/multihead-self-attention) — shape contract, fused QKV pattern, causal mask, inspection hook, and repeat-task demo.
2. Ashish Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — scaled dot-product attention, multi-head projections, concatenation and output projection, and the decoder's `-inf` masking rationale.
3. PyTorch, [`torch.nn.MultiheadAttention` documentation (2.15)](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html) — per-head weight shapes, boolean mask semantics, and the optimized SDPA path.
4. PyTorch, [`torch.nn.functional.scaled_dot_product_attention` documentation (2.15)](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html) — reference formula, `is_causal` behavior, dtype support, and the inverse boolean-mask convention used by the cross-check test.
5. Andrej Karpathy, [nanoGPT `model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py) — the single `3 * n_embd` projection split into Q/K/V, causal mask buffer pattern, and `N(0, 0.02)` initialization convention.

## License

MIT License, Copyright (c) 2026 Pedro Monteiro. See [LICENSE](LICENSE).
