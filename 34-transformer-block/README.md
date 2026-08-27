# Transformer Block from Scratch

## Overview

One transformer block is the repeating unit of a decoder-only language model. This module builds that unit from tensor operations rather than delegating to a library layer:

- A hand-rolled `LayerNorm` with learnable scale and shift.
- A causal `MultiHeadAttention` with one fused QKV projection and a registered mask.
- A position-wise GELU `FeedForward` network.
- A `TransformerBlock` whose `pre_ln` flag selects pre-LN or post-LN placement.
- A deterministic gradient probe that stacks six blocks of each variant, copies the weights and token ids, and compares the embedding gradient after one backward pass.

The block maps `(B, T, D) -> (B, T, D)` and supports dropout after the attention softmax and after the second feed-forward linear. The implementation is independent of sibling modules, uses only PyTorch, and runs offline on CPU.

The interesting result is not that the two variants produce different numbers. It is where the gradient goes. At initialization, pre-LN keeps a usable gradient at the embedding and spreads gradient magnitude across depth. Post-LN concentrates almost all of the block gradient near the output and leaves the embedding gradient about `8.6e4` times smaller, roughly five orders of magnitude. That is the mechanism behind the warmup requirement discussed by Xiong et al.

## Architecture

```text
(B, T, D)
   │
   ├── pre-LN:  x = x + MHA(LN1(x))
   │            x = x + FF(LN2(x))
   │
   └── post-LN: x = LN1(x + MHA(x))
                x = LN2(x + FF(x))
   │
   └── (B, T, D), same shape
```

The residual stream is the unnormalized tensor flowing from the embedding through every block. In pre-LN, each sublayer reads a normalized copy and adds its update back to the stream. In post-LN, the residual add happens first and the normalization is applied to the result; the next sublayer therefore reads an already normalized tensor.

### LayerNorm

`LayerNorm(d_model, eps=1e-5)` normalizes the last dimension of every token vector with the biased variance and applies per-element `weight` and `bias`:

```text
mean = x.mean(-1, keepdim=True)
var  = ((x - mean) ** 2).mean(-1, keepdim=True)
y    = (x - mean) / sqrt(var + eps) * weight + bias
```

The biased estimator and the placement of `eps` inside the square root match PyTorch's `nn.LayerNorm`. The implementation is tested against it in float64. No `nn.LayerNorm` is used inside the block.

### Causal multi-head attention

`MultiHeadAttention(D, H, L)` holds one `Linear(D, 3D)` projection and one `Linear(D, D)` output projection. The fused projection is split into Q, K, and V, reshaped to `(B, T, H, D/H)`, and transposed to `(B, H, T, D/H)`. Scores are `Q K^T / sqrt(D/H)`, the upper triangle is filled with `-inf`, softmax produces weights, dropout may be applied to those weights, and the weighted values are concatenated and projected back to `D`.

The causal mask is a boolean `L x L` buffer registered with `persistent=False`. `True` means blocked. It is sliced to the active sequence length on every forward pass, so constructing the block allocates it once. Because it is derived state it moves with `.to(device)` but never appears in `state_dict()`.

### Feed-forward network

`FeedForward` is a two-layer position-wise network:

```text
D -> 4D -> GELU -> dropout -> D
```

The dropout is after the second linear, before the residual add. No tokens interact inside this sublayer.

### Dropout placement

The block contains exactly two dropout modules: one on the attention weights after softmax and one after the second feed-forward linear. There is no dropout directly on the residual stream. Dropping the residual would break the additive identity that carries the signal and gradient through depth.

### Diagnostic stack and probe

The demo uses an embedding plus `N` blocks with no terminal LayerNorm. A final norm belongs to a full language model, not to the block-stack comparison; adding one to only one variant would confound the measurement. Both stacks are built with the same seed, then `post.load_state_dict(pre.state_dict())` so every projection, norm, and embedding value is identical. The only difference is normalization placement.

The probe loss is `output.square().mean()` on the stack output. It is an output-energy probe, not a language-model loss. After one backward pass the demo reports:

- the embedding weight gradient norm,
- the parameter gradient norm of every block,
- the pre/post ratio,
- the output difference,
- causal-mask and attention-row checks,
- parameter counts.

### Parameter counts

With bias in every linear and two LayerNorms per block:

| component | scalar parameters |
| --- | ---: |
| `LayerNorm` | `2D` |
| `MultiHeadAttention` | `4D² + 4D` |
| `FeedForward` | `8D² + 5D` |
| `TransformerBlock` | `12D² + 13D` |
| diagnostic stack with `L` blocks | `V·D + L·(12D² + 13D)` |

At `D=128`, `H=4`, `V=256`, a block has `198,272` parameters. Six blocks plus the embedding have `1,222,400`; twelve blocks have `2,412,032`. The tests pin these formulas.

## Quickstart

Requires `uv` and Python 3.11 or newer. From the repository root:

```bash
cd 34-transformer-block
./verify.sh
```

`verify.sh` syncs the frozen lockfile into the ignored `.venv/`, runs the tests and strict mypy, runs the default demo, and runs the twelve-layer variant. It exits 0 and is safe to run repeatedly. The first run downloads the pinned CPU wheels; later runs use the local environment.

Run the demo directly:

```bash
uv run python code/main.py
uv run python code/main.py demo --layers 12
python3 code/main.py          # works when torch is already installed
```

| flag | default | meaning |
| --- | ---: | --- |
| `--d-model` | `128` | model dimension `D` |
| `--heads` | `4` | attention heads `H`; `D % H` must be 0 |
| `--layers` | `6` | blocks per stack; must be at least 2 for the probe |
| `--context-length` | `16` | tokens per sequence `T`; must be at least 2 |
| `--batch-size` | `4` | sequences in the probe |
| `--vocab-size` | `256` | synthetic vocabulary size; must be at least 2 |
| `--seed` | `1234` | base seed for initialization and token ids |

The `demo` subcommand is optional. Semantic validation errors print one `error:` line to stderr and exit with status 2. The NumPy warning from PyTorch is harmless: NumPy is intentionally not a dependency and the module does not use it.

## Offline boundary and optional GPU mode

Everything the tests and demo do runs offline on CPU. There is no tokenizer, dataset, provider, paid API, remote service, or network call at runtime. Token ids are synthetic integers from a local generator, and every weight is initialized from a seeded local `torch.Generator`. The only network access is the one-time dependency installation performed by `uv`.

The modules are device-agnostic: parameters and the derived mask buffer move with `.to(device)`. GPU execution is possible but was not run in this environment, so no GPU timing or numerical result is claimed:

```python
import sys
sys.path.insert(0, "code")

import torch
from main import TransformerBlock

block = TransformerBlock(32, 4, 12, seed=0).to("cuda")
x = torch.randn(2, 12, 32, device="cuda")
print(block(x).shape)
```

## Measured results

Measured locally with Python 3.14.10, `torch==2.15.0+cpu`, `uv 0.13.9`, Linux x86_64, CPU only. These are single-run observations for a fixed seed, not benchmark claims. PyTorch does not guarantee identical floating-point results across releases, platforms, or CPU/GPU execution.

### Verification

```text
$ ./verify.sh
...
19 passed, 1 warning in 1.13s
Success: no issues found in 5 source files
verify: ok
```

A warm verification run took about 4.6 seconds on this machine. The warning is PyTorch's optional NumPy probe.

The 19 tests cover LayerNorm equivalence and validation; feed-forward shape, parameter count, and dropout; fused-QKV equivalence; explicit per-head attention; mask persistence and causality; pre/post parameter equality and divergence under copied weights; future-token isolation for both variants; gradient reachability; the two-dropout layout; configuration and input rejection; probe determinism, numeric ordering, and CLI output.

### Default six-layer probe

Command: `uv run --frozen python code/main.py`

```text
transformer block gradient probe
config: layers=6 d_model=128 num_heads=4 head_dim=32 max_context_length=16 batch_size=4 vocabulary_size=256 seed=1234
probe: loss=output.square().mean(), dropout=0.0, no final norm, identical token ids and copied weights
pre_ln: output_shape=(4, 16, 128) embedding_grad_norm=7.729714e-02 probe_loss=2.715585e-02
post_ln: output_shape=(4, 16, 128) embedding_grad_norm=9.000631e-07 probe_loss=9.999901e-01
embedding_grad_ratio=8.587969e+04
max_output_difference=3.918611e+00
pre_ln block_grad_norms: block_0=1.635712e-01 block_1=1.151357e-01 block_2=9.536710e-02 block_3=8.126924e-02 block_4=6.950606e-02 block_5=6.119957e-02
post_ln block_grad_norms: block_0=1.169225e-06 block_1=1.003413e-07 block_2=1.003249e-07 block_3=1.001655e-07 block_4=1.012642e-07 block_5=1.816508e-01
causal_check: future_weight_max=0.0 row_sum_max_error=2.38e-07
parameter_counts: pre_ln=1222400 post_ln=1222400
```

The pre-LN embedding gradient is `7.73e-02`; the post-LN embedding gradient is `9.00e-07`. The ratio is about `8.59e4`. The per-block norms explain the difference: pre-LN block gradients fall gently from `1.64e-01` to `6.12e-02`, while post-LN blocks before the last one sit between `1.0e-7` and `1.2e-6` and the last block is `1.82e-01`. Post-LN's gradient is concentrated near the output, exactly where a large learning-rate update is most destabilizing; the embedding path receives almost no signal. This is the initialization-time pattern that motivates warmup for post-LN and its removal for pre-LN.

The `probe_loss` values are not directly comparable as model quality: post-LN's last block normalizes its output, so the output-energy loss is about 1, while the pre-LN residual stream is unnormalized at the top and its loss is about 0.027. The per-block gradient profile shows the ordering is not a loss-scale artifact.

### Twelve-layer probe

Command: `uv run --frozen python code/main.py demo --layers 12`

```text
pre_ln: output_shape=(4, 16, 128) embedding_grad_norm=1.832322e-01 probe_loss=5.801152e-02
post_ln: output_shape=(4, 16, 128) embedding_grad_norm=8.903538e-07 probe_loss=9.999900e-01
embedding_grad_ratio=2.057970e+05
max_output_difference=3.694118e+00
pre_ln block_grad_norms: block_0=3.714493e-01 block_1=2.665681e-01 ... block_10=1.067305e-01 block_11=1.022539e-01
post_ln block_grad_norms: block_0=1.267032e-06 block_1=1.011012e-07 ... block_10=1.004349e-07 block_11=1.818765e-01
causal_check: future_weight_max=0.0 row_sum_max_error=2.38e-07
parameter_counts: pre_ln=2412032 post_ln=2412032
```

At twelve layers the gap grows to about `2.06e5`. Pre-LN's gradient remains distributed across all twelve blocks. Post-LN's first eleven blocks sit between `1.0e-7` and `1.3e-6` while the last block stays near `1.8e-01`.

## Design decisions

- **Hand-rolled LayerNorm.** The lesson is the block, so the normalization is written out with biased variance and `eps` inside the square root. The float64 test against `nn.LayerNorm` pins the semantics.
- **Fused QKV.** One `Linear(D, 3D)` is mathematically equivalent to three `D -> D` projections and matches the efficient implementation pattern. A test copies the fused rows into three independent linears and compares.
- **Non-persistent boolean mask.** `True` means blocked, `-inf` is applied before softmax, and the buffer is sliced to the active context. It is not checkpoint state.
- **Two dropout sites.** Attention weights are dropped after softmax; the feed-forward output is dropped after the second linear. No dropout touches the residual stream.
- **`N(0, 0.02)` weights and zero biases.** The initialization is explicit and seeded per module. GPT-2's residual scaling and final normalization are model-assembly concerns, not block concerns, and are not included here.
- **No terminal norm in the diagnostic stack.** Both variants end at a block output. A final LayerNorm is part of the next lesson's model assembly; adding one only to one side would make the probe unfair.
- **Output-energy probe at initialization.** The probe measures block behavior without a dataset, tokenizer, or language-model head. It runs one forward and one backward with dropout disabled and identical weights.
- **Six layers by default, twelve available.** Six matches the build step; twelve makes the depth trend visible and is exercised by `verify.sh`.
- **No sibling imports.** Small helpers are duplicated rather than coupling this module to earlier lessons.

## Limitations

- This is one backward pass at initialization, not a training run. It does not measure convergence, final quality, warmup schedules, or survival at a particular learning rate. The cited pre-LN/warmup result is the theoretical and empirical context, not something this demo re-trains.
- The probe loss is output energy, not cross-entropy. The two variants also produce differently scaled outputs because post-LN normalizes at the end of each block. The per-block norms are the fair part of the comparison; the absolute losses are descriptive.
- The diagnostic stack has no final LayerNorm, no LM head, no positional encoding, and no tokenizer.
- Only float32 CPU is measured. Float64 is used in arithmetic tests, but mixed precision and GPU numerics are not claimed.
- The ordering is depth-dependent. A one-block stack can reverse it; the probe therefore requires at least two blocks and defaults to six.
- Pre-LN is not universally better. This probe shows why post-LN is warmup-sensitive; it does not show that pre-LN trains to a better model.
- No KV cache, FlashAttention, RMSNorm, rotary/ALiBi positions, padding mask, or cross-attention is implemented. The block is the decoder self-attention unit, not a full GPT.
- PyTorch emits a harmless `Failed to initialize NumPy` warning because NumPy is intentionally absent.

## References

1. AI Engineering From Scratch, [Transformer Block capstone](https://aiengineeringfromscratch.com/learn/capstone-projects/transformer-block) — block contract, pre-LN/post-LN comparison, fused QKV, mask buffer, dropout placement, and gradient demo.
2. Ashish Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original post-LN transformer, multi-head attention, scaled dot-product attention, and residual/normalization structure.
3. Ruibin Xiong et al., [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — why post-LN needs warmup and why pre-LN's initialization behaves better.
4. Alec Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2's move of layer normalization to the input of each sub-block and the added final normalization.
5. PyTorch, [`torch.nn.LayerNorm` documentation (2.15)](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LayerNorm.html) — biased variance, `eps` placement, affine parameters, and initialization.

## License

MIT License, Copyright (c) 2026 Pedro Monteiro. See [LICENSE](LICENSE).
