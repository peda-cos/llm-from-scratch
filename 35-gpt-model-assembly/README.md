# GPT Model Assembly from Scratch

## Overview

This module assembles a GPT-2-small-shaped decoder from local tensor operations:

- A token embedding of shape `(50257, 768)`.
- A learned position embedding of shape `(1024, 768)` with a cached position-id buffer.
- Twelve pre-LN transformer blocks, each with fused-QKV causal attention and a GELU feed-forward network.
- A final LayerNorm and a language-model head whose weight is the token-embedding tensor itself when weight tying is enabled.
- Temperature scaling, exact top-k truncation, multinomial sampling, greedy decoding, and a sliding context window.

The default configuration has **124,439,808 unique parameters**. Its untied variant has **163,037,184**, a difference of exactly `50257 * 768 = 38,597,376`. The demo builds the full model, counts parameters, confirms the tied storage identity, and runs one forward pass. It also runs a tiny `d_model=64`, `num_layers=2` model end to end and prints the generated ids.

The model is untrained. Generated ids are synthetic vocabulary indices, not text; there is no tokenizer or pretrained checkpoint in this lesson.

## Architecture

```text
(B, T) token ids
   │
   ├── token embedding        (50257, 768)
   └── cached position ids ── position embedding (1024, 768)
   │
   └── add -> embedding dropout
          -> 12 × transformer block
          -> final LayerNorm
          -> tied LM head (768 -> 50257, no bias)
          -> (B, T, 50257) logits
```

Each block is pre-LN:

```text
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

`MultiHeadAttention` uses one fused `(768, 2304)` QKV projection, splits into 12 heads of width 64, applies a causal mask, computes scaled dot-product attention, and projects back with a `(768, 768)` output projection. `FeedForward` expands to `4 × 768 = 3072`, applies GELU, and projects back to 768. Dropout is applied to the attention weights and to the feed-forward output; the embedding dropout is outside the blocks.

### Parameter accounting

| component | unique parameters |
| --- | ---: |
| token embedding, `50257 × 768` | 38,597,376 |
| position embedding, `1024 × 768` | 786,432 |
| 12 blocks, `7,087,872` each | 85,054,464 |
| final LayerNorm | 1,536 |
| tied LM head | 0 additional |
| **total** | **124,439,808** |
| untied head delta | 38,597,376 |
| **untied total** | **163,037,184** |

Per block, with `D=768`, `E=4`, and linear/LayerNorm biases:

```text
(4 + 2E) D² + 4D + (E + 5) D = 12D² + 13D = 7,087,872
```

The exact count is pinned by an automated test. The GPT-2 paper's Table 2 lists the smallest 12-layer/768-d model as 117M, while the 124M label commonly used for GPT-2 small corresponds to the full 50,257-token vocabulary and 1,024-token context used here. The extra difference from a rounded 124M is not a wiring error: `124,439,808 - 124,000,000 = 439,808`.

## Weight tying

The token embedding has shape `(vocab, d_model)`. The LM head needs `d_model -> vocab`. These are transposes of one another, so the model assigns:

```python
self.lm_head.weight = self.token_embedding.weight
```

This is parameter-level sharing, not a copy. `model.lm_head.weight is model.token_embedding.weight` is `True`; the optimizer sees one parameter and gradients accumulate into the same tensor. `count_parameters` walks the module tree with an identity set, so the tied matrix is counted once. The demo prints `lm_head_shares_token_embedding=True` for both the 124M and tiny configurations, and a test checks the shared gradient tensor.

## Positions and generation

The learned position table is indexed by a non-persistent `position_ids` buffer created once for the maximum context:

```python
positions = self.position_ids[:length]
x = self.token_embedding(idx) + self.position_embedding(positions)
```

There is no `torch.arange(T)` in the forward path. The buffer is registered with `persistent=False`, so it moves with `.to(device)` but is not checkpoint state.

Generation takes only the last position of each forward:

```text
logits[:, -1, :]
  -> divide by temperature (unless greedy or temperature == 0)
  -> keep exactly top-k values via topk + scatter (unless k >= vocab)
  -> softmax
  -> torch.multinomial
  -> append
  -> crop the active window to context_length
```

`greedy=True` or `temperature=0` uses `argmax` and ignores top-k. `top_k=1` is equivalent to greedy; `top_k >= vocab_size` disables truncation. `generate` temporarily calls `.eval()` on an `nn.Module` and restores the previous training mode, so dropout is disabled during sampling without changing caller state. A supplied `torch.Generator` makes the sequence reproducible. There is no KV cache; each step recomputes the active window.

## Quickstart

Requires `uv` and Python 3.11 or newer. From the repository root:

```bash
cd 35-gpt-model-assembly
./verify.sh
```

`verify.sh` syncs the frozen lockfile into the ignored `.venv/`, runs pytest and strict mypy, runs the default 124M demo, runs the tiny-only demo, and exits 0. It is safe to run repeatedly. The first run downloads the pinned CPU wheels; later runs use the local environment.

Run the demo directly:

```bash
uv run --frozen python code/main.py
uv run --frozen python code/main.py demo --tiny-only
python3 code/main.py          # works when torch is already installed
```

The default demo builds the full 124M model and performs one forward pass. `--tiny-only` skips that allocation and runs only the small generation path. Semantic validation errors print one `error:` line to stderr and exit with status 2.

| flag | default | meaning |
| --- | ---: | --- |
| `--seed` | `1234` | model and sampling seed |
| `--forward-length` | `8` | tokens in the single 124M forward pass |
| `--tiny-only` | off | skip the 124M model |
| `--tiny-vocab-size` | `256` | tiny vocabulary size; must be at least 5 |
| `--tiny-context-length` | `32` | tiny context length |
| `--tiny-d-model` | `64` | tiny model dimension |
| `--tiny-heads` | `4` | tiny attention heads |
| `--tiny-layers` | `2` | tiny block count |
| `--max-new-tokens` | `8` | generated tokens in the tiny demo |
| `--temperature` | `0.8` | sampling temperature; `0` means greedy |
| `--top-k` | `40` | top-k truncation; values `>= vocab` disable it |
| `--greedy` | off | use argmax instead of sampling |

## Offline boundary and optional GPU mode

Everything the tests and demo do runs offline on CPU. Token ids are synthetic integers, weights are initialized from local seeded generators, and there is no tokenizer, dataset, paid API, provider, or network call at runtime. The only network access is the one-time dependency installation performed by `uv`.

The modules are device-agnostic: parameters and the non-persistent position/mask buffers move with `.to(device)`. GPU execution was not run in this environment, so no GPU timing or numerical result is claimed:

```python
import sys
sys.path.insert(0, "code")

import torch
from main import GPTConfig, GPTModel

model = GPTModel(GPTConfig(), seed=0).to("cuda")
ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
with torch.no_grad():
    logits = model(ids)
print(logits.shape)
```

For generation on GPU, pass a generator on the logits device, for example `torch.Generator(device=ids.device).manual_seed(0)`.

## Measured results

Measured locally with Python 3.14.10, `torch==2.15.0+cpu`, `uv 0.13.9`, Linux x86_64, CPU only. These are single-run observations for a fixed seed, not benchmark claims. PyTorch does not guarantee identical floating-point results across releases, platforms, or CPU/GPU execution.

### Verification

```text
$ ./verify.sh
...
21 passed, 1 warning in 1.51s
Success: no issues found in 5 source files
verify: ok
```

A warm verification run took about 7.0 seconds on this machine. The warning is PyTorch's optional NumPy probe; NumPy is intentionally not a dependency. The 21 tests cover LayerNorm equivalence; causal attention; configuration validation; the exact 124M count; parameter formulas; tied storage and gradients; forward shapes; full-model causality; cached positions; scaled residual initialization; `use_bias`; greedy/temperature/top-k sampling; top-k truncation; sliding-window inputs; mode restoration; reproducibility; report fields; and CLI validation.

### Default 124M report and tiny generation

Command: `uv run --frozen python code/main.py`

```text
gpt model assembly
large: parameters=124439808 (124.44M; reference GPT-2 small ~124M)
large: reference_parameters=124000000 rounded_delta=439808
large: tied_delta_vs_untied=38597376 expected=38597376
large: lm_head_shares_token_embedding=True
large: forward_shape=(1, 8, 50257) forward_ms=44.091
tiny: config=vocab_size=256 context_length=32 d_model=64 num_heads=4 num_layers=2 mlp_expansion=4 dropout=0.0 use_bias=True weight_tying=True
tiny: parameters=118528
tiny: lm_head_shares_token_embedding=True
tiny: prompt_ids=[1, 2, 3, 4]
tiny: generated_ids=[1, 2, 3, 4, 6, 182, 231, 185, 241, 244, 48, 254]
tiny: new_ids=[6, 182, 231, 185, 241, 244, 48, 254]
tiny: repeat_match=True
```

The full process took about 3.2 seconds wall time, including process startup, the 124M allocation, the count, and one forward pass. The reported `forward_ms=44.091` times only the single forward call, after model construction, on a cold CPU run; it is not a throughput benchmark.

### Tiny-only fast path

Command: `uv run --frozen python code/main.py demo --tiny-only`

```text
tiny: config=vocab_size=256 context_length=32 d_model=64 num_heads=4 num_layers=2 mlp_expansion=4 dropout=0.0 use_bias=True weight_tying=True
tiny: parameters=118528
tiny: lm_head_shares_token_embedding=True
tiny: prompt_ids=[1, 2, 3, 4]
tiny: generated_ids=[1, 2, 3, 4, 6, 182, 231, 185, 241, 244, 48, 254]
tiny: new_ids=[6, 182, 231, 185, 241, 244, 48, 254]
tiny: repeat_match=True
```

## Design decisions

- **Frozen `GPTConfig`.** The configuration cannot drift after a model is built, so the parameter count and position buffer cannot silently disagree with the recorded shape.
- **Local block implementation.** `code/main.py` reimplements the lesson-34 block locally; there are no sibling imports. The module is independently verifiable.
- **Pre-LN blocks and final LayerNorm.** Each sublayer reads a normalized copy and adds its update to the residual stream. The terminal LayerNorm is outside the block stack.
- **Fused QKV.** One `(768, 2304)` projection is split into Q, K, and V; this is mathematically equivalent to three separate projections and matches the efficient implementation pattern.
- **Two dropout sites per block.** Attention weights and the feed-forward output are dropped; the residual stream itself is not. A third dropout sits on the summed token/position embeddings.
- **Bias semantics.** `use_bias` toggles biases on all linear projections. LayerNorm keeps its affine weight and bias because they are part of the normalization. The LM head always has `bias=False`, matching GPT-2.
- **Embedding initialization.** Token and position embeddings and non-residual linears use `N(0, 0.02)`, following nanoGPT's GPT-2 initialization.
- **Scaled residual initialization.** The attention output projection and the second feed-forward linear use `0.02 / sqrt(2 * num_layers)`, so the residual stream does not grow with depth. At twelve layers this is `0.02 / sqrt(24) ≈ 0.00408`.
- **Cached positions.** `position_ids` is allocated once as a non-persistent buffer and sliced per forward; it is not reconstructed with `torch.arange`.
- **Parameter-level tying.** The tied head shares the token-embedding `Parameter` object, not a copy. `count_parameters` counts unique objects.
- **Meta-device untied count.** The untied comparison constructs the second module graph on `torch.device("meta")`, avoiding a second ~0.5 GB allocation while still counting the real wiring.
- **Exact top-k.** `topk` plus `scatter` keeps exactly `k` finite logits, including when the k-th logit is tied. `top_k >= vocab_size` skips truncation.
- **Inference hygiene.** `generate` disables autograd and dropout, restores the caller's training mode in a `finally` block, and accepts a seeded generator for reproducibility.
- **No 124M generation.** The full model is built, counted, and forwarded once; generation runs on the tiny model to keep the demo fast.

## Limitations

- The model is randomly initialized and untrained. Generated ids are vocabulary indices, not meaningful text.
- There is no tokenizer, dataset, pretrained GPT-2 checkpoint, or training loop. Loading pretrained weights is a later lesson.
- No KV cache: each generated token recomputes the active window, which is simple but not throughput-optimized.
- Top-p (nucleus) sampling, repetition penalty, and sinusoidal position embeddings are not implemented; they are optional exercises in the source lesson, not part of this Build It core.
- Only CPU execution was measured. The optional GPU snippet is not a measured result.
- The forward timing is one cold single run, not a benchmark; CPU scheduling and allocator state affect it.
- The 124M float32 model occupies roughly 0.5 GB before activations. The default demo builds it once; use `--tiny-only` on memory-constrained machines.
- PyTorch emits a harmless `Failed to initialize NumPy` warning because NumPy is intentionally absent.
- Batch generation uses one generator and one sampling row per sequence; per-row seeding is not exposed.

## References

1. AI Engineering From Scratch, [GPT Model Assembly capstone](https://aiengineeringfromscratch.com/learn/capstone-projects/gpt-model-assembly) — task contract, 124M configuration, parameter accounting, weight tying, and generation requirements.
2. Alec Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2 architecture, pre-LN placement, final LayerNorm, `1/sqrt(N)` residual scaling, vocabulary and context sizes.
3. Ofir Press and Lior Wolf, [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) — tying the input embedding and output projection, and the parameter/quality rationale.
4. Andrej Karpathy, [nanoGPT `model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py) — the GPT-2-style assembly, tied `wte`/`lm_head`, 0.02 initialization, and scaled `c_proj` initialization.
5. PyTorch, [`torch.multinomial` documentation](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html) — non-negative finite weights, row-wise sampling, and the generator argument used by the generation loop.

## License

MIT License, Copyright (c) 2026 Pedro Monteiro. See [LICENSE](LICENSE).
