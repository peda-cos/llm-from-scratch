# Token and Positional Embeddings

## Overview

A transformer never sees token ids as integers. The embedding stage turns a batch of ids of shape `(B, T)` into vectors of shape `(B, T, D)` by looking up one row per id, then adds a position-dependent vector of the same width. This module builds the token table, both standard position tables, and the composition:

- `TokenEmbedding` — learned `(V, D)` lookup table initialized at `N(0, 0.02)`.
- `LearnedPositionalEmbedding` — learned `(max_context_length, D)` table indexed by position.
- `SinusoidalPositionalEmbedding` — fixed sin/cos table precomputed once, with no parameters.
- `EmbeddingComposer` — sums token and positional vectors into the input for the next stage.

The composition is a sum, not a concatenation, so `D` stays constant through the network. This lesson stops before attention, an LM head, a loss, or training. The spec's `code/tests/test_embeddings.py` maps to this track's `tests/test_embeddings.py`; the module is independently verifiable and imports no sibling module.

## Architecture

```
(B, T) token ids
   │
   ├── TokenEmbedding (V, D) ───────────────────┐
   │      ids -> (B, T, D)                       │
   │                                             ├── elementwise sum -> (B, T, D)
   └── positions 0..T-1                          │
          ├── LearnedPositionalEmbedding (L, D) ─┤
          │      -> (1, T, D)                    │
          └── SinusoidalPositionalEmbedding (L, D)
                 -> (1, T, D)
```

Every batch element has the same context length `T`; the positional tensor is `(1, T, D)` and PyTorch broadcasts it across the batch. The token and positional dimensions must match, and both positional modules enforce a maximum context length.

### Token table

`TokenEmbedding` wraps `nn.Embedding(V, D)`. Forward is one indexing operation from `(B, T)` int64 (or int32) ids to `(B, T, D)` float32. Backprop touches only rows used in the forward pass; unused rows receive zero gradient on that step. `nn.Embedding` initializes with `N(0, 1)`, so the module explicitly overwrites the weight with `N(0, 0.02)`. A local `torch.Generator` makes `seed=` reproducible without changing the global RNG.

The token embedding and a future output projection are often tied in production models; this lesson exposes only the input side and does not build an LM head.

### Learned positional table

`LearnedPositionalEmbedding` wraps a second `nn.Embedding(max_context_length, D)`. Its `forward(seq_len)` looks up positions `0..seq_len-1` and returns `(1, seq_len, D)`. The table is a parameter, so it adds `max_context_length * D` parameters beyond the token table. A row that was never created cannot be queried: `forward(L + 1)` raises `ValueError` when the table has `L` rows. Production decoder-only models that use this scheme bake a maximum context into the architecture.

### Sinusoidal positional table

Position `p` and feature `i` produce, with `k = i // 2`:

```text
angle(p, k) = p / 10000 ** (2 * k / D)
table[p, 2k]     = sin(angle(p, k))
table[p, 2k + 1] = cos(angle(p, k))
```

`SinusoidalPositionalEmbedding` builds the full `(max_context_length, D)` table once in float64, casts it to float32, and registers it as a non-persistent buffer. It has zero parameters. For odd `D`, the last even feature has no cosine partner; the implementation slices the angle matrix accordingly.

Wavelengths are `2π * 10000 ** (2k/D)`, a geometric progression from `2π` upward. Lower feature pairs oscillate fastest (fine position); higher pairs oscillate slowest (coarse position). Because each sin/cos pair has norm 1, every row has norm `sqrt(D/2)`, and the cosine similarity for a fixed position offset is the same at every position.

The formula is defined for any position, but this lesson's module precomputes a fixed table and `forward` raises beyond `max_context_length`, exactly like the learned table. The formula's parameter-free extension is not a claim that a trained model generalizes to longer inputs.

### Composition

`EmbeddingComposer(token_embedding, positional_embedding)` validates matching `D`, then in forward:

```python
tokens = token_embedding(token_ids)      # (B, T, D)
positions = positional_embedding(T)      # (1, T, D)
return tokens + positions                # (B, T, D)
```

The same composer accepts either positional implementation. A `T` above the positional table's bound propagates the positional `ValueError`.

## Quickstart

Requires `uv` and Python 3.11 or newer. From the repository root:

```bash
cd 32-token-positional-embeddings
./verify.sh
```

`verify.sh` syncs the frozen lockfile, runs the tests, runs strict mypy, and exercises the demo at `D=64` and `D=32`. It exits 0 and is safe to run repeatedly. The first run downloads the pinned CPU wheels into the ignored `.venv/`; later runs work offline.

Run the demo directly:

```bash
uv run python main.py demo
uv run python main.py demo --dim 32
uv run python main.py demo --vocab-size 32 --context-length 8 --batch-size 8 --seed 7
```

| flag | default | meaning |
| --- | ---: | --- |
| `--vocab-size` | `64` | synthetic vocabulary size `V` |
| `--dim` | `64` | model dimension `D` |
| `--context-length` | `16` | tokens per sequence `T` |
| `--batch-size` | `4` | batch size `B` |
| `--seed` | `1234` | base seed for the token table, learned table, and id generator |

The demo uses synthetic ids; it does not need a tokenizer, corpus, or training run. A non-positive size, non-integer seed, or other invalid configuration exits with status 2 and one `error:` line.

## Offline boundary and optional GPU mode

Everything in this module runs offline on CPU. The tests and demo make no network calls and need no paid provider, remote service, dataset, or tokenizer; token ids are obviously synthetic integers. `verify.sh` downloads pinned packages only when the ignored `.venv/` is absent.

There is no real-provider mode to adapt: this lesson's entire boundary is local tensor math. GPU execution is possible but not exercised or measured here, because the environment has no CUDA device:

```python
import torch
from main import EmbeddingComposer, LearnedPositionalEmbedding, TokenEmbedding

device = torch.device("cuda")
token = TokenEmbedding(vocab_size=64, dim=64, seed=0).to(device)
positions = LearnedPositionalEmbedding(16, 64, seed=1).to(device)
model = EmbeddingComposer(token, positions).to(device)
ids = torch.randint(0, 64, (4, 16), device=device)
output = model(ids)  # (4, 16, 64) on cuda
```

The sinusoidal table is a buffer, so `.to(device)` moves it with the module. GPU timings would not be comparable to the CPU observations below.

## Measured results

Measured locally with Python 3.14.10, `torch==2.15.0+cpu`, `numpy==2.4.4`, and Linux 7.3.5 (x86_64). These are single-run local observations, not benchmark claims; the counts and similarity values are deterministic for the pinned versions and noted seeds.

Default command:

```text
$ uv run python main.py demo
token_ids: shape=(4, 16) dtype=torch.int64
token_embeddings: shape=(4, 16, 64)
learned_positions: shape=(1, 16, 64)
sinusoidal_positions: shape=(1, 16, 64)
composed_inputs: learned=(4, 16, 64) sinusoidal=(4, 16, 64)
parameter_counts:
  token=4096
  learned_positional=1024
  sinusoidal_positional=0
  learned_total=5120
  sinusoidal_total=4096
  overhead=25.0%
composition_sum=ok
batch_broadcast=ok
neighbour_cosine:
  gap=1 sinusoidal=0.9662 learned=-0.0166
  gap=2 sinusoidal=0.8845 learned=0.0158
  gap=4 sinusoidal=0.7479 learned=-0.0106
  gap=8 sinusoidal=0.7004 learned=-0.0165
  gap1_std: sinusoidal=0.000000 learned=0.099895
learned_positional: rejected at T+1 (seq_len 17 exceeds max_context_length 16)
sinusoidal_positional: accepted at T+1 (shape=(1, 17, 64))
sinusoidal_positional: rejected at max+1 (seq_len 65 exceeds max_context_length 64)
length_note: table-level bound only; downstream model length generalization is not tested here
wavelength_bands (dim=64):
  band=0 wavelength=6.28319
  band=1 wavelength=8.37876
  band=2 wavelength=11.1733
  band=4 wavelength=19.8692
  band=8 wavelength=62.8319
  band=31 wavelength=47117.3
```

The learned table here has `16 × 64 = 1024` parameters on top of the `64 × 64 = 4096` token parameters, a 25% increase. The sinusoidal table adds zero. The learned rows are independent random draws at initialization, so adjacent cosine is near zero with visible spread. The sinusoidal gap-1 cosine is `0.9662` at every position; its reported standard deviation rounds to `0.000000` (about `1.2e-7` in float32). Similarity decreases over the displayed gaps but is not monotone at every larger offset because the shortest-wavelength bands dominate some differences.

Changing `D` from 64 to 32:

| `D` | token params | learned pos params | sinusoidal pos params | learned total | sinusoidal total | overhead |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4096 | 1024 | 0 | 5120 | 4096 | 25.0% |
| 32 | 2048 | 512 | 0 | 2560 | 2048 | 25.0% |

| gap | `D=64` sinusoidal | `D=64` learned | `D=32` sinusoidal | `D=32` learned |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9662 | -0.0166 | 0.9571 | -0.0150 |
| 2 | 0.8845 | 0.0158 | 0.8581 | 0.0196 |
| 4 | 0.7479 | -0.0106 | 0.7288 | 0.0610 |
| 8 | 0.7004 | -0.0165 | 0.6576 | -0.0081 |
| gap-1 std | ~1.2e-7 | 0.099895 | ~1.1e-7 | 0.135735 |

Wavelength at a fixed band index grows when `D` shrinks, since the exponent is `2k/D`:

| band `k` | `D=64` wavelength | `D=32` wavelength |
| ---: | ---: | ---: |
| 0 | 6.28319 | 6.28319 |
| 1 | 8.37876 | 11.1733 |
| 2 | 11.1733 | 19.8692 |
| 4 | 19.8692 | 62.8319 |
| 8 | 62.8319 | 628.319 |
| last | 47117.3 (`k=31`) | 35332.9 (`k=15`) |

The `D=32` demo also reports the same learned-table bound at `T=16` and the same sinusoidal acceptance at `T+1` with shape `(1, 17, 32)`; the sinusoidal table has 64 rows either way because its maximum is `4T`.

Verification, measured once after a warm environment:

```text
27 passed in 0.88s
Success: no issues found in 2 source files
verify: ok
```

`./verify.sh` completes in about 4–5 seconds on this CPU when dependencies are already installed. Test runtime varies between runs.

## Design decisions

- **Sum, not concatenation.** The positional vector must not widen the model. Both positional modules return `(1, T, D)` so the batch broadcast is explicit and testable.
- **Seed local, initialize at 0.02.** `nn.Embedding` defaults to `N(0, 1)`, which is too wide for transformer-scale inputs. Both tables use `N(0, 0.02)` with a local generator; global RNG is never touched.
- **Both positional modules are bounded.** The sinusoidal formula can be evaluated at any position, but the lesson precomputes a table and rejects beyond it. This makes the learned-vs-sinusoidal comparison a parameter-count and representation comparison, not an unmeasured extrapolation claim.
- **Float64 construction, float32 storage.** The sinusoidal table is computed in float64 to keep the angle exponent accurate, then stored in the module's float32 dtype. Tests compare against a feature-by-feature float64 reference at `atol=1e-6`.
- **Non-persistent buffer.** The sinusoidal table is derived deterministically from `(max_context_length, D)`; storing it in every checkpoint would duplicate derived state. It still moves with `.to(device)`.
- **Strict input validation.** Token ids must be a 2D int32/int64 tensor; positional calls require positive integer lengths within the table; mismatched dimensions fail at composer construction.
- **Synthetic ids.** The lesson is about lookup tables, not tokenization. Using `torch.randint` avoids a corpus dependency and makes every diagnostic reproducible from `--seed`.
- **No training, tying, attention, RoPE, or ALiBi.** Those belong to later lessons. The input embedding and a future output projection are not tied here.

## Limitations

- The module does not train the embeddings. There is no loss, attention block, LM head, or trained-model length-generalization experiment. The length-bound demo is table-level only.
- Both positional tables are bounded by `max_context_length`. The sinusoidal implementation allocates `O(L * D)` storage even though its formula is parameter-free.
- Only dense 2D id batches are supported; there is no padding mask, ragged batch, or variable context length.
- Computation is CPU float32 in the measured configuration. GPU execution is documented but not run; no GPU timings are claimed.
- Token ids are synthetic. No tokenizer, corpus, or real vocabulary is included, so these token counts are not comparable to a production tokenizer's vocabulary.
- Weight tying, rotary position embeddings, and ALiBi are not implemented. RoPE and ALiBi apply at attention projections rather than at this embedding stage.
- Deterministic values are guaranteed within the pinned environment. PyTorch does not guarantee reproducibility across releases, platforms, or CPU/GPU execution.
- The adjacent-position cosine diagnostic is a local geometric observation at initialization, not evidence of downstream language-model quality.

## References

1. AI Engineering From Scratch, [Token and Positional Embeddings capstone](https://aiengineeringfromscratch.com/learn/capstone-projects/token-positional-embeddings) — shape contract, token/learned/sinusoidal modules, composition, and the contrastive diagnostics.
2. Ashish Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — sinusoidal formula, summing positional encodings with token embeddings, the near-identical learned variant, relative-offset rationale, and weight sharing.
3. PyTorch, [`torch.nn.Embedding` documentation](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Embedding.html) — lookup-table semantics, input/output shape, and the default `N(0, 1)` initialization that this module replaces.
4. Hugging Face, [GPT-2 model documentation](https://huggingface.co/docs/transformers/model_doc/gpt2) — absolute learned position embeddings, `n_positions=1024`, `initializer_range=0.02`, and tied word embeddings as a production contrast.
5. Amirhossein Kazemnejad et al., [The Impact of Positional Encoding on Length Generalization in Transformers](https://arxiv.org/abs/2305.19466) — systematic study finding explicit position encodings (learned APE, T5 relative, ALiBi, Rotary) poorly suited to downstream length generalization, with NoPE outperforming them; used to frame the limitation, not as a sinusoidal benchmark.

## License

MIT License, Copyright (c) 2026 Pedro Monteiro. See [LICENSE](LICENSE).
