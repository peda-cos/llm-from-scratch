# Loading Pretrained Weights

## Overview

Training a 124M-parameter GPT-2 is a budget decision; loading a published checkpoint should not be. This module loads GPT-2-style safetensors weights into a small, self-contained replica of the lesson-35 GPT model, without `transformers` and without network access at runtime.

The hard part is not the copy. Published GPT-2 tensors use a different naming convention (`wte`, `h.N.attn.c_attn`, `mlp.c_fc`), a transposed linear layout, and an LM head that is absent because it is tied to the token embedding. The loader in this module reconciles all three explicitly:

- **Names**: a fixed `make_pretrained_to_local(num_layers)` map covers every model parameter.
- **Shapes**: every tensor is checked against the expected pretrained orientation before assignment.
- **Layout**: attention and MLP projection weights are transposed from the published Conv1D layout into `nn.Linear` layout.
- **Coverage**: a `LoadReport` records loaded, missing, unexpected, ignored, and shape-mismatched tensors.
- **Safety**: any missing, unexpected, or mismatched tensor aborts the load before a single parameter is touched.

The demo writes a GPT-2-format stub checkpoint, samples tokens from a randomly initialized model, loads the stub, samples again, and shows that the model changed. It also corrupts one tensor in a second fixture and proves the loader refuses it without mutating the model.

## Architecture

```text
GPT-2 names                         local names
wte.weight ....................... tok_embed.weight
wpe.weight ....................... pos_embed.weight
h.N.attn.c_attn.{weight,bias} .... blocks.N.attn.qkv.{weight,bias}
h.N.attn.c_proj.{weight,bias} .... blocks.N.attn.out_proj.{weight,bias}
h.N.mlp.c_fc.{weight,bias} ....... blocks.N.mlp.fc1.{weight,bias}
h.N.mlp.c_proj.{weight,bias} ..... blocks.N.mlp.fc2.{weight,bias}
h.N.ln_1.{weight,bias} ........... blocks.N.ln1.{scale,shift}
h.N.ln_2.{weight,bias} ........... blocks.N.ln2.{scale,shift}
ln_f.{weight,bias} ............... final_ln.{scale,shift}

safetensors file
      │
      ▼
safe_open + get_slice             read names and shapes only
      │
      ▼
make_pretrained_to_local          canonicalize optional "transformer." prefix
      │
      ├── h.N.attn.bias ────────── ignore (HF causal-mask buffer)
      ├── lm_head.weight ───────── ignore (tied-head alias)
      │
      ▼
shape preflight                   source shape == reversed target shape?
      │                              for qkv / out_proj / fc1 / fc2 weights
      ├── mismatch ─────────────── LoadReport.shape_mismatch
      ├── unmapped ─────────────── LoadReport.unexpected
      └── absent ───────────────── LoadReport.missing
      │
      ▼
CheckpointLoadError?              yes: abort before assignment
      │ no
      ▼
torch.no_grad: parameter.copy_    transpose Conv1D weights, cast dtype
      │
      ▼
LoadReport.loaded                 one entry per assignment
```

The local model is mathematically the lesson-35 decoder: learned token and position embeddings, pre-LN transformer blocks, fused QKV causal self-attention, GELU feed-forward layers, a final LayerNorm, two dropout sites per block, and a tied LM head. Its attributes are named to match the lesson's local naming table, so the state-dict paths are the mapping targets directly.

## Quickstart

From this module's directory:

```bash
uv sync --frozen --all-groups
uv run python code/main.py
```

The first run writes two generated fixtures under `outputs/`:

- `gpt2-stub.safetensors` — a valid 12-block, `d_model=192` GPT-2-format checkpoint;
- `gpt2-stub-bad.safetensors` — the same checkpoint with one attention weight deliberately shaped incorrectly.

Both are ignored by git and total about 116 MB. Re-running reuses them; pass `--regenerate` after changing shape flags or the stub seed.

Run the full gate:

```bash
./verify.sh
```

It installs the pinned environment, runs the tests and strict typecheck, then runs the demo.

For a fast CPU run:

```bash
uv run python code/main.py \
  --vocab-size 64 --context-length 16 --d-model 16 \
  --num-heads 2 --num-layers 2 --mlp-expansion 2 \
  --max-new-tokens 4 --top-k 8 --fixture-dir /tmp/lpw-tiny
```

## The loader contract

```python
report = load_safetensors(model, "gpt2-stub.safetensors")
print(report.summary())
```

`load_safetensors(model, path)` is a two-phase operation:

1. **Preflight.** It opens the file with `safetensors.safe_open` and inspects names and shapes with `get_slice`; no tensor data is copied. It canonicalizes an optional leading `transformer.` prefix, maps each canonical name through `make_pretrained_to_local(model.config.num_layers)`, and computes the expected source shape. For projection weights the expected source shape is the reversed local shape; for embeddings, biases, and LayerNorm parameters it is identical.
2. **Assignment.** Only if the report has no `missing`, `unexpected`, or `shape_mismatch` entries does it open a `torch.no_grad()` block, fetch each validated tensor, transpose Conv1D-layout weights, and call `parameter.copy_(...)`. Copying into the existing parameter preserves object identity, so the tied `lm_head.weight` and `tok_embed.weight` stay aliased.

On failure it raises `CheckpointLoadError` with the complete `LoadReport` attached. `report.loaded` is empty because no assignment happened. The error message names every missing, unexpected, and mismatched tensor.

Known real-checkpoint entries are handled explicitly:

- `h.<N>.attn.bias` buffers (shape `(1, 1, context, context)`) are ignored; they are HF causal-mask buffers, not model parameters.
- `lm_head.weight` is ignored because GPT-2 ties it to `wte`; loading `wte` updates the head. A checkpoint that includes the alias does not produce a spurious "unexpected" report.
- Bias-less and untied configurations are rejected with `ValueError`: the fixed GPT-2 map includes biases, and published GPT-2 files do not contain a separate head.

The loader is strict by design. A checkpoint for 12 layers loaded into a 2-layer model reports extra block tensors as unexpected and the missing local blocks as missing; neither half-load is possible.

## Name map

`make_pretrained_to_local(num_layers)` returns `12 * num_layers + 4` fresh entries. The four global entries are `wte`, `wpe`, `ln_f.weight`, and `ln_f.bias`. Each block contributes:

| Pretrained | Local |
|---|---|
| `h.N.ln_1.weight` | `blocks.N.ln1.scale` |
| `h.N.ln_1.bias` | `blocks.N.ln1.shift` |
| `h.N.attn.c_attn.weight` | `blocks.N.attn.qkv.weight` |
| `h.N.attn.c_attn.bias` | `blocks.N.attn.qkv.bias` |
| `h.N.attn.c_proj.weight` | `blocks.N.attn.out_proj.weight` |
| `h.N.attn.c_proj.bias` | `blocks.N.attn.out_proj.bias` |
| `h.N.ln_2.weight` | `blocks.N.ln2.scale` |
| `h.N.ln_2.bias` | `blocks.N.ln2.shift` |
| `h.N.mlp.c_fc.weight` | `blocks.N.mlp.fc1.weight` |
| `h.N.mlp.c_fc.bias` | `blocks.N.mlp.fc1.bias` |
| `h.N.mlp.c_proj.weight` | `blocks.N.mlp.fc2.weight` |
| `h.N.mlp.c_proj.bias` | `blocks.N.mlp.fc2.bias` |

The transpose rule is derived from the local target, not guessed from matching shapes:

```text
blocks.N.attn.qkv.weight       published (d, 3d)  -> local (3d, d)
blocks.N.attn.out_proj.weight  published (d, d)   -> local (d, d), still transposed
blocks.N.mlp.fc1.weight        published (d, 4d)  -> local (4d, d)
blocks.N.mlp.fc2.weight        published (4d, d)  -> local (d, 4d)
```

The square attention projection is transposed too. Shape equality alone cannot detect this case, which is why the transpose decision is attached to the target name.

## Stub fixture

`make_stub_safetensors(path, config, *, seed, corrupt_name)` builds a checkpoint with the exact published GPT-2 naming convention and the tensor layout described above:

- token embeddings `(vocab_size, d_model)`;
- position embeddings `(context_length, d_model)`;
- fused QKV `(d_model, 3*d_model)`;
- attention output `(d_model, d_model)`;
- MLP `(d_model, expansion*d_model)` and `(expansion*d_model, d_model)`;
- LayerNorm scales near 1 and shifts near 0;
- biases in the published positions.

Weights are deterministic from the seed. The stub uses a deliberately different distribution from the model's initialization (`N(0, 0.05)` token embeddings, `N(0, 0.03)` projections, and jittered LayerNorm scales), so a successful load visibly changes the model. A `corrupt_name` argument gives exactly one tensor a wrong shape for the failure-path demo.

The default stub configuration is a 12-block model with `d_model=192`, 6 heads, GPT-2 vocabulary 50,257, and context 1,024. It has 15,184,704 parameters and writes 60,751,656 bytes. The bad fixture writes 60,530,464 bytes. Neither file is tracked.

## Measured results

Measured locally on this CPU with the pinned environment; these are single deterministic demo runs, not a benchmark.

```text
fixture: path=outputs/gpt2-stub.safetensors tensors=148 parameters=15184704 bytes=60751656
model: parameters=15184704 d_model=192 num_heads=6 num_layers=12 context_length=1024 vocab_size=50257
prompt_ids=[1, 2, 3, 4]
report: loaded=148 missing=0 unexpected=0 ignored=0 shape_mismatch=0
continuation_before_ids=[1, 2, 3, 4, 42701, 42701, 29187, 14924, 35226, 28329, 10072, 31360]
continuation_after_ids=[1, 2, 3, 4, 441, 27327, 15997, 26271, 20330, 74, 36483, 48262]
logits: max_abs_delta=3.202385
check: tokens_changed=true
load_error: checkpoint validation failed: 0 missing, 0 unexpected, 1 shape mismatch
report: loaded=0 missing=0 unexpected=0 ignored=0 shape_mismatch=1
mismatch: source=h.0.attn.c_attn.weight source_shape=(192, 288) expected_source_shape=(192, 576) target=blocks.0.attn.qkv.weight target_shape=(576, 192)
check: model_unchanged=true
summary: loaded=148 ignored=0 tokens_changed=true model_unchanged=true
```

- Fresh fixture creation + full demo: 2.198 s.
- Reusing the fixtures: 1.790 s.
- Test suite: `32 passed` in 1.84 s.
- Strict mypy: no issues in six source files.
- The 148 loaded entries are not shown above; the demo prints one line per assignment.

## Optional real GPT-2 checkpoint

The loader accepts the published `gpt2` safetensors file without code changes. The checkpoint download is deliberately outside the module; runtime code never touches the network.

```bash
curl -L --fail -o /tmp/gpt2.safetensors \
  https://huggingface.co/gpt2/resolve/main/model.safetensors

uv run python - <<'PY'
import sys
sys.path.insert(0, "code")

from main import GPTConfig, GPTModel, load_safetensors

model = GPTModel(GPTConfig(), seed=0)
report = load_safetensors(model, "/tmp/gpt2.safetensors")
print(report.summary())
print("tied_head:", model.lm_head.weight is model.tok_embed.weight)
PY
```

Expected shape of the report: `loaded=148 missing=0 unexpected=0 ignored=12 shape_mismatch=0`. The twelve ignored tensors are `h.<N>.attn.bias` causal-mask buffers; the file has 160 tensors total and no `lm_head.weight`. The model is about 0.5 GB in float32 and needs roughly that much free RAM in addition to the mmap-backed file. This exact load was verified locally against the published `model.safetensors` (`loaded=148`, `ignored=12`, tied head still shared).

This is a structural and numerical weight load, not a tokenizer or chat product. To generate text from real GPT-2, pair the loader with GPT-2 BPE tokenization elsewhere. The module's `generate` operates on token IDs.

## Tests

`tests/` contains 32 tests across five files:

- `test_model.py`: 124,439,808-parameter default count, closed-form tiny count, local parameter names, parameter-level weight tying, deterministic generation, context sliding, and argument validation.
- `test_name_map.py`: map size, exact coverage of every model parameter, spot checks for every mapping family, fresh-dict behavior, and invalid layer counts.
- `test_stub.py`: exact pretrained keys and shapes, metadata, deterministic values, one-tensor corruption, parent-directory creation, and argument validation.
- `test_loader.py`: exact source-to-target copies, transpose coverage including square projections, tied-head alias, shape/missing/unexpected aborts with full parameter snapshots, optional `transformer.` prefix, ignored attention-mask buffers and head alias, unsupported configurations, and missing files.
- `test_demo.py`: the printed CLI contract, before/after difference, failure report, model-unchanged check, determinism, fixture reuse, regeneration, and invalid arguments.

Tests use tiny configs and `tmp_path`; they never create the 60 MB fixture and never use the network.

## Design decisions

- **Local names are the contract.** The model uses the lesson's local names (`tok_embed`, `blocks.N.attn.qkv`, `final_ln.scale`) rather than the sibling lesson-35 attribute spellings, so the map is a literal dict and tests can compare state-dict paths directly.
- **Preflight before assignment.** Names and shapes are checked in a first pass with `safe_open.get_slice`; tensors are fetched only after the report is known to be clean. This makes a mismatch a safe no-op rather than a half-loaded model.
- **Transpose by target identity.** The loader knows the four projection suffix families. It does not transpose any 2-D tensor whose shape happens to be reversed.
- **Copy, do not replace.** `parameter.copy_` under `torch.no_grad()` keeps optimizer and tying semantics intact. Replacing `lm_head.weight` with a fresh parameter would silently double the parameter count.
- **Explicit ignores for real files.** HF causal-mask buffers and the tied-head alias are recognized by name. Everything else is matched or reported.
- **Deterministic fixtures.** A seeded `torch.Generator` drives every stub tensor; a fixed sample seed makes the demo repeatable. No timestamps or random global state affect output.
- **Fail loud.** Missing, unexpected, and mismatched tensors raise `CheckpointLoadError` with counts and per-tensor details. The demo returns non-zero if loading changes nothing or mutates the model on the bad path.

## Limitations

- The stub is random, not trained. Its continuations are token IDs with no linguistic meaning.
- No tokenizer is included; GPT-2 BPE text decoding is out of scope for this lesson.
- The replica uses exact `nn.GELU` (erf) as in lesson 35. Published HF GPT-2 declares `gelu_new` (tanh approximation), so real weights load and run but do not reproduce HF logits bit-for-bit.
- The name map assumes GPT-2's bias-enabled, tied-head layout. LLaMA/RMSNorm variants and untied heads are rejected, not silently coerced.
- Source dtype is not validated; `copy_` casts each tensor to the model parameter dtype. A dtype-target argument, expected-layer-count check, export path, and LLaMA map are source exercises and are not implemented.
- Only CPU execution was measured. No GPU path was benchmarked.
- Fixtures are reused by default. If shape flags change, pass `--regenerate`; otherwise the existing file is validated and may fail with a clear mismatch rather than being silently replaced.

## References

1. AI Engineering From Scratch, [Loading Pretrained Weights capstone](https://aiengineeringfromscratch.com/learn/capstone-projects/loading-pretrained-weights) — task contract, naming tables, loader requirements, and stub-fixture design.
2. Hugging Face, [safetensors](https://github.com/huggingface/safetensors) and its [Python API stubs](https://github.com/huggingface/safetensors/blob/main/bindings/python/py_src/safetensors/__init__.pyi) — `safe_open`, `get_slice`, `get_tensor`, metadata, and save behavior.
3. Hugging Face, [gpt2 model repository](https://huggingface.co/gpt2) — `config.json` (`n_embd=768`, `n_layer=12`, `n_head=12`, `activation_function=gelu_new`) and the published `model.safetensors` header (160 names, Conv1D-oriented weights, `h.N.attn.bias` buffers, no `lm_head.weight`).
4. Hugging Face Transformers, [`Conv1D` in pytorch_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py) and [`modeling_gpt2.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py) — the `(in_features, out_features)` Conv1D weight layout and the causal-mask buffer that motivate transpose-on-load and explicit ignores.
5. PyTorch, [`Tensor.copy_`](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.copy_.html) and [`torch.no_grad`](https://docs.pytorch.org/docs/2.15/generated/torch.no_grad.html) — copy semantics across dtype/device and gradient-disabled assignment.

## License

MIT License, Copyright (c) 2026 Pedro Monteiro. See [LICENSE](LICENSE).
