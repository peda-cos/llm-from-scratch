# Training Loop and Evaluation

## Overview

A training loop that does not measure is a loop that lies. This module trains a compact GPT-2-style decoder from scratch and measures the run three ways:

- cross entropy on the current training batch at every step;
- mean cross entropy on a fixed held-out slice every `eval_every` steps;
- a greedy continuation from a fixed prompt at the same probe points.

Every training step also appends one flushed JSONL record to `outputs/losses.jsonl`, so the run leaves a durable artifact that can be reloaded, compared, and plotted without a checkpoint format.

The model is deliberately small: 26,240 parameters with an 8-token toy vocabulary, 16-token context, and two pre-LN blocks. It trains on a deterministic periodic phrase on CPU. The demo is a mechanics fixture, not a language model; the loop is the deliverable.

## Architecture

```text
toy token ids (1-D)
   │
   └── make_batches: non-overlapping windows, targets shifted by one
          │
          ├── inputs  (B, T)
          └── targets (B, T)
                 │
                 ▼
          GPTModel.forward
          embeddings + positions
          2 × pre-LN block
          final LayerNorm
          tied LM head
                 │
                 ▼
          logits (B, T, vocab)
                 │
                 ▼
          calc_loss_batch: reshape (B*T, vocab) vs (B*T,)
          cross entropy -> backward
                 │
                 ▼
          clip_grad_norm_(max_norm=1.0, error_if_nonfinite=True)
                 │
                 ▼
          AdamW
          ├── matrix weights: weight_decay=0.1
          └── LayerNorm/bias: weight_decay=0.0
                 │
                 ▼
          cosine_with_warmup -> set every group's LR
                 │
                 ├── append {"step", "train_loss", "lr"} and flush
                 └── probe? eval fixed val slice; greedy sample from prompt
```

### Loss alignment

The batch builder emits `inputs = tokens[i:i+T]` and `targets = tokens[i+1:i+T+1]`. The model predicts the next token at every position, so the flattened cross entropy is `(B*T, vocab)` against `(B*T,)`. Targets are shifted exactly once, in `make_batches`; `calc_loss_batch` never shifts again. Tests pin the first and last window boundaries and compare the loss against a manual `torch.nn.functional.cross_entropy`.

### Decay split and schedule

`build_param_groups` walks unique parameters (the tied embedding/head appears once) and assigns every tensor with `ndim >= 2` to the decay group and every 1-D scale/bias to the no-decay group. The schedule is linear warmup from 0 to `max_lr`, then cosine decay to a floor:

```text
lr(step) = max_lr * step / warmup_steps                       step < warmup
         = min_lr + (max_lr-min_lr)/2 * (1+cos(pi*p))         warmup <= step < total
         = min_lr                                             step >= total
p = (step - warmup_steps) / (total_steps - warmup_steps)
```

The demo uses a 16-step warmup, 160 total steps, and `min_lr = 10% * max_lr`.

### Fixed held-out evaluation

`make_batches` slices the 10% validation tail once at script start. `evaluate_model` consumes exactly the first `eval_batches` entries, not a random draw, so two runs compare the model rather than the batch shuffle. It runs under `torch.no_grad()`, calls `model.eval()`, and restores the previous training mode in a `finally` block.

## Quickstart

Requires `uv` and Python 3.11 or newer. From the repository root:

```bash
cd 36-training-loop-eval
./verify.sh
```

`verify.sh` installs the pinned CPU wheels into the ignored `.venv/`, runs pytest, runs strict mypy, runs a short CLI smoke test, and prints `verify: ok`. It is safe to run repeatedly. The first run downloads dependencies; later runs reuse the local environment.

Run the full demo:

```bash
uv run --frozen python code/main.py
```

It prints a per-step loss line, an eval and a sample line every 40 steps, and writes `outputs/losses.jsonl`. The file is intentionally ignored by git: it is a run artifact, not source. Run from the module root so the default relative path lands in `outputs/`. If torch is already installed in the active environment, `python3 code/main.py` works too.

Inspect the JSONL with only the standard library:

```bash
uv run --frozen python - <<'PY'
import json
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path("outputs/losses.jsonl").read_text().splitlines()
]
print(len(rows), rows[0], rows[-1])
PY
```

| CLI flag | default | meaning |
| --- | ---: | --- |
| `--steps` | 160 | optimization steps |
| `--eval-every` | 40 | probe cadence; the final step is always probed |
| `--eval-batches` | 4 | fixed validation batches per probe |
| `--batch-size` | 8 | windows per training/eval batch |
| `--output` | `outputs/losses.jsonl` | JSONL path; parent directories are created |
| `--seed` | 1234 | model init, batch shuffle, and demo sampling seed |
| `--max-new-tokens` | 12 | tokens added by each qualitative probe |

Invalid values print one `error:` line to stderr and exit with status 2.

## Function map

| function | contract |
| --- | --- |
| `make_batches(token_ids, batch_size, context_length)` | non-overlapping full windows; inputs at `i..i+T-1`, targets at `i+1..i+T`; incomplete tail dropped |
| `calc_loss_batch(model, inputs, targets)` | forward, validate `(B,T,V)` logits, flattened cross entropy against pre-shifted targets |
| `evaluate_model(model, val_loader, max_batches)` | no-grad mean over exactly `max_batches` fixed entries; restores mode |
| `generate_and_print_sample(model, prompt, max_new_tokens)` | greedy continuation by default; prints `sample ids=[...]`; returns prompt + new ids |
| `build_param_groups(model, weight_decay)` | two AdamW groups: matrix parameters with decay, 1-D parameters without |
| `cosine_with_warmup(step, warmup_steps, total_steps, max_lr, min_lr)` | 0-based warmup + cosine LR with an exact floor |
| `train(model, train_batches, val_batches, config, log_path)` | the measured loop; writes/flushes JSONL and returns a `TrainResult` |

## Offline boundary and optional GPU mode

Everything in the tests and demo runs offline on CPU. Token ids come from a local deterministic phrase, the model is initialized from local seeded generators, and there is no tokenizer download, dataset fetch, paid API, provider, or network call at runtime. The only network access is the one-time `uv sync` dependency installation.

There is no real-provider mode in this lesson: it trains from scratch. To use real text, replace `make_synthetic_corpus()` with a token tensor from your own tokenizer and keep the same batching, evaluation, and logging path.

The model and batches are ordinary PyTorch tensors, so a GPU path is mechanical:

```python
import sys

import torch

sys.path.insert(0, "code")
from main import (
    GPTConfig, GPTModel, TrainConfig, make_batches,
    make_synthetic_corpus, train,
)

device = torch.device("cuda")
corpus = make_synthetic_corpus()
split = int(corpus.numel() * 0.9)
train_batches = [
    (x.to(device), y.to(device))
    for x, y in make_batches(corpus[:split], 8, 16)
]
val_batches = [
    (x.to(device), y.to(device))
    for x, y in make_batches(corpus[split:], 8, 16)
]
model = GPTModel(GPTConfig(), seed=0).to(device)
train(
    model,
    train_batches,
    val_batches,
    TrainConfig(steps=20, eval_every=10, warmup_steps=2),
    "outputs/gpu-losses.jsonl",
)
```

GPU execution was not run in this environment, so no GPU timing or numerical result is claimed. `train` places the probe prompt on the same device as the first training batch.

## Measured results

Measured locally with Python 3.14.10, `torch==2.15.0+cpu`, `uv 0.13.9`, Linux x86_64, CPU only. These are single-run observations for a fixed seed, not benchmark claims. PyTorch does not guarantee identical floating-point results across releases, platforms, or CPU/GPU execution.

### Verification

```text
$ ./verify.sh
...
25 passed, 1 warning in 2.80s
Success: no issues found in 7 source files
verify: ok
```

A second consecutive run reported `25 passed` and `Success: no issues found in 7 source files`; the two verification runs took about 6.8 s and 6.5 s wall time. The warning is PyTorch's optional NumPy probe; NumPy is intentionally not a dependency. The 25 tests cover attention causality, tied parameter storage, greedy/top-k generation and mode restoration, shifted batching, manual cross-entropy equivalence, gradient flow, parameter-group classification, fixed-batch no-grad evaluation, schedule boundaries and monotonicity, JSONL schema and LR integration, deterministic learning, exact minimum-window batching, probe cadence, overwrite behavior, CLI validation, and corpus periodicity.

### Full demo

Command: `uv run --frozen python code/main.py`

```text
training loop and evaluation
corpus: toy_phrase=[1, 2, 3, 4, 1, 5] words="the cat sat on the mat" vocab=8
data: train_tokens=16200 train_batches=126 val_tokens=1800 val_batches=14
model: parameters=26240 context_length=16 d_model=32 num_heads=4 num_layers=2
train: steps=160 batch_size=8 eval_every=40 eval_batches=4 warmup_steps=16 max_lr=0.003000 min_lr=0.000300 weight_decay=0.1 grad_clip=1.0
step=001 lr=0.000188 train_loss=2.1244 grad_norm=3.598
step=002 lr=0.000375 train_loss=2.0942 grad_norm=3.578
step=003 lr=0.000563 train_loss=2.0303 grad_norm=3.093
step=004 lr=0.000750 train_loss=1.9539 grad_norm=2.451
step=005 lr=0.000937 train_loss=1.8795 grad_norm=1.841
...
eval step=040 loss=0.2865
sample step=040 ids=[1, 2, 3, 4, 1, 5, 1, 2, 3, 4, 1, 5, 1, 2, 3, 4]
eval step=080 loss=0.0629
sample step=080 ids=[1, 2, 3, 4, 1, 5, 1, 2, 3, 4, 1, 5, 1, 2, 3, 4]
eval step=120 loss=0.0493
sample step=120 ids=[1, 2, 3, 4, 1, 5, 1, 2, 3, 4, 1, 5, 1, 2, 3, 4]
eval step=160 loss=0.0467
sample step=160 ids=[1, 2, 3, 4, 1, 5, 1, 2, 3, 4, 1, 5, 1, 2, 3, 4]
summary: first_train_loss=2.1244 final_train_loss=0.0494 best_eval_loss=0.0467 eval_points=4 elapsed_s=1.200
log: outputs/losses.jsonl records=160
```

The process took 3.386 s wall time including Python and torch startup; `elapsed_s=1.200` measures only the training loop. The model learns the six-token phrase: the final 16-token greedy sample is `1,2,3,4,1,5` repeated, including all 12 generated tokens. This is evidence that the loop optimizes and that the probe is wired to the trained model, not evidence of language ability.

### JSONL artifact

```text
$ wc -l outputs/losses.jsonl
160 outputs/losses.jsonl
$ head -2 outputs/losses.jsonl
{"step": 1, "train_loss": 2.1244168281555176, "lr": 0.0001875}
{"step": 2, "train_loss": 2.0941803455352783, "lr": 0.000375}
$ tail -2 outputs/losses.jsonl
{"step": 159, "train_loss": 0.04443052038550377, "lr": 0.00030032126344212265}
{"step": 160, "train_loss": 0.04940225929021835, "lr": 0.0003}
```

Step 160 uses exactly the 10% floor. Every line is independently parseable; the log is flushed per step, so a crash leaves every completed record readable.

## Design decisions

- **Non-overlapping windows, shifted once.** `make_batches` owns the shift; `calc_loss_batch` is a pure forward + cross entropy. This makes the highest-risk bug directly testable.
- **Tied embedding/head with identity dedupe.** The LM head shares the token embedding parameter. `count_parameters` and `build_param_groups` both dedupe by object identity so the tied matrix receives one optimizer update and one decay coefficient.
- **Decay by shape, not name.** Matrix-shaped tensors get decay; LayerNorm scales/biases and linear biases do not. This avoids coupling the split to module names.
- **Warmup and cosine boundaries are explicit.** Step 0 is 0 during warmup, the warmup boundary is exactly `max_lr`, and `total_steps` onward is exactly `min_lr`. The demo floor is 10% of the peak.
- **Fixed eval batches.** The validation slice is batched once and evaluated in order, never resampled. Eval loss is comparable between runs of the same config.
- **Flushed JSONL with exactly three keys.** `{"step", "train_loss", "lr"}` is intentionally narrow. Eval losses and samples are returned in `TrainResult` and printed; expanding the log schema would make the artifact less stable.
- **Probe on every `eval_every` step and on the final step.** A run that ends between multiples still gets a final held-out loss and sample. A coincident final multiple is recorded once.
- **Greedy qualitative probe.** Greedy decoding makes repetition and collapse visible without sampling noise. `generate` still supports temperature and exact top-k for callers that want stochastic samples.
- **Local RNG only.** Model init, layer init, and train batch shuffling use local `torch.Generator` instances. The demo uses dropout 0.0, so a run is reproducible from the seed without mutating global RNG state.
- **Fresh log, no silent append.** A new run opens the log with `"w"`. This lesson does not resume from a JSONL step; adding checkpoint/resume is a separate exercise.
- **Synthetic corpus.** A periodic toy phrase gives a learnable, deterministic target without a tokenizer, dataset, or external service. The cost is that a low loss here proves loop mechanics, not language modeling.

## Limitations

- The demo's periodic phrase is not natural language. The reported 0.05 loss reflects memorizing a six-token cycle, not a pretrained or useful LM.
- There is no tokenizer, real dataset, checkpoint save/load, resume path, throughput/MFU logging, gradient accumulation, mixed precision, or distributed training.
- Only CPU execution was measured. The GPU snippet is mechanical and unverified here.
- `train` seeds its shuffle generator but not torch's global RNG. With nonzero dropout, dropout masks are not reproducible from `TrainConfig.seed`; the demo keeps dropout at 0.
- The JSONL log stores training loss and LR only. Eval losses and samples are printed and returned, not written into the log.
- Generation recomputes the active window each token; there is no KV cache.
- The logged timings are one-shot observations and will vary with CPU scheduling and PyTorch build.
- PyTorch emits a harmless `Failed to initialize NumPy` warning because NumPy is intentionally absent.
- The demo writes relative to the current directory. Use `--output` or run from the module root.
- Eval loss is a scalar cross entropy over a toy distribution. It says nothing about calibration, downstream tasks, or text quality.

## References

1. AI Engineering From Scratch, [Training Loop and Evaluation capstone](https://aiengineeringfromscratch.com/learn/capstone-projects/training-loop-eval) — task contract, required helpers, JSONL schema, decay split, and probe cadence.
2. Andrej Karpathy, [nanoGPT `train.py`](https://github.com/karpathy/nanoGPT/blob/master/train.py) — weight-decay parameter split, warmup/cosine `get_lr`, gradient clipping before the optimizer step, held-out estimation, and training logging.
3. Hugging Face Transformers, [`optimization.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py) — cosine-with-warmup lambda and its boundary behavior.
4. Ilya Loshchilov and Frank Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — decoupled weight decay and the rationale for treating decay separately from adaptive updates.
5. PyTorch 2.15, [`torch.optim.AdamW`](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html) — AdamW parameter groups, default betas/eps, and per-group weight decay semantics.

## License

MIT License, Copyright (c) 2026 Pedro Monteiro. See [LICENSE](LICENSE).
