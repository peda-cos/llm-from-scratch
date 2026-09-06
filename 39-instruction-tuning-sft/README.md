# Instruction Tuning by Supervised Fine-Tuning

## Overview

A pretrained causal language model can extend a sequence, but next-token
prediction alone does not teach it the instruction/response contract. This
module builds the smallest version of supervised fine-tuning (SFT) that does:
every example is serialized as

```text
[INST] instruction [RESP] response <newline>
```

and a tiny transformer is trained to predict response bytes conditioned on the
instruction. The cross-entropy loss is masked so instruction tokens, the
`[RESP]` boundary, and padding contribute nothing (`ignore_index=-100`); only
response bytes and the terminating newline are scored.

The module is self-contained, CPU-only, deterministic, and independent of the
other track modules. It trains a 253,152-parameter transformer from random
initialization on a 200-pair generated fixture (160 train, 40 held out),
evaluates held-out exact-match, and prints per-category results.

## Architecture

```text
instruction + response
        │
        ▼
InstructionTokenizer ── [INST] + UTF-8 bytes + [RESP] + response bytes + \n
        │
        ▼
SFTDataset ── full ids and labels; labels[:response_start] = -100
        │
        ▼
sft_collate ── right-pad ids with PAD; labels with -100
        │
        ▼
TinyGPT ── token/position embeddings → 2 × pre-norm Block → final norm
        │       └── tied LM head: embeddingᵀ
        ▼
logits[:, :-1] vs labels[:, 1:]
        │
        ▼
cross_entropy(..., ignore_index=-100) → Adam
        │
        ▼
generate([INST] instruction [RESP]) ── greedy or sampled
        │
        ▼
exact_match(normalize(prediction), normalize(gold))
```

The body has hidden width 96, four attention heads, two pre-norm transformer
blocks, a 64-token context, and no dropout by default. The language-model head
is tied to the token embedding, so it adds no parameters. All linear layers in
attention and the MLP are bias-free; the only biases are in LayerNorm.

## The SFT objective

For a sequence

```text
tokens:  [INST]  h   i  [RESP]  y   o  \n
labels:  -100   -100 -100 -100   121 111  10
```

the loss compares `logits[:, :-1]` with `labels[:, 1:]`. The `[RESP]` position
therefore predicts the first response byte (`y`), and the last response byte
predicts the terminator. The label stored at the `[RESP]` index is `-100`: the
model is not trained to predict the boundary token from the last instruction
byte. The full sequence is still the model input, so response tokens attend to
the instruction; only their direct loss targets are excluded.

This is the response-only or "completion-only" objective. It differs from full
sequence language modeling in two ways: the model does not spend capacity
reconstructing user-provided instructions, and the mean loss divides by the
number of response targets rather than by every non-padding token. PyTorch's
`cross_entropy` excludes `ignore_index` targets from both the numerator and the
mean denominator.

### Why the newline terminator?

The fixture has no fourth reserved EOS token. A causal model trained only on
`[INST] instruction [RESP] response` has no supervised signal for stopping
after the final response byte. Each response therefore ends with a newline
byte; it lies inside the response region, is supervised, and is emitted
naturally at generation time. Generation stops on newline (or on a control id,
or on two consecutive sentence-ending bytes), and the newline is not part of
the decoded response. Exact-match normalization strips it anyway. A production
pipeline would normally reserve an explicit EOS token and leave its label
unmasked; this printable terminator is the local equivalent.

## Data fixture

`make_dataset(seed)` builds exactly 200 unique pairs across six task types:

| Category | Bases | Templates | Pairs | Held-out |
|---|---|---:|---:|---:|
| capitals | France, Spain, Italy | 14/13/13 | 40 | 8 |
| arithmetic | 2 + 3, 5 * 4, 9 - 7 | 10 each | 30 | 6 |
| lists | colors, planets, vowels | 10 each | 30 | 6 |
| summaries | three short texts | 10 each | 30 | 6 |
| code | print, sort, reverse | 10 each | 30 | 6 |
| definitions | variable, function, loop | 14/13/13 | 40 | 8 |

The instruction templates are paraphrases (`What is the capital of France?`,
`France's capital is?`, and so on). Responses are deterministic: short answers,
integer equations, comma-separated lists, extractive one-line summaries, small
Python expressions, and one-sentence definitions. Three bases per category keep
the answer space small enough for a 96-wide model to learn within the CPU
budget; the split still holds out whole instruction examples.

`split_dataset` uses one seeded RNG, shuffles each category independently, and
takes 20% as test: 160 train and 40 test. It verifies that both splits contain
all six categories, that no instruction appears in both, and that every test
example's base item also appears in training. The held-out set is therefore a
paraphrase-generalization test, not a test of unseen answers.

## Tokenizer and padding

`InstructionTokenizer` is byte-level with three reserved control ids:

| Symbol | Id | Meaning |
|---|---:|---|
| `INST_ID` | 256 | start of instruction |
| `RESP_ID` | 257 | instruction/response boundary |
| `PAD_ID` | 258 | right padding |

UTF-8 bytes occupy 0–255, so the vocabulary is 259. `encode_pair` adds the
newline terminator and returns the index of the first response byte;
`encode_prefix` returns `[INST] instruction [RESP]` for generation. A pair or
prefix that would exceed the 64-token context raises instead of silently
truncating the response. `decode_response` rejects control ids.

`SFTDataset` returns `(input_ids, labels)` with the instruction, boundary, and
(newly added) padding positions set to `-100`. `sft_collate` right-pads each
batch to its longest sequence. No key-padding mask is needed inside attention:
padding is always on the right and the causal mask prevents a real token from
attending to a later pad; padded query positions are trained on masked labels.

## Training

The demo uses:

- Adam, learning rate `1e-3`, no scheduler;
- batch size 8, 60 epochs;
- evaluation every 5 epochs on the 40 held-out pairs;
- local seeded generators for fixture order, model initialization, DataLoader
  shuffling, and sampled generation.

Sixty epochs is an adaptation of the lesson's 10–20 epoch suggestion. With 160
training pairs, the model reaches a low training loss but still needs many
passes before greedy exact-match on paraphrases stabilizes; 60 epochs keeps the
default run under 30 seconds on the measured CPU. The eval curve is retained,
including its oscillations, rather than selecting the best epoch on the test
set.

## Generation and exact match

`generate` feeds the instruction prefix and decodes one byte at a time:

- `temperature=0` uses argmax (the exact-match path);
- `temperature>0` samples from `softmax(logits / temperature)` with a seeded
  generator;
- generation stops at the 64-token context, at `max_new_tokens`, on any control
  id, on the response newline, or after two consecutive `.`, `!`, or `?` bytes;
- control ids and the newline are never included in the decoded response;
- the model's training/eval flag is restored on exit.

`exact_match` lowercases, strips, and collapses whitespace before comparison.
Punctuation and wording must still match exactly, so a paraphrase of the gold
response scores 0. `evaluate` runs greedy generation, aggregates overall and
per-category accuracy, stores predictions, and restores the model mode.

## Quickstart

Prerequisites: Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/). From this
module's directory:

```bash
uv sync --frozen --all-groups
uv run --frozen python code/main.py
```

The first command installs the pinned CPU environment into `.venv/`. The demo
builds the fixture, trains, evaluates, and prints the measurements; it writes
no fixtures, checkpoints, or result files. Python may create the usual ignored
`__pycache__/` bytecode.

A fast smoke run (tiny model, two epochs):

```bash
uv run --frozen python code/main.py \
  --epochs 2 --hidden 16 --num-heads 2 --num-layers 1 \
  --batch-size 8 --eval-every 1
```

Run the full gate:

```bash
./verify.sh
```

`verify.sh` syncs the frozen lockfile, runs the tests and strict typecheck, then
runs the default demo. It is safe to run twice and installs only into the
ignored `.venv/`.

PyTorch may print a one-time `Failed to initialize NumPy` warning because this
module intentionally has no NumPy dependency. It is an optional PyTorch
integration warning and does not affect execution or measured results.

## What runs offline

Everything after dependency installation is offline:

- the 200-pair fixture is generated in memory with `random.Random(seed)`;
- all training, generation, and evaluation are local PyTorch CPU operations;
- tests never access the network or download datasets or checkpoints;
- no API key, provider, cloud sandbox, or GPU is used.

`uv sync` may download the pinned CPU wheel from the configured PyTorch index
on a fresh machine. That is the only network boundary.

## Optional GPU / real-provider mode

No paid API, external dataset, GPU, or cloud service is required or used; the
local implementation is the real mechanism, not a simulation of a remote one.
The model, batches, and labels are device-agnostic PyTorch tensors, so a GPU
port would move them with `.to(device)` and update the training/eval loops. That
port is not implemented or measured here, and no GPU numbers are claimed.

## Measured results

Measured locally on CPU with the pinned environment; one seeded default run,
not a benchmark. The default demo took 25.3 s wall time (the `verify.sh` gate,
including the 20-test suite and mypy, took 40.7 s):

```text
instruction-tuning-sft: seed=0 train=160 test=40 max_len=64
dataset: pairs=200 capitals=40 arithmetic=30 lists=30 summaries=30 code=30 definitions=40
tokenizer: vocab=259 inst=256 resp=257 pad=258 terminator=10
model: parameters=253152 context_length=64 hidden=96 heads=4 layers=2
mask: targets=6683 response=2576 ignored=4107 response_fraction=0.385
baseline: exact_match=0.000 correct=0/40
epoch   5: loss=1.7143 exact_match=0.025
epoch  10: loss=0.5903 exact_match=0.125
epoch  15: loss=0.2526 exact_match=0.300
epoch  20: loss=0.0935 exact_match=0.425
epoch  25: loss=0.1030 exact_match=0.350
epoch  30: loss=0.0361 exact_match=0.575
epoch  35: loss=0.0365 exact_match=0.575
epoch  40: loss=0.0123 exact_match=0.700
epoch  45: loss=0.0390 exact_match=0.475
epoch  50: loss=0.0599 exact_match=0.475
epoch  55: loss=0.0418 exact_match=0.650
epoch  60: loss=0.0086 exact_match=0.775
final: exact_match=0.775 correct=31/40
per-category: arithmetic=1.000 (6/6)
per-category: capitals=1.000 (8/8)
per-category: code=0.333 (2/6)
per-category: definitions=0.625 (5/8)
per-category: lists=1.000 (6/6)
per-category: summaries=0.667 (4/6)
sample: [MATCH] instruction=What city is Spain's capital?
sample:         gold=Madrid.
sample:         prediction=Madrid.
sample: [MATCH] instruction=Where is the capital of France?
sample:         gold=Paris.
sample:         prediction=Paris.
sample: [MATCH] instruction=Name the capital city of Italy.
sample:         gold=Rome.
sample:         prediction=Rome.
check: loss_finite=true mask_only_response=true
```

The untrained baseline is 0/40. After training, 31/40 held-out paraphrases
produce the gold response character for character. The lookup tasks (capitals,
arithmetic, lists) reach 1.0; the copying/definition tasks remain weaker, which
is the interesting failure mode: a 253k-parameter model trained from random
initialization learns the format and many answers, but exact-match is strict
about every byte. The eval curve oscillates because each point measures only 40
examples and training is not early-stopped.

The mask line shows the objective directly: only 2,576 of 6,683 positions
(38.5%) are response targets. The other 4,107 instruction and boundary
positions, plus every right-padding position added by `sft_collate`, are
excluded from the loss numerator and denominator. Their logits receive no
direct target gradient; response loss still flows back through attention into
the instruction representations, which is what conditions the answer.

Verification on the same machine:

```text
20 passed, 1 warning in 7.79s
Success: no issues found in 2 source files
verify: ok
```

The warning is PyTorch's optional-NumPy notice described above.

## Design decisions

- **Response-only loss.** The labels are built before padding and the loss
  applies the one-token shift explicitly. A unit test proves the masked target
  logits receive zero direct gradient.
- **Tied LM head.** Tying reduces parameters and is standard for small
  decoder-only models. The PAD embedding row is zeroed at initialization;
  because the head is tied, that row can still receive gradient as a negative
  output class, but PAD labels are masked and padding never affects real-token
  logits.
- **No key-padding mask.** Right padding with a causal mask is sufficient for
  this training layout; tests pin that trailing pads do not change earlier
  logits.
- **Printable newline terminator.** It supplies a supervised stop signal
  without changing the three reserved ids. A production system should use an
  explicit EOS token instead.
- **Three bases per category.** The 200-pair, six-task fixture keeps a closed
  answer space so the tiny model can demonstrate SFT within the CPU budget.
  This is a deliberate simplification, not a claim about real instruction
  data.
- **Deterministic everything.** Fixture order, split, initialization, shuffling,
  and sampling are seeded. The demo is reproducible run to run.
- **Greedy exact match.** The metric is deterministic and unambiguous. It
  deliberately does not give partial credit for paraphrases.
- **No NumPy.** The fixture, masks, model, and metrics are pure PyTorch/Python;
  no NumPy operation is needed.

## Limitations

- The fixture is synthetic and small. Exact-match on 40 examples is a noisy
  estimate; the per-category scores are coarser still (6–8 examples each).
- Three bases per category and 60 epochs were chosen to fit the CPU budget and
  make the mechanism observable. Real SFT uses far more instructions, answer
  diversity, and pretrained weights.
- Summaries are short extractive sentences, not abstractive summaries.
- The model is trained from random initialization. It is not GPT-2 or a
  pretrained base checkpoint, so the observed exact-match does not transfer to
  a general instruction-following claim.
- Training loss falls much lower than held-out exact-match: the model partly
  memorizes instruction phrasings rather than generalizing robustly. The
  held-out set is also used for periodic reporting, with no separate validation
  split or model selection.
- There is no EOS special; the printable newline is the stop mechanism.
  Control-token and two-punctuation stops are heuristics, not a learned EOS.
- Exact-match is brittle. Token-level F1, BLEU/chrF, or a judge model are out of
  scope; a paraphrase with the same answer scores 0.
- No learning-rate warmup/cosine schedule, gradient clipping, early stopping,
  multi-turn chat template, GPU path, or real-provider adapter is implemented.
- The module intentionally has no NumPy dependency; PyTorch may warn once.

## Tests

`tests/test_main.py` contains 20 passing test cases covering:

- tokenizer specials, UTF-8 round trips, boundary index, and invalid inputs;
- fixture determinism, category counts, uniqueness, and 64-token bounds;
- stratified 160/40 split, disjoint instructions, and base coverage;
- dataset masking of instruction/boundary and preservation of response/newline
  labels;
- dynamic padding, pad-label masking, and malformed batch rejection;
- the explicit label shift, manual cross-entropy equivalence, and zero direct
  gradient on ignored target positions;
- model shapes, tied head, parameter count, causal invariance, and validation;
- greedy/sampled generation, all stop rules, context limits, and mode restore;
- exact-match normalization, per-category evaluation, and mode restore;
- training loss reduction, eval-hook scheduling, and all-ignored rejection;
- demo output contract, determinism, and invalid CLI configuration.

```bash
uv run --frozen pytest
uv run --frozen mypy code/main.py tests
```

## References

1. AI Engineering From Scratch, [Capstone Lesson 39: Instruction Tuning by
   Supervised Fine-Tuning](https://aiengineeringfromscratch.com/learn/capstone-projects/instruction-tuning-sft)
   — task contract, six fixtures, masking rules, generation heuristic, and
   exact-match objective.
2. Sebastian Raschka, [When should prompt tokens be masked out of the loss
   during instruction finetuning?](https://sebastianraschka.com/faq/docs/when-mask-prompt-tokens.html)
   — response-only objective, the label-shift off-by-one warning, end-token
   handling, and padding masks.
3. Hugging Face TRL, [SFT Trainer](https://huggingface.co/docs/trl/en/sft_trainer)
   — prompt-completion/completion-only dataset formats, completion-only loss,
   assistant-only loss, and causal label shifting.
4. Tatsu Lab, [Alpaca dataset card](https://huggingface.co/datasets/tatsu-lab/alpaca)
   — the instruction/input/output pair format that motivates the template.
5. PyTorch, [`torch.nn.functional.cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
   — `ignore_index`, mean reduction, and the `-100` convention.

## License

MIT License, Copyright (c) 2026 Pedro Monteiro. See [LICENSE](LICENSE).
