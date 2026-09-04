# Classifier Fine-Tuning by Head Swap

## Overview

A pretrained language model ends in a token-prediction head. For spam versus
ham, that head is wrong but the body is mostly right. This module keeps a tiny
pre-norm transformer body, pretrains it briefly with a causal language-model
objective, removes the language-model head, and attaches a two-class linear
head to a masked mean-pooled representation.

The same classifier is then trained twice:

- **Head-only** — the body stays frozen; only the 130-parameter head learns.
- **Full fine-tuning** — every body and head parameter learns, at a lower
  learning rate.

The held-out evaluation reports precision, recall, F1, and the four confusion
counts. Everything is local, deterministic, and CPU-only.

## Architecture

```text
byte text
   │  ByteTokenizer: UTF-8 bytes 0..255, PAD 256, ids 257..259 reserved
   ▼
ClassificationDataset: right-pad to 32 + boolean attention mask
   │
   ▼
LMBody: token embedding + position embedding → 2 × pre-norm Block → final norm
   │
   ▼
MeanPool: mask-weighted average over the sequence axis → (B, 64)
   │
   ▼
Classifier head: Linear(64, 2) → raw logits
   │
   ├── frozen body + Adam(head)      = head-only regime
   └── trainable body + Adam(all)    = full fine-tuning
   │
   ▼
evaluate: argmax → TN / FP / FN / TP → precision, recall, F1
```

The body is a deliberately small decoder: vocabulary 260, hidden width 64,
four attention heads, two blocks, maximum sequence length 32. The default body
has 118,784 scalar parameters; the classifier head has 130. Full fine-tuning
therefore updates 118,914 parameters, over 900 times the head-only count.

### Head swap

`LMBody` is a function from token ids to hidden states. `pretrain_quick` puts a
temporary `Linear(64, 260)` language-model head on top and trains the body for
five epochs with next-token cross-entropy. The temporary head is discarded.
`Classifier` then reaches the same body through masked mean pooling and a fresh
`Linear(64, 2)` head. No body parameter is reinitialised; `classifier.body is
body` is part of the contract, and the tests pin it.

### Pooling and padding

The classifier needs one vector per message. Mean pooling is:

```text
pooled[b] = sum_t(hidden[b, t] * mask[b, t]) / sum_t(mask[b, t])
```

The mask is true for real bytes and false for right-padding. Padding is always
on the right and attention is causal, so a real token never attends to a later
pad; the pool mask then removes pad positions from the representation.
Mid-sequence padding is not supported.

### Two regimes, one loop

`freeze_body` sets `requires_grad=False` on body parameters; `unfreeze_body`
restores it. PyTorch does not accumulate gradients for frozen leaves, and the
demo builds each optimizer from exactly the currently trainable parameters.
`train_classifier` is otherwise identical for both regimes. The full regime
uses a fresh optimizer and a lower learning rate (`2e-4` versus `1e-3`), since
small-data full fine-tuning is sensitive to an aggressive rate.

The demo follows the lesson's sequential contract: the same body instance is
pretrained, trained head-only, evaluated, then unfrozen and fine-tuned in full
and evaluated again. Full fine-tuning therefore warm-starts from the head-only
head; it is not a second independent initialization.

## Quickstart

Prerequisites: Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/). From this
module's directory:

```bash
uv sync --frozen --all-groups
uv run --frozen python code/main.py
```

The first command installs the pinned CPU environment into `.venv/`. The demo
generates the fixture in memory, pretrains the body, runs both regimes, and
prints the two reports. It saves nothing to disk.

A fast smoke run:

```bash
uv run --frozen python code/main.py \
  --messages 64 --d-model 16 --num-heads 2 --num-layers 1 \
  --pretrain-epochs 1 --head-epochs 2 --full-epochs 3 \
  --batch-size 16 --dropout 0 --head-lr 0.001 --full-lr 0.001 --seed 0
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
integration warning and does not affect execution or results.

## What runs offline

Everything after dependency installation is offline:

- the SMS fixture is generated in memory with `random.Random(seed)`;
- pretraining, both fine-tuning regimes, and evaluation are local PyTorch CPU
  operations;
- tests never access the network and never download datasets or checkpoints;
- no API key, provider, cloud sandbox, or GPU is used.

`uv sync` may download the pinned CPU wheel from the configured PyTorch index
on a fresh machine. That is the only network boundary.

## Deterministic fixture

`make_sms_fixture(count=800, seed=0)` enumerates two-slot templates and
substitutes short fillers for amounts, prizes, codes, times, names, and common
ham topics. Messages are 5–25 UTF-8 bytes; the default 800-message fixture
contains 400 spam and 400 ham messages, all unique, with byte lengths 8–18.

`stratified_split` shuffles each class with a fixed seed and takes the same
fraction from each, producing 640 training and 160 test messages at the default
80/20 split. The test set keeps the 50/50 balance and never shares an exact
message with training.

The fixture is deliberately clean and templatic. It proves the mechanism, not
real-world SMS robustness. The generator uses no labels to build the text, and
the pretraining pass sees only training messages, never the test split.

## Metrics

Class 1 is the positive class (spam). From raw logits, predictions are
`argmax(logits)`:

```text
TP = predicted spam and actually spam
FP = predicted spam and actually ham
FN = predicted ham and actually spam
TN = predicted ham and actually ham

precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

A zero denominator yields 0.0 rather than NaN, matching the `zero_division=0`
behavior documented by scikit-learn. The printed confusion matrix has rows as
true labels and columns as predictions: `[[TN, FP], [FN, TP]]`.

## Measured results

Measured locally on CPU with the pinned environment; one seeded default run,
not a benchmark. Wall time includes startup; phase seconds are printed by the
demo.

```text
demo: seed=0 messages=800 train=640 test=160
fixture: spam=400 ham=400 byte_lengths=8..18
split: train=640 test=160 train_spam=320 train_ham=320 test_spam=80 test_ham=80
tokenizer: vocab=260 pad_id=256
model: body_parameters=118784 head_parameters=130 context_length=32 d_model=64 num_heads=4 num_layers=2
pretrain: epochs=5 losses=[3.8013, 2.2983, 1.5828, 1.2261, 1.0121] seconds=2.95
regime: head-only trainable_parameters=130 body_frozen=true epochs=20
train: head-only epochs=20 final_loss=0.1124 final_accuracy=0.9828
head-only: precision=1.0000 recall=1.0000 f1=1.0000
head-only confusion (rows=true, cols=pred):
  ham  tn=80 fp=0
  spam fn=0 tp=80
head-only seconds=4.37
regime: full trainable_parameters=118914 body_frozen=false epochs=20
train: full epochs=20 final_loss=0.0001 final_accuracy=1.0000
full: precision=1.0000 recall=1.0000 f1=1.0000
full confusion (rows=true, cols=pred):
  ham  tn=80 fp=0
  spam fn=0 tp=80
full seconds=7.77
check: body_unchanged_in_head_only=true body_changed_in_full=true
summary: head_only_trainable=130 full_trainable=118914 head_seconds=4.37 full_seconds=7.77 ratio=1.78
```

Total wall time for the default demo was 16.48 s. Head-only training reached
perfect held-out F1 after 20 epochs; full fine-tuning also saturated, with a
much lower final training loss. On this easy fixture the meaningful differences
are parameter coverage and cost: 130 trainable weights versus 118,914, and a
measured full/head training-time ratio of 1.78× rather than the lesson's rough
"about three times" rule of thumb. The measured numbers are reported as-is; the
fixture was not tuned to manufacture an accuracy gap.

Verification on the same machine:

```text
21 passed, 1 warning in 2.30s
Success: no issues found in 2 source files
verify: ok
```

The warning is PyTorch's optional-NumPy notice described above.

## Design decisions

- **Byte tokens, fixed vocabulary.** Bytes 0–255 are ids, 256 is PAD, and
  257–259 are reserved so the model vocabulary is the lesson's 260. Encoding
  is lossless UTF-8; decoding rejects special ids instead of silently dropping
  them.
- **The body is the same object across regimes.** The head is replaced without
  touching body parameters. The head-only check clones and compares every body
  tensor before and after training; the demo fails if any body value moved.
- **Masked mean pooling, not CLS or last-token.** Mean pooling gives a stable
  signal for variable-length short messages and needs no pretrained summary
  token. The mask weighting is explicit, and an all-pad sequence raises.
- **Right padding plus causal attention.** Because the dataset always pads on
  the right, real tokens cannot see later pads and no key-padding mask is
  needed inside attention. The pool mask removes pad positions.
- **Pretrain on training text only.** The temporary language-model head learns
  next-byte prediction for five epochs. Test messages are held out from this
  step, so the held-out report is not contaminated by unsupervised pretraining.
- **One loop, caller-owned optimizer.** `train_classifier` never changes
  `requires_grad` or constructs an optimizer. The regime is entirely determined
  by the caller's freeze/unfreeze call and parameter list.
- **Lower full-fine-tuning learning rate.** The head trains at `1e-3`; full
  fine-tuning at `2e-4`. Aggressive rates are a known small-data instability,
  and the lower rate keeps the body drift useful rather than destructive.
- **Deterministic seeds everywhere.** Fixture generation, split shuffling,
  parameter initialization, DataLoader shuffling, and pretraining order all use
  local generators derived from the demo seed. Tests use tiny configurations.
- **Pure Python/torch metrics.** No scikit-learn or NumPy dependency is needed
  for four integer counts and three ratios.

## Limitations

- The fixture is synthetic and easy. Both regimes saturate at F1 1.0, so this
  run does not show an accuracy headroom difference; it demonstrates the
  mechanism and the parameter/time cost.
- `pretrain_quick` trains from a random initialization for five epochs. It is
  not a real GPT-2/BERT checkpoint, and the hidden representation is tiny.
- There is no validation split, early stopping, or hyperparameter search. The
  fixed 160-example test set is evaluated once per regime, after fixed
  hyperparameters were chosen.
- Full fine-tuning is sequential: it starts from the head-only model and uses a
  fresh optimizer. It is not an independent-run comparison.
- CPU only; no GPU path, provider adapter, or distributed training is
  implemented. Real SMS data, class imbalance, label noise, and domain drift
  are outside this fixture.
- Mid-sequence padding is unsupported. Right padding is required by the
  causal-mask/pool-mask combination.
- The lesson's stretch goals (partial fine-tuning of the last block, learning
  rate schedules, attention pooling) are not implemented.

### Using real data or a GPU

The tokenizer, dataset, training loop, and metrics are dataset-agnostic at the
text/label boundary. To use a real CSV of SMS messages, replace
`make_sms_fixture` and `stratified_split` with a loader that returns
`(texts, labels)`; keep `ClassificationDataset`, the classifier, and the two
regimes. A word-level tokenizer would also need a larger `context_length` (for
example 128) and a matching vocabulary.

No GPU adapter is implemented. Running on CUDA would require moving the model,
batches, and mask to the device explicitly; the module deliberately reports
measured CPU behavior only. No paid provider or external service is involved.

## Tests

`tests/test_main.py` contains 21 passing test cases (16 test functions, one
four-case metric parametrization, and three invalid-configuration cases). They
cover:

- UTF-8 byte round trips, reserved ids, padding, truncation, and dtype checks;
- fixture determinism, balance, uniqueness, byte bounds, and stratified splits;
- dataset padding/masking and empty-input rejection;
- body shape, causal invariance under trailing pads, and bad-id/length errors;
- mask-weighted mean pooling, including the all-pad failure;
- shared-body head swap, logits, and explicit-versus-derived masks;
- freeze/unfreeze parameter counts and head-only body immutability;
- body changes under full fine-tuning;
- precision, recall, F1, confusion counts, zero-division cases, and mode restore;
- pretraining history, body change, and the zero-epoch no-op;
- the CLI output contract, both regimes, determinism, and invalid arguments.

```bash
uv run --frozen pytest
uv run --frozen mypy code/main.py tests
```

## References

1. AI Engineering From Scratch, [Capstone Lesson 38: Classifier Fine-Tuning by
   Head Swap](https://aiengineeringfromscratch.com/learn/capstone-projects/classifier-finetuning)
   — task contract, head-swap model, pooling options, fixture, metrics, and the
   two training regimes.
2. Chi Sun, Xipeng Qiu, Yige Xu, Xuanjing Huang, [How to Fine-Tune BERT for
   Text Classification?](https://arxiv.org/abs/1905.05583) (arXiv:1905.05583)
   — further pretraining, layer selection/freezing, lower fine-tuning learning
   rates, and catastrophic forgetting.
3. Tianyi Zhang, Felix Wu, Arzoo Katiyar, Kilian Q. Weinberger, Yoav Artzi,
   [Revisiting Few-sample BERT
   Fine-tuning](https://arxiv.org/abs/2006.05987) (arXiv:2006.05987) — few-sample
   instability, debiased Adam, and longer training.
4. PyTorch, [Autograd mechanics](https://docs.pytorch.org/docs/2.15/notes/autograd.html)
   — `requires_grad` as the documented freezing mechanism, and the distinction
   between `requires_grad` and `eval()`.
5. scikit-learn, [`precision_recall_fscore_support`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)
   — precision/recall/F-score definitions and zero-division behavior.

## License

MIT License, Copyright (c) 2026 Pedro Monteiro. See [LICENSE](LICENSE).
