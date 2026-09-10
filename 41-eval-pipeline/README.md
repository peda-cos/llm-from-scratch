# Full Evaluation Pipeline from Scratch

## Overview

Training loss is a diagnostic, not an evaluation. A language model can fit its
training distribution while failing to answer short questions, paraphrase,
follow instructions, or score well on a judge's rubric. This module builds the
evaluation layer instead of another training loop: one tiny trained model, four
heterogeneous held-out evaluations, one weighted aggregate, and a JSON report
that a CI job can ingest.

The four evaluations are:

| Eval | Question it answers | Metric |
| --- | --- | --- |
| Perplexity | How well does the model fit held-out language? | exponentiated mean token NLL |
| Exact match | Does the model produce the gold short answer? | accuracy after normalisation |
| Token F1 | How much of the gold answer does it recover? | multiset token F1 |
| Mock judge | How good does the answer look qualitatively? | mean 1-5 rubric score |

Everything runs on CPU, offline, from a deterministic synthetic corpus. The
judge is a local deterministic scorer with the same callable interface a
provider-backed judge would use. No dataset download, checkpoint, API key, or
network call is required after dependency installation.

The measured default run is deliberately not flattering. The 259k-parameter
model learns the training prompts but only partially generalises to unseen
prompt templates. The pipeline reports that honestly: exact match 0.200, token
F1 0.188, judge 1.95/5, and aggregate 0.226. The report also preserves the
per-example failures, which is the point of an evaluation harness.

## Architecture

```text
synthetic fact/template tables
        │
        ├── 30 LM training sentences ──► LM pretraining (80 steps)
        │
        ├── 233 instruction/answer pairs ──► batched completion-only SFT
        │                                      │
        │                                      ▼
        │                                TinyGPT (259,296 params)
        │                                      │
        │              ┌───────────────────────┼───────────────────────┐
        │              ▼                       ▼                       ▼
        │       perplexity_eval        exact_match_eval        token_f1_eval
        │       held-out LM (20)       short facts (20)        open form (20)
        │              │                       │                       │
        │              │                       │                 judge_eval
        │              │                       │                 mock judge (20)
        │              └───────────────────────┴───────────────────────┘
        │                                      │
        │                                      ▼
        │                              Aggregator.normalise
        │                                      │
        │                                      ▼
        │                              weighted aggregate
        │                                      │
        └──────────────────────────────► report table + report.json
```

`EvalSuite` is a thin orchestrator: it builds one predictor and calls each
selected eval in a stable order. Each eval is a free function over a model (or
predictor) and a dataset, returning an `EvalResult` with per-example records.
The aggregator normalises each metric to `[0, 1]`, applies configurable weights,
and produces a `FinalReport` whose JSON schema is stable across runs.

## What is measured

### Held-out perplexity

Perplexity is `exp(mean negative log-likelihood per scored token)`. The
implementation has two traps it explicitly avoids:

- The denominator is the number of real target tokens, not documents, batches,
  or `batch * sequence`. Padding positions are masked out.
- Logits at position `i` are aligned with the token at position `i + 1`. The
  eval slices `logits[:, :-1]` against `ids[:, 1:]`; the tests pin this direction
  with a next-token-copy model that must yield perplexity near 1.

Each held-out document is encoded as raw UTF-8 bytes and scored in one padded
batch. The first byte of each document has no in-document predecessor and is
not counted. The per-token NLLs are also bucketed into a histogram
(`[0,0.5)`, `[0.5,1)`, `[1,2)`, `[2,4)`, `[4,8)`, `[8,inf)`). The JSON report
includes both the per-document values and the histogram.

This is a per-document perplexity, not a sliding-window corpus perplexity. The
fixtures are short enough to fit the 128-token context, so there is no
cross-document context and no stride overlap. That makes the accounting easy to
audit; it is not comparable to a subword-model WikiText number.

### Exact match

Both prediction and reference are normalised:

1. lowercase;
2. strip surrounding whitespace;
3. collapse internal whitespace runs to one space;
4. drop trailing terminal punctuation (`.`, `!`, `?`), repeated, then strip.

`"Paris."`, `"  paris  "`, and `"Paris"` all match. Different answers do not.
Non-ASCII answers are kept as UTF-8; the tokenizer is byte-level.

### Token F1

Token F1 is the harmonic mean of precision and recall over the whitespace
bag-of-tokens after the same normalisation:

```text
matches    = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
precision  = matches / len(prediction_tokens)
recall     = matches / len(reference_tokens)
F1         = 2 * precision * recall / (precision + recall)
```

Repeated tokens count as a multiset. Both sides empty is a vacuous match
(`1.0`); exactly one side empty is `0.0`; no overlap is `0.0`. This follows the
SQuAD evaluation convention. Commas remain part of comma-separated tokens by
design, because the spec stops at whitespace tokenisation; the fixtures use
consistent punctuation so that exact and partial answers are distinguishable.

### Mock LLM-as-a-judge

`mock_judge(instruction, prediction, reference) -> JudgeVerdict` is a
deterministic reference-based grader:

| Condition | Score |
| --- | --- |
| exact normalised match | 5 |
| token F1 >= 0.80 | 4 |
| token F1 in [0.50, 0.80) | 3 |
| token F1 in [0.20, 0.50) | 2 |
| otherwise | 1 |

Every verdict carries a one-line rationale with the computed F1 and band. The
instruction is accepted for interface parity but the mock rubric does not use
it; a real judge would. `judge_eval` takes a `judge` callable, so replacing the
mock is a one-function change and does not alter the pipeline.

### Aggregation

Each eval has a normalisation rule:

| Eval | Normalised score |
| --- | --- |
| Perplexity | `1 / (1 + log(perplexity))` |
| Exact match | already in `[0, 1]` |
| Token F1 | already in `[0, 1]` |
| Judge | `score / 5` |

The default weights are 0.2 perplexity, 0.3 exact match, 0.3 token F1, and 0.2
judge. The aggregator divides by the sum of the configured weights, so a
caller may use any positive scale. Contributions sum to the aggregate. Because
the spec defines judge normalisation as `score / 5`, the minimum normalised
judge contribution is 0.2, not 0.

## Data and held-out policy

All data is constructed in memory from module-level fact and template tables.
There are exactly 20 examples in each of the four evaluation fixtures:

- **LM fixture**: 20 held-out declarative sentences.
- **EM fixture**: 20 short-form prompts (8 capitals, 6 arithmetic, 6 lists).
- **F1 fixture**: 20 open-form prompts (8 lists, 12 definitions).
- **Judge fixture**: 20 prompts (10 definitions, 10 summaries).

Training has 233 instruction/answer pairs and 30 LM sentences. The eval prompt
strings do not appear in training. The split is **template-held-out**, not
fact-held-out: an eval fact appears in training under other prompt templates,
and an eval template appears in training on disjoint coverage facts. This is a
deliberate, documented generalisation test for a tiny model; it is not a claim
that the model has never seen the answer vocabulary.

The tests enforce deterministic construction, unique prompts within each
fixture, exact prompt disjointness between training and all eval fixtures,
balanced categories, single-line references, and context fit. Any violation
raises during fixture construction, not after a long training run.

## Quickstart

```bash
cd 41-eval-pipeline
./verify.sh
```

`verify.sh` installs the pinned CPU environment into `.venv`, runs 27 tests,
runs strict mypy, executes the default demo, and exits 0. It is safe to run
twice. The first run on a fresh machine downloads the pinned CPU wheel; after
that, everything is offline.

Run the demo directly:

```bash
uv run --frozen python code/main.py
```

It trains the model, runs all four evals, prints the report table and a few
per-example diagnostics, and writes `outputs/report.json`. Useful overrides:

```bash
uv run --frozen python code/main.py \
  --seed 0 --lm-steps 80 --sft-epochs 60 --sft-batch-size 8 \
  --hidden 96 --num-heads 4 --num-layers 2 --max-new-tokens 32 \
  --output outputs/report.json
```

The JSON schema is:

```json
{
  "schema_version": 1,
  "aggregate": 0.2262,
  "weights": {"perplexity": 0.2, "exact_match": 0.3, "token_f1": 0.3, "judge": 0.2},
  "evals": {
    "perplexity": {"metric": "perplexity", "value": 198.6689, "normalised": 0.1589,
      "weight": 0.2, "contribution": 0.0318, "num_examples": 20,
      "scored_tokens": 0, "examples": [], "histogram": []},
    "exact_match": {"metric": "accuracy", "value": 0.2, "normalised": 0.2,
      "weight": 0.3, "contribution": 0.06, "num_examples": 20, "examples": []},
    "token_f1": {"metric": "mean_token_f1", "value": 0.188, "normalised": 0.188,
      "weight": 0.3, "contribution": 0.0564, "num_examples": 20, "examples": []},
    "judge": {"metric": "mean_judge_score", "value": 1.95, "normalised": 0.39,
      "weight": 0.2, "contribution": 0.078, "num_examples": 20, "examples": []}
  },
  "metadata": {}
}
```

The actual file fills in the per-example arrays, the perplexity histogram, the
model/training metadata, and the real scored-token count. The keys and their
nesting do not change between runs.

## What runs offline

Everything after `uv sync` is local and deterministic:

- the synthetic fact, template, and sentence tables are built in memory;
- the byte tokenizer, transformer forward pass, LM pretraining, and SFT loop are
  pure PyTorch CPU operations;
- the mock judge is a local function, not a provider call;
- tests use no network, no downloaded fixtures, no pretrained checkpoint, and
  no API key;
- the JSON report is written to an ignored `outputs/` directory.

The only network boundary is the pinned dependency install.

## Optional real-provider / GPU mode

A real judge replaces `mock_judge` without changing `judge_eval`: implement a
callable with the signature

```python
def provider_judge(instruction: str, prediction: str, reference: str) -> JudgeVerdict:
    ...
```

and pass it as `judge=provider_judge`. The production implementation would call
a provider, request a strict JSON object with a 1-5 score and a one-line
rationale, validate the response, and map provider failures to an explicit
error rather than silently scoring. No such client, key handling, retry policy,
or cost accounting is shipped here because the module must run offline and its
tests must not touch the network.

The evaluation layer is model-agnostic at the `Predictor` boundary: any object
with `predict(instruction) -> str` can be evaluated by the text evals, and any
`nn.Module` returning `(B, T, VOCAB_SIZE)` logits can be used for perplexity and
greedy generation. A GPU port would move the model and batches to the device
and keep the same metric code; no GPU numbers are claimed or measured here.

## Measured results

Measured locally on CPU with the pinned environment (`torch 2.15.0+cpu`,
Python 3.14). This is one seeded default run, not a benchmark. The default demo
took 35.1 s wall time; the complete `verify.sh` gate took 40.4 s.

```text
lm step   1: loss=5.6041
lm step  20: loss=3.1822
lm step  40: loss=2.3483
lm step  60: loss=1.6875
lm step  80: loss=0.9639
sft epoch   1: loss=3.1262
sft epoch  10: loss=2.1749
sft epoch  20: loss=1.6388
sft epoch  30: loss=1.1310
sft epoch  40: loss=0.7083
sft epoch  50: loss=0.4317
sft epoch  60: loss=0.2602
eval-pipeline: seed=0 train_pairs=233 lm_train=30 eval_lm=20 eval_em=20 eval_f1=20 eval_judge=20
model: parameters=259296 hidden=96 heads=4 layers=2 context=128
training: lm_steps=80 lm_loss=5.6041->0.9639 sft_epochs=60 sft_loss=3.1262->0.2602
eval          metric                 value  normalised   weight  contribution
perplexity    perplexity          198.6689      0.1589    0.200        0.0318
exact_match   accuracy              0.2000      0.2000    0.300        0.0600
token_f1      mean_token_f1         0.1880      0.1880    0.300        0.0564
judge         mean_judge_score      1.9500      0.3900    0.200        0.0780
aggregate: 0.2262
sample: perplexity worst=842.66 tokens=33 text='When you compute 9 - 4, you get 5.'
sample: exact_match score=0.000 instruction='The capital of France is?' gold='Paris.' prediction='Parisbococons.'
sample: token_f1 score=0.000 instruction='Which three oceans come to mind?' gold='pacific, atlantic, indian.' prediction='cocps, suswaticollndig.'
sample: judge score=1.000 instruction='In one sentence, define variable.' gold='a name bound to a value.' prediction='ave ve thtath.'
report: outputs/report.json
```

The interesting result is not the aggregate alone. Training loss falls from
3.13 to 0.26, yet exact match is 0.200 and token F1 is 0.188. The model has
memorised much of the training split but has not learned to terminate answers
reliably on unseen templates: `Parisbococons.` begins with the right capital and
still fails exact match. Token F1 gives partial credit where exact match does
not, and the judge turns the same partial overlap into scores mostly in the
1-3 range. Perplexity is high because instruction tuning overwrites much of the
small model's language-modelling behaviour; the four metrics disagree, which is
exactly why the pipeline reports all of them.

Verification on the same machine:

```text
27 passed, 1 warning in 2.29s
Success: no issues found in 2 source files
verify: ok
```

The warning is PyTorch's optional-NumPy notice: this module does not import or
depend on NumPy.

## Design decisions

- **One implementation file.** The spec asks for one `main.py` plus tests.
  Sections are ordered by pipeline stage, and every public function is usable
  without the demo.
- **Byte tokenizer with `[INST]`, `[RESP]`, `[PAD]`.** No subword vocabulary is
  downloaded. Answers end with a newline byte that is both a supervised target
  and the generation stop marker. The newline keeps the vocabulary at the
  three specials named by the spec.
- **Two-stage training.** LM pretraining teaches byte-level language structure;
  completion-only SFT teaches instruction following without spending capacity
  reproducing the prompt. SFT uses mini-batches and multiple optimizer steps
  per epoch; one full-batch step per epoch under-trains the 233-pair fixture.
- **Template-held-out split.** Exact eval prompts are excluded from training,
  while facts and templates are each seen in other combinations. This is the
  only way a 259k model can demonstrate any generalisation without leaking the
  eval strings, and it is stated rather than hidden.
- **Masked perplexity accounting.** The denominator is the number of scored
  tokens. The first byte of a document and all pads are excluded; a test with
  uniform logits pins the count and a next-token-copy test pins the shift.
- **Per-document PPL.** Fixtures fit the context, so the simpler independent-
  document variant is auditable. The README does not compare it to sliding-
  window corpus numbers.
- **Deterministic mock judge.** The rubric is explicit, the rationales are
  reproducible, and the interface is the replacement point for a real judge.
  The mock is not presented as a human-preference measurement.
- **Weighted mean with a normalising denominator.** Weights need not sum to one;
  contributions always sum to the aggregate. The default mix is a product
  choice, not a discovered optimum.
- **Stable JSON keys, no volatile metadata.** CI can diff and plot the report.
  Timings and hostnames are printed to stdout, not stored in the report.
- **One CPU thread in the CLI.** The demo prints deterministic measurements
  across CPU thread configurations; tests use small configurations.

## Limitations

- The judge is a deterministic token-F1 proxy. It cannot assess factuality,
  style, safety, or instruction following beyond lexical overlap. Its scores
  correlate strongly with token F1 by construction, and the instruction argument
  is unused. Real judges also have position, verbosity, and self-enhancement
  biases that this mock neither exhibits nor measures.
- The model is tiny and the split is template-level. The absolute scores are
  not a claim about language models in general, and the measured result is one
  seed. There is no confidence interval, bootstrap, or multiple-seed variance
  estimate.
- Perplexity is byte-level and per-document. It is not comparable with subword
  perplexities on WikiText or with sliding-window corpus perplexity. The 20
  held-out sentences are short and synthetic.
- The eval fixtures share base facts with training. Exact prompt strings are
  held out, but the answer vocabulary is not. This measures template
  generalisation, not zero-shot factual knowledge.
- The model often fails to emit the answer terminator on unseen templates. This
  is reported as a failure, not hidden with a decoding heuristic or a substring
  match.
- Instruction tuning degrades held-out language modelling on this scale. The
  report shows the trade-off; it does not attempt replay, KL regularisation,
  early stopping, or checkpoint selection.
- Exact match and token F1 use whitespace tokenisation and the normalisation in
  the spec. They do not implement stemming, article removal, punctuation
  stripping inside tokens, or semantic equivalence.
- Weight tuning, calibration, robustness perturbations, and learned weights are
  not implemented. The default weights are fixed and arbitrary by design.
- No GPU path, real provider call, dataset download, or pretrained checkpoint
  is shipped or measured. The provider adapter is documented, not wired to a
  network client.

## Tests

`tests/test_main.py` contains 27 test functions covering:

- tokenizer specials, UTF-8 round trips, newline termination, and invalid input;
- model shape, tied head, causal prefix invariance, parameter count, seed
  determinism, and input validation;
- deterministic fixture sizes, category balance, prompt disjointness, and
  context fit;
- normalisation, exact match, token F1 multiset values, repeated tokens, and
  the empty-answer rules;
- all five mock-judge bands, including exact F1 boundaries, and rationale
  propagation;
- uniform-logit and next-token-copy perplexity models, pad/first-token masking,
  histogram accounting, and invalid corpora;
- scripted-predictor exact-match, token-F1, and judge evals;
- full and subset `EvalSuite` runs and unknown-eval rejection;
- aggregator normalisation, weighted means, contributions, and invalid
  weights/results;
- stable report keys, JSON round trips, recomputable aggregates, and per-example
  counts;
- newline-bounded greedy generation and the `max_new_tokens` limit;
- a tiny end-to-end demo, JSON output, CLI success, and CLI error handling.

```bash
uv run --frozen pytest
uv run --frozen mypy code/main.py tests
```

## References

1. AI Engineering From Scratch, [Capstone Lesson 41: Full Evaluation
   Pipeline](https://aiengineeringfromscratch.com/learn/capstone-projects/eval-pipeline)
   — task contract, the four evals, mock judge rubric, aggregation formula, and
   demo requirements.
2. Hugging Face Transformers, [Perplexity of fixed-length
   models](https://huggingface.co/docs/transformers/en/perplexity) — the PPL
   definition, why the denominator counts real tokens, and sliding-window
   masking context.
3. Pranav Rajpurkar et al., [SQuAD official evaluation
   script](https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py)
   — multiset token F1 and the both-empty/one-empty convention.
4. Lianmin Zheng et al., [Judging LLM-as-a-Judge with MT-Bench and Chatbot
   Arena](https://arxiv.org/abs/2306.05685) — the judge interface, 1-5 scoring,
   and the documented position/verbosity/self-enhancement limitations.
5. EleutherAI, [LM Evaluation Harness: Concepts &
   Architecture](https://lm-evaluation-harness.readthedocs.io/getting_started/concepts)
   — the dataset → model responses → scorer → reduce → aggregate pipeline shape
   and the separation between evals and aggregation.

## License

MIT. See [LICENSE](LICENSE).
