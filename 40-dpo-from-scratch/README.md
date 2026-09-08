# Direct Preference Optimization from Scratch

## Overview

A supervised fine-tuned language model can follow instructions, but its
completions are uneven. Preference data says which of two completions a human
prefers for the same prompt. The classical RLHF stack turns those preferences
into a reward model and then optimizes the policy with PPO and a KL penalty.
Direct Preference Optimization (DPO) collapses both stages into one supervised
loss: the policy is fitted directly on `(prompt, chosen, rejected)` triples,
with an implicit reward defined relative to a frozen reference model.

This module implements that mechanism from scratch on CPU:

- a byte-level tokenizer with `[INST]` and `[RESP]` specials;
- a tiny decoder-only transformer with a tied language-model head;
- twelve deterministic preference triples covering capitals, arithmetic, and
  lists;
- completion-only sequence log-probabilities (prompt tokens are not scored);
- a response-only warm-up that produces the reference model;
- a frozen reference and a policy initialized as an exact copy;
- the DPO sigmoid loss, its implicit reward margin, and 30 Adam steps;
- tests that pin the loss formula, gradient signs, reference invariance, and
  the direction of training.

Everything after dependency installation runs locally. There is no reward
model, no PPO loop, no external API, no downloaded dataset, and no pretrained
checkpoint.

## Architecture

```text
twelve (prompt, chosen, rejected) triples
        │
        ▼
InstructionTokenizer ── [INST] + UTF-8 bytes + [RESP] + completion bytes
        │
        ├── response-only warm-up on (prompt, chosen)
        │          │
        │          ▼
        │     reference TinyGPT ── frozen, requires_grad=False
        │          │
        │          └── load_state_dict ──► policy TinyGPT (trainable)
        │
        ▼
sequence_log_prob ── log π(chosen | prompt), log π(rejected | prompt)
        │
        ├── policy forward (gradient on)
        └── reference forward (torch.no_grad, precomputed once)
        │
        ▼
dpo_loss ── -log sigmoid(beta * ((log π_w - log π_ref_w)
                                  - (log π_l - log π_ref_l)))
        │
        ▼
Adam step on the policy only
```

The default model has hidden width 96, four attention heads, two pre-norm
transformer blocks, a 128-token context, no dropout, and 259,296 parameters.
The context is larger than lesson 39's 64 tokens so the wordy rejected
completions fit; the architecture is the same decoder-only shape with a tied
LM head.

## The DPO objective

Given a prompt `x` and completions `y_w` (chosen) and `y_l` (rejected), the
Bradley-Terry preference model is

$$
P(y_w \succ y_l \mid x) = \sigma\bigl(r(x, y_w) - r(x, y_l)\bigr).
$$

RLHF maximizes the reward with a KL anchor to the SFT reference:

$$
\max_{\pi}\; \mathbb{E}_{x,\,y\sim\pi}\bigl[r(x,y)\bigr]
- \beta\, \mathrm{KL}\bigl(\pi(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\bigr).
$$

The optimal policy for this objective has the closed form

$$
\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y\mid x)
\exp\!\left(\frac{r(x,y)}{\beta}\right).
$$

Rearranging for the reward gives

$$
r(x,y) = \beta\bigl(\log \pi^*(y\mid x) - \log \pi_{\mathrm{ref}}(y\mid x)\bigr)
+ \beta\log Z(x).
$$

The partition term `beta * log Z(x)` depends only on `x`, so it cancels in the
chosen-minus-rejected difference. Substituting into Bradley-Terry and taking
the negative log likelihood over preference pairs yields the implemented loss:

$$
\begin{aligned}
z &= \beta\Bigl[
\bigl(\log\pi_\theta(y_w\mid x) - \log\pi_{\mathrm{ref}}(y_w\mid x)\bigr) \\
&\qquad - \bigl(\log\pi_\theta(y_l\mid x) - \log\pi_{\mathrm{ref}}(y_l\mid x)\bigr)
\Bigr], \\
L_{\mathrm{DPO}} &= -\log\sigma(z).
\end{aligned}
$$

`dpo_loss` takes the four log-probability tensors and returns the per-example
loss plus `z`, the implicit reward delta
`r_theta(x, y_w) - r_theta(x, y_l)`. No separate reward model is fitted and no
KL term appears explicitly in the loss; the KL constraint is structural in the
closed-form derivation.

Differentiating the loss gives the identities pinned by the tests:

$$
\frac{\partial L}{\partial \log\pi_\theta(y_w\mid x)} = -\beta\bigl(1-\sigma(z)\bigr),
\qquad
\frac{\partial L}{\partial \log\pi_\theta(y_l\mid x)} = +\beta\bigl(1-\sigma(z)\bigr).
$$

The first is negative for every finite `z`: raising the chosen completion's
log-probability lowers the loss. The second is positive: raising the rejected
completion's log-probability raises it. `F.logsigmoid` is used rather than
`log(sigmoid(z))` so large `|z|` stays finite.

## Reference invariance

The reference is the warm-up model frozen in place. Three properties are
enforced and tested:

- Every reference parameter has `requires_grad=False`.
- Reference log-probabilities are computed once under `torch.no_grad()` and
  reused for all 30 policy steps.
- The policy is constructed with `policy.load_state_dict(reference.state_dict())`
  immediately after the reference is built, so the first DPO step has
  `z = 0` and loss `log(2)`.

The tests recompute reference log-probabilities after training and compare them
with the pre-training values within `1e-6`, while checking that policy
parameters did change.

## Data fixture

`make_preferences()` returns exactly twelve fixed triples, four per category.
Chosen completions are short and precise; rejected completions are wordy,
off-topic, or wrong. Exact strings live in `_PREFERENCE_ROWS` in
`code/main.py`.

| # | Category | Prompt | Chosen |
|---:|---|---|---|
| 1 | capital | What is the capital of France? | Paris. |
| 2 | capital | What is the capital of Spain? | Madrid. |
| 3 | capital | What is the capital of Italy? | Rome. |
| 4 | capital | What is the capital of Japan? | Tokyo. |
| 5 | arithmetic | What is 2 + 3? | 5. |
| 6 | arithmetic | What is 7 * 6? | 42. |
| 7 | arithmetic | What is 9 - 4? | 5. |
| 8 | arithmetic | What is 12 / 3? | 4. |
| 9 | list | List the primary colors. | red, blue, yellow. |
| 10 | list | Name the planets in order. | Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune. |
| 11 | list | List the vowels. | a, e, i, o, u. |
| 12 | list | List the days of the week. | Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday. |

The rejected strings include, for example, a long discussion of French
museums, an off-topic answer about Spanish football, and arithmetic answers
that end with the wrong number. Every chosen completion is shorter than its
rejected counterpart in UTF-8 bytes, and every serialized pair fits the
128-token context.

## Tokenizer and completion scoring

`InstructionTokenizer` is byte-level with three reserved control ids:

| Symbol | Id | Meaning |
|---|---:|---|
| `INST_ID` | 256 | start of instruction |
| `RESP_ID` | 257 | instruction/response boundary |
| `PAD_ID` | 258 | right padding |

UTF-8 bytes occupy 0–255, so the vocabulary is 259. `encode_completion`
returns the full sequence and the index of the first completion byte. No
newline or EOS is appended: this module scores a supplied completion and does
not generate text. A production generation pipeline would reserve an EOS token
and score its position; that is outside this module's objective.

`sequence_log_prob` runs one forward pass over the prompt plus completion and
sums the log-softmax probabilities of completion tokens:

```text
positions:  [INST] p r o m p t [RESP] c h o s e n
targets:                            c h o s e n
```

The `[RESP]` position predicts the first completion byte. Prompt positions and
the boundary contribute no direct log-probability terms. The sum has exactly
`len(completion.encode("utf-8"))` terms. A test compares the function with a
manual `log_softmax` gather over the same completion positions.

## Warm-up and training

The reference is produced by a response-only warm-up on the chosen
completions:

- full-batch next-token cross-entropy over `(prompt, chosen)` sequences;
- labels before `completion_start` are `-100`, including the `[RESP]` label;
- right padding with `PAD_ID`; padded labels are `-100`;
- 60 Adam epochs at `lr=1e-3` on the twelve prompts.

This is an SFT-like base, not a full-sequence reconstructor. After warm-up the
reference is switched to eval mode, frozen, and copied into the policy.

DPO training then runs 30 full-batch steps:

| Setting | Value |
|---|---|
| `beta` | `0.1` |
| optimizer | Adam |
| learning rate | `1e-5` |
| steps | 30 |
| batch | all 12 triples |
| reference log-probs | precomputed once, detached |
| policy log-probs | recomputed each step with gradients |

The learning rate is deliberately small. The reward is a sum of roughly 10–100
token log-probabilities, so raw log-probability differences are large; a larger
Adam step saturates the sigmoid within a few updates. At `lr=1e-5` the loss
moves gradually from `log(2)` to about `0.028` over 30 steps.

The demo prints one line per step:

```text
step   1: loss=0.69315 margin=335.2610 reward_delta=0.0000
```

`margin` is the raw mean chosen-minus-rejected log-probability under the policy.
`reward_delta` is the mean `z`, the implicit reward margin. The first loss is
exactly `log(2)` because the policy and reference start with identical weights.

## Quickstart

Prerequisites: Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/). From this
module's directory:

```bash
uv sync --frozen --all-groups
uv run --frozen python code/main.py
```

The first command installs the pinned CPU environment into `.venv/`. The demo
writes no checkpoints or result files; Python may create ignored `__pycache__/`
bytecode.

A fast smoke run with a tiny model:

```bash
uv run --frozen python code/main.py \
  --warmup-epochs 2 --steps 2 \
  --hidden 16 --num-heads 2 --num-layers 1
```

Run the complete gate:

```bash
./verify.sh
```

`verify.sh` syncs the frozen lockfile, runs the tests and strict typecheck, then
runs the default demo. It is safe to run twice and installs only into the
ignored `.venv/`.

PyTorch may print a one-time `Failed to initialize NumPy` warning because this
module intentionally has no NumPy dependency. It is an optional PyTorch
integration warning and does not affect execution. The CLI pins PyTorch to one
CPU thread so its printed measurements are reproducible across processes;
library functions use the caller's thread settings.

## What runs offline

Everything after dependency installation is offline:

- the twelve triples are constructed in memory;
- the warm-up, DPO loop, log-probability computation, and margin evaluation are
  local PyTorch CPU operations;
- tests use no network, no downloaded fixtures, and no external checkpoints;
- no API key, provider account, cloud sandbox, or GPU is used.

`uv sync` may download the pinned CPU wheel on a fresh machine; that is the only
network boundary.

## Optional GPU / real-provider mode

No paid API, external dataset, GPU, or cloud service is required. The local
implementation is the real DPO mechanism, not a simulation of a remote
service. What is simulated is only the preference source: the twelve triples
are a deterministic fixture standing in for a human-labelled preference
dataset.

To plug in a real preference dataset, replace `make_preferences()` with a
loader that returns `Preference` objects with the same four fields. To use a
larger or pretrained model, replace `TinyGPT` with any causal LM that exposes a
`forward(token_ids) -> logits` method, and keep the reference/policy pair
sharing one architecture. A GPU port would move the tensors and parameters with
`.to(device)` and update the warm-up and DPO loops; it is not implemented or
measured here, and no GPU numbers are claimed.

## Measured results

Measured locally on CPU with the pinned environment and the CLI's
single-thread setting. This is one seeded run, not a benchmark. The default
demo took 7.2 s wall time; the complete `verify.sh` gate took 15.5 s.

```text
dpo-from-scratch: seed=0 pairs=12 max_len=128
data: capital=4 arithmetic=4 list=4
model: parameters=259296 hidden=96 heads=4 layers=2
warmup: epochs=60 loss=5.5945 -> 0.3769
reference: chosen_mean=-5.9020 rejected_mean=-341.1630 margin=335.2610
dpo: beta=0.1000 lr=1e-05 steps=30
step   1: loss=0.69315 margin=335.2610 reward_delta=0.0000
step   2: loss=0.61492 margin=336.8943 reward_delta=0.1633
step   3: loss=0.54311 margin=338.5371 reward_delta=0.3276
step   4: loss=0.47785 margin=340.1853 reward_delta=0.4924
step   5: loss=0.41910 margin=341.8345 reward_delta=0.6574
step   6: loss=0.36669 margin=343.4797 reward_delta=0.8219
step   7: loss=0.32032 margin=345.1158 reward_delta=0.9855
step   8: loss=0.27959 margin=346.7377 reward_delta=1.1477
step   9: loss=0.24404 margin=348.3402 reward_delta=1.3079
step  10: loss=0.21317 margin=349.9185 reward_delta=1.4658
step  11: loss=0.18647 margin=351.4680 reward_delta=1.6207
step  12: loss=0.16345 margin=352.9845 reward_delta=1.7724
step  13: loss=0.14364 margin=354.4647 reward_delta=1.9204
step  14: loss=0.12660 margin=355.9055 reward_delta=2.0644
step  15: loss=0.11196 margin=357.3044 reward_delta=2.2043
step  16: loss=0.09937 margin=358.6595 reward_delta=2.3399
step  17: loss=0.08852 margin=359.9697 reward_delta=2.4709
step  18: loss=0.07917 margin=361.2339 reward_delta=2.5973
step  19: loss=0.07109 margin=362.4517 reward_delta=2.7191
step  20: loss=0.06408 margin=363.6231 reward_delta=2.8362
step  21: loss=0.05800 margin=364.7483 reward_delta=2.9487
step  22: loss=0.05271 margin=365.8280 reward_delta=3.0567
step  23: loss=0.04808 margin=366.8628 reward_delta=3.1602
step  24: loss=0.04403 margin=367.8538 reward_delta=3.2593
step  25: loss=0.04047 margin=368.8021 reward_delta=3.3541
step  26: loss=0.03733 margin=369.7091 reward_delta=3.4448
step  27: loss=0.03455 margin=370.5760 reward_delta=3.5315
step  28: loss=0.03209 margin=371.4044 reward_delta=3.6143
step  29: loss=0.02989 margin=372.1956 reward_delta=3.6935
step  30: loss=0.02794 margin=372.9513 reward_delta=3.7690
final: policy_chosen=-8.1316 policy_rejected=-381.8047 margin=373.6730
check: ref_unchanged=true loss_decreased=true margin_increased=true
```

The 30-step run reduces the DPO loss to 4.0% of its initial value (0.69315 to
0.02794). The raw chosen-minus-rejected margin grows from 335.26 to 373.67.
The absolute numbers also show the failure mode discussed in the TRL docs: the
policy's chosen log-probability falls from −5.9020 to −8.1316, but the rejected
log-probability falls much more, from −341.1630 to −381.8047. Training is
mostly suppressing the rejected completions rather than raising the chosen
ones in absolute terms. The relative gap is what DPO optimizes.

Verification on the same machine:

```text
24 passed, 1 warning in 8.14s
Success: no issues found in 2 source files
verify: ok
```

The warning is PyTorch's optional-NumPy notice.

## Design decisions

- **Response-only warm-up.** It mirrors an SFT reference on preferred
  completions and uses the same completion boundary that DPO scores. The
  `[RESP]` label is masked; the boundary position predicts the first completion
  byte.
- **Frozen reference, precomputed log-probs.** The reference never receives
  gradients, and its four log-probability vectors are computed once under
  `torch.no_grad()`. Tests recompute the reference after training to prove
  invariance.
- **Exact-copy initialization.** The policy is loaded from
  `reference.state_dict()`; the first DPO step has `z = 0` and loss `log(2)`.
- **Unnormalized sum log-probs.** This is the lesson's core DPO objective, not
  the length-normalized SimPO variant. The raw sums make the reward scale
  large, which is why the DPO learning rate is small.
- **No newline/EOS.** The module only scores completions; it never generates.
  Adding a terminator would silently change the four log-probabilities.
- **Full-batch training.** Twelve triples fit in one batch, so there is no
  shuffling or loader randomness. The CLI pins one CPU thread so printed
  measurements are reproducible across processes.
- **No NumPy.** The fixture, tokenizer, transformer, loss, and metrics are pure
  Python/PyTorch. One runtime dependency keeps the module easy to verify.

## Limitations

- The fixture is synthetic and tiny: twelve pairs, no held-out preference
  split, and evaluation only on the training prompts. The margin growth is not
  a generalization claim.
- The reference is a randomly initialized 259k-parameter transformer warmed on
  twelve preferred completions, not a pretrained SFT model. It is an
  educational stand-in for the SFT base in the DPO derivation.
- Sum log-probabilities are strongly length-sensitive. The chosen completions
  are shorter than the rejected ones, so the initial raw margin is already
  large and DPO mostly lowers the rejected probability. Length normalization
  is not implemented.
- There is no reward model, PPO baseline, generation quality evaluation, win
  rate, or beta sweep. The measured margin is a training diagnostic, not a
  human-preference benchmark.
- The stretch goals are intentionally skipped: length normalization, IPO,
  label smoothing, and a smaller distilled reference. The README cites IPO and
  length-bias sources because they explain the limitations, not because those
  variants are implemented.
- No GPU path or real-provider adapter is measured. The tensor code is
  device-aware where practical, but only CPU numbers are reported.
- The module has no NumPy dependency; PyTorch's optional-NumPy warning is
  expected.

## Tests

`tests/test_main.py` contains 24 test cases across 17 test functions:

- tokenizer specials, UTF-8 round trips, completion starts, and invalid input;
- fixture size, category balance, uniqueness, chosen/rejected length ordering,
  context fit, and determinism;
- `Preference` validation;
- model shape, tied head, parameter count, causal invariance, and input
  validation;
- manual completion-log-prob equivalence and gradient flow;
- DPO loss formula, fixed-value check, implicit reward delta, gradient signs,
  numerical stability, and shape/beta validation;
- response-only warm-up loss reduction and chosen log-prob improvement;
- reference copy/freeze invariants and reference invariance after DPO updates;
- first-loss `log(2)`, loss decrease, and margin increase;
- margin evaluation and mode restoration;
- CLI output contract, invalid configuration handling, and in-process
  determinism.

```bash
uv run --frozen pytest
uv run --frozen mypy code/main.py tests
```

## References

1. AI Engineering From Scratch, [Capstone Lesson 40: Direct Preference
   Optimization from Scratch](https://aiengineeringfromscratch.com/learn/capstone-projects/dpo-from-scratch)
   — task contract, twelve-triple fixture, loss derivation, reference
   invariance, and demo requirements.
2. Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D.
   Manning, Chelsea Finn, [Direct Preference Optimization: Your Language Model
   is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (NeurIPS 2023)
   — Bradley-Terry derivation, closed-form optimal policy, DPO loss, and the
   gradient interpretation.
3. Hugging Face TRL, [DPO Trainer](https://huggingface.co/docs/trl/main/dpo_trainer)
   — loss formula, precomputed reference log-probs, implicit-reward metrics, and
   the observation that training often suppresses dispreferred completions more
   than it raises preferred ones.
4. Mohammad Gheshlaghi Azar et al., [A General Theoretical Paradigm to Understand
   Learning from Human Preferences](https://arxiv.org/abs/2310.12036) (IPO)
   — identity-loss/overfitting caveat referenced in the limitations.
5. [Eliminating Biased Length Reliance of Direct Preference Optimization via
   Down-Sampled KL Divergence](https://arxiv.org/abs/2406.10957)
   — length-reliance context for the skipped length-normalization stretch goal.

## License

MIT License, Copyright (c) 2026 Pedro Monteiro. See [LICENSE](LICENSE).
