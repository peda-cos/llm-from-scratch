# BPE Tokenizer From Scratch

## Overview

A byte-level Byte Pair Encoding tokenizer in pure Python. It trains a merge table from a raw text corpus, applies that table in learned rank order to fresh text, and decodes the resulting ids back to exactly the same UTF-8 string.

The first 256 ids are the raw bytes, so every input string is representable before any merge happens. Two special ids sit immediately after the byte block, and the remaining vocabulary is learned merges. No runtime dependencies; the implementation is stdlib-only.

## Architecture

```
corpus text
  -> pretokenize (Unicode words | whitespace runs | single punctuation marks)
  -> byte-split each distinct pretoken
  -> count adjacent pairs, weighted by pretoken frequency
  -> merge the most frequent pair into the next free id
  -> append the merge to the ordered table and repeat
  -> encode fresh text by repeatedly applying the lowest-ranked applicable merge
  -> decode by concatenating each id's byte expansion
```

### Vocabulary layout

| ids | contents |
| --- | --- |
| `0..255` | raw bytes `0x00..0xFF` |
| `256` | `<|endoftext|>` |
| `257` | `<|pad|>` |
| `258+` | learned merges, in rank order |

`vocab_size` counts all of them: a target of 300 is 42 merges after the 258 base ids.

### Training

`BPETokenizer.train(text, vocab_size)` counts every distinct pretoken once, then repeats three operations until the target is reached:

1. Count adjacent symbol pairs across all pretokens, weighted by pretoken frequency.
2. Pick the highest-count pair. Ties are broken by the lexicographically smallest pair of ids, which makes training fully deterministic.
3. Rewrite every occurrence of that pair in every pretoken and assign the next free id, recording `(left, right, new_id)` in `self.merges`.

The 256-byte alphabet and the two special ids are present before the loop; new ids start at 258. Special ids are never symbols in the corpus or inputs to a merge, so training cannot consume or overwrite them. If the corpus runs out of pairs before the target, training stops and `vocab_size` reports the actual size.

This is the straightforward implementation: each merge step recounts the whole corpus. It is easy to inspect and intentionally not an incremental or parallel trainer.

### Encoding

`encode(text, allow_special=False)` pretokenizes the text and encodes each pretoken independently:

1. Expand the pretoken to its UTF-8 bytes.
2. Find the applicable merge with the smallest rank (the earliest learned).
3. Apply it to all non-overlapping occurrences, left to right.
4. Scan again; stop when no learned pair applies.

The rank order is the contract shared with training. Encoded pretokens are cached per tokenizer instance; `train` and `load` clear the cache.

With `allow_special=True`, occurrences of `<|endoftext|>` and `<|pad|>` are intercepted before pretokenization and emitted as ids 256/257. They are atomic and never merged. With the flag off, the literal strings are tokenized as ordinary bytes.

### Decoding and round-trip

Every id maps to bytes. Byte ids map to one byte; a merge id maps to the concatenation of its two parents; special ids map to the literal marker's UTF-8 bytes. `decode` concatenates the expansions in order and decodes strict UTF-8. Because every id ultimately expands to bytes from a UTF-8 input string, encoding then decoding returns the original string exactly. Special markers are returned literally rather than dropped, which keeps the round-trip property.

### Persistence

`save(path)` writes versioned JSON: the ordered merge triples, special table, byte-vocab size, and total vocabulary size. `load(path)` rebuilds token bytes and ranks, rejecting malformed JSON, unexpected formats, wrong ids, duplicate pairs, and merges that reference special ids.

## Quickstart

Requires `uv` and Python 3.11 or newer.

```bash
cd 30-bpe-tokenizer-from-scratch
./verify.sh              # sync pinned dev tools, run tests + typecheck + demo
uv run python main.py    # demo at the default vocabulary of 300
```

The default demo trains on `data/corpus.txt`, encodes a held-out sentence, prints the ids and decoded text, and checks the round-trip.

Show the compression curve (300 then 600):

```bash
uv run python main.py demo --curve
```

Run a single target size, including 600 as in the lesson:

```bash
uv run python main.py demo --vocab-size 600
```

Train once, then reuse the merge table for inference:

```bash
mkdir -p artifacts
uv run python main.py train --out artifacts/bpe-600.json --vocab-size 600

IDS=$(uv run python main.py encode --model artifacts/bpe-600.json "Hello, world!")
uv run python main.py decode --model artifacts/bpe-600.json $IDS
# Hello, world!
```

Special-token path:

```bash
uv run python main.py encode --model artifacts/bpe-600.json --allow-special "<|endoftext|>hello"
# starts with 256
uv run python main.py decode --model artifacts/bpe-600.json 256
# <|endoftext|>
```

Use another corpus with `--corpus PATH` on `demo` or `train`. A missing corpus/model file exits with status 2 and a single `error:` line.

## Offline and real-service boundary

Everything in this module runs offline on CPU. Runtime is standard-library Python; tests read the committed corpus and temporary files only, and never make network calls. There is no paid API, GPU, or external tokenizer service in the lesson, so no provider adapter is simulated and no real-service mode is hidden behind a fixture. `verify.sh` may download the pinned pytest and mypy packages into the ignored `.venv/` on its first run; later runs use the local environment.

## Measured results

Measured locally with Python 3.14.10 on Linux 7.3.4-1-cachyos x86_64. Command: `uv run python main.py demo --curve`. These are single-run local measurements, not benchmark claims.

Corpus: an excerpt from *Alice's Adventures in Wonderland* (Lewis Carroll, 1865), Project Gutenberg eBook #11. The Project Gutenberg header and footer were removed, then the first 24,000 characters were cut at a newline boundary: 23,945 characters, 4,559 whitespace-separated words, 10,348 pretokens, 1,092 unique pretokens. The text is public domain. It was retrieved through the [GITenberg mirror](https://github.com/GITenberg/Alice-s-Adventures-in-Wonderland_11) because `gutenberg.org` returned HTTP 503 from this environment.

Held-out sentence (78 characters, unseen during training):

> Tokenizers turn text into integers; the model never sees a single character. 🚀

| target vocab | actual vocab | merges | held-out ids | chars/token | train time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 300 | 42 | 64 | 1.219 | 0.085 s |
| 600 | 600 | 342 | 53 | 1.472 | 0.789 s |

The 600-id tokenizer emits **53 ids instead of 64**, a 17.2% reduction on the same sentence. Both decodes reproduce the sentence exactly. The test suite collects 27 cases across 16 test functions. Token counts are deterministic for the committed corpus; wall-clock timings vary between runs.

## Design decisions

- **Specials before merges.** The capstone spec puts `<|endoftext|>` and `<|pad|>` at ids 256 and 257, so learned merges start at 258. Some production tokenizers instead append specials after the merge block; this module follows the spec.
- **Deterministic tie-break.** The spec says to pick the most frequent pair but does not define ties. This implementation picks the smallest `(left_id, right_id)` lexicographically, and the tests pin that choice.
- **Small pretokenizer.** The regex `\s+|\w+|[^\w\s]` groups Unicode word runs and whitespace, and splits punctuation one character at a time. It is deliberately not the GPT-2/GPT-4 regex pretokenizer; it is enough to keep merges inside words while preserving every character for round-trip.
- **Special literals decode literally.** Decoding a special id returns its marker (`<|endoftext|>`) instead of dropping it. Dropping would require an explicit `skip_special_tokens` option and would weaken the round-trip guarantee.
- **JSON models, no pickle.** The model is a versioned, inspectable JSON document. Loading validates ids and merge structure before rebuilding state.
- **Cached pretoken encoding.** The cache is keyed by pretoken string and invalidated whenever training or loading changes the merge table.

## Limitations

- Training is O(merges × corpus) with a full pair recount per step. It is fast enough for the committed 24k-character corpus (well under a second at 600 ids) but not intended for million-token production corpora. Incremental pair updates or parallel counting would be the next step.
- Encoding is deterministic BPE, not BPE-dropout; there is no stochastic subword regularization.
- Only the two lesson special tokens are reserved. Adding more is a code change, not a configuration option.
- Input is a Python `str` containing valid Unicode. `decode` uses strict UTF-8 and raises on ids that do not form valid UTF-8; it does not silently replace invalid bytes.
- The pretokenizer is intentionally small: no digit grouping, contractions, or language-specific rules.
- Training stops early if the corpus contains no adjacent pair to merge; the requested target is not padded with arbitrary ids.

## References

1. AI Engineering From Scratch, [BPE Tokenizer From Scratch capstone](https://aiengineeringfromscratch.com/learn/capstone-projects/bpe-tokenizer-from-scratch) — task framing, byte alphabet, special-token behavior, and compression-curve goal.
2. Sebastian Raschka, [Implementing A Byte Pair Encoding (BPE) Tokenizer From Scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html) — merge-table training, rank-ordered encoding, and byte-level vocabulary rationale.
3. Hugging Face, [Tokenization algorithms](https://huggingface.co/docs/transformers/en/tokenizer_summary) — byte-level BPE as the 256-byte base that avoids an unknown token; vocabulary = base tokens + merges.
4. Andrej Karpathy, [minbpe](https://github.com/karpathy/minbpe) — byte-level BPE, pretokenization to avoid merges across categories, and opt-in special-token handling. This implementation is original and differs in special-id placement, pretokenizer, persistence format, and tie-break.
5. Project Gutenberg, [Alice's Adventures in Wonderland, eBook #11](https://www.gutenberg.org/ebooks/11) — public-domain training corpus, retrieved via the [GITenberg mirror](https://github.com/GITenberg/Alice-s-Adventures-in-Wonderland_11).

## License

MIT License, Copyright (c) 2026 Pedro Monteiro. See [LICENSE](LICENSE). The Project Gutenberg excerpt is public domain.
