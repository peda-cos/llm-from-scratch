# Tokenized Dataset with Sliding Window

## Overview

A causal language-model pretraining step is a function from token ids to gradients. This module builds the conveyor that feeds the ids in: a corpus is encoded once, sliced into overlapping fixed-length windows, exposed as input/target tensor pairs, and batched with a shuffle that is reproducible from a base seed and an epoch index.

The implementation is one file, `main.py`, plus a committed public-domain corpus and a pytest suite. It uses PyTorch for `Dataset`, `DataLoader`, tensors, and the seeded generator; NumPy is pinned because PyTorch imports its optional NumPy bridge at startup, but no code here calls NumPy directly.

## Architecture

```
data/corpus.txt
  -> ByteTokenizer.encode(text)                  (exactly once)
  -> flat int64 id tensor, shape (N,)
  -> SlidingWindowDataset(T, stride)
       __getitem__(i) -> window = ids[i*S : i*S + T + 1]
                       -> (input = window[:-1], target = window[1:])
  -> make_dataloader(seed = base_seed + epoch_index, B)
  -> batches: input (B, T), target (B, T)
```

### Shape and shift contract

Every training example covers `T + 1` raw ids. The input is the first `T`; the target is the same window shifted left by one. Position `t` in the target is the id that follows position `t` in the input, so `input[:, 1:] == target[:, :-1]` for every batch. The demo asserts this for every batch it streams.

The number of complete windows is

```text
max(0, 1 + (N - (T + 1)) // S)
```

where `N` is the id-stream length, `T` the context length, and `S` the stride. `SlidingWindowDataset.num_examples` exposes this as a static method so a trainer can compute steps per epoch without iterating. The tail is dropped rather than padded, so no loss mask is needed.

### Sliding window and stride

A stride of `T` gives non-overlapping windows. A stride of `T // 2` reuses half of each window and roughly doubles the example count; a stride of `1` gives every possible start position and roughly multiplies the count by `T`. Boundary effects make the exact ratios slightly smaller than these multipliers.

The slicer returns views into the one stored id tensor. It does not allocate a per-example table, so the dataset's id memory is `N * 8` bytes regardless of how many windows a small stride creates. The cost of a small stride is more examples per epoch, not more id storage.

### Deterministic shuffle

`make_dataloader` creates a local `torch.Generator` and seeds it with `base_seed + epoch_index`. The trainer rebuilds the loader at the top of each epoch with the next index. Consequently:

- two fresh loaders with the same `(base_seed, epoch_index)` produce the same index order;
- `base_seed=100, epoch_index=3` is exactly the same order as `base_seed=103, epoch_index=0`;
- a different epoch normally produces a different order while covering the same indices;
- no global RNG is touched.

A loader is intentionally single-epoch: iterating the same loader twice advances its generator. Rebuild it per epoch. PyTorch's `RandomSampler` rejects a zero-length dataset, so an empty dataset falls back to sequential iteration (which yields no batches).

Batches are collated by PyTorch's default collator, which stacks the per-example tensors into `(B, T)` long tensors. The final batch is smaller unless `drop_last=True`; the demo uses `drop_last=True` so every batch has the full shape.

### Code map

- `Tokenizer` / `ByteTokenizer` — the one-method encoder contract and the exact UTF-8 byte implementation.
- `_encode_corpus_to_ids` — the single `tokenizer.encode(text)` call that produces the flat `int64` stream.
- `SlidingWindowDataset` — validation, the window-count formula, and the shifted `(input, target)` views.
- `epoch_seed` / `make_dataloader` — `base_seed + epoch_index` and the single-epoch seeded `DataLoader`.
- `_run_epoch` / `_run_demo` / `main` — streaming, contract assertions, stride sweep, determinism replay, and the CLI.
- `tests/test_dataset.py` — the executable specification for every behavior above.

## Quickstart

Requires `uv` and Python 3.11 or newer. From the repository root:

```bash
cd 31-tokenized-dataset-sliding-window
./verify.sh
```

`verify.sh` syncs the frozen lockfile, runs the tests, runs strict mypy, and exercises the demo at context lengths 16 and 32. It exits 0 and is safe to run repeatedly.

Run the demo directly:

```bash
uv run python main.py demo
uv run python main.py demo --context-length 32
uv run python main.py demo --stride 1 --epochs 1
uv run python main.py demo --batch-size 16 --seed 7 --epochs 3
```

Default demo configuration: `T=16`, stride `T`, `B=8`, two epochs, base seed 1234, `drop_last=True`. The first run downloads the pinned CPU wheels into `.venv/`; later runs work offline.

| flag | default | meaning |
| --- | --- | --- |
| `--corpus PATH` | `data/corpus.txt` | UTF-8 input text |
| `--context-length` | `16` | input/target length `T` |
| `--stride` | `T` | window advance `S` |
| `--batch-size` | `8` | examples per batch `B` |
| `--epochs` | `2` | number of seeded epochs to stream |
| `--seed` | `1234` | base seed for `base_seed + epoch` |

A missing or non-UTF-8 corpus, an empty window set, or a batch larger than the dataset exits with status 2 and one `error:` line.

## Offline, GPU, and real-tokenizer boundary

Everything in this module runs offline on CPU. The tests and demo make no network calls; `verify.sh` downloads pinned packages only when the ignored `.venv/` is empty. There is no paid provider, remote service, or simulated service response in this lesson.

The committed manifest pins the official CPU-only PyTorch wheel (`torch==2.15.0+cpu`, about 209 MB) so verification does not pull the PyPI wheel's CUDA dependency stack. For GPU training, install a CUDA-enabled PyTorch build instead, move each batch to the device in the training loop, and keep this dataset/loader code unchanged. GPU execution is not exercised or measured here.

The previous lesson's BPE tokenizer is not imported: modules in this track must be independently verifiable. `Tokenizer` is a one-method protocol (`encode(text) -> list[int]`), and `_encode_corpus_to_ids` calls it exactly once. Pass the BPE tokenizer, a Hugging Face tokenizer wrapper, or any other encoder implementing that method; no dataset or loader change is required. The built-in `ByteTokenizer` is the deterministic in-process stand-in: it maps UTF-8 bytes to ids `0..255` and decodes exactly. It is intentionally not a subword tokenizer, so its token counts are byte counts, not BPE token counts.

`num_workers` is fixed at 0, matching the lesson. Production loading would expose `num_workers` and add a `worker_init_fn`; this pipeline's `__getitem__` is only a slice, so worker parallelism would mostly add process overhead.

## Measured results

Measured locally with Python 3.14.10, `torch==2.15.0+cpu`, and Linux 7.3.5 (x86_64). These are single-run local observations, not benchmark claims; wall-clock values vary between runs.

Corpus: 23,945 ASCII characters/bytes, 23,945 byte-token ids, vocabulary 256. The id tensor is `23,945 * 8 = 191,560` bytes (about 0.19 MB). The demo command below uses `drop_last=True`, so the last incomplete batch is excluded from the streamed example count.

```text
$ uv run python main.py demo --context-length 16 --epochs 2
examples_per_epoch=1496
batches_per_epoch=187
encode_elapsed=0.001418s
stride_sweep (context_length=16):
  stride=16 examples=1496 overlap=1.00x
  stride=8 examples=2992 overlap=2.00x
  stride=4 examples=5983 overlap=4.00x
  stride=1 examples=23929 overlap=16.00x
epoch 0: examples=1496 elapsed=0.0097s throughput=154814.0 examples/s
epoch 1: examples=1496 elapsed=0.0089s throughput=168654.0 examples/s
order_check=ok (epoch 0 replay with base_seed=1234)
epoch_order_change=ok
shift_contract=ok
```

| context `T` | stride `S` | examples/epoch | batches (`B=8`, drop last) | streamed examples | epoch time | throughput |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 16 | 1496 | 187 | 1496 | 0.0097 s | 154,814 examples/s |
| 32 | 32 | 748 | 93 | 744 | 0.0055 s | 135,504 examples/s |
| 16 | 1 | 23,929 | 2,991 | 23,928 | 0.2200 s | 108,779 examples/s |

Changing `T` from 16 to 32 reduces the number of examples per epoch from 1496 to 748, while the batch width doubles. The stride-1 run shows the other side of the trade-off: roughly 16x the examples, with a measured epoch time of 0.2200 s versus 0.0097 s for the non-overlapping run.

The test suite collects 37 passing cases (0.78–1.22 s across local runs) and strict mypy reports no issues. `./verify.sh` completes with `verify: ok`.

## Design decisions

- **Byte tokenizer, not a duplicated BPE trainer.** Module boundaries forbid importing lesson 30. A protocol plus a 256-id byte tokenizer keeps the lesson focused on windowing, batching, and seeding while accepting the real BPE tokenizer unchanged.
- **Stride defaults to `T`.** Non-overlapping windows are the common pretraining default; overlap is opt-in because it multiplies work.
- **One id tensor, view-based examples.** `__getitem__` returns slices, not copies, so a stride of 1 does not increase storage. The `T + 1`th id is a target only.
- **Explicit seed arithmetic.** The seed is a local generator, not the global RNG. The factory's `(base_seed, epoch_index)` contract is pinned by tests, including the seed-offset equivalence.
- **Strict input validation.** Id tensors must be one-dimensional non-negative integers; window parameters must be positive ints; empty streams yield zero examples rather than a partial window.
- **Drop the tail.** Padding would require a loss mask and is deliberately left to a later lesson. The demo's `drop_last=True` gives exact `(B, T)` batches; the default factory keeps the partial tail.
- **CPU-only pinned wheel.** The official CPU index keeps `verify.sh` practical. NumPy is pinned because PyTorch imports its optional NumPy bridge at startup; this code does not otherwise use it.
- **Corpus.** A public-domain excerpt from *Alice's Adventures in Wonderland* (Project Gutenberg eBook #11) is committed so tests and demos are deterministic and offline.

## Limitations

- The whole id stream is held in one in-memory tensor; there is no disk streaming, memory-mapping, or sharded dataset. At `N` ids the storage is about `8N` bytes.
- The tail is always dropped. There is no padding, packing, or loss mask.
- The stream is one continuous text. Document boundaries and `<|endoftext|>` insertion are not handled; multi-document packing would insert boundary ids before encoding or between documents.
- `ByteTokenizer` is a byte-level stand-in, not the lesson 30 BPE. One token is one byte for ASCII, so token counts are not comparable to subword-tokenizer counts.
- `num_workers` is fixed at 0; worker seeding and the production prefetch path are not demonstrated.
- Deterministic orders are guaranteed within the pinned environment. PyTorch explicitly does not guarantee reproducibility across releases, platforms, or CPU/GPU executions, even with identical seeds.
- The module stops at the data pipeline. It does not define a transformer, a loss, an optimizer, or a training loop; throughput numbers measure slicing and collation, not model compute.
- The committed corpus is a small lesson fixture, not a production-scale pretraining corpus.

## References

1. AI Engineering From Scratch, [Tokenized Dataset with Sliding Window capstone](https://aiengineeringfromscratch.com/learn/capstone-projects/tokenized-dataset-sliding-window) — shape contract, sliding-window semantics, stride trade-off, seed contract, and tail policy.
2. PyTorch, [`torch.utils.data` documentation](https://docs.pytorch.org/docs/2.15/data.html) — map-style datasets, default collation, `DataLoader` batching, and the `generator` argument used for seeded shuffling.
3. PyTorch, [Reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html) — local generator seeding, worker reseeding, and the limits of cross-platform determinism.
4. Sebastian Raschka, [How are input-target training examples constructed for LLM pretraining?](https://sebastianraschka.com/faq/docs/input-target-pretraining-examples.html) — `L+1`-token windows, shift-by-one targets, stride/overlap, and the cost of short-tail handling.
5. Project Gutenberg, [*Alice's Adventures in Wonderland*, eBook #11](https://www.gutenberg.org/ebooks/11) — public-domain corpus, retrieved through the [GITenberg mirror](https://github.com/GITenberg/Alice-s-Adventures-in-Wonderland_11) and previously committed in this track; the excerpt here is ASCII and unmodified.

## License

MIT License, Copyright (c) 2026 Pedro Monteiro. See [LICENSE](LICENSE). The Project Gutenberg excerpt is public domain.
