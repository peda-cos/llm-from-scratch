# HDF5 Tokenized Corpus

A streaming tokenizer and reader for language-model pretraining corpora. Documents
are encoded once, buffered in memory, and appended to resizable, chunked HDF5
datasets in chunk-sized strides. Each input shard becomes one HDF5 file; a
`shards.json` index records token counts, document counts, SHA-256 digests, and
byte sizes. At training time a worker opens its shard subset with SWMR reads,
concatenates the selected shards into one logical token stream, and samples
fixed-length `(input, target)` windows with an explicit document-boundary token.

The implementation is Python 3.12+ with pinned `h5py==3.16.0` and
`numpy==2.5.5`. Everything in the demo and test suite runs offline.

## Quickstart

Prerequisites: [`uv`](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
cd 43-hdf5-tokenized-corpus
uv sync --frozen --all-groups
uv run --frozen python code/main.py
```

The zero-argument command builds the tiny in-memory corpus, writes two HDF5
shards, opens them through the chunked reader, prints ten batch shapes and
checksums, and validates the index:

```text
corpus: shards=2 documents=6 tokens=316 chunk_size=64
  shard-0000: tokens=155 docs=3 bytes=11008 sha256=7ce703dc3c2c121e...
  shard-0001: tokens=161 docs=3 bytes=11008 sha256=14c10e7eb0d6307f...
build: 0.0093s
batch=00 input=(4, 16) target=(4, 16) checksum=32913afef55a84a2
batch=01 input=(4, 16) target=(4, 16) checksum=09adfd8ac4c2bf9a
batch=02 input=(4, 16) target=(4, 16) checksum=0c7ff7c534f4de4b
batch=03 input=(4, 16) target=(4, 16) checksum=2106dc51fa33671c
batch=04 input=(4, 16) target=(4, 16) checksum=c2871de40d4f444b
batch=05 input=(4, 16) target=(4, 16) checksum=1752c86fe8374cef
batch=06 input=(4, 16) target=(4, 16) checksum=c462501f82f828e9
batch=07 input=(4, 16) target=(4, 16) checksum=4cee1bc87d77097b
batch=08 input=(4, 16) target=(4, 16) checksum=9fc7b3ff65e9f278
batch=09 input=(4, 16) target=(4, 16) checksum=f1ee0b47db9d5761
read: 0.0104s samples=10 samples_per_second=961.5 tokens_per_second=61538.8
validated: shards=2 tokens=316 bytes=22016
index=outputs/demo/shards.json
```

Timing lines vary between runs; the batch checksums do not.

Run the complete gate (frozen install, tests, strict mypy, demo, validation):

```bash
./verify.sh
```

`verify.sh` is safe to run repeatedly: `outputs/`, `.venv/`, and caches are
ignored, and every artifact uses a fixed replaceable name.

Other commands:

```bash
# Tokenize a directory of lesson-42 JSONL shards into HDF5.
uv run --frozen python code/main.py build \
  --input-dir ../42-large-corpus-downloader/outputs/demo/corpus \
  --workdir outputs/corpus --chunk-size 8192

# Recompute every shard SHA-256 and compare it with shards.json.
uv run --frozen python code/main.py validate --index outputs/corpus/shards.json

# Compare chunk sizes at half, equal, and double the window.
uv run --frozen python code/main.py benchmark \
  --workdir outputs/benchmark --window 128 --batch-size 16 --samples 1000
```

## Architecture

```mermaid
flowchart TD
  JSONL[JSONL documents] --> Tok[ByteTokenizer.encode]
  Tok --> Buf[Buffer: chunk_size uint32 ids]
  Buf --> Full{buffer full?}
  Full -- no --> JSONL
  Full -- yes --> Resize[resize dataset + chunk_size]
  Resize --> Write[write contiguous chunk + flush]
  Write --> Buf
  Buf --> Close[close: write residual, set attrs]
  Close --> Shard[HDF5 shard]
  Shard --> Index[shards.json]
  Index --> Store[MmapTokenStore<br/>per-worker shard subset]
  Store --> Slice[get_slice(start, stop)]
  Slice --> Window[SlidingWindowDataloader]
  Window --> Batch[(input, target) uint32]
```

### On-disk format

Each shard is one HDF5 file with one dataset:

| Property | Value |
| --- | --- |
| Dataset | `tokens` |
| Dtype | little-endian `uint32` (`<u4`) |
| Initial shape | `(0,)` |
| Maximum shape | `(None,)` |
| Chunks | `(chunk_size,)` |
| Layout | chunked; optional gzip filter |
| Attributes | `token_count`, `sha256` |

`token_count` is the number of real tokens, not the logical dataset length.
The store always truncates reads at this attribute, so a padded or externally
grown dataset cannot leak trailing zeros into a training batch. The writer
records the exact length, so the normal shape and attribute agree. The SHA-256
is over the token bytes in append order as canonical little-endian uint32,
with no padding and no header.

Every document is followed by one reserved boundary token. The byte tokenizer
uses ids `0..255` for UTF-8 bytes and id `256` for the separator, so
`vocab_size` is 257. Consecutive shards therefore still have a separator
between the last document of one shard and the first document of the next.
The boundary token costs one id per document; the dataloader never enforces
document boundaries and may return a window that straddles two documents.

`shards.json` is the source of truth. The fields are:

```json
{
  "schema_version": 1,
  "generator": "hdf5-tokenized-corpus/0.1.0",
  "chunk_size": 64,
  "vocab_size": 257,
  "boundary_token_id": 256,
  "total_tokens": 316,
  "shards": [
    {
      "shard_id": "shard-0000",
      "path": "shard-0000.h5",
      "token_count": 155,
      "document_count": 3,
      "sha256": "7ce703dc3c2c121e4b6bb04250b816fd964bfa14ebaba1278ea724fc46051d8a",
      "byte_size": 11008
    },
    {
      "shard_id": "shard-0001",
      "path": "shard-0001.h5",
      "token_count": 161,
      "document_count": 3,
      "sha256": "14c10e7eb0d6307f7efeb82e92227e33c0490387656d30790d97989cca219d1a",
      "byte_size": 11008
    }
  ]
}
```

Shard order defines the global stream. The reader computes offsets by
cumulative token counts; it does not store redundant offsets in the index.
Paths are bare filenames relative to the index directory. Duplicate ids or
paths, traversal, non-lowercase hashes, a bad total, and a wrong schema are
rejected when the index is loaded.

### Write discipline

`HDF5ShardWriter` allocates a NumPy buffer of exactly `chunk_size` ids. A
document append copies only the bytes that fit; when the buffer fills, the
writer performs **one** `resize` by exactly `chunk_size`, writes the buffer into
the new range, updates the running SHA-256, and calls `dataset.flush()`. It
never resizes per document and never materialises the whole shard in memory.
At close, a residual buffer is written into the final partial range, then the
attributes are recorded.

The writer opens every file with `libver="latest"`, creates the dataset and
attribute placeholders, and only then enables `file.swmr_mode = True`. A reader
can therefore open the file while the writer is still appending. The tests
prove that a live SWMR reader sees the first committed chunk, sees the second
chunk after `dataset.refresh()`, and sees the final attributes after close.

The pipeline writes to `<shard_id>.h5.partial` and renames it into place only
after the writer closes. The index is written last through a sibling
`shards.json.tmp`, `fsync`, and `os.replace`. A failure while consuming a shard
removes that shard's partial file; completed earlier shards remain, and the
next run overwrites them.

### Read path

`MmapTokenStore` is named for the API, not because h5py maps the file. h5py
reads whole chunks through the HDF5 chunk cache and the OS page cache and
copies the requested hyperslab into a NumPy array. The store opens only a
worker's selected shard subset in `swmr=True` mode, computes cumulative global
offsets in index order, and implements `get_slice(start, stop)` with
`dataset.read_direct` into the destination buffer. A slice crossing a shard
boundary is one call and several chunk reads, not a full-file materialisation.
`rdcc_nbytes` defaults to 1 MiB, the HDF5 chunk-cache default.

`SlidingWindowDataloader` samples `batch_size` starts uniformly with
replacement from `0..num_tokens - (window_size + 1)`, reads exactly
`window_size + 1` ids per start, and fills preallocated `(batch_size,
window_size)` uint32 arrays:

```text
window = tokens[start : start + window_size + 1]
input  = window[:-1]
target = window[1:]
```

The target shift is therefore exact within every row. A private
`numpy.random.Generator(numpy.random.PCG64(seed))` makes the stream
reproducible for the pinned NumPy version without touching global RNG state.
Workers are expected to build their own store and dataloader with different
shard subsets and seeds.

## Real corpus mode

There is no external provider, API key, or GPU path in this lesson. The real
integration point is local: lesson 42 emits decompressed `*.jsonl` shards under
its `corpus/` directory. Point `build --input-dir` at that directory; each
`<name>.jsonl` produces `<name>.h5`, preserving the one-file-per-source-shard
mapping. A trainer consumes the uint32 NumPy batches at the batch boundary and
moves them to its own device; this module deliberately does not depend on
PyTorch or pinned memory.

For a larger local corpus, use a chunk size that is an integer multiple of the
training window, and give each dataloader worker a disjoint `shard_ids` subset.
The index remains the single startup file: adding a shard means writing its
HDF5 file, hashing it, and appending a record.

## What runs offline

- `python code/main.py` / `demo`: builds the fixed two-shard corpus in process,
  writes under `outputs/`, reads it, and validates it.
- `build`: reads local `*.jsonl` files; no HTTP, credentials, or remote store.
- `validate`: local HDF5 reads and local SHA-256 comparison.
- `benchmark`: local synthetic corpus and local reads.
- `pytest`: temporary directories only; no network, fixed ports, or secrets.

The only network use is `uv` installing the pinned packages into the ignored
`.venv/`.

## Verification

```bash
./verify.sh
```

The gate runs `uv sync --frozen --all-groups`, 38 pytest cases, strict mypy over
`code/main.py` and `tests/`, the demo, and `validate` on the demo index. The
last line is `verify: ok`. The suite covers UTF-8 byte round trips, chunk-stride
buffering, empty streams, gzip round trips, live SWMR visibility, boundary
placement, empty shards, malformed JSONL, index contract violations, checksum
corruption, missing/truncated shards, cross-shard slices, attribute truncation,
shard subset ordering, shift-by-one batches, seeded replay, and the CLI paths.
No test asserts only that an exception was not raised.

## Measured results

Measured locally on Python 3.14.10, h5py 3.16.0, NumPy 2.5.5, HDF5 2.0.0,
Linux x86-64. These are short, warm-cache runs, not hardware benchmarks.
Numbers vary with filesystem and load.

Demo corpus, two shards, six documents, 316 tokens, chunk 64:

| Variant | Shard bytes | Build | 10 read batches | Samples/s |
| --- | ---: | ---: | ---: | ---: |
| uncompressed | 22,016 | 0.0082–0.0109 s | 0.0068–0.0148 s | 675–1,472 |
| gzip | 21,068 | 0.0100 s | 0.0084 s | 1,197 |

The tiny demo is dominated by fixed h5py open/close costs. It shows that gzip
saves only about 4% on this short, mixed text and does not justify making it
the default.

Benchmark, 523,944 tokens across four shards, `batch_size=16`, 1,000 batches:

| Chunk | Window | Bytes | Build | Read | Samples/s | Mtoken/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 128 | 2,260,056 | 0.6622 s | 0.5573 s | 1,794.4 | 0.23 |
| 128 | 128 | 2,195,456 | 0.3423 s | 0.4592 s | 2,177.7 | 0.28 |
| 256 | 128 | 2,162,688 | 0.2006 s | 0.4219 s | 2,370.4 | 0.30 |

With a larger window (`window=4096`, `batch_size=8`, 1,000 batches), the
per-sample h5py call overhead is amortised:

| Chunk | Window | Bytes | Build | Read | Samples/s | Mtoken/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 4,096 | 2,138,112 | 0.0771 s | 0.2814 s | 3,553.2 | 14.55 |
| 4,096 | 4,096 | 2,138,112 | 0.0613 s | 0.2697 s | 3,707.7 | 15.19 |
| 8,192 | 4,096 | 2,138,112 | 0.0517 s | 0.2501 s | 3,999.1 | 16.38 |

Observations:

- Larger chunks cut **build** time because there are fewer resize calls:
  0.662 s at chunk 64 versus 0.201 s at chunk 256 in the first table.
- At the small window, read throughput stays around 0.23–0.30 Mtoken/s; the
  per-sample HDF5 call is the bottleneck, not the chunk transition.
- At the larger window, half-window chunks are about 11% slower than
  equal/double chunks (14.55 versus 16.38 Mtoken/s). This is the expected
  alignment direction, but it did **not** reproduce the spec's claim that
  mismatched chunks halve throughput on a warm-cache 2 MiB file.
- Gzip at `window=128`, chunk 256 reduced bytes from 2,162,688 to 492,852 and
  build time from 0.2006 s to 0.2166 s; read throughput stayed in the same
  0.29–0.30 Mtoken/s band. The synthetic corpus is highly repetitive, so this
  compression ratio overstates what natural text would achieve. The default
  remains uncompressed: it avoids decompression in the hot read path, and the
  demo's mixed text showed only a 4% space saving.
- No cold-cache measurement was taken; dropping page caches needs privileges
  and would not be reproducible in the ordinary test environment. No
  16-worker process stress test was run.

## Design decisions

- **UTF-8 byte tokenizer, boundary id 256.** The lesson needs a deterministic
  tokenizer, not another BPE implementation. Bytes make the encoding exactly
  reversible and keep the boundary id outside the text vocabulary.
- **One boundary token after every document.** This inserts exactly one
  separator per document and makes shard concatenation safe. The cost is one
  id per document; a loss mask is a trainer decision, not a storage decision.
- **Little-endian uint32 and one canonical hash.** Explicit `<u4` avoids
  platform-endian ambiguity in the checksum. The hash covers exactly the
  append-order token bytes, so `validate` can recompute it in blocks.
- **Exact-size resizes, attribute as an additional guard.** Full buffers
  resize by `chunk_size`; the residual resizes to its true length. The
  `token_count` attribute is still authoritative for readers, and a test grows
  a dataset beyond it to prove trailing zeros are not returned.
- **SWMR on the write side.** `libver="latest"` plus `swmr_mode=True` and a
  flush after each committed chunk match the SWMR writer contract [2]. The
  writer also records final attributes after entering SWMR; this works on the
  pinned HDF5 2.0.0 and is covered by a live-reader test.
- **Atomic shard and index publication.** Partial shards and a temp index keep
  half-written files out of the source-of-truth listing. Multi-shard
  replacement is still not a filesystem transaction; see limitations.
- **Per-worker shard subsets.** A store opens only its selected files, in
  index order. This is the mechanism that lets workers share the corpus
  without sharing file handles.
- **Seeded PCG64 sampling with replacement.** Uniform starts match the
  infinite-stream training pattern; the local generator makes replays
  reproducible. NumPy stream stability is tied to the pinned version.
- **Chunk cache, not mmap.** h5py does not memory-map the file; chunked reads
  use the HDF5 chunk cache and the OS page cache [3][4]. The store's name is
  retained for the lesson's API, and its docstring says this explicitly.

## Limitations

- The pipeline writes shards sequentially in one process. Shards are
  independent files, so a caller can run multiple pipelines/processes, but no
  built-in parallel writer is provided.
- Publishing a new multi-shard set is not transactional across shard files and
  the index. A crash after replacing some shards but before writing the index
  can leave the old index pointing at changed files; rerun the pipeline and
  validate before training.
- `MmapTokenStore` targets finalized shards. The files are SWMR-readable and
  the writer flushes committed chunks, but the store does not refresh a
  live-growing dataset; use a fresh store after a shard is finalized.
- `verify_shards` checks shards sequentially. The per-shard hashes are
  independent and can be fanned out to processes by a CI driver.
- Windows are sampled with replacement and may straddle documents and shards.
  There is no epoch, no without-replacement coverage guarantee, and no loss
  mask for the boundary token.
- Token ids are uint32, so vocabularies above `2^32 - 1` are unsupported (as
  are negative ids).
- Very long documents are tokenized whole; there is no read-time or write-time
  `max_document_tokens` truncation.
- The benchmark's synthetic text is repetitive and warm-cache; it is useful for
  relative chunk-sizing behavior, not for predicting production storage
  throughput.
- The module stops at NumPy batches. It does not implement pinned memory, a
  trainer, model compute, a GPU transfer, or a distributed worker pool.

## References

1. AI Engineering From Scratch, [HDF5 Tokenized Corpus capstone](https://aiengineeringfromscratch.com/learn/capstone-projects/hdf5-tokenized-corpus) — pipeline stages, shard index contract, boundary-token packing, chunking, and SWMR production patterns.
2. h5py, [Single Writer Multiple Reader (SWMR)](https://docs.h5py.org/en/stable/swmr.html) — `libver="latest"`, `swmr_mode`, dataset flushes, reader `refresh()`, and the writer/reader sequence.
3. h5py, [Datasets — chunked storage](https://docs.h5py.org/en/stable/high/dataset.html#chunked-storage) — resizable `maxshape`, fixed chunks, compression filters, and the chunk cache behavior used by `get_slice`.
4. The HDF Group, [Chunking in HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/hdf5_chunking.html) — chunked layout, partial I/O, and aligning chunk size with the selection size.
5. The HDF Group, [Improve HDF5 performance using caching](https://www.hdfgroup.org/2022/10/17/improve-hdf5-performance-using-caching/) — per-dataset chunk cache, default 1 MiB, `rdcc_nbytes`, and eviction behavior.

## License

MIT License, Copyright (c) 2026 Pedro Monteiro. See [LICENSE](LICENSE).
