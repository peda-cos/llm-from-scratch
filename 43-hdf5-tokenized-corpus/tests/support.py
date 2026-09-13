from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from main import (
    ByteTokenizer,
    CorpusDocument,
    ShardIndex,
    ShardedTokenizationPipeline,
    TOKENS_DTYPE,
    TokenArray,
    load_shard_index,
)

DEFAULT_DOCUMENTS: dict[str, tuple[CorpusDocument, ...]] = {
    "shard-0000": (
        CorpusDocument("a-0000", "alpha beta gamma delta"),
        CorpusDocument("a-0001", "Olá, 世界! UTF-8 stays intact."),
    ),
    "shard-0001": (
        CorpusDocument("b-0000", "epsilon zeta eta theta"),
        CorpusDocument("b-0001", "A longer document keeps the buffer busy."),
        CorpusDocument("b-0002", "🚀 rocket endpoint"),
    ),
}


@dataclass(frozen=True, slots=True)
class BuiltCorpus:
    workdir: Path
    index_path: Path
    index: ShardIndex
    tokenizer: ByteTokenizer
    documents: dict[str, tuple[CorpusDocument, ...]]
    shard_streams: dict[str, TokenArray]
    reference: TokenArray


def _stream_for(
    tokenizer: ByteTokenizer, documents: Sequence[CorpusDocument]
) -> TokenArray:
    ids: list[int] = []
    for document in documents:
        ids.extend(tokenizer.encode(document.text))
        ids.append(tokenizer.boundary_token_id)
    return np.asarray(ids, dtype=TOKENS_DTYPE)


def build_corpus(
    workdir: Path,
    *,
    chunk_size: int = 8,
    documents: Mapping[str, Sequence[CorpusDocument]] | None = None,
) -> BuiltCorpus:
    tokenizer = ByteTokenizer()
    selected = (
        {shard_id: tuple(docs) for shard_id, docs in documents.items()}
        if documents is not None
        else dict(DEFAULT_DOCUMENTS)
    )
    pipeline = ShardedTokenizationPipeline(
        workdir, tokenizer, chunk_size=chunk_size
    )
    index = pipeline.run(selected)
    index_path = workdir / "shards.json"
    shard_streams = {
        shard_id: _stream_for(tokenizer, docs)
        for shard_id, docs in selected.items()
    }
    if shard_streams:
        reference: TokenArray = np.concatenate(tuple(shard_streams.values()))
    else:
        reference = np.empty(0, dtype=TOKENS_DTYPE)
    return BuiltCorpus(
        workdir=workdir,
        index_path=index_path,
        index=load_shard_index(index_path),
        tokenizer=tokenizer,
        documents=dict(selected),
        shard_streams=shard_streams,
        reference=reference,
    )


def shard_path(built: BuiltCorpus, shard_id: str) -> Path:
    record = next(
        shard for shard in built.index.shards if shard.shard_id == shard_id
    )
    return built.workdir / record.path
