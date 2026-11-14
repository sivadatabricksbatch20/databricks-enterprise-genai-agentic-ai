"""Index builder: chunk → embed → upsert, replacing the old chunks of documents that changed."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.embeddings.embedders import Embedder
from src.ingestion.documents import Document
from src.processing.chunking import Chunk, chunk_document
from src.retrieval.vector_store import VectorStore
from src.utils.logging import get_logger, log_event
from src.utils.debug import debug_trace, resolve_debug

log = get_logger("eika.indexing")


@dataclass
class IndexReport:
    documents: int
    chunks: int
    replaced_chunks: int
    embedding_model: str
    embedding_version: str
    index_size: int

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def chunk_documents(docs: list[Document], cfg: dict[str, Any],
                    debug: bool | str | None = None) -> list[Chunk]:
    debug_on = resolve_debug(debug, cfg)
    c = cfg["chunking"]
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, int(c["chunk_size"]), int(c["chunk_overlap"]),
                                     int(c.get("min_chunk_chars", 1)), debug=debug_on))
    debug_trace(debug_on, "CHUNKING", f"all_documents={len(docs)} total_chunks={len(chunks)}")
    return chunks


def index_documents(docs: list[Document], cfg: dict[str, Any], embedder: Embedder, store: VectorStore,
                    updated_doc_ids: set[str] | None = None,
                    debug: bool | str | None = None) -> IndexReport:
    debug_on = resolve_debug(debug, cfg)
    debug_trace(debug_on, "INDEXING",
                f"starting documents={len(docs)} updated_documents={len(updated_doc_ids or set())} "
                f"store={type(store).__name__}")
    replaced = sum(store.delete_document(doc_id) for doc_id in (updated_doc_ids or set()))
    chunks = chunk_documents(docs, cfg, debug=debug_on)
    if chunks:
        vectors = embedder.embed([c.text for c in chunks])
        store.upsert(chunks, vectors)
    report = IndexReport(len(docs), len(chunks), replaced, embedder.model_name, embedder.model_version, store.count())
    log_event(log, "indexing.done", **report.as_dict())
    debug_trace(debug_on, "INDEXING", f"complete report={report.as_dict()}")
    return report
