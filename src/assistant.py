"""Entry points used by the notebooks, the app and the tests."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.orchestrator import Orchestrator
from src.embeddings.embedders import Embedder, build_embedder
from src.ingestion.pipeline import IngestionResult, ingest_directory
from src.ingestion.registry import FileRegistry
from src.rag.llm import LLM, build_llm
from src.rag.pipeline import RAGPipeline
from src.retrieval.indexing import IndexReport, index_documents
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import (
    DatabricksVectorSearchStore, InMemoryVectorStore, VectorStore, build_vector_store,
)
from src.tools.registry import build_registry
from src.utils.debug import debug_trace, resolve_debug


def _index_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["work_dir"]) / cfg["vector_store"].get("persist_file", "vector_index.npz")


def build_index(cfg: dict[str, Any], embedder: Embedder | None = None, rebuild: bool = False,
                debug: bool | str | None = None) -> tuple[VectorStore, IngestionResult, IndexReport]:
    """Ingest documents incrementally and update the vector index (learning backend persists to ``work_dir``)."""
    debug_on = resolve_debug(debug, cfg)
    debug_trace(debug_on, "PIPELINE",
                f"build_index starting profile={cfg.get('profile')!r} rebuild={rebuild} "
                f"documents_root={cfg['paths']['sample_documents']!r}")
    embedder = embedder or build_embedder(cfg, debug=debug_on)
    work = Path(cfg["paths"]["work_dir"])
    manifest, index_file = work / "document_manifest.json", _index_path(cfg)
    in_memory = cfg["vector_store"].get("backend", "in_memory") == "in_memory"
    if rebuild or (in_memory and not index_file.exists()):
        manifest.unlink(missing_ok=True)                       # index missing → re-ingest everything
    store: VectorStore
    if in_memory and index_file.exists() and not rebuild:
        store = InMemoryVectorStore.load(index_file)
        if store.dimension != embedder.dimension:
            manifest.unlink(missing_ok=True)
            store = InMemoryVectorStore(dimension=embedder.dimension, debug=debug_on)
    else:
        store = build_vector_store(cfg, embedder.dimension, debug=debug_on)
    if isinstance(store, DatabricksVectorSearchStore):
        created = store.ensure_direct_index()
        if created or (manifest.exists() and store.count() == 0):
            manifest.unlink(missing_ok=True)
    registry = FileRegistry(manifest)
    result = ingest_directory(cfg["paths"]["sample_documents"], registry, cfg["ingestion"]["supported_types"],
                              debug=debug_on, record=False)
    report = index_documents(result.to_process, cfg, embedder, store, {d.doc_id for d in result.updated},
                             debug=debug_on)
    if in_memory:
        store.save(index_file)  # type: ignore[union-attr]
    for document in result.to_process:
        registry.upsert(document)
    debug_trace(debug_on, "PIPELINE", f"build_index complete ingestion={result.summary()} index={report.as_dict()}")
    return store, result, report


def load_rag(cfg: dict[str, Any], llm: LLM | None = None, embedder: Embedder | None = None,
             store: VectorStore | None = None,
             debug: bool | str | None = None) -> RAGPipeline:
    debug_on = resolve_debug(debug, cfg)
    debug_trace(debug_on, "RAG", f"loading profile={cfg.get('profile')!r}")
    embedder = embedder or build_embedder(cfg, debug=debug_on)
    if store is None:
        if cfg["vector_store"].get("backend", "in_memory") == "in_memory":
            store = InMemoryVectorStore.load(_index_path(cfg), debug=debug_on) if _index_path(cfg).exists() else \
                build_index(cfg, embedder, debug=debug_on)[0]
        else:
            store = build_vector_store(cfg, embedder.dimension, debug=debug_on)
    pipeline = RAGPipeline(Retriever.from_config(cfg, embedder, store, debug=debug_on),
                           llm or build_llm(cfg, debug=debug_on), debug=debug_on)
    debug_trace(debug_on, "RAG",
                f"loaded store={type(store).__name__} embedding_model={embedder.model_name!r} "
                f"llm={type(pipeline.llm).__name__}")
    return pipeline


def load_agent(cfg: dict[str, Any], llm: LLM | None = None, rag: RAGPipeline | None = None, spark=None,
               use_llm: bool | None = None) -> Orchestrator:
    rag = rag or load_rag(cfg, llm)
    tools = build_registry(cfg, rag.retriever, spark=spark)
    wants_llm = cfg["agent"].get("mode", "rules") == "llm" if use_llm is None else use_llm
    return Orchestrator(tools, rag.llm if wants_llm else None, "llm" if wants_llm else "rules",
                        int(cfg["agent"].get("max_steps", 6)))
