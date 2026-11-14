"""Vector stores behind one interface, selected by ``vector_store.backend``.

* ``InMemoryVectorStore`` (learning): exact cosine search with numpy, or FAISS when installed. Persists to a .npz file.
* ``DatabricksVectorSearchStore`` (enterprise): a Databricks AI Search index with self-managed embeddings,
  queried with ``databricks-ai-search``. Needs an AI Search endpoint and a Delta source table.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import os

import numpy as np

from src.processing.chunking import Chunk
from src.utils.debug import debug_trace, resolve_debug


@dataclass
class SearchHit:
    chunk: dict[str, Any]
    score: float


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...
    def delete_document(self, doc_id: str) -> int: ...
    def search(self, query_vector: np.ndarray, k: int, filters: dict[str, Any] | None = None) -> list[SearchHit]: ...
    def count(self) -> int: ...


def matches_filters(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Filter on metadata, e.g. ``{"doc_type": "md"}`` or ``{"source_name": ["a.md", "b.txt"]}``."""
    for key, wanted in (filters or {}).items():
        value = chunk.get(key)
        if isinstance(value, list):
            wanted_set = set(wanted if isinstance(wanted, (list, tuple, set)) else [wanted])
            if not wanted_set & set(value):
                return False
        elif isinstance(wanted, (list, tuple, set)):
            if value not in wanted:
                return False
        elif value != wanted:
            return False
    return True


@dataclass
class InMemoryVectorStore:
    """LEARNING backend: every vector is kept in memory and compared exactly (fine for thousands of chunks)."""

    dimension: int
    use_faiss: bool = True
    debug: bool = False
    _vectors: np.ndarray = field(init=False)
    _chunks: list[dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._vectors = np.zeros((0, self.dimension), dtype=np.float32)
        try:
            import faiss  # noqa: F401
            self._faiss = faiss if self.use_faiss else None
        except ImportError:
            self._faiss = None

    @property
    def engine(self) -> str:
        return "faiss" if self._faiss else "numpy"

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if len(vectors) and vectors.shape[1] != self.dimension:
            raise ValueError(f"vector dimension {vectors.shape[1]} != index dimension {self.dimension}")
        incoming = {c.chunk_id for c in chunks}
        keep = [i for i, c in enumerate(self._chunks) if c["chunk_id"] not in incoming]
        self._chunks = [self._chunks[i] for i in keep] + [c.to_dict() for c in chunks]
        self._vectors = np.vstack([self._vectors[keep], vectors.astype(np.float32)])
        debug_trace(self.debug, "VECTOR STORE",
                    f"backend={self.engine} upserted={len(chunks)} total={len(self._chunks)} dimension={self.dimension}")

    def delete_document(self, doc_id: str) -> int:
        keep = [i for i, c in enumerate(self._chunks) if c["doc_id"] != doc_id]
        removed = len(self._chunks) - len(keep)
        self._chunks = [self._chunks[i] for i in keep]
        self._vectors = self._vectors[keep]
        return removed

    def search(self, query_vector: np.ndarray, k: int, filters: dict[str, Any] | None = None) -> list[SearchHit]:
        if not self._chunks:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if self._faiss is not None and not filters:
            index = self._faiss.IndexFlatIP(self.dimension)
            index.add(self._vectors)
            scores, ids = index.search(q.reshape(1, -1), min(k, len(self._chunks)))
            hits = [SearchHit(self._chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]
            debug_trace(self.debug, "VECTOR STORE", f"backend=faiss requested={k} returned={len(hits)} filters={{}}")
            return hits
        scores = self._vectors @ q                      # cosine similarity: vectors are unit length
        order = np.argsort(-scores)
        hits = []
        for i in order:
            if matches_filters(self._chunks[i], filters):
                hits.append(SearchHit(self._chunks[i], float(scores[i])))
                if len(hits) == k:
                    break
        debug_trace(self.debug, "VECTOR STORE",
                    f"backend=numpy requested={k} returned={len(hits)} filters={filters or {}}")
        return hits

    def count(self) -> int:
        return len(self._chunks)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=self._vectors, chunks=np.array([json.dumps(self._chunks)]))

    @classmethod
    def load(cls, path: str | Path, debug: bool | str | None = None) -> "InMemoryVectorStore":
        data = np.load(Path(path), allow_pickle=False)
        store = cls(dimension=int(data["vectors"].shape[1]), debug=resolve_debug(debug))
        store._vectors = data["vectors"].astype(np.float32)
        store._chunks = json.loads(str(data["chunks"][0]))
        return store


class DatabricksVectorSearchStore:  # pragma: no cover - needs a Vector Search endpoint
    """ENTERPRISE backend: Databricks Vector Search (direct-access index with self-managed embeddings).

    For production, prefer a Delta Sync index on ``unity_catalog.chunks_table`` so the index follows the table;
    ``create_delta_sync_index`` shows that setup. This class also supports direct upserts for simple pipelines.
    """

    COLUMNS = ["chunk_id", "doc_id", "chunk_index", "text", "source_name", "source_path", "doc_type", "groups"]

    def __init__(self, cfg: dict[str, Any], debug: bool | str | None = None) -> None:
        from databricks.ai_search.client import AISearchClient
        vs = cfg["vector_store"]
        self.endpoint, self.index_name = vs["endpoint_name"], vs["index_name"]
        self.dimension = int(vs.get("embedding_dimension", 1024))
        self.debug = resolve_debug(debug, cfg)
        if os.environ.get("DATABRICKS_CLIENT_ID") and os.environ.get("DATABRICKS_CLIENT_SECRET"):
            workspace_url = os.environ["DATABRICKS_HOST"].strip()
            if "://" not in workspace_url:
                workspace_url = "https://" + workspace_url
            self.client = AISearchClient(
                workspace_url=workspace_url.rstrip("/"),
                service_principal_client_id=os.environ["DATABRICKS_CLIENT_ID"],
                service_principal_client_secret=os.environ["DATABRICKS_CLIENT_SECRET"],
                disable_notice=True,
            )
        else:
            self.client = AISearchClient(disable_notice=True)
        self._index = None

    def ensure_direct_index(self) -> bool:
        """Ensure the index is ready and report whether it was created."""
        debug_trace(self.debug, "VECTOR STORE",
                    f"checking endpoint={self.endpoint!r} index={self.index_name!r}")
        created = not self.client.index_exists(endpoint_name=self.endpoint, index_name=self.index_name)
        if not created:
            self._index = self.client.get_index(self.endpoint, self.index_name)
        else:
            self._index = self.client.create_direct_access_index(
                endpoint_name=self.endpoint, index_name=self.index_name, primary_key="chunk_id",
                embedding_dimension=self.dimension, embedding_vector_column="embedding",
                schema={"chunk_id": "string", "doc_id": "string", "chunk_index": "int", "text": "string",
                        "source_name": "string", "source_path": "string", "doc_type": "string",
                        "groups": "array<string>", "embedding": "array<float>"})
        self._index.wait_until_ready()
        debug_trace(self.debug, "VECTOR STORE", f"index_ready={self.index_name!r}")
        return created

    def create_delta_sync_index(self, source_table: str, pipeline_type: str = "TRIGGERED") -> None:
        self.client.create_delta_sync_index(
            endpoint_name=self.endpoint, index_name=self.index_name, source_table_name=source_table,
            pipeline_type=pipeline_type, primary_key="chunk_id", embedding_dimension=self.dimension,
            embedding_vector_column="embedding")

    @property
    def index(self):
        if self._index is None:
            self._index = self.client.get_index(self.endpoint, self.index_name)
        return self._index

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if self._index is None:
            self.ensure_direct_index()
        rows = [{**{k: c.to_dict()[k] for k in self.COLUMNS}, "embedding": v.tolist()} for c, v in zip(chunks, vectors)]
        for i in range(0, len(rows), 500):
            self.index.upsert(rows[i:i + 500])
        # Upsert is asynchronous. Do not let a following notebook query while the data plane is still updating.
        self.index.wait_until_ready(wait_for_updates=True)
        debug_trace(self.debug, "VECTOR STORE",
                    f"backend=databricks_ai_search upserted={len(rows)} index={self.index_name!r} updates_ready=True")

    def delete_document(self, doc_id: str) -> int:
        res = self.index.similarity_search(query_vector=[0.0] * self.dimension, columns=["chunk_id"],
                                           filters={"doc_id": doc_id}, num_results=1000)
        ids = [r[0] for r in res.get("result", {}).get("data_array", [])]
        if ids:
            self.index.delete(ids)
        return len(ids)

    def search(self, query_vector: np.ndarray, k: int, filters: dict[str, Any] | None = None) -> list[SearchHit]:
        res = self.index.similarity_search(query_vector=np.asarray(query_vector).tolist(), columns=self.COLUMNS,
                                           filters=filters or None, num_results=k)
        cols = [c["name"] for c in res["manifest"]["columns"]]
        hits = []
        for row in res.get("result", {}).get("data_array", []):
            rec = dict(zip(cols, row))
            score = float(rec.pop("score", 0.0))
            hits.append(SearchHit(rec, score))
        debug_trace(self.debug, "VECTOR STORE",
                    f"backend=databricks_ai_search requested={k} returned={len(hits)} filters={filters or {}}")
        return hits

    def count(self) -> int:
        return int(self.index.describe().get("status", {}).get("indexed_row_count", 0))


def build_vector_store(cfg: dict[str, Any], dimension: int,
                       debug: bool | str | None = None) -> VectorStore:
    debug_on = resolve_debug(debug, cfg)
    backend = cfg["vector_store"].get("backend", "in_memory")
    debug_trace(debug_on, "VECTOR STORE", f"building backend={backend!r} dimension={dimension}")
    if backend == "in_memory":
        return InMemoryVectorStore(dimension=dimension, debug=debug_on)
    if backend == "databricks_vector_search":
        return DatabricksVectorSearchStore(cfg, debug=debug_on)
    raise ValueError(f"Unknown vector store backend '{backend}'")
