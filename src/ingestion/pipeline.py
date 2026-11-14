"""Incremental ingestion: load new or changed documents, skip unchanged ones and exact duplicates."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.documents import Document, UnsupportedDocumentError, discover_files, load_file
from src.ingestion.registry import Registry
from src.security.redaction import redact
from src.utils.debug import debug_trace, resolve_debug
from src.utils.logging import get_logger, log_event

log = get_logger("eika.ingestion")


@dataclass
class IngestionResult:
    new: list[Document] = field(default_factory=list)
    updated: list[Document] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def to_process(self) -> list[Document]:
        return self.new + self.updated

    def summary(self) -> dict[str, int]:
        return {"new": len(self.new), "updated": len(self.updated), "unchanged": len(self.unchanged),
                "duplicates": len(self.duplicates), "failed": len(self.failed)}


def ingest_directory(root: str | Path, registry: Registry, supported: tuple[str, ...] | list[str],
                     groups_by_name: dict[str, list[str]] | None = None,
                     debug: bool | str | None = None, *, record: bool = True) -> IngestionResult:
    """Load every supported file under ``root``. Set ``record=False`` to commit after indexing.

    * A file whose path was never seen is **new**.
    * A known path with a different content hash is **updated** (its old chunks must be replaced).
    * A known path with the same hash is **unchanged** and skipped.
    * A new path whose content hash matches another document is a **duplicate** and skipped.
    """
    debug_on = resolve_debug(debug)
    result = IngestionResult()
    known_hashes = {row["content_hash"]: row["doc_id"] for row in registry.all()}
    debug_trace(debug_on, "INGESTION", f"starting root={str(root)!r} known_documents={len(known_hashes)}")
    for path in discover_files(root, supported, debug=debug_on):
        try:
            doc = load_file(path, (groups_by_name or {}).get(path.name), debug=debug_on)
        except (UnsupportedDocumentError, ValueError, OSError) as exc:
            result.failed[path.name] = redact(str(exc))
            debug_trace(debug_on, "INGESTION", f"decision=failed file={path.name!r} reason={result.failed[path.name]!r}")
            continue
        if not doc.text.strip():
            result.failed[path.name] = "no extractable text"
            debug_trace(debug_on, "INGESTION", f"decision=failed file={path.name!r} reason='no extractable text'")
            continue
        previous = registry.get(doc.doc_id)
        if previous and previous["content_hash"] == doc.content_hash:
            result.unchanged.append(doc.source_name)
            debug_trace(debug_on, "INGESTION", f"decision=unchanged file={doc.source_name!r}")
            continue
        if not previous and doc.content_hash in known_hashes:
            result.duplicates.append(doc.source_name)
            debug_trace(debug_on, "INGESTION", f"decision=duplicate file={doc.source_name!r}")
            continue
        doc.status = "updated" if previous else "new"
        debug_trace(debug_on, "INGESTION", f"decision={doc.status} file={doc.source_name!r}")
        (result.updated if previous else result.new).append(doc)
        known_hashes[doc.content_hash] = doc.doc_id
        if record:
            registry.upsert(doc)
    log_event(log, "ingestion.done", **result.summary())
    debug_trace(debug_on, "INGESTION", f"complete summary={result.summary()}")
    return result
