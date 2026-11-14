"""Append-only audit trail for tool calls and approvals. Arguments are redacted before they are written."""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from src.security.redaction import redact


class AuditLog:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def record(self, actor: str, action: str, outcome: str, **details: Any) -> dict[str, Any]:
        entry = {
            "audit_id": str(uuid.uuid4()),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "actor": redact(actor),
            "action": action,
            "outcome": outcome,
            "details": json.loads(redact(json.dumps(details, default=str))),
        }
        with self._lock:
            self.records.append(entry)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self.path.is_relative_to("/Volumes"):
                    # Unity Catalog volumes support file creation but not append writes.
                    entries = self.path.with_suffix(".entries")
                    entries.mkdir(exist_ok=True)
                    (entries / f"{entry['audit_id']}.json").write_text(json.dumps(entry) + "\n", encoding="utf-8")
                else:
                    with self.path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(entry) + "\n")
        return entry
