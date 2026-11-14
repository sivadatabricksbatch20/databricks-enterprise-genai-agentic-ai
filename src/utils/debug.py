"""Small, safe console traces used by the learner-facing notebooks."""
from __future__ import annotations

from typing import Any


def debug_enabled(value: bool | str | None) -> bool:
    """Interpret booleans, notebook widget values and environment-style strings."""
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_debug(debug: bool | str | None = None, cfg: dict[str, Any] | None = None) -> bool:
    """Use an explicit flag first, otherwise inherit the runtime flag added by ``load_config``."""
    return debug_enabled(debug if debug is not None else (cfg or {}).get("_debug"))


def debug_trace(debug: bool | str | None, stage: str, message: str) -> None:
    """Print one consistently labelled line only when learner tracing is enabled."""
    if debug_enabled(debug):
        print(f"[EIKA {stage.upper()}] {message}")


def short_text(value: str, limit: int = 100) -> str:
    """Return a single-line preview without flooding notebook output."""
    compact = " ".join((value or "").split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"
