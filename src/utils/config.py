"""Configuration loading for the learning and enterprise profiles.

A profile is a YAML file in ``config/``. Any value can be overridden with an environment variable named
``EIKA__<SECTION>__<KEY>`` (for example ``EIKA__LLM__ENDPOINT=databricks-claude-sonnet-5``), so secrets and
workspace-specific values never need to be written into the files.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
ENV_PREFIX = "EIKA__"
DEBUG_ENV = "EIKA_DEBUG"
SENSITIVE_PARTS = ("password", "secret", "token", "api_key", "credential", "key")


class ConfigError(ValueError):
    """Raised when a profile is missing or invalid."""


def _coerce(value: str) -> Any:
    """Turn an environment-variable string into bool/int/float when it looks like one."""
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _apply_env_overrides(cfg: dict[str, Any], environ: dict[str, str]) -> dict[str, Any]:
    for name, raw in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        keys = [k.lower() for k in name[len(ENV_PREFIX):].split("__") if k]
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        if keys:
            node[keys[-1]] = _coerce(raw)
    return cfg


def _resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Make relative paths in the ``paths`` section absolute, relative to the project root."""
    for key, value in cfg.get("paths", {}).items():
        if isinstance(value, str) and value and not value.startswith(("/", "dbfs:", "abfss:")) and ":" not in value[:3]:
            cfg["paths"][key] = str(PROJECT_ROOT / value)
    return cfg


def _enabled(value: bool | str | None) -> bool:
    """Interpret notebook-widget and environment values as a debug flag."""
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _redact(value: Any, key: str = "") -> Any:
    """Return a printable configuration copy with sensitive values hidden."""
    if any(part in key.lower() for part in SENSITIVE_PARTS):
        return "***REDACTED***"
    if isinstance(value, dict):
        return {k: _redact(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, key) for v in value]
    return value


def _trace_config(cfg: dict[str, Any], env_names: list[str], overrides: dict[str, Any] | None) -> None:
    """Print a concise, safe trace that is useful while running lessons one by one."""
    selected = {
        key: cfg.get(key)
        for key in ("profile", "paths", "ingestion", "chunking", "embedding", "vector_store", "retrieval", "llm", "agent")
        if key in cfg
    }
    print("[EIKA CONFIG] Environment override names:", env_names or "none")
    print("[EIKA CONFIG] Explicit overrides:")
    print(json.dumps(_redact(overrides or {}), indent=2, default=str))
    print("[EIKA CONFIG] Effective notebook settings (sensitive values redacted):")
    print(json.dumps(_redact(selected), indent=2, default=str))


REQUIRED_SECTIONS = ("profile", "paths", "chunking", "embedding", "vector_store", "retrieval", "llm", "agent", "security")


def load_config(profile: str | None = None, overrides: dict[str, Any] | None = None,
                environ: dict[str, str] | None = None,
                debug: bool | str | None = None) -> dict[str, Any]:
    """Load a profile (``learning`` or ``enterprise``), then apply environment and explicit overrides.

    The profile name comes from the argument, else the ``EIKA_PROFILE`` environment variable, else ``learning``.
    """
    environ = dict(os.environ if environ is None else environ)
    if profile:
        profile_source = "function/notebook argument"
    elif environ.get("EIKA_PROFILE"):
        profile_source = "EIKA_PROFILE environment variable"
    else:
        profile_source = "default"
    profile = profile or environ.get("EIKA_PROFILE", "learning")
    path = CONFIG_DIR / f"{profile}.yaml"
    print(f"[EIKA CONFIG] profile={profile!r} source={profile_source}")
    print(f"[EIKA CONFIG] yaml={path.resolve()}")
    if not path.exists():
        raise ConfigError(f"Unknown profile '{profile}': {path} does not exist")

    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"Profile '{profile}' is missing sections: {', '.join(missing)}")
    env_names = sorted(name for name in environ if name.startswith(ENV_PREFIX))
    cfg = _apply_env_overrides(cfg, environ)
    for section, values in (overrides or {}).items():
        if isinstance(values, dict):
            cfg.setdefault(section, {}).update(values)
        else:
            cfg[section] = values
    cfg = _resolve_paths(copy.deepcopy(cfg))
    debug_on = _enabled(debug if debug is not None else environ.get(DEBUG_ENV))
    if debug_on:
        _trace_config(cfg, env_names, overrides)
    # Runtime-only flag: downstream learner-facing functions inherit the notebook choice.
    # It is added after the configuration trace so it is not confused with a YAML setting.
    cfg["_debug"] = debug_on
    return cfg
