"""Automatic Unity Catalog setup for the enterprise learning profile."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any


SYSTEM_CATALOGS = {"hive_metastore", "spark_catalog", "system", "samples"}


def _quote(identifier: str) -> str:
    """Quote a Unity Catalog identifier safely."""
    return f"`{identifier.replace('`', '``')}`"


def _workspace_catalog(spark, configured: str | None) -> str:
    """Resolve the workspace catalog without silently choosing an unrelated catalog."""
    catalogs = [row[0] for row in spark.sql("SHOW CATALOGS").collect()]
    lookup = {name.lower(): name for name in catalogs}

    if configured and configured.lower() not in {"auto", "workspace"}:
        match = lookup.get(configured.lower())
        if not match:
            raise ValueError(
                f"Configured catalog {configured!r} was not found. Available catalogs: {', '.join(catalogs)}"
            )
        return match

    try:
        workspace_name = spark.conf.get("spark.databricks.clusterUsageTags.workspaceName")
    except Exception:  # noqa: BLE001
        workspace_name = ""
    if workspace_name and workspace_name.lower() in lookup:
        return lookup[workspace_name.lower()]

    current = spark.sql("SELECT current_catalog()").first()[0]
    if current and current.lower() not in SYSTEM_CATALOGS and current.lower() in lookup:
        return lookup[current.lower()]
    if "workspace" in lookup:
        return lookup["workspace"]

    candidates = [name for name in catalogs if name.lower() not in SYSTEM_CATALOGS]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "Could not determine the student's workspace catalog. "
        f"Available catalogs: {', '.join(catalogs)}. "
        "Set unity_catalog.catalog in config/enterprise.yaml or EIKA__UNITY_CATALOG__CATALOG."
    )


def prepare_enterprise_config(
    cfg: dict[str, Any], spark, project_root: str | Path, *, debug: bool | str = False
) -> dict[str, Any]:
    """Create schema/Volume, seed samples, and resolve enterprise paths idempotently."""
    if cfg.get("profile") != "enterprise":
        return cfg

    cfg = copy.deepcopy(cfg)
    catalog = _workspace_catalog(spark, cfg.get("unity_catalog", {}).get("catalog"))
    schema = cfg.get("unity_catalog", {}).get("schema", "eika")
    volume = cfg.get("unity_catalog", {}).get("volume", "eika_data")

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {_quote(catalog)}.{_quote(schema)}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {_quote(catalog)}.{_quote(schema)}.{_quote(volume)}")

    volume_root = Path(f"/Volumes/{catalog}/{schema}/{volume}")
    sample_target, work_dir = volume_root / "sample_documents", volume_root / "work"
    sample_target.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    sample_source = Path(project_root) / "data" / "sample_documents"
    if not sample_source.exists():
        raise FileNotFoundError(f"Bundled sample documents were not found: {sample_source}")
    copied = []
    for source in sample_source.iterdir():
        if source.is_file():
            target = sample_target / source.name
            if not target.exists() or target.stat().st_size != source.stat().st_size:
                shutil.copy2(source, target)
                copied.append(source.name)

    cfg.setdefault("unity_catalog", {}).update({
        "catalog": catalog, "schema": schema, "volume": volume,
        "documents_table": f"{catalog}.{schema}.documents",
        "chunks_table": f"{catalog}.{schema}.document_chunks",
    })
    cfg.setdefault("paths", {}).update({
        "sample_documents": str(sample_target), "work_dir": str(work_dir)
    })
    cfg.setdefault("vector_store", {})["index_name"] = f"{catalog}.{schema}.document_chunks_index"

    if str(debug).strip().lower() in {"1", "true", "yes", "on"}:
        print(f"[EIKA ENTERPRISE] catalog={catalog}")
        print(f"[EIKA ENTERPRISE] schema={catalog}.{schema}")
        print(f"[EIKA ENTERPRISE] volume={catalog}.{schema}.{volume}")
        print(f"[EIKA ENTERPRISE] sample_documents={sample_target}")
        print(f"[EIKA ENTERPRISE] copied={copied or 'none; files already present'}")
    return cfg
