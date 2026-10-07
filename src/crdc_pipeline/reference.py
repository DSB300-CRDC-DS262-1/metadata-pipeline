"""The hand-built references in data/reference/: metadata, statistics and contexts for each document."""

from __future__ import annotations

import json
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"


def _load(name: str) -> list[dict]:
    return json.loads((REFERENCE_DIR / name).read_text(encoding="utf-8"))["documents"]


def metadata_reference() -> dict[str, dict]:
    """Reference metadata by document_id."""
    return {d["document_id"]: d for d in _load("metadata.json")}


def findings_reference() -> list[dict]:
    """Each document's statistics joined with its context items and contexts, as one record per document."""
    contexts = {d["document_id"]: d for d in _load("contexts.json")}
    return [{"document_id": d["document_id"], "context_items": contexts[d["document_id"]]["context_items"],
             "contexts": contexts[d["document_id"]]["contexts"], "statistics": d["statistics"]}
            for d in _load("statistics.json")]
