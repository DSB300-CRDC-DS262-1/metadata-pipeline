"""Layout and page-count checks on the real sample PDFs (skipped if absent)."""

import json
from pathlib import Path

import pytest

from crdc_pipeline.ingestion.loader import load_document

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MANIFEST_PATH = DATA_DIR / "document_manifest.json"


def _load_manifest_entries() -> list[dict]:
    if not MANIFEST_PATH.is_file():
        return []
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return manifest["documents"]


def _expected_columns(expected_layout: str) -> int:
    return 2 if expected_layout.startswith("two column") else 1


@pytest.mark.parametrize("entry", _load_manifest_entries(), ids=lambda e: e["document_id"])
def test_sample_document_layout_matches_ground_truth(entry: dict) -> None:
    if not entry.get("filename"):
        pytest.skip(f"{entry['document_id']}: no filename set in document_manifest.json yet")

    pdf_path = DATA_DIR / "raw_pdfs" / entry["filename"]
    if not pdf_path.is_file():
        pytest.skip(f"{entry['document_id']}: {pdf_path.name} not found under data/raw_pdfs/")

    doc = load_document(pdf_path, document_id=entry["document_id"])

    assert doc.page_count == entry["expected_page_count"]

    expected_columns = _expected_columns(entry["expected_layout"])
    majority_columns = [p.column_count for p in doc.pages].count(expected_columns)
    # Abstract and reference pages are often full-width, so require a majority.
    assert majority_columns >= doc.page_count / 2, (
        f"expected most pages to be {expected_columns}-column for "
        f"{entry['document_id']}, got column counts: "
        f"{[p.column_count for p in doc.pages]}"
    )

    failed_pages = [p.page_number for p in doc.pages if p.char_count == 0]
    if failed_pages:
        pytest.skip(
            f"{entry['document_id']}: pages {failed_pages} produced no text "
            "-- expected for known text-layer gaps (weaknesses_and_fixes.md "
            "issue 2), needs OCR/multimodal fallback to resolve, not this test"
        )
