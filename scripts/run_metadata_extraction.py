"""Run frontier-model metadata extraction across the four CRDC sample
documents and compare the result against the reference metadata in
data/schema_worked_examples.json.

Usage:
    .venv/Scripts/python.exe scripts/run_metadata_extraction.py

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment or a
.env file at the project root (copy .env.example to .env and fill it in).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crdc_pipeline.extraction.metadata import (  # noqa: E402
    extract_metadata,
    make_client,
)
from crdc_pipeline.ingestion.loader import load_document  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "metadata"


def load_manifest() -> list[dict]:
    manifest = json.loads((DATA_DIR / "document_manifest.json").read_text(encoding="utf-8"))
    return manifest["documents"]


def load_reference_metadata() -> dict[str, dict]:
    schema = json.loads((DATA_DIR / "schema_worked_examples.json").read_text(encoding="utf-8"))
    return {d["document_id"]: d for d in schema["documents"]}


def compare_field(label: str, extracted, reference) -> str:
    if reference is None:
        return f"    {label:<16} {str(extracted)!s:<60} (no reference to compare)"
    match = "OK" if _loosely_equal(extracted, reference) else "??"
    return f"  {match} {label:<16} extracted: {extracted!s:<40} reference: {reference!s}"


def _loosely_equal(a, b) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        norm = lambda xs: {str(x).strip().lower() for x in xs}
        return norm(a) == norm(b)
    if a is None or b is None:
        return a == b
    return str(a).strip().lower() == str(b).strip().lower()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    client = make_client()
    reference = load_reference_metadata()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for entry in load_manifest():
        document_id = entry["document_id"]
        filename = entry.get("filename")
        if not filename:
            print(f"[skip] {document_id}: no filename in document_manifest.json yet")
            continue
        pdf_path = DATA_DIR / "raw_pdfs" / filename
        if not pdf_path.is_file():
            print(f"[skip] {document_id}: {pdf_path} not found")
            continue

        print(f"\n=== {document_id} ({filename}) ===")
        document = load_document(pdf_path, document_id=document_id)
        pages = [p.text for p in document.pages]

        extraction = extract_metadata(pages, client)
        metadata = extraction.metadata
        result = {
            "document_id": document_id,
            "page_count": document.page_count,  # from ingestion, not the model
            "resolved_model": extraction.resolved_model,
            **metadata.model_dump(),
        }

        out_path = OUTPUT_DIR / f"{document_id}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  wrote {out_path.relative_to(PROJECT_ROOT)} (model: {extraction.resolved_model})")

        ref = reference.get(document_id)
        if ref is None:
            print("  (no reference entry in schema_worked_examples.json for this document)")
            continue
        print(compare_field("title", metadata.title, ref.get("title")))
        print(compare_field("authors", metadata.authors, ref.get("authors")))
        print(compare_field("organisation", metadata.organisation, ref.get("organisation")))
        # No reference values yet for these newer fields; printed for inspection only.
        print(compare_field("publisher", metadata.publisher, None))
        print(compare_field("funder", metadata.funder, None))
        print(
            compare_field("publication_date", metadata.publication_date, ref.get("publication_date"))
        )
        print(compare_field("submission_date", metadata.submission_date, None))
        print(compare_field("acceptance_date", metadata.acceptance_date, None))
        print(compare_field("source_url", metadata.source_url, ref.get("source_url")))
        if metadata.supersedes_note:
            print(f"    supersedes_note: {metadata.supersedes_note}")


if __name__ == "__main__":
    main()
