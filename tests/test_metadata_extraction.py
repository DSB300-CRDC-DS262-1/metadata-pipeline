"""Prompt, schema, and page-selection tests; the live Gemini test needs an API key."""

import os

import pytest

from crdc_pipeline.extraction.metadata import DocumentMetadata, build_context_text, build_prompt


def test_document_metadata_has_separate_organisation_publisher_funder_fields() -> None:
    metadata = DocumentMetadata(
        title="A Title",
        organisation="A University",
        publisher="A Publisher",
        funder="A Funder",
    )
    assert metadata.organisation == "A University"
    assert metadata.publisher == "A Publisher"
    assert metadata.funder == "A Funder"


def test_document_metadata_has_separate_date_fields_defaulting_to_none() -> None:
    metadata = DocumentMetadata(title="A Title", publication_date="2019")
    assert metadata.publication_date == "2019"
    assert metadata.submission_date is None
    assert metadata.acceptance_date is None


def test_short_document_uses_all_pages() -> None:
    pages = ["page one", "page two"]
    assert build_context_text(pages) == "page one\n\n=== page break ===\n\npage two"


def test_long_document_uses_leading_and_trailing_pages_only() -> None:
    pages = [f"page {i}" for i in range(1, 11)]
    result = build_context_text(pages)
    assert "page 1" in result
    assert "page 2" in result
    assert "page 10" in result
    assert "page 5" not in result
    assert "page 9" not in result


def test_prompt_includes_rules_and_selected_text() -> None:
    prompt = build_prompt(["Title Page Text"])
    assert "Title Page Text" in prompt
    assert "return null" in prompt
    assert "Do not fabricate a source_url" in prompt
    assert "organisation: the author's own affiliated institution" in prompt
    assert "publisher:" in prompt
    assert "funder:" in prompt
    assert "publication_date for whichever one represents" in prompt


@pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="no GEMINI_API_KEY/GOOGLE_API_KEY set -- skipping live API call",
)
def test_live_extraction_on_aghealth2019() -> None:
    from pathlib import Path

    from crdc_pipeline.extraction import extract_metadata, make_backend
    from crdc_pipeline.ingestion.loader import load_document

    pdf_path = Path(__file__).resolve().parent.parent / "data" / "raw_pdfs" / "aghealth2019.pdf"
    if not pdf_path.is_file():
        pytest.skip("data/raw_pdfs/aghealth2019.pdf not found")

    document = load_document(pdf_path, document_id="doc_aghealth2019")
    pages = [p.text for p in document.pages]

    result = extract_metadata(pages, make_backend("gemini"))
    metadata = result.metadata

    assert metadata.title is not None
    assert "cotton" in metadata.title.lower()
    assert any("lower" in a.lower() for a in metadata.authors)
