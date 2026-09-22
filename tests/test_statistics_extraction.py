"""Tests for statistics extraction.

Page-selection and prompt-building are pure functions, tested directly.
The actual Gemini call is only tested end-to-end when GEMINI_API_KEY is
available -- these tests are not run by default and cost real API
calls when they do run.
"""

import os

import pytest

from crdc_pipeline.extraction.statistics import StatisticEntry, build_context_text, build_prompt


def test_build_context_text_marks_every_page() -> None:
    result = build_context_text(["first page", "second page"])
    assert "=== page 1 ===\nfirst page" in result
    assert "=== page 2 ===\nsecond page" in result


def test_build_context_text_keeps_all_pages_unlike_metadata() -> None:
    pages = [f"page {i}" for i in range(1, 11)]
    result = build_context_text(pages)
    assert "page 5" in result
    assert "page 9" in result


def test_prompt_includes_rules_and_selected_text() -> None:
    prompt = build_prompt(["Some Page Text"])
    assert "Some Page Text" in prompt
    assert "censored_below" in prompt
    assert "model_assumption" in prompt


def test_statistic_entry_note_defaults_to_none() -> None:
    entry = StatisticEntry(
        variable="a measurement",
        value=1.0,
        value_status="exact",
        reported_text="one",
        unit="things",
        record_type="measurement",
        time_period=None,
        page=1,
        source_excerpt="There was one thing.",
    )
    assert entry.note is None


@pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="no GEMINI_API_KEY/GOOGLE_API_KEY set -- skipping live API call",
)
def test_live_extraction_on_aghealth2019() -> None:
    from pathlib import Path

    from crdc_pipeline.extraction.statistics import extract_statistics, make_client
    from crdc_pipeline.ingestion.loader import load_document

    pdf_path = Path(__file__).resolve().parent.parent / "data" / "raw_pdfs" / "aghealth2019.pdf"
    if not pdf_path.is_file():
        pytest.skip("data/raw_pdfs/aghealth2019.pdf not found")

    document = load_document(pdf_path, document_id="doc_aghealth2019")
    pages = [p.text for p in document.pages]

    result = extract_statistics(pages, make_client())

    assert len(result.statistics) > 0
    assert any(s.value == 9 for s in result.statistics)
