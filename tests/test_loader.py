"""Loader tests on synthetic PDFs."""

from pathlib import Path

from crdc_pipeline.ingestion.loader import load_document


def test_single_column_document_uses_layout_preserving_extraction(single_column_pdf: Path) -> None:
    doc = load_document(single_column_pdf, document_id="test_single")
    assert doc.page_count == 1
    assert doc.detected_layout == "single_column"
    page = doc.pages[0]
    assert page.column_count == 1
    assert page.extraction_method == "pdfplumber_layout"
    assert "line number 1 " in page.text
    assert "line number 20" in page.text
    assert not page.warnings


def test_two_column_document_reconstructs_reading_order(two_column_pdf: Path) -> None:
    doc = load_document(two_column_pdf, document_id="test_two_col")
    assert doc.detected_layout == "two_column"
    page = doc.pages[0]
    assert page.column_count == 2
    assert page.extraction_method == "pdfplumber_column_crop"

    # All LEFT lines must precede all RIGHT lines.
    last_left = page.text.rindex("LEFT-10")
    first_right = page.text.index("RIGHT-1 ")
    assert last_left < first_right

    for i in range(1, 11):
        assert f"LEFT-{i} column sentence about topic A." in page.text
        assert f"RIGHT-{i} column sentence about topic B." in page.text


def test_mixed_layout_document_labelled_mixed(mixed_layout_pdf: Path) -> None:
    doc = load_document(mixed_layout_pdf, document_id="test_mixed")
    assert doc.page_count == 2
    assert doc.detected_layout == "mixed"
    assert doc.pages[0].column_count == 1
    assert doc.pages[1].column_count == 2


def test_missing_file_raises() -> None:
    try:
        load_document("does_not_exist.pdf", document_id="missing")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for a missing PDF path")
