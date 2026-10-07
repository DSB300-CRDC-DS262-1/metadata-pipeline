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


def _pdf(tmp_path, draw):
    from reportlab.pdfgen import canvas

    path = tmp_path / "doc.pdf"
    c = canvas.Canvas(str(path), pagesize=(612, 792))
    draw(c)
    c.showPage()
    c.save()
    return path


def test_email_addresses_kept_unless_redaction_is_switched_on(tmp_path, monkeypatch) -> None:
    from crdc_pipeline.ingestion import loader

    def draw(c):
        c.setFont("Helvetica", 11)
        c.drawString(72, 740, "Contact: jane.citizen@example.gov.au for details.")

    path = _pdf(tmp_path, draw)
    assert "jane.citizen@example.gov.au" in load_document(path, "t").pages[0].text

    monkeypatch.setattr(loader, "REDACT_EMAILS", True)
    page = load_document(path, "t").pages[0]
    assert "example.gov.au" not in page.text
    assert "[email removed]" in page.text


def test_vision_is_used_for_pages_without_text(tmp_path) -> None:
    seen = []

    def vision(png: bytes) -> str:
        seen.append(png[:8])
        return "Scanned page text 73%"

    path = _pdf(tmp_path, lambda c: c.rect(100, 100, 200, 200))
    page = load_document(path, "t", vision=vision).pages[0]
    assert page.extraction_method == "vision"
    assert page.text == "Scanned page text 73%"
    assert seen == [b"\x89PNG\r\n\x1a\n"]


def test_failed_vision_keeps_text_layer_with_warning(tmp_path) -> None:
    def vision(png: bytes) -> str:
        raise RuntimeError("model returned nothing")

    path = _pdf(tmp_path, lambda c: c.rect(100, 100, 200, 200))
    page = load_document(path, "t", vision=vision).pages[0]
    assert page.extraction_method == "pdfplumber_layout"
    assert any("vision fallback failed" in w for w in page.warnings)


def test_full_width_block_is_its_own_band() -> None:
    from crdc_pipeline.ingestion.loader import _bands

    def word(x0, x1, top):
        return {"x0": x0, "x1": x1, "top": top, "bottom": top + 10}

    column = [word(50, 250, t) for t in (0, 15)] + [word(350, 550, t) for t in (0, 15)]
    table = [word(50, 550, t) for t in (40, 55, 70)]
    stray = [word(200, 400, 95)]  # one crossing line stays in the columns
    below = [word(50, 250, t) for t in (110, 125)]
    bands = _bands(column + table + stray + below, gutter_x=300)
    assert [full for full, _, _ in bands] == [False, True, False]


def test_garbled_text_is_recognised() -> None:
    from crdc_pipeline.ingestion.loader import is_garbled

    prose = "The economic model includes loss of future earnings and household contributions. " * 4
    ocr = "Devel pm i te ated. f 11. ate sf ariage tsy co o rodi. ICti ,,, g R I @F. 70 re\"@t r, @\", v, @ " * 3
    spaced = "C o t t o n G r o w i n g P r a c t i c e s 2 0 1 6 F i n d i n g s " * 2
    assert not is_garbled(prose)
    assert is_garbled(ocr)
    assert is_garbled(spaced)
    assert not is_garbled("Figure 1 @@ ## 3")  # too short to judge


def test_emails_split_by_ocr_are_recognised() -> None:
    from crdc_pipeline.ingestion.loader import has_broken_email

    assert has_broken_email("Email sunil. tennakoon@CSiro. au, scott. johnslon@CSiro. all")
    assert not has_broken_email("Email sunil.tennakoon@csiro.au")
    assert not has_broken_email("marketing@austcottonshippers.\ncom.au")
