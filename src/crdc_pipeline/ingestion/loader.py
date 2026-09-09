"""PDF ingestion: turns a source PDF into per-page text, choosing an
extraction path per page based on its detected column layout.

Single-column pages use `pdftotext -layout`, which preserves reading
order well when there is only one column to preserve. Two-column pages
are extracted column-by-column with pdfplumber -- crop the left half,
crop the right half at the detected gutter, extract each top-to-bottom,
then concatenate left-then-right -- because `pdftotext -layout`
corrupts two-column reading order (docs/weaknesses_and_fixes.md, issue
1) and plain `pdftotext`'s stream order isn't guaranteed to match
visual reading order either. Reconstructing order from word geometry
sidesteps both tools' ordering assumptions.

Requires the poppler `pdftotext` binary on PATH (used only for
single-column pages); pdfplumber (used for layout detection and all
two-column extraction) is a pure Python dependency.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

from .layout import Word, detect_columns

PDFTOTEXT_BIN = "pdftotext"


@dataclass
class Page:
    page_number: int  # 1-indexed
    text: str
    column_count: int
    extraction_method: str
    char_count: int
    word_count: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class Document:
    document_id: str
    source_path: str
    page_count: int
    pages: list[Page]
    detected_layout: str  # "single_column" | "two_column" | "mixed" | "unknown"
    warnings: list[str] = field(default_factory=list)


def load_document(pdf_path: str | Path, document_id: str) -> Document:
    """Load a PDF, detecting and correcting for column layout per page.

    Raises FileNotFoundError if pdf_path doesn't exist, and
    RuntimeError if the pdftotext binary is missing or fails on a
    single-column page.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    pages: list[Page] = []
    column_counts: list[int] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, plumber_page in enumerate(pdf.pages, start=1):
            words = [
                Word(w["text"], w["x0"], w["x1"], w["top"], w["bottom"])
                for w in plumber_page.extract_words()
            ]
            detection = detect_columns(words, float(plumber_page.width))
            column_counts.append(detection.column_count)

            if detection.column_count == 2 and detection.gutter_x is not None:
                text = _extract_two_column_text(plumber_page, detection.gutter_x)
                method = "pdfplumber_column_crop"
            else:
                text = _run_pdftotext(pdf_path, i, layout=True)
                method = "pdftotext_layout"

            page_warnings: list[str] = []
            char_count = len(text.strip())
            if char_count == 0:
                # Known failure mode: some PDFs have glyphs missing from
                # the text layer entirely (docs/weaknesses_and_fixes.md,
                # issue 2). This module can only flag it -- recovering
                # the value needs an OCR or multimodal fallback, which
                # is a later stage, not ingestion.
                page_warnings.append(
                    "no text extracted from this page; possible text-layer "
                    "gap (see weaknesses_and_fixes.md issue 2), needs OCR "
                    "or multimodal fallback to confirm"
                )

            pages.append(
                Page(
                    page_number=i,
                    text=text,
                    column_count=detection.column_count,
                    extraction_method=method,
                    char_count=char_count,
                    word_count=len(text.split()),
                    warnings=page_warnings,
                )
            )

    return Document(
        document_id=document_id,
        source_path=str(pdf_path),
        page_count=len(pages),
        pages=pages,
        detected_layout=_summarise_layout(column_counts),
    )


def _extract_two_column_text(plumber_page: pdfplumber.page.Page, gutter_x: float) -> str:
    """Extract a two-column page in correct reading order via cropping."""
    width, height = plumber_page.width, plumber_page.height
    left = plumber_page.crop((0, 0, gutter_x, height))
    right = plumber_page.crop((gutter_x, 0, width, height))
    left_text = (left.extract_text() or "").strip()
    right_text = (right.extract_text() or "").strip()
    return f"{left_text}\n\n{right_text}"


def _run_pdftotext(pdf_path: Path, page_number: int, layout: bool) -> str:
    args = [PDFTOTEXT_BIN, "-f", str(page_number), "-l", str(page_number)]
    if layout:
        args.append("-layout")
    args += [str(pdf_path), "-"]
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "pdftotext binary not found on PATH (requires poppler-utils)"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"pdftotext failed on {pdf_path} page {page_number}: {result.stderr.strip()}"
        )
    return result.stdout


def _summarise_layout(column_counts: list[int]) -> str:
    if not column_counts:
        return "unknown"
    unique = set(column_counts)
    if unique == {1}:
        return "single_column"
    if unique == {2}:
        return "two_column"
    return "mixed"
