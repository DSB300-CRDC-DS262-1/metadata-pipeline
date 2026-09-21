"""Loads a PDF into per-page text, choosing the extraction method per page by column layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

from .layout import Word, detect_columns

X_TOLERANCE = 1.5

_DASH_TRANSLATION = str.maketrans(
    {
        "‐": "-",  # hyphen
        "‑": "-",  # non-breaking hyphen
        "‒": "-",  # figure dash
        "–": "-",  # en dash
        "—": "-",  # em dash
        "−": "-",  # minus sign
    }
)


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
    """Load a PDF, detecting the column layout of each page."""
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
            detection = detect_columns(
                words, float(plumber_page.width), float(plumber_page.height)
            )
            column_counts.append(detection.column_count)

            if detection.column_count == 2 and detection.gutter_x is not None:
                text = _extract_two_column_text(plumber_page, detection.gutter_x)
                method = "pdfplumber_column_crop"
            else:
                text = _normalise(
                    plumber_page.extract_text(layout=True, x_tolerance=X_TOLERANCE) or ""
                )
                method = "pdfplumber_layout"

            page_warnings: list[str] = []
            char_count = len(text.strip())
            if char_count == 0:
                # No text at all: flag it for a later OCR fallback.
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
    width, height = plumber_page.width, plumber_page.height
    left = plumber_page.crop((0, 0, gutter_x, height))
    right = plumber_page.crop((gutter_x, 0, width, height))
    left_text = _normalise((left.extract_text(x_tolerance=X_TOLERANCE) or "").strip())
    right_text = _normalise((right.extract_text(x_tolerance=X_TOLERANCE) or "").strip())
    return f"{left_text}\n\n{right_text}"


def _normalise(text: str) -> str:
    return text.translate(_DASH_TRANSLATION)


def _summarise_layout(column_counts: list[int]) -> str:
    if not column_counts:
        return "unknown"
    unique = set(column_counts)
    if unique == {1}:
        return "single_column"
    if unique == {2}:
        return "two_column"
    return "mixed"
