"""Loads a PDF into per-page text, choosing the extraction method per page by column layout."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pdfplumber

from .layout import Word, detect_columns
from .symbols import LIGATURES, SymbolRules, hidden_glyphs

X_TOLERANCE = 1.5

# Learned font-and-glyph corrections, shared across documents (see symbols.py).
DEFAULT_RULES_PATH = Path(__file__).resolve().parents[3] / "data" / "symbol_rules.json"

# Set True to strip email addresses before any text reaches a model.
REDACT_EMAILS = False
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

_DASH_TRANSLATION = str.maketrans(
    {
        "‐": "-",  # hyphen
        "‑": "-",  # non-breaking hyphen
        "‒": "-",  # figure dash
        "–": "-",  # en dash
        "—": "-",  # em dash
        "−": "-",  # minus sign
        **LIGATURES,
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
    unreadable_glyphs: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class Document:
    document_id: str
    source_path: str
    page_count: int
    pages: list[Page]
    detected_layout: str  # "single_column" | "two_column" | "mixed" | "unknown"
    warnings: list[str] = field(default_factory=list)


def load_document(
    pdf_path: str | Path,
    document_id: str,
    vision: Callable[[bytes], str] | None = None,
    read_text: Callable[[bytes], str] | None = None,
    rules: SymbolRules | None = None,
) -> Document:
    """Load a PDF, detecting the column layout of each page.

    Glyphs with learned corrections are repaired first. If `read_text` is given, mislabelled
    glyphs without a rule are read from a word image and new rules are saved. If `vision` is
    given, pages that still have unreadable glyphs are transcribed from a page image instead.
    """
    if rules is None:
        rules = SymbolRules(DEFAULT_RULES_PATH)
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    pages: list[Page] = []
    column_counts: list[int] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, plumber_page in enumerate(pdf.pages, start=1):
            unreadable = _repair_chars(plumber_page, rules, read_text, document_id)
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
            if unreadable or not text.strip():
                if vision is not None:
                    image = plumber_page.to_image(resolution=150).original
                    try:
                        text = _normalise(vision(_png_bytes(image)))
                        method = "vision"
                    except RuntimeError as exc:
                        page_warnings.append(f"vision fallback failed, kept the text layer: {exc}")
                elif unreadable:
                    page_warnings.append(
                        f"{unreadable} characters show on the page but have no text in the "
                        "PDF text layer; numbers on this page may be missing digits"
                    )
            if REDACT_EMAILS:
                text = _EMAIL.sub("[email removed]", text)
            char_count = len(text.strip())
            if char_count == 0:
                page_warnings.append(
                    "no text extracted from this page; pass vision= to read it from the page image"
                )

            pages.append(
                Page(
                    page_number=i,
                    text=text,
                    column_count=detection.column_count,
                    extraction_method=method,
                    char_count=char_count,
                    word_count=len(text.split()),
                    unreadable_glyphs=unreadable,
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


def _repair_chars(
    plumber_page: pdfplumber.page.Page,
    rules: SymbolRules,
    read_text: Callable[[bytes], str] | None,
    document_id: str,
) -> int:
    """Repair mislabelled glyphs in place and return how many glyphs remain unreadable."""
    chars = plumber_page.chars
    rules.apply(chars)
    if read_text is not None and rules.suspicious(chars):
        rules.learn(chars, lambda bbox: _crop_png(plumber_page, bbox), read_text, document_id)
        rules.apply(chars)
    return len(rules.suspicious(chars)) + hidden_glyphs(chars)


def _crop_png(plumber_page: pdfplumber.page.Page, bbox: tuple[float, float, float, float]) -> bytes:
    x0, top, x1, bottom = bbox
    clamped = (max(0, x0), max(0, top), min(plumber_page.width, x1), min(plumber_page.height, bottom))
    return _png_bytes(plumber_page.crop(clamped).to_image(resolution=300).original)


def _png_bytes(image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


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
