"""PDF ingestion: turns a source PDF into per-page text, choosing an
extraction path per page based on its detected column layout.

Everything here runs through pdfplumber -- no external binary
dependency. Single-column pages use `extract_text(layout=True)`,
pdfplumber's own whitespace-preserving reconstruction (equivalent to
`pdftotext -layout`, confirmed to match it on real single-column pages
in this corpus, content-for-content). Two-column pages are extracted
column-by-column -- crop the left half, crop the right half at the
detected gutter, extract each top-to-bottom, then concatenate
left-then-right -- because layout-preserving extraction corrupts
two-column reading order by interleaving both columns line by line
(docs/weaknesses_and_fixes.md, issue 1). Reconstructing order from word
geometry sidesteps that entirely.

An earlier version of this module shelled out to `pdftotext` for the
single-column path. That was dropped: it added a dependency on
whichever `pdftotext` build happened to be on PATH, and different
builds disagree on supported flags (the poppler build supports
crop-box options a project might eventually want; the Xpdf-derived
build bundled with Git for Windows does not) -- exactly the kind of
works-on-my-machine fragility a pure Python dependency avoids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

from .layout import Word, detect_columns

#: pdfplumber's own default x_tolerance (3pt) merges adjacent words with
#: no visible space into a single run on some tightly-kerned justified
#: fonts (confirmed on the Hedayati PDF: "agriculturalresources(Kong..."
#: instead of "agricultural resources (Kong..."). A tighter tolerance
#: fixes it. Applied everywhere, not just the two-column path, since
#: nothing rules out the same font behaviour on a single-column page.
X_TOLERANCE = 1.5

#: PDF fonts often use a typographic hyphen/dash glyph distinct from the
#: plain ASCII "-" (confirmed on aghealth2019: "3‐5 years", using
#: HYPHEN U+2010, where a naive read might expect "-"). This pipeline
#: only cares about the underlying value, not preserving exact source
#: typography, so every dash-like glyph is normalised to ASCII "-" here
#: rather than leaving every later regex to account for each variant.
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
    """Load a PDF, detecting and correcting for column layout per page.

    Raises FileNotFoundError if pdf_path doesn't exist.
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
