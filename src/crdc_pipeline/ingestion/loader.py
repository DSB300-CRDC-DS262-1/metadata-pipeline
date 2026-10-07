"""Loads a PDF into per-page text, choosing the extraction method per page by column layout."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pdfplumber

from .layout import LayoutDetection, Word, detect_columns
from .symbols import LIGATURES, SymbolRules, hidden_glyphs

X_TOLERANCE = 1.5
# A page gutter further than this (share of page width) from the document's usual one is rechecked.
USUAL_GUTTER_TOLERANCE = 0.05

# Learned font-and-glyph corrections, shared across documents (see symbols.py).
DEFAULT_RULES_PATH = Path(__file__).resolve().parents[3] / "data" / "symbol_rules.json"

# A page whose tokens are mostly not word-like is read from its image when vision is available.
GARBLED_MIN_TOKENS = 30
GARBLED_THRESHOLD = 0.5
# One unreadable character is usually a symbol (an accented letter); lost digits come several to a page.
VISION_MIN_UNREADABLE = 2
BROKEN_EMAIL = re.compile(r"\w@\w+\.[ \t]+(?:au|com|org|net|edu|gov)\b", re.I)  # a split on one line, not a wrapped domain
PLAUSIBLE_TOKEN = re.compile(r"[A-Za-z][a-z]{2,}[.,;:)]?|[A-Z]{2,}[.,;:]?|[a-z]{1,2}[.,]?|\d[\d.,%]*")

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
    vision_on_unreadable: bool = True,
) -> Document:
    """Load a PDF, detecting the column layout of each page.

    Glyphs with learned corrections are repaired first. If `read_text` is given, mislabelled
    glyphs without a rule are read from a word image and new rules are saved. If `vision` is
    given, pages that still have unreadable glyphs, garbled text or no text are transcribed from a
    page image instead; with `vision_on_unreadable=False` only garbled or empty pages are (a few
    unreadable digits matter for statistics, but a transcription can lose other text).
    """
    if rules is None:
        rules = SymbolRules(DEFAULT_RULES_PATH)
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    pages: list[Page] = []
    column_counts: list[int] = []

    with pdfplumber.open(pdf_path) as pdf:
        unreadable_counts = [_repair_chars(p, rules, read_text, document_id) for p in pdf.pages]
        detections = _detect_layouts(pdf.pages)
        for i, (plumber_page, unreadable, detection) in enumerate(
            zip(pdf.pages, unreadable_counts, detections), start=1
        ):
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
            garbled = is_garbled(text) or has_broken_email(text)
            if (unreadable >= VISION_MIN_UNREADABLE and vision_on_unreadable) or garbled or not text.strip():
                if vision is not None:
                    image = plumber_page.to_image(resolution=150).original
                    try:
                        text = _normalise(vision(_png_bytes(image)))
                        method = "vision"
                    except RuntimeError as exc:
                        page_warnings.append(f"vision fallback failed, kept the text layer: {exc}")
            if unreadable and method != "vision":
                page_warnings.append(
                    f"{unreadable} characters show on the page but have no text in the "
                    "PDF text layer; numbers on this page may be missing digits"
                )
            if garbled and method != "vision":
                page_warnings.append(
                    "the text layer looks garbled (e.g. poor OCR of a scan); pass vision= to read the page image"
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


def _detect_layouts(plumber_pages: list) -> list[LayoutDetection]:
    """Column layout per page, checking odd pages against the document's usual gutter.

    Pages of one document share a template, so where a page's own search finds no gutter or
    one far from the others (a figure leaving part of a column empty), the search is repeated
    close to the usual gutter.
    """
    pages = [
        ([Word(w["text"], w["x0"], w["x1"], w["top"], w["bottom"]) for w in p.extract_words()],
         float(p.width), float(p.height))
        for p in plumber_pages
    ]
    detections = [detect_columns(words, width, height) for words, width, height in pages]
    gutters = sorted(d.gutter_x / w for d, (_, w, _) in zip(detections, pages) if d.gutter_x is not None)
    if len(gutters) < 2:
        return detections
    usual = gutters[len(gutters) // 2]
    for i, ((words, width, height), detection) in enumerate(zip(pages, detections)):
        if detection.gutter_x is None or abs(detection.gutter_x / width - usual) > USUAL_GUTTER_TOLERANCE:
            near = (usual - USUAL_GUTTER_TOLERANCE, usual + USUAL_GUTTER_TOLERANCE)
            retry = detect_columns(words, width, height, search=near)
            if retry.column_count == 2:
                detections[i] = retry
    return detections


def has_broken_email(text: str) -> bool:
    """An email address the OCR split with spaces ("sunil. tennakoon@CSiro. au"); the image has it whole."""
    return bool(BROKEN_EMAIL.search(text))


def is_garbled(text: str) -> bool:
    """True when too few tokens look like words or numbers, as with poor OCR or spaced-out letters."""
    tokens = text.split()
    if len(tokens) < GARBLED_MIN_TOKENS:
        return False
    plausible = sum(1 for t in tokens if PLAUSIBLE_TOKEN.fullmatch(t) and re.search(r"[aeiouyAEIOUY\d]", t))
    return plausible / len(tokens) < GARBLED_THRESHOLD


def _extract_two_column_text(plumber_page: pdfplumber.page.Page, gutter_x: float) -> str:
    """Left column then right column, band by band.

    A block of full-width lines (a figure or table across both columns) is read across the
    page, and the columns above and below it are read separately, so the block does not get
    mixed into the column text.
    """
    # The page box need not start at 0 (some PDFs have a negative origin), so crop within it.
    left, page_top, right, page_bottom = plumber_page.bbox
    bands = _bands(plumber_page.extract_words(x_tolerance=X_TOLERANCE), gutter_x)
    middles = [(bands[i][2] + bands[i + 1][1]) / 2 for i in range(len(bands) - 1)]
    edges = [page_top] + [min(max(m, page_top), page_bottom) for m in middles] + [page_bottom]
    parts = []
    for (full_width, _, _), top, bottom in zip(bands, edges, edges[1:]):
        if bottom <= top:
            continue
        boxes = [(left, top, right, bottom)] if full_width else [(left, top, gutter_x, bottom), (gutter_x, top, right, bottom)]
        for box in boxes:
            text = _normalise((plumber_page.crop(box).extract_text(x_tolerance=X_TOLERANCE) or "").strip())
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _bands(words: list[dict], gutter_x: float, min_block_lines: int = 3) -> list[tuple[bool, float, float]]:
    """(full_width, top, bottom) bands from top to bottom; single crossing lines stay in the columns."""
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(w["top"] - lines[-1][0]["top"]) <= 3:
            lines[-1].append(w)
        else:
            lines.append([w])
    crossing = [any(w["x0"] < gutter_x < w["x1"] for w in line) for line in lines]
    # Only runs of several crossing lines are full-width blocks.
    full = crossing[:]
    i = 0
    while i < len(lines):
        j = i
        while j < len(lines) and crossing[j]:
            j += 1
        if j - i < min_block_lines:
            full[i:j] = [False] * (j - i)
        i = max(j, i + 1)
    bands: list[tuple[bool, float, float]] = []
    for line, is_full in zip(lines, full):
        top, bottom = min(w["top"] for w in line), max(w["bottom"] for w in line)
        if bands and bands[-1][0] == is_full:
            bands[-1] = (is_full, bands[-1][1], bottom)
        else:
            bands.append((is_full, top, bottom))
    return bands or [(False, 0.0, 0.0)]


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
