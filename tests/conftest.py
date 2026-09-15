"""Shared fixtures for building synthetic test PDFs with reportlab.

These stand in for the real sample documents until they're added under
data/raw_pdfs/ -- they're deliberately simple (no real prose) because
their only job is to exercise the ingestion module's column-routing
logic against known, controlled geometry.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

PAGE_WIDTH, PAGE_HEIGHT = 612, 792  # US letter, points

# Cycling filler word-counts so each line ends at a different x position --
# real single-column body text has ragged line lengths from word-wrap, and
# a detector needs to see that variation. Uniform-length lines leave a
# consistent gap near the right margin that looks exactly like a column
# gutter (this is what the first version of this fixture got wrong).
_FILLER_WORD_COUNTS = [0, 5, 10, 2, 8, 1, 11, 4, 7, 0, 9, 3, 6, 10, 1, 8, 4, 11, 2, 6]
_FILLER_WORDS = (
    "across multiple sites over consecutive seasons under varying "
    "conditions and independently verified against agency records"
).split()


def _single_column_line(i: int) -> str:
    base = f"This is single column line number {i} of the report."
    filler_count = _FILLER_WORD_COUNTS[(i - 1) % len(_FILLER_WORD_COUNTS)]
    filler = " ".join(_FILLER_WORDS[j % len(_FILLER_WORDS)] for j in range(filler_count))
    return f"{base} {filler}".rstrip()


def _write_lines(c: canvas.Canvas, lines: list[str], x: float, top_y: float, leading: float = 14) -> None:
    y = top_y
    for line in lines:
        c.drawString(x, y, line)
        y -= leading


@pytest.fixture
def single_column_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "single_column.pdf"
    c = canvas.Canvas(str(path), pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    c.setFont("Helvetica", 11)
    lines = [_single_column_line(i) for i in range(1, 21)]
    _write_lines(c, lines, x=72, top_y=740)
    c.showPage()
    c.save()
    return path


@pytest.fixture
def two_column_pdf(tmp_path: Path) -> Path:
    """A page with a left column (LEFT-1..LEFT-10) and a right column
    (RIGHT-1..RIGHT-10), interleaved row by row -- the way a real
    two-column PDF's content stream is often ordered, which is exactly
    what breaks plain reading-order assumptions. A correct loader must
    still emit all LEFT lines before all RIGHT lines.
    """
    path = tmp_path / "two_column.pdf"
    c = canvas.Canvas(str(path), pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    c.setFont("Helvetica", 11)
    y = 740
    for i in range(1, 11):
        # Interleaved draw order: right-column text is written to the
        # content stream before the left-column text on the same row.
        c.drawString(320, y, f"RIGHT-{i} column sentence about topic B.")
        c.drawString(72, y, f"LEFT-{i} column sentence about topic A.")
        y -= 14
    c.showPage()
    c.save()
    return path


@pytest.fixture
def mixed_layout_pdf(tmp_path: Path) -> Path:
    """Page 1 single-column, page 2 two-column -- exercises per-page
    (not per-document) layout detection and the "mixed" summary label.
    """
    path = tmp_path / "mixed_layout.pdf"
    c = canvas.Canvas(str(path), pagesize=(PAGE_WIDTH, PAGE_HEIGHT))

    c.setFont("Helvetica", 11)
    _write_lines(c, [_single_column_line(i) for i in range(1, 16)], x=72, top_y=740)
    c.showPage()

    c.setFont("Helvetica", 11)
    y = 740
    for i in range(1, 11):
        c.drawString(320, y, f"RIGHT-{i} on the two column page.")
        c.drawString(72, y, f"LEFT-{i} on the two column page.")
        y -= 14
    c.showPage()
    c.save()
    return path
