"""Synthetic test PDFs built with reportlab."""

from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

PAGE_WIDTH, PAGE_HEIGHT = 612, 792

# Varying line lengths, like real wrapped text (uniform lines look like a gutter).
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
    """Left and right column lines drawn interleaved, as in real PDFs."""
    path = tmp_path / "two_column.pdf"
    c = canvas.Canvas(str(path), pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    c.setFont("Helvetica", 11)
    y = 740
    for i in range(1, 11):
        c.drawString(320, y, f"RIGHT-{i} column sentence about topic B.")
        c.drawString(72, y, f"LEFT-{i} column sentence about topic A.")
        y -= 14
    c.showPage()
    c.save()
    return path


@pytest.fixture
def mixed_layout_pdf(tmp_path: Path) -> Path:
    """Page 1 single-column, page 2 two-column."""
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
