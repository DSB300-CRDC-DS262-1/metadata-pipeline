"""Unit tests for column detection. These use synthetic word lists, not
real PDFs, so they can run before any sample document is available and
pin down the detection logic in isolation.
"""

from crdc_pipeline.ingestion.layout import Word, detect_columns

PAGE_WIDTH = 600.0


def _line(text_words: list[str], top: float, x0: float, word_width: float = 40.0) -> list[Word]:
    """Lay out words left-to-right on one line starting at x0."""
    words = []
    x = x0
    for t in text_words:
        words.append(Word(t, x, x + word_width, top, top + 10))
        x += word_width + 5
    return words


def test_single_column_full_width_text() -> None:
    words: list[Word] = []
    for row in range(10):
        top = row * 15.0
        words += _line(["word"] * 10, top, x0=50)  # spans ~50 to ~495, no consistent gap
    result = detect_columns(words, PAGE_WIDTH)
    assert result.column_count == 1


def test_two_column_layout_detected() -> None:
    words: list[Word] = []
    for row in range(10):
        top = row * 15.0
        words += _line(["left"] * 4, top, x0=50)  # ~50 to ~245
        words += _line(["right"] * 4, top, x0=350)  # ~350 to ~545
    result = detect_columns(words, PAGE_WIDTH)
    assert result.column_count == 2
    assert result.gutter_x is not None
    assert 245 < result.gutter_x < 350


def test_full_width_heading_above_two_column_body_still_detects_two_columns() -> None:
    """A title/heading spanning the full width shouldn't stop the page
    below it from being correctly read as two-column, since most lines
    still respect the gutter."""
    words: list[Word] = []
    words += _line(["A", "Full", "Width", "Heading", "Spanning", "Everything"], top=0, x0=50)
    for row in range(1, 11):
        top = row * 15.0
        words += _line(["left"] * 4, top, x0=50)
        words += _line(["right"] * 4, top, x0=350)
    result = detect_columns(words, PAGE_WIDTH)
    assert result.column_count == 2


def test_mostly_full_width_with_incidental_gap_is_single_column() -> None:
    """A handful of short lines that happen to leave a gap shouldn't be
    enough to call the page two-column if most lines cross it."""
    words: list[Word] = []
    for row in range(10):
        top = row * 15.0
        if row < 8:
            words += _line(["word"] * 12, top, x0=50)  # crosses the whole middle
        else:
            words += _line(["left"] * 3, top, x0=50)
            words += _line(["right"] * 3, top, x0=350)
    result = detect_columns(words, PAGE_WIDTH)
    assert result.column_count == 1


def test_sparse_page_defaults_to_single_column() -> None:
    words = _line(["Figure", "1"], top=0, x0=50)
    result = detect_columns(words, PAGE_WIDTH)
    assert result.column_count == 1


def test_empty_page_defaults_to_single_column() -> None:
    result = detect_columns([], PAGE_WIDTH)
    assert result.column_count == 1
    assert result.line_count == 0
