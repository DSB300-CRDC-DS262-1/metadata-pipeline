"""Column detection tests on synthetic word positions."""

from crdc_pipeline.ingestion.layout import Word, detect_columns

PAGE_WIDTH = 600.0
PAGE_HEIGHT = 800.0


def _line(text_words: list[str], top: float, x0: float, word_width: float = 40.0) -> list[Word]:
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
        words += _line(["word"] * 10, top, x0=50)
    result = detect_columns(words, PAGE_WIDTH, PAGE_HEIGHT)
    assert result.column_count == 1


def test_two_column_layout_detected() -> None:
    words: list[Word] = []
    for row in range(10):
        top = row * 15.0
        words += _line(["left"] * 4, top, x0=50)
        words += _line(["right"] * 4, top, x0=350)
    result = detect_columns(words, PAGE_WIDTH, PAGE_HEIGHT)
    assert result.column_count == 2
    assert result.gutter_x is not None
    assert 245 < result.gutter_x < 350


def test_full_width_heading_above_two_column_body_still_detects_two_columns() -> None:
    """A full-width heading above a two-column body still reads as two columns."""
    words: list[Word] = []
    words += _line(["A", "Full", "Width", "Heading", "Spanning", "Everything"], top=0, x0=50)
    for row in range(1, 11):
        top = row * 15.0
        words += _line(["left"] * 4, top, x0=50)
        words += _line(["right"] * 4, top, x0=350)
    result = detect_columns(words, PAGE_WIDTH, PAGE_HEIGHT)
    assert result.column_count == 2


def test_mostly_full_width_with_incidental_gap_is_single_column() -> None:
    """A few short lines with a gap do not make a mostly full-width page two-column."""
    words: list[Word] = []
    for row in range(10):
        top = row * 15.0
        if row < 8:
            words += _line(["word"] * 12, top, x0=50)
        else:
            words += _line(["left"] * 3, top, x0=50)
            words += _line(["right"] * 3, top, x0=350)
    result = detect_columns(words, PAGE_WIDTH, PAGE_HEIGHT)
    assert result.column_count == 1


def test_sparse_page_defaults_to_single_column() -> None:
    words = _line(["Figure", "1"], top=0, x0=50)
    result = detect_columns(words, PAGE_WIDTH, PAGE_HEIGHT)
    assert result.column_count == 1


def test_empty_page_defaults_to_single_column() -> None:
    result = detect_columns([], PAGE_WIDTH, PAGE_HEIGHT)
    assert result.column_count == 1
    assert result.line_count == 0


def test_full_width_running_header_does_not_hide_a_narrow_gutter() -> None:
    """A full-width merged header must not hide a narrow gutter."""
    words: list[Word] = []
    words.append(Word("RunningHeaderCrossingTheGutter", 200, 400, top=5, bottom=13))
    for row in range(1, 11):
        top = row * 15.0
        words += _line(["left"] * 6, top, x0=50)
        words += _line(["right"] * 6, top, x0=345)

    result = detect_columns(words, PAGE_WIDTH, PAGE_HEIGHT, min_gap_ratio=0.02)
    assert result.column_count == 2


def test_full_width_table_above_two_columns_still_detects_the_gutter() -> None:
    """A table spanning both columns leaves no empty strip, but the columns below still count."""
    words: list[Word] = []
    for row in range(6):
        words += _line(["Label", "12.5", "13.0", "14.2", "15.1", "16.0", "17.3", "18.8"], row * 15.0, x0=50, word_width=55)
    for row in range(6, 20):
        top = row * 15.0
        words += _line(["left"] * 5, top, x0=50)
        words += _line(["right"] * 5, top, x0=330)
    result = detect_columns(words, PAGE_WIDTH, PAGE_HEIGHT)
    assert result.column_count == 2
    assert 270 < result.gutter_x < 330


def test_table_page_with_prose_on_one_side_is_single_column() -> None:
    """Table rows (a label beside numbers) leave a gap, but there is prose on only one side."""
    words: list[Word] = []
    for row in range(12):
        top = row * 15.0
        words += _line(["Falls", "on", "the", "same", "level"], top, x0=50)
        words += _line(["25", "(6)", "60", "(12)"], top, x0=350)
    result = detect_columns(words, PAGE_WIDTH, PAGE_HEIGHT)
    assert result.column_count == 1
