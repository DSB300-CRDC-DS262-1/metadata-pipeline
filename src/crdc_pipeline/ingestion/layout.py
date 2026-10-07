"""Per-page single- vs two-column layout detection from word positions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    """A word and its bounding box (origin top-left, as in pdfplumber)."""

    text: str
    x0: float
    x1: float
    top: float
    bottom: float


@dataclass(frozen=True)
class LayoutDetection:
    column_count: int
    gutter_x: float | None
    confidence: float
    line_count: int
    lines_respecting_gutter: int


def detect_columns(
    words: list[Word],
    page_width: float,
    page_height: float,
    min_lines: int = 4,
    respect_threshold: float = 0.6,
    margin_ratio: float = 0.30,
    min_gap_ratio: float = 0.02,
    vertical_margin_ratio: float = 0.08,
    y_tolerance: float = 3.0,
    search: tuple[float, float] | None = None,
) -> LayoutDetection:
    """Classify a page as single- or two-column from its word positions.

    `search` (low, high), as shares of the page width, narrows where the gutter is looked for;
    by default it is anywhere between the margins.
    """
    lines = _group_into_lines(words, y_tolerance)
    if len(lines) < min_lines:
        return LayoutDetection(1, None, 1.0, len(lines), len(lines))

    # Running headers/footers can span the gutter and hide it.
    body_words = [
        w
        for w in words
        if w.top >= page_height * vertical_margin_ratio
        and w.bottom <= page_height * (1 - vertical_margin_ratio)
    ]
    gutter = _find_candidate_gutter(
        _group_into_lines(body_words, y_tolerance),
        page_width,
        search or (margin_ratio, 1 - margin_ratio),
        min_gap_ratio,
    )
    if gutter is None:
        return LayoutDetection(1, None, 1.0, len(lines), len(lines))

    gutter_start, gutter_end = gutter
    respecting = sum(
        1
        for line in lines
        if not any(w.x0 < gutter_end and w.x1 > gutter_start for w in line)
    )
    # Both sides must hold prose. A table (labels beside numbers) or a cover page with a short
    # line off to one side leaves a gap too, but not prose on both sides of it.
    clear = [line for line in lines if not any(w.x0 < gutter_end and w.x1 > gutter_start for w in line)]
    left_prose = sum(_prose([w for w in line if w.x1 <= gutter_start]) for line in clear)
    right_prose = sum(_prose([w for w in line if w.x0 >= gutter_end]) for line in clear)
    confidence = respecting / len(lines)
    gutter_x = (gutter_start + gutter_end) / 2

    if confidence >= respect_threshold and min(left_prose, right_prose) >= min_lines:
        return LayoutDetection(2, gutter_x, confidence, len(lines), respecting)
    return LayoutDetection(1, None, confidence, len(lines), respecting)


def _prose(words: list[Word]) -> bool:
    return sum(1 for w in words if sum(c.isalpha() for c in w.text) >= 3) >= 3


def _group_into_lines(words: list[Word], y_tolerance: float) -> list[list[Word]]:
    lines: list[list[Word]] = []
    for w in sorted(words, key=lambda w: (w.top, w.x0)):
        if lines and abs(w.top - lines[-1][0].top) <= y_tolerance:
            lines[-1].append(w)
        else:
            lines.append([w])
    return lines


def _find_candidate_gutter(
    lines: list[list[Word]],
    page_width: float,
    search: tuple[float, float],
    min_gap_ratio: float,
    bins: int = 400,
) -> tuple[float, float] | None:
    """The gutter-wide vertical strip that the fewest lines cross, nearest the page centre.

    Counting crossings rather than requiring an empty strip keeps the gutter visible when a
    full-width figure or table sits above or below the columns.
    """
    if not lines:
        return None

    bin_width = page_width / bins
    crossings = [0] * bins
    for line in lines:
        covered: set[int] = set()
        for w in line:
            start_bin = max(0, min(bins - 1, int(w.x0 / bin_width)))
            end_bin = max(0, min(bins - 1, int(w.x1 / bin_width)))
            covered.update(range(start_bin, end_bin + 1))
        for b in covered:
            crossings[b] += 1

    low = max(0, int(bins * search[0]))
    high = min(bins, int(bins * search[1]))
    width = max(1, round(bins * min_gap_ratio))
    if high - low < width:
        return None

    centre = bins / 2
    start = min(
        range(low, high - width + 1),
        key=lambda s: (max(crossings[s:s + width]), abs(s + width / 2 - centre)),
    )
    worst = max(crossings[start:start + width])
    end = start + width - 1
    while start > low and crossings[start - 1] <= worst:
        start -= 1
    while end < high - 1 and crossings[end + 1] <= worst:
        end += 1
    return start * bin_width, (end + 1) * bin_width
