"""Column-layout detection for PDF pages.

Detects whether a page uses a single-column or two-column layout from
the horizontal positions of its extracted words, independent of any
particular PDF text-extraction tool. This has to run *before* any text
is handed to a layout-preserving extractor: `pdftotext -layout` reads a
two-column page left-to-right, line by line, splicing unrelated
sentences from each column together on the same output line (see
docs/weaknesses_and_fixes.md, issue 1). Detecting the column count
first lets the loader route each page through whichever extraction
method actually produces correct reading order for that page.

Detection is two-pass and per page (not per document), because a
two-column journal article can still have full-width elements --
titles, abstracts, figure captions -- on the same page as its
two-column body (the Hedayati paper is like this: ground truth cites
its abstract ("page: abstract") separately from the two-column body
pages).

    1. Find a candidate gutter: the widest empty vertical band away
       from the page margins, from a histogram of word positions.
    2. Validate it: check what fraction of actual text *lines* respect
       that gutter (no word crosses it). A full-width heading creates
       a misleading gap in the global histogram without any individual
       line straddling it in a way that breaks the check, but a page
       that is mostly full-width text with only an incidental gap will
       fail this per-line check and correctly come back single-column.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    """A single extracted word and its bounding box, in page coordinates
    (origin top-left, matching pdfplumber's convention)."""

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
    min_lines: int = 4,
    respect_threshold: float = 0.6,
    margin_ratio: float = 0.30,
    min_gap_ratio: float = 0.02,
    y_tolerance: float = 3.0,
) -> LayoutDetection:
    """Classify a page as single- or two-column from its word positions.

    Args:
        words: words on the page with bounding boxes.
        page_width: page width in the same units as the word coordinates.
        min_lines: pages with fewer text lines than this (e.g. a mostly
            blank cover page) default to single-column rather than
            risk a confident call on too little evidence.
        respect_threshold: minimum fraction of lines that must respect
            the candidate gutter for the page to be called two-column.
        margin_ratio: fraction of page width excluded from gutter search
            at each edge. A real column gutter sits roughly in the middle
            of the page, not near the edges -- a wide margin_ratio also
            protects against ragged-right single-column text, where lines
            of similar length can otherwise leave a misleadingly
            consistent gap near the right margin.
        min_gap_ratio: minimum width (as a fraction of page width) an
            empty band must have to be considered a candidate gutter. A
            real two-column journal gutter can be narrow -- as little as
            ~2-3% of page width -- so this needs to be set low enough to
            catch that, not just the wide gaps of a magazine-style layout.
        y_tolerance: vertical distance within which words are grouped
            into the same line.

    Returns:
        A LayoutDetection with column_count 1 or 2. gutter_x is the
        detected column boundary (in page x-coordinates) when
        column_count is 2, else None.
    """
    lines = _group_into_lines(words, y_tolerance)
    if len(lines) < min_lines:
        return LayoutDetection(1, None, 1.0, len(lines), len(lines))

    gutter = _find_candidate_gutter(words, page_width, margin_ratio, min_gap_ratio)
    if gutter is None:
        return LayoutDetection(1, None, 1.0, len(lines), len(lines))

    gutter_start, gutter_end = gutter
    respecting = sum(
        1
        for line in lines
        if not any(w.x0 < gutter_end and w.x1 > gutter_start for w in line)
    )
    confidence = respecting / len(lines)
    gutter_x = (gutter_start + gutter_end) / 2

    if confidence >= respect_threshold:
        return LayoutDetection(2, gutter_x, confidence, len(lines), respecting)
    return LayoutDetection(1, None, confidence, len(lines), respecting)


def _group_into_lines(words: list[Word], y_tolerance: float) -> list[list[Word]]:
    """Group words into text lines by vertical position.

    Sorting by (top, x0) first means a simple forward scan comparing
    each word only to the running line's first word is enough --
    there's no need to re-sort or merge clusters afterwards.
    """
    lines: list[list[Word]] = []
    for w in sorted(words, key=lambda w: (w.top, w.x0)):
        if lines and abs(w.top - lines[-1][0].top) <= y_tolerance:
            lines[-1].append(w)
        else:
            lines.append([w])
    return lines


def _find_candidate_gutter(
    words: list[Word],
    page_width: float,
    margin_ratio: float,
    min_gap_ratio: float,
    bins: int = 400,
) -> tuple[float, float] | None:
    """Find the widest empty vertical strip in the middle of the page.

    Returns (gutter_start, gutter_end) in page coordinates, or None if
    no strip wide enough to plausibly be a column gutter exists.
    """
    if not words:
        return None

    bin_width = page_width / bins
    occupied = [False] * bins
    for w in words:
        start_bin = max(0, min(bins - 1, int(w.x0 / bin_width)))
        end_bin = max(0, min(bins - 1, int(w.x1 / bin_width)))
        for b in range(start_bin, end_bin + 1):
            occupied[b] = True

    low = int(bins * margin_ratio)
    high = int(bins * (1 - margin_ratio))

    best_run: tuple[int, int] = (0, -1)  # inclusive empty-bin range
    run_start: int | None = None
    for b in range(low, high):
        if not occupied[b]:
            if run_start is None:
                run_start = b
        elif run_start is not None:
            if (b - 1 - run_start) > (best_run[1] - best_run[0]):
                best_run = (run_start, b - 1)
            run_start = None
    if run_start is not None and (high - 1 - run_start) > (best_run[1] - best_run[0]):
        best_run = (run_start, high - 1)

    if best_run[1] < best_run[0]:
        return None
    run_width = (best_run[1] - best_run[0] + 1) * bin_width
    if run_width < min_gap_ratio * page_width:
        return None

    return best_run[0] * bin_width, (best_run[1] + 1) * bin_width
