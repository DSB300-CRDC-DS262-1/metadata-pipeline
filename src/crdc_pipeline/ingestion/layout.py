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
) -> LayoutDetection:
    """Classify a page as single- or two-column from its word positions."""
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
    gutter = _find_candidate_gutter(body_words, page_width, margin_ratio, min_gap_ratio)
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
