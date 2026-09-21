from __future__ import annotations

from datetime import datetime

DATE_FIELDS = {"publication_date", "submission_date", "acceptance_date"}
DATE_FORMATS = [
    ("%Y-%m-%d", 3),
    ("%B %d, %Y", 3),
    ("%d %B %Y", 3),
    ("%Y-%m", 2),
    ("%B %Y", 2),
    ("%Y", 1),
]


def parse_date(value: str) -> tuple[tuple[int, ...], int] | None:
    """Return (year, month, day) truncated to the precision the text states, and that precision."""
    text = value.strip()
    for fmt, precision in DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return (parsed.year, parsed.month, parsed.day)[:precision], precision
    return None


def strict_equal(a, b) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return {str(x).strip().lower() for x in a} == {str(x).strip().lower() for x in b}
    if a is None or b is None:
        return a == b
    return str(a).strip().lower() == str(b).strip().lower()


def dates_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return a == b
    left, right = parse_date(a), parse_date(b)
    if left is None or right is None:
        return strict_equal(a, b)
    precision = min(left[1], right[1])
    return left[0][:precision] == right[0][:precision]


def field_matches(field: str, extracted, expected, normalise_dates: bool = False) -> bool:
    if normalise_dates and field in DATE_FIELDS:
        return dates_equal(extracted, expected)
    return strict_equal(extracted, expected)
