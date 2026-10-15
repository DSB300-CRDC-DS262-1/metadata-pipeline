"""Periods read from the printed text, at the precision printed."""

from __future__ import annotations

import re

from ..contexts.schema import ContextItem, Period
from .schema import Statistic


MONTHS = {m[:3].lower(): f"{i:02d}" for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
     "November", "December"], start=1)}


_MONTH = r"(?:(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?"


_YEAR = r"((?:19|20)\d\d)"


# Longest forms first: "2012-13 to 2016-17", "Oct 2019 - Dec 2023", "2001-13", "since 2001", "by 2014", "in 2020".
_DAY_MONTH = r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"


PERIODS = [
    # "30 October to 17 November 2025", "between 30 October and 17 November 2025"
    ("days", re.compile(_DAY_MONTH + r"(?:\s+" + _YEAR + r")?\s*(?:-|to|and)\s*" + _DAY_MONTH + r"\s+" + _YEAR)),
    ("span", re.compile(_YEAR + r"[-/]\d\d\s*(?:-|to)\s*(?:19|20)\d\d[-/](\d\d)\b")),
    ("span", re.compile(r"\b" + _MONTH + _YEAR + r"\s*(?:-|to)\s*" + _MONTH + r"((?:19|20)?\d\d)\b(?![-/]?\d)")),
    ("between", re.compile(r"\bbetween\s+" + _MONTH + _YEAR + r"\s+and\s+" + _MONTH + _YEAR + r"\b", re.I)),
    ("since", re.compile(r"\bsince\s+" + _MONTH + _YEAR, re.I)),
    ("by", re.compile(r"\bby\s+" + _MONTH + _YEAR, re.I)),
    # "in 2023 Australian dollars" is a currency year, not a period.
    ("year", re.compile(r"(?:\bin|\bduring|\bof|\bfor|\()\s*(?:the\s+)?" + _MONTH + _YEAR
                        + r"\b(?![-/]\d)(?!\s+(?:Australian\s+|AU\s*)?dollars)", re.I)),
]


def _date(month: str | None, year: str) -> str:
    return f"{year}-{MONTHS[month[:3].lower()]}" if month else year


def period_from_quote(quote: str, reported_text: str) -> Period | None:
    """The printed period nearest the value in its sentence, at the precision printed; citation years ignored."""
    text = NARRATIVE_CITATION.sub(lambda m: " " * len(m.group(0)), CITATION.sub(lambda m: " " * len(m.group(0)), quote))
    at = text.find(reported_text.strip())
    at = at if at >= 0 else 0
    found: list[tuple[int, Period]] = []
    taken: list[tuple[int, int]] = []
    for kind, pattern in PERIODS:
        for m in pattern.finditer(text):
            if any(s < m.end() and m.start() < e for s, e in taken):
                continue
            if kind == "days":
                day1, month1, year1, day2, month2, year2 = m.groups()
                start = f"{year1 or year2}-{MONTHS[month1[:3].lower()]}-{int(day1):02d}"
                end = f"{year2}-{MONTHS[month2[:3].lower()]}-{int(day2):02d}"
            elif kind == "span" and m.re is PERIODS[1][1]:
                start, end = m.group(1), m.group(1)[:2] + m.group(2)
            elif kind == "span":
                short_end = len(m.group(4)) == 2
                if short_end and (re.search(r"\bto\b", m.group(0)) or re.match(r"\s*%", text[m.end():])):
                    continue  # "in 2023 to 51% in 2025" is not the span 2023-2051; "2001-13" is
                start = _date(m.group(1), m.group(2))
                year = m.group(4) if not short_end else m.group(2)[:2] + m.group(4)
                end = _date(m.group(3), year)
            elif kind == "between":
                start, end = _date(m.group(1), m.group(2)), _date(m.group(3), m.group(4))
            elif kind == "since":
                start, end = _date(m.group(1), m.group(2)), None
            elif kind == "by":
                start, end = None, _date(m.group(1), m.group(2))
            else:
                start = end = _date(m.group(1), m.group(2))
            if start and end and end < start:
                continue
            taken.append(m.span())  # only a period that was read holds its text against shorter forms
            # A period with another number between it and the value belongs to that number.
            between = text[min(m.end(), at + len(reported_text)):max(m.start(), at)]
            # A period opening the sentence, before any other number, covers every value in it.
            opens_sentence = not NUMBER.search(FIGURE_REF.sub("", text[:m.start()]))
            if NUMBER.search(FIGURE_REF.sub("", between)) and not opens_sentence:
                continue
            gap = max(0, m.start() - (at + len(reported_text))) + max(0, at - m.end())
            found.append((gap, Period(start=start, end=end)))
    return min(found, key=lambda f: f[0])[1] if found else None


ISO_DATE = re.compile(r"^(?:19|20)\d\d(?:-\d\d(?:-\d\d)?)?$")


def with_period(statistic: Statistic) -> Statistic:
    """Fills an empty period from the sentence; models rarely fill it themselves."""
    period = statistic.period
    if period is not None and all(d is None or ISO_DATE.match(d) for d in (period.start, period.end))             and (period.start or period.end):
        return statistic
    # Empty, or words where a date belongs ("last week"): read it from the sentence instead.
    return statistic.model_copy(update={"period": period_from_quote(statistic.quote, statistic.reported_text)})


ISO = r"((?:19|20)\d\d(?:-(?:0[1-9]|1[0-2])(?!\d)(?:-(?:0[1-9]|[12]\d|3[01])(?!\d))?)?)"


ISO_PERIOD = re.compile(r"^\s*" + ISO + r"(?:\s*(?:to|-|/)\s*" + ISO + r")?\s*$")


def with_item_period(item: ContextItem) -> ContextItem:
    """Fills an empty time_period item's period from its value ("2025-10-30 to 2025-11-17", "2014-19")."""
    if item.item_type != "time_period" or (item.period and (item.period.start or item.period.end)):
        return item
    iso = ISO_PERIOD.match(item.value)
    if iso:
        period = Period(start=iso.group(1), end=iso.group(2) or iso.group(1))
    else:
        period = _spanning(item.value) or _spanning(item.quote)
    return item.model_copy(update={"period": period}) if period else item


def _spanning(text: str) -> Period | None:
    """One period covering every period printed in the text ("2011-2012, 2012-2013, and 2013-2014")."""
    # Each year with the few words before it ("since", "in", a day and month).
    found = [period_from_quote(text[max(0, m.start() - 30):], m.group(0)) for m in re.finditer(r"(?:19|20)\d\d", text)]
    found = [p for p in found if p is not None]
    if not found:
        return None
    starts = [p.start for p in found if p.start]
    ends = [p.end for p in found if p.end]
    return Period(start=min(starts) if starts else None, end=max(ends) if ends else None)


def survey_year(items: list[ContextItem], study: list[str]) -> Period | None:
    """The year of the study's own fieldwork: its most precisely dated whole-study period within one year.

    Without one, a period dated to the month or day anywhere in the document (fieldwork dates are
    usually the only such period).
    """
    single_year = [
        i for i in items
        if i.item_type == "time_period" and i.period and i.period.start and i.period.end
        and i.period.start[:4] == i.period.end[:4]
    ]
    periods = [i.period for i in single_year if i.item_id in study]
    periods = periods or [i.period for i in single_year if len(i.period.start) >= len("2025-10")]
    if not periods:
        return None
    precision = max(len(p.start) for p in periods)
    years = {p.start[:4] for p in periods if len(p.start) == precision}
    return Period(start=years.pop(), end=None) if len(years) == 1 else None


def date_survey_answers(statistics: list[Statistic], year: Period | None) -> list[Statistic]:
    """Survey answers with no period of their own were given in the survey's year."""
    if year is None:
        return statistics
    dated = Period(start=year.start, end=year.start)
    return [s.model_copy(update={"period": dated}) if s.record_type == "survey_response" and s.period is None else s
            for s in statistics]


NUMBER = re.compile(r"\d")


FIGURE_REF = re.compile(r"\b(?:Figure|Fig\.|Table)\s*\d+[a-z]?", re.I)


CITATION = re.compile(r"\((?:[A-Z][A-Za-z&.' -]+?)(?: et al\.?)?, (?:19|20)\d\d[a-z]?(?:[;,][^)]*)?\)")


# "Tian et al. (2015)", "Welke and Smith (2011)": a citation written into the sentence.
NARRATIVE_CITATION = re.compile(r"\b[A-Z][A-Za-z'-]+(?:\s+et\s+al\.?|\s+and\s+[A-Z][A-Za-z'-]+)?\s*\((?:19|20)\d\d[a-z]?\)")
