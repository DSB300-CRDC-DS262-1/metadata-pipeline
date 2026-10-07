"""Text and numbers to leave out: reference lists, table rows, chart labels and numbers that only
describe how a report or study is set up."""

from __future__ import annotations

import re

from .findings import STATISTIC_LEVEL, ContextItem, Statistic
from .statistics_values import YEARS, printed_share

# Lines that are mostly numbers: table rows and chart labels.
TABLE_TOKEN = re.compile(r"[\d,.()%$#p*-]+")

def without_tables(text: str) -> str:
    """Drops lines that are mostly numbers (table rows, chart labels), keeping running text."""
    return "\n".join(line for line in text.splitlines() if not _mostly_numbers(line))


def _mostly_numbers(line: str) -> bool:
    tokens = line.split()
    numeric = sum(bool(TABLE_TOKEN.fullmatch(t)) for t in tokens)
    words = sum(sum(c.isalpha() for c in t) >= 3 for t in tokens)
    return numeric >= 3 and numeric > 2 * words


def before_references(pages: list[str]) -> list[str]:
    """Pages with the reference list and everything after it removed; cited works are not findings."""
    kept: list[str] = []
    for text in pages:
        lines = text.splitlines()
        cut = next((i for i, line in enumerate(lines) if line.strip().lower() in ("references", "reference list")), None)
        if cut is not None:
            kept.append("\n".join(lines[:cut]))
            return kept + [""] * (len(pages) - len(kept))
        kept.append(text)
    return kept


LABEL_MAX_WORDS = 8
MIN_QUOTE_WORDS = 4


def is_label(item: ContextItem, page_text: str) -> bool:
    """A quote that is a short line of its own with no sentence ending: a chart legend or heading."""
    quote = " ".join(item.quote.split())
    if item.item_type in STATISTIC_LEVEL:
        return False
    if len(quote.split()) < MIN_QUOTE_WORDS:
        return True  # "Australia's cotton industry": a phrase, not a statement of the fact
    if len(quote.split()) >= LABEL_MAX_WORDS or quote.endswith((".", "?")):
        return False
    return any(" ".join(line.split()) == quote for line in page_text.splitlines())


# Numbers that define a subgroup ("those who knew 6-10 people"), not results about it.
SUBGROUP = re.compile(r"\b(?:who|that)\s+(?:knew|know|had|have|were|are|aged)\s+(?:about\s+)?$", re.I)
# The ends of a rating scale ("from 1 (Not at all) to 5 (Extremely)").
SCALE = re.compile(r"\bscale\s+(?:of|from)\s+(\d+)\b[^.;]{0,40}?\bto\s+(\d+)\b", re.I)
# A count of the report's own parts ("three main themes", "two core items").
OWN_PARTS = re.compile(
    r"\b(?:themes?|items?|measures?|groupings?|groups?|variables?|areas?|categories|intervals?|questions?|"
    r"topics?|sections?|chapters?|tests?|parts?|steps?|phases?|options?|strategies|scenarios?|"
    r"replications?|replicates?|rows?|plots?|blocks?)\b", re.I)
# A number defined by an equals sign ("NAWF = 5", "1 = Strongly disagree").
DEFINED = re.compile(r"=\s*$")
# A hyphenated dimension of equipment or plots ("38-in rows", "9-mil tubing").
SPEC = re.compile(r"^\d+(?:\.\d+)?-(?:in|inch|mil|ft|cm|mm)$", re.I)
# A count used as an adjective ("25-boll", "10-saw"); time spans such as "6-month" are durations.
COUNT_ADJECTIVE = re.compile(r"^\d+-(?!(?:month|year|week|day|hour|minute)s?$)[a-z]+$", re.I)
TIMING = re.compile(r"^(?:\d+|one|two|three|four|five|six|seven|eight)\s+(?:weeks?|days?|months?)$", re.I)
SHARE_RESTATED_WITHIN = 10  # percentage points between "close to half" and the exact figure after it
SMALL_WORD_NUMBER = re.compile(r"^(?:one|two|three|four|five|six|seven|eight|nine|ten)$", re.I)


def is_structural(statistic: Statistic) -> bool:
    """Numbers that describe how the report or survey is set up rather than a finding."""
    reported = statistic.reported_text.strip()
    quote = " ".join(statistic.quote.split())
    at = quote.find(reported)
    if at >= 0 and SUBGROUP.search(quote[:at]):
        return True
    scale = SCALE.search(quote)
    if scale and reported in scale.groups():
        return True
    if CAPTION.match(quote):
        return True  # a value read off a chart, quoted from its caption
    if SPEC.match(reported) or COUNT_ADJECTIVE.match(reported):
        return True  # "38-in rows", "a 25-boll sample", "a 10-saw gin": how the trial was set up
    if at >= 0 and TIMING.match(reported) and re.match(r"\s*(?:prior to|before|after|past)\b", quote[at + len(reported):]):
        return True  # "one week prior to first cracked boll": when a treatment was applied
    if at >= 0 and (DEFINED.search(quote[:at]) or re.match(r"\s*=", quote[at + len(reported):])):
        return True
    share = printed_share(reported) if not re.search(r"\d", reported) else None  # "... at 51%" is the exact one
    if share is not None and at >= 0:
        exact = [float(p) for p in re.findall(r"(\d+(?:\.\d+)?)\s*%", quote[at:])]
        if any(abs(p - share) <= SHARE_RESTATED_WITHIN for p in exact):
            return True  # "Close to half ... agreed ... at 47%": the exact figure is the statistic
    numbers = [n for n in re.findall(r"\d[\d.,]*", YEARS.sub("", quote))]
    tokens = quote.split()
    numeric = sum(bool(re.fullmatch(r"[\d.,%]+", t)) for t in tokens)
    if not re.search(r"[A-Za-z]{3,}", quote) and len(numbers) >= 2 and 2 * numeric >= len(tokens):
        return True  # "n 3.54 3.51": a chart's labels, no words ("53% in 2023" is kept)
    if SMALL_WORD_NUMBER.match(reported) and at >= 0:
        following = re.findall(r"[A-Za-z]+", quote[at + len(reported):at + len(reported) + 40])[:3]
        if any(OWN_PARTS.fullmatch(word) for word in following):
            return True
    return False


TABLE_ROW_MAX_WORDS = 15


def is_table_row(statistic: Statistic, page_text: str) -> bool:
    """A quote that is a whole short line ending in the value, with no sentence ending: a table row."""
    quote = " ".join(statistic.quote.split())
    if _mostly_numbers(quote):
        return True
    if len(quote.split()) >= TABLE_ROW_MAX_WORDS or quote.endswith((".", "?", "!")):
        return False
    if not quote.endswith(statistic.reported_text.strip()):
        return False
    return any(" ".join(line.split()) == quote for line in page_text.splitlines())


WORD_NUMBER = re.compile(
    r"\b(?:(?:up to|over|more than|almost|nearly|close to|about|around|just over|fewer than|less than)\s+)?"
    r"(?:(?:a|one)\s+(?:third|quarter|half)|half|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:-\w+)?|"
    r"(?:one|two|three)-(?:thirds?|quarters?)|\d[\d,.]*\s+per\s+cent)\b",
    re.I,
)


def word_number_note(text: str) -> str:
    """Numbers written in words on the page, which small models tend to skip, listed as a reminder."""
    prose = "\n".join(line for line in text.splitlines() if not _mostly_numbers(line))
    found = list(dict.fromkeys(m.group(0).lower() for m in WORD_NUMBER.finditer(prose)))
    if not found:
        return ""
    return ("\nNumbers written in words on this page (list each one that states a statistic, skipping "
            "the kinds above): " + "; ".join(found) + "\n")


CAPTION = re.compile(r"^\W*(figure|fig\.|table)\s*\d", re.I)
