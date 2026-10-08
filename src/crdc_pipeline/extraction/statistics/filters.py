"""Text and numbers to leave out: reference lists, table rows, chart labels and numbers that only
describe how a report or study is set up."""

from __future__ import annotations

import re

from .schema import Statistic
from .values import YEARS, printed_share


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


# Numbers that define a subgroup ("those who knew 6-10 people"), not results about it.
SUBGROUP = re.compile(r"\b(?:who|that)\s+(?:knew|know|had|have|were|are|aged)\s+(?:about\s+)?(?:\d+\s*[-–]\s*)?$", re.I)
# A group's bound in brackets after its name: "Serious injury claims (5+ days)".
GROUP_BOUND = re.compile(r"^\d+(?:\+|\s*[-–]\s*\d+)\s*(?:days?|weeks?|months?|years?)$", re.I)
# A number in a rule the report applies: "if case numbers are below five", "less than 1% were excluded",
# "below the 1% cut-off", "removal of surveys where ... less than 5 minutes".
CONDITION = re.compile(r"\b(?:if|where|unless)\b[^.;]{0,90}$", re.I)
RULE_WORDS = re.compile(r"\b(?:excluded|exclusion|removal|removed|required|cut-off|threshold)\b", re.I)
RULE_AFTER = re.compile(r"^[^.;]{0,40}?\b(?:cut-off|were excluded|threshold)\b", re.I)
# The length of a period or reporting interval: "the five-year period", "five-year reporting intervals",
# "the last two years of this study period", "only three years in duration".
PERIOD_LENGTH = re.compile(
    r"^\S+?[- ](?:years?|months?)\b(?:\s+(?:of|in)\s+(?:this|the)\s+(?:study\s+)?period|\s+in duration|"
    r"\s*(?:reporting\s+)?(?:period|intervals?)\b)", re.I)
# The functional unit an LCA's results are expressed per ("a functional unit of one tonne").
FUNCTIONAL_UNIT = re.compile(r"\bfunctional unit\s+(?:was|is|of)\s+(?:a|an|the)?\s*$", re.I)


# The ends of a rating scale ("from 1 (Not at all) to 5 (Extremely)").
SCALE = re.compile(r"\bscale\s+(?:ranging\s+)?(?:of|from)\s+(\d+)\b[^.;]{0,40}?\bto\s+(\d+)\b", re.I)


# A count of the report's own parts ("three main themes", "two core items").
OWN_PARTS = re.compile(
    r"\b(?:themes?|items?|measures?|groupings?|groups?|variables?|areas?|categories|intervals?|questions?|"
    r"topics?|sections?|chapters?|tests?|parts?|steps?|phases?|options?|strategies|scenarios?|"
    r"replications?|replicates?|rows?|plots?|blocks?|seasons?|crops?|inputs?|findings?)\b", re.I)
OWN_PARTS_REACH = 5  # words after the number to look for what it counts


# A number defined by an equals sign ("NAWF = 5", "1 = Strongly disagree").
DEFINED = re.compile(r"=\s*$")


# A hyphenated dimension of equipment or plots ("38-in rows", "9-mil tubing").
SPEC = re.compile(r"^\d+(?:\.\d+)?-(?:in|inch|mil|ft|cm|mm)$", re.I)


# A count used as an adjective ("25-boll", "10-saw"); time spans such as "6-month" are durations.
COUNT_ADJECTIVE = re.compile(r"^\d+-(?!(?:month|year|week|day|hour|minute)s?$)[a-z]+$", re.I)


TIMING = re.compile(r"^(?:\d+|one|two|three|four|five|six|seven|eight)\s+(?:weeks?|days?|months?)$", re.I)


SHARE_RESTATED_WITHIN = 10  # percentage points between "close to half" and the exact figure after it


SMALL_WORD_NUMBER = re.compile(r"^(?:single|one|two|three|four|five|six|seven|eight|nine|ten)$", re.I)


NUMBER_WORDS = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|half|quarter|"
                "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth")
# What may follow a number word's hyphen and still leave a number: "two-thirds", "five-fold", "twenty-five".
STILL_A_NUMBER = re.compile(r"^(?:thirds?|quarters?|halves|fold|" + NUMBER_WORDS + r")$", re.I)


def _only_as_adjective(reported: str, quote: str) -> bool:
    """Every number word of the value appears in the quote only inside a hyphenated word ("third-party",
    "five-year"), so the model read a compound as a number."""
    words = re.findall(r"\b(?:" + NUMBER_WORDS + r")\b", reported, re.I)
    if not words:
        return False
    for word in words:
        uses = list(re.finditer(r"\b" + word + r"\b(-[A-Za-z]+)?", quote, re.I))
        if not uses or any(not m.group(1) or STILL_A_NUMBER.match(m.group(1)[1:]) for m in uses):
            return False
    return True


def is_structural(statistic: Statistic) -> bool:
    """Numbers that describe how the report or survey is set up rather than a finding."""
    reported = statistic.reported_text.strip()
    quote = " ".join(statistic.quote.split())
    at = quote.find(reported)
    if at >= 0 and SUBGROUP.search(quote[:at]):
        return True
    if at >= 0 and GROUP_BOUND.match(reported) and quote[at - 1:at] == "(":
        return True
    after = quote[at + len(reported):] if at >= 0 else ""
    if at >= 0 and ((CONDITION.search(quote[:at]) and RULE_WORDS.search(quote)) or RULE_AFTER.match(after)):
        return True  # a threshold in a rule, not a finding
    if at >= 0 and (PERIOD_LENGTH.match(reported + after) or FUNCTIONAL_UNIT.search(quote[:at])):
        return True
    scale = SCALE.search(quote)
    if scale and (reported in scale.groups() or reported == "{} to {}".format(*scale.groups())):
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
    if not re.search(r"\d", reported) and _only_as_adjective(reported, quote):
        return True  # "third-party", "a five-year average": a number word inside a compound word
    first = reported.split()[0] if reported.split() else ""
    if SMALL_WORD_NUMBER.match(first) and at >= 0:
        following = re.findall(r"[A-Za-z]+", quote[at + len(first):at + len(first) + 60])[:OWN_PARTS_REACH]
        if any(OWN_PARTS.fullmatch(word) for word in following):
            return True
    return False


TABLE_ROW_MAX_WORDS = 15
CHART_LABEL_MAX_WORDS = 12
NUMBER_RUN = 3  # numbers in a row with no words between: table cells or chart labels ("35% 21% 37% 7%")


def _in_number_run(quote: str, reported: str) -> bool:
    """The value sits in a run of numbers with no words between them: a table row or chart labels
    ("35% 21% 37% 7%"). A sentence separates its numbers with words; a run elsewhere in the quote (a
    journal header pasted into the sentence) says nothing about this value."""
    value = reported.split()[0] if reported.split() else ""
    runs, run = [], []
    for token in quote.split():
        if TABLE_TOKEN.fullmatch(token) and re.search(r"\d", token):
            run.append(token)
        else:
            runs.append(run)
            run = []
    runs.append(run)
    return bool(value) and any(len(r) >= NUMBER_RUN and any(value in t for t in r) for r in runs)


def is_table_row(statistic: Statistic, page_text: str) -> bool:
    """A quote that is a whole short line ending in the value, with no sentence ending: a table row."""
    quote = " ".join(statistic.quote.split())
    if _mostly_numbers(quote) or _in_number_run(quote, statistic.reported_text):
        return True
    if len(quote.split()) >= TABLE_ROW_MAX_WORDS or quote.endswith((".", "?", "!")):
        return False
    if (len(quote.split()) <= CHART_LABEL_MAX_WORDS and quote.endswith(statistic.reported_text.strip())
            and not re.match(r"^\W*\d+\.\s", quote)):
        return True  # "Prefer not to say: 0.16%": a chart or infographic label, not a sentence or numbered list item
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


def value_not_in_quote(statistic: Statistic) -> bool:
    """The value's digits do not appear in its own quote: the model paired the number with the wrong text,
    which happens on pages whose columns or charts are read out of order. Values in words ("eight") are kept."""
    digits = re.findall(r"\d+(?:\.\d+)?", statistic.reported_text.replace(",", ""))
    quote = statistic.quote.replace(",", "")
    return bool(digits) and not all(d in quote for d in digits)
