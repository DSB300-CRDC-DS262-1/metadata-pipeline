"""Rules that read each statistic's value, bounds, type and record type from its printed wording."""

from __future__ import annotations

import re

from .findings import Statistic
from .statistics_periods import CITATION

# Wording that fixes how a value should be read, checked against the model's value_status.
STATUS_WORDS = [
    ("lower_bound", re.compile(r"\b(just over|over|more than|surpass\w*|exceed\w*|at least|greater than)\b|[≥>]", re.I)),
    ("upper_bound", re.compile(r"\b(fewer than|less than|under|within|approaching|approached|up to|at most)\b|[≤<]", re.I)),
    ("approximate", re.compile(r"~|\b(about|around|approximately|roughly|close to|nearly|some)\b", re.I)),
]
RANGE = re.compile(r"(\d[\d,.]*)\s*(?:-|to)\s*\$?(\d[\d,.]*)")  # "$3.88 to $7.76/ac" too
# A year, a span of years or a currency year: a date, not a statistic.
DATE_ONLY = re.compile(
    r"^\(?(?:(?:19|20)\d\d(?:[-/](?:19|20)?\d\d)?(?:\s*(?:-|to)\s*(?:19|20)\d\d(?:[-/](?:19|20)?\d\d)?)?)"
    r"(?:\s+(?:australian\s+)?dollars)?\)?$",
    re.I,
)
YEARS = re.compile(r"(?<![$\d.,])\b(?:19\d\d|20[0-4]\d)(?:[-/]\d\d(?!\d))?\b")

def _number(text: str) -> float:
    return float(text.replace(",", "").rstrip("."))


# Shares written in words, as percentages ("the third largest" is a rank, not a share).
SHARE_WORDS = [
    (re.compile(r"\bthree[- ]quarters\b", re.I), 75.0),
    (re.compile(r"\btwo[- ]thirds\b", re.I), 66.7),
    (re.compile(r"\b(?:a|one)[- ]third\b", re.I), 33.3),
    (re.compile(r"\b(?:a|one)[- ]quarter\b", re.I), 25.0),
    (re.compile(r"\bhalf\b", re.I), 50.0),
]
PER_CENT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*per\s*cent\b", re.I)


SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
SCALED = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(thousand|million|billion)\b", re.I)


def printed_scale(statistic: Statistic) -> Statistic:
    """"$32 million" is 32 with the unit "... million", as printed; models often write 32000000."""
    found = SCALED.search(statistic.reported_text)
    if not found:
        return statistic
    printed, word = _number(found.group(1)), found.group(2).lower()
    update = {}
    for key in ("value", "value_min", "value_max"):
        current = getattr(statistic, key)
        if current is not None and abs(current - printed * SCALES[word]) < 1e-6 * printed * SCALES[word]:
            update[key] = printed
    unit = statistic.unit or ""
    if update or (unit and word not in unit.lower()):
        if word not in unit.lower():
            update["unit"] = f"{word} {unit}".strip() if not re.search(r"\b(?:USD|AUD|AU\$|\$)", unit) else \
                re.sub(r"\b(USD|AUD)\b", rf"\1 {word}", unit, count=1)
    return statistic.model_copy(update=update) if update else statistic


def printed_share(reported: str) -> float | None:
    """A share written in words or as "per cent", as a percentage; None for anything else."""
    per_cent = PER_CENT.search(reported)
    if per_cent:
        return _number(per_cent.group(1))
    if re.search(r"\d", reported):
        return None  # "Close to half ... at 47%": the printed number is the value
    return next((value for pattern, value in SHARE_WORDS if pattern.search(reported)), None)


BOUND_REACH = 25  # characters before the value to look for "surpasses", "approaching", "within"
UNCONFIRMED = re.compile(r"\b(?:possibly|possible|could have|may have|unconfirmed|suspected)\b", re.I)


def tidy_statistic(statistic: Statistic) -> Statistic | None:
    """The reading the wording states, bounds in the right fields, and dates dropped."""
    reported = statistic.reported_text.strip()
    if DATE_ONLY.match(reported):
        return None
    if YEARS.search(reported) and not re.search(r"\d", YEARS.sub("", reported)):
        return None  # "five-year period 2012-13 to 2016-17": its only numbers are years
    statistic = printed_scale(statistic)
    share = printed_share(reported)
    if share is not None:  # "half", "up to a third", "more than 99 per cent": a percentage as printed
        statistic = statistic.model_copy(update={"value": share, "value_min": None, "value_max": None, "unit": "%"})
    update: dict = {}
    status = next((s for s, pattern in STATUS_WORDS if pattern.search(reported)), None)
    if status is None:
        # The model often leaves the wording out of the value ("surpasses $48 million" -> "$48 million").
        quote = " ".join(statistic.quote.split())
        at = quote.find(reported)
        before = quote[max(0, at - BOUND_REACH):at] if at > 0 else ""
        status = next((s for s, pattern in STATUS_WORDS if pattern.search(before) and before.rstrip().endswith(
            tuple(m.group(0) for m in pattern.finditer(before)))), None)
        if status and re.search(r"\bnot\s+\w+\s*$", before):
            status = None  # "does not exceed -130 kPa" is a threshold, not a bound on a result
    if UNCONFIRMED.search(statistic.quote):
        update["confirmed"] = False
    span = RANGE.search(reported)
    if span and status is None and not re.search(r"(19|20)\d\d", reported):
        status = "range"
    status = status or statistic.value_status
    known = statistic.value if statistic.value is not None else (statistic.value_min or statistic.value_max)
    if status == "range" and span:
        update.update(value=None, value_min=_number(span.group(1)), value_max=_number(span.group(2)))
    elif status == "lower_bound":
        update.update(value=None, value_min=statistic.value_min if statistic.value_min is not None else known)
    elif status == "upper_bound":
        update.update(value=None, value_max=statistic.value_max if statistic.value_max is not None else known)
    elif status in ("exact", "approximate") and statistic.value is None and known is not None:
        update.update(value=known)
    update["value_status"] = status
    if "%" in reported:
        # Models sometimes write "80%" as 0.8; the value is the number as printed.
        printed = [_number(n) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", reported)]
        for key in ("value", "value_min", "value_max"):
            current = update.get(key, getattr(statistic, key))
            if current is not None and current not in printed and any(abs(current * 100 - p) < 1e-6 for p in printed):
                update[key] = round(current * 100, 6)
    return statistic.model_copy(update=update)



ANSWER = re.compile(r"\b(disagree|agree|neutral)(?:d|ly|ment)?\b", re.I)  # "in agreement" counts as agree
ANSWER_REACH = 40
CHANGE = re.compile(r"\b(?:grown|grew|increased?|increase of|decreased?|decrease of|rose|risen|fell|fallen|dropped|change of)\b", re.I)


def answer_category(statistic: Statistic) -> str | None:
    """agree, neutral or disagree from the words around the value in its quote."""
    quote, value = statistic.quote, statistic.reported_text.strip()
    found_value = re.search(r"(?<![\d.])" + re.escape(value), quote)
    if found_value is None or "%" not in value:
        return None
    at = found_value.start()
    if CHANGE.search(quote[max(0, at - 25):at]):
        return None  # "has grown 0.9%", "an increase of 6%": a change, not an answer
    # The nearest answer word after the value, then before it, without passing another percentage.
    after = quote[at + len(value):][:ANSWER_REACH].split("%")[0]
    found = ANSWER.search(after)
    if found is None:
        before = quote[:at][-ANSWER_REACH:].split("%")[-1]
        found = list(ANSWER.finditer(before))[-1] if ANSWER.search(before) else None
    if found is None:
        # A long survey statement can sit between the answer word and the value; one answer word in the sentence is enough.
        named = {m.group(1).lower() for m in ANSWER.finditer(quote)}
        return named.pop() if len(named) == 1 else None
    return found.group(1).lower()

TIME_UNIT = re.compile(r"(?:days?|weeks?|months?|years?|hours?|minutes?)")
MONEY = re.compile(r"\$|\bAUD\b|\bdollars?\b", re.I)


# The amount of a change ("an increase of 6%", "increased by 0.09", "grown 0.9%"), not the level a change
# starts from or ends at ("increased from 49%", "dropped to 3.72"): CQE's up/down change categories.
CHANGE_AMOUNT = re.compile(
    r"\b(?:increase[ds]?|decrease[ds]?|grown|grew|rose|risen|fell|fallen|dropped|declined|change[ds]?)"
    r"(?:\s+(?:by|of))?\s*$", re.I)


def label_change(statistic: Statistic) -> Statistic:
    """A value that is the amount of a change is derived from the two levels it compares."""
    if statistic.record_type not in ("measurement", "survey_response"):
        return statistic
    quote = " ".join(statistic.quote.split())
    at = quote.find(statistic.reported_text.strip())
    if at > 0 and CHANGE_AMOUNT.search(quote[max(0, at - 30):at]):
        return statistic.model_copy(update={"record_type": "derived", "response_category": None})
    return statistic


def shared_unit(statistic: Statistic) -> Statistic:
    """A number joined to the next one by "and", "or" or "to" shares its unit: in "24 and 67% below",
    24 is a percentage too (the shared-unit rule of rule-based quantity extractors such as CQE)."""
    if statistic.unit or not re.fullmatch(r"\d[\d,.]*", statistic.reported_text.strip()):
        return statistic
    quote = " ".join(statistic.quote.split())
    shared = re.search(re.escape(statistic.reported_text.strip()) + r"\s+(?:and|or|to|-)\s+\d[\d,.]*\s*(%|per\s*cent\b)", quote)
    return statistic.model_copy(update={"unit": "%"}) if shared else statistic


def label_type(statistic: Statistic) -> Statistic:
    """A percentage is a percentage; money is a monetary value unless it is a median, an average or a ratio."""
    unit, variable = statistic.unit or "", statistic.variable.lower()
    kind = statistic.statistic_type
    if "%" in unit or (not unit and statistic.reported_text.strip().endswith("%")):
        kind = "percentage"
    elif MONEY.search(statistic.reported_text + " " + unit) and not re.search(r"/|\bper\b", unit + statistic.reported_text):
        kind = "median" if "median" in variable else "average" if re.search(r"\b(average|mean)\b", variable) else "monetary_value"
    elif TIME_UNIT.fullmatch(unit.strip().lower()):
        kind = "duration"
    elif unit.strip().lower() in ("rank", "ranking", "position"):
        kind = "other"  # "the sixth largest producer" is a position, not a count
    elif re.search(r"\bmedian\b", variable):
        kind = "median"
    elif re.search(r"\b(average|mean)\b", variable) and not re.search(r"\b(change|increase|decrease|difference|drop|rise)\b", variable):
        kind = "average"  # an average, but not a change in one
    per_unit = bool(re.search(r"/|\bper\b", unit))
    if P_VALUE.match(statistic.reported_text):
        kind = "other"  # a test's p-value is not a share
    elif kind == "rate" and MONEY.search(statistic.reported_text + " " + unit):
        kind = "monetary_value"  # "$3.88 to $7.76/ac" is money per acre
    elif kind == "rate" and not per_unit:
        kind = "other"  # a rate is per something; "250,000 acres" or "333 GDDs" is a quantity
    elif kind == "count" and PHYSICAL_UNIT.fullmatch(unit.strip().lower()):
        kind = "other"  # acres or hectares are measured, not counted
    return statistic if kind == statistic.statistic_type else statistic.model_copy(update={"statistic_type": kind})


P_VALUE = re.compile(r"^\s*p\s*[<>≥≤=]", re.I)
PHYSICAL_UNIT = re.compile(
    r"(?:million |thousand )?(?:acres?|ac|ha|hectares?|ac-in|in|inch(?:es)?|ft|m|km|mm|cm|kg|g|t|tonnes?|kpa|"
    r"gdds?|dd60|l|ml|gl|kwh|mj)", re.I)



def label_cited(statistic: Statistic) -> Statistic:
    """A measurement whose sentence cites another work ("(Roth, 2014)") is reported from that work."""
    if statistic.record_type in ("measurement", "derived") and CITATION.search(statistic.quote):
        return statistic.model_copy(update={"record_type": "cited"})
    return statistic


# A numbered answer option with its share: "2. Television - news (21%)."
LIST_OPTION = re.compile(r"^\W*\d+\.\s*(?P<label>[^()]+?)\s*\(\s*\d[\d.]*\s*%\s*\)")


def option_label(quote: str) -> str | None:
    found = LIST_OPTION.match(" ".join(quote.split()))
    if not found:
        return None
    label = re.split(r",\s*e\.g\.", found.group("label"))[0]
    return re.sub(r"\s*-\s*", " ", label).strip(" .,").lower() or None


def label_survey_answer(statistic: Statistic, page_text: str = "") -> Statistic:
    if statistic.record_type == "derived":
        return statistic.model_copy(update={"response_category": None})  # a change in a share answers nothing
    category = statistic.response_category or option_label(statistic.quote) or answer_category(statistic)
    if category is None and page_text:
        # The model's quote often stops at the value; the answer word can be earlier in the sentence.
        sentence = full_sentence(statistic.quote, page_text)
        if sentence != statistic.quote:
            category = answer_category(statistic.model_copy(update={"quote": sentence}))
    if category is None:
        return statistic
    record_type = "survey_response" if statistic.record_type == "measurement" else statistic.record_type
    return statistic.model_copy(update={"response_category": category, "record_type": record_type})


def full_sentence(quote: str, page_text: str) -> str:
    """The whole sentence (or sentences) of the page that contain the quote; the quote if not found."""
    text = " ".join(page_text.split())
    part = " ".join(quote.split())
    at = text.find(part)
    if at < 0 or not part:
        return quote
    start = max(text.rfind(". ", 0, at), text.rfind("? ", 0, at))
    start = start + 2 if start >= 0 else 0
    end = text.find(". ", at + len(part))
    return text[start:end + 1 if end >= 0 else len(text)]
