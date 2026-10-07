"""Rules for the study's own conditions: dropping labels, and swapping vague whole-study items for
the specific ones stated in the method."""

from __future__ import annotations

import re

from .schema import STATISTIC_LEVEL, ContextItem


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


# Asides that are not about how the data were produced: thanks, disclaimers, copyright.
ASIDE = re.compile(r"\b(?:our thanks|we thank|thanks are extended|acknowledge?ments?|disclaimer|the reader is advised|"
                   r"general statements based on|accept(?:s)? no (?:liability|responsibility)|copyright)\b|©", re.I)
# A value that is only a generic word for the survey or its people ("Survey", "Australians").
GENERIC_WORDS = {"the", "survey", "surveys", "participants", "respondents", "australians", "community", "members",
                 "responses", "response", "people", "data"}
RUNNING_LINE_PAGES = 2
RUNNING_LINE_MIN_WORDS = 3


def _line_key(text: str) -> str:
    return re.sub(r"\s*\d{1,3}$", "", " ".join(text.split())).lower()  # a trailing page number differs per page


def running_lines(pages: list[str]) -> set[str]:
    """Lines printed on several pages: running headers and footers such as the report's title."""
    seen: dict[str, set[int]] = {}
    for n, text in enumerate(pages):
        for line in text.splitlines():
            key = _line_key(line)
            if len(key.split()) >= RUNNING_LINE_MIN_WORDS:
                seen.setdefault(key, set()).add(n)
    return {key for key, found in seen.items() if len(found) >= RUNNING_LINE_PAGES}


def not_a_condition(item: ContextItem, running: set[str]) -> bool:
    """An item that only repeats a running header or footer, an aside, or a generic word."""
    if item.item_type in STATISTIC_LEVEL:
        return False
    if _line_key(item.quote) in running or _line_key(item.value) in running:
        return True
    if ASIDE.search(item.quote):
        return True
    words = re.findall(r"[a-z]+", item.value.lower())
    return bool(words) and len(words) <= 3 and set(words) <= GENERIC_WORDS


def sharpen_scope(study: list[str], items: list[ContextItem]) -> list[str]:
    """Swaps vague whole-study items for the specific ones in the method: the period, the sample, the
    collection method and the region; then fills any key kind still missing from the method's page.

    A model asked for the study's period often picks a span of years ("2020-2025") over the
    fieldwork dates printed in the method ("30 October to 17 November 2025"). The period goes
    first because the sample is looked for on the method's page.
    """
    by_id = {i.item_id: i for i in items}
    study = _sharpen_sample(_sharpen_period(study, items, by_id), items, by_id)
    study = _sharpen_region(_sharpen_collection(study, items, by_id), items, by_id)
    return _fill_missing_kinds(study, items, by_id)


# The kinds every study states for its own data; the scope step sometimes leaves one out.
KEY_KINDS = ("time_period", "geographic_region", "sample_population", "data_source", "collection_method")


def _fill_missing_kinds(study: list[str], items: list[ContextItem], by_id: dict[str, ContextItem]) -> list[str]:
    """A key kind the scope step left out is taken from the method's page, if the page states one.

    The method's page is the page of the study's chosen period, sample or collection method (e.g. a survey's
    data source, "an online research panel", printed beside its fieldwork dates).
    """
    pages = {by_id[i].page for i in study if by_id[i].item_type in ("time_period", "sample_population", "collection_method")}
    chosen = {by_id[i].item_type for i in study}
    added = [next(i.item_id for i in items if i.item_type == kind and i.page in pages)
             for kind in KEY_KINDS if kind not in chosen and any(i.item_type == kind and i.page in pages for i in items)]
    return study + added


def _method_pages(study: list[str], by_id: dict[str, ContextItem]) -> set[int]:
    return {by_id[i].page for i in study if by_id[i].item_type in ("time_period", "collection_method", "data_source")}


def _sharpen_region(study: list[str], items: list[ContextItem], by_id: dict[str, ContextItem]) -> list[str]:
    """A whole-study region from outside the method ("mid-southern United States" in the abstract)
    gives way to the region stated on the method's page ("Starkville, MS and Stoneville, MS")."""
    chosen = [i for i in study if by_id[i].item_type == "geographic_region"]
    pages = _method_pages(study, by_id)
    if not chosen or any(by_id[i].page in pages for i in chosen):
        return study
    local = [i for i in items if i.item_type == "geographic_region" and i.page in pages]
    if not local:
        return study
    return [local[0].item_id if i == chosen[0] else i for i in study if i not in chosen[1:]]


def _sharpen_collection(study: list[str], items: list[ContextItem], by_id: dict[str, ContextItem]) -> list[str]:
    """A collection method from outside the method ("Survey responses" under one chart) gives way to the
    one stated on the page of the study's period, data source or sample ("online survey methodology")."""
    chosen = [i for i in study if by_id[i].item_type == "collection_method"]
    pages = {by_id[i].page for i in study if by_id[i].item_type in ("time_period", "data_source", "sample_population")}
    if not chosen or any(by_id[i].page in pages for i in chosen):
        return study
    local = [i for i in items if i.item_type == "collection_method" and i.page in pages]
    if not local:
        return study
    return [local[0].item_id if i == chosen[0] else i for i in study if i not in chosen[1:]]


def _sharpen_period(study: list[str], items: list[ContextItem], by_id: dict[str, ContextItem]) -> list[str]:
    chosen = [i for i in study if by_id[i].item_type == "time_period"]
    if not chosen:
        return study
    best = max(_precision(by_id[i]) for i in chosen)
    # Fieldwork dates fall within one year; a dated span over several years is usually a sub-period.
    sharper = [i for i in items if i.item_type == "time_period" and _precision(i) > max(best, len("2025"))
               and i.period.start[:4] == i.period.end[:4]]
    if not sharper:
        return study
    pick = max(sharper, key=_precision).item_id
    return [pick if i == chosen[0] else i for i in study if i not in chosen[1:]]


def _sharpen_sample(study: list[str], items: list[ContextItem], by_id: dict[str, ContextItem]) -> list[str]:
    """A whole-study sample with no size ("Australians") gives way to the sized sample in the method.

    The method is the page of the study's chosen period, collection method or data source.
    """
    chosen = [i for i in study if by_id[i].item_type == "sample_population"]
    if any(re.search(r"\d", by_id[i].value) for i in chosen):
        return study
    method_pages = {by_id[i].page for i in study if by_id[i].item_type in ("time_period", "collection_method", "data_source")}
    sized = [i for i in items if i.item_type == "sample_population" and i.page in method_pages and re.search(r"\d", i.value)]
    if not sized:
        return study
    rest = [i for i in study if i not in chosen]
    return rest + [sized[0].item_id]


def _precision(item: ContextItem) -> int:
    period = item.period
    return len(period.start) if period and period.start and period.end else 0
