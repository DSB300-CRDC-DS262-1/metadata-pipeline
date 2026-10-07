"""Rules over all of a document's statistics: the study's scope, duplicates, links and record types."""

from __future__ import annotations

import re

from .findings import ContextItem, Statistic
from .statistics_prompts import DUPLICATE_PROMPT, MAX_DUPLICATE_PAIRS

def sharpen_scope(study: list[str], items: list[ContextItem]) -> list[str]:
    """Swaps vague whole-study items for the specific ones in the method: the period, the sample, the
    collection method and the region.

    A model asked for the study's period often picks a span of years ("2020-2025") over the
    fieldwork dates printed in the method ("30 October to 17 November 2025"). The period goes
    first because the sample is looked for on the method's page.
    """
    by_id = {i.item_id: i for i in items}
    study = _sharpen_sample(_sharpen_period(study, items, by_id), items, by_id)
    return _sharpen_region(_sharpen_collection(study, items, by_id), items, by_id)


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


def _numbers(statistic: Statistic) -> tuple:
    return (statistic.value, statistic.value_min, statistic.value_max)


def _units_agree(first: Statistic, second: Statistic) -> bool:
    a, b = (first.unit or "").strip().lower(), (second.unit or "").strip().lower()
    return not a or not b or a == b or a in b or b in a


def duplicate_candidates(statistics: list[Statistic]) -> list[tuple[Statistic, Statistic]]:
    """Pairs with the same numbers and compatible units, each later number paired with its earliest match."""
    pairs = []
    for j, later in enumerate(statistics):
        if all(n is None for n in _numbers(later)):
            continue
        earlier = next((s for s in statistics[:j] if _numbers(s) == _numbers(later) and _units_agree(s, later)), None)
        if earlier is not None:
            pairs.append((earlier, later))
    return pairs[:MAX_DUPLICATE_PAIRS]


def duplicate_prompt(pairs: list[tuple[Statistic, Statistic]]) -> str:
    def line(s: Statistic) -> str:
        return f"p{s.page} | {s.variable} | {s.reported_text} | {' '.join(s.quote.split())}"

    listed = "\n".join(f"pair_{k:03d}\n  A: {line(a)}\n  B: {line(b)}" for k, (a, b) in enumerate(pairs, start=1))
    return DUPLICATE_PROMPT.format(pairs=listed)


def pair_numbers(pair_ids: list[str], count: int) -> set[int]:
    """'pair_007' -> 7, ignoring IDs the model made up."""
    found = (re.search(r"(\d+)", i) for i in pair_ids)
    return {int(m.group(1)) for m in found if m and 1 <= int(m.group(1)) <= count}


RANGING = re.compile(
    r"\b(?:rang\w*\s+(?:from\s+)?(?P<low>[^\s,;]+)(?P<middle>[^.;]{0,80}?)\bto|between\s+(?P<low2>[^\s,;]+)\s+and)\s+"
    r"(?P<high>[^\s,;]+)", re.I)


def merge_ranges(statistics: list[Statistic]) -> list[Statistic]:
    """Two numbers a sentence joins as a range ("ranging from 0.0033 for ... to 0.0105 for ...") are one
    statistic with a lowest and highest value, as printed."""
    merged, dropped = [], set()
    for low in statistics:
        if id(low) in dropped:
            continue
        found = RANGING.search(" ".join(low.quote.split()))
        high = None
        first = (found.group("low") or found.group("low2")) if found else None
        if found and low.reported_text.strip() == first and low.value is not None:
            high = next((s for s in statistics if s is not low and s.page == low.page and id(s) not in dropped
                         and s.reported_text.strip() == found.group("high") and s.value is not None), None)
        if high is None:
            merged.append(low)
            continue
        dropped.add(id(high))
        merged.append(low.model_copy(update={
            "reported_text": f"{first} ... to {found.group('high')}",
            "value": None, "value_min": low.value, "value_max": high.value, "value_status": "range",
            "context_item_ids": list(dict.fromkeys(low.context_item_ids + high.context_item_ids)),
        }))
    return [s for s in merged if id(s) not in dropped]


def drop_conversions(statistics: list[Statistic]) -> list[Statistic]:
    """Drops a number printed in brackets straight after another one on its page: the same quantity in
    another unit ("12 bales/ha (2.7 t/ha)", "$2070/t ($470/bale)")."""
    def unit(s: Statistic) -> str:
        return (s.unit or "").strip().lower()

    dropped = set()
    for b in statistics:
        if not unit(b) or "%" in unit(b) + b.reported_text:
            continue  # "(2.6% reduction)", "(38% neutral)" are results, not conversions
        quote = " ".join(b.quote.split())
        inner = re.escape(b.reported_text.strip())
        for a in statistics:
            if a is b or a.page != b.page or id(a) in dropped or not unit(a) or unit(a) == unit(b):
                continue
            if re.search(re.escape(a.reported_text.strip()) + r"\s*\(\s*" + inner, quote):
                dropped.add(id(b))  # by object: the model's IDs can repeat
                break
    return [s for s in statistics if id(s) not in dropped]


def drop_duplicates(statistics: list[Statistic], same: set[int],
                    pairs: list[tuple[Statistic, Statistic]]) -> list[Statistic]:
    """Removes the later number of each pair judged the same measurement; the earlier keeps both sets of links."""
    merged: dict[str, list[str]] = {}
    dropped: set[str] = set()
    for k in sorted(same):
        earlier, later = pairs[k - 1]
        if earlier.statistic_id in dropped:
            continue
        dropped.add(later.statistic_id)
        ids = merged.get(earlier.statistic_id, earlier.context_item_ids)
        merged[earlier.statistic_id] = list(dict.fromkeys(ids + later.context_item_ids))
    return [s.model_copy(update={"context_item_ids": merged[s.statistic_id]}) if s.statistic_id in merged else s
            for s in statistics if s.statistic_id not in dropped]


# When the study's own item of a kind already reaches a statistic, other items of that kind added by
# the linking step are almost always wrong: always for these kinds...
SETTLED_BY_STUDY = {"time_period", "geographic_region"}
# ...and, for these, when they come from another page (a subgroup on the same page can be right).
SETTLED_UNLESS_SAME_PAGE = {"sample_population", "collection_method", "data_source"}


def trim_links(statistics: list[Statistic], items: list[ContextItem], study: list[str]) -> list[Statistic]:
    """Drops linked items that repeat a kind the study's own items already cover for that statistic."""
    by_id = {i.item_id: i for i in items}
    study_set = set(study)
    trimmed = []
    for s in statistics:
        covered = {by_id[i].item_type for i in s.context_item_ids if i in study_set and i in by_id}
        keep = []
        for i in s.context_item_ids:
            item = by_id.get(i)
            if item is None or i in study_set or item.item_type not in covered:
                keep.append(i)
            elif item.item_type in SETTLED_BY_STUDY:
                continue
            elif item.item_type in SETTLED_UNLESS_SAME_PAGE and item.page != s.page:
                continue
            else:
                keep.append(i)
        trimmed.append(s if keep == s.context_item_ids else s.model_copy(update={"context_item_ids": keep}))
    return trimmed


MODELLING_STUDY_SCENARIOS = 2


def modelled_results(statistics: list[Statistic], items: list[ContextItem]) -> list[Statistic]:
    """In a modelling study (scenarios among its context items), a computed value is a model result.

    Models label an LCA's stage emissions or breakdowns "derived"; they are outputs of the study's
    model. Cited figures were already recorded as cited.
    """
    if sum(i.item_type == "scenario" for i in items) < MODELLING_STUDY_SCENARIOS:
        return statistics
    return [s.model_copy(update={"record_type": "scenario_result"}) if s.record_type == "derived" else s
            for s in statistics]


def answers_to_questions(statistics: list[Statistic], items: list[ContextItem]) -> list[Statistic]:
    """A measurement linked to a survey question (an average rating, a share agreeing) is a survey answer."""
    questions = {i.item_id for i in items if i.item_type == "survey_question"}

    def answered(s: Statistic) -> bool:
        # In a survey report, a share of people who agreed, said or recalled something is an answer.
        share = "%" in (s.unit or "") + s.reported_text
        return bool(questions) and share and bool(ANSWER_WORDS.search(s.quote))

    def rating(s: Statistic) -> bool:
        # An average score on the survey's scale ("scored an average of 3.46"); changes are below 1.
        return (bool(questions) and s.value is not None and 1 <= s.value <= 5 and "%" not in (s.unit or "")
                and bool(RATING_WORDS.search(s.quote)))

    return [s.model_copy(update={"record_type": "survey_response"})
            if (s.record_type == "measurement"
                and (questions & set(s.context_item_ids) or RATING_UNIT.search(s.unit or "") or answered(s)))
            or (s.record_type in ("measurement", "derived") and rating(s))
            else s
            for s in statistics]


RATING_WORDS = re.compile(r"\b(?:average|mean|scored?|rating)\b", re.I)


ANSWER_WORDS = re.compile(
    r"\b(?:agreed|agree|disagreed|neutral|neutrally|said|responded|recall|heard|participants|respondents)\b", re.I)


RATING_UNIT = re.compile(r"\b(?:rating|score|scale|mean|likert)\b", re.I)  # a rating measured by a survey
