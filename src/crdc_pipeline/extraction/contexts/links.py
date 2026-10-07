"""Links between statistics and context items: trimming repeats and the whole-document linking pass."""

from __future__ import annotations

import re

from ..backends import Backend
from ..statistics.records import MODELLING_STUDY_SCENARIOS
from ..statistics.schema import Statistic
from .prompts import DOCUMENT_LINK_PROMPT, DocumentLinks
from .schema import Context, ContextItem, group_contexts


# A model input is read with the study's period, region and data source, not its method or limitations.
INPUT_KINDS = {"time_period", "geographic_region", "data_source"}
RATING_SCALE = re.compile(r"\bscale\b|\b1\s*=|\bfrom\s+1\b", re.I)
AGREEMENT = re.compile(r"\bagree", re.I)


def fit_study_links(statistics: list[Statistic], items: list[ContextItem], study: list[str]) -> list[Statistic]:
    """Takes study items off statistics they cannot qualify.

    Model inputs keep only the study's period, region and data source, and so do measurements in a modelling
    study (scenarios among its items): it measured nothing itself, so they are background figures such as a
    region's share of production. A rating scale reaches only
    survey answers, and a scale without agreement wording ("1 (Not at all) to 5 (Extremely)") only
    the average ratings, not the shares agreeing.
    """
    by_id = {i.item_id: i for i in items}
    inputs = {"model_assumption"}
    if sum(i.item_type == "scenario" for i in items) >= MODELLING_STUDY_SCENARIOS:
        inputs.add("measurement")

    def fits(statistic: Statistic, item: ContextItem) -> bool:
        if statistic.record_type in inputs and item.item_type not in INPUT_KINDS:
            return False
        text = f"{item.value} {item.quote}"
        if RATING_SCALE.search(text):
            if statistic.record_type != "survey_response":
                return False
            share = "%" in (statistic.unit or "") + statistic.reported_text
            if share and not AGREEMENT.search(text):
                return False
        return True

    fitted = []
    for s in statistics:
        keep = [i for i in s.context_item_ids if i not in study or i not in by_id or fits(s, by_id[i])]
        fitted.append(s if keep == s.context_item_ids else s.model_copy(update={"context_item_ids": keep}))
    return fitted


def document_link_prompt(items: list[ContextItem], statistics: list[Statistic]) -> str:
    listed = "\n".join(f"{s.statistic_id} | {s.page} | {s.variable} | {s.reported_text} | {s.quote}" for s in statistics)
    return DOCUMENT_LINK_PROMPT.format(items=item_list(items), statistics=listed)


def relink(
    items: list[ContextItem], statistics: list[Statistic], backend: Backend, document_id: str
) -> tuple[list[Statistic], list[Context], dict[str, str | None], list[str]]:
    """Replaces every statistic's links with one whole-document linking call; returns failures too."""
    known = {i.item_id for i in items}
    try:
        links = backend.generate(document_link_prompt(items, statistics), DocumentLinks).parsed.links
    except RuntimeError as exc:
        return statistics, *group_contexts(document_id, items, statistics), [f"document links: {exc}"]
    chosen = {entry.statistic_id: [i for i in dict.fromkeys(entry.item_ids) if i in known] for entry in links}
    linked = [s.model_copy(update={"context_item_ids": chosen.get(s.statistic_id, [])}) for s in statistics]
    return linked, *group_contexts(document_id, items, linked), []


def item_list(items: list[ContextItem]) -> str:
    return "\n".join(f"{i.item_id} | {i.item_type} | {i.value}" for i in items) or "(none)"


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


# Words too common to show that a statistic answers a particular survey question.
COMMON_WORDS = set("the a an of to in for and or is are be was were that this it its on by with as at from their they "
                   "our we you not can all have has".split())
QUESTION_WORDS_FOUND = 0.4  # share of a question's words that must appear in the statistic's variable or sentence


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in COMMON_WORDS and len(w) > 2}


def fit_questions(statistics: list[Statistic], items: list[ContextItem]) -> list[Statistic]:
    """Takes off survey questions a statistic does not answer.

    The linking step often adds questions from other pages; a statistic answering a question repeats its
    words ("agree with: the cotton industry uses water responsibly"), so a question whose words are mostly
    absent from the statistic's variable and sentence belongs to another number.
    """
    questions = {i.item_id: _content_words(i.value) for i in items if i.item_type == "survey_question"}
    fitted = []
    for s in statistics:
        text = _content_words(f"{s.variable} {s.quote}")
        keep = [i for i in s.context_item_ids
                if i not in questions or (questions[i] and len(questions[i] & text) >= QUESTION_WORDS_FOUND * len(questions[i]))]
        fitted.append(s if keep == s.context_item_ids else s.model_copy(update={"context_item_ids": keep}))
    return fitted
