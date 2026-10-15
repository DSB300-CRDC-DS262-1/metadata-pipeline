"""Record types decided across a document: model results in a modelling study, survey answers."""

from __future__ import annotations

import re

from ..contexts.schema import ContextItem
from .schema import Statistic


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
