"""Schema for statistics and the contexts that explain them.

A document's context items are single facts with a source quote (a data source, a population, a
limitation). Each statistic lists the items that apply to it; statistics with the same shared
conditions share one context, built here in code rather than by the model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ItemType = Literal[
    "data_source", "collection_method", "sample_population", "geographic_region", "time_period",
    "inclusion_criteria", "exclusion_criteria", "study_design", "preprocessing", "assumption",
    "limitation", "functional_unit", "currency_basis", "survey_question", "scenario", "other",
]
StatisticType = Literal[
    "count", "percentage", "rate", "average", "median", "ratio", "monetary_value", "duration", "index", "other",
]
RecordType = Literal["measurement", "survey_response", "scenario_result", "model_assumption", "derived", "cited"]
ValueStatus = Literal["exact", "approximate", "lower_bound", "upper_bound", "range"]
STATISTIC_LEVEL = {"survey_question", "scenario"}

CONSTRAINTS: dict[str, list[str]] = {
    "count": ["non_negative", "integer"],
    "percentage": ["non_negative", "at_most_100"],
    "rate": ["non_negative"],
    "monetary_value": ["non_negative"],
    "duration": ["non_negative"],
}


class Period(BaseModel):
    start: str | None = Field(default=None, description="ISO 8601 at the precision printed: '2014', '2019-10' or '2025-10-30'.")
    end: str | None = Field(default=None, description="ISO 8601 at the precision printed. Never add a month or day the text does not give.")


class ContextItem(BaseModel):
    item_id: str = Field(description="Unique within the document, e.g. 'item_01'.")
    item_type: ItemType
    value: str = Field(description="The fact in a short phrase, e.g. 'costs in 2018 Australian dollars'.")
    period: Period | None = Field(default=None, description="Only for time_period items.")
    whole_document: bool = Field(
        default=False, description="True if it applies to every statistic in the document."
    )
    quote: str = Field(description="Exact sentence the fact comes from.")
    page: int = Field(description="Page number from the '=== page N ===' marker.")


class Statistic(BaseModel):
    statistic_id: str = Field(description="Unique within the document, e.g. 'stat_01'.")
    variable: str = Field(description="What is measured, as a short noun phrase.")
    value: float | None = Field(description="The number, for exact and approximate values. Null for bounds and ranges.")
    value_min: float | None = Field(default=None, description="For lower_bound and range: the lowest value stated.")
    value_max: float | None = Field(default=None, description="For upper_bound and range: the highest value stated.")
    reported_text: str = Field(description="The value exactly as written, e.g. '47%', 'fewer than five'.")
    unit: str | None = Field(default=None, description="Unit only, e.g. '%', 'deaths', 'AUD million'.")
    statistic_type: StatisticType
    record_type: RecordType = Field(
        description="measurement: observed data. survey_response: answers to a survey question. "
        "scenario_result: a modelled outcome. model_assumption: an input the authors assumed. "
        "derived: computed from other statistics in the document. cited: reported from another study."
    )
    value_status: ValueStatus = Field(
        description="exact: stated plainly. approximate: '~', 'about', 'around'. lower_bound: 'over', 'more than', "
        "'surpasses'. upper_bound: 'fewer than', 'within', 'approaching'. range: '3-5'."
    )
    confirmed: bool = Field(default=True, description="False if the document says the value is unconfirmed or possible.")
    period: Period | None = Field(default=None, description="The period this value refers to.")
    response_category: str | None = Field(
        default=None, description="For survey answers: agree, neutral, disagree, important, etc."
    )
    context_item_ids: list[str] = Field(
        default_factory=list, description="IDs of the context items that apply to this statistic only."
    )
    page: int = Field(description="Page number from the '=== page N ===' marker.")
    quote: str = Field(description="Exact sentence(s) the value comes from.")


class Context(BaseModel):
    context_id: str
    document_id: str
    item_ids: list[str]


def constraints_for(statistic: Statistic) -> list[str]:
    """Value checks implied by the statistic type, set in code rather than by the model."""
    return CONSTRAINTS.get(statistic.statistic_type, [])


def group_contexts(
    document_id: str, items: list[ContextItem], statistics: list[Statistic]
) -> tuple[list[Context], dict[str, str | None]]:
    """One context per distinct set of shared conditions; returns the contexts and each statistic's context_id.

    Survey questions and scenarios describe a single statistic, so they stay on the statistic and
    do not split its context.
    """
    known = {item.item_id for item in items if item.item_type not in STATISTIC_LEVEL}
    shared = [item.item_id for item in items if item.whole_document]
    contexts: dict[tuple[str, ...], Context] = {}
    assigned: dict[str, str | None] = {}
    for statistic in statistics:
        own = [i for i in statistic.context_item_ids if i in known and i not in shared]
        key = tuple(dict.fromkeys(shared + sorted(own)))
        if not key:
            assigned[statistic.statistic_id] = None
            continue
        if key not in contexts:
            contexts[key] = Context(
                context_id=f"{document_id}_ctx_{len(contexts) + 1:02d}", document_id=document_id, item_ids=list(key)
            )
        assigned[statistic.statistic_id] = contexts[key].context_id
    return list(contexts.values()), assigned
