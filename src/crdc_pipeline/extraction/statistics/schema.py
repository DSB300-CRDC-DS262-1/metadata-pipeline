"""Schema for statistics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..contexts.schema import Period


StatisticType = Literal[
    "count", "percentage", "rate", "average", "median", "ratio", "monetary_value", "duration", "index", "other",
]


RecordType = Literal["measurement", "survey_response", "scenario_result", "model_assumption", "derived", "cited"]


ValueStatus = Literal["exact", "approximate", "lower_bound", "upper_bound", "range"]


CONSTRAINTS: dict[str, list[str]] = {
    "count": ["non_negative", "integer"],
    "percentage": ["non_negative", "at_most_100"],
    "rate": ["non_negative"],
    "monetary_value": ["non_negative"],
    "duration": ["non_negative"],
}


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


def constraints_for(statistic: Statistic) -> list[str]:
    """Value checks implied by the statistic type, set in code rather than by the model."""
    return CONSTRAINTS.get(statistic.statistic_type, [])
