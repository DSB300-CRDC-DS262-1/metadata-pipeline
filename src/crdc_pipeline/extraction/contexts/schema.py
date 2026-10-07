"""Context items (single facts with a source quote) and the contexts built from them.

A document's context items are facts such as a data source, a population or a limitation. Each
statistic lists the items that apply to it; statistics with the same shared conditions share one
context, built here in code rather than by the model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..statistics.schema import Statistic


ItemType = Literal[
    "data_source", "collection_method", "sample_population", "geographic_region", "time_period",
    "inclusion_criteria", "exclusion_criteria", "study_design", "preprocessing", "assumption",
    "limitation", "functional_unit", "currency_basis", "survey_question", "scenario", "other",
]


STATISTIC_LEVEL = {"survey_question", "scenario"}


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


class Context(BaseModel):
    context_id: str
    document_id: str
    item_ids: list[str]


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
