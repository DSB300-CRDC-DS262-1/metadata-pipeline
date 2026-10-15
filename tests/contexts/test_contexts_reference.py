"""Checks the hand-built contexts reference against the schema and, when the PDFs are present, the source text."""

import re

import pytest

from crdc_pipeline.extraction.contexts.schema import ContextItem, group_contexts
from crdc_pipeline.extraction.statistics.schema import Statistic
from crdc_pipeline.reference import findings_reference

DATE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


def documents():
    return [pytest.param(d, id=d["document_id"]) for d in findings_reference()]


@pytest.mark.parametrize("document", documents())
def test_items_fit_the_schema_and_rules(document) -> None:
    items = [ContextItem(**i) for i in document["context_items"]]
    item_ids = [i.item_id for i in items]
    assert len(item_ids) == len(set(item_ids))
    for period in [i.period for i in items]:
        for date in [period.start, period.end] if period else []:
            assert date is None or DATE.match(date), date


@pytest.mark.parametrize("document", documents())
def test_stored_contexts_match_the_grouping(document) -> None:
    items = [ContextItem(**i) for i in document["context_items"]]
    statistics = [Statistic(**s) for s in document["statistics"]]
    contexts, assigned = group_contexts(document["document_id"], items, statistics)
    assert document["contexts"] == [c.model_dump() for c in contexts]
    assert {s["statistic_id"]: s["context_id"] for s in document["statistics"]} == assigned


@pytest.mark.parametrize("document", documents())
def test_item_quotes_appear_on_their_page(document, quote_checker) -> None:
    alternatives = [{**q, "item_id": i["item_id"]} for i in document["context_items"] for q in i.get("also_stated", [])]
    missing = quote_checker(document["document_id"], document["context_items"] + alternatives)
    assert not missing, "quotes not found: " + ", ".join(missing)
