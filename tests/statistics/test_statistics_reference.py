"""Checks the hand-built statistics reference against the schema and, when the PDFs are present, the source text."""

import re

import pytest

from crdc_pipeline.extraction.statistics.schema import Statistic
from crdc_pipeline.reference import findings_reference

DATE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


def documents():
    return [pytest.param(d, id=d["document_id"]) for d in findings_reference()]


@pytest.mark.parametrize("document", documents())
def test_statistics_fit_the_schema_and_rules(document) -> None:
    item_ids = {i["item_id"] for i in document["context_items"]}
    statistics = [Statistic(**s) for s in document["statistics"]]
    assert len({s.statistic_id for s in statistics}) == len(statistics)
    for statistic in statistics:
        assert set(statistic.context_item_ids) <= item_ids, statistic.statistic_id
        bounded = statistic.value_status in ("lower_bound", "upper_bound", "range")
        assert (statistic.value is None) == bounded, statistic.statistic_id
        if statistic.value_status in ("lower_bound", "range"):
            assert statistic.value_min is not None, statistic.statistic_id
        if statistic.value_status in ("upper_bound", "range"):
            assert statistic.value_max is not None, statistic.statistic_id
    for record in document["statistics"]:
        acceptable = set(record.get("acceptable_item_ids", []))
        assert acceptable <= item_ids, record["statistic_id"]
        assert not acceptable & set(record["context_item_ids"]), record["statistic_id"]
    for period in [s.period for s in statistics]:
        for date in [period.start, period.end] if period else []:
            assert date is None or DATE.match(date), date


@pytest.mark.parametrize("document", documents())
def test_statistic_quotes_appear_on_their_page(document, quote_checker) -> None:
    missing = quote_checker(document["document_id"], document["statistics"])
    assert not missing, "quotes not found: " + ", ".join(missing)
