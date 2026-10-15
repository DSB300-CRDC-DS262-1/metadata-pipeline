"""Statistic schema."""

from crdc_pipeline.extraction.statistics.schema import Statistic, constraints_for


def statistic(statistic_id: str, item_ids: list[str], statistic_type: str = "count") -> Statistic:
    return Statistic(statistic_id=statistic_id, variable="deaths", value=6, reported_text="six", unit="deaths",
                     statistic_type=statistic_type, record_type="measurement", value_status="exact",
                     context_item_ids=item_ids, page=1, quote="q")


def test_constraints_come_from_the_statistic_type() -> None:
    assert constraints_for(statistic("s1", [], "percentage")) == ["non_negative", "at_most_100"]
    assert constraints_for(statistic("s1", [], "index")) == []
