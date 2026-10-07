from crdc_pipeline.extraction.findings import ContextItem, Statistic, constraints_for, group_contexts


def item(item_id: str, whole_document: bool = False) -> ContextItem:
    return ContextItem(item_id=item_id, item_type="limitation", value="v", whole_document=whole_document,
                       quote="q", page=1)


def statistic(statistic_id: str, item_ids: list[str], statistic_type: str = "count") -> Statistic:
    return Statistic(statistic_id=statistic_id, variable="deaths", value=6, reported_text="six", unit="deaths",
                     statistic_type=statistic_type, record_type="measurement", value_status="exact",
                     context_item_ids=item_ids, page=1, quote="q")


def test_statistics_with_the_same_items_share_one_context() -> None:
    items = [item("source", whole_document=True), item("a"), item("b")]
    stats = [statistic("s1", ["b", "a"]), statistic("s2", ["a", "b"]), statistic("s3", ["a"])]
    contexts, assigned = group_contexts("doc", items, stats)
    assert assigned["s1"] == assigned["s2"] != assigned["s3"]
    assert [c.item_ids for c in contexts] == [["source", "a", "b"], ["source", "a"]]


def test_survey_questions_stay_on_the_statistic() -> None:
    question = ContextItem(item_id="q1", item_type="survey_question", value="v", quote="q", page=1)
    contexts, assigned = group_contexts("doc", [item("a"), question], [statistic("s1", ["a", "q1"]), statistic("s2", ["a"])])
    assert assigned["s1"] == assigned["s2"]
    assert contexts[0].item_ids == ["a"]


def test_unknown_item_ids_are_ignored() -> None:
    contexts, assigned = group_contexts("doc", [item("a")], [statistic("s1", ["a", "made_up"])])
    assert contexts[0].item_ids == ["a"]


def test_constraints_come_from_the_statistic_type() -> None:
    assert constraints_for(statistic("s1", [], "percentage")) == ["non_negative", "at_most_100"]
    assert constraints_for(statistic("s1", [], "index")) == []
