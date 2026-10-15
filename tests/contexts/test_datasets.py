"""Splitting the study's own items by dataset, and narrowing method facts to the numbers they cover."""

from crdc_pipeline.extraction.contexts.datasets import method_reach, split_by_dataset, with_study_items
from crdc_pipeline.extraction.contexts.prompts import (
    Dataset,
    DatasetAssignments,
    DatasetNumbers,
    MethodReach,
    MethodReaches,
    StudyDatasets,
)
from crdc_pipeline.extraction.contexts.schema import ContextItem
from crdc_pipeline.extraction.statistics.schema import Statistic


def item(item_id, kind):
    return ContextItem(item_id=item_id, item_type=kind, value=item_id, quote=f"The {item_id} sentence.", page=1)


def stat(statistic_id, links=(), record_type="measurement"):
    return Statistic(statistic_id=statistic_id, variable="v", value=1, reported_text="1", value_status="exact",
                     statistic_type="count", record_type=record_type, page=1, quote="q", context_item_ids=list(links))


ITEMS = [item("deaths_src", "data_source"), item("deaths_lim", "limitation"), item("claims_src", "data_source"),
         item("claims_lim", "limitation"), item("region", "geographic_region")]
STUDY = [i.item_id for i in ITEMS]


def answers(*parsed):
    queue = list(parsed)
    return lambda prompt, schema, step: queue.pop(0)


def test_each_statistic_takes_its_own_datasets_items_and_the_shared_ones() -> None:
    groups = StudyDatasets(datasets=[Dataset(name="Deaths", item_ids=["deaths_src", "deaths_lim"]),
                                     Dataset(name="Claims", item_ids=["claims_src", "claims_lim"])],
                           shared_item_ids=["region"])
    # The model sometimes copies the whole listed line instead of the ID.
    picked = DatasetAssignments(datasets=[DatasetNumbers(dataset="Deaths", statistic_ids=["s1 | 5 | deaths | six | ..."]),
                                          DatasetNumbers(dataset="Claims", statistic_ids=["s2"])])
    statistics = [stat("s1"), stat("s2", links=["deaths_lim"]), stat("s3", record_type="cited")]
    datasets, shared, index = split_by_dataset(STUDY, ITEMS, statistics, answers(groups, picked))
    assert shared == ["region"] and index == {"s1": 0, "s2": 1}
    linked = [with_study_items(s, STUDY, datasets, shared, index) for s in statistics]
    assert linked[0].context_item_ids == ["region", "deaths_src", "deaths_lim"]
    assert linked[1].context_item_ids == ["region", "claims_src", "claims_lim"]  # another dataset's link is dropped
    assert linked[2].context_item_ids == []  # cited figures take none of the study's items


def test_one_dataset_or_a_group_of_method_facts_alone_keeps_every_item_shared() -> None:
    items = ITEMS + [item("path_model", "study_design")]
    groups = StudyDatasets(datasets=[Dataset(name="Survey", item_ids=STUDY),
                                     Dataset(name="Modelling", item_ids=["path_model"])])
    datasets, shared, index = split_by_dataset(STUDY + ["path_model"], items, [stat("s1")], answers(groups))
    assert datasets == {} and shared == STUDY + ["path_model"] and index == {}
    assert with_study_items(stat("s1"), shared, datasets, shared, index).context_item_ids == shared


def test_method_facts_are_narrowed_only_when_they_cover_few_numbers() -> None:
    items = ITEMS + [item("path_model", "study_design"), item("cleaning", "preprocessing"), item("scale", "study_design")]
    study = ["region", "path_model", "cleaning", "scale"]
    statistics = [stat(f"s{n}") for n in range(1, 10)]
    reach = MethodReaches(facts=[
        MethodReach(item_id="path_model", applies_to_all=False, statistic_ids=["s1"]),
        MethodReach(item_id="cleaning", applies_to_all=True),
        MethodReach(item_id="scale", applies_to_all=False, statistic_ids=["s1", "s2", "s3", "s4"]),  # wide: kept for all
        MethodReach(item_id="region", applies_to_all=False, statistic_ids=["s1"]),  # not a method fact: ignored
    ])
    assert method_reach(study, items, statistics, answers(reach)) == {"path_model": {"s1"}}
