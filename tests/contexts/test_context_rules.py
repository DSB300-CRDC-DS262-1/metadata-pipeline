"""Rules for context items: the study's scope, labels, item periods and links."""

from crdc_pipeline.extraction.backends import GenerationResult
from crdc_pipeline.extraction.contexts.schema import ContextItem
from crdc_pipeline.extraction.statistics.schema import Statistic


def stat(reported, value=None, status="exact", **extra):
    defaults = {"quote": "q", "record_type": "measurement", "variable": "v", "statistic_type": "other",
                "statistic_id": "s", "page": 1}
    return Statistic(value=value, reported_text=reported, value_status=status, **{**defaults, **extra})


def test_relink_replaces_links_with_one_document_call() -> None:
    from crdc_pipeline.extraction.contexts.links import relink
    from crdc_pipeline.extraction.contexts.prompts import DocumentLinks, DocumentStatisticLinks

    class Linker:
        def generate(self, prompt, schema, images=None):
            assert schema is DocumentLinks and "s1 | 1 |" in prompt
            parsed = DocumentLinks(links=[DocumentStatisticLinks(statistic_id="s1", item_ids=["a", "a", "gone"])])
            return GenerationResult(parsed=parsed, model="fake", seconds=1.0, prompt_tokens=1, output_tokens=1)

    item = ContextItem(item_id="a", item_type="limitation", value="v", quote="Data were provisional at the time of writing.", page=1)
    old = stat("six", 6).model_copy(update={"statistic_id": "s1", "context_item_ids": ["a", "old"]})
    linked, contexts, assigned, failed = relink([item], [old], Linker(), "doc_x")
    assert linked[0].context_item_ids == ["a"] and assigned["s1"] == contexts[0].context_id and failed == []


def test_time_period_items_cover_every_period_they_print() -> None:
    from crdc_pipeline.extraction.statistics.periods import with_item_period

    def period(value):
        found = with_item_period(ContextItem(item_id="a", item_type="time_period", value=value, page=1, quote=value)).period
        return (found.start, found.end) if found else None

    assert period("three production seasons of 2011-2012, 2012-2013, and 2013-2014") == ("2011", "2014")
    assert period("recruited between 30 October and 17 November 2025") == ("2025-10-30", "2025-11-17")
    assert period("since 2001") == ("2001", None)
    assert period("Oct 2019 - Dec 2023") == ("2019-10", "2023-12")


def test_a_span_of_years_gives_way_to_the_fieldwork_dates() -> None:
    from crdc_pipeline.extraction.contexts.scope import sharpen_scope
    from crdc_pipeline.extraction.statistics.periods import with_item_period

    def item(item_id, value):
        return with_item_period(ContextItem(item_id=item_id, item_type="time_period", value=value, page=1, quote=value))

    items = [item("a", "2020-2025"), item("b", "between 30 October and 17 November 2025"), item("c", "Oct 2019 - Dec 2023"),
             ContextItem(item_id="x", item_type="geographic_region", value="Australia", page=1, quote="across Australia")]
    assert sharpen_scope(["a", "x"], items) == ["b", "x"]
    assert sharpen_scope(["c"], [items[0], items[2]]) == ["c"]  # a dated span over several years is not fieldwork


def test_a_sample_without_a_size_gives_way_to_the_sized_sample_in_the_method() -> None:
    from crdc_pipeline.extraction.contexts.scope import sharpen_scope

    items = [
        ContextItem(item_id="m", item_type="collection_method", value="online survey", page=5,
                    quote="An online survey methodology was used to access the views of Australians."),
        ContextItem(item_id="v", item_type="sample_population", value="Australians", page=15,
                    quote="When asked if the industry was prepared to change, Australians agreed."),
        ContextItem(item_id="s", item_type="sample_population", value="3,063 surveys analysed", page=5,
                    quote="3,063 surveys were included for analysis after data cleaning."),
        ContextItem(item_id="p", item_type="sample_population", value="70,000 members in 14 countries", page=2,
                    quote="The platform engaged over 70,000 community members in 14 countries."),
    ]
    assert sharpen_scope(["m", "v"], items) == ["m", "s"]


def test_items_quoting_only_a_phrase_are_dropped() -> None:
    from crdc_pipeline.extraction.contexts.scope import is_label

    phrase = ContextItem(item_id="a", item_type="geographic_region", value="Australia", page=9,
                         quote="Australia's cotton industry")
    question = ContextItem(item_id="b", item_type="survey_question", value="water", page=9, quote="uses water responsibly")
    assert is_label(phrase, "")
    assert not is_label(question, "")


def test_the_sample_is_found_on_the_page_of_the_sharpened_period() -> None:
    from crdc_pipeline.extraction.contexts.scope import sharpen_scope
    from crdc_pipeline.extraction.statistics.periods import with_item_period

    def period(item_id, value, page):
        return with_item_period(ContextItem(item_id=item_id, item_type="time_period", value=value, page=page, quote=value))

    items = [period("a", "2020-2025", 15), period("b", "between 30 October and 17 November 2025", 5),
             ContextItem(item_id="v", item_type="sample_population", value="Australians", page=15,
                         quote="Most Australians agreed with the statement."),
             ContextItem(item_id="s", item_type="sample_population", value="3,063 surveys analysed", page=5,
                         quote="3,063 surveys were included for analysis after data cleaning.")]
    assert sharpen_scope(["a", "v"], items) == ["b", "s"]


def test_linked_items_that_repeat_a_kind_the_study_covers_are_dropped() -> None:
    from crdc_pipeline.extraction.contexts.links import trim_links

    def item(item_id, kind, page):
        return ContextItem(item_id=item_id, item_type=kind, value="v", page=page, quote="A fact stated in a sentence.")

    items = [item("st", "time_period", 5), item("ss", "sample_population", 5), item("t2", "time_period", 9),
             item("sub", "sample_population", 9), item("far", "sample_population", 2), item("q", "survey_question", 9)]
    answer = stat("38%", 38, page=9, context_item_ids=["st", "ss", "t2", "sub", "far", "q"])
    trimmed = trim_links([answer], items, ["st", "ss"])[0]
    assert trimmed.context_item_ids == ["st", "ss", "sub", "q"]


def test_a_region_from_the_abstract_gives_way_to_the_sites_in_the_method() -> None:
    from crdc_pipeline.extraction.contexts.scope import sharpen_scope
    from crdc_pipeline.extraction.statistics.periods import with_item_period

    def item(item_id, kind, value, page):
        made = ContextItem(item_id=item_id, item_type=kind, value=value, page=page, quote=f"The study reports {value}.")
        return with_item_period(made)

    items = [item("r1", "geographic_region", "mid-southern United States", 1),
             item("t", "time_period", "2015-2017", 2), item("r2", "geographic_region", "Starkville and Stoneville, MS", 2)]
    assert sharpen_scope(["r1", "t"], items) == ["r2", "t"]


def test_a_collection_method_under_one_chart_gives_way_to_the_one_in_the_method() -> None:
    from crdc_pipeline.extraction.contexts.scope import sharpen_scope
    from crdc_pipeline.extraction.statistics.periods import with_item_period

    def item(item_id, kind, value, page):
        made = ContextItem(item_id=item_id, item_type=kind, value=value, page=page, quote=f"The study reports {value}.")
        return with_item_period(made)

    items = [item("t", "time_period", "2025-10-30 to 2025-11-17", 5), item("m", "collection_method", "online survey", 5),
             item("c1", "collection_method", "Survey responses", 15), item("c2", "collection_method", "a survey", 11)]
    assert sharpen_scope(["t", "c1", "c2"], items) == ["t", "m"]
    assert sharpen_scope(["t", "m"], items) == ["t", "m"]
