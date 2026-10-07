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


def test_running_headers_asides_and_generic_words_are_not_conditions() -> None:
    from crdc_pipeline.extraction.contexts.scope import not_a_condition, running_lines

    pages = ["Method\nData came from NCIS.\nSentinel Health and Safety Data for Cotton Farms 1",
             "Results\nSix deaths were found.\nSentinel Health and Safety Data for Cotton Farms 2"]
    running = running_lines(pages)

    def item(value, quote, kind="data_source"):
        return ContextItem(item_id="i", item_type=kind, value=value, quote=quote, page=1)

    assert not_a_condition(item("Sentinel data", "Sentinel Health and Safety Data for Cotton Farms 2"), running)
    assert not_a_condition(item("NCIS", "Our thanks are extended to the staff of the NCIS."), running)
    assert not_a_condition(item("information may be incomplete",
                                "The reader is advised that such information may be incomplete."), running)
    assert not_a_condition(item("Survey", "73% of survey participants agreed.", "collection_method"), running)
    assert not not_a_condition(item("national survey", "Who completed the 2025 national survey?", "collection_method"), running)
    assert not not_a_condition(item("NCIS", "Data came from NCIS."), running)
    assert not not_a_condition(item("Survey", "Survey", "survey_question"), running)


def test_study_items_fit_the_statistics_they_reach() -> None:
    from crdc_pipeline.extraction.contexts.links import fit_study_links

    def item(item_id, kind, value):
        return ContextItem(item_id=item_id, item_type=kind, value=value, quote=f"{value}.", page=1)

    items = [item("region", "geographic_region", "northwest NSW"), item("unit", "functional_unit", "one tonne of lint"),
             item("trust", "study_design", "Measured on a scale from 1 (Not at all) to 5 (Extremely)"),
             item("likert", "study_design", "5-point agreement scales, 1 = Strongly disagree")]
    study = ["region", "unit", "trust", "likert"]
    input_value = stat("85%", 85, unit="%", record_type="model_assumption", context_item_ids=study)
    share = stat("47%", 47, unit="%", record_type="survey_response", context_item_ids=study + ["q1"])
    rating = stat("3.19", 3.19, record_type="survey_response", context_item_ids=study)
    result = stat("1601", 1601, record_type="scenario_result", context_item_ids=study)
    fitted = fit_study_links([input_value, share, rating, result], items, study)
    assert fitted[0].context_item_ids == ["region"]  # a model input keeps the period, region and source
    assert fitted[1].context_item_ids == ["region", "unit", "likert", "q1"]  # a share: the agreement scale only
    assert fitted[2].context_item_ids == study  # a rating: both scales
    assert fitted[3].context_item_ids == ["region", "unit"]  # not a survey answer: no rating scale

    # In a modelling study (scenarios among its items) a measurement is a background figure, fitted like an input.
    background = stat("96%", 96, unit="%", record_type="measurement", context_item_ids=["region", "unit"])
    scenarios = [item("s1", "scenario", "N cut to 180 kg N/ha"), item("s2", "scenario", "biodiesel B20")]
    assert fit_study_links([background], items, study)[0].context_item_ids == ["region", "unit"]
    assert fit_study_links([background], items + scenarios, study)[0].context_item_ids == ["region"]


def test_a_key_kind_missing_from_the_scope_is_taken_from_the_method_page() -> None:
    from crdc_pipeline.extraction.contexts.scope import sharpen_scope
    from crdc_pipeline.extraction.statistics.periods import with_item_period

    def item(item_id, kind, value, page):
        return with_item_period(ContextItem(item_id=item_id, item_type=kind, value=value, page=page,
                                            quote=f"The study reports {value}."))

    items = [item("t", "time_period", "2025-10-30 to 2025-11-17", 5), item("s", "sample_population", "3,063 surveys", 5),
             item("d", "data_source", "an online research panel", 5), item("other", "data_source", "Kmart Q&A", 9)]
    assert sharpen_scope(["t", "s"], items) == ["t", "s", "d"]
    assert sharpen_scope(["t", "s", "other"], items) == ["t", "s", "other"]  # a source was chosen: keep it


def test_a_statistic_keeps_only_the_survey_questions_it_answers() -> None:
    from crdc_pipeline.extraction.contexts.links import fit_questions

    items = [ContextItem(item_id=i, item_type="survey_question", value=v, quote=f"{v}.", page=11)
             for i, v in (("water", "the cotton industry uses water responsibly"),
                          ("pesticides", "the use of pesticides really bothers me"))]
    answer = stat("47%", 47, unit="%", record_type="survey_response", variable="agree with: the cotton industry uses water responsibly",
                  context_item_ids=["region", "water", "pesticides"])
    assert fit_questions([answer], items)[0].context_item_ids == ["region", "water"]


def test_a_survey_answer_without_a_question_takes_the_statement_its_sentence_quotes() -> None:
    from crdc_pipeline.extraction.contexts.links import questions_from_quotes

    quote = ('Most agreed that "plays an important role in regional communities" (71%) and "creates an important '
             'export product for Australia" (80%).')
    first = stat("71%", 71, unit="%", record_type="survey_response", page=8, quote=quote,
                 variable="agree with: plays an important role in regional communities")
    again = first.model_copy(update={"statistic_id": "s2", "reported_text": "71"})
    other = stat("80%", 80, unit="%", record_type="survey_response", page=8, quote=quote,
                 variable="agree with: creates an important export product")
    linked, items = questions_from_quotes([first, again, other], [])
    assert [i.value for i in items] == ["plays an important role in regional communities",
                                        "creates an important export product for Australia"]
    assert [s.context_item_ids for s in linked] == [["p8_q01"], ["p8_q01"], ["p8_q02"]]
