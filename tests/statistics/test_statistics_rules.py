"""Rules applied to each statistic: readings, types, periods, filters and duplicates."""

from crdc_pipeline.extraction.contexts.prompts import CONTEXT_PROMPT, PageContextItems
from crdc_pipeline.extraction.contexts.schema import ContextItem
from crdc_pipeline.extraction.statistics.prompts import STATISTICS_PROMPT, PageStatistics
from crdc_pipeline.extraction.statistics.schema import Statistic


def stat(reported, value=None, status="exact", **extra):
    defaults = {"quote": "q", "record_type": "measurement", "variable": "v", "statistic_type": "other",
                "statistic_id": "s", "page": 1}
    return Statistic(value=value, reported_text=reported, value_status=status, **{**defaults, **extra})


def test_prompts_keep_dates_at_printed_precision() -> None:
    for prompt in (CONTEXT_PROMPT, STATISTICS_PROMPT):
        assert "Never add a month or day" in prompt


def test_schemas_cap_list_lengths() -> None:
    assert PageStatistics.model_json_schema()["properties"]["statistics"]["maxItems"] == 60
    assert PageContextItems.model_json_schema()["properties"]["items"]["maxItems"] == 30


def test_wording_sets_the_reading_and_fills_bounds() -> None:
    from crdc_pipeline.extraction.statistics.values import tidy_statistic

    over = tidy_statistic(stat("just over $3.4 million", 3.4, "upper_bound"))
    assert (over.value_status, over.value, over.value_min) == ("lower_bound", None, 3.4)
    within = tidy_statistic(stat("usually within seven days", 7))
    assert (within.value_status, within.value_max) == ("upper_bound", 7)
    span = tidy_statistic(stat("3-5 years", status="range"))
    assert (span.value_min, span.value_max) == (3, 5)
    assert tidy_statistic(stat("~99%", 99)).value_status == "approximate"


def test_years_and_periods_are_not_statistics() -> None:
    from crdc_pipeline.extraction.statistics.values import tidy_statistic

    for reported in ("2001", "2014-19", "2008-09 to 2011-12", "2018 Australian dollars", "(2001-12)"):
        assert tidy_statistic(stat(reported, 2001)) is None, reported
    assert tidy_statistic(stat("2,360", 2360)) is not None


def test_table_rows_are_dropped_but_number_heavy_prose_is_kept() -> None:
    from crdc_pipeline.extraction.statistics.filters import without_tables

    text = "2009-10 721 1.2 1,170.0 1.1\n(55% agreed, 34% neutral, 10% disagreed; Figure 5)\n2. Television - news (21%)."
    assert without_tables(text).splitlines() == text.splitlines()[1:]


def test_survey_answers_are_labelled_from_the_quote() -> None:
    from crdc_pipeline.extraction.statistics.values import label_survey_answer

    quote = "(73% agreed, 24% neutral, 3% disagreed; Figure 12)"
    for reported, category in (("73%", "agree"), ("24%", "neutral"), ("3%", "disagree")):
        labelled = label_survey_answer(stat(reported, float(reported[:-1]), quote=quote))
        assert (labelled.response_category, labelled.record_type) == (category, "survey_response")
    change = label_survey_answer(stat("0.9%", 0.9, quote="While agreement to this question has grown 0.9% since"))
    assert change.response_category is None


def test_percentages_written_as_fractions_take_the_printed_number() -> None:
    from crdc_pipeline.extraction.statistics.values import tidy_statistic

    assert tidy_statistic(stat("80%", 0.8)).value == 80
    assert tidy_statistic(stat("0.5%", 0.5)).value == 0.5


def test_measurements_with_a_citation_are_cited() -> None:
    from crdc_pipeline.extraction.statistics.values import label_cited

    cited = stat("3.9 million bales", 3.9, quote="production was 3.9 million bales (Cotton Australia, 2016).",
                 record_type="measurement")
    dated = stat("five", 5, quote="there were five cases (Oct 2019).", record_type="measurement")
    assert label_cited(cited).record_type == "cited"
    assert label_cited(dated).record_type == "measurement"


def test_period_is_read_from_the_sentence_nearest_the_value() -> None:
    from crdc_pipeline.extraction.statistics.periods import period_from_quote

    def span(quote, value):
        found = period_from_quote(quote, value)
        return (found.start, found.end) if found else None

    claims = "claims increased from a median of $2.18 million (2009-14) to just over $3.4 million (2014-17)"
    assert span(claims, "$2.18 million") == ("2009", "2014")
    assert span(claims, "just over $3.4 million") == ("2014", "2017")
    assert span("there have been fewer than five cases (Oct 2019 - Dec 2023)", "fewer than five") == ("2019-10", "2023-12")
    assert span("seventeen fatal incidents since 2001 surpasses $48 million", "$48 million") == ("2001", None)
    assert span("uses water responsibly at 47% (Figure 4) compared to 42% in 2020.", "47%") is None
    assert span("production was 3.9 million bales (Cotton Australia, 2016).", "3.9 million bales") is None


def test_survey_answers_take_the_fieldwork_year() -> None:
    from crdc_pipeline.extraction.statistics.periods import date_survey_answers, survey_year, with_item_period

    items = [with_item_period(ContextItem(item_id=i, item_type="time_period", value=v, page=1, quote=v))
             for i, v in [("a", "2025-10-30 to 2025-11-17"), ("b", "Program began in 2020"), ("c", "2014-19")]]
    assert (items[2].period.start, items[2].period.end) == ("2014", "2019")
    answer = stat("80%", 80, record_type="survey_response")
    dated = date_survey_answers([answer], survey_year(items, ["a", "b"]))[0]
    assert (dated.period.start, dated.period.end) == ("2025", "2025")


def test_reference_lists_and_chart_labels_are_left_out() -> None:
    from crdc_pipeline.extraction.contexts.scope import is_label
    from crdc_pipeline.extraction.statistics.filters import before_references

    pages = ["Results 5%", "Discussion\nReferences\nLal, 2004. Soil carbon.", "More references"]
    assert before_references(pages) == ["Results 5%", "Discussion", ""]
    legend = ContextItem(item_id="a", item_type="data_source", value="IPCC", page=1, quote="Using the IPCC emissions values")
    sentence = ContextItem(item_id="b", item_type="data_source", value="IPCC", page=1,
                           quote="We used the IPCC emissions values.")
    page = "Using the IPCC emissions values\nWe used the IPCC emissions values."
    assert is_label(legend, page) and not is_label(sentence, page)


def test_statistic_type_follows_the_unit() -> None:
    from crdc_pipeline.extraction.statistics.values import label_type

    assert label_type(stat("~99%", 99, unit="%")).statistic_type == "percentage"
    assert label_type(stat("approached $18 million", 18, unit="AUD million")).statistic_type == "monetary_value"
    median = stat("$2.18 million", 2.18, unit="AUD million", variable="median annual claim costs")
    assert label_type(median).statistic_type == "median"
    ratio = stat("AU$ 1.29/CO2e", 1.29, unit="AU$ per kg CO2e", statistic_type="ratio")
    assert label_type(ratio).statistic_type == "ratio"


def test_values_whose_only_numbers_are_years_are_dropped() -> None:
    from crdc_pipeline.extraction.statistics.values import tidy_statistic

    assert tidy_statistic(stat("five-year period 2012-13 to 2016-17")) is None
    assert tidy_statistic(stat("$2070/t", 2070)) is not None
    assert tidy_statistic(stat("$2.18 million (2009-14)", 2.18)) is not None


def test_a_measurement_linked_to_a_survey_question_is_a_survey_answer() -> None:
    from crdc_pipeline.extraction.statistics.records import answers_to_questions

    question = ContextItem(item_id="q", item_type="survey_question", value="trust", page=9,
                           quote="the average level of trust in Australia's cotton industry")
    rating = stat("3.19", 3.19, context_item_ids=["q"])
    other = stat("six", 6)
    labelled = answers_to_questions([rating, other], [question])
    assert [s.record_type for s in labelled] == ["survey_response", "measurement"]


def test_periods_are_dates_and_a_sentence_opening_period_covers_the_sentence() -> None:
    from crdc_pipeline.extraction.contexts.schema import Period
    from crdc_pipeline.extraction.statistics.periods import period_from_quote, with_period

    words = stat("10%", 10, quote="10% said they had heard about the industry in the last week",
                 period=Period(start="last week", end="last week"))
    assert with_period(words).period is None
    opening = "The five year (2009-2014) average production was 3.9 million bales from 334,775 ha"
    found = period_from_quote(opening, "334,775 ha")
    assert (found.start, found.end) == ("2009", "2014")
    assert period_from_quote("the average cost of a fatality in 2023 Australian dollars is $3.5 million", "$3.5 million") is None


def test_time_units_are_durations_and_averages_are_averages_unless_they_change() -> None:
    from crdc_pipeline.extraction.statistics.values import label_type

    assert label_type(stat("usually within seven days", 7, unit="days")).statistic_type == "duration"
    assert label_type(stat("3.68", 3.68, unit="score", variable="average rating, knows 3-5 people")).statistic_type == "average"
    change = stat("an increase of 0.07", 0.07, unit="rating points", variable="change in average trust")
    assert label_type(change).statistic_type == "other"


def test_numbers_written_in_words_are_listed_as_a_reminder() -> None:
    from crdc_pipeline.extraction.statistics.filters import word_number_note

    note = word_number_note("half of the systems were back-to-back cotton\nwhich may offset up to a third of the increase\n"
                            "with more than 99 per cent of production exported\n12 15 18 21 two")
    assert "half; up to a third; more than 99 per cent" in note
    assert "two" not in note  # the mostly numeric (table) line is skipped
    assert word_number_note("Results were 45% and 12 kg.") == ""


def test_the_answer_word_is_found_in_the_full_sentence_on_the_page() -> None:
    from crdc_pipeline.extraction.statistics.values import label_survey_answer

    page = ('More than three-quarters agreed that the cotton industry "creates an important export product for '
            'Australia" (78%; Figure 3). Fewer were neutral.')
    cut = stat("78%", 78, quote='creates an important export product for Australia" (78%;')
    assert label_survey_answer(cut).response_category is None
    assert label_survey_answer(cut, page).response_category == "agree"


def test_a_year_followed_by_a_percentage_is_not_a_span() -> None:
    from crdc_pipeline.extraction.statistics.periods import period_from_quote

    quote = "neutral responses increased from 49% in 2023 to 51% in 2025, reinforcing uncertainty"
    first, second = period_from_quote(quote, "49%"), period_from_quote(quote, "51%")
    assert (first.start, first.end) == ("2023", "2023")
    assert (second.start, second.end) == ("2025", "2025")
    found = period_from_quote("nine deaths in the 2001-13 period", "nine")
    assert (found.start, found.end) == ("2001", "2013")


def test_a_rating_is_a_survey_answer_even_without_a_question_link() -> None:
    from crdc_pipeline.extraction.statistics.records import answers_to_questions

    rating = stat("3.19", 3.19, unit="mean rating (1-5)")
    change = stat("an increase of 0.07", 0.07, unit="rating points", record_type="derived")
    assert [s.record_type for s in answers_to_questions([rating, change], [])] == ["survey_response", "derived"]


def test_duplicate_candidates_need_the_same_numbers_and_compatible_units() -> None:
    from crdc_pipeline.extraction.statistics.duplicates import duplicate_candidates

    abstract = stat("46%", 46, statistic_id="a", unit="%", page=1)
    results = stat("46%", 46, statistic_id="b", unit="%", page=6)
    other_unit = stat("46", 46, statistic_id="c", unit="kg", page=7)
    different = stat("47%", 47, statistic_id="d", unit="%", page=8)
    pairs = duplicate_candidates([abstract, results, other_unit, different])
    assert [(x.statistic_id, y.statistic_id) for x, y in pairs] == [("a", "b")]


def test_a_duplicate_judged_the_same_is_dropped_and_its_links_kept() -> None:
    from crdc_pipeline.extraction.statistics.duplicates import (
        drop_duplicates,
        duplicate_candidates,
        pair_numbers,
    )

    abstract = stat("46%", 46, statistic_id="a", unit="%", page=1, context_item_ids=["i1"])
    results = stat("46%", 46, statistic_id="b", unit="%", page=6, context_item_ids=["i2"])
    pairs = duplicate_candidates([abstract, results])
    kept = drop_duplicates([abstract, results], pair_numbers(["pair_001", "pair_099"], len(pairs)), pairs)
    assert [(s.statistic_id, s.context_item_ids) for s in kept] == [("a", ["i1", "i2"])]
    assert drop_duplicates([abstract, results], set(), pairs) == [abstract, results]


def test_numbers_that_set_up_the_survey_or_report_are_not_findings() -> None:
    from crdc_pipeline.extraction.statistics.filters import is_structural

    subgroup = stat("6-10", None, "range", value_min=6, value_max=10,
                    quote="Those who knew 6-10 gave the highest average rating")
    scale_end = stat("5", 5, quote="Measured on a scale from 1 (Not at all) to 5 (Extremely), the average was 3.19")
    chart = stat("3.54", 3.54, quote="n 3.54 3.51")
    parts = stat("three", 3, quote="Broadly, the community are focused on three main themes: water and biodiversity")
    finding = stat("3.19", 3.19, quote="Measured on a scale from 1 (Not at all) to 5 (Extremely), the average was 3.19")
    dated = stat("53%", 53, quote="53% in 2023")
    fragment = stat("AU$ 1.11/CO2e", 1.11, quote="or AU$ 1.11/CO2e @ 240 kg N/ha.")
    deaths = stat("two", 2, quote="a further two cases could have possibly occurred in cotton production")
    assert all(is_structural(s) for s in (subgroup, scale_end, chart, parts))
    assert not any(is_structural(s) for s in (finding, dated, fragment, deaths))


def test_shares_in_words_and_per_cent_are_percentages_as_printed() -> None:
    from crdc_pipeline.extraction.statistics.values import tidy_statistic

    half = tidy_statistic(stat("half", 0.5))
    third = tidy_statistic(stat("up to a third", None))
    exported = tidy_statistic(stat("more than 99 per cent", None))
    rank = tidy_statistic(stat("the third largest saving overall", 3))
    assert (half.value, half.unit) == (50.0, "%")
    assert (third.value_max, third.value_status) == (33.3, "upper_bound")
    assert (exported.value_min, exported.value_status) == (99.0, "lower_bound")
    assert rank.value == 3 and rank.unit is None


def test_a_bracketed_conversion_into_another_unit_is_the_same_number() -> None:
    from crdc_pipeline.extraction.statistics.duplicates import drop_conversions

    quote = "Average lint yield in Australia is the highest in the world at 12 bales/ha (2.7 t/ha)"
    bales = stat("12 bales/ha", 12, statistic_id="a", unit="bales/ha", quote=quote)
    tonnes = stat("2.7 t/ha", 2.7, statistic_id="b", unit="t/ha", quote=quote)
    result = "reducing N fertiliser to 240 kg N/ha (2.6% reduction)"
    rate = stat("240 kg N/ha", 240, statistic_id="c", unit="kg N/ha", quote=result)
    share = stat("2.6%", 2.6, statistic_id="d", unit="%", quote=result)
    assert [s.statistic_id for s in drop_conversions([bales, tonnes, rate, share])] == ["a", "c", "d"]


def test_computed_values_in_a_modelling_study_are_model_results() -> None:
    from crdc_pipeline.extraction.statistics.records import modelled_results
    from crdc_pipeline.extraction.statistics.values import label_cited

    scenarios = [ContextItem(item_id=f"s{k}", item_type="scenario", value="option", page=3,
                             quote=f"Option {k} reduces the fertiliser rate.") for k in (1, 2)]
    stage = stat("406 kg CO2e", 406, record_type="derived", quote="the pre-farm stage was 406 kg CO2e")
    cited = label_cited(stat("3.3 Pg C/year", 3.3, record_type="derived", quote="estimated at 3.3 Pg C/year (Lal, 2004)"))
    assert [s.record_type for s in modelled_results([stage, cited], scenarios)] == ["scenario_result", "cited"]
    assert modelled_results([stage], scenarios[:1])[0].record_type == "derived"  # not a modelling study


def test_ranks_are_positions_and_medians_are_medians() -> None:
    from crdc_pipeline.extraction.statistics.values import label_type

    assert label_type(stat("sixth", 6, unit="rank", statistic_type="count")).statistic_type == "other"
    median = stat("10.28 bales per ha", 10.28, unit="bales/ha", variable="median farm yield", statistic_type="average")
    assert label_type(median).statistic_type == "median"


def test_two_ends_of_a_printed_range_are_one_statistic() -> None:
    from crdc_pipeline.extraction.statistics.duplicates import merge_ranges

    quote = "N2O emissions factors, ranging from 0.0033 for the Darling Downs to 0.0105 for Southern Queensland"
    low = stat("0.0033", 0.0033, statistic_id="a", quote=quote)
    high = stat("0.0105", 0.0105, statistic_id="b", quote=quote)
    other = stat("298", 298, statistic_id="c", quote="global warming potentials of 1, 298 and 25")
    merged = merge_ranges([low, high, other])
    assert [(s.value_min, s.value_max, s.value_status) for s in merged[:1]] == [(0.0033, 0.0105, "range")]
    assert [s.statistic_id for s in merged] == ["a", "c"]


def test_chart_values_and_word_shares_restating_an_exact_figure_are_dropped() -> None:
    from crdc_pipeline.extraction.statistics.filters import is_structural

    chart = stat("3.71", 3.71, quote="Figure 15: Average scores for responsiveness by people known. 2 3.71")
    restated = stat("Close to half", 50, quote="Close to half of the community agreed that Australia receives a fair share at 47%")
    exact = stat("47%", 47, quote="Close to half of the community agreed that Australia receives a fair share at 47%")
    alone = stat("close to a quarter", 25, quote="46% could not recall; 10% heard last week, and close to a quarter more than six months ago")
    assert is_structural(chart) and is_structural(restated)
    assert not is_structural(exact) and not is_structural(alone)


def test_shares_of_people_answering_are_survey_answers_in_a_survey_report() -> None:
    from crdc_pipeline.extraction.statistics.records import answers_to_questions
    from crdc_pipeline.extraction.statistics.periods import period_from_quote

    question = ContextItem(item_id="q", item_type="survey_question", value="importance", page=8,
                           quote="the cotton industry is important to our way of life")
    recall = stat("46%", 46, unit="%", quote="close to half of all participants could not recall the industry at 46%")
    assert answers_to_questions([recall], [question])[0].record_type == "survey_response"
    assert answers_to_questions([recall], [])[0].record_type == "measurement"  # not a survey report
    assert period_from_quote("a 52% reduction, as per Tian et al. (2015).", "52%") is None


def test_answer_options_agreement_and_changes() -> None:
    from crdc_pipeline.extraction.statistics.values import label_survey_answer

    option = label_survey_answer(stat("21%", 21, quote="2. Television - news (21%)."))
    agreement = label_survey_answer(stat("62%", 62, quote="The community were largely in agreement that water should be used carefully (62%)"))
    change = label_survey_answer(stat("6%", 6, quote="73% of survey participants agreed, an increase of 6% since the first survey"))
    derived = label_survey_answer(stat("0.07", 0.07, record_type="derived", response_category="agree"))
    assert option.response_category == "television news"
    assert agreement.response_category == "agree"
    assert change.response_category is None and derived.response_category is None


def test_bound_wording_just_before_the_value_and_unconfirmed_wording() -> None:
    from crdc_pipeline.extraction.statistics.values import tidy_statistic

    over = tidy_statistic(stat("$48 million", 48, quote="the total direct impact since 2001 surpasses $48 million"))
    plain = tidy_statistic(stat("46%", 46, quote="over the period, 46% of claims were serious"))
    possible = tidy_statistic(stat("two", 2, quote="a further two cases could have possibly occurred"))
    assert (over.value_status, over.value_min) == ("lower_bound", 48)
    assert plain.value_status == "exact"
    assert possible.confirmed is False


def test_an_average_score_on_the_scale_is_a_survey_answer() -> None:
    from crdc_pipeline.extraction.statistics.records import answers_to_questions

    question = ContextItem(item_id="q", item_type="survey_question", value="trust", page=9, quote="the level of trust in the industry")
    score = stat("3.46", 3.46, quote="acceptance of the cotton industry scored an average of 3.46 in 2025")
    change = stat("0.07", 0.07, record_type="derived", quote="the average level of trust increased by 0.07")
    labelled = answers_to_questions([score, change], [question])
    assert [s.record_type for s in labelled] == ["survey_response", "derived"]


def test_between_two_years_is_a_period() -> None:
    from crdc_pipeline.extraction.statistics.periods import period_from_quote

    found = period_from_quote("Ratings increased by 0.09 between 2020 and 2025.", "0.09")
    assert (found.start, found.end) == ("2020", "2025")
    assert period_from_quote("levels of over 3.4 in both 2023 and 2025", "over 3.4") is None


def test_scaled_money_ranges_with_currency_and_between_ranges() -> None:
    from crdc_pipeline.extraction.statistics.duplicates import merge_ranges
    from crdc_pipeline.extraction.statistics.values import tidy_statistic

    lost = tidy_statistic(stat("$32 million dollars", 32000000, unit="dollars", quote="resulting in more than $32 million dollars in lost revenue"))
    cost = tidy_statistic(stat("$3.88 to $7.76/ac", 3.88, unit="USD/ac", quote="is $3.88 to $7.76/ac"))
    assert (lost.value_min, lost.unit, lost.value_status) == (32, "million dollars", "lower_bound")
    assert (cost.value_min, cost.value_max, cost.value_status) == (3.88, 7.76, "range")
    quote = "Typical furrow irrigation events apply between 2 and 4 in/ac of water."
    merged = merge_ranges([stat("2", 2, statistic_id="a", quote=quote), stat("4", 4, statistic_id="b", quote=quote)])
    assert [(s.value_min, s.value_max) for s in merged] == [(2, 4)]


def test_definitions_design_counts_and_equipment_specs_are_setup() -> None:
    from crdc_pipeline.extraction.statistics.filters import is_structural

    assert is_structural(stat("5", 5, quote="weeks following cutout (NAWF =5) and first cracked boll"))
    assert is_structural(stat("four", 4, quote="a randomized complete block design with four replications."))
    assert is_structural(stat("38-in", 38, quote="Plots consisted of eight - 38-in rows that were 40-ft in length"))
    assert not is_structural(stat("1 inch", 1, quote="planted at 45,000 seeds/ac and a depth of 1 inch"))
    assert not is_structural(stat("6-month", 6, quote="Cotton is an annual summer crop, with a 6-month production season."))


def test_treatment_timings_and_count_adjectives_are_setup() -> None:
    from crdc_pipeline.extraction.statistics.filters import is_structural

    assert is_structural(stat("one week", 1, quote="terminated at two weeks prior to first cracked boll, one week prior to first cracked boll"))
    assert is_structural(stat("25-boll", 25, quote="by harvesting a 25-boll sample by hand"))
    assert not is_structural(stat("within four weeks", 4, quote="which occurs within four weeks of harvest"))
    assert not is_structural(stat("350 GDDs", 350, quote="when 350 GDDs past cutout in northern Arkansas"))


def test_p_values_bounds_symbols_and_physical_quantities() -> None:
    from crdc_pipeline.extraction.statistics.values import label_type, tidy_statistic

    p = tidy_statistic(stat("p ≥ 0.6107", 0.6107, quote="had an effect (p ≥ 0.6107)", statistic_type="percentage"))
    assert (p.value_status, p.value_min) == ("lower_bound", 0.6107)
    assert label_type(p).statistic_type == "other"
    threshold = tidy_statistic(stat("-130 kPa", -130, unit="kPa", quote="if soil water potential does not exceed -130 kPa"))
    assert threshold.value_status == "exact"
    assert label_type(stat("250,000 acres", 250000, unit="acres", statistic_type="count")).statistic_type == "other"
    assert label_type(stat("333 GDDs", 333, unit="GDDs", statistic_type="rate")).statistic_type == "other"
    assert label_type(stat("45,000 seeds/ac", 45000, unit="seeds/ac", statistic_type="rate")).statistic_type == "rate"
    assert label_type(stat("21", 21, unit="deaths", statistic_type="count")).statistic_type == "count"


def test_shared_units_and_the_amount_of_a_change() -> None:
    from crdc_pipeline.extraction.statistics.values import label_change, shared_unit

    first = shared_unit(stat("24", 24, quote="rainfall ranged from 24 and 67% below the 10-year average"))
    assert first.unit == "%"
    change = label_change(stat("0.09", 0.09, record_type="survey_response", quote="having increased by 0.09 between 2020 and 2025"))
    level = label_change(stat("49%", 49, record_type="survey_response", quote="neutral responses increased from 49% in 2023"))
    assert change.record_type == "derived" and level.record_type == "survey_response"


def test_thresholds_period_lengths_group_bounds_and_counts_are_setup() -> None:
    from crdc_pipeline.extraction.statistics.filters import is_structural

    setup = [
        stat("five", 5, quote="In compliance with the ethics agreement, if case numbers are below five, we are required not to provide the precise number."),
        stat("1%", 1, quote="All process flows with a cumulative mass or energy of less than 1% were excluded, e.g. packaging glues."),
        stat("1%", 1, quote="The increase in emissions was negligible and below the 1% cut-off for inclusion in the LCA."),
        stat("five", 5, quote="Workers compensation data were accessed for the five-year period 2012-13 to 2016-17."),
        stat("two", 2, quote="This may be reflected in the lower number of weeks lost in the last two years of this study period."),
        stat("three years", 3, quote="the first interval (2008-09 to 2011-12), is only three years in duration"),
        stat("5+ days", 5, quote="Serious injury claims (5+ days), comprised virtually all of the time lost (~99%)."),
        stat("10", 10, quote="those who knew 6-10 people in the cotton industry gave the highest average score of 3.94"),
        stat("three", 3, quote="The foreground data were for the three production seasons from 2011-2012 to 2013-2014."),
        stat("one", 1, quote="The study used a functional unit of one tonne of cotton lint at port."),
        stat("two", 2, quote="The relationship is measured through two core trust and acceptance items."),
    ]
    for s in setup:
        assert is_structural(s), s.quote
    findings = [
        stat("fewer than five", None, quote="In the more recent period, there have been fewer than five cases on a cotton property."),
        stat("6-month", 6, quote="Cotton is an annual summer crop, with a 6-month production season."),
        stat("3-5 years or longer", None, quote="(which in some instances can take 3-5 years or longer)"),
        stat("3.94", 3.94, quote="those who knew 6-10 people in the cotton industry gave the highest average score of 3.94"),
    ]
    for s in findings:
        assert not is_structural(s), s.quote


def test_chart_labels_are_dropped_but_numbered_answers_are_kept() -> None:
    from crdc_pipeline.extraction.statistics.filters import is_table_row

    assert is_table_row(stat("0.16%", 0.16, quote="Prefer not to say: 0.16%"), "")
    assert is_table_row(stat("7.35%", 7.35, quote="Yes 7.35%"), "")
    assert not is_table_row(stat("30%", 30, quote="1. Internet (30%)."), "")
    assert not is_table_row(stat("12%", 12, quote="5. Friends or family (12%)"), "")


def test_a_value_restated_as_a_bound_or_rounded_is_a_duplicate_candidate() -> None:
    from crdc_pipeline.extraction.statistics.duplicates import duplicate_candidates

    bound = stat("approaching $3.5 million", None, status="upper_bound", value_max=3.5, unit="AUD million", statistic_id="a")
    again = stat("approximately $3.5 million", 3.5, status="approximate", unit="million AUD", statistic_id="b")
    rounded = stat("10.3 bales/ha", 10.3, unit="bales/ha", statistic_id="c")
    exact = stat("10.28 bales/ha", 10.28, unit="bales/ha", statistic_id="d")
    whole = stat("three", 3, statistic_id="e")
    share = stat("2.6%", 2.6, unit="%", statistic_id="f")
    pairs = duplicate_candidates([bound, again, rounded, exact, whole, share])
    assert [(a.statistic_id, b.statistic_id) for a, b in pairs] == [("a", "b"), ("c", "d")]


def test_table_rows_chart_labels_and_misread_values_are_left_out() -> None:
    from crdc_pipeline.extraction.statistics.filters import is_structural, is_table_row, value_not_in_quote

    row = stat("23%", 23, quote="Weather monitoring services are accurate enough for me to determine the inversion risk 2% 4% 23% 30% 33% 8%")
    label = stat("0.6", 0.6, quote="Australian Raw Cotton Exports (million tonnes) 0.6")
    header = stat("9.85 bales/ha", 9.85, quote="at 12 bales/ha (2.7 t/ha)ner Production 212 (2019) 974-985 975 (irrigated) in 2014-15 "
                                            "and 9.85 bales/ha (2.2 t/ha) (irrigated) in 2009-14 (Roth, 2014) due to breeding.")
    assert is_table_row(row, "") and is_table_row(label, "")
    assert not is_table_row(header, "")  # a journal header pasted into the sentence, away from the value
    assert value_not_in_quote(stat("0.79 bales/ML", 0.79, quote="The range of water use efficiency has slowed since 2007."))
    assert not value_not_in_quote(stat("eight", 8, quote="taking the total for the period to eight."))
    assert is_structural(stat("a third", quote="These assessments provide a third-party expert assessment of the industry."))
    assert not is_structural(stat("up to a third", quote="which may offset up to a third of the annual increase in CO2"))
    assert not is_structural(stat("two-thirds", quote="Two-thirds of growers used the app."))


def test_scale_ends_and_counts_of_the_reports_own_findings_are_not_findings() -> None:
    from crdc_pipeline.extraction.statistics.filters import is_structural

    scale = "trust is assessed through three distinct questions, employing a scale ranging from 1 (Not at all) to 5 (Extremely)."
    assert is_structural(stat("1 to 5", quote=scale)) and is_structural(stat("5", 5, quote=scale))
    assert is_structural(stat("two", 2, quote="The responses to these inquiries revealed two key findings."))
    assert is_structural(stat("single", 1, quote="Acceptance is measured using a single item, using the same scale."))
