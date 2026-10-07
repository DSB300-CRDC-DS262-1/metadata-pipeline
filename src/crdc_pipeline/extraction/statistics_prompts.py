"""Prompts and response schemas for the statistics and context extraction steps."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .findings import ContextItem, Statistic

MAX_ITEMS_PER_PAGE = 30
MAX_STATISTICS_PER_PAGE = 60

class PageContextItems(BaseModel):
    items: list[ContextItem] = Field(default_factory=list, max_length=MAX_ITEMS_PER_PAGE)


class PageStatistics(BaseModel):
    statistics: list[Statistic] = Field(default_factory=list, max_length=MAX_STATISTICS_PER_PAGE)


class StatisticLinks(BaseModel):
    statistic_id: str
    item_ids: list[str] = Field(default_factory=list, max_length=MAX_ITEMS_PER_PAGE)


class PageLinks(BaseModel):
    links: list[StatisticLinks] = Field(default_factory=list, max_length=MAX_STATISTICS_PER_PAGE)


class StudyItems(BaseModel):
    item_ids: list[str] = Field(default_factory=list, max_length=MAX_ITEMS_PER_PAGE * 3)


MAX_DUPLICATE_PAIRS = 120


class SamePairs(BaseModel):
    pair_ids: list[str] = Field(default_factory=list, max_length=MAX_DUPLICATE_PAIRS)


DUPLICATE_PROMPT = """Each pair below is two numbers from the same research document that have the same value. \
Decide which pairs report the same measurement: the same quantity of the same thing, stated twice (for \
example the abstract repeating a result from the body, or one sentence restated on the same page).

A pair is not the same measurement when the numbers belong to different things: different survey \
questions or answer options, different groups, years, scenarios, regions or stages, or one is a target or \
threshold and the other a result. When unsure, it is not the same.

Return the IDs of the pairs that are the same measurement, or an empty list.

Pairs (pair ID, then each number as: page | what | value | sentence):
{pairs}
"""


SCOPE_PROMPT = """Below are context items found in a research document. Pick the items that describe the \
document's own study as a whole: facts a reader needs for every number the study itself reports.

For each of these kinds, pick the one item (two at most) that best states it for the whole study, if there is one:
- time_period: when the study's own data were collected or apply; for a survey, its fieldwork dates (not a \
figure's years, the publication date, or when a program began)
- geographic_region: where the study applies
- sample_population: who or what was studied, including the final sample size
- data_source: the study's own main data source(s)
- collection_method: how the study's data were collected
Then add every method fact that the text applies to all of the study's results: the overall method or \
model, the system boundary, the functional unit, the measurement scale, how results were allocated, \
combined or cleaned, the emission factors or formulae used throughout, and exclusions, assumptions and \
limitations stated for the study as a whole.

Never pick: survey questions, scenarios, subgroups, a figure's or table's own source, anything about only \
one of several datasets, publication or report dates, background about the organisation, or studies \
cited from elsewhere.

Return the IDs of the chosen items only.

Context items (ID | kind | fact):
{items}
"""


METHOD_KINDS = {"study_design", "preprocessing", "exclusion_criteria", "inclusion_criteria", "assumption",
                "functional_unit", "limitation"}
MAX_METHOD_ITEMS = 8

METHOD_SCOPE_PROMPT = """Below are method facts found in a research document. Pick only the few that the text \
itself says apply to the whole study or to all of its results: the system boundary, the functional unit or \
unit all results are expressed in, the rating scale used throughout, a currency year applied to all figures, \
how all results were allocated, combined or cleaned, and exclusions or limitations stated for the study as a \
whole.

Do not pick facts about one result, one scenario, one subgroup, one dataset, one table or figure, a \
sensitivity test, or a study cited from elsewhere. Pick at most 8.

Return the IDs of the chosen facts only, or an empty list.

Method facts (ID | kind | fact):
{items}
"""


LINK_PROMPT = """Below are numbers taken from page {page} of a research document, and the context items \
found anywhere in the same document. For each number, list the IDs of every context item that applies \
to it: the facts a reader needs to interpret that number correctly.

Most numbers from the document's own study need several items: where the data came from, how it was \
collected, who or what was measured, the region, the time period, what was included or excluded, the \
method or model, how data were processed, the currency year for money, and any assumption or \
limitation that covers it. Items can come from any page. A survey answer also needs its survey \
question; a modelled result needs its scenario. A figure cited from another study needs only that \
study's source item. Do not link items that belong to a different dataset, period or scenario.

Return one entry per number, using the statistic_id shown.

Context items (ID | kind | fact):
{items}

Numbers on page {page} (statistic_id | what | value | sentence):
{statistics}
"""


CONTEXT_PROMPT = """You are reading page {page} of a research document. List the facts on this page that \
explain how the document's numbers were produced or how they should be read.

Kinds of fact (item_type):
- data_source: where the data came from (a dataset, database, survey, earlier report or cited study)
- collection_method: how it was collected (online survey, case review, interviews)
- sample_population: who or what was measured, including sample size and subgroups
- geographic_region, time_period: where and when the data apply
- inclusion_criteria, exclusion_criteria: what was counted or left out
- study_design, preprocessing: method, model, rating scales, how data were combined, rounded or revised
- assumption, limitation: stated assumptions, caveats, uncertainty, provisional or missing data
- functional_unit: what results are expressed per (e.g. one tonne of lint)
- currency_basis: the year of dollars (e.g. "2018 Australian dollars")
- survey_question: the wording of a survey statement or question
- scenario: a modelled option or setting (e.g. "N fertiliser cut to 180 kg N/ha")
- other

Rules:
- One fact per item. value: a short phrase. quote: the exact sentence it comes from.
- Use only this page. Skip reference lists, contact details, figure labels and table cells.
- whole_document: true only if the fact applies to every number in the document.
- period (time_period items only): dates exactly as precise as printed. "2014-19" is start "2014", \
end "2019"; "Oct 2019" is "2019-10". Never add a month or day that is not printed.
- item_id: "item_01", "item_02" and so on. page: {page}.
- Return an empty list if the page has none.

=== page {page} ===
{text}
"""

def _whole_document(prompt: str, swaps: list[tuple[str, str]]) -> str:
    for old, new in swaps:
        assert prompt.count(old) == 1, old
        prompt = prompt.replace(old, new)
    return prompt


# The same instructions over a whole document in one call, for models with a large context window.
DOCUMENT_CONTEXT_PROMPT = _whole_document(CONTEXT_PROMPT, [
    ("You are reading page {page} of a research document. List the facts on this page that",
     "You are reading a whole research document; each page starts with '=== page N ==='. List the facts that"),
    ("- Use only this page. Skip", "- Skip"),
    ('page: {page}.', "page: the number of the page the quote is on."),
    ("- Return an empty list if the page has none.\n\n=== page {page} ===\n{text}", "{text}"),
])
DOCUMENT_LINK_PROMPT = _whole_document(LINK_PROMPT, [
    ("Below are numbers taken from page {page} of a research document", "Below are numbers taken from a research document"),
    ("Numbers on page {page} (statistic_id | what | value | sentence):", "Numbers (statistic_id | page | what | value | sentence):"),
])


class DocumentContextItems(BaseModel):
    items: list[ContextItem] = Field(default_factory=list, max_length=MAX_ITEMS_PER_PAGE * 10)


class DocumentStatisticLinks(BaseModel):
    statistic_id: str
    item_ids: list[str] = Field(default_factory=list, max_length=MAX_ITEMS_PER_PAGE * 3)


class DocumentLinks(BaseModel):
    links: list[DocumentStatisticLinks] = Field(default_factory=list, max_length=MAX_STATISTICS_PER_PAGE * 10)


STATISTICS_PROMPT = """You are reading page {page} of a research document. List every number stated in \
the running text of this page as a statistic.

Include counts, percentages, money, rates, averages, durations, ranges, model inputs and results, \
and figures the document cites from other studies. Words count too ("six", "half", "a third").
Skip: table cells, figure labels, dates used as dates, years in citations, section, list, page and \
footnote numbers, codes and identifiers, software versions, and reference lists. Also skip counts \
of the document's own parts (two groupings, three measures, four themes), rating-scale labels \
("1 = Strongly disagree"), and the bounds that define a category ("0-4 days", "5+ days", "aged over \
18") unless a result is reported for them. If the same value for the same thing is repeated on the \
page, list it once.

Fields:
- variable: what is measured, as a short noun phrase.
- reported_text: the value exactly as written ("~99%", "fewer than five", "just over $3.4 million").
- value_status and value:
  exact: stated plainly. approximate: "~", "about", "around", "approximately". Put the number in value.
  lower_bound: "over", "more than", "surpasses", "just over". value null, number in value_min.
  upper_bound: "fewer than", "within", "approaching", "up to". value null, number in value_max.
  range: "3-5". value null, value_min and value_max.
  Numbers are written as printed: "80%" is value 80 with unit "%", never 0.8.
- unit: the unit only ("%", "deaths", "AUD million", "kg CO2e per tonne lint").
- record_type: measurement (observed data), survey_response (answers to a survey question, \
including shares agreeing and average ratings), scenario_result (an output of the study's model, \
such as calculated emissions or their breakdown, for the baseline or any option), model_assumption (an \
input the authors assumed or applied), derived (computed from other numbers in the document, e.g. a \
total or a change), cited (reported from another study, often with a citation such as "(Roth, 2014)").
- confirmed: false only if the text says the value is possible or unconfirmed.
- period: the period the value refers to, exactly as precise as printed. Never add a month or day. \
Years and periods ("2014-19", "since 2001", "2018 dollars") belong in period or context items, never \
as statistics of their own. Fill period whenever the sentence or its paragraph gives one.
- response_category: for survey answers only (agree, neutral, disagree, ...).
- context_item_ids: IDs from the list below that apply to this number. Use only listed IDs.
- quote: the exact sentence the value comes from. page: {page}. statistic_id: "stat_01", "stat_02" and so on.

Return an empty list if the page has no numbers in running text.

Context items for this document (ID | kind | fact):
{items}
{word_numbers}
=== page {page} ===
{text}
"""
