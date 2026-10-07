"""Prompts and response schemas for the statistics steps."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .schema import Statistic


MAX_STATISTICS_PER_PAGE = 60


class PageStatistics(BaseModel):
    statistics: list[Statistic] = Field(default_factory=list, max_length=MAX_STATISTICS_PER_PAGE)


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
