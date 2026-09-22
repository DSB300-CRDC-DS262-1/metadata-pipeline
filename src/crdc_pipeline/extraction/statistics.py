"""Quantitative statistics extraction via a frontier model (Gemini).

Pulls explicit, prose-reported quantitative findings (percentages,
counts, dollar figures, emissions figures, etc.) from a document's own
ingested text -- the harder half of the CRDC AI Metadata Pipeline's
extraction stage, scoped around value_status / record_type as used in
data/ground_truth_all_documents.json.

Deliberately does NOT attempt:
- context_id linking (geographic region, sample/population, collection
  method, etc.) -- that needs its own extraction pass and/or dedup
  against other statistics sharing the same underlying survey question
  or methodology description. This module extracts each statistic's
  own self-contained facts only.
- supersedes_statistic_id linking -- like metadata's
  supersedes_document_id, this needs corpus-wide reconciliation
  (matching one document's revised figure to another document's
  original one), not something a single document's own text can
  answer alone. Left as a separate, not-yet-built step.
- Recovering values that are visible only in a chart/image rendering
  and never appear in the text layer at all (see the
  "extraction_failed" entries in the ground truth) -- this module
  works from extracted text only, so those are expected, known misses,
  not bugs.
"""

from __future__ import annotations

import json
import os
import time

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

#: See metadata.py for why this retry exists (free-tier 503s).
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 5

#: See metadata.py for why this is pinned rather than an alias.
DEFAULT_MODEL = "gemini-3.6-flash"


class StatisticEntry(BaseModel):
    variable: str = Field(
        description="Plain-English description of what was measured, e.g. "
        "'community agreement that the cotton industry uses water responsibly'."
    )
    value: float | None = Field(
        description="The numeric value, as a plain number (no % sign, no commas, no units). "
        "Null if the source explicitly withholds or censors the value (e.g. 'fewer than five') "
        "rather than simply not reporting one."
    )
    value_status: str = Field(
        description="How the value is reported: 'exact' for a plainly stated figure, "
        "'approximate' if the source hedges it (e.g. 'surpasses', 'approaching', '~'), "
        "'censored_below' if the source deliberately withholds an exact value below a "
        "threshold (e.g. 'fewer than five'). Briefly note the hedge word/phrase used, if any."
    )
    reported_text: str = Field(
        description="The value exactly as printed in the source, e.g. '47%', '$2.8 million', "
        "'fewer than five'."
    )
    unit: str = Field(
        description="What the number measures, e.g. 'percent agreed', 'deaths', "
        "'kg CO2e per tonne of lint', 'AUD millions'."
    )
    record_type: str = Field(
        description="One of: 'measurement' (an observed/reported real-world fact), "
        "'scenario_result' (an output of a modelled what-if scenario), 'model_assumption' "
        "(an input the authors assumed rather than measured), 'survey_response' (a survey "
        "agreement/response percentage). Use the closest fit if none match exactly."
    )
    time_period: str | None = Field(
        description="The time period the statistic covers, as stated or clearly inferable "
        "from the immediate surrounding text (e.g. '2020', '2014-2019', '2017/18-2021/22'). "
        "Null if genuinely not stated anywhere nearby."
    )
    page: int = Field(description="The page number (as given in the page markers below) this statistic was found on.")
    source_excerpt: str = Field(
        description="The sentence(s) from the source text that this statistic was extracted from, "
        "quoted closely enough to verify the extraction against the original."
    )
    note: str | None = Field(
        default=None,
        description="Anything unusual worth flagging: the source explicitly calls the figure "
        "unconfirmed, the time period is ambiguous, the statistic is a computed total of other "
        "statistics, etc. Null if nothing unusual applies.",
    )


class StatisticsExtraction(BaseModel):
    statistics: list[StatisticEntry]


PROMPT_TEMPLATE = """You are extracting quantitative statistics from a research document for a \
government research agency's document pipeline. Accuracy matters more than completeness: it is \
much better to omit a statistic, or return a null value, than to guess or infer a \
plausible-sounding number.

Extract EVERY explicit quantitative finding reported in the text below -- percentages, counts, \
dollar/cost figures, emissions figures, survey agreement rates, and modelled scenario results. \
Do not extract page numbers, section numbers, years used only as labels (e.g. a table column \
header), or figures that are purely decorative/structural rather than a reported finding.

Rules:
- Use ONLY information literally present in the text below. Never calculate, infer, or round a \
value that is not itself printed in the text.
- If the source deliberately withholds an exact value (e.g. "fewer than five"), set value to \
null and value_status to 'censored_below', keeping the withheld phrase in reported_text.
- Distinguish a real measured/reported finding (record_type 'measurement' or 'survey_response') \
from a value the authors merely assumed as a modelling input (record_type 'model_assumption') \
or a modelled what-if outcome (record_type 'scenario_result'). Read the surrounding sentence \
carefully -- don't assume every number is a finding.
- If the same real-world figure is restated in a summary/abstract and again in the body text, \
extract it once, from wherever it is stated most precisely (with page and full context).
- If a single sentence reports multiple distinct numbers (e.g. a count alongside a dollar figure, or several percentages in one sentence), extract EACH number as its own separate statistic entry, even if they share the same source_excerpt. Do not only extract the most prominent number in a sentence.
- Also extract explicit numeric facts about the study's own methodology or process (e.g. how long a step in data collection typically takes, sample sizes, response rates), not just headline research findings. Use record_type 'measurement' for these unless a better fit applies.
- If a time period is ambiguous or requires cross-referencing another part of the document to \
resolve, still extract the statistic, set time_period to null, and use the note field to flag \
the ambiguity.
- page must be the page marker number the statistic appears under, not a guess.

Document text (all pages, each marked with its page number):
---
{text}
---

Extract all statistics now."""


def build_context_text(pages: list[str]) -> str:
    """Join all pages with explicit page-number markers.

    Unlike metadata extraction, statistics can appear on any page, so
    (unlike metadata.py's build_context_text) nothing is trimmed here.
    Exposed separately from extract_statistics so this can be
    unit-tested without a live API call.
    """
    parts = [f"=== page {i} ===\n{page_text}" for i, page_text in enumerate(pages, start=1)]
    return "\n\n".join(parts)


def build_prompt(pages: list[str]) -> str:
    return PROMPT_TEMPLATE.format(text=build_context_text(pages))


class StatisticsExtractionResult(BaseModel):
    statistics: list[StatisticEntry]
    #: See metadata.py's MetadataExtractionResult for why this is kept.
    resolved_model: str


def _generate_with_retry(client: genai.Client, model: str, prompt: str):
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=StatisticsExtraction,
        temperature=0,
    )
    last_error: errors.ServerError | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.generate_content(model=model, contents=prompt, config=config)
        except errors.ServerError as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last_error


def extract_statistics(
    pages: list[str],
    client: genai.Client,
    model: str = DEFAULT_MODEL,
) -> StatisticsExtractionResult:
    """Extract StatisticEntry records from a document's page texts.

    `pages` is a document's full list of per-page text (e.g.
    `[p.text for p in document.pages]` from crdc_pipeline.ingestion).
    """
    prompt = build_prompt(pages)
    response = _generate_with_retry(client, model, prompt)
    if response.parsed is not None:
        extraction = response.parsed
    else:
        # Fallback if the SDK didn't auto-parse for some reason --
        # validate the raw JSON text against the same schema ourselves.
        extraction = StatisticsExtraction.model_validate(json.loads(response.text))
    resolved_model = getattr(response, "model_version", None) or model
    return StatisticsExtractionResult(statistics=extraction.statistics, resolved_model=resolved_model)


def make_client() -> genai.Client:
    """Build a Gemini client, failing with a clear message if no API
    key is configured rather than a raw SDK error.

    Duplicated from metadata.py deliberately -- both extraction
    modules should be independently importable/usable without one
    depending on the other.
    """
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError(
            "No GEMINI_API_KEY (or GOOGLE_API_KEY) found in the environment. "
            "Copy .env.example to .env and fill in your key, or export it directly."
        )
    return genai.Client()
