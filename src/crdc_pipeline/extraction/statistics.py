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
DEFAULT_MODEL = "gemini-3.5-flash"

class AssociatedDateTimes(BaseModel):
    start: str | None = Field(
        default=None,
        description="ISO-8601 date (YYYY-MM-DD) representing the beginning of the reporting period."
    )

    end: str | None = Field(
        default=None,
        description="ISO-8601 date (YYYY-MM-DD) representing the end of the reporting period."
    )


class StatisticEntry(BaseModel):
    variable: str = Field(
        description=(
            "The real-world quantity being measured. "
            "Use a concise noun phrase such as "
            "'cotton-related deaths', "
            "'injury rate', "
            "'water use efficiency', "
            "'survey agreement with industry sustainability practices'."
        )
    )

    value: float | None = Field(
        description=(
            "The reported numeric value. "
            "Null only when the document deliberately withholds the value "
            "(e.g. 'fewer than five cases')."
        )
    )

    reported_text: str = Field(
        description=(
            "The exact value expression as written in the document. "
            "Examples: '47%', '$2.8 million', "
            "'fewer than five', 'eight'."
        )
    )

    unit: str | None = Field(
        default=None,
        description=(
            "Unit associated with the value. "
            "Examples: '%', 'deaths', 'respondents', "
            "'kg CO2e per tonne lint', 'AUD million'."
        )
    )

    statistic_type: str = Field(
        description=(
            "One of: "
            "count, percentage, rate, average, median, "
            "ratio, monetary_value, duration, index, other."
        )
    )

    value_status: str = Field(
        description=(
            "One of: "
            "exact, approximate, censored_below, censored_above."
        )
    )

    constraints: list[str] = Field(
        default_factory=list,
        description=(
            "Logical constraints on the value. "
            "Examples: "
            "'non_negative', "
            "'integer_count', "
            "'percentage_range_0_100'."
        )
    )

    associated_datetimes: AssociatedDateTimes | None = Field(
        default=None,
        description=(
            "Reporting period associated with the statistic."
        )
    )

    limitations: list[str] = Field(
        default_factory=list,
        description=(
            "Limitations, caveats, assumptions or data-quality concerns "
            "explicitly stated in the surrounding text."
        )
    )

    page: int = Field(
        description="Page number where the statistic occurs."
    )

    source_text: str = Field(
        description=(
            "The complete contextual passage used for extraction. "
            "Include enough surrounding text to verify the value, "
            "time period and limitations."
        )
    )

class StatisticsExtraction(BaseModel):
    statistics: list[StatisticEntry]

PROMPT_TEMPLATE = """You are an information extraction system.
Your task is to extract quantitative findings from a research document.
Return structured records describing quantitative findings, not merely isolated numbers.

A finding consists of:
- the variable being measured
- the reported value
- any unit
- the reporting period
- explicit limitations
- supporting source text

CRITICAL RULES
1. Use ONLY information explicitly present in the supplied text.
2. Never calculate, estimate, infer or invent values.
3. If a value is deliberately withheld
   (e.g. "fewer than five cases"),
   set:
       value = null
       value_status = "censored_below"
   and preserve the original wording in reported_text.
4. Capture limitations whenever they are explicitly stated nearby
   or in immediately preceding contextual paragraphs that apply
   to the finding being reported.
5. Extract complete findings rather than isolated numbers.
   A finding may require multiple consecutive sentences or paragraphs
   to establish the variable, reporting period, limitations and final value.
6. Extract final reported values as primary findings, not multiple values
   that describe the evolution of the same underlying statistic
   (e.g. original count, revised count, updated total).
   Do not create separate records solely for intermediate values
   unless the document explicitly treats them as distinct findings.
7. Preserve document provenance:
   - page must match the page marker
   - source_text must contain all text required
     to verify the extraction.
8. Do not create findings that are unsupported by source_text.
9. If information is genuinely unavailable, return null rather than guessing.
10. Return every distinct quantitative finding as a separate record.
11. source_text should contain every sentence used to determine:
    - the value
    - the variable
    - the reporting period
    - limitations
    - assumptions
    Do not truncate source_text to a single sentence if adjacent
    sentences were required to interpret the finding.

---------------------------------------
FINDING CONSOLIDATION AND DEDUPLICATION
---------------------------------------

A record represents a complete quantitative finding, not an isolated number.

Before creating a record:
1. Locate the primary quantitative statement.
2. Expand context backward and forward until:
   - the variable is unambiguous
   - units are identified
   - reporting periods are identified
   - limitations and assumptions are identified
3. Include all contextual sentences used in interpretation
   within source_text.
4. If several numbers describe the evolution of the same
   finding (e.g. an initial value, revisions, additions,
   updates, or a final total), extract a single record using
   the final reported value and preserve supporting context in source_text.
5. If multiple candidate records describe the same variable,
   population and reporting period, keep only the most complete record.

Return no duplicate findings.

---------------------
DEDUPLICATION EXAMPLE
---------------------

Input text:

The previous review identified a total of six deaths
that definitely occurred within cotton production
in the 2014-19 period.

Since the previous review, two additional cases
were identified to have occurred during this period,
taking the total for the 2014-2019 period to eight.

Incorrect extraction:

[
  {{
    "variable": "cotton-related deaths",
    "value": 6,
    "unit": "deaths"
  }},
  {{
    "variable": "additional cotton-related deaths",
    "value": 2,
    "unit": "deaths"
  }},
  {{
    "variable": "cotton-related deaths",
    "value": 8,
    "unit": "deaths"
  }}
]

Reason:

The values 6 and 2 are intermediate values used to
explain the final reported total of 8. These are not separate findings.

Correct extraction:

[
  {{
    "variable": "cotton-related deaths",
    "value": 8,
    "reported_text": "eight",
    "unit": "deaths",
    "statistic_type": "count",
    "value_status": "exact",
    "constraints": [
      "non_negative",
      "integer_count"
    ],
    "associated_datetimes": {{
      "start": "2014-01-01",
      "end": "2019-12-31"
    }},
    "limitations": [],
    "source_text": "The previous review identified a total of six deaths that definitely occurred within cotton production in the 2014-19 period. Since the previous review, two additional cases were identified to have occurred during this period, taking the total for the 2014-2019 period to eight."
  }}
]

Rule:

When multiple numeric values describe intermediate steps,
revisions, additions, or calculations that lead to a final
reported total, extract a single statistic record using the
final reported value and preserve the supporting context
within source_text.

--------------------------------------------------
FIELD DEFINITIONS
--------------------------------------------------

variable
- The real-world quantity being measured.
- Use concise noun phrases.
- Examples:
    "cotton-related deaths"
    "injury rate"
    "survey agreement"

value
- Numeric value only.
- No units.
- Null only for censored values.

reported_text
- Exact wording used by the document.

unit
- Measurement unit.
Examples:
- deaths
- %
- respondents
- AUD million

statistic_type
Must be one of:
- count
- percentage
- rate
- average
- median
- ratio
- monetary_value
- duration
- index
- other

value_status
Must be one of:
- exact
- approximate
- censored_below
- censored_above

constraints
Examples:
- non_negative
- integer_count
- percentage_range_0_100

associated_datetimes
Where possible convert reporting periods to:
{{
  "start": "YYYY-MM-DD",
  "end": "YYYY-MM-DD"
}}

Examples:
"2014-2019" becomes
{{
  "start": "2014-01-01",
  "end": "2019-12-31"
}}

If insufficient information exists,
set either field to null.

limitations
Capture explicit limitations, caveats,
assumptions, exclusions, uncertainties,
data quality issues, or under-reporting concerns.

source_text
Include all surrounding sentences required to:
- verify value
- verify unit
- verify reporting period
- verify limitations

--------------------------------------------------
EXAMPLE
--------------------------------------------------

Input text:

The previous review identified a total of six deaths
that definitely occurred within cotton production
in the 2014-19 period.

Since the previous review,
two additional cases were identified to have occurred
during this period,
taking the total for the 2014-2019 period to eight.

Output:

{{
  "variable": "cotton-related deaths",
  "value": 8,
  "reported_text": "eight",
  "unit": "deaths",
  "statistic_type": "count",
  "value_status": "exact",
  "constraints": [
    "non_negative",
    "integer_count"
  ],
  "associated_datetimes": {{
    "start": "2014-01-01",
    "end": "2019-12-31"
  }},
  "limitations": [],
  "page": 3,
  "source_text": "The previous review identified a total of six deaths that definitely occurred within cotton production in the 2014-19 period. Since the previous review, two additional cases were identified to have occurred during this period, taking the total for the 2014-2019 period to eight."
}}

--------------------------------------------------
DOCUMENT TEXT
--------------------------------------------------
{text}
--------------------------------------------------
EXTRACT ALL QUANTITATIVE FINDINGS
--------------------------------------------------"""


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
