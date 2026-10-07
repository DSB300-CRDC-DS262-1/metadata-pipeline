"""
Document context extraction via a frontier model (Gemini).

Pulls explicit contextual information (study populations, geographic
regions, data sources, collection methods, limitations, assumptions,
industry sectors, reporting periods, inclusion/exclusion criteria,
etc.) from a document's ingested text. This forms the contextual
foundation of the CRDC AI Metadata Pipeline's extraction stage, where
context entities extracted by this module are intended to serve as
reusable, normalised references that may subsequently be linked to
multiple statistics, findings or document sections.

Deliberately does NOT attempt:

- statistic extraction (percentages, counts, monetary values, rates,
 emissions figures, etc.) -- quantitative finding extraction is
 handled by the statistics extraction module. This module focuses on
 identifying the contextual information required to interpret those
 findings.

- statistic-to-context linking -- while context entities may be
 classified by scope (e.g. global, section, finding-specific), the
 decision of which contexts are relevant to a particular statistic
 requires access to the extracted statistics themselves and is
 performed in a later linking stage.

- corpus-wide context reconciliation and deduplication -- multiple
 documents may reference the same real-world population, methodology,
 organisation, geographic region or limitation using different
 terminology. Resolving these to a shared canonical context entity is
 a corpus-level problem beyond what a single document can determine.

- supersedes_document_id linking or other document lineage
 determination -- like metadata extraction, identifying whether a
 document supersedes, revises or updates another document requires
 cross-document comparison and is left as a separate stage.

- Recovering contextual information that exists only in charts,
 diagrams, images or other non-text renderings and never appears in
 the document's extracted text layer (see the "extraction_failed"
 records in the ground truth). This module operates on extracted text
 only, so such omissions are expected limitations rather than bugs.
"""

from __future__ import annotations

import json
import os
import time

from google import genai
from google.genai import errors, types
from typing import Literal
from pydantic import BaseModel, Field

#: See metadata.py for why this retry exists (free-tier 503s).
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 5

#: See metadata.py for why this is pinned rather than an alias.
DEFAULT_MODEL = "gemini-3.5-flash"

ContextType = Literal[
    "population",
    "sample_population",
    "geographic_region",
    "country",
    "collection_method",
    "data_source",
    "time_period",
    "limitation",
    "assumption",
    "inclusion_criteria",
    "exclusion_criteria",
    "study_design",
    "functional_unit",
    "preprocessing",
    "other"
]

class DateRange(BaseModel):
    start: str | None = Field(
        default=None,
        description="ISO-8601 date YYYY-MM-DD"
    )

    end: str | None = Field(
        default=None,
        description="ISO-8601 date YYYY-MM-DD"
    )

class ContextValue(BaseModel):
    text_value: str | None = None

    date_range: DateRange | None = None

class ContextEntry(BaseModel):
    source_group: int = Field(
        description=(
            "Grouping identifier. Contexts extracted from the "
            "same sentence or paragraph must share the same value."
        )
    )

    context_type: ContextType

    value: ContextValue

    original_text: str = Field(
        description=(
            "At least one complete sentence from the source "
            "document. Never a sentence fragment."
        )
    )

    page: int


class ContextExtraction(BaseModel):
    document: str

    contexts: list[ContextEntry]

class ContextExtraction(BaseModel):
    contexts: list[ContextEntry]

PROMPT_TEMPLATE = """
 You are an information extraction system.

 Your task is to extract contextual information from a document.

 A context is any explicitly stated piece of information that helps a
 reader interpret findings, methodology, data sources, assumptions,
 limitations, populations, geographic scope, reporting periods, study
 design, preprocessing methods or data exclusions.

 Extract all contextual information that may be relevant to later
 interpretation of quantitative findings.

 Return data using the supplied schema.

 ==================================================
 PROVENANCE REQUIREMENTS
 ==================================================

 Preserving provenance is critical.

 1. original_text must always contain at least one complete sentence.

 2. Never extract sentence fragments.

 3. original_text may contain multiple sentences when additional
    surrounding context is necessary.

 4. Preserve the wording used in the document.

 5. original_text should contain all text required to verify the
    extracted context.

 6. Every context must be directly supported by original_text.

 ==================================================
 CONTEXT EXTRACTION RULES
 ==================================================

 Extract contexts such as:

 - population
 - sample population
 - geographic regions
 - countries
 - industry sectors
 - collection methods
 - data sources
 - reporting periods
 - functional units
 - limitations
 - assumptions
 - inclusion criteria
 - exclusion criteria
 - study designs
 - preprocessing methods
 - normalisation methods
 - management practices

 Do not extract statistics.

 Do not extract quantitative findings.

 Do not attempt context-to-statistic linking.

 Extract contextual information only.

 ==================================================
 GROUPING RULES
 ==================================================

 A single sentence or paragraph may contain multiple contexts.

 When multiple contexts originate from the same sentence or passage:

 - create one ContextEntry per context
 - assign the same source_group value
 - preserve the same original_text

 Example:

 "The study was conducted from 2011 to 2014 in northwest New South
 Wales with a functional unit of one tonne of cotton lint."

 Produces:

 source_group = 1

 Context A:
 - geographic_region

 Context B:
 - time_period

 Context C:
 - functional_unit

 All share:

 source_group = 1

 and

 the same original_text.

 ==================================================
 TIME PERIOD RULES
 ==================================================

 When reporting periods are identified:

 Populate value.date_range.

 Convert periods to ISO format where possible.

 Examples:

 2011-2014

 becomes

 start = 2011-01-01
 end = 2014-12-31

 If the exact dates cannot be determined, provide the most precise
 range supported by the text.

 ==================================================
 VALUE NORMALISATION
 ==================================================

 Create concise normalized values.

 Examples:

 Original text:
 "northwest New South Wales"

 Value:
 "Northwest New South Wales"

 Original text:
 "one tonne of cotton lint at port"

 Value:
 "One tonne of cotton lint at port"

 Normalize wording but preserve meaning.

 ==================================================
 LIMITATIONS AND ASSUMPTIONS
 ==================================================

 Extract all explicitly stated:

 - limitations
 - caveats
 - assumptions
 - uncertainties
 - exclusions
 - methodological constraints
 - data quality concerns

 Do not infer additional limitations.

 ==================================================
 DEDUPLICATION
 ==================================================

 Do not create duplicate contexts.

 If the same contextual information appears multiple times,
 retain only the most informative occurrence.

 Prefer occurrences containing:

 - more complete wording
 - stronger provenance
 - more explanatory detail

 ==================================================
 EXAMPLE
 ==================================================

 Source text:

 The foreground data for the study were for the three production
 seasons from 2011-2012 to 2013-2014 in northwest New South Wales,
 with a functional unit of one tonne of cotton lint at port.

 Expected extraction:

 source_group = 1

 Context 1:
 context_type = geographic_region
 value.text_value = Northwest New South Wales

 Context 2:
 context_type = study_design
 value.text_value = Three production seasons

 Context 3:
 context_type = time_period
 value.date_range.start = 2011-01-01
 value.date_range.end = 2014-12-31

 Context 4:
 context_type = functional_unit
 value.text_value = One tonne of cotton lint at port

 All contexts share the same source_group and original_text.

 ==================================================
 FINAL CHECK
 ==================================================

 Before returning results:

 1. Ensure every context is explicitly supported by original_text.

 2. Ensure original_text contains at least one complete sentence.

 3. Ensure related contexts share the same source_group.

 4. Ensure no duplicate contexts remain.

 5. Ensure provenance is preserved.

 ==================================================
 DOCUMENT
 ==================================================
 {text}
"""


def build_context_text(pages: list[str]) -> str:
    """Join all pages with explicit page-number markers.

    Unlike metadata extraction, context can appear on any page,
    so nothing is trimmed here.
    Exposed separately from extract_contexts so this can be
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

def assign_context_ids(
    extraction: ContextExtraction,
    document_id: str,
):
    counters = {}

    for ctx in extraction.contexts:

        group = ctx.source_group

        counters.setdefault(group, 0)
        counters[group] += 1

        suffix = chr(ord("a") + counters[group] - 1)

        ctx.context_group_id = (
            f"{document_id}_{group:03d}"
        )

        ctx.context_id = (
            f"{document_id}_{group:03d}{suffix}"
        )

    return extraction

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
