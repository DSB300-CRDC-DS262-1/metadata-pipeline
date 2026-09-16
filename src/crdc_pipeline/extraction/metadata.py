"""Document metadata extraction via a frontier model (Gemini).

Pulls bibliographic metadata (title, authors, organisation, publisher,
funder, publication/submission/acceptance dates, source URL, and any
explicit self-described revision relationship) from a document's own
ingested text. This is the "easy" half of the CRDC AI Metadata
Pipeline's extraction stage, run first, on a frontier model, before
the statistics extraction (value_status / record_type / context) that
the ground truth and weaknesses_and_fixes.md are actually scoped
around.

Schema decisions below (organisation = author affiliation specifically,
separate publisher/funder fields, multiple date fields, source_url kept
even when empty) were confirmed with Burton in the 9 Sept 2026 project
meeting -- see docs/metadata_extraction.md for the full reasoning and
the web-interface-vs-API finding he asked to have documented.

Deliberately does NOT ask the model for:
- page_count: already known exactly from ingestion (Document.page_count),
  asking an LLM to recount it would just add a chance to get a fact right
  for the wrong reason.
- supersedes_document_id: linking one document's ID to another's needs
  corpus-wide reconciliation (matching titles/organisations across
  documents), not something a single document's own text can answer.
  This module extracts `supersedes_note` instead, whether a document
  explicitly describes itself as updating an earlier report, in its
  own words, and leaves matching that note to a specific other
  document_id as a separate, not-yet-built reconciliation step.
"""

from __future__ import annotations

import json
import os
import time

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

#: The free-tier flash model returns 503 ("high demand") often enough
#: in practice that a single unretried call isn't reliable -- observed
#: repeatedly during development, on requests that then succeeded
#: immediately on the next attempt. The SDK's own built-in retry
#: doesn't cover this consistently, hence a small explicit retry here.
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 5

#: Pro-tier Gemini models moved behind billing in 2026; the team's
#: current access is free-tier, which covers Flash models only.
#:
#: Pinned rather than using the "gemini-flash-latest" alias: the alias
#: returned repeated 503s (server overloaded) when this was tested,
#: while this pinned id served the same request cleanly -- an alias
#: pointing at whatever is newest isn't automatically the reliable
#: choice. "gemini-2.5-flash" was tried first and rejected outright
#: (404: "no longer available to new users... use models/gemini-3.6-flash"),
#: which is exactly the failure mode to expect again eventually -- when
#: this one starts 404ing, the error message itself names the
#: replacement, so re-pin from that rather than guessing.
DEFAULT_MODEL = "gemini-3.6-flash"

#: How many leading/trailing pages to give the model. Title, authors,
#: and publication info are almost always on page 1; DOIs and source
#: URLs sometimes only appear in a running header/footer or a
#: references/citation block near the end.
LEADING_PAGES = 2
TRAILING_PAGES = 1


class DocumentMetadata(BaseModel):
    title: str | None = Field(
        default=None, description="The document's full title, as printed."
    )
    authors: list[str] = Field(
        default_factory=list,
        description="Author names as printed, in the order given. Empty list if none are named "
        "(e.g. the document credits an organisation rather than individuals).",
    )
    organisation: str | None = Field(
        default=None,
        description="The author's affiliated institution ONLY (e.g. the university or research "
        "institute an author works for). Never the publisher or the funder, even if the "
        "author's own organisation also funded or published the work; use the publisher/funder "
        "fields for those. If a document is credited to an organisation rather than named "
        "individuals, that organisation still belongs here, not in authors."
    )
    publisher: str | None = Field(
        default=None,
        description="The entity that published or distributes the document (e.g. a journal "
        "publisher, a government agency releasing its own report). Distinct from organisation "
        "(author affiliation) and funder, even when the same name could plausibly fill more "
        "than one role."
    )
    funder: str | None = Field(
        default=None,
        description="The entity that funded or commissioned the work, if the document credits "
        "one explicitly (e.g. 'commissioned by the Cotton Research and Development "
        "Corporation'). Distinct from organisation and publisher."
    )
    publication_date: str | None = Field(
        default=None,
        description="The document's main publication or issue date, exactly as it can be "
        "determined from the text (ISO 8601 'YYYY-MM-DD' if a full date is given, 'YYYY' if "
        "only a year is stated). This is the date that answers 'how new or old is this "
        "document', which is its primary use downstream, so prefer the officially published/"
        "issued date over a submission or acceptance date when a document states more than one."
    )
    submission_date: str | None = Field(
        default=None,
        description="The date the work was originally submitted, if the document states one "
        "separately from its publication date (common in journal articles). Same date format "
        "as publication_date. Null if not stated or not distinguishable from publication_date."
    )
    acceptance_date: str | None = Field(
        default=None,
        description="The date the work was accepted for publication, if the document states "
        "one separately from its publication date. Same date format as publication_date. Null "
        "if not stated or not distinguishable from publication_date."
    )
    source_url: str | None = Field(
        default=None,
        description="A DOI or URL for this document, ONLY if one literally appears in the "
        "given text. Never construct or guess one. Keep this field even when null: most "
        "documents in this corpus have no in-text URL, but future documents or ingestion "
        "sources may supply one, and the field stays part of the schema either way."
    )
    supersedes_note: str | None = Field(
        default=None,
        description="Quote or closely paraphrase any sentence where the document explicitly "
        "describes itself as updating, revising, or replacing an earlier report. Null if the "
        "document makes no such claim about itself."
    )


PROMPT_TEMPLATE = """You are extracting bibliographic metadata from a research document for a \
government research agency's document pipeline. Accuracy matters more than completeness: it is \
much better to return null for a field than to guess or infer a plausible-sounding value.

Rules:
- Use ONLY information literally present in the text below.
- If a field cannot be determined from this text, return null (or an empty list for authors).
- A document commonly involves several distinct organisations. Keep them separate rather than \
picking whichever one is most prominent:
  - organisation: the author's own affiliated institution (e.g. the university or research \
institute an author works for), never the publisher or funder.
  - publisher: who published or distributes the document (e.g. a journal publisher, or a \
government agency releasing its own report).
  - funder: who funded or commissioned the work, only if the document says so explicitly (e.g. \
"commissioned by X").
  The same real-world entity can genuinely fill more than one of these roles for a given \
document (e.g. an agency that both commissioned and published a report); when the text supports \
it, fill in more than one field rather than forcing a single choice.
- A document can state more than one date (publication/issue date, submission date, acceptance \
date). Use publication_date for whichever one represents the document's official publication or \
issue date, since that is the date used downstream to judge how current the document is. Only \
fill submission_date or acceptance_date when the document distinguishes them from the \
publication date explicitly.
- Do not fabricate a source_url. Only return one if a DOI or URL is actually printed in the text.
- For supersedes_note: only fill this in if the document itself explicitly says it updates, \
revises, or replaces an earlier report. Do not infer this from context alone.

Document text (leading and trailing pages; other pages omitted):
---
{text}
---

Extract the metadata now."""


def build_context_text(pages: list[str]) -> str:
    """Select and join the pages worth giving the model.

    Exposed separately from extract_metadata so the page-selection
    logic (which pages, how they're joined) can be unit-tested without
    a live API call.
    """
    if len(pages) <= LEADING_PAGES + TRAILING_PAGES:
        selected = pages
    else:
        selected = pages[:LEADING_PAGES] + pages[-TRAILING_PAGES:]
    return "\n\n=== page break ===\n\n".join(selected)


def build_prompt(pages: list[str]) -> str:
    return PROMPT_TEMPLATE.format(text=build_context_text(pages))


class MetadataExtractionResult(BaseModel):
    metadata: DocumentMetadata
    #: The specific model version that actually served the request, not
    #: just the (possibly "-latest") alias that was requested -- worth
    #: keeping alongside results run with a moving alias, so a later
    #: comparison isn't silently mixing runs from different model
    #: versions without anyone noticing.
    resolved_model: str


def _generate_with_retry(client: genai.Client, model: str, prompt: str):
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=DocumentMetadata,
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


def extract_metadata(
    pages: list[str],
    client: genai.Client,
    model: str = DEFAULT_MODEL,
) -> MetadataExtractionResult:
    """Extract DocumentMetadata from a document's page texts.

    `pages` is a document's full list of per-page text (e.g.
    `[p.text for p in document.pages]` from crdc_pipeline.ingestion).
    """
    prompt = build_prompt(pages)
    response = _generate_with_retry(client, model, prompt)
    if response.parsed is not None:
        metadata = response.parsed
    else:
        # Fallback if the SDK didn't auto-parse for some reason --
        # validate the raw JSON text against the same schema ourselves.
        metadata = DocumentMetadata.model_validate(json.loads(response.text))
    resolved_model = getattr(response, "model_version", None) or model
    return MetadataExtractionResult(metadata=metadata, resolved_model=resolved_model)


def make_client() -> genai.Client:
    """Build a Gemini client, failing with a clear message if no API
    key is configured rather than a raw SDK error."""
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError(
            "No GEMINI_API_KEY (or GOOGLE_API_KEY) found in the environment. "
            "Copy .env.example to .env and fill in your key, or export it directly."
        )
    return genai.Client()
