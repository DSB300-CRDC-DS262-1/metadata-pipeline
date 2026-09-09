"""Document metadata extraction via a frontier model (Gemini).

Pulls bibliographic metadata (title, authors, organisation,
publication date, source URL, and any explicit self-described
revision relationship) from a document's own ingested text. This is
the "easy" half of the CRDC AI Metadata Pipeline's extraction stage --
run first, on a frontier model, before the statistics extraction
(value_status / record_type / context) that the ground truth and
weaknesses_and_fixes.md are actually scoped around.

Deliberately does NOT ask the model for:
- page_count -- already known exactly from ingestion (Document.page_count),
  asking an LLM to recount it would just add a chance to get a fact right
  for the wrong reason.
- supersedes_document_id -- linking one document's ID to another's needs
  corpus-wide reconciliation (matching titles/organisations across
  documents), not something a single document's own text can answer.
  This module extracts `supersedes_note` instead -- whether a document
  explicitly describes itself as updating an earlier report, in its
  own words -- and leaves matching that note to a specific other
  document_id as a separate, not-yet-built reconciliation step.
"""

from __future__ import annotations

import json
import os

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

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
    title: str | None = Field(description="The document's full title, as printed.")
    authors: list[str] = Field(
        default_factory=list,
        description="Author names as printed, in the order given. Empty list if none are named "
        "(e.g. the document credits an organisation rather than individuals).",
    )
    organisation: str | None = Field(
        description="The publishing or commissioning organisation, as named in the document."
    )
    publication_date: str | None = Field(
        description="Publication date exactly as it can be determined from the text "
        "(ISO 8601 'YYYY-MM-DD' if a full date is given, 'YYYY' if only a year is stated)."
    )
    source_url: str | None = Field(
        description="A DOI or URL for this document, ONLY if one literally appears in the "
        "given text. Never construct or guess one."
    )
    supersedes_note: str | None = Field(
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
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DocumentMetadata,
            temperature=0,
        ),
    )
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
