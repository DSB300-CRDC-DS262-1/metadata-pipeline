"""Document metadata schema, prompt, and extraction."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .backends import Backend

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
    """Join the first two and last pages, which carry the metadata."""
    if len(pages) <= LEADING_PAGES + TRAILING_PAGES:
        selected = pages
    else:
        selected = pages[:LEADING_PAGES] + pages[-TRAILING_PAGES:]
    return "\n\n=== page break ===\n\n".join(selected)


def build_prompt(pages: list[str]) -> str:
    return PROMPT_TEMPLATE.format(text=build_context_text(pages))


class MetadataExtractionResult(BaseModel):
    metadata: DocumentMetadata
    resolved_model: str
    seconds: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None


def extract_metadata(pages: list[str], backend: Backend) -> MetadataExtractionResult:
    result = backend.generate(build_prompt(pages), DocumentMetadata)
    return MetadataExtractionResult(
        metadata=result.parsed,
        resolved_model=result.model,
        seconds=result.seconds,
        prompt_tokens=result.prompt_tokens,
        output_tokens=result.output_tokens,
    )
